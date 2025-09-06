from flask import Flask, render_template, Response, jsonify
from utils.logger import logger
from queue import Empty
import cv2
import time
import config

def create_flask_app(main_app_instance):
    app = Flask(__name__)
    app.main_app = main_app_instance

    def generate_frames():
        while app.main_app.is_running:
            try:
                frame = app.main_app.display_frame_queue.get(timeout=1)
                with app.main_app.latest_detections_lock:
                    face_bbox = app.main_app.latest_detections.get("face_bbox")
                    gesture = app.main_app.latest_detections.get("gesture")

                if face_bbox:
                    x, y, w, h = face_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if gesture:
                    cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                battery = app.main_app.drone_controller.tello.get_battery() if config.CONNECT_TO_DRONE else 100
                state_val = app.main_app.drone_controller.state.value
                cv2.putText(frame, f"State: {state_val}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Battery: {battery}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error generating video frame: {e}")

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/status')
    def status():
        drone = app.main_app.drone_controller
        survivor_pos = app.main_app.survivor_detected_pos
        
        status_data = {
            "state": drone.state.value,
            "battery": drone.tello.get_battery() if config.CONNECT_TO_DRONE else 100,
            "position": (drone.position_cm / 100).tolist(),
            "survivor_position": (survivor_pos / 100).tolist() if survivor_pos is not None else None
        }
        return jsonify(status_data)

    @app.route('/takeoff', methods=['POST'])
    def takeoff():
        logger.info("Takeoff command received from web GUI.")
        app.main_app.drone_controller.takeoff()
        return jsonify(success=True, message="Takeoff command sent.")

    @app.route('/land', methods=['POST'])
    def land():
        """API endpoint to command the drone to land."""
        logger.info("Land command received from web GUI.")
        app.main_app.drone_controller.land()
        return jsonify(success=True, message="Land command sent.")

    return app
