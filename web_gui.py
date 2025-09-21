from flask import Flask, render_template, Response, jsonify
from utils.logger import logger
from queue import Empty
import cv2
import time
import config
import numpy as np

def create_flask_app(main_app_instance):
    """
    Creates and configures the Flask application.
    A factory function to allow passing the main application instance.
    """
    app = Flask(__name__)
    app.main_app = main_app_instance
    app.drone_path = [] # Store drone path history

    def generate_frames():
        """
        Generator function to stream video frames to the web client.
        It grabs real-time frames and overlays the latest detection data.
        """
        while app.main_app.is_running:
            try:
                frame = app.main_app.display_frame_queue.get(timeout=1)
                
                with app.main_app.latest_detections_lock:
                    face_bbox = app.main_app.latest_detections.get('face_bbox')
                    gesture = app.main_app.latest_detections.get('gesture')

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
        """Render the main HTML page."""
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route."""
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/status')
    def status():
        """API endpoint to get the current drone status."""
        drone = app.main_app.drone_controller
        survivor_pos = app.main_app.survivor_detected_pos
        
        current_pos_m = (drone.position_cm / 100).tolist()
        
        # Add current position to path history
        if not app.drone_path or np.linalg.norm(np.array(current_pos_m) - np.array(app.drone_path[-1])) > 0.1:
             app.drone_path.append(current_pos_m)

        status_data = {
            'state': drone.state.value,
            'battery': drone.tello.get_battery() if config.CONNECT_TO_DRONE else 100,
            'position': current_pos_m,
            'survivor_position': (survivor_pos / 100).tolist() if survivor_pos is not None else None,
            'joystick_connected': drone.joystick_connected,
            'joystick_mode': drone.joystick_mode,
            'drone_path': app.drone_path
        }
        return jsonify(status_data)

    @app.route('/takeoff', methods=['POST'])
    def takeoff():
        """API endpoint to command the drone to take off."""
        logger.info("Takeoff command received from web GUI.")
        app.main_app.drone_controller.takeoff()
        return jsonify(success=True, message="Takeoff command sent.")

    @app.route('/land', methods=['POST'])
    def land():
        """API endpoint to command the drone to land."""
        logger.info("Land command received from web GUI.")
        app.main_app.drone_controller.land()
        return jsonify(success=True, message="Land command sent.")

    @app.route('/toggle_joystick', methods=['POST'])
    def toggle_joystick():
        """API endpoint to toggle joystick mode."""
        logger.info("Toggle joystick command received from web GUI.")
        app.main_app.drone_controller.toggle_joystick_mode()
        return jsonify(success=True, message="Toggle joystick command sent.")

    return app
