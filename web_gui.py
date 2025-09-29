from flask import Flask, render_template, Response, jsonify
from utils.logger import logger
from queue import Empty
import cv2
import time
import config
import numpy as np

def create_flask_app(main_app_instance):
    app = Flask(__name__)
    app.main_app = main_app_instance
    app.drone_path = [] # Store drone path history

    def generate_frames():
        while app.main_app.is_running:
            try:
                frame = app.main_app.display_frame_queue.get(timeout=1)
                
                battery = app.main_app.drone_controller.tello.get_battery() if config.CONNECT_TO_DRONE else 100
                state_val = app.main_app.drone_controller.state.value
                cv2.putText(frame, f"State: {state_val}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Battery: {battery}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        
        current_pos_m = (drone.position_cm / 100).tolist()
        
        if not app.drone_path or np.linalg.norm(np.array(current_pos_m) - np.array(app.drone_path[-1])) > 0.1:
             app.drone_path.append(current_pos_m)

        photo_msg = None
        if app.main_app.photo_message and (time.time() - app.main_app.photo_message_time < 5):
             photo_msg = app.main_app.photo_message
        else:
             app.main_app.photo_message = None


        status_data = {
            'state': drone.state.value,
            'battery': drone.tello.get_battery() if config.CONNECT_TO_DRONE else 100,
            'position': current_pos_m,
            'survivor_position': (survivor_pos / 100).tolist() if survivor_pos is not None else None,
            'joystick_connected': drone.joystick_connected,
            'joystick_mode': drone.joystick_mode,
            'drone_path': app.drone_path,
            'planned_path': app.main_app.planned_return_path,
            'search_path': (np.array(drone.search_path) / 100).tolist(),
            'photo_message': photo_msg
        }
        return jsonify(status_data)

    @app.route('/takeoff', methods=['POST'])
    def takeoff():
        logger.info("Takeoff command received from web GUI.")
        app.main_app.drone_controller.takeoff()
        return jsonify(success=True, message="Takeoff command sent.")

    @app.route('/land', methods=['POST'])
    def land():
        logger.info("Land command received from web GUI.")
        app.main_app.drone_controller.land()
        return jsonify(success=True, message="Land command sent.")

    @app.route('/return_to_home', methods=['POST'])
    def return_to_home():
        logger.info("Return to Home command received from web GUI.")
        app.main_app.drone_controller.trigger_return_to_home()
        return jsonify(success=True, message="Return to Home command sent.")

    @app.route('/toggle_joystick', methods=['POST'])
    def toggle_joystick():
        logger.info("Toggle joystick command received from web GUI.")
        app.main_app.drone_controller.toggle_joystick_mode()
        return jsonify(success=True, message="Toggle joystick command sent.")

    @app.route('/generate_summary', methods=['POST'])
    def generate_summary():
        logger.info("Mission summary requested.")
        main_app = app.main_app
        drone = main_app.drone_controller

        if main_app.mission_start_time is None:
            return jsonify(summary="Mission has not started yet.")

        duration_seconds = time.time() - main_app.mission_start_time
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration_seconds))
        
        survivor_pos_str = "Not detected"
        if main_app.survivor_detected_pos is not None:
            pos_m = main_app.survivor_detected_pos / 100
            survivor_pos_str = f"({pos_m[0]:.1f}, {pos_m[1]:.1f}, {pos_m[2]:.1f}) meters"

        summary_text = (
            f"Mission Summary:\n"
            f"-----------------\n"
            f"Total Flight Time: {duration_str}\n"
            f"Final Drone Status: {drone.state.value}\n"
            f"Final Battery: {drone.tello.get_battery() if config.CONNECT_TO_DRONE else 100}%\n"
            f"Survivor Detected At: {survivor_pos_str}\n"
            f"Total Waypoints Flown: {len(app.drone_path)}"
        )
        return jsonify(summary=summary_text)

    return app