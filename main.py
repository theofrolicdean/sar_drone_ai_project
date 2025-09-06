# main.py
import threading
import time
import sys
from queue import Queue, Empty
import cv2
from waitress import serve
from web_gui import create_flask_app
from control.drone_controller import DroneController, DroneState
from vision.vision_processor import VisionProcessor
from utils.logger import logger
import config


class MainApplication:
    def __init__(self):
        self.is_running = True
        self.processing_frame_queue = Queue(maxsize=1) 
        self.display_frame_queue = Queue(maxsize=1)    
        self.results_queue = Queue(maxsize=1)          
        self.latest_detections = {"face_bbox": None, "gesture": None}
        self.latest_detections_lock = threading.Lock()
        self.vision_processor = VisionProcessor()
        self.drone_controller = DroneController()
        self.survivor_detected_pos = None
        self.threads = []
        self.flask_app = create_flask_app(self)

    def run(self):
        try:
            self.drone_controller.connect()
        except Exception as e:
            logger.error(f"Error connecting to drone. Application will exit. {e}")
            
        self._start_thread(self._video_capture_loop, "VideoCaptureThread")
        self._start_thread(self._vision_processing_loop, "VisionProcessingThread")
        self._start_thread(self._drone_control_loop, "DroneControlThread")

        logger.info("Starting Flask server on http://0.0.0.0:8080")
        try:
            serve(self.flask_app, host='0.0.0.0', port=8080, threads=8)
        except OSError as e:
            logger.error(f"Could not start server: {e}")
        
        self.on_closing()

    def _start_thread(self, target_func, name):
        thread = threading.Thread(target=target_func, name=name, daemon=True)
        self.threads.append(thread)
        thread.start()
        logger.info(f"Thread {name} started.")

    def _video_capture_loop(self):
        if not config.CONNECT_TO_DRONE:
            logger.error("Drone connection is disabled in config.")
            return

        address = self.drone_controller.tello.get_udp_video_address()
        logger.info(f"Connecting to Tello video stream at: {address}")
        cap = cv2.VideoCapture(address)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        if not cap.isOpened():
            logger.error("Failed to open Tello video stream.")
            self.is_running = False
            return

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                continue

            while not self.processing_frame_queue.empty():
                self.processing_frame_queue.get_nowait()
            self.processing_frame_queue.put(frame)

        cap.release()



    def _vision_processing_loop(self):
        while self.is_running:
            try:
                frame = self.processing_frame_queue.get(timeout=1)
                
                # draw face/hand boxes and landmarks
                processed_frame, face_bbox, hand_bbox, gesture = self.vision_processor.process_frame(frame)

                if gesture is not None:
                    cv2.putText(processed_frame, f"Gesture: {gesture.upper()}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                while not self.display_frame_queue.empty():
                    self.display_frame_queue.get_nowait()
                self.display_frame_queue.put(processed_frame)

                if self.results_queue.full():
                    self.results_queue.get_nowait()
                self.results_queue.put((face_bbox, gesture))

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in vision processing loop: {e}", exc_info=True)




    def _drone_control_loop(self):
        self.drone_controller.start()
        last_update_time = time.time()

        while self.is_running:
            try:
                face_bbox, gesture = None, None
                try:
                    face_bbox, gesture = self.results_queue.get(timeout=0.1)
                except Empty:
                    pass

                current_time = time.time()
                dt = current_time - last_update_time
                last_update_time = current_time

                self.drone_controller.update((face_bbox, gesture), dt)

                if face_bbox is not None and self.survivor_detected_pos is None:
                    if self.drone_controller.state in [DroneState.SEARCHING, DroneState.TRACKING]:
                         self.survivor_detected_pos = self.drone_controller.position_cm.copy()
                         logger.info(f"Survivor located at position: {self.survivor_detected_pos}")
                time.sleep(1 / 30)

            except Exception as e:
                logger.error(f"Error in drone control loop: {e}", exc_info=True)
                time.sleep(1)
        
        self.drone_controller.stop()

    def on_closing(self):
        logger.info("Application closing...")
        self.is_running = False

def run_vision_tester():
    logger.info("Starting Vision Tester mode...")
    vision_processor = VisionProcessor(model_path="models/best_model_gesture.pkl")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, face_bbox, hand_bbox, gesture = vision_processor.process_frame(frame)

        # Show the processed frame
        cv2.imshow("Vision Tester - Press q to quit", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Vision Tester closed.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--test-vision':
        run_vision_tester()
    else:
        try:
            logger.info("Starting SAR Drone Application...")
            app = MainApplication()
            app.run()
        except Exception as e:
            print("\n" + "="*50)
            print(f"ERROR: {e}")
