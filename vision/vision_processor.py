import cv2
import mediapipe as mp
import numpy as np
import pickle
from utils.logger import logger
import config

class VisionProcessor:
    def __init__(self, model_path="models/best_model_gesture.pkl"):
        try:
            # Load face detector
            self.face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_PATH)
            if self.face_cascade.empty():
                logger.error(f"Failed to load Haarcascade from path: {config.FACE_CASCADE_PATH}")
                raise IOError("Haarcascade file not found.")

            # Initialize Mediapipe hands
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils

            # Load ML model trained on landmarks
            model_dict = pickle.load(open(model_path, 'rb'))
            self.model = model_dict["model"]
            self.label_map = {
                "A": "fist",
                "B": "palm",
                "S": "sos"
            }
            self.valid_classes = list(set(self.label_map.values()))

            logger.info("VisionProcessor initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing VisionProcessor: {e}", exc_info=True)
            raise

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)
        )
        if len(faces) == 0:
            return None, None
        largest_face = max(faces, key=lambda rect: rect[2]*rect[3])
        x, y, w, h = largest_face
        center_x, center_y = x + w // 2, y + h // 2
        return largest_face, (center_x, center_y)

    def detect_gesture(self, frame):
        gesture = "Not Detected"
        hand_bbox = None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use a persistent Hands object for performance
        with self.mp_hands.Hands(static_image_mode=False, 
                                max_num_hands=1, 
                                min_detection_confidence=0.8,
                                min_tracking_confidence=0.8) as hands:
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                x_coords = []
                y_coords = []
                data_aux = []

                for hand_landmarks in results.multi_hand_landmarks:
                    # draw landmarks on original frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(236, 255, 3), thickness=2)
                    )

                    # collect landmark positions
                    for lm in hand_landmarks.landmark:
                        x_coords.append(lm.x)
                        y_coords.append(lm.y)
                        data_aux.extend([lm.x, lm.y])

                if len(data_aux) >= 42:
                    h_frame, w_frame, _ = frame.shape
                    x_min = int(min(x_coords) * w_frame)
                    y_min = int(min(y_coords) * h_frame)
                    x_max = int(max(x_coords) * w_frame)
                    y_max = int(max(y_coords) * h_frame)
                    hand_bbox = (x_min, y_min, x_max, y_max)

                    pred_class = self.model.predict([np.array(data_aux)[0:42]])[0]
                    gesture = pred_class if pred_class in self.valid_classes else "Not Detected"

        return gesture, hand_bbox


    def process_frame(self, frame):
        face_bbox, face_center = self.detect_face(frame)
        gesture, hand_bbox = self.detect_gesture(frame)

        # Draw face box safely
        if face_bbox is not None:
            x, y, w, h = face_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue
            if face_center is not None:
                cv2.circle(frame, face_center, 5, (0, 0, 255), -1)

        # Draw hand box safely
        if hand_bbox is not None:
            x_min, y_min, x_max, y_max = hand_bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green

        # Display gesture label
        if gesture is not None:
            cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, face_bbox, hand_bbox, gesture

