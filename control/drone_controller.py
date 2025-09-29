from djitellopy import Tello
import time
import math
from enum import Enum
from utils.logger import logger
import config
from control.pid_controller import PIDController
from navigation.path_planner import PathPlanner
import numpy as np
import threading
from inputs import get_gamepad


class DroneState(Enum):
    ON_GROUND = "On Ground"
    SEARCHING = "Searching"
    TRACKING = "Tracking Survivor"
    GUIDING = "Guiding Survivor"
    RETURNING_HOME = "Returning to Home"
    SOS_HOVER = "SOS! Hovering for Rescue"
    LANDED = "Landed"
    JOYSTICK_CONTROL = "Joystick Control"

class DroneController:
    def __init__(self, main_app):
        self.main_app = main_app
        self.tello = Tello()
        self.state = DroneState.ON_GROUND
        self.is_running = False
        self.position_cm = np.array([0.0, 0.0, 0.0]) # x, y, z
        self.yaw_deg = 0
        self.home_position = np.array([0.0, 0.0, 0.0])
        self.pid_yaw = PIDController(config.PID_GAINS['P'], config.PID_GAINS['I'], config.PID_GAINS['D'])
        self.pid_dist = PIDController(config.PID_GAINS['P'], config.PID_GAINS['I'], config.PID_GAINS['D'], setpoint=config.FACE_TARGET_AREA)
        self.path_planner = PathPlanner(config.MAP_SIZE_METERS, config.GRID_CELL_SIZE_CM)
        self.search_path = self._generate_spiral_search_path()
        self.current_search_waypoint_index = 0
        self.return_path = None
        self.current_waypoint_index = 0
        self.joystick_mode = False
        self.joystick_thread = None
        self.joystick_connected = False


    def connect(self):
        if not config.CONNECT_TO_DRONE:
            logger.warning("Drone connection disabled.")
            return
        try:
            self.tello.connect()
            self.tello.streamon()
            logger.info(f"Tello connected. Battery: {self.tello.get_battery()}%")
            self.home_position[2] = self.tello.get_height()
        except Exception as e:
            logger.error(f"Failed to connect to Tello: {e}", exc_info=True)
            raise

    def start(self):
        self.is_running = True
        if get_gamepad:
            self.joystick_thread = threading.Thread(target=self._joystick_thread, daemon=True)
            self.joystick_thread.start()
        logger.info("DroneController started.")

    def stop(self):
        self.is_running = False
        if config.CONNECT_TO_DRONE and self.tello.is_flying:
            logger.info("Landing drone...")
            try:
                self.land()
            except Exception as e:
                logger.error(f"Error during landing: {e}")
        logger.info("DroneController stopped.")

    def _joystick_thread(self):
        rc_commands = {'ABS_X': 0, 'ABS_Y': 0, 'ABS_RX': 0, 'ABS_RY': 0}
        
        while self.is_running:
            try:
                events = get_gamepad()
                self.joystick_connected = True
                for event in events:
                    if event.ev_type == 'Absolute' and event.code in rc_commands:
                        value = int((event.state / 32768) * 100)
                        rc_commands[event.code] = value
                    elif event.ev_type == 'Key' and event.code == 'BTN_SOUTH' and event.state == 1:
                        logger.info("Photo capture command received from joystick.")
                        self.main_app.capture_photo_flag.set()
                if self.joystick_mode:
                    dead_zone = 10
                    roll = rc_commands['ABS_RX'] if abs(rc_commands['ABS_RX']) > dead_zone else 0
                    pitch = -rc_commands['ABS_RY'] if abs(rc_commands['ABS_RY']) > dead_zone else 0
                    throttle = -rc_commands['ABS_Y'] if abs(rc_commands['ABS_Y']) > dead_zone else 0
                    yaw = rc_commands['ABS_X'] if abs(rc_commands['ABS_X']) > dead_zone else 0
                    self.tello.send_rc_control(roll, pitch, throttle, yaw)
            except Exception:
                if self.joystick_connected:
                    logger.warning("Joystick disconnected or error.")
                self.joystick_connected = False
                rc_commands = {k: 0 for k in rc_commands}
                time.sleep(1)

    def toggle_joystick_mode(self):
        if not self.joystick_connected:
            logger.warning("Cannot enable joystick mode: No joystick connected.")
            return
        battery = self.tello.get_battery()
        if battery < config.BATTERY_FAILSAFE_PERCENTAGE:
            logger.warning(f"Cannot enable joystick mode: Battery is at {battery}%.")
            return
        self.joystick_mode = not self.joystick_mode
        if self.joystick_mode:
            self.state = DroneState.JOYSTICK_CONTROL
            logger.info("Joystick mode enabled.")
        else:
            self.state = DroneState.SEARCHING
            self.tello.send_rc_control(0,0,0,0)
            logger.info("Joystick mode disabled.")

    def update(self, vision_data, dt):
        if not self.is_running or (config.CONNECT_TO_DRONE and not self.tello.is_flying):
            return

        battery = self.tello.get_battery()
        if battery <= config.EMERGENCY_LAND_BATTERY:
            logger.critical(f"CRITICAL BATTERY ({battery}%)! EMERGENCY LANDING!")
            self.tello.emergency()
            self.state = DroneState.LANDED
            return
        
        if battery < config.BATTERY_FAILSAFE_PERCENTAGE:
            if self.joystick_mode:
                logger.warning(f"Low battery ({battery}%), disabling joystick mode and returning to home.")
                self.joystick_mode = False
            if self.state not in [DroneState.RETURNING_HOME, DroneState.LANDED, DroneState.SEARCHING]:
                logger.warning(f"Low battery ({battery}%), triggering return to home.")
                self.state = DroneState.RETURNING_HOME
                self.return_path = self.path_planner.plan_path(self.position_cm[:2], self.home_position[:2])
                self.current_waypoint_index = 0

        if self.joystick_mode:
            return

        face_bbox, gesture = vision_data

        if self.state == DroneState.SEARCHING:
            self._handle_searching(face_bbox, dt)
        elif self.state == DroneState.TRACKING:
            self._handle_tracking(face_bbox, gesture, dt)
        elif self.state == DroneState.GUIDING:
            self._handle_guiding(face_bbox, gesture, dt)
        elif self.state == DroneState.RETURNING_HOME:
            self._handle_return_to_home(dt)
        elif self.state == DroneState.SOS_HOVER:
            self.tello.send_rc_control(0, 0, 0, 0)
        elif self.state == DroneState.ON_GROUND or self.state == DroneState.LANDED:
            # In webcam mode, we need an initial state to begin testing.
            if not config.CONNECT_TO_DRONE and self.state == DroneState.ON_GROUND:
                logger.info("Webcam mode: Setting state to SEARCHING to begin.")
                self.state = DroneState.SEARCHING
            pass
            
    def trigger_return_to_home(self):
        """Manually triggers the return to home sequence."""
        if self.tello.is_flying and self.state not in [DroneState.RETURNING_HOME, DroneState.LANDED]:
            logger.info("Manual Return to Home triggered.")
            self.state = DroneState.RETURNING_HOME
            self.return_path = self.path_planner.plan_path(self.position_cm[:2], self.home_position[:2])
            self.current_waypoint_index = 0
        else:
            logger.warning("Return to Home command ignored: drone is not in a valid state to RTH.")


    def _handle_searching(self, face_bbox, dt):
        if face_bbox is not None:
            logger.info("Face detected. Switching to TRACKING mode.")
            self.state = DroneState.TRACKING
            return
        if self.current_search_waypoint_index >= len(self.search_path):
            logger.info("Search pattern complete. Restarting search...")
            self.current_search_waypoint_index = 0
            return
        target_pos = self.search_path[self.current_search_waypoint_index]
        current_pos = self.position_cm[:2]
        distance_to_target = np.linalg.norm(target_pos - current_pos)
        if distance_to_target < (config.GRID_CELL_SIZE_CM / 2):
            self.current_search_waypoint_index += 1
            logger.info(f"Reached search waypoint. Moving to next.")
            self.tello.send_rc_control(0, 0, 0, 0)
        else:
            self._move_towards(target_pos, dt)

    def _handle_tracking(self, face_bbox, gesture, dt):
        if face_bbox is None:
            logger.warning("Lost track of face. Reverting to SEARCHING.")
            self.state = DroneState.SEARCHING
            self.tello.send_rc_control(0, 0, 0, 0)
            return
        if gesture == "sos":
            logger.critical("SOS gesture detected! Engaging SOS_HOVER mode.")
            self.state = DroneState.SOS_HOVER
            return
        
        if gesture == "fist":
            distance_from_home = np.linalg.norm(self.position_cm[:2] - self.home_position[:2])
            if distance_from_home > 300:
                logger.info("Fist gesture detected. Planning path and starting to guide survivor.")
                self.return_path = self.path_planner.plan_path(self.position_cm[:2], self.home_position[:2])
                self.current_waypoint_index = 0
                if self.return_path:
                    self.state = DroneState.GUIDING
                else:
                    logger.error("Could not plan guiding path. Returning to tracking.")
            else:
                logger.warning("Fist gesture ignored: too close to home to begin guiding.")
            return
        
        if gesture == "palm":
            logger.info("Palm gesture detected. Pausing.")
            self.tello.send_rc_control(0, 0, 0, 0)
            return
        self._track_face(face_bbox, dt)

    def _follow_path(self, dt):
        if self.return_path is None or self.current_waypoint_index >= len(self.return_path):
            logger.info("Path finished. Landing.")
            self.land()
            return

        target_pos = self.return_path[self.current_waypoint_index]
        current_pos = self.position_cm[:2]
        distance_to_target = np.linalg.norm(target_pos - current_pos)

        if distance_to_target < (config.GRID_CELL_SIZE_CM / 2):
            self.current_waypoint_index += 1
            logger.info(f"Reached waypoint {self.current_waypoint_index}/{len(self.return_path)}.")
        else:
            self._move_towards(target_pos, dt)

    def _handle_guiding(self, face_bbox, gesture, dt):
        if face_bbox is None:
            logger.warning("Lost track of survivor during guiding. Pausing and reverting to TRACKING.")
            self.state = DroneState.TRACKING
            self.tello.send_rc_control(0, 0, 0, 0)
            return
        if gesture == "sos":
            logger.critical("SOS gesture detected! Engaging SOS_HOVER mode.")
            self.state = DroneState.SOS_HOVER
            return
        if gesture == "palm":
            logger.info("Palm gesture detected. Pausing guidance.")
            self.state = DroneState.TRACKING
            self.tello.send_rc_control(0, 0, 0, 0)
            return
        
        self._follow_path(dt)

    def _handle_return_to_home(self, dt):
        self._follow_path(dt)
        
    def _track_face(self, face_bbox, dt):
        x, y, w, h = face_bbox
        cx = x + w // 2
        area = w * h
        yaw_error = cx - config.FRAME_WIDTH / 2
        yaw_speed = -int(self.pid_yaw.update(yaw_error, dt))
        yaw_speed = np.clip(yaw_speed, -config.YAW_VELOCITY, config.YAW_VELOCITY).astype(int)
        dist_error = area
        fwd_speed = int(self.pid_dist.update(dist_error, dt))
        fwd_speed = np.clip(fwd_speed, -config.FORWARD_VELOCITY, config.FORWARD_VELOCITY).astype(int)
        self.tello.send_rc_control(0, fwd_speed, 0, yaw_speed)
        self._update_position(0, fwd_speed, 0, yaw_speed, dt)

    def _move_towards(self, target_pos_cm, dt):
        current_pos_cm = self.position_cm[:2]
        vector_to_target = target_pos_cm - current_pos_cm
        target_yaw_rad = np.arctan2(vector_to_target[1], vector_to_target[0])
        target_yaw_deg = np.degrees(target_yaw_rad)
        current_yaw_deg = self.yaw_deg
        yaw_error = (target_yaw_deg - current_yaw_deg + 180) % 360 - 180
        yaw_speed = 0
        fwd_speed = 0
        if abs(yaw_error) > 10:
            yaw_speed = np.clip(int(yaw_error * 0.8), -config.YAW_VELOCITY, config.YAW_VELOCITY)
        else:
            fwd_speed = config.FORWARD_VELOCITY
        self.tello.send_rc_control(0, fwd_speed, 0, yaw_speed)
        self._update_position(0, fwd_speed, 0, yaw_speed, dt)

    def _update_position(self, lr, fb, ud, yaw, dt):
        if dt <= 0: return
        self.yaw_deg += (yaw * dt)
        self.yaw_deg %= 360
        distance = fb * dt
        rad = math.radians(self.yaw_deg)
        self.position_cm[0] += distance * math.cos(rad)
        self.position_cm[1] += distance * math.sin(rad)
        self.position_cm[2] = self.tello.get_height()

    def _generate_spiral_search_path(self):
        path = []
        radius = 0
        angle = 0
        max_radius_cm = config.SEARCH_PATTERN_RADIUS_M * 100
        step_cm = config.GRID_CELL_SIZE_CM
        path.append(np.array([0, 0]))
        while radius <= max_radius_cm:
            x = int(radius * np.cos(angle))
            y = int(radius * np.sin(angle))
            if not path or (path[-1][0] != x or path[-1][1] != y):
                 path.append(np.array([x, y]))
            angle += np.pi / 8
            radius += step_cm / 20
        logger.info(f"Generated spiral search path with {len(path)} waypoints.")
        return path
    
    def takeoff(self):
        if self.state == DroneState.ON_GROUND and config.CONNECT_TO_DRONE and not self.tello.is_flying:
            logger.info("Taking off...")
            self.tello.takeoff()
            time.sleep(1)
            self.state = DroneState.SEARCHING
            self.home_position = np.array([0.0, 0.0, self.tello.get_height()])
            self.position_cm = self.home_position.copy()
            
    def land(self):
        logger.info("Landing command received.")
        self.state = DroneState.LANDED
        if config.CONNECT_TO_DRONE and self.tello.is_flying:
            self.tello.land()