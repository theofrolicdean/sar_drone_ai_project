import time

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, current_value, dt):
        if dt <= 0:
            return 0
        error = self.setpoint - current_value        
        self.integral += error * dt
        # Limit integral value
        self.integral = max(min(self.integral, 10000), -10000)
        derivative = (error - self.last_error) / dt
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.last_error = error
        return output

    def reset(self):
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
