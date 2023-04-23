class PID(object):
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.cumulative_error = 0
        self.last_step_error = 0

    def action(self, error):
        """Calculate the PID action given the current error."""
        self.cumulative_error += error
        derivative_error = (error - self.last_step_error) / self.dt
        self.last_step_error = error
        return self.Kp * error + self.Ki * self.cumulative_error * self.dt + self.Kd * derivative_error

