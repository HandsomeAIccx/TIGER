import numpy as np
from gymnasium.spaces import Box

from src.control_env.servo_signal import CommandSignal


class ServoSystem(object):
    def __init__(self) -> None:
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(1, 6), dtype=np.float32)
        self.action_space = Box(low=-1, high=1, shape=(1, 1), dtype=np.float32)
        
        self.command_signal = CommandSignal()

    def reset(self):
        """
        The reset function is called at the beginning of each episode.
        """
        self.rad_pre_step = 0  # rad of servo previous step
        self.rad_current = 0  # rad of servo
        self.rad_target = 0  # target rad of servo
        self.rad_target_2_step = 0  # target rad of servo 2 step later
        self.error_pre_step = 0  # error of servo previous step
        self.error_current = 0  # error of servo
        self.electric_pre_step = 0  # electric of servo previous step
        self.step_count = 0  # step count of servo
        self.dt = 0.0001  # time interval of servo
        
        return np.array(
            [
                self.rad_pre_step,
                self.rad_current,
                self.rad_target,
                self.rad_target_2_step,
                self.error_pre_step,
                self.error_current,
            ]
        )[np.newaxis, :]

    def step(self, action):
        
        error_current = self.command_signal.trapezoidal(self.step_count * self.dt) - self.rad_current
        
        rad = self.system(electric=action)
        
        # update state
        self.rad_pre_step = self.rad_current
        self.rad_current = rad
        self.rad_target = self.command_signal.trapezoidal((self.step_count + 1) * self.dt)
        self.rad_target_2_step = self.command_signal.trapezoidal((self.step_count + 2) * self.dt)
        self.error_pre_step = self.error_current
        self.error_current = error_current
        
        self.step_count += 1
        
        return np.array(
            [
                self.rad_pre_step,
                self.rad_current,
                self.rad_target,
                self.rad_target_2_step,
                self.error_pre_step,
                self.error_current,
            ]
        )[np.newaxis, :]

    def render(self):
        pass
    
    def system(self, electric):
        return self.rad_current - 3.478 * 0.0001 * self.rad_pre_step + 1.388 * electric + 0.1986 * self.electric_pre_step + 0.1 * np.random.normal(0, 1)



        