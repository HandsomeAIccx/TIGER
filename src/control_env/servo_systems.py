import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box

from src.control_env.servo_signal import CommandSignal


class ServoSystem(object):
    def __init__(self, configs) -> None:
        """
        The init function is called at the beginning of each episode.
        """
        self.configs = configs
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

        self.command_signal = CommandSignal()

        self.signal_x, self.signal_y = [], []
        for time_count in range(self.configs["simulate_times"]):
            self.signal_x.append(time_count)
            self.signal_y.append(
                self.command_signal.trapezoidal(time_count * self.configs["dt"])
            )

        self.track_x, self.track_y, self.track_error = [], [], []

    def reset(self):
        self.rad_pre_step = 0  # rad of servo previous step
        self.rad_current = 0  # rad of servo
        self.rad_target = 0  # target rad of servo
        self.rad_target_2_step = 0  # target rad of servo 2 step later
        self.error_pre_step = 0  # error of servo previous step
        self.error_current = 0  # error of servo
        self.electric_pre_step = 0  # electric of servo previous step
        self.step_count = 0  # step count of servo
        self.dt = 0.0001  # time interval of servo

        self.track_x.append(self.step_count)
        self.track_y.append(self.rad_current)
        self.track_error.append(self.error_current)
        return np.array(
            [
                self.rad_pre_step,
                self.rad_current,
                self.rad_target,
                self.rad_target_2_step,
                self.error_pre_step,
                self.error_current,
            ]
        )

    def step(self, action):
        assert action in self.action_space, "action is not in action space"
        error_current = (
            self.command_signal.trapezoidal(self.step_count * self.dt)
            - self.rad_current
        )

        rad = self.system(electric=action)

        # update state
        self.rad_pre_step = self.rad_current
        self.rad_current = rad
        self.rad_target = self.command_signal.trapezoidal(
            (self.step_count + 1) * self.dt
        )
        self.rad_target_2_step = self.command_signal.trapezoidal(
            (self.step_count + 2) * self.dt
        )
        self.error_pre_step = self.error_current
        self.error_current = error_current

        self.step_count += 1

        self.track_x.append(self.step_count)
        self.track_y.append(self.rad_current)
        self.track_error.append(self.error_current)
        return np.array(
            [
                self.rad_pre_step,
                self.rad_current,
                self.rad_target,
                self.rad_target_2_step,
                self.error_pre_step,
                self.error_current,
            ]
        )

    def render(self):
        plt.cla()
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axs[0].plot(self.signal_x, self.signal_y, "-r", label="CommandSignal")
        axs[0].plot(self.track_x, self.track_y, "ob", label="Trajectory")
        axs[0].set_title(
            "speed:"
            + str(round(float(self.rad_current), 2))
            + ",target index:"
            + str(round(float(self.rad_target), 2))
        )
        axs[0].legend()

        axs[1].plot(self.track_x, self.track_error, "-b", label="CommandError")
        axs[1].set_title("error:" + str(round(float(self.error_current), 2)))
        axs[1].legend(loc="best")
        axs[1].grid(True)
        plt.pause(0.0001)

    def system(self, electric):
        if isinstance(electric, np.ndarray):
            assert electric.shape == (1,), "electric shape is not (1,)"
            electric = electric[0]
        else:
            raise ValueError("electric is not np.ndarray")
        return (
            self.rad_current
            - 3.478 * 0.0001 * self.rad_pre_step
            + 1.388 * electric
            + 0.1986 * self.electric_pre_step
            + 0.1 * np.random.normal(0, 1)
        )
