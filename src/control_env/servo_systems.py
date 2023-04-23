import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box

from src.control_env.servo_signal import CommandSignal


class ServoSystem(object):
    """
    This class is used to simulate the servo system.
    """

    def __init__(self, configs) -> None:
        """
        Initialize function for ServoSystem class.

        Args:
            configs (dict): The configuration of the servo system.

        Returns:
            None
        """
        self.configs = configs

        assert "simulate_times" in self.configs, "simulate_times must be in configs"
        assert "dt" in self.configs, "dt must be in configs"

        self.simulate_times = self.configs["simulate_times"]
        self.dt = self.configs["dt"]

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )  # set observation space for 6 dimensions.
        self.action_space = Box(
            low=-5.0, high=5.0, shape=(1,), dtype=np.float32
        )  # action bounds are [-5, 5]

        self.command_signal = CommandSignal()

        self.signal_x, self.signal_y = [], []
        for time_count in range(self.configs["simulate_times"]):
            self.signal_x.append(time_count)
            self.signal_y.append(
                self.command_signal.trapezoidal(time_count * self.configs["dt"])
            )

        self.track_x, self.track_y, self.track_error = [], [], []

    def reset(self):
        """
        Reset function for ServoSystem class.

        Args:
            None

        Returns:
            obs (np.array): The observation of the servo system.
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

        self.track_x.append(self.step_count)
        self.track_y.append(self.rad_current)
        self.track_error.append(self.error_current)
        obs = np.array(
            [
                self.rad_pre_step,
                self.rad_current,
                self.rad_target,
                self.rad_target_2_step,
                self.error_pre_step,
                self.error_current,
            ]
        )
        return obs

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
        obs = np.array(
            [
                self.rad_pre_step,
                self.rad_current,
                self.rad_target,
                self.rad_target_2_step,
                self.error_pre_step,
                self.error_current,
            ]
        )
        return obs

    def render(self):
        ax1 = plt.subplot(121)
        ax1.figure.set_size_inches(12, 6)
        ax1.cla()
        ax1.plot(self.signal_x, self.signal_y, "-r", label="CommandSignal")
        ax1.plot(self.track_x, self.track_y, "ob", label="Trajectory")
        ax1.set_title(
            "speed:"
            + str(round(float(self.rad_current), 2))
            + ",target index:"
            + str(round(float(self.rad_target), 2))
        )
        ax1.grid(True)
        ax1.legend(loc="upper right")

        ax2 = plt.subplot(122)
        ax2.figure.set_size_inches(12, 6)
        ax2.cla()
        ax2.plot(self.track_x, self.track_error, "-b", label="CommandError")
        ax2.set_title("error:" + str(round(float(self.error_current), 2)))
        ax2.grid(True)
        ax2.legend(loc="upper right")
        plt.pause(0.00001)

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
