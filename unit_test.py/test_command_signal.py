from src.control_env.servo_signal import CommandSignal

import matplotlib.pyplot as plt


def test_trapezoidal_signal():
    """
    This function is used to test the trapezoidal signal.
    """
    signal = CommandSignal()
    x, y = [], []
    for time_count in range(4600):
        x.append(time_count)
        y.append(signal.trapezoidal(time_count * 0.0001))
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    test_trapezoidal_signal()
