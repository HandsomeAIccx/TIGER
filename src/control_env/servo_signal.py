class CommandSignal(object):
    """
    This class is used to generate the command signal.
    """

    def __init__(self) -> None:
        pass

    def trapezoidal(
        self, current_time, total_time=0.45, amplitude=1000, split_multiple=4
    ):
        """
        This function is used to generate the trapezoidal signal.

        Args:
            current_time (float): The current time.
            total_time (float): The total time of the signal.
            amplitude (float): The amplitude of the signal.
            split_multiple (int): The multiple phase of the signal.

        Returns:
            float: The value of the signal.
        """
        if current_time >= total_time:
            return 0.0
        if current_time <= total_time / split_multiple:
            return amplitude / (total_time / split_multiple) * current_time
        elif current_time <= total_time * (split_multiple - 1) / split_multiple:
            return amplitude
        else:
            assert (
                current_time <= total_time
            ), "current_time must be less than total_time"
            return (
                amplitude / (total_time / split_multiple) * (total_time - current_time)
            )
