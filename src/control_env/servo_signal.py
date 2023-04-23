class CommandSignal(object):
    def __init__(self) -> None:
        pass

    def trapezoidal(self, current_time, total_time=0.45, amplitude=1000, split_multiple=4):
        if current_time <= total_time / split_multiple:
            return amplitude / (total_time / split_multiple) * current_time
        elif current_time <= total_time * (split_multiple - 1) / split_multiple:
            return amplitude
        else:
            assert current_time <= total_time, "current_time must be less than total_time"
            return amplitude / (total_time / split_multiple) * (total_time - current_time)
