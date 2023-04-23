import numpy as np

from src.control_env.pid import PID
from src.control_env.servo_systems import ServoSystem

def test_pid_control_servo_system():
    configs = {
        "simulate_times": 4600,
        "dt": 0.0001,
    }
    env = ServoSystem(configs)
    pid_agent = PID(Kp=0.5, Ki=2.985, Kd=0.0, dt=0.0001)
    obs = env.reset()
    for time in range(configs["simulate_times"]):
        env.render()
        action = pid_agent.action(error=obs[-1])
        obs = env.step(np.array([action], dtype=np.float32))
        

if __name__ == "__main__":
    test_pid_control_servo_system()
    