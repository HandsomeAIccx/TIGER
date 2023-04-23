import matplotlib.pyplot as plt

from src.control_env.servo_systems import ServoSystem
from src.control_env.pid import PID

def test_pid_control_servo_system():
    # env control
    rad_current_list = []
    rad_target_list = []
    env = ServoSystem()
    pid_agent = PID(Kp=0.5, Ki=2.985, Kd=0.0, dt=0.0001)
    obs = env.reset()
    for time in range(4400):
        rad_current_list.append(obs[0][1])
        rad_target_list.append(obs[0][2])
        action = pid_agent.action(error=obs[0][-1])
        obs = env.step(action)
    plt.plot(rad_current_list)
    plt.plot(rad_target_list)
    plt.show()

if __name__ == "__main__":
    test_pid_control_servo_system()
    