from env_peg_in_hole import PegInHoleEnv
from scipy.spatial.transform import Rotation
import time
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
data_dir = current_file.parent / "data"
data_dir.mkdir(exist_ok=True)
task_name = current_file.stem.replace("test_", "")
task_dir = data_dir / task_name
task_dir.mkdir(exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
run_dir = task_dir / timestamp
run_dir.mkdir(exist_ok=True)
print(f"Data will be saved to: {run_dir}")


if __name__ == "__main__":
    env = PegInHoleEnv(initial_pose=np.array([0.490, 0.001, 0.440, 0.924, -0.385, -0.000, 0.000]), initial_factor=0.01, 
                        k_estimator_mode=False, default_k=0.08, 
                        tac3d_name1="HDL1-GWH0046", tac3d_name2="HDL1-GWH0047",
                        grasping_force=10.0)
    # env = PegInHoleEnv(initial_pose=np.array([0.490, 0.001, 0.440, 1.000, -0.000, -0.000, 0.000]), initial_factor=0.002, 
    #                     k_estimator_mode=False, default_k=0.08, 
    #                     tac3d_name1="HDL1-GWH0046", tac3d_name2="HDL1-GWH0047",
    #                     grasping_force=10.0)
    env.client.contact(contact_speed=8, preload_force=2, quick_move_speed=15, quick_move_pos=10)
    env.client.grasp(goal_force=5.0, load_time=5.0)
    env.move(dx=0.0, dy=0.0, dz=-0.0, droll=5.0, dpitch=10.0, dyaw=-5.0, velocity=1.0, servo=False)
    time.sleep(1)
    goal_pressure_force = 1
    env.start_recording()
    while True:
        try:
            env.client.force_servo(goal_force=5.0)
            env.get_current_observation()
            f_ext = env._get_external_force()
            vz = -0.001 * (goal_pressure_force - f_ext[2])
            print(f"External force: {f_ext}, vz: {vz}")
            env.move_velocity(vx=0.0, vy=0.0, vz=vz, vroll=0.0, vpitch=0.0, vyaw=0.0, servo=True)
            time.sleep(0.005)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping environment movement.")
            env.stop_recording()
            env.save_data(savepath=run_dir)
            break
    env.close()