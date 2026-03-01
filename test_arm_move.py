from env_peg_in_hole import PegInHoleEnv
from config_loader import get_config, load_config
import time
import numpy as np

cfg = get_config()
INITIAL_POSE = np.array(cfg.init_params.initial_pose)
INITIAL_FACTOR = cfg.init_params.initial_factor
K_ESTIMATOR_MODE = cfg.init_params.k_estimator_mode
DEFAULT_K = cfg.init_params.default_k
TAC3D_NAME1 = cfg.init_params.tac3d_name1
TAC3D_NAME2 = cfg.init_params.tac3d_name2
GRASPING_FORCE = cfg.init_params.grasping_force

if __name__ == "__main__":

    env = PegInHoleEnv(initial_pose=INITIAL_POSE, initial_factor=INITIAL_FACTOR, 
                        k_estimator_mode=K_ESTIMATOR_MODE, default_k=DEFAULT_K, 
                        tac3d_name1=TAC3D_NAME1, tac3d_name2=TAC3D_NAME2,
                        grasping_force=GRASPING_FORCE)
    # env = PegInHoleEnv(initial_pose=np.array([0.490, 0.001, 0.440, 1.000, -0.000, -0.000, 0.000]), initial_factor=0.002, 
    #                     k_estimator_mode=False, default_k=0.08, 
    #                     tac3d_name1="HDL1-GWH0046", tac3d_name2="HDL1-GWH0047",
    #                     grasping_force=10.0)
    while True:
        try:
            # env.move(dx=0.002, dy=0.0, dz=0.0, velocity=0.5, servo=True)
            # env.move(dx=0.0, dy=0.002, dz=-0.0, velocity=0.5, servo=True)
            # env.move(dx=0.0, dy=0.0, dz=-0.002, velocity=0.5, servo=True)
            # env.move(droll=5.0, dpitch=0.0, dyaw=0.0, velocity=0.5, servo=True)
            # env.move(droll=0.0, dpitch=5.0, dyaw=0.0, velocity=0.5, servo=True)
            # env.move(droll=0.0, dpitch=0.0, dyaw=10.0, velocity=0.5, servo=True)
            # env.move_velocity(vx=0.001, vy=0.0, vz=0.0, vroll=0.0, vpitch=0.0, vyaw=0.0,  servo=True)
            # env.move_velocity(vx=0.0, vy=0.001, vz=0.0, vroll=0.0, vpitch=0.0, vyaw=0.0,  servo=True)
            # env.move_velocity(vx=0.0, vy=0.0, vz=0.001, vroll=0.0, vpitch=0.0, vyaw=0.0,  servo=True)
            # env.move_velocity(vx=0.0, vy=0.0, vz=0.0, vroll=0.01, vpitch=0.0, vyaw=0.0,  servo=True)
            # env.move_velocity(vx=0.0, vy=0.0, vz=0.0, vroll=0.0, vpitch=0.001, vyaw=0.0,  servo=True)
            # env.move_velocity(vx=0.0, vy=0.0, vz=0.0, vroll=0.0, vpitch=0.0, vyaw=0.001,  servo=True)

            time.sleep(0.1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping environment movement.")
            break
    env.close()