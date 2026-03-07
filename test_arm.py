from franky import *

robot = Robot("172.16.0.2")
initial_factor = 0.01
robot.relative_dynamics_factor = initial_factor
curr_pos = robot.current_cartesian_state.pose.end_effector_pose
initial_postion = curr_pos.translation.copy()
initial_rotation = curr_pos.quaternion.copy()
print(f"Initial end effector position: {curr_pos}")
print(f"Initial end effector position (translation): {initial_postion}")
print(f"Initial end effector position (quaternion): {initial_rotation}")
robot.move(CartesianMotion(Affine(initial_postion, [0.924, -0.385, -0.000, 0.000])), asynchronous=False)
curr_pos = robot.current_cartesian_state.pose.end_effector_pose
initial_postion = curr_pos.translation.copy()
initial_rotation = curr_pos.quaternion.copy()
print(f"Final end effector position: {curr_pos}")
print(f"Final end effector position (translation): {initial_postion}")
print(f"Final end effector position (quaternion): {initial_rotation}")
