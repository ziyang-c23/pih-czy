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

# (pih-czy) ad102@ad102-Lenovo-Legion-Y7000P2020H:~/projects/czy$ python read_arm_pose.py 
# Initial end effector position: Affine(t=[0.302404 0.00115767 0.577608], q=[0.926212 -0.376861 -0.00644796 0.00808602])



# k = 0
# while(1):
#     k += 1
#     curr_pos = robot.current_cartesian_state.pose.end_effector_pose
#     print(f"current end effector position: {curr_pos}")
#     joint = robot.current_joint_state.position
#     print("joint:", joint)
#     input("Press Enter to continue or Ctrl+C to exit...")
#     T_curr = robot.current_cartesian_state.pose.end_effector_pose.translation
    
#     T_target = initial_postion.copy()
#     T_target[2] += k*0.002  # Move up by 2 mm

#     Q_curr = initial_rotation
#     robot.move(CartesianMotion(Affine(T_target, Q_curr)), asynchronous=True)