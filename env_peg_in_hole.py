import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from franky import *
# import DexHandClient
from dexhand_client import DexHandClient
import PyTac3D


class Tac3DInfo:
    """Tac3D传感器数据结构"""

    def __init__(self, SN):
        self.SN = SN
        self.frameIndex = -1
        self.sendTimestamp = None
        self.recvTimestamp = None
        self.P = np.zeros((400, 3))
        self.D = np.zeros((400, 3))
        self.F = np.zeros((400, 3))
        self.Fr = np.zeros((1, 3))
        self.Mr = np.zeros((1, 3))
        self.Fr_base = np.zeros((1, 3))
        # 传感器左右顺序，根据实际调整
        self.right_sn = "HDL1-GWH0031-H"
        self.left_sn = "HDL1-GWH0032-H"
        # reset位置
        self.init_pose6 = None
        # 卡滞判据（已经不用了，但是为了和仿真对齐）
        self.stuck_force_tangential_thr = 6000.0


class PegInHoleEnv:
    """
    插孔任务环境，集成机器人、DexHand、Tac3D传感器。
    主要接口：reset(), step(action, ...), 属性访问（如initial_postion、obs等）
    """

    def __init__(self, initial_pose=None,  initial_factor=0.002, k_estimator_mode=False, default_k=0.08, tac3d_name1="HDL1-GWH0031-H", tac3d_name2="HDL1-GWH0032-H",grasping_force=10.0):
        self.initial_factor = initial_factor
        self.initial_postion = None
        self.initial_rotation = None
        self.grasping_force = grasping_force
        self.observation = None

        # --- 机器人初始化 ---
        self._init_robot(initial_pose, initial_factor=initial_factor)
        # --- 机械手初始化 ---
        self._init_hand(k_estimator_mode=k_estimator_mode, default_k=default_k)
        # --- Tac3D初始化 ---
        self._init_tac3d(tac3d_name1=tac3d_name1, tac3d_name2=tac3d_name2)
        # --- 观测与数据缓冲区 ---
        self._init_buffers()
        # --- 其他参数 ---
        self.force_threshold = 3.0
        self.action_scale = np.array([0.001, 0.001, 0.0015])
        self.step_count = 0
        self.distance_ini = None
        self.observation_last = None

        self._ee_pose_prev = None

    def _init_robot(self, initial_pose, initial_factor=0.002):
        self.robot = Robot("172.16.0.2")
        self.initial_factor = initial_factor
        self.robot.relative_dynamics_factor = self.initial_factor
        if initial_pose is not None:
            self.robot.move(CartesianMotion(Affine(initial_pose[:3], initial_pose[3:])))
        self.curr_pos = self.robot.current_cartesian_state.pose.end_effector_pose
        self.initial_postion = self.curr_pos.translation.copy()
        self.initial_rotation = self.curr_pos.quaternion.copy()
        print(f"Initial end effector position: {self.curr_pos}")
        time.sleep(2.0)

    def _init_hand(self, k_estimator_mode=False, default_k=0.08):
        self.hand_obs = pd.DataFrame(columns=['nowtime', 'nowforce', 'nowforce1', 'nowforce2', 'nowpos', 'nowstiffness'])
        self.client = DexHandClient(ip="192.168.2.100", port=60031, recvCallback_hand=self._hand_recv_callback)
        self.client.stop_server()
        self.client.start_server()
        self.client.acquire_hand()
        self.client.set_home(goal_speed=8)
        self.client.calibrate_force_zero()
        if k_estimator_mode:
            self.client.switch_k_mode(use_estimator=True, default_k=default_k)   
        else:
            self.client.switch_k_mode(use_estimator=False, default_k=default_k)
        time.sleep(3.0)

        # 暂不需要

    def _init_tac3d(self, tac3d_name1="HDL1-GWH0031-H", tac3d_name2="HDL1-GWH0032-H"):
        self.tac_obs = {}
        self.tac3d_name1 = tac3d_name1
        self.tac3d_name2 = tac3d_name2
        print(f"Initializing Tac3D sensors: {self.tac3d_name1}, {self.tac3d_name2}")
        self.tacinfo1 = Tac3DInfo(self.tac3d_name1)
        self.tacinfo2 = Tac3DInfo(self.tac3d_name2)
        self.tac_dict = {self.tac3d_name1: self.tacinfo1, self.tac3d_name2: self.tacinfo2}
        self.tac3d = PyTac3D.Sensor(recvCallback=self._tac3d_recv_callback, port=9987, maxQSize=5,
                                    callbackParam=self.tac_dict)
        time.sleep(5.0)
        self.recorder1 = PyTac3D.DataRecorder(self.tac3d_name1)
        self.recorder2 = PyTac3D.DataRecorder(self.tac3d_name2)
        self.tac3d.calibrate(self.tac3d_name1)
        self.tac3d.calibrate(self.tac3d_name2)

    def _init_buffers(self):

        self.obs_buffer = {
            "ee_pose": [],
            "hand_force": [],
            "tac3d_1": [],
            "tac3d_2": []
        }

    # ==================== 回调函数 ====================
    def _hand_recv_callback(self, client: DexHandClient):
        info = client.hand_info
        now_avg_force = info.avg_force
        now_force1 = info.now_force[0]
        now_force2 = info.now_force[1]
        now_pos = info.now_pos
        now_stiffness = info.stiffness
        now_time = time.time()
        # Save to DataFrame
        if hasattr(self, 'start_record') and self.start_record and not self.end_record:
            new_row = pd.DataFrame([[now_time, now_avg_force, now_force1, now_force2, now_pos, now_stiffness]],
                                   columns=['nowtime', 'nowforce', 'nowforce1', 'nowforce2', 'nowpos', 'nowstiffness'])
            self.hand_obs = pd.concat([self.hand_obs, new_row], ignore_index=True)

    def _tac3d_recv_callback(self, frame, param):
        SN = frame["SN"]
        tacinfo = param[SN]
        tacinfo.frameIndex = frame["index"]
        tacinfo.sendTimestamp = frame["sendTimestamp"]
        tacinfo.recvTimestamp = frame["recvTimestamp"]
        tacinfo.P = frame.get("3D_Positions")
        tacinfo.D = frame.get("3D_Displacements")
        tacinfo.F = frame.get("3D_Forces")
        tacinfo.Fr = frame.get("3D_ResultantForce")
        tacinfo.Mr = frame.get("3D_ResultantMoment")
        # 坐标变换
        Fr_base = np.zeros(3)
        if tacinfo.Fr is not None:
            ee_pose = self.robot.current_cartesian_state.pose.end_effector_pose
            ee_quat = ee_pose.quaternion
            ee_rot = Rotation.from_quat(ee_quat).as_matrix()
            tac_force = tacinfo.Fr.reshape(3)
            if SN == self.tac3d_name1:
                tac_to_ee = np.array([[0,       0.7071,     -0.7071], 
                                      [0.0,     0.7071,     0.7071], 
                                      [1,       0.,         0]])
            elif SN == self.tac3d_name2:
                tac_to_ee = np.array([[0,       -0.7071,    0.7071], 
                                      [0.0,     -0.7071,    -0.7071], 
                                      [1,       0.,         0]])
            else:
                tac_to_ee = np.eye(3)
            tac_force_ee = tac_to_ee @ tac_force
            Fr_base = ee_rot @ tac_force_ee
        tacinfo.Fr_base = Fr_base
        self.tac_obs[SN] = {
            "frameIndex": frame["index"],
            "sendTimestamp": frame["sendTimestamp"],
            "recvTimestamp": frame["recvTimestamp"],
            "P": frame.get("3D_Positions"),
            "D": frame.get("3D_Displacements"),
            "F": frame.get("3D_Forces"),
            "Fr": frame.get("3D_ResultantForce"),
            "Mr": frame.get("3D_ResultantMoment"),
            "Fr_base": Fr_base
        }
        if hasattr(self, 'start_record') and self.start_record and not self.end_record:
            if frame['SN'] == self.tac3d_name1:
                self.recorder1.put(frame)
            elif frame['SN'] == self.tac3d_name2:
                self.recorder2.put(frame)

    # ==================== 主要接口 ====================
    def reset(self, initial_pose=None):
        """环境复位，重置机器人、机械手、Tac3D、观测等"""
        self.client.grasp(goal_force=self.grasping_force)
        if initial_pose is not None:
            self.robot.move(CartesianMotion(Affine(initial_pose[:3], initial_pose[3:])))
        else:
            self.robot.move(CartesianMotion(Affine(self.initial_postion, self.initial_rotation)))

        self.tac_obs.clear()
        self.recorder1.clear()
        self.recorder2.clear()
        self.hand_obs = pd.DataFrame(columns=['nowtime', 'nowforce', 'nowforce1', 'nowforce2', 'nowpos', 'nowstiffness'])
        self.observation = None
        self._init_buffers()
        time.sleep(2.0)


    def step1_line_force_diff(self, action_xyz, droll=0.0, grasp_force=10.0,
                              fx_diff_delta_ratio=0.2, keep_sign=True,
                              ds_max=1e-3,
                              max_iter=4000, velocity=1.0):
        """
        底层连续控制：沿增量方向规定推进最大值；使切向力差减小一定比例；保正负号
        """
        import numpy as np
        '''记录迭代起点'''
        ee_prev = self.robot.current_cartesian_state.pose.end_effector_pose
        v = np.asarray(action_xyz, dtype=float).reshape(3)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-12:
            return self.get_current_observation(), False, {'fx_ok': True, 'sign_ok': True, 's_travel': 0.0}
        uhat = v / v_norm
        max_s = v_norm
        Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R = self.read_pad_forces_base()
        print(f"in center, all pad forces are {Fx_L}, {Fy_L}, {Fz_L}, {Fx_R}, {Fy_R}, {Fz_R}")
        fx_raw0 = Fx_L - Fx_R
        print(f"in center, force difference is {fx_raw0}")
        fx_abs0 = abs(fx_raw0)
        fx_abs_target = max(fx_abs0 * (1.0 - fx_diff_delta_ratio), 0.0)
        sign0 = np.sign(fx_raw0) if fx_abs0 > 1e-4 else 0.0
        s = 0.0
        info = {'fx_ok': False, 'sign_ok': True, 's_travel': 0.0}
        for itr in range(int(max_iter)):
            self.client.force_servo(goal_force=grasp_force)
            ds = min(ds_max, max_s - s)
            if ds <= 1e-9: break
            d = uhat * ds
            print(f"in iteration of center, moving {d[0]}, {d[1]}, {d[2]}")
            self.move(float(d[0]), float(d[1]), float(d[2]), velocity=velocity, servo=False)
            # self.move(0.00, -0.002, 0.00, velocity=velocity, servo=False)
            obs = self.get_current_observation()
            Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R = self.read_pad_forces_base()
            fx_now = Fx_L - Fx_R
            fx_abs_now = abs(fx_now)
            fx_ok = (fx_abs_now <= fx_abs_target + max(1e-3, 0.02 * fx_abs0))
            sign_ok = True
            if keep_sign and sign0 != 0.0:
                sign_ok = (np.sign(fx_now) == sign0) and (fx_abs_now > 1e-4)
            if fx_ok and sign_ok:
                info.update({'fx_ok': True, 'sign_ok': True, 's_travel': s + ds})
                break
            s += ds
            if s >= max_s - 1e-9:
                info.update({'fx_ok': fx_ok, 'sign_ok': sign_ok, 's_travel': s})
                break
        print(f"in center, before manipulation pose is {ee_prev}")
        print(f"in center, after manipulation pose is {self.robot.current_cartesian_state.pose.end_effector_pose}")
        self._ee_pose_prev = self.robot.current_cartesian_state.pose.end_effector_pose
        return obs, False, info

    def step(self, action, velocity=1.0, k_f=np.array([0.001, 0.001, 0.0005]), arm_servo=False):
        """执行一步动作，返回观测、奖励、done、动作、权重"""
        self.client.force_servo(goal_force=self.grasping_force)
        self.get_current_observation()
        external_force = self._get_external_force()
        displacement_main = np.array(action)
        adjustment = external_force * k_f
        max_adjustment = 0.6 * np.linalg.norm(displacement_main)
        if np.linalg.norm(adjustment) > max_adjustment and np.linalg.norm(adjustment) > 0:
            adjustment = adjustment / np.linalg.norm(adjustment) * max_adjustment
        displacement_adjusted = displacement_main + adjustment
        action_modified = displacement_adjusted / np.linalg.norm(displacement_adjusted) * self.action_scale
        # 保证action_modified的各个分量的绝对值不低于0.0004
        min_val = 0.0004
        action_modified = np.where(np.abs(action_modified) < min_val, np.sign(action_modified) * min_val,
                                   action_modified)

        self.move(action_modified[0], action_modified[1], action_modified[2], velocity=velocity, servo=arm_servo)

        obs = self.get_current_observation()
        rewards, task_weight, done = self._calculate_reward()
        self.step_count += 1
        Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R = self.read_pad_forces_base()
        print(f"in pose_control step, forces are {Fx_L}, {Fy_L}, {Fz_L}, {Fx_R}, {Fy_R}, {Fz_R}")
        return obs, rewards, done, action_modified, task_weight

    def move(self, dx=0.0, dy=0.0, dz=0.0, droll=0.0, dpitch=0.0, dyaw=0.0, velocity=1.0, servo=False):
        """机器人运动控制接口，支持增量位置和增量姿态控制
        param dx: x轴增量，单位米
        param dy: y轴增量，单位米
        param dz: z轴增量，单位米
        param droll: 绕x轴增量，单位度
        param dpitch: 绕y轴增量，单位度
        param dyaw: 绕z轴增量，单位度
        """
        curr_pos = self.robot.current_cartesian_state.pose.end_effector_pose
        T_curr = curr_pos.translation
        Q_curr_xyzw = curr_pos.quaternion

        T_target = T_curr.copy()
        T_target[0] += dx
        T_target[1] += dy
        T_target[2] += dz

        rot_curr = Rotation.from_quat(Q_curr_xyzw)
        deuler = np.array([droll, dpitch, dyaw])
        drot = Rotation.from_euler('xyz', deuler, degrees=True)
        Q_target_xyzw = (drot * rot_curr).as_quat()
        # print(f"Moving from T {T_curr} to {T_target}, Q {Q_curr_xyzw} to {Q_target}, velocity {velocity}, servo {servo}")

        self.robot.relative_dynamics_factor = self.initial_factor * velocity
        if servo:
            self.robot.move(CartesianMotion(Affine(T_target, Q_target_xyzw)), asynchronous=True)
        else:
            self.robot.move(CartesianMotion(Affine(T_target, Q_target_xyzw)))
        self.robot.relative_dynamics_factor = self.initial_factor

    def move_velocity(self, vx=0.0, vy=0.0, vz=0.0, vroll=0.0, vpitch=0.0, vyaw=0.0, servo=False):
        """机器人速度控制接口，支持线速度和角速度控制
        param vx: x轴线速度，单位m/s
        param vy: y轴线速度，单位m/s
        param vz: z轴线速度，单位m/s
        param vroll: 绕x轴角速度，单位rad/s
        param vpitch: 绕y轴角速度，单位rad/s
        param vyaw: 绕z轴角速度，单位rad/s
        """
        move = CartesianVelocityMotion(Twist(linear_velocity=[vx, vy, vz], angular_velocity=[vroll, vpitch, vyaw]))
        if servo:
            self.robot.move(move, asynchronous=True)
        else:
            self.robot.move(move, asynchronous=False)

    def get_current_observation(self):
        """获取当前观测"""
        if self.tac3d_name1 in self.tac_obs and self.tac3d_name2 in self.tac_obs:
            ee_pose = self.robot.current_cartesian_state.pose.end_effector_pose
            hand_force = self.hand_obs['nowforce'].iloc[-1] if not self.hand_obs.empty else 0.0
            obs = {
                'ee_pose': ee_pose,
                'hand_force': hand_force,
                'tac3d_1': self.tac_obs[self.tac3d_name1],
                'tac3d_2': self.tac_obs[self.tac3d_name2]
            }
            self.obs_buffer["ee_pose"].append(ee_pose)
            self.obs_buffer["hand_force"].append(hand_force)
            self.obs_buffer["tac3d_1"].append(self.tac_obs[self.tac3d_name1])
            self.obs_buffer["tac3d_2"].append(self.tac_obs[self.tac3d_name2])
            # self.observation_last = self.observation
            self.observation = obs
            return obs
        else:
            missing = []
            if self.tac3d_name1 not in self.tac_obs:
                missing.append(self.tac3d_name1)
            if self.tac3d_name2 not in self.tac_obs:
                missing.append(self.tac3d_name2)
            print(f"无法获取以下触觉传感器数据: {', '.join(missing)}")
            return None

    def read_pad_forces_base(self):
        """返回世界系下的六维力"""
        if not self.obs_buffer["tac3d_1"] or not self.obs_buffer["tac3d_2"]:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        tac1 = self.obs_buffer["tac3d_1"][-1]
        tac2 = self.obs_buffer["tac3d_2"][-1]
        F1 = tac1.get("Fr_base", np.zeros(3))
        F2 = tac2.get("Fr_base", np.zeros(3))
        Fx_L, Fy_L, Fz_L = float(F1[2]), float(F1[1]), float(F1[0])         #这里维度很奇怪，但是根据夹持力合力等可以判断，第三个维度为切向力竖直分量，第一个维度为法向力
        Fx_R, Fy_R, Fz_R = float(F2[2]), float(F2[1]), float(F2[0])
        return Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R

    def read_hand_real_dz_mm(self, prev_ee_pose=None, curr_ee_pose=None):
        """返回当前步的末端执行器实际下压量(mm，向下为正)"""
        if prev_ee_pose is None:
            prev_ee_pose = getattr(self, "_ee_pose_prev", None)
        if curr_ee_pose is None:
            curr_ee_pose = self.robot.current_cartesian_state.pose.end_effector_pose
        if prev_ee_pose is None:
            return 0.0
        z_prev = float(prev_ee_pose.translation[2])
        z_now = float(curr_ee_pose.translation[2])
        dz_mm = (z_prev - z_now) * 1000.0  # 向下为正
        return dz_mm

    def load_ratio_S_LUT(self, csv_path, ema_alpha=0.6):
        """查表，记录EMA参数"""
        import pandas as pd
        lut = pd.read_csv(csv_path)
        self._lut_ratio = lut.iloc[:, 0].to_numpy(dtype=float)
        self._lut_S = lut.iloc[:, 1].to_numpy(dtype=float)
        self._S_prev = 1.0
        self._S_ema_a = float(ema_alpha)

    def ratio_to_S(self, ratio):
        """线性插值+EMA"""
        if ratio <= self._lut_ratio[0]:
            S_lut = float(self._lut_S[0])
        elif ratio >= self._lut_ratio[-1]:
            S_lut = float(self._lut_S[-1])
        else:
            import numpy as np
            i = np.searchsorted(self._lut_ratio, ratio) - 1
            r0, r1 = self._lut_ratio[i], self._lut_ratio[i + 1]
            s0, s1 = self._lut_S[i], self._lut_S[i + 1]
            t = (ratio - r0) / max(r1 - r0, 1e-12)
            S_lut = float((1 - t) * s0 + t * s1)
        # EMA
        a = self._S_ema_a
        self._S_prev = (1 - a) * self._S_prev + a * S_lut
        return self._S_prev

    def depth_weight(self, depth_accum_mm, x0_cm=2, tau_cm=1):
        x_cm = max(0.0, depth_accum_mm / 10.0)
        import numpy as np
        return 0.5 * (1.0 - np.tanh((x_cm - x0_cm) / max(tau_cm, 1e-6)))

    # ==================== 奖励与观测辅助 ====================
    def _calculate_reward(self):
        object_pos, _ = self._get_object_pose()
        ext_force = self._get_external_force()
        ee_pose = self.robot.current_cartesian_state.pose.end_effector_pose.translation
        hand_pos = ee_pose.copy()
        # 获取当前物体位置，暂时用机械手末端代替
        object_pos = hand_pos
        current_distance = np.linalg.norm(object_pos - hand_pos)
        dis_error = np.abs(self.distance_ini - current_distance) * 1000  # [mm]

        # 位姿奖励（以z轴变化为主）
        # 这里只用z轴变化，实际可根据需求调整
        object_pos_last = getattr(self, 'prev_pos', object_pos)
        position_reward = -2500 * (object_pos[2] - object_pos_last[2])
        position_reward = np.sign(position_reward) * min(5, abs(position_reward))
        # print(f"object_pos: {object_pos}, object_pos_last: {object_pos_last}, position_reward: {position_reward}")
        task_reward = position_reward

        # 接触约束奖励（外力变化）
        ext_force_last = getattr(self, 'prev_ext_force', object_pos)
        contact_reward = -10 * (np.linalg.norm(ext_force[:2]) - np.linalg.norm(ext_force_last[:2]))
        contact_reward = np.sign(contact_reward) * min(5, abs(contact_reward))
        # print(f"ext_force: {ext_force}, ext_force_last: {ext_force_last}, contact_reward: {contact_reward}")
        constriant_reward = contact_reward

        # 约束权重
        force_xy = np.linalg.norm(ext_force[:2])
        force_threshold = self.force_threshold
        min_threshold = 0.5 * force_threshold
        if force_xy < min_threshold:
            constraint_weight = 0.0
        else:
            constraint_weight = (force_xy - min_threshold) / (force_threshold - min_threshold)
            constraint_weight = np.clip(constraint_weight, 0.0, 1.0)
        task_weight = 1.0 - constraint_weight

        done = False
        if object_pos[2] < -0.0 and dis_error < 20:
            done = True
        if dis_error > 20:
            done = True
        if self.step_count > 20:
            done = True
        self.prev_pos = object_pos.copy()
        self.prev_ext_force = ext_force.copy()
        return np.array([task_reward, constriant_reward]), task_weight, done

    def _get_object_pose(self):
        if not self.obs_buffer["ee_pose"]:
            raise RuntimeError("obs_buffer['ee_pose'] is empty!")
        hand_pose = self.obs_buffer["ee_pose"][-1]
        hand_position = hand_pose.translation
        # hand_orientation = hand_pose.translation
        object_position = hand_position.copy()
        return object_position, np.array([0.0, 0.0, 0.0])

    def _get_external_force(self):
        tac1 = self.obs_buffer["tac3d_1"][-1]
        tac2 = self.obs_buffer["tac3d_2"][-1]
        self.ext_force = tac1["Fr_base"] + tac2["Fr_base"] if "Fr_base" in tac1 and "Fr_base" in tac2 else np.zeros(3)
        # tac1_last = self.obs_buffer["tac3d_1"][-1]
        # tac2_last = self.obs_buffer["tac3d_2"][-2]
        # self.prev_ext_force = tac1_last["Fr_base"] + tac2_last["Fr_base"] if "Fr_base" in tac1_last and "Fr_base" in tac2_last else np.zeros(3)
        return self.ext_force

    # ==================== 数据记录与资源管理 ====================
    def start_recording(self):
        self.start_record = True
        self.end_record = False
        print("开始记录数据...")

    def stop_recording(self):
        """停止记录数据"""
        self.start_record = False
        self.end_record = True
        print("停止记录数据...")
    
    def save_data(self, savepath=None, filename=None):
        """保存记录的数据到文件"""
        if filename is None:
            filename = "dexhand.csv"

        self.recorder1.save(path=str(savepath))
        self.recorder2.save(path=str(savepath))

        if not self.hand_obs.empty:
            self.hand_obs.to_csv(str(savepath / filename), index=False)

    def close(self):
        """环境关闭，重置机器人、机械手、Tac3D、观测等"""
        time.sleep(2)
        self.hand_obs = pd.DataFrame(columns=['nowtime', 'nowforce', 'nowforce1', 'nowforce2', 'nowpos', 'nowstiffness'])
        self.tac_obs.clear()
        self.recorder1.clear()
        self.recorder2.clear()
        self._init_buffers()
        self.get_current_observation()
        self.distance_ini = 0.0
        self.step_count = 0
        self.observation = None
        self.observation_last = None
        try:
            self.client.pos_goto(goal_pos=2.0)  # 重置机械手位置
            self.client.release_hand()
        except Exception:
            print("Exception: Failed to release hand.")
            pass
        try:
            self.robot.move(CartesianMotion(Affine(self.initial_postion, self.initial_rotation)))  # 重置机器人位置
        except Exception:
            print("Exception: Failed to reset robot position.")

if __name__ == "__main__":
    env = PegInHoleEnv(initial_pose=np.array([0.490, 0.001, 0.440, 0.924, -0.385, -0.000, 0.000]), initial_factor=0.002, 
                        k_estimator_mode=False, default_k=0.08, 
                        tac3d_name1="HDL1-GWH0046", tac3d_name2="HDL1-GWH0047",
                        grasping_force=10.0)
    # env = PegInHoleEnv(initial_pose=np.array([0.490, 0.001, 0.440, 1.000, -0.000, -0.000, 0.000]), initial_factor=0.002, 
    #                     k_estimator_mode=False, default_k=0.08, 
    #                     tac3d_name1="HDL1-GWH0046", tac3d_name2="HDL1-GWH0047",
    #                     grasping_force=10.0)
    while True:
        try:
            # env.move(dx=0.002, dy=0.0, dz=0.0, velocity=0.5, servo=True)
            time.sleep(0.1)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Stopping environment movement.")
            break
    env.close()