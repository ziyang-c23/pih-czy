# controller0928.py — Phase-B always in compliance
import time
import numpy as np

from env_peg_in_hole import PegInHoleEnv

FIXED_DZ                        = -0.002
HSTEP_Y                         =  0.001
DY_CLIP                         =  0.0015
S_TRIGGER, S_HYST, S_FLIP       =  0.70, 0.02, 0.01   # 仍保留定义，但阶段B不再用它们做触发/翻转
CENTER_STEPS_MAX                =  5
GRASP_FORCE                     = 10.0
LUT_CSV                         = "lookup_table.csv"

# 用于判定 (Fy_L+Fy_R) 符号的抗噪阈值（根据现场噪声可调）
FY_EPS                          = 0.10

def _sign_eps(x: float, eps: float) -> int:
    """有阈值的符号函数：|x|<=eps 视为0"""
    return 1 if x > eps else (-1 if x < -eps else 0)

def main():
    env = PegInHoleEnv(initial_pose=None, grasping_force=10)
    env.reset()
    env._ee_pose_prev = env.robot.current_cartesian_state.pose.end_effector_pose

    env.load_ratio_S_LUT(LUT_CSV, ema_alpha=0.6)

    depth_accum_mm = 0.0
    S_prev = 1.0
    rand_active = False
    dy_sign = 0.0

    # ===== 阶段A：力差居中（保持不变） =====
    for k in range(CENTER_STEPS_MAX):
        Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R = env.read_pad_forces_base()
        print(f"in controller_center, pad forces are: L({Fx_L:.2f},{Fy_L:.2f},{Fz_L:.2f})  R({Fx_R:.2f},{Fy_R:.2f},{Fz_R:.2f})")
        dFx = Fx_L - Fx_R
        print(f"in controller_center, dFx is {dFx}")
        dx = 0.003 if dFx > 0 else -0.003
        dz = -0.001
        obs, done, info = env.step1_line_force_diff(
            [dx, 0.0, dz],
            grasp_force=GRASP_FORCE,
            fx_diff_delta_ratio=0.2, keep_sign=True,
            ds_max=1e-3, velocity=1.0
        )
        dz_mm = env.read_hand_real_dz_mm()
        depth_accum_mm += max(0.0, dz_mm)
        print(f"now doing center")
        time.sleep(0.02)
        print("-----------------next center step-------------------")

    # ===== 阶段B：始终顺应（按你的新逻辑） =====
    max_steps = 40
    last_cmd_dz_mm = 0.0
    last_real_dz_mm = 0.0

    dFx_prev = None
    fy_sum_sign_prev = None  # 记录上一帧 (Fy_L+Fy_R) 的“有阈值符号”

    for step in range(max_steps):
        # 读取受力
        Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R = env.read_pad_forces_base()
        dFx = Fx_L - Fx_R
        fy_sum = Fy_L + Fy_R
        fy_sign = _sign_eps(fy_sum, FY_EPS)

        # （仅用于打印的历史 actual_gt_cmd；不再用于控制是否顺应）
        actual_gt_cmd = abs(last_real_dz_mm) > abs(last_cmd_dz_mm)

        # 深度权
        w = env.depth_weight(depth_accum_mm)

        # 二阶段回中保险（保持不变，叠加在 dx 上）
        dx_cmd = 0.0
        if dFx_prev is not None:
            if (dFx * dFx_prev > 0.0) and (abs(dFx) > abs(dFx_prev)):
                dx_cmd = w * (0.002 if dFx > 0.0 else -0.002)
                print(f"doing center insurance in stage 2")

        # ——关键改动：阶段B“始终顺应”——
        prev_rand_active = rand_active
        rand_active = True   # 进入B就开启顺应，不再受 actual_gt_cmd 影响

        # 首次进入顺应：用 (Fy_L+Fy_R) 的符号来确定 dy_sign
        if rand_active and (not prev_rand_active):
            if fy_sign == 0:
                # 噪声区：沿用上一方向；若无，则默认 +1
                dy_sign = dy_sign if abs(dy_sign) > 1e-6 else +1.0
            else:
                dy_sign = -1.0 if fy_sign > 0 else +1.0
            fy_sum_sign_prev = fy_sign
            print(f"in controller stage2 [compliance enter]: Fy_sum={fy_sum:+.3f}N -> dy_sign={dy_sign:+.0f}")

        # 顺应进行中：当 (Fy_L+Fy_R) 的“有阈值符号”变号，则翻转方向
        if rand_active:
            if (fy_sum_sign_prev is not None) and (fy_sign * fy_sum_sign_prev < 0):
                dy_sign *= -1.0
                fy_sum_sign_prev = fy_sign
                print(f"in controller stage2 [compliance flip]: Fy_sum sign changed -> dy_sign={dy_sign:+.0f}")
            elif fy_sum_sign_prev is None and fy_sign != 0:
                fy_sum_sign_prev = fy_sign

        # —— 幅度由 S 与深度权 w 控制 ——（保持你的公式）
        S = float(np.clip(S_prev, 0.0, 0.99)); T = 1.0 - S
        S_eff = 1.0 - w * (1.0 - S)
        T_eff = w * (1.0 - S)

        # 始终顺应：dz 用 S_eff，dy 用 dy_sign*T_eff；dx 叠加“回中保险”
        dz = S_eff * FIXED_DZ
        dy = 0.0
        if dy_sign != 0.0:
            dy = float(np.clip(dy_sign * T_eff * HSTEP_Y, -DY_CLIP, +DY_CLIP))
            print(f"in pose control, dy is {dy}")
            dy = -4 * dy  # 保持原放大倍数
            print(f"in pose control, now dy is {dy}")
        dx = dx_cmd

        # 调试：再次打印受力（保持你原来的调试风格）
        Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R = env.read_pad_forces_base()
        print(f"in controller pose control, pad forces are: L({Fx_L:.2f},{Fy_L:.2f},{Fz_L:.2f})  R({Fx_R:.2f},{Fy_R:.2f},{Fz_R:.2f})")

        # 执行（保持原有阻塞式 move）
        env.move(dx, dy, dz, velocity=1.0, servo=False)

        # 观测、奖励（保持不变）
        obs = env.get_current_observation()
        rewards, task_weight, done = env._calculate_reward()
        last_cmd_dz_mm = dz * 1000.0

        ee_prev = env._ee_pose_prev
        env._ee_pose_prev = env.robot.current_cartesian_state.pose.end_effector_pose
        dz_mm = env.read_hand_real_dz_mm(prev_ee_pose=ee_prev, curr_ee_pose=env._ee_pose_prev)
        last_real_dz_mm = max(0.0, dz_mm)
        depth_accum_mm += last_real_dz_mm
        dFx_prev = dFx

        # 下压有效时更新 S（仅更新，不再用 dS 触发翻转/随机）
        if dz < -1e-6 and last_real_dz_mm > 0.02:
            Fx_L, Fy_L, Fz_L, Fx_R, Fy_R, Fz_R = env.read_pad_forces_base()
            ratio = abs((Fx_L + Fx_R)) / max(last_real_dz_mm, 1e-6)
            S_new = env.ratio_to_S(ratio)
            S_prev = S_new

        print(f"[{step:03d}] S={S:.3f} depth={depth_accum_mm/10:.2f} cm "
              f"cmd(dy,dz)=({dy * 1000:+.2f},{dz * 1000:+.2f})mm  "
              f"real_dz={last_real_dz_mm:+.3f}mm  gt={actual_gt_cmd}  "
              f"Fx(L,R)=({Fx_L:+.2f},{Fx_R:+.2f})N")

        if done:
            print("Episode done.")
            break

if __name__ == "__main__":
    main()
