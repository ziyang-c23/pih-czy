# from env_peg_in_hole import PegInHoleEnv
import PyTac3D

from scipy.spatial.transform import Rotation
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

plt.ion()  # 开启交互模式
current_file = Path(__file__).resolve()
data_dir = current_file.parent / "data"
data_dir.mkdir(exist_ok=True)
task_name = current_file.stem.replace("demo_", "")
task_dir = data_dir / task_name
task_dir.mkdir(exist_ok=True)
timestamp = "20260228-213103"  
run_dir = task_dir / timestamp
run_dir.mkdir(exist_ok=True)


tac3d_name1 = "HDL1-GWH0046"
tac3d_name2 = "HDL1-GWH0047"
sn1_dataloader = PyTac3D.DataLoader(SN=tac3d_name1, path=str(run_dir), skip=0)
sn2_dataloader = PyTac3D.DataLoader(SN=tac3d_name2, path=str(run_dir), skip=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 初始化空数据
sn1_positions = np.zeros((400, 3))
sn1_displacements = np.zeros((400, 3))

scat = ax.scatter(
    sn1_positions[:, 0],
    sn1_positions[:, 1],
    sn1_positions[:, 2],
    s=5
)

quiver = ax.quiver(
    sn1_positions[:, 0] - sn1_displacements[:, 0],
    sn1_positions[:, 1] - sn1_displacements[:, 1],
    sn1_positions[:, 2] - sn1_displacements[:, 2],
    sn1_displacements[:, 0],
    sn1_displacements[:, 1],
    sn1_displacements[:, 2],
    length=1,
    normalize=False,
    color='r'
)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(5, 15)
if __name__ == "__main__":
    while True:
        sn1_frame, sn1_frame_time, sn1_endflag = sn1_dataloader.get()
        sn2_frame, sn2_frame_time, sn2_endflag = sn2_dataloader.get()
        sn1_positions = np.array(sn1_frame["3D_Positions"])
        sn1_displacements = np.array(sn1_frame["3D_Displacements"])
        scat._offsets3d = (sn1_positions[:, 0], sn1_positions[:, 1] , sn1_positions[:, 2])

        # 删除旧箭头
        quiver.remove()

        # 画新箭头
        quiver = ax.quiver(
            sn1_positions[:, 0] - sn1_displacements[:, 0],
            sn1_positions[:, 1] - sn1_displacements[:, 1],
            sn1_positions[:, 2] - sn1_displacements[:, 2],
            sn1_displacements[:, 0],
            sn1_displacements[:, 1],
            sn1_displacements[:, 2],
            length=1,       # 可以调节箭头缩放
            normalize=False,
            color='b'
        )

        plt.pause(0.001)
        print(f"Tac3D Frame 1 time: {sn1_frame_time:.4f} s, Tac3D Frame 2 time: {sn2_frame_time:.4f} s")
