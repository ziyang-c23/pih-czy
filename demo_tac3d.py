# from env_peg_in_hole import PegInHoleEnv
import PyTac3D

import time
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pathlib import Path
from config_loader import get_config

cfg = get_config()
TAC3D_NAME1 = cfg.init_params.tac3d_name1
TAC3D_NAME2 = cfg.init_params.tac3d_name2

class Tac3D_info:
    def __init__(self, SN):
        self.SN = SN  # 传感器SN
        self.frameIndex = -1  # 帧序号
        self.sendTimestamp = None  # 时间戳
        self.recvTimestamp = None  # 时间戳
        self.P = np.zeros((400, 3))  # 三维形貌测量结果，400行分别对应400个标志点，3列分别为各标志点的x，y和z坐标
        self.D = np.zeros((400, 3))  # 三维变形场测量结果，400行分别对应400个标志点，3列分别为各标志点的x，y和z变形
        self.F = np.zeros((400, 3))  # 三维分布力场测量结果，400行分别对应400个标志点，3列分别为各标志点的x，y和z受力
        self.Fr = np.zeros((1, 3))  # 整个传感器接触面受到的x,y,z三维合力
        self.Mr = np.zeros((1, 3))  # 整个传感器接触面受到的x,y,z三维合力矩

class Tac3D_GUI:
    def __init__(self):
        self.app = gui.Application.instance
        self.app.initialize()
        self.win = self.app.create_window("PyTac3D Visualization", 1280, 720)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.win.renderer)

        # 设置背景色和初始视角
        self.scene_widget.scene.set_background([0.15, 0.15, 0.15, 1.0]) # 背景色
        center = np.array([0, 0, 10], dtype=np.float32)  # 观察点
        eye = np.array([17, 0, 16], dtype=np.float32)     # 相机位置
        up = np.array([0, 0, 1], dtype=np.float32)        # 朝上方向
        self.scene_widget.look_at(center, eye, up)
        self.win.add_child(self.scene_widget)

        # 添加信息标签
        self.esc_text = "Press 'esc' to exit. \n"
        self.i_text = "Press 'i' to record initial displacements. \n"
        text = "Status: Initializing...\n" + self.esc_text + self.i_text
        self.info_label = gui.Label(text)
        self.info_label.text_color = gui.Color(1.0, 1.0, 1.0) # 白色文字
        self.win.add_child(self.info_label)
        self.win.set_on_layout(self.on_layout)

        # 设置点云和线段的渲染属性
        self.pcd_mat = rendering.MaterialRecord()
        self.pcd_mat.shader = "defaultUnlit"
        self.pcd_mat.point_size = 5.0  

        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 3.0

        self.grid = create_grid_bbox(        
            x_range=(-10, 10),
            y_range=(-10, 10), 
            z_range=(5, 15),
            step=2.0  # 每2个单位一条网格线
        )    # 创建网格线对象
        self.add_line_geometry("grid", self.grid) 

    def on_layout(self, layout_context):
        rect = self.win.content_rect
        self.scene_widget.frame = rect
        label_size = self.info_label.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.info_label.frame = gui.Rect(rect.x + 10, rect.y + 10, label_size.width, label_size.height)

    def add_pcd_geometry(self, name, pcd):
        self.scene_widget.scene.add_geometry(name, pcd, self.pcd_mat)

    def add_line_geometry(self, name, line_set):
        self.scene_widget.scene.add_geometry(name, line_set, self.line_mat)

    def update_pcd_geometry(self, name, pcd):
        self.scene_widget.scene.remove_geometry(name)
        self.scene_widget.scene.add_geometry(name, pcd, self.pcd_mat)
    
    def update_line_geometry(self, name, line_set):
        self.scene_widget.scene.remove_geometry(name)
        self.scene_widget.scene.add_geometry(name, line_set, self.line_mat)

    def update_info_label(self, text):
        self.info_label.text = text

    def refresh(self):
        self.win.post_redraw()
        self.app.run_one_tick()

    def quit(self):
        self.app.quit()

def create_grid_bbox(x_range=(-10, 10), y_range=(-10, 10), z_range=(5, 15), step=2.0):
    """
    创建类似 plt 的 3D 网格线框
    包含：边界框 + 网格线 + 坐标轴
    """
    lines = []
    colors = []
    
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    # 颜色定义
    color_bbox = [0.5, 0.5, 0.5]    # 灰色边界
    color_grid_xy = [0.3, 0.3, 0.3] # 深灰色网格
    color_grid_xz = [0.3, 0.3, 0.3]
    color_grid_yz = [0.3, 0.3, 0.3]
    color_axis_x = [1, 0, 0]        # 红色X轴
    color_axis_y = [0, 1, 0]        # 绿色Y轴
    color_axis_z = [0, 0, 1]        # 蓝色Z轴
    
    # ========== 1. 绘制边界框 (12条边) ==========
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])
    
    # 底面
    lines.extend([[0,1], [1,2], [2,3], [3,0]])
    # 顶面
    lines.extend([[4,5], [5,6], [6,7], [7,4]])
    # 垂直边
    lines.extend([[0,4], [1,5], [2,6], [3,7]])
    colors.extend([color_bbox] * 12)
    
    # ========== 2. 绘制XY平面网格 (Z=z_min 和 Z=z_max) ==========
    for z in [z_min, z_max]:
        # X方向线
        for y in np.arange(y_min, y_max + step, step):
            lines.append([len(corners), len(corners) + 1])
            corners = np.vstack([corners, [[x_min, y, z], [x_max, y, z]]])
            colors.append(color_grid_xy)
        # Y方向线
        for x in np.arange(x_min, x_max + step, step):
            lines.append([len(corners), len(corners) + 1])
            corners = np.vstack([corners, [[x, y_min, z], [x, y_max, z]]])
            colors.append(color_grid_xy)
    
    # ========== 3. 绘制XZ平面网格 (Y=y_min 和 Y=y_max) ==========
    for y in [y_min, y_max]:
        # X方向线
        for z in np.arange(z_min, z_max + step, step):
            lines.append([len(corners), len(corners) + 1])
            corners = np.vstack([corners, [[x_min, y, z], [x_max, y, z]]])
            colors.append(color_grid_xz)
        # Z方向线
        for x in np.arange(x_min, x_max + step, step):
            lines.append([len(corners), len(corners) + 1])
            corners = np.vstack([corners, [[x, y, z_min], [x, y, z_max]]])
            colors.append(color_grid_xz)
    
    # ========== 4. 绘制YZ平面网格 (X=x_min 和 X=x_max) ==========
    for x in [x_min, x_max]:
        # Y方向线
        for z in np.arange(z_min, z_max + step, step):
            lines.append([len(corners), len(corners) + 1])
            corners = np.vstack([corners, [[x, y_min, z], [x, y_max, z]]])
            colors.append(color_grid_yz)
        # Z方向线
        for y in np.arange(y_min, y_max + step, step):
            lines.append([len(corners), len(corners) + 1])
            corners = np.vstack([corners, [[x, y, z_min], [x, y, z_max]]])
            colors.append(color_grid_yz)
    
    # ========== 5. 绘制坐标轴 (在角落，长度=step) ==========
    origin = np.array([x_min, y_min, z_min])
    # X轴
    lines.extend([[len(corners), len(corners) + 1]])
    corners = np.vstack([corners, [origin, origin + [step, 0, 0]]])
    colors.append(color_axis_x)
    # Y轴
    lines.extend([[len(corners), len(corners) + 1]])
    corners = np.vstack([corners, [origin, origin + [0, step, 0]]])
    colors.append(color_axis_y)
    # Z轴
    lines.extend([[len(corners), len(corners) + 1]])
    corners = np.vstack([corners, [origin, origin + [0, 0, step]]])
    colors.append(color_axis_z)
    
    # 创建 LineSet
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(corners)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector(colors)
    
    return grid

def update_point_cloud(pcd_obj, positions):
    """更新点云数据，不创建新对象"""
    if len(positions) > 0:
        pcd_obj.points = o3d.utility.Vector3dVector(positions)
        pcd_obj.paint_uniform_color([0, 1, 0])  # 绿色
    return pcd_obj

def update_lines(line_set_obj, positions, vectors, color, scale=1.0):
    """更新线段数据，不创建新对象"""
    if len(positions) == 0 or len(vectors) == 0:
        # 如果数据为空，设置一个虚拟点避免报错
        line_set_obj.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
        line_set_obj.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set_obj.paint_uniform_color(color)
        return line_set_obj
    
        # 向量化计算起点和终点
    starts = positions
    if np.any(np.abs(vectors) > 1e-6):  # 检查是否有非零位移
        ends = positions + vectors * scale
    else:
        ends = positions + np.ones_like(positions) * 0.001  # 微小偏移避免零长度
    
    # 交错排列: [start0, end0, start1, end1, ...]
    all_points = np.empty((len(positions) * 2, 3))
    all_points[0::2] = starts
    all_points[1::2] = ends

    n_points = 400  # 你有400个传感器点
    # 创建400条线段的索引
    # 数组是 [start0, end0, start1, end1...] 交错排列
    lines = np.array([[i*2, i*2+1] for i in range(n_points)])

    # 直接修改现有对象的属性
    line_set_obj.points = o3d.utility.Vector3dVector(all_points)
    line_set_obj.lines = o3d.utility.Vector2iVector(lines)
    line_set_obj.paint_uniform_color(color)
    
    return line_set_obj

if __name__ == "__main__":
    import keyboard

    current_file = Path(__file__).resolve()
    data_dir = current_file.parent / "data"
    data_dir.mkdir(exist_ok=True)
    task_name = current_file.stem.replace("demo_", "")
    task_dir = data_dir / task_name
    task_dir.mkdir(exist_ok=True)
    timestamp = "20260303-134814"  
    run_dir = task_dir / timestamp
    run_dir.mkdir(exist_ok=True)


    dataloader1 = PyTac3D.DataLoader(SN=TAC3D_NAME1, path=str(run_dir), skip=0)
    # dataloader2 = PyTac3D.DataLoader(SN=TAC3D_NAME2, path=str(run_dir), skip=0)

    app = Tac3D_GUI()
    
    pcd = o3d.geometry.PointCloud()
    lines_D_contact = o3d.geometry.LineSet()
    lines_D_constraint = o3d.geometry.LineSet()

    # 将初始几何体添加到场景
    app.add_pcd_geometry("pcd", pcd)
    app.add_line_geometry("lines_D_contact", lines_D_contact)
    app.add_line_geometry("lines_D_constraint", lines_D_constraint)
    
    cnt = 0
    st_time = time.time()
    sn1_D_init = np.zeros((400, 3))  # 用于记录初始位移，默认为全零
    sn1_endflag = False
    sn2_endflag = False
    try:
        while True:
            cnt += 1
            now_time = time.time() - st_time
            sn1_frame, sn1_frame_time, sn1_endflag = dataloader1.get()
            # sn2_frame, sn2_frame_time, sn2_endflag = dataloader2.get()
            sn1_P = np.array(sn1_frame["3D_Positions"])
            sn1_D = np.array(sn1_frame["3D_Displacements"])
            sn1_F = np.array(sn1_frame["3D_Forces"])
            if sn1_endflag or sn2_endflag:
                dataloader1.reset()
                # dataloader2.reset()
                print(f"Data reset. Restarting visualization loop.")

            if keyboard.is_pressed('i'):
                sn1_D_init = sn1_D.copy()  # 记录初始位移，用于后续计算相对位移
                
                print("Initial displacements recorded.")
                time.sleep(0.5)  # 防止按键抖动导致多次触发
                continue

            sn1_D_constraint = sn1_D - sn1_D_init

            # 关键：更新现有对象的数据，不创建新对象！
            update_point_cloud(pcd, sn1_P)
            update_lines(lines_D_contact, sn1_P, sn1_D_init, [1, 0, 0], scale=5.0)
            update_lines(lines_D_constraint, sn1_P, sn1_D_constraint, [0, 1, 0], scale=5.0)

            # 5. 更新平面文字内容
            
            text = app.i_text + app.esc_text + f"Task: {task_name}\n" + f"Frames: {cnt}"
            app.update_info_label(text)

            # 通知 Open3D 数据已更新
            app.update_pcd_geometry("pcd", pcd)
            app.update_line_geometry("lines_D_contact", lines_D_contact)
            app.update_line_geometry("lines_D_constraint", lines_D_constraint)
            # 刷新渲染
            app.refresh()
            time.sleep(0.001)

    except Exception as e:
        print(e)

    app.quit()

    print(f"Total frames visualized: {cnt}")
    print(f"Total time: {time.time() - st_time:.2f} seconds")
    print(f"Average time per frame: {(time.time() - st_time) / cnt:.4f} seconds")
