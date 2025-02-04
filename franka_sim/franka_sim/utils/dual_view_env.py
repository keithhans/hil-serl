import mujoco
import glfw
import numpy as np
from typing import Callable

class DualViewEnv:
    def __init__(self, model_path: str = "../envs/xmls/arena.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 窗口参数
        self.window_width = 800
        self.window_height = 600
        self._init_glfw()
        
    def _init_glfw(self):
        if not glfw.init():
            raise Exception("GLFW 初始化失败")
            
        # 创建双窗口
        self.window1 = glfw.create_window(self.window_width, self.window_height, "View 1 - 俯视", None, None)
        self.window2 = glfw.create_window(self.window_width, self.window_height, "View 2 - 侧视", None, None)
        glfw.set_window_pos(self.window1, 100, 100)
        glfw.set_window_pos(self.window2, 1000, 100)
        
        # 初始化渲染配置
        self.config1 = self._setup_window(self.window1, self.model, "top")
        self.config2 = self._setup_window(self.window2, self.model, "side")

    # 自定义视角配置函数
    def _setup_custom_view(self, cam, view_type):
        if view_type == "top":
            # 俯视视角
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat = np.array([0, 0, 0])  # 看向原点
            cam.distance = 3.0    # 观察距离
            cam.elevation = -90   # 俯视角度（-90度垂直向下）
            cam.azimuth = 0       # 水平旋转角度
        elif view_type == "side":
            # 侧视视角
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = 0   # 跟踪根物体
            cam.lookat = np.array([0, 0, 1.5])  # 看向腰部高度
            cam.distance = 4.0
            cam.elevation = -20   # 稍微俯视
            cam.azimuth = 90      # 正侧面视角

    # 窗口配置函数
    def _setup_window(self, window, model, view_type):
        glfw.make_context_current(window)
        
        # 创建渲染上下文
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # 初始化场景和相机
        scene = mujoco.MjvScene(model, maxgeom=1000)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        
        # 应用自定义视角
        self._setup_custom_view(cam, view_type)
        
        return {
            "scene": scene,
            "cam": cam,
            "opt": opt,
            "context": context,
            "viewport": mujoco.MjrRect(0, 0, self.window_width, self.window_height)
        }

    def run(self, 
           on_step: Callable[[mujoco.MjModel, mujoco.MjData], None] = None,
           on_render: Callable[[int], None] = None,
           speed: float = 1.0,
           pause: bool = False):
        """主运行循环
        Args:
            on_step: 每次物理步进后调用的函数，接收model和data参数
            on_render: 每次渲染前调用的函数，接收窗口标识（1或2）
            speed: 模拟速度倍数
            pause: 是否暂停模拟
        """
        while not self.should_close:
            if not pause:
                # 执行物理模拟
                mujoco.mj_step(self.model, self.data, nstep=int(speed))
                
                # 用户自定义步进逻辑
                if on_step:
                    on_step(self.model, self.data)

            # 渲染窗口1
            if on_render:
                on_render(1)
            self._render_window(self.window1, self.config1)
            
            # 渲染窗口2
            if on_render:
                on_render(2)
            self._render_window(self.window2, self.config2)
            
            glfw.poll_events()

    def _render_window(self, window, config):
        glfw.make_context_current(window)
        mujoco.mjv_updateScene(self.model, self.data, config["opt"], None, 
                             config["cam"], mujoco.mjtCatBit.mjCAT_ALL.value, config["scene"])
        mujoco.mjr_render(config["viewport"], config["scene"], config["context"])
        glfw.swap_buffers(window)

    @property
    def should_close(self):
        return glfw.window_should_close(self.window1) or glfw.window_should_close(self.window2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        glfw.terminate() 

def custom_step(model, data):
    # 在这里添加自定义控制逻辑
    # data.ctrl[:] = ...  # 设置控制指令
    pass

def custom_render(window_id):
    # 在渲染前执行自定义操作
    print(f"正在渲染窗口 {window_id}")

if __name__ == '__main__':
    with DualViewEnv() as env:
        env.run(
            on_step=custom_step,
            on_render=custom_render,
            speed=1.5,  # 1.5倍速运行
            pause=False
        )