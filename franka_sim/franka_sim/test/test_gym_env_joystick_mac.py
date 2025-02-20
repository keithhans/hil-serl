import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs
import gymnasium as gym

# import joystick wrapper
from franka_env.envs.wrappers import JoystickIntervention
from franka_env.spacemouse.spacemouse_expert import ControllerType

from franka_sim.utils.dual_view_env import DualViewEnv


# env = envs.PandaPickCubeGymEnv(render_mode="human", image_obs=True)
env = gym.make("PandaPickCubeVision-v0", render_mode="human", image_obs=True)
env = JoystickIntervention(env)

env.reset()
m = env.unwrapped.model
d = env.unwrapped.data

# Create the dual viewer
dual_viewer = DualViewEnv(env.unwrapped.model, env.unwrapped.data)

def custom_step(model, data):
    # 在这里添加自定义控制逻辑
    # data.ctrl[:] = ...  # 设置控制指令
    env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def custom_render(window_id):
    # 在渲染前执行自定义操作
    #print(f"正在渲染窗口 {window_id}")
    pass


# intervene on position control
with dual_viewer as viewer:
    viewer.run(
        on_step=custom_step,
        on_render=custom_render,
        speed=1.0,  # 1.5倍速运行
        pause=False
    )



