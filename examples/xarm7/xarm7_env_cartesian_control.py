import logging
from dataclasses import dataclass
from time import sleep
from typing import Literal

import mujoco as mj
import tyro

from rcs._core.sim import SimConfig
from rcs.envs.base import (
    ControlMode,
    CoverWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
    SimEnv,
)
from rcs.envs.configs import EmptyWorldXArm7
from rcs.envs.sim import RobotSimWrapper

import rcs
from rcs import sim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
The example shows how to create a xArm7 environment with Cartesian control
and a relative action space. The example works both with a real robot and in
simulation.

To test with a real robot, pass '--platform hardware' and pass the '--ip YOUR_ROBOT_IP',
install the rcs_xarm7 extension (`pip install extensions/rcs_xarm7`)
"""


@dataclass
class Args:
    platform: Literal["simulation", "hardware"] = "simulation"
    """The platform to use for this demo ('simulation' | 'hardware')"""

    ip: str = "192.168.1.245"
    """The IP of the robot hardware platform to connect to"""

    gfx: Literal["vulkan", "opengl", "opengl_software"] = "vulkan"
    """The graphics API to be used by the experimental filament renderer"""


MJ_VERSION = tuple(map(int, mj.__version__.split(".")))
MJ_USE_FILAMENT = True


def main():
    args = tyro.cli(Args)

    if args.platform == "hardware":
        from rcs_xarm7.configs import DefaultXArm7HardwareEnv

        env_creator = DefaultXArm7HardwareEnv()
        env_creator.ip = args.ip
        hw_cfg = env_creator.config()
        hw_cfg.control_mode = ControlMode.CARTESIAN_TQuat
        hw_cfg.max_relative_movement = 0.5
        hw_cfg.relative_to = RelativeTo.LAST_STEP
        env_rel = env_creator.create_env(hw_cfg)
    else:
        scene = EmptyWorldXArm7()
        sim_cfg_data = scene.prefixed_cfg(scene.config())
        xarm7 = scene.lead_robot_name(sim_cfg_data)

        robot_cfg = sim_cfg_data.robot_cfgs[xarm7]
        sim_cfg = SimConfig(
            realtime=False,
            async_control=False,
        )

        mjmodel = scene.create_model(sim_cfg_data)
        simulation = sim.Sim(mjmodel, sim_cfg)

        kinematic_model_path, attachment_site = scene.kinematics_cfg(sim_cfg_data)[xarm7]
        ik = rcs.common.Pin(
            kinematic_model_path,
            attachment_site,
            joint_limits=robot_cfg.joint_limits,
        )

        robot = rcs.sim.SimRobot(simulation, ik, robot_cfg)
        env_rel = SimEnv(simulation)
        env_rel = RobotWrapper(env_rel, robot, ControlMode.CARTESIAN_TQuat)
        env_rel = RobotSimWrapper(env_rel)
        env_rel = RelativeActionSpace(
            env_rel,
            max_mov=0.5,
            relative_to=RelativeTo.LAST_STEP,
        )
        env_rel = CoverWrapper(env_rel)

        # # NOTE(wilbert): the mp gui server stuff seems to not play well with the exp-filament stuff
        # if MJ_VERSION != (3, 16, 1) or not MJ_USE_FILAMENT:
        #     env_rel.get_wrapper_attr("sim").open_gui()
        #     sleep(3)  # wait for gui to open

    obs, info = env_rel.reset()

    if MJ_VERSION == (3, 16, 1) and MJ_USE_FILAMENT:
        from mujoco.experimental.studio import launch_passive, step_control, viewer_app, viewer_protocol

        config = viewer_protocol.ViewerConfig(
            title="RCS MuJoCo viewer",
            width=1024,
            height=768,
            gfx=args.gfx,
        )

        # sim_plugins = [step_control.StepControl()]
        # viewer_plugins = [viewer_app.ViewerApp()]

        plugins = [
            step_control.StepControl(),
            viewer_app.ViewerApp(),
        ]

        launch_passive.run(
            config,
            model=simulation.model,
            data=simulation.data,
            plugins=plugins,
        )

        # # with launch_passive.launch(config, sim_plugins=sim_plugins, viewer_plugins=viewer_plugins) as viewer:
        # with launch_passive.launch(config, plugins=[*sim_plugins, *viewer_plugins]) as viewer:
        #     while viewer.is_running():
        #         act = {"tquat": [0, 0, 0, 0, 0, 0, 1]}
        #         _ = env_rel.step(act)
        #
        #         viewer.sync()
    else:
        for _ in range(100):
            for _ in range(10):
                # move 1cm in x direction (forward) and close gripper
                act = {"tquat": [0.01, 0, 0, 0, 0, 0, 1]}
                obs, reward, terminated, truncated, info = env_rel.step(act)
                sleep(0.6)
            for _ in range(10):
                # move 1cm in negative x direction (backward) and open gripper
                act = {"tquat": [-0.01, 0, 0, 0, 0, 0, 1]}
                obs, reward, terminated, truncated, info = env_rel.step(act)
                sleep(0.6)


if __name__ == "__main__":
    main()
