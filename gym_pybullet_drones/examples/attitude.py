"""Script demonstrating the joint use of Attitude input.

The simulation is run by a `AttitudeAviary` environment.

Example
-------
In a terminal, run as:

    $ python Attitude.py

Notes
-----
The drones use interal PID control to track a target Attitude.

"""
import time
import numpy as np
from icecream import install
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from gym_pybullet_drones.envs.AttitudeAviary import AttitudeAviary

install()

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_AGGREGATE = True
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 500
DEFAULT_CONTROL_FREQ_HZ = 500
DEFAULT_DURATION_SEC = 10.0
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def run(
        drone=DEFAULT_DRONE,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VIDEO,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        aggregate=DEFAULT_AGGREGATE,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
):
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([
        [0, 1.0, 1.0],
    ])
    INIT_RPYS = np.array([
        [0.0, 0, 0],
    ])
    AGGR_PHY_STEPS = int(simulation_freq_hz /
                         control_freq_hz) if aggregate else 1
    PHY = Physics.PYB

    #### Create the environment ################################
    env = AttitudeAviary(drone_model=drone,
                         num_drones=1,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=Physics.PYB,
                         neighbourhood_radius=10,
                         freq=simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=gui,
                         record=record_video,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui
                         )

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Run the simulation ####################################
    # generate trajectory
    num_substeps = 50

    t = np.arange(0, duration_sec, 1/500*num_substeps) # 0.1
    t = np.tile(t, (3,1)).transpose()
    A_base = 0.5
    w_base = 2 * np.pi / 5.0
    traj_xyz = np.zeros((len(t), 3))
    traj_xyz[:, 2] += 1.0
    traj_vxyz = np.zeros((len(t), 3))
    for i in range(0,2,1):
        phase = np.random.rand(3) * 2 * np.pi
        scale = np.random.rand(3) * 0.3 + 0.7
        A = A_base * scale * (2.0**(-i))
        w = w_base * (2**i)
        traj_xyz += A * np.cos(t*w+phase)
        traj_vxyz += - w * A * np.sin(t*w+phase)
    # show traj_xyz as curve in bullet 
    for i in range(len(t)-1):
        p.addUserDebugLine(traj_xyz[i], traj_xyz[i+1], lineColorRGB=[1,0,0], lineWidth=3)


    START = time.time()
    for i in range(0, int(duration_sec*env.SIM_FREQ/num_substeps), AGGR_PHY_STEPS):
        # PID controller
        state = env._getDroneStateVector(0)
        total_force_drone = - np.array([0,0,-0.027*9.8]) - (state[:3] - traj_xyz[i]) * 0.2 - (state[10:13] - traj_vxyz[i]) * 0.1
        total_force_drone_projected = (np.linalg.inv(rpy2rotmat(state[7:10])) @ total_force_drone)[2]
        thrust_pid = np.clip(total_force_drone_projected, 0, 1.0)
        ctl_row_pid = np.clip(np.arctan2(-total_force_drone[1], np.sqrt(total_force_drone[0]**2 + total_force_drone[2]**2)), -np.pi/3, np.pi/3)
        ctl_pitch_pid = np.clip(np.arctan2(total_force_drone[0], total_force_drone[2]), -np.pi/3, np.pi/3)
        action = {str(i): np.array([thrust_pid, ctl_row_pid, ctl_pitch_pid, 0]) for i in range(1)}
        for _ in range(num_substeps):
            obs, reward, done, info = env.step(action)
            #### Log the simulation ####################################
            for j in range(1):
                act_log = np.zeros(12)
                act_log[:4] = action[str(j)]
                act_log[4:7] = traj_xyz[i]
                act_log[7:10] = traj_vxyz[i]
                logger.log(drone=j,
                        timestamp=i/env.SIM_FREQ,
                        state=obs[str(j)]["state"],
                        control=act_log,
                        )
        if i % env.SIM_FREQ == 0:
            env.render()
        if gui:
            sync(i, START, env.TIMESTEP)
    env.close()

    logger.save_as_csv("att")  # Optional CSV save
    if plot:
        logger.plot()

def rpy2rotmat(rpy):
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    rotmat = np.zeros([3, 3])
    rotmat[0, 0] = np.cos(yaw) * np.cos(pitch)
    rotmat[0, 1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    rotmat[0, 2] = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    rotmat[1, 0] = np.sin(yaw) * np.cos(pitch)
    rotmat[1, 1] = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    rotmat[1, 2] = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
    rotmat[2, 0] = -np.sin(pitch)
    rotmat[2, 1] = np.cos(pitch) * np.sin(roll)
    rotmat[2, 2] = np.cos(pitch) * np.cos(roll)
    return rotmat


if __name__ == "__main__":
    run()
