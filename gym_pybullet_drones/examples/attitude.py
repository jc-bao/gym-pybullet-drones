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
import torch

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
DEFAULT_DURATION_SEC = 8.0
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
    # set torch numpy print precision
    torch.set_printoptions(precision=2)
    np.set_printoptions(precision=2)

    #### Initialize the simulation #############################
    INIT_XYZS = np.array([
        [0, 0.0, 1.0],
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

    delta_t = 0.1
    base_w = 2 * np.pi / (40.0 * delta_t)
    t = np.arange(0, int(duration_sec/delta_t*2)) * delta_t
    t = np.tile(t, (3,1)).transpose()
    traj_xyz = np.zeros((len(t), 3))
    traj_xyz[:, 2] += 1.0
    traj_vxyz = np.zeros((len(t), 3))
    for i in range(0,2,1):
        A = 0.5 * (np.random.rand(3) * 0.3 + 0.7) * (2.0**(-i))

        # DEBUG
        # A[:] = 0.5 * (2.0**(-i))

        w = base_w*(2**i)

        phase = np.random.rand(3) * 2 * np.pi
        
        # DEBUG
        # phase[0] = 0.0
        # phase[1] = np.pi/2
        # phase[2] = np.pi/4

        traj_xyz += A * np.cos(t*w+phase)
        traj_vxyz += - w * A * np.sin(t*w+phase)
    # show traj_xyz as curve in bullet 
    traj_xyz_drone = traj_xyz.copy()
    traj_xyz_drone[:, 2] += 0.2
    for i in range(len(t)-1):
        p.addUserDebugLine(traj_xyz_drone[i], traj_xyz_drone[i+1], lineColorRGB=[1,0,0], lineWidth=3)
    # create a sphere to show current target position
    target_sphere = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1,0,0,1], physicsClientId = env.CLIENT)
    # create body from visual shape
    target_sphere_body = p.createMultiBody(baseVisualShapeIndex=target_sphere, physicsClientId = env.CLIENT)

    # load PPO controller
    loaded_agent = torch.load('/Users/reedpan/Desktop/Research/gym-pybullet-drones/gym_pybullet_drones/examples/results/ppo_track_robust_6.pt', map_location='cpu')
    policy = loaded_agent['actor']
    compressor = loaded_agent['compressor']

    START = time.time()
    for i in range(0, int(duration_sec*env.SIM_FREQ/num_substeps), AGGR_PHY_STEPS):
        state = env._getDroneStateVector(0)
        # PPO controller
        # get observation
        xyz_drone = state[:3]
        xyz_drone_normed = (xyz_drone - np.array([0.,0.,1.])) / np.ones(3)
        xyz_obj = state[:3]
        xyz_obj[2] -= 0.2
        xyz_obj_normed = (xyz_obj - np.array([0.,0.,1.])) / np.ones(3)
        xyz_target = traj_xyz[i]
        xyz_target_normed = (xyz_target - np.array([0., 0., 1.0])) / (np.ones(3)*0.7)
        vxyz_drone = state[10:13]
        vxyz_drone_normed = (vxyz_drone - np.zeros(3)) / (np.ones(3) * 2.0)
        vxyz_obj_normed = vxyz_drone_normed
        rpy_drone = state[7:10]
        rpy_drone_normed = (rpy_drone - np.zeros(3)) / np.array([np.pi/3, np.pi/3, 1.0])
        future_traj_x = traj_xyz[i:i+5].copy()
        future_traj_x[:, 2] -= 1.0
        future_traj_x = future_traj_x.flatten()
        future_traj_v = traj_vxyz[i:i+5].flatten()
        # ic(xyz_drone_normed, xyz_obj_normed, xyz_target_normed, vxyz_drone_normed, vxyz_obj_normed, rpy_drone_normed, xyz_obj - xyz_target, vxyz_drone - traj_vxyz[i], future_traj_x, future_traj_v)
        obs = np.concatenate([xyz_drone_normed, xyz_obj_normed, xyz_target_normed, vxyz_drone_normed, vxyz_obj_normed, rpy_drone_normed, xyz_obj - xyz_target, vxyz_drone - traj_vxyz[i], future_traj_x, future_traj_v], axis=0)
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        # get expert info
        mass_obj_normed = (np.array([0.0]) - 0.0) / 1.0
        length_rope_normed = (np.array([0.2]) - 0.2) / 1.0
        mass_drone_normed = (np.array([0.027]) - 0.0275) / 0.0025
        damping_rate_drone_normed = (np.array([0.0]) - 0.025) / 0.025
        damping_rate_obj_normed = (np.array([0.0]) - 0.025) / 0.025
        attitude_pid_p = (np.array([0.075]) - 0.075) / 0.025
        attitude_pid_d = (np.array([0.5]) - 0.5) / 0.1
        e = np.concatenate([mass_obj_normed, length_rope_normed, mass_drone_normed, damping_rate_drone_normed, damping_rate_obj_normed, attitude_pid_p, attitude_pid_d], axis=0)
        e = torch.from_numpy(e).float().unsqueeze(0)
        # get action
        with torch.no_grad():
            action = policy(obs, compressor(e))
        action = action.cpu().numpy().squeeze()
        ctl_thrust = action[0] * 0.3 + 0.3
        ctl_roll = action[1] * np.pi/6
        ctl_pitch = action[2] * np.pi/6
        # PID controller
        # total_force_drone = - np.array([0,0,-0.027*9.8]) - (state[:3] - traj_xyz_drone[i]) * 0.1 - (state[10:13] - traj_vxyz[i]) * 0.2
        # total_force_drone_projected = (np.linalg.inv(rpy2rotmat(state[7:10])) @ total_force_drone)[2]
        # ctl_thrust = np.clip(total_force_drone_projected, 0, 1.0)
        # ctl_roll = np.clip(np.arctan2(-total_force_drone[1], np.sqrt(total_force_drone[0]**2 + total_force_drone[2]**2)), -np.pi/3, np.pi/3)
        # ctl_pitch = np.clip(np.arctan2(total_force_drone[0], total_force_drone[2]), -np.pi/3, np.pi/3)
        '''
        # Debug controller
        roll = state[7]
        pitch = state[8]
        ctl_thrust = 0.027 * 9.8 / np.cos(np.sqrt(roll**2 + pitch**2))
        if i%40 < 10:
            ctl_pitch = np.pi/24
        elif i%40 < 30:
            ctl_pitch = -np.pi/24
        else:
            ctl_pitch = np.pi/24
        ctl_roll = ctl_pitch
        '''
        action = {str(i): np.array([ctl_thrust, ctl_roll, ctl_pitch, 0]) for i in range(1)}
        # set target_sphere to the traj_xyz_drone[i]
        p.resetBasePositionAndOrientation(target_sphere_body, traj_xyz_drone[i], (0,0,0,1), physicsClientId = env.CLIENT)
        for _ in range(num_substeps):
            obs, reward, done, info = env.step(action)
            #### Log the simulation ####################################
            for j in range(1):
                act_log = np.zeros(12)
                act_log[:4] = action[str(j)]
                act_log[4:7] = traj_xyz_drone[i]
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
