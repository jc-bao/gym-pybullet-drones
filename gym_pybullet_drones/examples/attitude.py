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
import argparse
import numpy as np
from icecream import install

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
DEFAULT_DURATION_SEC = 2
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
        [0, 0, .0],
    ])
    INIT_RPYS = np.array([
        [0, 0, 0],
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
    START = time.time()
    for i in range(0, int(duration_sec*env.SIM_FREQ/50), AGGR_PHY_STEPS):
        action = {str(i): np.array([0.027 * 9.8 * 1.0, 0, 0, 0]) for i in range(1)}
        for _ in range(50):
            obs, reward, done, info = env.step(action)

        #### Log the simulation ####################################
        for j in range(1):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state=obs[str(j)]["state"],
                       )
        if i % env.SIM_FREQ == 0:
            env.render()
        if gui:
            sync(i, START, env.TIMESTEP)
    env.close()

    logger.save_as_csv("att")  # Optional CSV save
    if plot:
        logger.plot()


if __name__ == "__main__":
    run()
