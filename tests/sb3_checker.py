#!/usr/bin/env python
from stable_baselines3.common.env_checker import check_env
from rl_environments.single_agent.simple import RVOSimulationEnv
env = RVOSimulationEnv('./simulator/worlds/base_scenario.yaml')
check_env(env)
