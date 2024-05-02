#!/usr/bin/env python
from stable_baselines3.common.env_checker import check_env
from multi_agent_environment.simple import RVOSimulationEnv
env = RVOSimulationEnv('./simulator/worlds/base_scenario.yaml')
check_env(env)
