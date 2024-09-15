#!/usr/bin/env python
from stable_baselines3.common.env_checker import check_env
from rl_environments.single_agent.simple_v2 import RVOSimulationEnv2
env = RVOSimulationEnv2('./simulator/worlds/simple_v2.yaml')
check_env(env)
