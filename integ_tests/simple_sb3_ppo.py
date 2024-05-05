#!/usr/bin/env python
import gymnasium as gym
from multi_agent_environment.simple import RVOSimulationEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

if __name__ == "__main__":
    vec_env = make_vec_env(RVOSimulationEnv, n_envs=4,  env_kwargs={"config_file":'./simulator/worlds/base_scenario.yaml'})
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo_rvo")

    del model
    model = PPO.load("ppo_rvo")
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")