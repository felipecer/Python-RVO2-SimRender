#!/usr/bin/env python
import gymnasium as gym
from rl_environments.single_agent.simple_v5 import RVOSimulationEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

if __name__ == "__main__":
    vec_env = make_vec_env(RVOSimulationEnv, n_envs=32,  env_kwargs={
                           "config_file": './simulator/worlds/simple.yaml', "render_mode": None})
    model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')
    # model = PPO.load("ppo_rvo", vec_env, verbose=1, device='cpu')
    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("ppo_rvo")
    print("Entrenamiento terminado")
    del model
    vec_env = make_vec_env(RVOSimulationEnv, n_envs=4,  env_kwargs={
                           "config_file": './simulator/worlds/simple.yaml', "render_mode": 'rgb'})
    model = PPO.load("ppo_rvo")
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("rgb")
