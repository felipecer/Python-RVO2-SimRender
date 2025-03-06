#!/usr/bin/env python
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from rl_environments.single_agent.miac.circle import RVOMiacCircle


def train():
    vec_env = make_vec_env(RVOMiacCircle, n_envs=64, env_kwargs={
                           "config_file": './simulator/worlds/miac/d_lite.yaml', "render_mode": None, "seed": 13})
    model = PPO("MlpPolicy", vec_env,  n_steps=1024, verbose=1, device='cpu',
                tensorboard_log="./tests/logs/ppo_rvo_miac_d_lite_test/")
    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("./tests/logs/saves/ppo_rvo_miac_d_lite_test1")
    print("Entrenamiento terminado")
    del model


def test():
    vec_env = make_vec_env(RVOMiacCircle, n_envs=1, env_kwargs={
                           "config_file": './simulator/worlds/miac/d_lite.yaml', "render_mode": 'rgb', "seed": 13})
    model = PPO.load("./tests/logs/saves/ppo_rvo_miac_d_lite_test1")
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

        if dones:
            vec_env.reset()
        vec_env.render("rgb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrenar o probar el modelo PPO en RVOSimulationEnvMIAC a.')
    parser.add_argument('--mode', choices=['train', 'test'],
                        required=True, help='Modo de operación: train o test')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
