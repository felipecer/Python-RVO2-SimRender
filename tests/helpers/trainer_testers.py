#!/usr/bin/env python
import argparse
from datetime import datetime
import uuid
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import os
import csv
from torch import nn


def parse_cli_args(script_dir):
    parser = argparse.ArgumentParser(
        description='Train or test the PPO model in a simulation environment.')
    parser.add_argument('--mode', choices=['train', 'test'],
                        required=True, help='Operation mode: train or test', default='train')
    parser.add_argument('--config_file', default='',
                        help='Environment configuration file')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total number of timesteps for training (default: 1000000)')
    parser.add_argument('--n_steps', type=int, default=1024,
                        help='Number of steps per environment per update (default: 1024)')
    parser.add_argument('--n_envs', type=int, default=64,
                        help='Number of parallel environments (default: 64)')
    parser.add_argument('--render_mode', choices=['rgb', 'ansi', None],
                        default=None, help='Render mode')
    parser.add_argument('--seed', type=int, default=13,
                        help='Seed for the simulation')
    parser.add_argument('--tag', type=str, default='',
                        help='Optional tag for the run')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to the saved model for testing')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for tensorboard logs')
    parser.add_argument('--progress_bar', type=bool, default=False,
                        help='Display progress bar during training (default: True)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                        help='Device to use for training (default: cpu)')
    parser.add_argument('--env_name', type=str, default=None,
                        help='Environment name for better logging')
    parser.add_argument('--level', type=str, default=None,
                        help='Level identifier for better logging')
    args = parser.parse_args()

    # Generate a unique ID for the run
    unique_id = str(uuid.uuid4())

    # Set log_dir and save_path relative to the script directory ONLY if not provided
    if args.log_dir is None:
        args.log_dir = os.path.join(script_dir, 'logs', unique_id)
    if args.save_path is None:
        args.save_path = os.path.join(
            script_dir, 'saves', f'ppo_model_{unique_id}')

    args.render_mode = args.render_mode if args.mode == 'test' else None
    args.unique_id = unique_id
    return args


class PPOTrainerTester:
    def __init__(self, env_class, config_file, log_dir, save_path, render_mode=None, seed=13, unique_id=None, tag='', hyperparams=None, env_name=None, level=None):
        self.env_class = env_class
        self.config_file = config_file
        self.log_dir = log_dir
        self.save_path = save_path
        self.render_mode = render_mode
        self.seed = seed
        self.unique_id = unique_id
        self.tag = tag
        self.hyperparams = hyperparams if hyperparams is not None else {}
        self.env_name = env_name
        self.level = level
        # Create directories if they don't exist
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        if self.save_path:
            # Create directory for the save path (excluding the file itself)
            save_dir = os.path.dirname(self.save_path)
            os.makedirs(save_dir, exist_ok=True)

    def create_env(self, n_envs):
        # vec_env_cls=SubprocVecEnv
        return make_vec_env(self.env_class, n_envs=n_envs, vec_env_cls=DummyVecEnv, env_kwargs={
            # return make_vec_env(self.env_class, n_envs=n_envs, env_kwargs={
            "config_file": self.config_file, "render_mode": self.render_mode, "seed": self.seed,
        })

    def log_parameters(self, params):
        # Use the log_dir as is, don't get its parent directory
        os.makedirs(self.log_dir, exist_ok=True)

        # Write the CSV file directly in the log directory
        log_file = os.path.join(self.log_dir, 'run_log.csv')
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['timestamp', 'tag', 'unique_id', 'env_name', 'level', 'config_file', 'total_timesteps',
                                'n_steps', 'n_envs', 'seed', 'log_dir', 'save_path', 'hyperparameters', 'mean_reward'])
            writer.writerow([
                datetime.now().isoformat(),
                self.tag,
                self.unique_id,
                self.env_name,
                self.level,
                params['config_file'],
                params['total_timesteps'],
                params['n_steps'],
                params['n_envs'],
                params['seed'],
                params['log_dir'],
                params['save_path'],
                str(params['hyperparameters']),
                params['mean_reward']
            ])

    def train(self, n_envs=16, total_timesteps=1000000, n_steps=256, device='cpu', progress_bar=True):
        vec_env = self.create_env(n_envs=n_envs)

        # Create a run name that includes environment and level information
        run_name = f"PPO"
        if self.env_name and self.level != None:
            run_name = f"PPO_{self.env_name}_level_{self.level}"
        elif self.env_name:
            run_name = f"PPO_{self.env_name}"
        
        policy_kwargs = dict(
            # asymmetric nets
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
            ortho_init=True,  # or False, depending on stability in your env
        )
        
        model = PPO("MlpPolicy", vec_env, n_steps=n_steps, verbose=1, policy_kwargs=policy_kwargs, device=device,
                    tensorboard_log=self.log_dir, **self.hyperparams)
        
        # Include run_name in the learn method
        model.learn(total_timesteps=total_timesteps,
                    progress_bar=progress_bar, tb_log_name=run_name, log_interval=10)
        
        model.save(self.save_path)
        print("Training completed")
        del model

        # Log parameters
        params = {
            'config_file': self.config_file,
            'total_timesteps': total_timesteps,
            'n_steps': n_steps,
            'n_envs': n_envs,
            'seed': self.seed,
            'log_dir': self.log_dir,
            'save_path': self.save_path,
            'hyperparameters': self.hyperparams,
            'mean_reward': -999999999
        }
        self.log_parameters(params)

    def test(self):
        vec_env = self.create_env(n_envs=1)
        model = PPO.load(self.save_path)
        obs = vec_env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            if dones:
                vec_env.reset()
            vec_env.render("rgb")
