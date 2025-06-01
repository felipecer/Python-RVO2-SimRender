#!/usr/bin/env python
"""
Enhanced PPO Trainer/Tester with metadata storage, evaluation, and video recording capabilities.
"""

import argparse
import csv
import json
import os
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import RecordVideo

class EnhancedPPOTrainerTester:
    """Enhanced PPO trainer/tester with comprehensive metadata storage and evaluation."""
    
    def __init__(self, env_class, config_file, log_dir, save_path, render_mode=None, 
                 seed=13, unique_id=None, tag='', hyperparams=None, env_name=None, 
                 level=None):
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
        
        # Enhanced paths for metadata and evaluation
        self.metadata_path = f"{self.save_path}_metadata.json"
        self.evaluation_path = f"{self.save_path}_evaluation.json"
        self.architecture_path = f"{self.save_path}_architecture.pkl"
        
        # Create directories if they don't exist
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        if self.save_path:
            save_dir = os.path.dirname(self.save_path)
            os.makedirs(save_dir, exist_ok=True)
    
    def create_env(self, n_envs, record_video=False, video_folder=None, 
                   video_name_prefix=None):
        """Create vectorized environment with optional video recording."""
        def _init_env():
            env = self.env_class(
                config_file=self.config_file,
                render_mode="rgb_array" if record_video else self.render_mode,
                seed=self.seed
            )
            
            if record_video and video_folder and n_envs == 1:
                # Only record video for single environment evaluation
                env = RecordVideo(
                    env,
                    video_folder=video_folder,
                    name_prefix=video_name_prefix or "evaluation",
                    episode_trigger=lambda x: True  # Record all episodes
                )
            
            return env
        
        if n_envs == 1:
            return DummyVecEnv([_init_env])
        else:
            return make_vec_env(
                lambda: self.env_class(
                    config_file=self.config_file,
                    render_mode=self.render_mode,
                    seed=self.seed
                ),
                n_envs=n_envs
            )
    
    def save_enhanced_metadata(self, model, training_params: Dict[str, Any], 
                               training_time: float, mean_reward: float = None):
        """Save comprehensive model metadata including architecture and training details."""
        
        # Extract model architecture information
        policy = model.policy
        architecture_info = {
            'network_architecture': {
                'policy_net': str(policy.mlp_extractor.policy_net),
                'value_net': str(policy.mlp_extractor.value_net),
                'features_extractor': str(policy.features_extractor),
            },
            'policy_kwargs': {
                'net_arch': policy.net_arch if hasattr(policy, 'net_arch') else None,
                'activation_fn': str(policy.activation_fn) if hasattr(policy, 'activation_fn') else None,
                'ortho_init': getattr(policy, 'ortho_init', None)
            },
            'observation_space': {
                'shape': model.observation_space.shape,
                'dtype': str(model.observation_space.dtype),
                'low': model.observation_space.low.tolist() if hasattr(model.observation_space, 'low') else None,
                'high': model.observation_space.high.tolist() if hasattr(model.observation_space, 'high') else None,
            },
            'action_space': {
                'shape': model.action_space.shape if hasattr(model.action_space, 'shape') else None,
                'dtype': str(model.action_space.dtype) if hasattr(model.action_space, 'dtype') else None,
                'low': model.action_space.low.tolist() if hasattr(model.action_space, 'low') else None,
                'high': model.action_space.high.tolist() if hasattr(model.action_space, 'high') else None,
            }
        }
        
        # Compile comprehensive metadata
        metadata = {
            'model_info': {
                'algorithm': 'PPO',
                'sb3_version': '2.0.0',  # Could get this dynamically
                'pytorch_version': torch.__version__,
                'model_class': str(type(model).__name__),
                'policy_class': str(type(policy).__name__),
            },
            'environment_info': {
                'env_name': self.env_name,
                'level': self.level,
                'config_file': self.config_file,
                'env_class': str(self.env_class),
            },
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'unique_id': self.unique_id,
                'tag': self.tag,
                'seed': self.seed,
                'training_time_seconds': training_time,
                'training_params': training_params,
                'hyperparameters': self.hyperparams,
                'mean_reward': mean_reward,
            },
            'architecture': architecture_info,
            'file_paths': {
                'model_path': self.save_path,
                'metadata_path': self.metadata_path,
                'evaluation_path': self.evaluation_path,
                'architecture_path': self.architecture_path,
                'log_dir': self.log_dir,
            }
        }
        
        # Save metadata as JSON
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save architecture details as pickle for exact reconstruction
        # Extract learning rate value (handle both float and schedule functions)
        lr_value = model.learning_rate
        if callable(lr_value):
            # If it's a schedule function, get the current value
            lr_value = float(lr_value(1.0))  # Get value at progress 1.0
        
        architecture_data = {
            'policy_state_dict': policy.state_dict(),
            'model_parameters': {
                'n_steps': model.n_steps,
                'batch_size': model.batch_size,
                'n_epochs': model.n_epochs,
                'gamma': model.gamma,
                'gae_lambda': model.gae_lambda,
                'clip_range': float(model.clip_range(1.0)) if callable(model.clip_range) else model.clip_range,
                'learning_rate': lr_value,
                'ent_coef': model.ent_coef,
                'vf_coef': model.vf_coef,
            }
        }
        
        with open(self.architecture_path, 'wb') as f:
            pickle.dump(architecture_data, f)
        
        print(f"Enhanced metadata saved to: {self.metadata_path}")
        print(f"Architecture data saved to: {self.architecture_path}")
    
    def load_model_with_metadata(self, model_path: str = None) -> Tuple[PPO, Dict]:
        """Load model along with its metadata."""
        if model_path is None:
            model_path = self.save_path
        
        # Load the model
        model = PPO.load(model_path)
        
        # Load metadata
        metadata_file = f"{model_path}_metadata.json"
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def evaluate_model(self, model_path: str = None, n_eval_episodes: int = 10, 
                       record_best_video: bool = True, video_folder: str = None) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple episodes and optional video recording.
        """
        if model_path is None:
            model_path = self.save_path
        
        model, metadata = self.load_model_with_metadata(model_path)
        
        # Setup video recording folder
        if record_best_video and video_folder is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_folder = os.path.join(
                os.path.dirname(model_path), 
                'evaluation_videos', 
                f"{self.env_name}_level_{self.level}_{timestamp}"
            )
            os.makedirs(video_folder, exist_ok=True)
        
        # Evaluate without video recording first to get performance metrics
        eval_env = self.create_env(n_envs=1, record_video=False)
        
        print(f"Evaluating model over {n_eval_episodes} episodes...")
        episode_rewards = []
        episode_lengths = []
        episode_success_rates = []
        
        try:
            for episode in range(n_eval_episodes):
                obs = eval_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward[0]
                    episode_length += 1
                    
                    if done[0]:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Extract success information if available
                success = info[0].get('success', 0) if info and len(info) > 0 else 0
                episode_success_rates.append(success)
                
                print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        finally:
            # Always close the evaluation environment
            eval_env.close()
            del eval_env
        
        # Calculate statistics
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'n_eval_episodes': n_eval_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_success_rates': episode_success_rates,
            'statistics': {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'min_reward': float(np.min(episode_rewards)),
                'max_reward': float(np.max(episode_rewards)),
                'mean_length': float(np.mean(episode_lengths)),
                'std_length': float(np.std(episode_lengths)),
                'success_rate': float(np.mean(episode_success_rates)),
                'best_episode_idx': int(np.argmax(episode_rewards)),
                'worst_episode_idx': int(np.argmin(episode_rewards)),
            }
        }
        
        # Record video of the best performing run
        if record_best_video and video_folder:
            print(f"Recording video of best performing run (Episode {evaluation_results['statistics']['best_episode_idx'] + 1})...")
            
            video_name_prefix = f"best_run_{self.env_name}_level_{self.level}"
            video_env = self.create_env(
                n_envs=1, 
                record_video=True, 
                video_folder=video_folder,
                video_name_prefix=video_name_prefix
            )
            
            try:
                # Run one episode for video recording
                obs = video_env.reset()
                done = False
                total_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = video_env.step(action)
                    total_reward += reward[0]
                    
                    if done[0]:
                        break
                
                evaluation_results['video_folder'] = video_folder
                evaluation_results['video_recorded'] = True
                print(f"Video saved to: {video_folder}")
                
            finally:
                # Always close the video environment
                video_env.close()
                del video_env
        
        # Save evaluation results
        with open(self.evaluation_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved to: {self.evaluation_path}")
        print(f"Mean reward: {evaluation_results['statistics']['mean_reward']:.2f} ± {evaluation_results['statistics']['std_reward']:.2f}")
        print(f"Success rate: {evaluation_results['statistics']['success_rate']:.2%}")
        
        return evaluation_results
    
    def train(self, n_envs=16, total_timesteps=1000000, n_steps=256, device='cpu', 
              progress_bar=True):
        """Enhanced training with metadata storage."""
        start_time = time.time()
        
        vec_env = self.create_env(n_envs=n_envs)
        
        # Create a run name that includes environment and level information
        run_name = f"PPO"
        if self.env_name and self.level is not None:
            run_name = f"PPO_{self.env_name}_level_{self.level}"
        elif self.env_name:
            run_name = f"PPO_{self.env_name}"
        
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
            ortho_init=True,
        )
        
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            n_steps=n_steps, 
            verbose=1, 
            policy_kwargs=policy_kwargs, 
            device=device,
            tensorboard_log=self.log_dir, 
            **self.hyperparams
        )
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar, 
            tb_log_name=run_name, 
            log_interval=10
        )
        
        training_time = time.time() - start_time
        
        # Save model
        model.save(self.save_path)
        print("Training completed")
        
        # Quick evaluation to get mean reward
        mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=5)
        
        # Save enhanced metadata
        training_params = {
            'total_timesteps': total_timesteps,
            'n_steps': n_steps,
            'n_envs': n_envs,
            'device': device,
            'run_name': run_name,
        }
        
        self.save_enhanced_metadata(model, training_params, training_time, mean_reward)
        
        # Log basic parameters for compatibility
        self.log_parameters({
            'config_file': self.config_file,
            'total_timesteps': total_timesteps,
            'n_steps': n_steps,
            'n_envs': n_envs,
            'seed': self.seed,
            'log_dir': self.log_dir,
            'save_path': self.save_path,
            'hyperparameters': self.hyperparams,
            'mean_reward': mean_reward
        })
        
        # Close environment and clean up
        vec_env.close()
        del model
        del vec_env
    
    def test(self, n_eval_episodes: int = 10, record_video: bool = True):
        """Enhanced testing with evaluation metrics and video recording."""
        print("Starting enhanced evaluation...")
        evaluation_results = self.evaluate_model(
            n_eval_episodes=n_eval_episodes,
            record_best_video=record_video
        )
        
        print("Enhanced evaluation completed!")
        return evaluation_results
    
    def log_parameters(self, params):
        """Legacy parameter logging for compatibility."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, 'run_log.csv')
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'tag', 'unique_id', 'env_name', 'level', 
                    'config_file', 'total_timesteps', 'n_steps', 'n_envs', 
                    'seed', 'log_dir', 'save_path', 'hyperparameters', 'mean_reward'
                ])
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


def parse_cli_args(script_dir):
    """Parse command-line arguments with enhanced options."""
    parser = argparse.ArgumentParser(
        description='Enhanced PPO training/testing with metadata and video recording.')
    
    # Basic arguments (compatible with existing scripts)
    parser.add_argument('--mode', choices=['train', 'test', 'evaluate'],
                        required=True, help='Operation mode', default='train')
    parser.add_argument('--config_file', default='',
                        help='Environment configuration file')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help='Total number of timesteps for training')
    parser.add_argument('--n_steps', type=int, default=1024,
                        help='Number of steps per environment per update')
    parser.add_argument('--n_envs', type=int, default=64,
                        help='Number of parallel environments')
    parser.add_argument('--render_mode', choices=['rgb', 'ansi', None],
                        default=None, help='Render mode')
    parser.add_argument('--seed', type=int, default=13,
                        help='Seed for the simulation')
    parser.add_argument('--device', default='cpu',
                        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--progress_bar', type=bool, default=True,
                        help='Show progress bar during training')
    
    # Enhanced arguments
    parser.add_argument('--env_name', type=str,
                        help='Environment name')
    parser.add_argument('--level', type=int,
                        help='Environment difficulty level')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--record_video', type=bool, default=True,
                        help='Record video during evaluation')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model for evaluation')
    
    # Directory arguments
    parser.add_argument('--log_dir', default=os.path.join(script_dir, 'logs'),
                        help='Directory to save logs')
    parser.add_argument('--save_path', default=os.path.join(script_dir, 'saves', 'ppo_model'),
                        help='Path to save/load the trained model')
    parser.add_argument('--unique_id', default=None,
                        help='Unique identifier for this run')
    
    return parser.parse_args()
