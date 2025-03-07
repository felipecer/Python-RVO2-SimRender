from datetime import datetime
import uuid
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from tests.miac.trainer_testers import PPOTrainerTester
import os
import argparse
import importlib

def parse_args(defaults):
    parser = argparse.ArgumentParser(description="Optimize PPO hyperparameters using Optuna")
    parser.add_argument('--env_class', type=str, default=defaults['env_class'], help="Environment class to use")
    parser.add_argument('--config_file', type=str, default=defaults['config_file'], help="Path to the environment configuration file")
    parser.add_argument('--total_timesteps', type=int, default=defaults['total_timesteps'], help="Total timesteps for training")
    parser.add_argument('--n_steps', type=int, default=defaults['n_steps'], help="Number of steps per environment per update")
    parser.add_argument('--n_envs', type=int, default=defaults['n_envs'], help="Number of parallel environments")
    parser.add_argument('--seed', type=int, default=defaults['seed'], help="Random seed")
    parser.add_argument('--n_trials', type=int, default=defaults['n_trials'], help="Number of Optuna trials")
    return parser.parse_args()

def objective(trial, env_class, config_file, total_timesteps, n_steps, n_envs, seed, base_path):
    # Define the hyperparameters to optimize
    hyperparams = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'gamma': trial.suggest_uniform('gamma', 0.9, 0.9999),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-2),
    }

    # Generate a unique ID for the trial
    unique_id = str(uuid.uuid4())
    log_dir = os.path.join(base_path, 'logs', unique_id)
    save_path = os.path.join(base_path, 'saves', f'ppo_model_{unique_id}')

    # Create the PPOTrainerTester instance
    trainer_tester = PPOTrainerTester(
        env_class=env_class,
        config_file=config_file,
        log_dir=log_dir,
        save_path=save_path,
        render_mode=None,
        seed=seed,
        unique_id=unique_id,
        hyperparams=hyperparams
    )

    # Train the model
    trainer_tester.train(n_envs=n_envs, total_timesteps=total_timesteps)

    # Evaluate the model
    vec_env = trainer_tester.create_env(n_envs=1)
    model = PPO.load(save_path)
    mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=10)

    # Log parameters
    params = {
        'timestamp': datetime.now().isoformat(),
        'tag': 'optuna',
        'unique_id': unique_id,
        'config_file': config_file,
        'total_timesteps': total_timesteps,
        'n_steps': n_steps,
        'n_envs': n_envs,
        'seed': seed,
        'log_dir': log_dir,
        'save_path': save_path,
        'hyperparameters': str(hyperparams),
        'mean_reward': mean_reward
    }
    trainer_tester.log_parameters(params)

    return mean_reward

if __name__ == "__main__":
    args = parse_args()

    # Dynamically import the environment class
    module_name, class_name = args.env_class.rsplit('.', 1)
    env_module = importlib.import_module(module_name)
    env_class = getattr(env_module, class_name)

    # Define the base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, env_class, args.config_file, args.total_timesteps, args.n_steps, args.n_envs, args.seed, base_path), n_trials=args.n_trials)

    print("Best hyperparameters: ", study.best_params)