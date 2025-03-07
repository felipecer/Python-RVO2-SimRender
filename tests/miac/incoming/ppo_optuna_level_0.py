from datetime import datetime
import uuid
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from rl_environments.single_agent.miac.incoming import RVOMiacIncoming
from tests.miac.trainer_testers import PPOTrainerTester
import os

def objective(trial):
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'logs', unique_id)
    save_path = os.path.join(script_dir, 'saves', f'ppo_model_{unique_id}')

    # Create the PPOTrainerTester instance
    trainer_tester = PPOTrainerTester(
        env_class=RVOMiacIncoming,
        config_file='./simulator/worlds/miac/incoming/incoming_level_0.yaml',
        log_dir=log_dir,
        save_path=save_path,
        render_mode=None,
        seed=13,
        unique_id=unique_id,
        hyperparams=hyperparams
    )

    # Train the model
    trainer_tester.train(n_envs=4, total_timesteps=20000)

    # Evaluate the model
    vec_env = trainer_tester.create_env(n_envs=1)
    model = PPO.load(save_path)
    mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=10)

    # Log parameters
    params = {
        'timestamp': datetime.now().isoformat(),
        'tag': 'optuna',
        'unique_id': unique_id,
        'config_file': './simulator/worlds/miac/incoming/incoming_level_0.yaml',
        'total_timesteps': 20000,
        'n_steps': 1024,
        'n_envs': 4,
        'seed': 13,
        'log_dir': log_dir,
        'save_path': save_path,
        'hyperparameters': hyperparams,
        'mean_reward': mean_reward
    }
    trainer_tester.log_parameters(params)

    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters: ", study.best_params)