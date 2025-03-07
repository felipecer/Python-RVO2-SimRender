import importlib
from tests.miac.ppo_hyperparameter_optimization import parse_args, objective
import optuna
import os

if __name__ == "__main__":
    defaults = {
        'env_class': 'rl_environments.single_agent.miac.perp2.RVOMiacPerp2',
        'config_file': './simulator/worlds/miac/perp2/perp2_level_0.yaml',
        'total_timesteps': 20000,
        'n_steps': 1024,
        'n_envs': 4,
        'seed': 13,
        'n_trials': 10
    }
    args = parse_args(defaults=defaults)

    # Dynamically import the environment class
    module_name, class_name = args.env_class.rsplit('.', 1)
    env_module = importlib.import_module(module_name)
    env_class = getattr(env_module, class_name)

    # Define the base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, env_class, args.config_file, args.total_timesteps, args.n_steps, args.n_envs, args.seed, base_path), n_trials=args.n_trials)

    print("Best hyperparameters: ", study.best_params)