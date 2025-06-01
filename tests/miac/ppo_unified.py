#!/usr/bin/env python
"""
Unified PPO test file that handles all MIAC environments and difficulty levels.
Usage: python ppo_unified.py --env_name <environment> --level <level> [other args]
"""
import os
import sys
from pathlib import Path
from tests.helpers.trainer_testers import parse_cli_args, PPOTrainerTester


# Environment class mappings
ENV_CLASSES = {
    'incoming': 'rl_environments.single_agent.miac.v2.incoming.RVOMiacIncoming2',
    'circle': 'rl_environments.single_agent.miac.v2.circle.RVOMiacCircleV2',
    'perp1': 'rl_environments.single_agent.miac.v2.perp1.RVOMiacPerp1V2',
    'perp2': 'rl_environments.single_agent.miac.v2.perp2.RVOMiacPerp2V2',
    'two_paths': 'rl_environments.single_agent.miac.v2.two_paths.RVOMiacTwoPathsV2'
}

def import_env_class(env_name):
    """Dynamically import the environment class based on environment name."""
    if env_name not in ENV_CLASSES:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_CLASSES.keys())}")
    
    module_path, class_name = ENV_CLASSES[env_name].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def get_config_file(env_name, level):
    """Get the configuration file path for the given environment and level."""
    # Get the project root directory (two levels up from this script)
    project_root = Path(__file__).parent.parent.parent.absolute()
    config_file = project_root / f'simulator/worlds/miac/{env_name}/{env_name}_level_{level}.yaml'
    
    if not config_file.exists():
        raise ValueError(f"Configuration file not found: {config_file}")
    
    return str(config_file)


def main():
    """Main function that handles unified PPO training/testing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_cli_args(script_dir)
    
    # Validate required arguments
    if not args.env_name:
        raise ValueError("--env_name is required. Available: " + ", ".join(ENV_CLASSES.keys()))
    
    if args.level is None:
        args.level = 0
        print(f"No level specified, defaulting to level 0")
    
    # Convert level to int if it's a string
    if isinstance(args.level, str):
        args.level = int(args.level)
    
    print(f"Running PPO for environment: {args.env_name}, level: {args.level}")
    
    # Get environment class and config file
    env_class = import_env_class(args.env_name)
    
    # Use provided config file or determine from env_name and level
    if args.config_file and args.config_file != '':
        config_file = args.config_file
    else:
        config_file = get_config_file(args.env_name, args.level)
    
    print(f"Using configuration file: {config_file}")
    
    # Create trainer/tester instance
    trainer_tester = PPOTrainerTester(
        env_class=env_class,
        config_file=config_file,
        log_dir=args.log_dir,
        save_path=args.save_path,
        render_mode=args.render_mode,
        seed=args.seed,
        unique_id=args.unique_id,
        level=args.level,
        env_name=args.env_name
    )
    
    # Execute based on mode
    device = args.device
    progress_bar = args.progress_bar
    
    if args.mode == 'train':
        print(f"Starting training with {args.total_timesteps} timesteps...")
        trainer_tester.train(
            total_timesteps=args.total_timesteps, 
            device=device,
            progress_bar=progress_bar, 
            n_envs=args.n_envs, 
            n_steps=args.n_steps
        )
        print("Training completed!")
    elif args.mode == 'test':
        print("Starting testing...")
        trainer_tester.test()
        print("Testing completed!")


if __name__ == "__main__":
    main()
