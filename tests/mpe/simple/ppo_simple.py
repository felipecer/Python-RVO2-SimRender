#!/usr/bin/env python
import os
from rl_environments.single_agent.mpe.simple import RVOSimulationEnv
from tests.helpers.trainer_testers import parse_cli_args, PPOTrainerTester

def main(env_class, args):
    config_file = args.config_file if args.config_file != '' else './simulator/worlds/mpe/simple.yaml'
    trainer_tester = PPOTrainerTester(
        env_class=env_class,
        config_file=config_file,
        log_dir=args.log_dir,
        save_path=args.save_path,
        render_mode=args.render_mode,
        seed=args.seed,
        unique_id=args.unique_id
    )

    if args.mode == 'train':
        trainer_tester.train(n_envs=args.n_envs, total_timesteps=args.total_timesteps, n_steps=args.n_steps)
    elif args.mode == 'test':
        trainer_tester.test()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_cli_args(script_dir)
    main(RVOSimulationEnv, args)