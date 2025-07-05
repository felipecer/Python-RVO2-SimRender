#!/usr/bin/env python
"""
Enhanced unified PPO test file with comprehensive model management.
Features:
- Saves models with detailed metadata (architecture, hyperparams, training info)
- Enables loading and testing of trained models 
- Records videos of best performing evaluation runs
- Flexible evaluation with configurable episode counts
- All data saved in organized tests/miac directory structure

Usage: 
  # Training
  python ppo_unified_enhanced.py --env_name incoming --level 0 --mode train --total_timesteps 1000000
  
  # Evaluation  
  python ppo_unified_enhanced.py --env_name incoming --level 0 --mode evaluate --model_path <path> --n_eval_episodes 10 --record_video True
  
  # Batch evaluation of all models
  python ppo_unified_enhanced.py --mode batch_evaluate --models_base_dir tests/miac --n_eval_episodes 5
"""
import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
from tests.helpers.enhanced_trainer_testers import parse_cli_args, EnhancedPPOTrainerTester


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
    project_root = Path(__file__).parent.parent.parent.absolute()
    config_file = project_root / f'simulator/worlds/miac/{env_name}/{env_name}_level_{level}.yaml'
    
    if not config_file.exists():
        raise ValueError(f"Configuration file not found: {config_file}")
    
    return str(config_file)


def find_trained_models(base_dir: str, env_name: Optional[str] = None, 
                       level: Optional[int] = None) -> List[Dict]:
    """Find all trained models in the base directory with their metadata."""
    models = []
    search_pattern = f"{base_dir}/**/*.zip"
    
    for model_path in glob.glob(search_pattern, recursive=True):
        # Skip files that don't look like models
        if '_metadata.json' in model_path or '_evaluation.json' in model_path:
            continue
            
        metadata_path = model_path.replace('.zip', '_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                model_env = metadata.get('environment_info', {}).get('env_name')
                model_level = metadata.get('environment_info', {}).get('level')
                
                # Filter by environment and level if specified
                if env_name and model_env != env_name:
                    continue
                if level is not None and model_level != level:
                    continue
                
                models.append({
                    'model_path': model_path,
                    'metadata_path': metadata_path,
                    'metadata': metadata,
                    'env_name': model_env,
                    'level': model_level
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse metadata for {model_path}: {e}")
                continue
    
    return models


def batch_evaluate_models(models_base_dir: str, n_eval_episodes: int = 10, 
                         record_videos: bool = True, env_filter: Optional[List[str]] = None,
                         level_filter: Optional[List[int]] = None):
    """Evaluate all trained models in the base directory."""
    print(f"🔍 Searching for trained models in: {models_base_dir}")
    
    # Find all models
    all_models = find_trained_models(models_base_dir)
    
    # Apply filters
    if env_filter:
        all_models = [m for m in all_models if m['env_name'] in env_filter]
    if level_filter:
        all_models = [m for m in all_models if m['level'] in level_filter]
    
    if not all_models:
        print("❌ No trained models found matching the criteria")
        return
    
    print(f"📊 Found {len(all_models)} trained models to evaluate")
    
    # Create evaluation summary
    evaluation_summary = {
        'batch_evaluation_timestamp': datetime.now().isoformat(),
        'total_models': len(all_models),
        'n_eval_episodes': n_eval_episodes,
        'models_evaluated': []
    }
    
    for i, model_info in enumerate(all_models):
        print(f"\n📈 Evaluating model {i+1}/{len(all_models)}")
        print(f"   Environment: {model_info['env_name']} level {model_info['level']}")
        print(f"   Model: {model_info['model_path']}")
        
        try:
            # Get environment class and config
            env_class = import_env_class(model_info['env_name'])
            config_file = get_config_file(model_info['env_name'], model_info['level'])
            
            # Create trainer/tester instance
            trainer_tester = EnhancedPPOTrainerTester(
                env_class=env_class,
                config_file=config_file,
                log_dir=None,  # Not needed for evaluation
                save_path=model_info['model_path'].replace('.zip', ''),
                env_name=model_info['env_name'],
                level=model_info['level']
            )
            
            # Evaluate the model
            results = trainer_tester.evaluate_model(
                model_path=model_info['model_path'],
                n_eval_episodes=n_eval_episodes,
                record_best_video=record_videos
            )
            
            # Add to summary
            model_summary = {
                'model_path': model_info['model_path'],
                'env_name': model_info['env_name'],
                'level': model_info['level'],
                'evaluation_results': results['statistics'],
                'video_recorded': results.get('video_recorded', False),
                'video_folder': results.get('video_folder', None)
            }
            evaluation_summary['models_evaluated'].append(model_summary)
            
            print(f"   ✅ Mean reward: {results['statistics']['mean_reward']:.2f}")
            print(f"   📊 Success rate: {results['statistics']['success_rate']:.2%}")
            if results.get('video_recorded'):
                print(f"   🎥 Video saved to: {results['video_folder']}")
                
        except Exception as e:
            print(f"   ❌ Error evaluating model: {e}")
            model_summary = {
                'model_path': model_info['model_path'],
                'env_name': model_info['env_name'],
                'level': model_info['level'],
                'error': str(e)
            }
            evaluation_summary['models_evaluated'].append(model_summary)
    
    # Save batch evaluation summary
    summary_path = os.path.join(models_base_dir, f"batch_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"\n📋 Batch evaluation summary saved to: {summary_path}")
    
    # Print overall summary
    successful_evals = [m for m in evaluation_summary['models_evaluated'] if 'error' not in m]
    if successful_evals:
        print(f"\n📊 Overall Results ({len(successful_evals)} successful evaluations):")
        avg_reward = sum(m['evaluation_results']['mean_reward'] for m in successful_evals) / len(successful_evals)
        avg_success_rate = sum(m['evaluation_results']['success_rate'] for m in successful_evals) / len(successful_evals)
        print(f"   Average reward across all models: {avg_reward:.2f}")
        print(f"   Average success rate: {avg_success_rate:.2%}")


def main():
    """Main function that handles enhanced unified PPO training/testing/evaluation."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = parse_cli_args(script_dir)
    
    # Handle batch evaluation mode
    if args.mode == 'batch_evaluate':
        if not hasattr(args, 'models_base_dir') or not args.models_base_dir:
            models_base_dir = "tests/miac"  # Default to tests/miac directory
        else:
            models_base_dir = args.models_base_dir
            
        env_filter = getattr(args, 'env_filter', None)
        level_filter = getattr(args, 'level_filter', None)
        
        batch_evaluate_models(
            models_base_dir=models_base_dir,
            n_eval_episodes=getattr(args, 'n_eval_episodes', 10),
            record_videos=getattr(args, 'record_video', True),
            env_filter=env_filter,
            level_filter=level_filter
        )
        return
    
    # Validate required arguments for individual training/evaluation
    if not args.env_name:
        raise ValueError("--env_name is required. Available: " + ", ".join(ENV_CLASSES.keys()))
    
    if args.level is None:
        args.level = 0
        print(f"No level specified, defaulting to level 0")
    
    # Convert level to int if it's a string
    if isinstance(args.level, str):
        args.level = int(args.level)
    
    print(f"🚀 Running PPO for environment: {args.env_name}, level: {args.level}")
    
    # Get environment class and config file
    env_class = import_env_class(args.env_name)
    
    # Use provided config file or determine from env_name and level
    if args.config_file and args.config_file != '':
        config_file = args.config_file
    else:
        config_file = get_config_file(args.env_name, args.level)
    
    print(f"📄 Using configuration file: {config_file}")
    
    # Create trainer/tester instance
    trainer_tester = EnhancedPPOTrainerTester(
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
        print(f"🏋️ Starting training with {args.total_timesteps} timesteps...")
        trainer_tester.train(
            total_timesteps=args.total_timesteps, 
            device=device,
            progress_bar=progress_bar, 
            n_envs=args.n_envs, 
            n_steps=args.n_steps
        )
        print("✅ Training completed!")
        
        # Automatically run evaluation after training
        print("\n🧪 Running post-training evaluation...")
        trainer_tester.test(
            n_eval_episodes=getattr(args, 'n_eval_episodes', 5),
            record_video=getattr(args, 'record_video', True)
        )
        
    elif args.mode == 'test' or args.mode == 'evaluate':
        print("🧪 Starting evaluation...")
        model_path = args.model_path if hasattr(args, 'model_path') and args.model_path else trainer_tester.save_path
        
        if not os.path.exists(model_path):
            # Try with .zip extension
            if not model_path.endswith('.zip'):
                model_path_zip = model_path + '.zip'
                if os.path.exists(model_path_zip):
                    model_path = model_path_zip
                else:
                    raise ValueError(f"Model not found at: {model_path} or {model_path_zip}")
        
        evaluation_results = trainer_tester.evaluate_model(
            model_path=model_path,
            n_eval_episodes=getattr(args, 'n_eval_episodes', 10),
            record_best_video=getattr(args, 'record_video', True)
        )
        
        print("✅ Evaluation completed!")
        print(f"📊 Results: Mean reward = {evaluation_results['statistics']['mean_reward']:.2f}")
        print(f"📈 Success rate = {evaluation_results['statistics']['success_rate']:.2%}")
        
        if evaluation_results.get('video_recorded'):
            print(f"🎥 Video saved to: {evaluation_results['video_folder']}")


if __name__ == "__main__":
    from datetime import datetime
    main()
