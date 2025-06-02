#!/usr/bin/env python
"""
Comprehensive PPO Training and Evaluation Script for All MIAC Environment-Level Combinations

This script trains all available environment-level combinations for 100,000 timesteps each,
then evaluates each model 10 times to calculate averaged rewards and records videos of 
the best performances.

Features:
- Trains all env-level combinations (20 total: 5 environments × 4 levels each)
- 100,000 timesteps per training run
- 10-episode evaluation per trained model
- Video recording of best performance using single environments (no moviepy dependency)
- Comprehensive metadata storage and performance tracking
- Detailed progress reporting and error handling
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers.enhanced_trainer_testers import EnhancedPPOTrainerTester

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
    config_file = PROJECT_ROOT / f'simulator/worlds/miac/{env_name}/{env_name}_level_{level}.yaml'
    
    if not config_file.exists():
        raise ValueError(f"Configuration file not found: {config_file}")
    
    return str(config_file)

def discover_valid_combinations() -> List[Tuple[str, int]]:
    """Discover all valid environment-level combinations based on available config files."""
    combinations = []
    environments = ['incoming', 'circle', 'perp1', 'perp2', 'two_paths']
    levels = [0, 1, 2, 3]
    
    for env_name in environments:
        for level in levels:
            try:
                config_file = get_config_file(env_name, level)
                combinations.append((env_name, level))
                print(f"✓ Found valid combination: {env_name} level {level}")
            except ValueError:
                print(f"✗ Skipping {env_name} level {level} - config file not found")
    
    return combinations

def setup_directories(base_dir: str, batch_timestamp: str) -> Dict[str, str]:
    """Set up directory structure for comprehensive training."""
    directories = {
        'base': base_dir,
        'logs': os.path.join(base_dir, 'logs', batch_timestamp),
        'models': os.path.join(base_dir, 'models', batch_timestamp),
        'videos': os.path.join(base_dir, 'videos', batch_timestamp),
        'reports': os.path.join(base_dir, 'reports', batch_timestamp)
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def train_and_evaluate_model(env_name: str, level: int, directories: Dict[str, str], 
                           timesteps: int = 100000, eval_episodes: int = 10) -> Dict:
    """Train and evaluate a single model."""
    print(f"\n{'='*80}")
    print(f"Training and Evaluating: {env_name} level {level}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Get environment class and config
        env_class = import_env_class(env_name)
        config_file = get_config_file(env_name, level)
        
        # Set up model paths
        model_name = f"ppo_{env_name}_level_{level}"
        model_path = os.path.join(directories['models'], model_name)
        log_dir = os.path.join(directories['logs'], f"{env_name}_level_{level}")
        video_dir = os.path.join(directories['videos'], f"{env_name}_level_{level}")
        
        # Create unique ID for this training run
        unique_id = f"{env_name}_level_{level}_{int(time.time())}"
        
        # Initialize enhanced trainer
        trainer = EnhancedPPOTrainerTester(
            env_class=env_class,
            config_file=config_file,
            log_dir=log_dir,
            save_path=model_path,
            render_mode="rgb_array",
            seed=42,
            unique_id=unique_id,
            env_name=env_name,
            level=level
        )
        
        print(f"Starting training for {timesteps:,} timesteps...")
        training_start = time.time()
        
        # Train the model with optimized parameters for 100k timesteps
        trainer.train(
            n_envs=4,                    # Reduced for stability
            total_timesteps=timesteps,
            n_steps=32,                  # Smaller steps for faster updates
            device='cpu',
            progress_bar=True
        )
        
        training_time = time.time() - training_start
        print(f"Training completed in {training_time:.1f} seconds")
        
        # Evaluate the model
        print(f"Starting evaluation with {eval_episodes} episodes...")
        evaluation_start = time.time()
        
        evaluation_results = trainer.evaluate_model(
            model_path=model_path,
            n_eval_episodes=eval_episodes,
            record_best_video=True,
            video_folder=video_dir
        )
        
        evaluation_time = time.time() - evaluation_start
        print(f"Evaluation completed in {evaluation_time:.1f} seconds")
        
        # Compile results
        total_time = time.time() - start_time
        results = {
            'env_name': env_name,
            'level': level,
            'model_path': model_path,
            'training_timesteps': timesteps,
            'training_time': training_time,
            'evaluation_time': evaluation_time,
            'total_time': total_time,
            'success': True,
            'evaluation_results': evaluation_results,
            'mean_reward': evaluation_results['statistics']['mean_reward'],
            'std_reward': evaluation_results['statistics']['std_reward'],
            'success_rate': evaluation_results['statistics']['success_rate'],
            'video_recorded': evaluation_results.get('video_recorded', False),
            'video_folder': evaluation_results.get('video_folder', None)
        }
        
        print(f"✓ SUCCESS: {env_name} level {level}")
        print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Success rate: {results['success_rate']:.2%}")
        print(f"  Total time: {total_time:.1f}s")
        
        return results
        
    except Exception as e:
        error_time = time.time() - start_time
        print(f"✗ FAILED: {env_name} level {level}")
        print(f"  Error: {str(e)}")
        print(f"  Time: {error_time:.1f}s")
        print(f"  Traceback: {traceback.format_exc()}")
        
        return {
            'env_name': env_name,
            'level': level,
            'success': False,
            'error': str(e),
            'error_time': error_time,
            'traceback': traceback.format_exc()
        }

def generate_comprehensive_report(all_results: List[Dict], directories: Dict[str, str]):
    """Generate comprehensive performance report across all environments."""
    timestamp = datetime.now().isoformat()
    
    # Separate successful and failed results
    successful_results = [r for r in all_results if r.get('success', False)]
    failed_results = [r for r in all_results if not r.get('success', False)]
    
    # Calculate summary statistics
    if successful_results:
        mean_rewards = [r['mean_reward'] for r in successful_results]
        success_rates = [r['success_rate'] for r in successful_results]
        training_times = [r['training_time'] for r in successful_results]
        
        summary_stats = {
            'total_successful': len(successful_results),
            'total_failed': len(failed_results),
            'overall_mean_reward': float(np.mean(mean_rewards)),
            'overall_std_reward': float(np.std(mean_rewards)),
            'best_performance': {
                'env_name': max(successful_results, key=lambda x: x['mean_reward'])['env_name'],
                'level': max(successful_results, key=lambda x: x['mean_reward'])['level'],
                'mean_reward': max(mean_rewards),
            },
            'worst_performance': {
                'env_name': min(successful_results, key=lambda x: x['mean_reward'])['env_name'],
                'level': min(successful_results, key=lambda x: x['mean_reward'])['level'],
                'mean_reward': min(mean_rewards),
            },
            'average_success_rate': float(np.mean(success_rates)),
            'average_training_time': float(np.mean(training_times)),
            'total_training_time': float(np.sum(training_times))
        }
    else:
        summary_stats = {
            'total_successful': 0,
            'total_failed': len(failed_results),
            'overall_mean_reward': None,
            'overall_std_reward': None,
            'best_performance': None,
            'worst_performance': None,
            'average_success_rate': None,
            'average_training_time': None,
            'total_training_time': None
        }
    
    # Create comprehensive report
    report = {
        'timestamp': timestamp,
        'summary': summary_stats,
        'successful_runs': successful_results,
        'failed_runs': failed_results,
        'environment_performance': {}
    }
    
    # Performance by environment
    for env_name in ENV_CLASSES.keys():
        env_results = [r for r in successful_results if r['env_name'] == env_name]
        if env_results:
            env_rewards = [r['mean_reward'] for r in env_results]
            report['environment_performance'][env_name] = {
                'successful_levels': len(env_results),
                'mean_reward': float(np.mean(env_rewards)),
                'std_reward': float(np.std(env_rewards)),
                'best_level': max(env_results, key=lambda x: x['mean_reward'])['level'],
                'worst_level': min(env_results, key=lambda x: x['mean_reward'])['level'],
                'results_by_level': {str(r['level']): r['mean_reward'] for r in env_results}
            }
    
    # Save report as JSON
    report_path = os.path.join(directories['reports'], 'comprehensive_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate human-readable summary
    summary_path = os.path.join(directories['reports'], 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("MIAC PPO Comprehensive Training and Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Total combinations attempted: {len(all_results)}\n")
        f.write(f"Successful runs: {summary_stats['total_successful']}\n")
        f.write(f"Failed runs: {summary_stats['total_failed']}\n\n")
        
        if successful_results:
            f.write(f"Overall Performance:\n")
            f.write(f"  Mean reward across all environments: {summary_stats['overall_mean_reward']:.2f} ± {summary_stats['overall_std_reward']:.2f}\n")
            f.write(f"  Average success rate: {summary_stats['average_success_rate']:.2%}\n")
            f.write(f"  Total training time: {summary_stats['total_training_time']/3600:.1f} hours\n\n")
            
            f.write(f"Best Performance: {summary_stats['best_performance']['env_name']} level {summary_stats['best_performance']['level']} (reward: {summary_stats['best_performance']['mean_reward']:.2f})\n")
            f.write(f"Worst Performance: {summary_stats['worst_performance']['env_name']} level {summary_stats['worst_performance']['level']} (reward: {summary_stats['worst_performance']['mean_reward']:.2f})\n\n")
            
            f.write("Performance by Environment:\n")
            for env_name, perf in report['environment_performance'].items():
                f.write(f"  {env_name}: {perf['mean_reward']:.2f} ± {perf['std_reward']:.2f} (best level: {perf['best_level']})\n")
        
        if failed_results:
            f.write(f"\nFailed Runs:\n")
            for result in failed_results:
                f.write(f"  {result['env_name']} level {result['level']}: {result['error']}\n")
    
    print(f"\nComprehensive report saved to: {report_path}")
    print(f"Summary saved to: {summary_path}")
    
    return report

def main():
    """Main function to run comprehensive training and evaluation."""
    print("MIAC PPO Comprehensive Training and Evaluation")
    print("=" * 60)
    print("Training all environment-level combinations for 100,000 timesteps each")
    print("Then evaluating each model with 10 episodes and video recording")
    print("=" * 60)
    
    # Configuration
    TIMESTEPS = 100000  # 100k timesteps per model
    EVAL_EPISODES = 10  # 10 episodes per evaluation
    
    # Set up directories
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(str(PROJECT_ROOT), 'tests', 'miac', 'comprehensive_results')
    directories = setup_directories(base_dir, batch_timestamp)
    
    print(f"Batch timestamp: {batch_timestamp}")
    print(f"Results will be saved to: {directories['base']}")
    
    # Discover valid combinations
    print("\nDiscovering valid environment-level combinations...")
    valid_combinations = discover_valid_combinations()
    
    if not valid_combinations:
        print("No valid combinations found. Exiting.")
        return
    
    total_combinations = len(valid_combinations)
    print(f"\nFound {total_combinations} valid combinations to process")
    print(f"Estimated total training time: {(total_combinations * TIMESTEPS * 4) / 1000000:.1f} million steps")
    
    # Process all combinations
    print(f"\nStarting comprehensive training and evaluation...")
    print(f"Training parameters: {TIMESTEPS:,} timesteps, {EVAL_EPISODES} evaluation episodes per model")
    
    all_results = []
    overall_start_time = time.time()
    
    for i, (env_name, level) in enumerate(valid_combinations, 1):
        print(f"\nProgress: {i}/{total_combinations} - Processing {env_name} level {level}")
        
        result = train_and_evaluate_model(
            env_name=env_name,
            level=level,
            directories=directories,
            timesteps=TIMESTEPS,
            eval_episodes=EVAL_EPISODES
        )
        
        all_results.append(result)
        
        # Save intermediate results
        intermediate_path = os.path.join(directories['reports'], f'intermediate_results_{i:02d}.json')
        with open(intermediate_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Generate final comprehensive report
    total_time = time.time() - overall_start_time
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Processed: {len(all_results)} combinations")
    
    successful_runs = sum(1 for r in all_results if r.get('success', False))
    print(f"Successful: {successful_runs}/{len(all_results)}")
    
    print("\nGenerating comprehensive report...")
    report = generate_comprehensive_report(all_results, directories)
    
    print(f"\nAll results saved to: {directories['base']}")
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
