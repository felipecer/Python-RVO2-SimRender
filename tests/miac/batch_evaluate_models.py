#!/usr/bin/env python
"""
Enhanced batch evaluation script for all trained MIAC models.
Evaluates models, records videos of best runs, and generates performance reports.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

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
    project_root = Path(__file__).parent.parent.parent.absolute()
    config_file = project_root / f'simulator/worlds/miac/{env_name}/{env_name}_level_{level}.yaml'
    
    if not config_file.exists():
        raise ValueError(f"Configuration file not found: {config_file}")
    
    return str(config_file)

def find_trained_models(base_dir: str) -> List[Tuple[str, str, int]]:
    """
    Find all trained models in the base directory.
    Returns list of (env_name, model_path, level) tuples.
    """
    models = []
    base_path = Path(base_dir)
    
    # Look for model files with timestamp subdirectories
    for env_dir in base_path.iterdir():
        if env_dir.is_dir() and env_dir.name in ENV_CLASSES:
            env_name = env_dir.name
            saves_dir = env_dir / 'saves'
            
            if saves_dir.exists():
                # Look through timestamped subdirectories
                for timestamp_dir in saves_dir.iterdir():
                    if timestamp_dir.is_dir():
                        # Look for model files
                        for model_file in timestamp_dir.glob('*.zip'):
                            # Extract level from filename
                            filename = model_file.stem
                            if f'{env_name}_level_' in filename:
                                try:
                                    level = int(filename.split(f'{env_name}_level_')[1].split('_')[0])
                                    models.append((env_name, str(model_file), level))
                                except (IndexError, ValueError):
                                    continue
    
    return models

def evaluate_model(env_name: str, model_path: str, level: int, n_eval_episodes: int = 10,
                   record_video: bool = True) -> Dict:
    """Evaluate a single model and return results."""
    print(f"\nEvaluating {env_name} level {level}: {model_path}")
    
    try:
        # Get environment class and config
        env_class = import_env_class(env_name)
        config_file = get_config_file(env_name, level)
        
        # Create evaluator
        evaluator = EnhancedPPOTrainerTester(
            env_class=env_class,
            config_file=config_file,
            log_dir=os.path.dirname(model_path),
            save_path=model_path.replace('.zip', ''),  # Remove .zip extension
            env_name=env_name,
            level=level,
            seed=42  # Use consistent seed for evaluation
        )
        
        # Run evaluation
        results = evaluator.evaluate_model(
            model_path=model_path,
            n_eval_episodes=n_eval_episodes,
            record_best_video=record_video
        )
        
        # Add model identification info
        results['env_name'] = env_name
        results['level'] = level
        results['model_file'] = os.path.basename(model_path)
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {env_name} level {level}: {e}")
        return None

def generate_performance_report(all_results: List[Dict], output_dir: str):
    """Generate comprehensive performance report."""
    if not all_results:
        print("No evaluation results to report")
        return
    
    # Create DataFrame for analysis
    df_data = []
    for result in all_results:
        if result is None:
            continue
            
        stats = result['statistics']
        df_data.append({
            'env_name': result['env_name'],
            'level': result['level'],
            'model_file': result['model_file'],
            'mean_reward': stats['mean_reward'],
            'std_reward': stats['std_reward'],
            'max_reward': stats['max_reward'],
            'min_reward': stats['min_reward'],
            'mean_length': stats['mean_length'],
            'success_rate': stats['success_rate'],
            'n_episodes': result['n_eval_episodes'],
            'video_recorded': result.get('video_recorded', False),
            'video_folder': result.get('video_folder', ''),
        })
    
    df = pd.DataFrame(df_data)
    
    # Save detailed results
    detailed_csv = os.path.join(output_dir, 'detailed_evaluation_results.csv')
    df.to_csv(detailed_csv, index=False)
    print(f"Detailed results saved to: {detailed_csv}")
    
    # Generate summary statistics
    summary_stats = []
    
    # Overall best performers
    best_overall = df.loc[df['mean_reward'].idxmax()]
    summary_stats.append(f"\\n=== BEST OVERALL PERFORMER ===")
    summary_stats.append(f"Environment: {best_overall['env_name']} Level {best_overall['level']}")
    summary_stats.append(f"Mean Reward: {best_overall['mean_reward']:.3f} ± {best_overall['std_reward']:.3f}")
    summary_stats.append(f"Success Rate: {best_overall['success_rate']:.2%}")
    summary_stats.append(f"Model: {best_overall['model_file']}")
    
    # Best performer per environment
    summary_stats.append(f"\\n=== BEST PERFORMERS PER ENVIRONMENT ===")
    for env_name in df['env_name'].unique():
        env_df = df[df['env_name'] == env_name]
        best_env = env_df.loc[env_df['mean_reward'].idxmax()]
        summary_stats.append(f"{env_name.upper()}:")
        summary_stats.append(f"  Level {best_env['level']}: {best_env['mean_reward']:.3f} ± {best_env['std_reward']:.3f} (Success: {best_env['success_rate']:.2%})")
    
    # Best performer per level across environments
    summary_stats.append(f"\\n=== BEST PERFORMERS PER LEVEL ===")
    for level in sorted(df['level'].unique()):
        level_df = df[df['level'] == level]
        best_level = level_df.loc[level_df['mean_reward'].idxmax()]
        summary_stats.append(f"Level {level}: {best_level['env_name']} - {best_level['mean_reward']:.3f} ± {best_level['std_reward']:.3f}")
    
    # Environment difficulty analysis
    summary_stats.append(f"\\n=== DIFFICULTY ANALYSIS ===")
    env_difficulty = df.groupby('env_name')['mean_reward'].agg(['mean', 'std', 'count']).round(3)
    env_difficulty = env_difficulty.sort_values('mean', ascending=False)
    summary_stats.append("Environment ranking by average performance:")
    for idx, (env_name, row) in enumerate(env_difficulty.iterrows(), 1):
        summary_stats.append(f"{idx}. {env_name}: {row['mean']:.3f} ± {row['std']:.3f} ({row['count']} models)")
    
    # Level difficulty analysis
    summary_stats.append(f"\\n=== LEVEL DIFFICULTY ANALYSIS ===")
    level_difficulty = df.groupby('level')['mean_reward'].agg(['mean', 'std', 'count']).round(3)
    level_difficulty = level_difficulty.sort_values('mean', ascending=False)
    summary_stats.append("Level ranking by average performance:")
    for idx, (level, row) in enumerate(level_difficulty.iterrows(), 1):
        summary_stats.append(f"{idx}. Level {level}: {row['mean']:.3f} ± {row['std']:.3f} ({row['count']} models)")
    
    # Success rate analysis
    summary_stats.append(f"\\n=== SUCCESS RATE ANALYSIS ===")
    high_success = df[df['success_rate'] >= 0.8]
    if len(high_success) > 0:
        summary_stats.append(f"Models with ≥80% success rate: {len(high_success)}")
        for _, row in high_success.iterrows():
            summary_stats.append(f"  {row['env_name']} L{row['level']}: {row['success_rate']:.2%}")
    else:
        summary_stats.append("No models achieved ≥80% success rate")
    
    # Save summary report
    summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("\\n".join(summary_stats))
    
    print(f"Summary report saved to: {summary_file}")
    print("\\n" + "\\n".join(summary_stats))

def main():
    parser = argparse.ArgumentParser(description='Batch evaluate all trained MIAC models')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Base directory containing trained models')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: models_dir/evaluation_results)')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                        help='Number of episodes per evaluation')
    parser.add_argument('--record_videos', action='store_true', default=True,
                        help='Record videos of best runs')
    parser.add_argument('--env_filter', type=str, nargs='+', default=None,
                        help='Only evaluate specific environments')
    parser.add_argument('--level_filter', type=int, nargs='+', default=None,
                        help='Only evaluate specific levels')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.models_dir):
        raise ValueError(f"Models directory does not exist: {args.models_dir}")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.models_dir, 'evaluation_results')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all trained models
    print(f"Searching for trained models in: {args.models_dir}")
    all_models = find_trained_models(args.models_dir)
    
    # Apply filters
    if args.env_filter:
        all_models = [(env, path, level) for env, path, level in all_models if env in args.env_filter]
    
    if args.level_filter:
        all_models = [(env, path, level) for env, path, level in all_models if level in args.level_filter]
    
    print(f"Found {len(all_models)} models to evaluate")
    
    if not all_models:
        print("No models found for evaluation")
        return
    
    # Evaluate all models
    all_results = []
    for i, (env_name, model_path, level) in enumerate(all_models, 1):
        print(f"\\nProgress: {i}/{len(all_models)}")
        result = evaluate_model(
            env_name=env_name,
            model_path=model_path,
            level=level,
            n_eval_episodes=args.n_eval_episodes,
            record_video=args.record_videos
        )
        if result:
            all_results.append(result)
    
    # Generate performance report
    print(f"\\nGenerating performance report...")
    generate_performance_report(all_results, args.output_dir)
    
    # Save raw results
    raw_results_file = os.path.join(args.output_dir, 'raw_evaluation_results.json')
    with open(raw_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total models evaluated: {len(all_results)}")

if __name__ == "__main__":
    main()
