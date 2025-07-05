#!/usr/bin/env python
"""
PPO Model Evaluation Script

This script evaluates a trained PPO model from Stable-Baselines3 using deterministic inference.
It uses the same environment setup and configuration as in the EnhancedPPOTrainerTester class.

Features:
- Load pretrained PPO models with metadata
- Deterministic evaluation over multiple episodes
- CSV logging of evaluation results
- Environment auto-detection from model metadata
- Comprehensive error handling
- Summary statistics reporting
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

# Environment class mappings - support both v1 and v2 classes
ENV_CLASSES_V1 = {
    'circle': 'rl_environments.single_agent.miac.circle.RVOMiacCircle',
    'incoming': 'rl_environments.single_agent.miac.incoming.RVOMiacIncoming',
    'perp1': 'rl_environments.single_agent.miac.perp1.RVOMiacPerp1', 
    'perp2': 'rl_environments.single_agent.miac.perp2.RVOMiacPerp2',
    'two_paths': 'rl_environments.single_agent.miac.two_paths.RVOMiacTwoPaths'
}

ENV_CLASSES_V2 = {
    'circle': 'rl_environments.single_agent.miac.v2.circle.RVOMiacCircleV2',
    'incoming': 'rl_environments.single_agent.miac.v2.incoming.RVOMiacIncoming2',
    'perp1': 'rl_environments.single_agent.miac.v2.perp1.RVOMiacPerp1V2', 
    'perp2': 'rl_environments.single_agent.miac.v2.perp2.RVOMiacPerp2V2',
    'two_paths': 'rl_environments.single_agent.miac.v2.two_paths.RVOMiacTwoPathsV2'
}

def import_env_class(env_name: str, use_v2: bool = True):
    """Dynamically import the environment class based on environment name and version."""
    env_classes = ENV_CLASSES_V2 if use_v2 else ENV_CLASSES_V1
    
    if env_name not in env_classes:
        available = list(env_classes.keys())
        raise ValueError(f"Unknown environment: {env_name}. Available: {available}")
    
    module_path, class_name = env_classes[env_name].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def get_config_file(env_name: str, level: int) -> str:
    """Get the configuration file path for the given environment and level."""
    config_file = PROJECT_ROOT / f'simulator/worlds/miac/{env_name}/{env_name}_level_{level}.yaml'
    
    if not config_file.exists():
        raise ValueError(f"Configuration file not found: {config_file}")
    
    return str(config_file)

def load_model_metadata(model_path: str) -> Optional[Dict]:
    """Load model metadata if available."""
    metadata_path = f"{model_path}_metadata.json"
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata from: {metadata_path}")
            return metadata
        except Exception as e:
            print(f"Warning: Could not load metadata from {metadata_path}: {e}")
    else:
        print(f"No metadata file found at: {metadata_path}")
    
    return None

def extract_env_info_from_metadata(metadata: Dict) -> Tuple[str, int, bool]:
    """Extract environment name, level, and version info from metadata."""
    env_info = metadata.get('environment_info', {})
    
    env_name = env_info.get('env_name')
    level = env_info.get('level')
    
    # Determine if it's v2 based on the env_class string
    env_class_str = env_info.get('env_class', '')
    use_v2 = 'v2' in env_class_str.lower()
    
    return env_name, level, use_v2

def create_environment(env_name: str, level: int, use_v2: bool = True, 
                      render_mode: str = None, seed: int = 42):
    """Create environment instance."""
    env_class = import_env_class(env_name, use_v2=use_v2)
    config_file = get_config_file(env_name, level)
    
    # Create environment - check if it uses step_mode parameter
    try:
        env = env_class(
            config_file=config_file,
            render_mode=render_mode,
            seed=seed,
            step_mode='min_dist'  # Try with step_mode first
        )
    except TypeError:
        # If step_mode is not supported, create without it
        env = env_class(
            config_file=config_file,
            render_mode=render_mode,
            seed=seed
        )
    
    return env

def evaluate_model(model_path: str, env_name: str = None, level: int = None, 
                  num_episodes: int = 100, seed: int = 42, 
                  output_csv: str = None, use_v2: bool = True) -> Dict[str, Any]:
    """Evaluate a trained PPO model."""
    
    print(f"Loading model from: {model_path}")
    
    # Load the model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    # Load metadata if available
    metadata = load_model_metadata(model_path)
    
    # Determine environment info
    if metadata and (env_name is None or level is None):
        meta_env_name, meta_level, meta_use_v2 = extract_env_info_from_metadata(metadata)
        env_name = env_name or meta_env_name
        level = level if level is not None else meta_level
        use_v2 = meta_use_v2  # Use version from metadata
        print(f"Environment info from metadata: {env_name} level {level} (v2: {use_v2})")
    
    if env_name is None or level is None:
        raise ValueError("Environment name and level must be provided either as arguments or in model metadata")
    
    # Set up output CSV path
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"evaluations/{env_name}_level_{level}_{timestamp}.csv"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"Evaluating: {env_name} level {level}")
    print(f"Episodes: {num_episodes}")
    print(f"Results will be saved to: {output_csv}")
    
    # Create environment
    try:
        env = create_environment(env_name, level, use_v2=use_v2, seed=seed)
        print("✅ Environment created successfully")
    except Exception as e:
        raise ValueError(f"Failed to create environment {env_name} level {level}: {e}")
    
    # Run evaluation episodes
    results = []
    successful_episodes = 0
    failed_episodes = 0
    
    print(f"\nStarting evaluation...")
    
    for episode in range(num_episodes):
        try:
            # Reset environment
            reset_result = env.reset(seed=seed + episode)
            
            # Handle both Gym API (obs, info) and old API (obs)
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            episode_start_time = datetime.now()
            
            while not done:
                # Get action from model (deterministic)
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step in environment
                step_result = env.step(action)
                
                # Handle both new Gym API and old API
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                episode_length += 1
                
                # Safety break for very long episodes
                if episode_length > 10000:
                    print(f"Warning: Episode {episode + 1} exceeded 10000 steps, terminating")
                    done = True
            
            # Check success (if environment supports it)
            success = False
            try:
                success = env.is_done(0)  # Check if goal was reached
            except (AttributeError, TypeError):
                # Some environments might not have is_done method
                # Try to infer from reward or info
                success = episode_reward > 0 or info.get('success', False)
            
            # Record episode results
            episode_result = {
                'episode': episode + 1,
                'total_reward': float(episode_reward),
                'episode_length': episode_length,
                'success': bool(success),
                'env_name': env_name,
                'level': level,
                'timestamp': episode_start_time.isoformat(),
                'seed': seed + episode
            }
            
            results.append(episode_result)
            successful_episodes += 1
            
            # Progress reporting
            if (episode + 1) % 10 == 0:
                success_rate = successful_episodes / (episode + 1)
                avg_reward = np.mean([r['total_reward'] for r in results])
                print(f"  Progress: {episode + 1}/{num_episodes} episodes, "
                      f"Success rate: {success_rate:.2%}, "
                      f"Avg reward: {avg_reward:.2f}")
        
        except Exception as e:
            print(f"Warning: Episode {episode + 1} failed: {e}")
            failed_episodes += 1
            
            # Record failed episode
            episode_result = {
                'episode': episode + 1,
                'total_reward': 0.0,
                'episode_length': 0,
                'success': False,
                'env_name': env_name,
                'level': level,
                'timestamp': datetime.now().isoformat(),
                'seed': seed + episode,
                'error': str(e)
            }
            results.append(episode_result)
    
    # Clean up environment
    env.close()
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # Calculate summary statistics
    valid_results = df[df['success'].notna()]  # Exclude failed episodes from stats
    
    if len(valid_results) > 0:
        summary_stats = {
            'total_episodes': num_episodes,
            'successful_episodes': successful_episodes,
            'failed_episodes': failed_episodes,
            'mean_reward': float(valid_results['total_reward'].mean()),
            'std_reward': float(valid_results['total_reward'].std()),
            'min_reward': float(valid_results['total_reward'].min()),
            'max_reward': float(valid_results['total_reward'].max()),
            'mean_episode_length': float(valid_results['episode_length'].mean()),
            'std_episode_length': float(valid_results['episode_length'].std()),
            'success_rate': float(valid_results['success'].mean()),
            'env_name': env_name,
            'level': level,
            'model_path': model_path,
            'output_csv': output_csv,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    else:
        summary_stats = {
            'total_episodes': num_episodes,
            'successful_episodes': 0,
            'failed_episodes': failed_episodes,
            'error': 'All episodes failed'
        }
    
    return {
        'summary': summary_stats,
        'detailed_results': df,
        'metadata': metadata
    }

def print_summary_stats(summary: Dict[str, Any]):
    """Print formatted summary statistics."""
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Environment: {summary['env_name']} level {summary['level']}")
    print(f"Total episodes: {summary['total_episodes']}")
    print(f"Successful episodes: {summary['successful_episodes']}")
    print(f"Failed episodes: {summary['failed_episodes']}")
    
    if summary['successful_episodes'] > 0:
        print(f"\nPerformance Metrics:")
        print(f"  Mean reward: {summary['mean_reward']:.3f} ± {summary['std_reward']:.3f}")
        print(f"  Reward range: [{summary['min_reward']:.3f}, {summary['max_reward']:.3f}]")
        print(f"  Mean episode length: {summary['mean_episode_length']:.1f} ± {summary['std_episode_length']:.1f}")
        print(f"  Success rate: {summary['success_rate']:.2%}")
    
    print(f"\nResults saved to: {summary['output_csv']}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Evaluate a trained PPO model using deterministic inference'
    )
    parser.add_argument('model_path', type=str,
                        help='Path to the trained PPO model (.zip file)')
    parser.add_argument('--env', type=str, choices=list(ENV_CLASSES_V2.keys()),
                        help='Environment name (auto-detected from metadata if not provided)')
    parser.add_argument('--level', type=int, choices=[0, 1, 2, 3],
                        help='Environment level (auto-detected from metadata if not provided)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str,
                        help='Output CSV file path (default: auto-generated)')
    parser.add_argument('--use-v1', action='store_true',
                        help='Force use of v1 environment classes')
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return 1
    
    try:
        # Run evaluation
        results = evaluate_model(
            model_path=args.model_path,
            env_name=args.env,
            level=args.level,
            num_episodes=args.episodes,
            seed=args.seed,
            output_csv=args.output,
            use_v2=not args.use_v1
        )
        
        # Print summary
        print_summary_stats(results['summary'])
        
        print(f"\n✅ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
