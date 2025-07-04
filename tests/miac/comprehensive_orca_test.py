#!/usr/bin/env python
"""
Comprehensive ORCA Testing Script

This script consolidates all the individual baseline_orca_*.py scripts into one unified test.
It automatically discovers all available environment-level combinations and runs ORCA baseline
tests with data collection and optional video recording.

Features:
- Auto-discovery of all MIAC environment-level combinations
- Unified data collection with central CSV logging
- Optional video recording using RecordablePygameRenderer
- Parallel execution support (optional)
- Comprehensive error handling and progress reporting
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import time
import csv
import fcntl
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Environment class mappings for v2 classes (the actual classes that exist)
ENV_CLASSES = {
    'circle': 'rl_environments.single_agent.miac.v2.circle.RVOMiacCircleV2',
    'incoming': 'rl_environments.single_agent.miac.v2.incoming.RVOMiacIncoming2',
    'perp1': 'rl_environments.single_agent.miac.v2.perp1.RVOMiacPerp1V2',
    'perp2': 'rl_environments.single_agent.miac.v2.perp2.RVOMiacPerp2V2',
    'two_paths': 'rl_environments.single_agent.miac.v2.two_paths.RVOMiacTwoPathsV2'
}

def import_env_class(env_name: str):
    """Dynamically import the environment class based on environment name."""
    if env_name not in ENV_CLASSES:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_CLASSES.keys())}")
    
    module_path, class_name = ENV_CLASSES[env_name].rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def get_config_file(env_name: str, level: int) -> str:
    """Get the configuration file path for the given environment and level."""
    config_file = PROJECT_ROOT / f'simulator/worlds/miac/{env_name}/{env_name}_level_{level}.yaml'
    
    if not config_file.exists():
        raise ValueError(f"Configuration file not found: {config_file}")
    
    return str(config_file)

def discover_valid_combinations() -> List[Tuple[str, int]]:
    """Discover all valid environment-level combinations based on available config files."""
    combinations = []
    environments = list(ENV_CLASSES.keys())
    levels = [0, 1, 2, 3]
    
    for env_name in environments:
        for level in levels:
            try:
                config_file = get_config_file(env_name, level)
                combinations.append((env_name, level))
                print(f"  ✓ Found: {env_name} level {level}")
            except ValueError:
                print(f"  ✗ Missing: {env_name} level {level}")
    
    return combinations

def setup_directories(base_dir: str) -> Dict[str, str]:
    """Set up directory structure for comprehensive ORCA testing."""
    directories = {
        'base': base_dir,
        'baseline_results': os.path.join(base_dir, 'baseline_results'),
        'videos': os.path.join(base_dir, 'videos'),
        'detailed_results': os.path.join(base_dir, 'detailed_results')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def log_run_to_central_csv(central_log_file: str, result: Dict[str, Any]):
    """Log a run to the central CSV file with file locking."""
    file_exists = os.path.isfile(central_log_file)
    
    with open(central_log_file, mode='a', newline='') as file:
        # Acquire an exclusive lock
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        
        writer = csv.writer(file)
        # Only write header if file is newly created
        if not file_exists:
            writer.writerow([
                'timestamp', 'tag', 'unique_id', 'config_file', 'algorithm', 
                'env_name', 'level', 'success_rate', 'avg_steps', 'std_steps', 
                'avg_reward', 'std_reward', 'avg_duration', 'num_runs', 'seed',
                'video_recorded', 'video_folder'
            ])
        
        writer.writerow([
            datetime.now().isoformat(),
            result['tag'],
            result['unique_id'],
            result['config_file'],
            'baseline_orca',
            result['env_name'],
            result['level'],
            f"{result['success_rate']:.4f}",
            f"{result['avg_steps']:.2f}",
            f"{result['std_steps']:.2f}",
            f"{result['avg_reward']:.2f}",
            f"{result['std_reward']:.2f}",
            f"{result['avg_duration']:.2f}",
            result['num_runs'],
            result['seed'],
            result.get('video_recorded', False),
            result.get('video_folder', '')
        ])
        
        # Release the lock
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)

def run_orca_baseline_for_env_level(env_name: str, level: int, num_runs: int = 1000, 
                                   seed: int = 42, record_video: bool = False, 
                                   video_folder: str = None, tag: str = 'comprehensive_orca') -> Dict[str, Any]:
    """Run baseline ORCA algorithm for a specific environment-level combination."""
    
    print(f"\n{'='*80}")
    print(f"Running ORCA Baseline: {env_name} level {level}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Get environment class and config file
        env_class = import_env_class(env_name)
        config_file = get_config_file(env_name, level)
        
        # Generate timestamp-based ID for this batch of runs
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_id = f"{timestamp}_{env_name}_level_{level}"
        
        results = []
        
        # Set up video recording if requested
        render_mode = None
        env_video_folder = None
        videos_saved = []
        
        if record_video:
            render_mode = "rgb_array"
            if video_folder:
                # Create well-organized video folder structure
                timestamp_short = datetime.now().strftime("%Y%m%d_%H%M")
                env_video_folder = os.path.join(
                    video_folder, 
                    f"{env_name}_level_{level}",
                    f"orca_baseline_{timestamp_short}"
                )
                os.makedirs(env_video_folder, exist_ok=True)
                
                # Create a README for this video set
                readme_path = os.path.join(env_video_folder, "README.txt")
                with open(readme_path, 'w') as f:
                    f.write(f"ORCA Baseline Videos - {env_name} Level {level}\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Configuration: {config_file}\n")
                    f.write(f"Runs: {num_runs}, Seed: {seed}\n")
                    f.write(f"Algorithm: Pure ORCA (action=[0,0])\n")
                    f.write("="*60 + "\n\n")
                    f.write("Video filename format:\n")
                    f.write("run_XXXX_STATUS_steps_XXX_reward_XX.X.mp4\n\n")
                    f.write("Where:\n")
                    f.write("- XXXX: Run number (0000-9999)\n")
                    f.write("- STATUS: SUCCESS or FAILED\n")
                    f.write("- steps_XXX: Number of simulation steps\n")
                    f.write("- reward_XX.X: Episode reward\n\n")
        
        for run in range(num_runs):
            # Create environment with min_dist mode (same as existing baseline scripts)
            env = env_class(
                config_file=config_file,
                render_mode=render_mode,
                seed=seed + run,  # Vary seed per run
                step_mode='min_dist'
            )
            
            # Reset environment
            obs, _ = env.reset()
            
            # Track metrics
            episode_reward = 0
            steps = 0
            run_start_time = time.time()
            
            # Run episode
            done = False
            truncated = False
            
            # Store frames for video if recording
            frames = []
            
            while not done and not truncated:
                # Execute action [0,0] (pure ORCA)
                action = np.array([0.0, 0.0])
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                # Capture frame for video if recording
                if record_video and render_mode == "rgb_array":
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
            
            # Calculate runtime
            duration = time.time() - run_start_time
            success = env.is_done(0)  # Check if goal was reached
            
            # Save video for this run if recording
            if record_video and frames and video_folder:
                try:
                    # Save as video using simple method (without moviepy dependency)
                    import imageio
                    
                    # Create descriptive filename with performance info
                    status = "SUCCESS" if success else "FAILED"
                    video_filename = f"run_{run:04d}_{status}_steps_{steps}_reward_{episode_reward:.1f}.mp4"
                    video_path = os.path.join(env_video_folder, video_filename)
                    
                    # Save video (limit to reasonable number of videos per environment)
                    # Save all successful runs and first 5 failed runs for variety
                    failed_videos_count = len([v for v in videos_saved if not v.get('success', True)])
                    should_save = success or (not success and failed_videos_count < 5)
                    
                    if should_save:
                        imageio.mimsave(video_path, frames, fps=30)
                        videos_saved.append({
                            'filename': video_filename,
                            'run': run,
                            'success': success,
                            'steps': steps,
                            'reward': episode_reward,
                            'duration': duration
                        })
                        print(f"    📹 Video saved: {video_filename}")
                    
                except ImportError:
                    print(f"Warning: imageio not available for video saving")
                except Exception as e:
                    print(f"Warning: Could not save video for run {run}: {e}")
            
            # Record results
            results.append({
                'run': run,
                'seed': seed + run,
                'steps': steps,
                'reward': episode_reward,
                'success': success,
                'duration': duration,
                'truncated': truncated,
                'timestamp': datetime.now().isoformat(),
                'tag': tag,
                'unique_id': unique_id,
                'config_file': config_file,
                'env_name': env_name,
                'level': level
            })
            
            # Progress reporting
            if (run + 1) % 100 == 0 or run + 1 == num_runs:
                print(f"  Progress: {run + 1}/{num_runs} runs completed")
            
            # Close environment to free resources
            env.close()
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary_stats = {
            'env_name': env_name,
            'level': level,
            'success_rate': df['success'].mean(),
            'avg_steps': df['steps'].mean(),
            'std_steps': df['steps'].std(),
            'avg_reward': df['reward'].mean(),
            'std_reward': df['reward'].std(),
            'avg_duration': df['duration'].mean(),
            'num_runs': num_runs,
            'seed': seed,
            'tag': tag,
            'unique_id': unique_id,
            'config_file': config_file,
            'total_time': time.time() - start_time,
            'video_recorded': record_video,
            'video_folder': env_video_folder if record_video and video_folder else '',
            'videos_saved': videos_saved if record_video else []
        }
        
        # Print summary statistics
        print(f"\n✅ Completed: {env_name} level {level}")
        print(f"Success rate: {summary_stats['success_rate']:.2%}")
        print(f"Average steps: {summary_stats['avg_steps']:.2f} ± {summary_stats['std_steps']:.2f}")
        print(f"Average reward: {summary_stats['avg_reward']:.2f} ± {summary_stats['std_reward']:.2f}")
        print(f"Average duration: {summary_stats['avg_duration']:.2f} seconds")
        print(f"Total runtime: {summary_stats['total_time']:.2f} seconds")
        
        # Video summary
        if record_video and videos_saved:
            print(f"📹 Videos saved: {len(videos_saved)} total")
            successful_videos = [v for v in videos_saved if v['success']]
            failed_videos = [v for v in videos_saved if not v['success']]
            print(f"   - Successful runs: {len(successful_videos)}")
            print(f"   - Failed runs: {len(failed_videos)}")
            print(f"   - Video folder: {env_video_folder}")
            
            # Update README with video summary
            if env_video_folder:
                readme_path = os.path.join(env_video_folder, "README.txt")
                with open(readme_path, 'a') as f:
                    f.write(f"RESULTS SUMMARY:\n")
                    f.write(f"Total runs: {num_runs}\n")
                    f.write(f"Success rate: {summary_stats['success_rate']:.2%}\n")
                    f.write(f"Videos saved: {len(videos_saved)}\n")
                    f.write(f"  - Successful: {len(successful_videos)}\n")
                    f.write(f"  - Failed: {len(failed_videos)}\n\n")
                    
                    if successful_videos:
                        f.write("BEST SUCCESSFUL RUNS (by reward):\n")
                        best_videos = sorted(successful_videos, key=lambda x: x['reward'], reverse=True)[:3]
                        for i, video in enumerate(best_videos, 1):
                            f.write(f"{i}. {video['filename']} (reward: {video['reward']:.1f}, steps: {video['steps']})\n")
                        f.write("\n")
                    
                    if failed_videos:
                        f.write("SAMPLE FAILED RUNS:\n")
                        for video in failed_videos[:3]:
                            f.write(f"- {video['filename']} (reward: {video['reward']:.1f}, steps: {video['steps']})\n")
                        f.write("\n")
        
        return {
            'success': True,
            'summary': summary_stats,
            'detailed_results': df,
            'error': None
        }
        
    except Exception as e:
        error_msg = f"Error running {env_name} level {level}: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            'success': False,
            'summary': {
                'env_name': env_name,
                'level': level,
                'error': error_msg
            },
            'detailed_results': None,
            'error': str(e)
        }

def main():
    """Main function to run comprehensive ORCA testing."""
    parser = argparse.ArgumentParser(description='Comprehensive ORCA Testing Script')
    parser.add_argument('--num-runs', type=int, default=1000, 
                        help='Number of runs per environment-level combination (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Base random seed (default: 42)')
    parser.add_argument('--record-video', action='store_true', 
                        help='Record videos of successful runs')
    parser.add_argument('--tag', type=str, default='comprehensive_orca', 
                        help='Tag for this test run (default: comprehensive_orca)')
    parser.add_argument('--env', type=str, choices=list(ENV_CLASSES.keys()), 
                        help='Run only specific environment (default: all)')
    parser.add_argument('--level', type=int, choices=[0, 1, 2, 3], 
                        help='Run only specific level (default: all)')
    
    args = parser.parse_args()
    
    print("MIAC ORCA Comprehensive Testing")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Runs per env-level: {args.num_runs}")
    print(f"  Base seed: {args.seed}")
    print(f"  Record videos: {args.record_video}")
    print(f"  Tag: {args.tag}")
    print("=" * 60)
    
    # Set up directories
    base_dir = str(PROJECT_ROOT / 'tests' / 'miac')
    directories = setup_directories(base_dir)
    
    print(f"Results will be saved to: {directories['base']}")
    
    # Discover valid combinations
    print("\nDiscovering valid environment-level combinations...")
    if args.env and args.level is not None:
        # Run only specific combination
        valid_combinations = [(args.env, args.level)]
    elif args.env:
        # Run only specific environment, all levels
        valid_combinations = []
        for level in [0, 1, 2, 3]:
            try:
                get_config_file(args.env, level)
                valid_combinations.append((args.env, level))
            except ValueError:
                pass
    else:
        # Discover all combinations
        valid_combinations = discover_valid_combinations()
    
    if not valid_combinations:
        print("❌ No valid combinations found!")
        return 1
    
    total_combinations = len(valid_combinations)
    print(f"\nFound {total_combinations} valid combinations to process")
    estimated_time = (total_combinations * args.num_runs * 2) / 3600  # Rough estimate: 2 seconds per run
    print(f"Estimated total time: {estimated_time:.1f} hours")
    
    # Process all combinations
    print(f"\nStarting comprehensive ORCA testing...")
    
    all_results = []
    overall_start_time = time.time()
    central_log_file = os.path.join(directories['baseline_results'], 'baseline_run_log.csv')
    
    for i, (env_name, level) in enumerate(valid_combinations, 1):
        print(f"\n[{i}/{total_combinations}] Processing: {env_name} level {level}")
        
        # Set up video folder if recording
        video_folder = None
        if args.record_video:
            video_folder = directories['videos']
        
        # Run ORCA baseline for this env-level combination
        result = run_orca_baseline_for_env_level(
            env_name=env_name,
            level=level,
            num_runs=args.num_runs,
            seed=args.seed,
            record_video=args.record_video,
            video_folder=video_folder,
            tag=args.tag
        )
        
        all_results.append(result)
        
        # Log to central CSV if successful
        if result['success']:
            log_run_to_central_csv(central_log_file, result['summary'])
            
            # Save detailed results to individual CSV
            detailed_csv_path = os.path.join(
                directories['detailed_results'], 
                f"{env_name}_level_{level}_{result['summary']['unique_id']}.csv"
            )
            result['detailed_results'].to_csv(detailed_csv_path, index=False)
            print(f"Detailed results saved to: {detailed_csv_path}")
    
    # Generate final summary
    total_time = time.time() - overall_start_time
    successful_runs = sum(1 for r in all_results if r['success'])
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ORCA TESTING COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Processed: {len(all_results)} combinations")
    print(f"Successful: {successful_runs}/{len(all_results)}")
    print(f"Central log: {central_log_file}")
    print(f"Detailed results: {directories['detailed_results']}")
    if args.record_video:
        print(f"Videos: {directories['videos']}")
    
    # Show any failed runs
    failed_runs = [r for r in all_results if not r['success']]
    if failed_runs:
        print(f"\n❌ Failed runs:")
        for result in failed_runs:
            print(f"  {result['summary']['env_name']} level {result['summary']['level']}: {result['error']}")
    
    print("\nTesting completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
