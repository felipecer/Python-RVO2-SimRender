#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import time
import uuid
import csv

from rl_environments.single_agent.miac.circle import RVOMiacCircle

def run_baseline_orca_circle_level1(num_runs=10, render_mode=None, seed=42, tag='baseline_orca'):
    """Run baseline ORCA algorithm on circle_level_1 with action=[0,0]"""
    
    config_file = './simulator/worlds/miac/circle/circle_level_1.yaml'
    results = []
    
    # Generate unique ID for this batch of runs
    unique_id = str(uuid.uuid4())
    
    # Set up paths similar to PPOTrainerTester structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'logs', unique_id)
    save_dir = os.path.join(script_dir, 'saves')
    
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup a central CSV file for all baseline run logging
    # Put it at the project root level in a dedicated directory
    # Setup a central CSV file for all baseline run logging in the miac directory
    miac_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)))  # This gets the miac directory
    central_log_dir = os.path.join(miac_dir, 'baseline_results')
    os.makedirs(central_log_dir, exist_ok=True)
    run_log_file = os.path.join(central_log_dir, 'baseline_run_log.csv')
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(run_log_file)
    
    for run in range(num_runs):
        # Create environment with min_dist mode
        env = RVOMiacCircle(
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
        start_time = time.time()
        
        # Run episode
        done = False
        truncated = False
        
        while not done and not truncated:
            # Execute action [0,0] (pure ORCA)
            action = np.array([0.0, 0.0])
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Optional: add a small delay if rendering to see what's happening
            if render_mode:
                time.sleep(0.05)
        
        # Calculate runtime
        duration = time.time() - start_time
        success = env.is_done(0)  # Check if goal was reached
        
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
            'config_file': config_file
        })
        
        print(f"Run {run+1}/{num_runs}: {'Success' if success else 'Failure'}, Steps: {steps}, Reward: {episode_reward:.2f}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Success rate: {df['success'].mean():.2f}")
    print(f"Average steps: {df['steps'].mean():.2f} ± {df['steps'].std():.2f}")
    print(f"Average reward: {df['reward'].mean():.2f} ± {df['reward'].std():.2f}")
    print(f"Average duration: {df['duration'].mean():.2f} seconds")
    
    # Save detailed results to saves folder
    results_file = os.path.join(save_dir, f'circle_level1_baseline_orca_{unique_id}.csv')
    df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    import fcntl
    
    with open(run_log_file, mode='a', newline='') as file:
        # Acquire an exclusive lock
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        
        writer = csv.writer(file)
        # Only write header if file is newly created
        if not file_exists:
            writer.writerow(['timestamp', 'tag', 'unique_id', 'config_file', 'algorithm', 'success_rate', 
                            'avg_steps', 'std_steps', 'avg_reward', 'std_reward', 'avg_duration', 'num_runs', 'seed'])
        
        writer.writerow([
            datetime.now().isoformat(),
            tag,
            unique_id,
            config_file,
            'baseline_orca',
            f"{df['success'].mean():.4f}",
            f"{df['steps'].mean():.2f}",
            f"{df['steps'].std():.2f}",
            f"{df['reward'].mean():.2f}",
            f"{df['reward'].std():.2f}",
            f"{df['duration'].mean():.2f}",
            num_runs,
            seed
        ])
        
        # Release the lock
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    
    print(f"Summary logged to: {run_log_file}")
    return df

if __name__ == "__main__":
    # Run without visualization by default
    df = run_baseline_orca_circle_level1(num_runs=10, render_mode=None, seed=42, tag='baseline_orca_circle_1')