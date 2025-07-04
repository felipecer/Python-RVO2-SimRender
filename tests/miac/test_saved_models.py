#!/usr/bin/env python
"""
Console application for testing and evaluating saved PPO models from comprehensive training.

This script allows you to:
- Browse previous training runs from the master CSV log
- Filter runs by environment, level, timestamp, or performance
- Select specific models for re-evaluation
- Generate new videos and performance metrics
- Compare multiple models

Usage:
    python test_saved_models.py
"""

import os
import sys
import csv
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers.enhanced_trainer_testers import EnhancedPPOTrainerTester

class SavedModelTester:
    """Console application for testing saved PPO models."""
    
    def __init__(self):
        self.base_dir = PROJECT_ROOT / 'tests' / 'miac' / 'comprehensive_results'
        self.csv_path = self.base_dir / 'master_runs_log.csv'
        self.runs_df = None
        
    def load_runs_data(self) -> bool:
        """Load the runs data from CSV file."""
        if not self.csv_path.exists():
            print(f"❌ Master runs log not found: {self.csv_path}")
            print("Run comprehensive_training.py first to generate training data.")
            return False
        
        try:
            self.runs_df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded {len(self.runs_df)} runs from {self.csv_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
    
    def display_summary(self):
        """Display a summary of all runs."""
        if self.runs_df is None:
            return
        
        print(f"\n{'='*80}")
        print("TRAINING RUNS SUMMARY")
        print(f"{'='*80}")
        
        total_runs = len(self.runs_df)
        successful_runs = len(self.runs_df[self.runs_df['status'] == 'SUCCESS'])
        failed_runs = total_runs - successful_runs
        
        print(f"Total runs: {total_runs}")
        print(f"Successful: {successful_runs}")
        print(f"Failed: {failed_runs}")
        
        if successful_runs > 0:
            # Performance statistics for successful runs
            success_df = self.runs_df[self.runs_df['status'] == 'SUCCESS']
            print(f"\nPerformance Statistics (Successful Runs):")
            print(f"  Mean reward: {success_df['mean_reward'].mean():.2f} ± {success_df['mean_reward'].std():.2f}")
            print(f"  Best reward: {success_df['mean_reward'].max():.2f}")
            print(f"  Worst reward: {success_df['mean_reward'].min():.2f}")
            
            # Best performing run
            best_run = success_df.loc[success_df['mean_reward'].idxmax()]
            print(f"\nBest performing run:")
            print(f"  Run ID: {best_run['run_id']}")
            print(f"  Environment: {best_run['env_name']} level {best_run['level']}")
            print(f"  Reward: {best_run['mean_reward']:.2f}")
            
        # Runs by environment
        print(f"\nRuns by Environment:")
        env_summary = self.runs_df.groupby('env_name').agg({
            'status': lambda x: (x == 'SUCCESS').sum(),
            'mean_reward': 'mean'
        }).round(2)
        
        for env_name, row in env_summary.iterrows():
            successful_count = int(row['status'])
            mean_reward = row['mean_reward'] if not pd.isna(row['mean_reward']) else 0
            print(f"  {env_name}: {successful_count} successful runs, avg reward: {mean_reward:.2f}")
    
    def display_runs_table(self, df: pd.DataFrame = None, limit: int = 20):
        """Display runs in a formatted table."""
        if df is None:
            df = self.runs_df
        
        if df is None or len(df) == 0:
            print("No runs to display.")
            return
        
        # Limit the number of displayed runs
        display_df = df.head(limit)
        
        print(f"\n{'='*120}")
        print(f"RUNS TABLE (showing {len(display_df)} of {len(df)} runs)")
        print(f"{'='*120}")
        
        # Format columns for display
        for idx, row in display_df.iterrows():
            status_icon = "✅" if row['status'] == 'SUCCESS' else "❌"
            reward = f"{row['mean_reward']:.1f}" if pd.notna(row['mean_reward']) else "N/A"
            time_str = row['completed_timestamp'][:16] if pd.notna(row['completed_timestamp']) else "N/A"
            
            print(f"{idx+1:2d}. {status_icon} {row['run_id'][:25]:<25} | "
                  f"{row['env_name']:<10} L{row['level']} | "
                  f"Reward: {reward:>8} | {time_str}")
        
        if len(df) > limit:
            print(f"\n... and {len(df) - limit} more runs (use filters to narrow down)")
    
    def filter_runs_menu(self) -> pd.DataFrame:
        """Interactive menu for filtering runs."""
        filtered_df = self.runs_df.copy()
        
        while True:
            print(f"\n{'='*60}")
            print("FILTER RUNS")
            print(f"{'='*60}")
            print(f"Current selection: {len(filtered_df)} runs")
            print("\nFilter options:")
            print("1. By environment")
            print("2. By level") 
            print("3. By status (SUCCESS/FAILED)")
            print("4. By batch timestamp")
            print("5. By reward range")
            print("6. Reset filters")
            print("7. Continue with current selection")
            
            choice = input("\nSelect filter option (1-7): ").strip()
            
            if choice == '1':
                envs = filtered_df['env_name'].unique()
                print(f"\nAvailable environments: {', '.join(envs)}")
                env_choice = input("Enter environment name: ").strip()
                if env_choice in envs:
                    filtered_df = filtered_df[filtered_df['env_name'] == env_choice]
                    print(f"Filtered to {len(filtered_df)} runs")
                else:
                    print("Invalid environment name")
                    
            elif choice == '2':
                levels = sorted(filtered_df['level'].unique())
                print(f"\nAvailable levels: {levels}")
                try:
                    level_choice = int(input("Enter level: ").strip())
                    if level_choice in levels:
                        filtered_df = filtered_df[filtered_df['level'] == level_choice]
                        print(f"Filtered to {len(filtered_df)} runs")
                    else:
                        print("Invalid level")
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == '3':
                print("\nAvailable statuses: SUCCESS, FAILED")
                status_choice = input("Enter status: ").strip().upper()
                if status_choice in ['SUCCESS', 'FAILED']:
                    filtered_df = filtered_df[filtered_df['status'] == status_choice]
                    print(f"Filtered to {len(filtered_df)} runs")
                else:
                    print("Invalid status")
                    
            elif choice == '4':
                timestamps = sorted(filtered_df['batch_timestamp'].unique())
                print(f"\nAvailable batch timestamps:")
                for i, ts in enumerate(timestamps, 1):
                    count = len(filtered_df[filtered_df['batch_timestamp'] == ts])
                    print(f"  {i}. {ts} ({count} runs)")
                
                try:
                    ts_idx = int(input("\nSelect timestamp number: ").strip()) - 1
                    if 0 <= ts_idx < len(timestamps):
                        selected_ts = timestamps[ts_idx]
                        filtered_df = filtered_df[filtered_df['batch_timestamp'] == selected_ts]
                        print(f"Filtered to {len(filtered_df)} runs")
                    else:
                        print("Invalid selection")
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == '5':
                success_df = filtered_df[filtered_df['status'] == 'SUCCESS']
                if len(success_df) == 0:
                    print("No successful runs to filter by reward")
                    continue
                    
                min_reward = success_df['mean_reward'].min()
                max_reward = success_df['mean_reward'].max()
                print(f"\nReward range: {min_reward:.2f} to {max_reward:.2f}")
                
                try:
                    min_input = input(f"Enter minimum reward (or press Enter for {min_reward:.2f}): ").strip()
                    max_input = input(f"Enter maximum reward (or press Enter for {max_reward:.2f}): ").strip()
                    
                    min_val = float(min_input) if min_input else min_reward
                    max_val = float(max_input) if max_input else max_reward
                    
                    filtered_df = filtered_df[
                        (filtered_df['mean_reward'] >= min_val) & 
                        (filtered_df['mean_reward'] <= max_val)
                    ]
                    print(f"Filtered to {len(filtered_df)} runs")
                except ValueError:
                    print("Please enter valid numbers")
                    
            elif choice == '6':
                filtered_df = self.runs_df.copy()
                print("Filters reset")
                
            elif choice == '7':
                break
                
            else:
                print("Invalid choice. Please select 1-7.")
        
        return filtered_df
    
    def select_run_for_testing(self, filtered_df: pd.DataFrame) -> Optional[Dict]:
        """Allow user to select a specific run for testing."""
        if len(filtered_df) == 0:
            print("No runs available for selection.")
            return None
        
        # Display runs with index numbers
        self.display_runs_table(filtered_df)
        
        while True:
            try:
                choice = input(f"\nSelect run number (1-{len(filtered_df)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                
                run_idx = int(choice) - 1
                if 0 <= run_idx < len(filtered_df):
                    selected_run = filtered_df.iloc[run_idx].to_dict()
                    return selected_run
                else:
                    print(f"Please enter a number between 1 and {len(filtered_df)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    
    def test_selected_model(self, run_data: Dict):
        """Test/evaluate a selected model."""
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {run_data['run_id']}")
        print(f"{'='*80}")
        
        # Check if model exists
        model_path = run_data['model_path']
        if not os.path.exists(model_path) and not os.path.exists(f"{model_path}.zip"):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        # Ensure model path has .zip extension if needed
        if not model_path.endswith('.zip') and os.path.exists(f"{model_path}.zip"):
            model_path = f"{model_path}.zip"
        
        print(f"Model: {model_path}")
        print(f"Environment: {run_data['env_name']} level {run_data['level']}")
        print(f"Original performance: {run_data['mean_reward']:.2f} ± {run_data['std_reward']:.2f}")
        
        # Get number of evaluation episodes
        try:
            n_episodes = int(input("\nNumber of evaluation episodes (default 10): ").strip() or "10")
        except ValueError:
            n_episodes = 10
        
        # Ask about video recording
        record_video = input("Record video of best run? (y/n, default y): ").strip().lower()
        record_video = record_video != 'n'
        
        # Set up evaluation
        try:
            # Import environment class
            from comprehensive_training import import_env_class, get_config_file
            
            env_class = import_env_class(run_data['env_name'])
            config_file = get_config_file(run_data['env_name'], run_data['level'])
            
            # Create video directory for new evaluation
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_dir = None
            if record_video:
                video_dir = self.base_dir / 'evaluation_videos' / f"{run_data['run_id']}_{timestamp}"
                os.makedirs(video_dir, exist_ok=True)
            
            # Create trainer for evaluation
            trainer = EnhancedPPOTrainerTester(
                env_class=env_class,
                config_file=config_file,
                log_dir=None,  # Not needed for evaluation
                save_path=model_path,
                render_mode="rgb_array",
                seed=42,
                unique_id=f"eval_{timestamp}",
                env_name=run_data['env_name'],
                level=run_data['level']
            )
            
            print(f"\n🧪 Starting evaluation with {n_episodes} episodes...")
            
            # Run evaluation
            eval_results = trainer.evaluate_model(
                model_path=model_path,
                n_eval_episodes=n_episodes,
                record_best_video=record_video,
                video_folder=str(video_dir) if video_dir else None
            )
            
            # Display results
            print(f"\n{'='*60}")
            print("EVALUATION RESULTS")
            print(f"{'='*60}")
            print(f"Mean reward: {eval_results['statistics']['mean_reward']:.2f} ± {eval_results['statistics']['std_reward']:.2f}")
            print(f"Success rate: {eval_results['statistics']['success_rate']:.2%}")
            print(f"Episode length: {eval_results['statistics']['mean_length']:.1f} ± {eval_results['statistics']['std_length']:.1f}")
            print(f"Best episode reward: {eval_results['statistics']['max_reward']:.2f}")
            print(f"Worst episode reward: {eval_results['statistics']['min_reward']:.2f}")
            
            if record_video and eval_results.get('video_recorded'):
                print(f"\n🎥 Video recorded: {video_dir}")
            
            # Save evaluation results
            eval_file = video_dir / 'evaluation_results.json' if video_dir else None
            if eval_file:
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                print(f"📊 Results saved: {eval_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def main_menu(self):
        """Main menu loop."""
        print(f"\n{'='*80}")
        print("SAVED MODEL TESTER")
        print(f"{'='*80}")
        print("Console application for testing and evaluating saved PPO models")
        
        if not self.load_runs_data():
            return
        
        while True:
            self.display_summary()
            
            print(f"\n{'='*60}")
            print("MAIN MENU")
            print(f"{'='*60}")
            print("1. View all runs")
            print("2. Filter and select runs")
            print("3. Test a specific model")
            print("4. Quit")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                self.display_runs_table()
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                filtered_df = self.filter_runs_menu()
                if len(filtered_df) > 0:
                    selected_run = self.select_run_for_testing(filtered_df)
                    if selected_run:
                        self.test_selected_model(selected_run)
                        input("\nPress Enter to continue...")
                
            elif choice == '3':
                # Quick selection from successful runs
                success_runs = self.runs_df[self.runs_df['status'] == 'SUCCESS']
                if len(success_runs) == 0:
                    print("No successful runs available for testing.")
                    input("Press Enter to continue...")
                    continue
                    
                selected_run = self.select_run_for_testing(success_runs)
                if selected_run:
                    self.test_selected_model(selected_run)
                    input("\nPress Enter to continue...")
                
            elif choice == '4':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-4.")

def main():
    """Main function."""
    try:
        app = SavedModelTester()
        app.main_menu()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
