#!/usr/bin/env python
"""
Script to export TensorBoard scalar data as matplotlib plots.
This script reads TensorBoard event files and generates high-quality plots as images.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    print("✅ TensorBoard backend available")
except ImportError:
    print("❌ TensorBoard backend not available. Install with: pip install tensorboard")
    sys.exit(1)

class TensorBoardPlotExporter:
    """Export TensorBoard scalar data as matplotlib plots."""
    
    def __init__(self, logdir: str, output_dir: str = None):
        self.logdir = Path(logdir)
        self.output_dir = Path(output_dir) if output_dir else self.logdir.parent / 'tensorboard_plots'
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    def find_event_files(self):
        """Find all TensorBoard event files in the log directory."""
        event_files = {}
        
        print(f"Searching for event files in: {self.logdir}")
        
        for root, dirs, files in os.walk(self.logdir):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    path = Path(root)
                    run_name = path.relative_to(self.logdir)
                    event_files[str(run_name)] = Path(root) / file
                    
        print(f"Found {len(event_files)} event files:")
        for run_name, file_path in event_files.items():
            print(f"  {run_name}: {file_path.name}")
            
        return event_files
    
    def extract_scalar_data(self, event_file: Path):
        """Extract scalar data from a TensorBoard event file."""
        print(f"Processing: {event_file}")
        
        # Create EventAccumulator
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        
        # Get available scalar tags
        scalar_tags = ea.Tags()['scalars']
        print(f"  Available scalar tags: {scalar_tags}")
        
        data = {}
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            timestamps = [event.wall_time for event in scalar_events]
            
            data[tag] = {
                'steps': np.array(steps),
                'values': np.array(values),
                'timestamps': np.array(timestamps)
            }
            
        return data
    
    def plot_training_curves(self, all_data: dict, save_individual: bool = True):
        """Create training curve plots."""
        print("Creating training curve plots...")
        
        # Common metrics to plot
        metrics_to_plot = [
            'rollout/ep_len_mean',
            'rollout/ep_rew_mean', 
            'time/fps',
            'train/entropy_loss',
            'train/explained_variance',
            'train/learning_rate',
            'train/policy_gradient_loss',
            'train/value_loss'
        ]
        
        # Find which metrics are actually available
        available_metrics = set()
        for run_data in all_data.values():
            available_metrics.update(run_data.keys())
        
        # Filter to only available metrics
        available_metrics_to_plot = [m for m in metrics_to_plot if m in available_metrics]
        
        print(f"Available metrics to plot: {available_metrics_to_plot}")
        
        if save_individual:
            # Create individual plots for each metric
            for metric in available_metrics_to_plot:
                self._plot_single_metric(all_data, metric)
        
        # Create combined plots
        self._plot_combined_metrics(all_data, available_metrics_to_plot)
        
    def _plot_single_metric(self, all_data: dict, metric: str):
        """Plot a single metric across all runs."""
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
        
        for i, (run_name, run_data) in enumerate(all_data.items()):
            if metric in run_data:
                data = run_data[metric]
                plt.plot(data['steps'], data['values'], 
                        label=run_name, color=colors[i], linewidth=2, alpha=0.8)
        
        plt.xlabel('Training Steps')
        plt.ylabel(metric.split('/')[-1].replace('_', ' ').title())
        plt.title(f'Training Progress: {metric}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        safe_filename = metric.replace('/', '_').replace(' ', '_')
        output_path = self.output_dir / f'{safe_filename}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")
    
    def _plot_combined_metrics(self, all_data: dict, metrics: list):
        """Create a combined plot with multiple subplots."""
        if not metrics:
            return
            
        # Create subplots grid
        n_metrics = len(metrics)
        cols = 2
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
        if rows == 1:
            axes = [axes] if n_metrics == 1 else axes
        else:
            axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for j, (run_name, run_data) in enumerate(all_data.items()):
                if metric in run_data:
                    data = run_data[metric]
                    ax.plot(data['steps'], data['values'], 
                           label=run_name, color=colors[j], linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(metric.split('/')[-1].replace('_', ' ').title())
            ax.set_title(f'{metric}')
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Add legend to first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Hide extra subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save combined plot
        output_path = self.output_dir / 'combined_training_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved combined plot: {output_path}")
    
    def create_summary_report(self, all_data: dict):
        """Create a summary report with final values."""
        print("Creating summary report...")
        
        summary_data = []
        
        for run_name, run_data in all_data.items():
            row = {'run_name': run_name}
            
            # Extract final values for key metrics
            key_metrics = ['rollout/ep_rew_mean', 'rollout/ep_len_mean', 'time/fps']
            
            for metric in key_metrics:
                if metric in run_data and len(run_data[metric]['values']) > 0:
                    final_value = run_data[metric]['values'][-1]
                    row[metric] = final_value
                else:
                    row[metric] = None
            
            summary_data.append(row)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / 'training_summary.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"  Saved summary: {csv_path}")
        
        return df
    
    def export_all_plots(self):
        """Main function to export all plots."""
        print(f"🎨 TensorBoard Plot Exporter")
        print(f"Log directory: {self.logdir}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        # Find event files
        event_files = self.find_event_files()
        
        if not event_files:
            print("❌ No TensorBoard event files found!")
            return False
        
        # Extract data from all runs
        all_data = {}
        for run_name, event_file in event_files.items():
            try:
                data = self.extract_scalar_data(event_file)
                if data:  # Only add if data was extracted
                    all_data[run_name] = data
            except Exception as e:
                print(f"❌ Error processing {run_name}: {e}")
        
        if not all_data:
            print("❌ No data could be extracted from event files!")
            return False
        
        # Create plots
        self.plot_training_curves(all_data)
        
        # Create summary report
        summary_df = self.create_summary_report(all_data)
        
        print("\n" + "=" * 60)
        print(f"✅ Export completed!")
        print(f"📊 Processed {len(all_data)} runs")
        print(f"📁 Plots saved to: {self.output_dir}")
        
        # Show summary
        print("\nTraining Summary:")
        print(summary_df.to_string(index=False, float_format='%.2f'))
        
        return True

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export TensorBoard plots as images')
    parser.add_argument('--logdir', type=str, 
                       default='tests/miac/comprehensive_results/logs',
                       help='TensorBoard log directory')
    parser.add_argument('--output', type=str, 
                       default=None,
                       help='Output directory for plots (default: logdir/../tensorboard_plots)')
    parser.add_argument('--batch', type=str, 
                       default=None,
                       help='Specific batch timestamp to process')
    
    args = parser.parse_args()
    
    # Construct full log directory path
    logdir = Path(args.logdir)
    if not logdir.is_absolute():
        logdir = Path.cwd() / logdir
    
    # If batch specified, append it to logdir
    if args.batch:
        logdir = logdir / args.batch
    
    if not logdir.exists():
        print(f"❌ Log directory not found: {logdir}")
        return False
    
    # Create exporter and run
    exporter = TensorBoardPlotExporter(str(logdir), args.output)
    return exporter.export_all_plots()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
