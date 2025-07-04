#!/usr/bin/env python
"""
Simple TensorBoard data extractor and plotter.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def find_event_files(logdir):
    """Find all TensorBoard event files."""
    event_files = {}
    
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                path = Path(root)
                run_name = path.name  # Just use the directory name
                event_files[run_name] = Path(root) / file
                
    return event_files

def extract_and_plot_simple():
    """Simple extraction and plotting."""
    logdir = Path("tests/miac/comprehensive_results/logs/20250609_144558")
    output_dir = Path("tests/miac/comprehensive_results/tensorboard_plots")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Looking for event files in: {logdir}")
    
    # Find event files
    event_files = find_event_files(logdir)
    print(f"Found {len(event_files)} runs:")
    for run_name, file_path in event_files.items():
        print(f"  {run_name}: {file_path}")
    
    if not event_files:
        print("No event files found!")
        return
    
    # Try to import TensorBoard
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard not available for data extraction")
        return
    
    # Process each run
    all_data = {}
    
    for run_name, event_file in event_files.items():
        print(f"Processing {run_name}...")
        
        try:
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            
            scalar_tags = ea.Tags()['scalars']
            print(f"  Available metrics: {scalar_tags}")
            
            run_data = {}
            for tag in scalar_tags:
                scalar_events = ea.Scalars(tag)
                steps = [event.step for event in scalar_events]
                values = [event.value for event in scalar_events]
                run_data[tag] = {'steps': steps, 'values': values}
            
            all_data[run_name] = run_data
            
        except Exception as e:
            print(f"  Error processing {run_name}: {e}")
    
    # Create simple plots
    if all_data:
        print("Creating plots...")
        
        # Find common metrics
        all_metrics = set()
        for run_data in all_data.values():
            all_metrics.update(run_data.keys())
        
        print(f"All available metrics: {list(all_metrics)}")
        
        # Plot each metric
        for metric in all_metrics:
            plt.figure(figsize=(10, 6))
            
            for run_name, run_data in all_data.items():
                if metric in run_data:
                    data = run_data[metric]
                    plt.plot(data['steps'], data['values'], label=run_name, linewidth=2)
            
            plt.xlabel('Steps')
            plt.ylabel(metric.split('/')[-1])
            plt.title(f'Training Progress: {metric}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            safe_name = metric.replace('/', '_').replace(' ', '_')
            output_path = output_dir / f'{safe_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {output_path}")
        
        print(f"\nAll plots saved to: {output_dir}")
    else:
        print("No data extracted!")

if __name__ == "__main__":
    extract_and_plot_simple()
