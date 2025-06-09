#!/usr/bin/env python
"""
Test script for CSV logging functionality
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import the CSV logging function (now from same directory)
from comprehensive_training import log_run_to_csv

def test_csv_logging():
    """Test the CSV logging functionality"""
    print("Testing CSV logging functionality...")
    
    # Test directory
    base_dir = "/tmp/test_comprehensive_results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Test timestamp
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Test successful run
    successful_result = {
        'env_name': 'incoming',
        'level': 0,
        'success': True,
        'model_path': '/test/path/model.zip',
        'training_timesteps': 100000,
        'training_time': 120.5,
        'evaluation_time': 15.2,
        'total_time': 135.7,
        'mean_reward': -5000.0,
        'std_reward': 500.0,
        'success_rate': 0.0,
        'video_recorded': True,
        'video_folder': '/test/videos'
    }
    
    # Test failed run
    failed_result = {
        'env_name': 'circle',
        'level': 1,
        'success': False,
        'error': 'Test error message',
        'error_time': 5.0
    }
    
    # Log both runs
    print("Logging successful run...")
    run_id_1 = log_run_to_csv(base_dir, batch_timestamp, successful_result)
    
    print("Logging failed run...")
    run_id_2 = log_run_to_csv(base_dir, batch_timestamp, failed_result)
    
    # Check if CSV file was created and contains expected data
    csv_path = os.path.join(base_dir, 'master_runs_log.csv')
    
    if os.path.exists(csv_path):
        print(f"\n✅ CSV file created successfully: {csv_path}")
        
        # Read and display the content
        with open(csv_path, 'r') as f:
            content = f.read()
            print("\nCSV Content:")
            print("-" * 80)
            print(content)
            print("-" * 80)
        
        # Count lines (header + 2 data rows = 3 lines)
        lines = content.strip().split('\n')
        if len(lines) == 3:
            print(f"✅ Correct number of lines: {len(lines)} (header + 2 data rows)")
        else:
            print(f"❌ Unexpected number of lines: {len(lines)}")
        
        print(f"✅ Run IDs generated: {run_id_1}, {run_id_2}")
        return True
    else:
        print(f"❌ CSV file was not created: {csv_path}")
        return False

if __name__ == "__main__":
    success = test_csv_logging()
    if success:
        print("\n🎉 CSV logging test passed!")
        sys.exit(0)
    else:
        print("\n💥 CSV logging test failed!")
        sys.exit(1)
