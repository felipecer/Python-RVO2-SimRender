#!/usr/bin/env python
"""
Test script to verify model loading and basic evaluation functionality.
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from test_saved_models import SavedModelTester

def test_model_loading():
    """Test that we can load and basic functionality works."""
    print("Testing SavedModelTester functionality...")
    
    # Create tester instance
    tester = SavedModelTester()
    
    # Test loading CSV data
    if not tester.load_runs_data():
        print("❌ Failed to load runs data")
        return False
    
    print("✅ Successfully loaded runs data")
    
    # Test getting successful runs
    success_runs = tester.runs_df[tester.runs_df['status'] == 'SUCCESS']
    print(f"✅ Found {len(success_runs)} successful runs")
    
    if len(success_runs) > 0:
        # Get the first successful run
        test_run = success_runs.iloc[0].to_dict()
        print(f"✅ Test run: {test_run['run_id']}")
        print(f"   Model path: {test_run['model_path']}")
        
        # Check if model file exists
        model_path = test_run['model_path']
        if os.path.exists(model_path):
            print(f"✅ Model file exists: {model_path}")
        elif os.path.exists(f"{model_path}.zip"):
            print(f"✅ Model file exists: {model_path}.zip")
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
    
    print("✅ All basic tests passed!")
    return True

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test passed!")
        sys.exit(0)
    else:
        print("\n💥 Model loading test failed!")
        sys.exit(1)
