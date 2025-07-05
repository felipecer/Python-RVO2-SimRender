#!/usr/bin/env python3
"""
Test script for the enhanced training system with video recording.
This will run a quick test to verify that the system works correctly.
"""

import sys
import os
import time

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.helpers.enhanced_trainer_testers import EnhancedPPOTrainerTester
from rl_environments.single_agent.miac.v2.incoming import RVOMiacIncoming2

def test_enhanced_system():
    """Test the enhanced training system with a quick training run."""
    
    # Use a valid MIAC environment for testing
    config_file = '/home/felipecerda/memoria/Python-RVO2-SimRender/simulator/worlds/miac/incoming/incoming_level_0.yaml'
    
    print("=== Testing Enhanced Training System ===")
    print(f"Environment: incoming level 0")
    print(f"Config file: {config_file}")
    
    # Create trainer with proper parameters
    trainer = EnhancedPPOTrainerTester(
        env_class=RVOMiacIncoming2,
        config_file=config_file,
        log_dir='/home/felipecerda/memoria/Python-RVO2-SimRender/tests/miac/test_logs',
        save_path='/home/felipecerda/memoria/Python-RVO2-SimRender/tests/miac/test_model',
        render_mode='rgb_array',
        seed=42,
        unique_id='test_enhanced',
        env_name='incoming',
        level=0
    )
    
    # Quick training run (very small for testing)
    print("\n--- Starting Training ---")
    model = trainer.train(
        n_envs=4,  # Small number for quick test
        total_timesteps=1000,  # Very small for quick test
        n_steps=64,
        device='cpu',
        progress_bar=True
    )
    
    # Test evaluation with video recording
    print("\n--- Starting Evaluation ---")
    evaluation_results = trainer.evaluate_model(
        n_eval_episodes=3,  # Small number for quick test
        record_best_video=True
    )
    
    print("\n=== Test Results ===")
    print(f"Mean reward: {evaluation_results['statistics']['mean_reward']:.2f}")
    print(f"Success rate: {evaluation_results['statistics']['success_rate']:.2%}")
    print(f"Video recorded: {evaluation_results.get('video_recorded', False)}")
    if evaluation_results.get('video_folder'):
        print(f"Video folder: {evaluation_results['video_folder']}")
    
    # Test loading model with metadata
    print("\n--- Testing Model Loading ---")
    loaded_model, metadata = trainer.load_model_with_metadata()
    print(f"Model loaded successfully: {loaded_model is not None}")
    print(f"Metadata loaded: {len(metadata)} fields")
    print(f"Training timestamp: {metadata.get('training_info', {}).get('timestamp', 'N/A')}")
    
    print("\n=== Test Complete ===")
    print("Enhanced training system is working correctly!")
    
    return True

if __name__ == "__main__":
    try:
        success = test_enhanced_system()
        if success:
            print("✅ Test passed!")
            sys.exit(0)
        else:
            print("❌ Test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
