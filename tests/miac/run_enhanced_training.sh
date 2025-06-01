#!/bin/bash

# Enhanced PPO testing script with metadata storage and evaluation
# This script runs enhanced PPO training with comprehensive metadata storage

set -e

# Get the directory of the current script (tests/miac)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the project root (two levels up from script directory)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
TIMESTEPS=${1:-1000}  # Short timesteps for quick testing, can be overridden
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=== Enhanced PPO Training with Metadata Storage ==="
echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"
echo "Batch timestamp: $BATCH_TIMESTAMP"
echo "Training timesteps: $TIMESTEPS"
echo ""

# Create summary directory
SUMMARY_DIR="${SCRIPT_DIR}/enhanced_runs/${BATCH_TIMESTAMP}"
mkdir -p "$SUMMARY_DIR"

# Summary file
SUMMARY_FILE="${SUMMARY_DIR}/training_summary.txt"
echo "Enhanced PPO Training Summary - $(date)" > "$SUMMARY_FILE"
echo "Timesteps per run: $TIMESTEPS" >> "$SUMMARY_FILE"
echo "================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Counter for progress tracking
TOTAL_RUNS=0
SUCCESSFUL_RUNS=0

# Test a subset of environment-level combinations for demonstration
declare -a test_combinations=(
    "perp1:0"
    "perp1:1"
    "perp2:0"
    "perp2:1"
    "circle:0"
)

echo "Testing ${#test_combinations[@]} environment-level combinations:"
for combo in "${test_combinations[@]}"; do
    echo "  - $combo"
done
echo ""

run_enhanced_training() {
    local env_name=$1
    local level=$2
    local combo="${env_name}:${level}"
    
    echo "=== Running enhanced training: $combo ==="
    
    # Create timestamped directories
    local env_dir="${SCRIPT_DIR}/${env_name}"
    local env_logs_dir="${env_dir}/logs/${BATCH_TIMESTAMP}"
    local env_saves_dir="${env_dir}/saves/${BATCH_TIMESTAMP}"
    mkdir -p "$env_logs_dir"
    mkdir -p "$env_saves_dir"
    
    local model_name="ppo_${env_name}_level_${level}"
    local model_save_path="${env_saves_dir}/${model_name}"
    local unique_id="${BATCH_TIMESTAMP}_${env_name}_${level}"
    
    echo "Training model: $model_name"
    echo "Save path: $model_save_path"
    echo "Unique ID: $unique_id"
    
    # Run from project root with correct PYTHONPATH
    (
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        
        # Training
        echo "Starting training..."
        python "tests/miac/ppo_unified.py" \
            --mode train \
            --env_name "$env_name" \
            --level "$level" \
            --total_timesteps $TIMESTEPS \
            --device cpu \
            --log_dir "tests/miac/${env_logs_dir}" \
            --save_path "tests/miac/${model_save_path}" \
            --unique_id "$unique_id" \
            --n_envs 4 \
            --n_steps 32 \
            --n_eval_episodes 5 \
            --record_video True \
            --progress_bar True
        
        echo $? > "${SUMMARY_DIR}/.exit_status_${env_name}_${level}"
    )
    
    # Check exit status
    local exit_status=$(cat "${SUMMARY_DIR}/.exit_status_${env_name}_${level}" 2>/dev/null || echo "1")
    
    if [ "$exit_status" -eq 0 ]; then
        echo "✓ Enhanced training completed successfully: $combo"
        echo "✓ $combo - Training successful" >> "$SUMMARY_FILE"
        
        # Log metadata file locations
        if [ -f "tests/miac/${model_save_path}_metadata.json" ]; then
            echo "  Metadata: ${model_save_path}_metadata.json" >> "$SUMMARY_FILE"
        fi
        if [ -f "tests/miac/${model_save_path}_evaluation.json" ]; then
            echo "  Evaluation: ${model_save_path}_evaluation.json" >> "$SUMMARY_FILE"
        fi
        
        SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
    else
        echo "✗ Enhanced training failed: $combo (exit code: $exit_status)"
        echo "✗ $combo - Training failed (exit code: $exit_status)" >> "$SUMMARY_FILE"
    fi
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    echo ""
}

# Run training for each combination
for combo in "${test_combinations[@]}"; do
    IFS=':' read -r env_name level <<< "$combo"
    run_enhanced_training "$env_name" "$level"
done

# Final summary
echo "=== Enhanced Training Summary ===" | tee -a "$SUMMARY_FILE"
echo "Total runs: $TOTAL_RUNS" | tee -a "$SUMMARY_FILE"
echo "Successful runs: $SUCCESSFUL_RUNS" | tee -a "$SUMMARY_FILE"
echo "Failed runs: $((TOTAL_RUNS - SUCCESSFUL_RUNS))" | tee -a "$SUMMARY_FILE"
echo "Success rate: $(( (SUCCESSFUL_RUNS * 100) / TOTAL_RUNS ))%" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Results saved to: $SUMMARY_DIR" | tee -a "$SUMMARY_FILE"

# Show directory structure
echo "" | tee -a "$SUMMARY_FILE"
echo "=== Generated Files Structure ===" | tee -a "$SUMMARY_FILE"
cd "$SCRIPT_DIR"
find . -name "*${BATCH_TIMESTAMP}*" -type f | head -20 | tee -a "$SUMMARY_FILE"

echo ""
echo "Enhanced training pipeline completed!"
echo "Summary available at: $SUMMARY_FILE"
