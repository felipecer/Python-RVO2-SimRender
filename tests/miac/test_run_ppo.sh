#!/bin/bash
# Quick test version of run_ppo_tests.sh for validation
# Set up project root for correct imports and path resolution
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "Project root: $PROJECT_ROOT"

# Set minimal parameters for testing
TIMESTEPS=100

# Test all environments and levels
ENVIRONMENTS=("incoming" "circle" "perp1" "perp2" "two_paths")
LEVELS=(0 1 2 3)

# Generate a common timestamp for this batch run
BATCH_TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
echo "Using batch timestamp: $BATCH_TIMESTAMP"

# Set up centralized summary directory and file
SUMMARY_DIR="tests/miac/ppo_results"
SUMMARY_FILE="${SUMMARY_DIR}/test_ppo_summary_${BATCH_TIMESTAMP}.csv"
LOG_FILE="${SUMMARY_DIR}/test_ppo_results_${BATCH_TIMESTAMP}.log"

# Create summary directory if it doesn't exist
mkdir -p "$SUMMARY_DIR"

# Initialize log file
echo "PPO Test Results - Started at $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG_FILE"
echo "=========================================================" >> "$LOG_FILE"

# Initialize summary file with headers
echo "Timestamp,Test File,Environment,Level,Duration,Exit Status" > "$SUMMARY_FILE"

# Function to run a PPO test with unified test file and measure execution time
run_ppo_test() {
    local env_name=$1
    local level=$2
    local start_time=$(date +%s)
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting: $env_name level $level with $TIMESTEPS timesteps"
    
    # Create environment-specific timestamped subdirectories within tests/miac
    local env_logs_dir="tests/miac/${env_name}/logs/${BATCH_TIMESTAMP}"
    local env_saves_dir="tests/miac/${env_name}/saves/${BATCH_TIMESTAMP}"
    mkdir -p "$env_logs_dir"
    mkdir -p "$env_saves_dir"
    
    local model_name="ppo_${env_name}_level_${level}"
    local model_save_path="${env_saves_dir}/${model_name}"
    
    # Run from project root with correct PYTHONPATH using unified test file
    (
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        python "tests/miac/ppo_unified.py" \
            --mode train \
            --env_name "$env_name" \
            --level "$level" \
            --total_timesteps $TIMESTEPS \
            --device cpu \
            --log_dir "${env_logs_dir}" \
            --save_path "${model_save_path}" \
            --n_envs 2 \
            --n_steps 8 \
            --progress_bar True
        
        echo $? > "${SUMMARY_DIR}/.exit_status"
    )
    
    local exit_status=$(cat "${SUMMARY_DIR}/.exit_status")
    rm -f "${SUMMARY_DIR}/.exit_status"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))
    
    local time_str=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ $exit_status -eq 0 ]; then
        echo "$timestamp - Completed: $env_name level $level (Duration: $time_str)"
    else
        echo "$timestamp - FAILED: $env_name level $level (Duration: $time_str)"
    fi
    
    # Log to file
    echo "$timestamp | $env_name level $level | $time_str | Exit: $exit_status" >> "$LOG_FILE"
    
    # Log to summary file
    echo "$timestamp,ppo_unified.py,$env_name,$level,$time_str,$exit_status" >> "$SUMMARY_FILE"
}

echo "======================================================================="
echo "Starting quick PPO test at $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================="

# Generate all environment-level combinations
echo "Generating environment-level combinations..."
ALL_COMBINATIONS=()
for env in "${ENVIRONMENTS[@]}"; do
    for level in "${LEVELS[@]}"; do
        # Check if config file exists for this combination
        config_file="$PROJECT_ROOT/simulator/worlds/miac/$env/${env}_level_${level}.yaml"
        if [ -f "$config_file" ]; then
            ALL_COMBINATIONS+=("$env:$level")
        else
            echo "Warning: Config file not found for $env level $level, skipping..."
        fi
    done
done

TOTAL_COMBINATIONS=${#ALL_COMBINATIONS[@]}

if [ $TOTAL_COMBINATIONS -eq 0 ]; then
    echo "No valid environment-level combinations found."
    exit 1
fi

echo "Found $TOTAL_COMBINATIONS total combinations to process"

# Process combinations one at a time
for ((i=0; i<$TOTAL_COMBINATIONS; i++)); do
    IFS=':' read -r env_name level <<< "${ALL_COMBINATIONS[$i]}"
    echo "Testing: $env_name level $level"
    run_ppo_test "$env_name" "$level"
done

echo "Quick PPO test completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Summary results available in $SUMMARY_FILE"
