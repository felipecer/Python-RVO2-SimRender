#!/bin/bash
# filepath: /home/felipecerda/repos/memoria/Python-RVO2-SimRender/tests/miac/run_ppo_tests.sh

# Set up project root for correct imports and path resolution
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "Project root: $PROJECT_ROOT"

# Set the number of timesteps
TIMESTEPS=1000000

# Generate a common timestamp for this batch run
BATCH_TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
echo "Using batch timestamp: $BATCH_TIMESTAMP"

# Set up centralized summary directory and file
SUMMARY_DIR="./ppo_results"
SUMMARY_FILE="${SUMMARY_DIR}/ppo_summary_${BATCH_TIMESTAMP}.csv"
LOG_FILE="${SUMMARY_DIR}/ppo_test_results_${BATCH_TIMESTAMP}.log"

# Create summary directory if it doesn't exist
mkdir -p "$SUMMARY_DIR"

# Initialize log file
echo "PPO Test Results - Started at $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG_FILE"
echo "=========================================================" >> "$LOG_FILE"

# Initialize summary file with headers
echo "Timestamp,Test File,Environment,Level,Duration,Exit Status" > "$SUMMARY_FILE"

# Function to run a PPO test file and measure execution time
run_ppo_test() {
    local test_file=$1
    local start_time=$(date +%s)
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting: $test_file with $TIMESTEPS timesteps"
    
    # Extract environment directory, name and level from filename
    local env_dir=$(dirname "$test_file")  # This gives us 'circle', 'incoming', etc.
    local env_name=$(echo "$test_file" | grep -o -E '[a-z_]+_level_[0-9]+' | cut -d '_' -f 1)
    local level=$(echo "$test_file" | grep -o -E 'level_[0-9]+' | cut -d '_' -f 2)
    
    # Create environment-specific timestamped subdirectories
    local env_logs_dir="${env_dir}/logs/${BATCH_TIMESTAMP}"
    local env_saves_dir="${env_dir}/saves/${BATCH_TIMESTAMP}"
    mkdir -p "$env_logs_dir"
    mkdir -p "$env_saves_dir"
    
    local model_name=$(basename "$test_file" .py)
    local model_save_path="${env_saves_dir}/${model_name}"
    
    # Run from project root with correct PYTHONPATH
    (
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        python "tests/miac/$test_file" \
            --mode train \
            --total_timesteps $TIMESTEPS \
            --device cpu \
            --log_dir "tests/miac/${env_logs_dir}" \
            --save_path "tests/miac/${model_save_path}" \
            --env_name "$env_name" \
            --level "$level" \
            --n_envs 16 \
            --n_steps 64 \
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
        echo "$timestamp - Completed: $test_file (Duration: $time_str)"
    else
        echo "$timestamp - FAILED: $test_file (Duration: $time_str)"
    fi
    
    # Log to file
    echo "$timestamp | $test_file | $time_str | Exit: $exit_status" >> "$LOG_FILE"
    
    # Log to summary file with locking to prevent race conditions
    (
        flock -x 200
        echo "$timestamp,$test_file,$env_name,$level,$time_str,$exit_status" >> "$SUMMARY_FILE"
    ) 200>"${SUMMARY_DIR}/.lock"
}

# Main execution
echo "======================================================================="
echo "Starting PPO test batch run at $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================="
echo "Batch timestamp: $BATCH_TIMESTAMP"
echo "Summary will be logged in real-time to $SUMMARY_FILE"
echo "======================================================================="

# Find all PPO test files excluding optuna related ones
echo "Scanning for PPO test files..."
# ALL_PPO_FILES=$(find . -name "ppo_*_level_0.py" | grep -v "optuna" | grep -v "circle" | sed 's|^./||')
ALL_PPO_FILES=$(find . -name "ppo_incoming_level_*.py" | grep -v "optuna" | sed 's|^./||')
TOTAL_FILES=$(echo "$ALL_PPO_FILES" | wc -l)

if [ $TOTAL_FILES -eq 0 ]; then
    echo "No PPO test files found. Please check your directory structure."
    echo "No PPO test files found. Exiting." >> "$LOG_FILE"
    exit 1
fi

# Shuffle the files
echo "Shuffling $TOTAL_FILES test files..."
SHUFFLED_FILES=($(echo "$ALL_PPO_FILES" | sort -R))

echo "Found and shuffled $TOTAL_FILES total PPO test files to process"
echo "Found $TOTAL_FILES total PPO test files to process" >> "$LOG_FILE"
echo "=========================================================" >> "$LOG_FILE"
echo "Format: TIMESTAMP | FILE | DURATION (HH:MM:SS) | Exit Status" >> "$LOG_FILE"
echo "=========================================================" >> "$LOG_FILE"

# List all files that will be processed
echo "Files to process in this order:"
for ((i=0; i<$TOTAL_FILES; i++)); do
    echo "  $(($i+1)). ${SHUFFLED_FILES[$i]}"
done
echo "======================================================================="

# Process files one at a time
TOTAL_BATCHES=$TOTAL_FILES

for ((i=0; i<$TOTAL_FILES; i++)); do
    BATCH_NUM=$((i+1))
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting batch $BATCH_NUM of $TOTAL_BATCHES"
    echo "----- Batch $BATCH_NUM of $TOTAL_BATCHES -----" >> "$LOG_FILE"
    
    echo "Launching process for: ${SHUFFLED_FILES[$i]}"
    run_ppo_test "${SHUFFLED_FILES[$i]}"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed batch $BATCH_NUM of $TOTAL_BATCHES"
    echo "-----------------------------------------------------"
done

echo "All PPO tests completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "All PPO tests completed at $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "Summary results available in $SUMMARY_FILE"