#!/bin/bash
# filepath: /home/felipecerda/repos/memoria/Python-RVO2-SimRender/run_ppo_tests.sh

# Set the number of timesteps
TIMESTEPS=5000000
LOG_FILE="ppo_test_results.log"

# Initialize log file
echo "PPO Test Results - Started at $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG_FILE"
echo "=========================================================" >> "$LOG_FILE"

# Function to run a PPO test file and measure execution time
run_ppo_test() {
    local test_file=$1
    local start_time=$(date +%s)
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting: $test_file with $TIMESTEPS timesteps"
    
    python "$test_file" --mode train --total_timesteps $TIMESTEPS --device cuda
    
    local exit_status=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))
    
    local time_str=$(printf "%02d:%02d:%02d" $hours $minutes $seconds)
    
    if [ $exit_status -eq 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed: $test_file (Duration: $time_str)"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - FAILED: $test_file (Duration: $time_str)"
    fi
    
    # Log to file
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $test_file | $time_str | Exit: $exit_status" >> "$LOG_FILE"
}

# Main execution
echo "======================================================================="
echo "Starting PPO test execution at $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================="

# Find all PPO test files excluding optuna related ones
echo "Scanning for PPO test files..."
ALL_PPO_FILES=$(find ./tests/miac -name "ppo_*_level_*.py" | grep -v "optuna")
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

# Process files in batches of 5
TOTAL_BATCHES=$(( (TOTAL_FILES+4)/5 ))

for ((i=0; i<$TOTAL_FILES; i+=5)); do
    BATCH_NUM=$((i/5+1))
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting batch $BATCH_NUM of $TOTAL_BATCHES"
    echo "----- Batch $BATCH_NUM of $TOTAL_BATCHES -----" >> "$LOG_FILE"
    
    # Run up to 5 tests in parallel
    for ((j=0; j<5 && i+j<$TOTAL_FILES; j++)); do
        echo "Launching process for: ${SHUFFLED_FILES[$i+j]}"
        run_ppo_test "${SHUFFLED_FILES[$i+j]}" &
    done
    
    # Wait for all processes in this batch to complete
    wait
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed batch $BATCH_NUM of $TOTAL_BATCHES"
    echo "-----------------------------------------------------"
done

echo "All PPO tests completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "All PPO tests completed at $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"