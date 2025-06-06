#!/bin/bash
# filepath: /home/felipecerda/repos/memoria/Python-RVO2-SimRender/tests/miac/run_all_baseline_orca.sh

# Set up project root for correct imports and path resolution
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "Project root: $PROJECT_ROOT"

# Create baseline results directory for centralized logging
mkdir -p baseline_results

# Create logs and saves directories if they don't exist
mkdir -p circle/logs
mkdir -p circle/saves
mkdir -p incoming/logs
mkdir -p incoming/saves
mkdir -p perp1/logs
mkdir -p perp1/saves
mkdir -p perp2/logs
mkdir -p perp2/saves
mkdir -p two_paths/logs
mkdir -p two_paths/saves

# Number of runs for each environment
NUM_RUNS=1000

# Create a temporary directory to track progress
TEMP_DIR=$(mktemp -d)
TOTAL_TESTS=20
COMPLETED=0

# Function to update progress
update_progress() {
    COMPLETED=$((COMPLETED + 1))
    PERCENT=$((COMPLETED * 100 / TOTAL_TESTS))
    BAR_LENGTH=50
    FILLED_LENGTH=$((BAR_LENGTH * COMPLETED / TOTAL_TESTS))
    
    # Create the progress bar
    BAR=""
    for ((i=0; i<FILLED_LENGTH; i++)); do
        BAR="${BAR}="
    done
    for ((i=FILLED_LENGTH; i<BAR_LENGTH; i++)); do
        BAR="${BAR} "
    done
    
    # Print the progress
    printf "\rProgress: [%s] %d%% (%d/%d)" "$BAR" "$PERCENT" "$COMPLETED" "$TOTAL_TESTS"
}

# Function to run a script in the background and notify when done
run_script() {
    local script_path=$1
    local script_name=$(basename "$script_path")
    local script_dir=$(dirname "$script_path")
    local flag_file="${TEMP_DIR}/${script_name}.done"
    
    # Create log file in the logs directory instead of next to the script
    local log_dir="${script_dir}/logs"
    local log_file="${log_dir}/${script_name%.py}.log"
    
    echo "Starting $script_path (logging to $log_file)"
    
    # Run the script and create a flag file when done
    (
        # Run from project root with correct PYTHONPATH
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
        
        # The script path needs to be adjusted to be run from project root
        python "tests/miac/$script_path" > "tests/miac/$log_file" 2>&1
        local status=$?
        echo "$status" > "$flag_file"
        
        # Display completion message
        echo -e "\n✅ Completed: $script_path (exit code: $status)"
        
        # Update progress
        update_progress
    ) &
    
    echo "Started $script_name with PID $!"
}

echo "Starting all baseline ORCA scripts in parallel processes..."
echo "Total tests to run: $TOTAL_TESTS"

# Display initial progress bar
update_progress

# Circle environment - all levels
run_script circle/baseline_orca_circle_level_0.py
run_script circle/baseline_orca_circle_level_1.py
run_script circle/baseline_orca_circle_level_2.py
run_script circle/baseline_orca_circle_level_3.py

# Incoming environment - all levels
run_script incoming/baseline_orca_incoming_level_0.py
run_script incoming/baseline_orca_incoming_level_1.py
run_script incoming/baseline_orca_incoming_level_2.py
run_script incoming/baseline_orca_incoming_level_3.py

# Perp1 environment - all levels
run_script perp1/baseline_orca_perp1_level_0.py
run_script perp1/baseline_orca_perp1_level_1.py
run_script perp1/baseline_orca_perp1_level_2.py
run_script perp1/baseline_orca_perp1_level_3.py

# Perp2 environment - all levels
run_script perp2/baseline_orca_perp2_level_0.py
run_script perp2/baseline_orca_perp2_level_1.py
run_script perp2/baseline_orca_perp2_level_2.py
run_script perp2/baseline_orca_perp2_level_3.py

# Two Paths environment - all levels
run_script two_paths/baseline_orca_two_paths_level_0.py
run_script two_paths/baseline_orca_two_paths_level_1.py
run_script two_paths/baseline_orca_two_paths_level_2.py
run_script two_paths/baseline_orca_two_paths_level_3.py

echo -e "\nAll scripts have been started in the background."
echo "You can check the status using 'ps aux | grep baseline_orca'"
echo "Logs are being written to the logs directories in the respective test folders"

# Wait for all background processes to complete
echo "Waiting for all processes to complete..."
wait

# Final progress bar update
update_progress
echo -e "\n\nAll baseline ORCA evaluations completed!"
echo "Results are saved in the respective 'saves' folders and logs are available in the 'logs' folders"

# Clean up temporary directory
rm -rf "$TEMP_DIR"
exit 0