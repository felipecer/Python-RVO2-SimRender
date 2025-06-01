#!/bin/bash
# Test script to show what combinations would be generated
# filepath: /Users/felipecerda/Universidad/Memoria/Python-RVO2-SimRender/tests/miac/test_combinations.sh

# Set up project root for correct imports and path resolution
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
echo "Project root: $PROJECT_ROOT"

# Define available environments and levels
ENVIRONMENTS=("incoming" "circle" "perp1" "perp2" "two_paths")
LEVELS=(0 1 2 3)

echo "======================================================================="
echo "Testing PPO unified combinations generation"
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
            echo "✓ Found: $env level $level"
        else
            echo "✗ Missing: $config_file"
        fi
    done
done

TOTAL_COMBINATIONS=${#ALL_COMBINATIONS[@]}
echo ""
echo "Total valid combinations found: $TOTAL_COMBINATIONS"
echo ""

if [ $TOTAL_COMBINATIONS -eq 0 ]; then
    echo "No valid environment-level combinations found."
    exit 1
fi

# Show what would be shuffled and run
echo "All valid combinations:"
for ((i=0; i<$TOTAL_COMBINATIONS; i++)); do
    IFS=':' read -r env level <<< "${ALL_COMBINATIONS[$i]}"
    echo "  $(($i+1)). $env level $level"
done

echo ""
echo "The script would run: python tests/miac/ppo_unified.py --mode train --env_name <ENV> --level <LEVEL> --total_timesteps 1000000 --device cpu --n_envs 8 --n_steps 32 --progress_bar True"
