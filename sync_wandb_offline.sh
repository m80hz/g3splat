#!/bin/bash
duration=$((6 * 3600))  # Run for 6 hours (change as needed)
start_time=$(date +%s)

while true; do
    echo $(date)
    wandb sync /path/to/wandb/offline/logs
    sleep 900  # Sleep for 900 seconds (15 minutes)

    # Stop after the specified duration
    current_time=$(date +%s)
    if (( current_time - start_time >= duration )); then
        break
    fi
done


# run this script in the background: 
# nohup ./sync_wandb_offline.sh > sync_wandb_offline.output.log 2>&1 &