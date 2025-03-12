#!/bin/bash

# Define the file path
FILE="checkpoints/YOUR_CHECKPOINT"

# Loop indefinitely
while true; do
    # Check if the file exists
    if [ -e "$FILE" ]; then
        echo "Checkpoint exists. Running the command..."
        # Replace the following line with the command you want to run
        # sleep 30
        sbatch run_ap.sh
        sbatch run_md.sh
        sbatch run_sg.sh
        sbatch run_si.sh
        sbatch run_pa.sh
        sbatch run_pt.sh
        
        # Optionally break the loop if you only want to run the command once
        break
    else
        echo "Checkpoint does not exist. Checking again..."
    fi
    
    # Wait for a specified amount of time before checking again
    sleep 5
done
