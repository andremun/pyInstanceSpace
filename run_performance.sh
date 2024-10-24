#!/bin/bash

# This is a shell script to conduct performance testing
#Run integration_demo.py 10 times
for i in {1..10}
do
    poetry run python integration_demo.py
done

# Read from collected.csv and calculate the average of the second column
csv_files=("preprocessing_time.csv",
            "prelim_time.csv",
            "pilot.csv", 
            "cloister_time.csv",
            "pythia_time.csv",
            "stifed_time.csv",
            "trace_time.csv",
            "trace_time.csv")

for csv_file in "${csv_files[@]}"
do
    if [[ -f "$csv_file" ]]; then
        echo "Calculating average for $csv_file..."
        avg=$(awk -F',' '{sum += $2; count++} END {if (count > 0) print sum / count}' "$csv_file")
        echo "Average of second column in $csv_file: $avg"
    else
        echo "$csv_file does not exist."
    fi
done