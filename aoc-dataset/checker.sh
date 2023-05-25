#!/bin/bash

shopt -s globstar

status=0
for pass in **/pass*.py; do
    output=$(python submit.py --file "$pass")
    if [[ "$output" != "OK" ]]; then
        echo "Execution failed for $pass"
        ((status++))
    fi
done

for fail in **/fail*.py; do
    output=$(python submit.py --file "$fail")
    if [[ "$output" == "OK" ]]; then
        echo "Execution failed for $fail"
        ((status++))
    fi
done

if [[ $status -ne 0 ]]; then
    exit 1
fi
