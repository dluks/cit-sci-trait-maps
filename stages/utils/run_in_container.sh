#!/bin/bash

# Function to get container command based on type
get_container_cmd() {
    local container_type=$1
    local container_name=$2
    local container_tag=$3
    local container_image=$4

    if [ "$container_type" = "docker" ]; then
        echo "docker run --rm -v \$(pwd):/app -w /app $container_name:$container_tag"
    else
        # Set APPTAINER_BINDPATH environment variable for Apptainer/Singularity
        # This will bind all the paths defined in the config file
        unset APPTAINER_BINDPATH
        export APPTAINER_BINDPATH="\
            src:/app/src,\
            tests:/app/tests,\
            models:/app/models,\
            data:/app/data,\
            reference:/app/reference,\
            results:/app/results,\
            .env:/app/.env,\
            params.yaml:/app/params.yaml"
        echo "apptainer run $container_image"
    fi
}

# Parse container configuration (first 4 arguments)
container_type=${1:-"singularity"}
container_name=${2:-"cit-sci-traits"}
container_tag=${3:-"latest"}
container_image=${4:-"cit-sci-traits.sif"}

# The command to run is everything after the first 4 arguments
COMMAND="${@:5}"

# Get container command
container_cmd=$(get_container_cmd "$container_type" "$container_name" "$container_tag" "$container_image")

# Source the mountpoints
source scripts/set_container_mountpoints.sh

# Run the command in the container
eval "$container_cmd $COMMAND" 