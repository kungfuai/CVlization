#!/usr/bin/env bash
#
# Set $DOCKER_RUNTIME environment variable based on available container runtimes.

if [ "$(docker info | grep Runtimes | grep -o nvidia)" == "nvidia" ]; then
  export DOCKER_RUNTIME="nvidia"
else
  export DOCKER_RUNTIME="runc"
fi
