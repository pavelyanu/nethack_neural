#!/bin/bash

docker run \
  --env "PYTHONPATH=/workspaces" \
  --volume "$(pwd):/workspaces" \
  --workdir "/workspaces" \
  -it -u 0 docker.io/pavelyanu/nethack_neural:dev /bin/bash 
