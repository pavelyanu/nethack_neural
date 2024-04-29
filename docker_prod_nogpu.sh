#!/bin/bash

docker run \
    --volume "$(pwd):/app/model_save_site" \
    -it -u 0 docker.io/pavelyanu/nethack_neural:prod
