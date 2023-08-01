podman run \
	--device nvidia.com/gpu=0 \
	--security-opt label=disable \
    --security-opt seccomp=unconfined \
	--env "PYTHONPATH=/workspaces" \
    --volume "$(pwd):/workspaces" --workdir "/workspaces" \
	-it -u 0 docker.io/pavelyanu/yanushonak /bin/bash
