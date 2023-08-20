#!/bin/bash

# For setup.py
current_version_setup=$(grep -oE "version='[0-9]+\.[0-9]+'" setup.py | grep -oE '[0-9]+\.[0-9]+')
major_version_setup=$(echo $current_version_setup | cut -d'.' -f1)
minor_version_setup=$(echo $current_version_setup | cut -d'.' -f2)
new_minor_version_setup=$((minor_version_setup+1))

# Replace old version with new version in setup.py
sed -i "s/version='$major_version_setup\.$minor_version_setup'/version='$major_version_setup\.$new_minor_version_setup'/" setup.py

# For pyproject.toml
current_version_toml=$(grep -oE "version = \"[0-9]+\.[0-9]+\"" pyproject.toml | grep -oE '[0-9]+\.[0-9]+')
major_version_toml=$(echo $current_version_toml | cut -d'.' -f1)
minor_version_toml=$(echo $current_version_toml | cut -d'.' -f2)
new_minor_version_toml=$((minor_version_toml+1))

# Replace old version with new version in pyproject.toml
sed -i "s/version = \"$major_version_toml\.$minor_version_toml\"/version = \"$major_version_toml\.$new_minor_version_toml\"/" pyproject.toml