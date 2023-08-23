#!/bin/bash

# Exit on any error
set -e

echo "Updating system..."
sudo apt update

echo "Installing required packages..."
sudo apt install -y build-essential autoconf libtool pkg-config python3-dev \
    python3-pip python3-numpy git flex bison libbz2-dev

echo "Adding Kitware (for CMake) repository..."
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update

echo "Installing CMake and associated keys..."
sudo apt-get --allow-unauthenticated install -y cmake kitware-archive-keyring

echo "Making sure /usr/bin/cmake is the default..."
# Using `command -v` to get the path of the command
CMAKE_PATH=$(command -v cmake)

# If the default path isn't /usr/bin/cmake, then remove it
if [ "$CMAKE_PATH" != "/usr/bin/cmake" ]; then
    sudo rm "$CMAKE_PATH"
fi

echo "Installed CMake version:"
cmake --version

echo "All dependencies installed successfully!"
