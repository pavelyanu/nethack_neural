#!/bin/bash

# Default download directory
DOWNLOAD_DIR="./dataset"

# Verbosity level
VERBOSE=0

# Dry run mode
DRY_RUN=0

while getopts "d:v:n:" opt; do
  case ${opt} in
    -d | --directory )
      DOWNLOAD_DIR=$OPTARG
      ;;
    -v | --verbose )
      VERBOSE=$OPTARG
      ;;
    -n | --dry_run )
      DRY_RUN=$OPTARG
      ;;
    \? )
      echo "Usage: cmd [-d download_directory] [-v verbosity] [-n dry_run]"
      ;;
  esac
done
shift $((OPTIND -1))

# Define your download links
declare -A LINKS=( 
  # ["nld-aa.zip"]=""
  # ["nld-nao_part1.zip"]=""
  # ["nld-nao_part2.zip"]=""
  # ["nld-nao_part3.zip"]=""
  # ["nld-nao_part4.zip"]=""
  # ["nld-nao_xlogfiles.zip"]=""
)

# Download and unzip
for file in "${!LINKS[@]}"; do
  link=${LINKS[$file]}
  if [ $VERBOSE -eq 1 ] || [ $DRY_RUN -eq 1 ]; then
    echo "Would download $file from $link"
    echo "Would unzip $file"
  fi
  if [ $DRY_RUN -eq 0 ]; then
    wget -P $DOWNLOAD_DIR $link
    if [[ $file == "nld-aa.zip" ]]; then
      unzip $DOWNLOAD_DIR/$file -d $DOWNLOAD_DIR/nld-aa
    else
      unzip $DOWNLOAD_DIR/$file -d $DOWNLOAD_DIR/nld-nao
    fi
  fi
done
