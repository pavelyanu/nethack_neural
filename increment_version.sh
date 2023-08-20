#!/bin/bash

sed -i -E "s/(version='[0-9]+\.)([0-9]+)'/\1$(($(grep -oE "version='[0-9]+\.[0-9]+'" setup.py | grep -oE '[0-9]+$')+1))'/" setup.py

sed -i -E "s/(version = \"[0-9]+\.)([0-9]+)\"/\1$(($(grep -oE "version = \"[0-9]+\.[0-9]+\"" pyproject.toml | grep -oE '[0-9]+$')+1))\"/" pyproject.toml
