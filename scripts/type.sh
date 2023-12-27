#!/usr/bin/env sh

echo "Checking labller"
mypy --check-untyped-defs src/main.py

echo "Checking data-loader"
mypy loader/*.py

echo "Checking tests"
mypy test/*.py
