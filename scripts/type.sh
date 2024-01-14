#!/usr/bin/env sh

export PYTHONPATH=".:third_party/yolov5"

echo "Checking labller"
mypy --check-untyped-defs src/main.py

echo "Checking data-loader"
mypy loader/*.py

echo "Checking benchmarks"
mypy benchmarks/*.py

echo "Checking tests"
mypy test/*.py
