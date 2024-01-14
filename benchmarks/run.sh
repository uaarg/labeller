#!/usr/bin/env bash

set -o errexit

benchmark="$1"

if ! git status 2>&1 >/dev/null; then
  echo "Please run this script from inside the labeller repository"
  exit 1
fi

cd $(git worktree list | awk '{ print $1 }')

if ! test -f "benchmarks/$benchmark.py"; then
  echo "ERROR: Please pass a benchmark which is one of:"
  ls benchmarks/*.py | sed 's/^benchmarks\// - /' | sed 's/\.py$//' | grep -v -E '__init__|detector'
  exit 1
fi

PYTHONPATH=".:third_party/yolov5" python3 "benchmarks/$benchmark.py"
