#!/usr/bin/env bash

set -o errexit

example="$1"

if ! git status 2&>1 >/dev/null; then
  echo "Please run this script from inside the labeller repository"
  exit 1
fi

cd $(git worktree list | awk '{ print $1 }')

if ! test -f "examples/$example.py"; then
  echo "ERROR: Please pass an example which is one of:"
  ls examples/*.py | sed 's/^examples\// - /' | sed 's/\.py$//'
  exit 1
fi

PYTHONPATH="." python3 "examples/$example.py"
