#!/usr/bin/env bash

mypy --check-untyped-defs src/main.py
mypy loader/*.py
