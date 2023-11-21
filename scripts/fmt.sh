#!/usr/bin/env sh

yapf --parallel --style pep8 --in-place pylint $(git ls-files '*.py')
