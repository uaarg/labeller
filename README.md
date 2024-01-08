# Labeller

This is a tool used to view and annotate large image datasets for use in image
algorithm training/evaluation.

## Setup

With python 3.10+ installed (and pip/venv) run the following on MacOS/Linux:

```sh
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

or on Windows:

```sh
python -m venv venv
./venv/Scripts/activate.ps1
pip install -r requirements.txt
```

## Running

Then the application can be started with the following on MacOS/Linux:

```sh
python3 src/main.py
```

or on Windows:

```sh
python src/main.py
```

## Development

When developing `labeller`, the following scripts may be useful:

```sh
./scripts/fmt.sh  # Run the formatter
./scripts/lint.sh  # Run the linter
./scripts/type.sh  # Run the type checker
./scripts/test.sh  # Run the test suite (WIP)
```

## Benchmarks

Benchmarks for specific algorithms can be run with:

```sh
benchmarks/run.sh <name>
```

Leave `name` blank to see a list of benchmarks you can run.
