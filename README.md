# Usage

Make sure to install UV to set up the project. For convinience we used uv to set up the dependencys and the venv for running the code. But we have updated the `requirements.txt` to reflect the packages we have added and you can also use pip to install the deps and seting up the venv

## Install UV

Linux and MacOS
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows
```zsh
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Setup

The deps of this project have been adjusted to allow for installing the correct version of tensorflow based on the OS running. Also due to a bug in `tensorflow-macos` related to gpu not being utilised on M1 Pro chips we have moved to a newer version of tensorflow.

Install deps
```zsh
uv sync
```

This will install all the deps and and set up the venv at the latest avalible version of python 3.10.

Activate venv
```zsh
source .venv/bin/activate
```

Run app
```zsh
uv run src/baselines.py # To run the baseline code
uv run src/hill_climbing.py # To run the HC code
uv run src/benchmark_hill_climbing.py # To run the HC benchmark code
```

`hill_climbing_writeup_images.py` also exits to output the generated images and save them like its done in the provided `baseline.py`

## Notes on the flake.nix file

If using NixOS those files can also be used to set up the envoriment with venv automaticly created. Just run `nix develop` to use them. Added for convinience as a member is using NixOS for testing the cuda runtime.