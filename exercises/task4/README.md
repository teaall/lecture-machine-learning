# Python Project for the SS26 Machine Learning Course @ KIT

This folder contains Homework 4: **Neural Networks**.

## Environment & Jupyter Setup

You need `uv` to set up the Python environment. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation).

We recommend Jupyter Lab because it simplifies environment handling and protects cells that should not be edited.

- Jupyter Lab:
    - Run `uv run jupyter lab` in a terminal within this folder.
    - Open the link printed in the terminal.
- Visual Studio Code:
    - Install the Jupyter extension.
    - Run `uv sync` in a terminal within this folder.
    - Open the notebook and select the Python environment created in `.venv`.
    - VS Code allows every cell to be edited, so only change designated solution cells.

## Dependencies

The required packages are declared in `pyproject.toml`:

- `numpy`
- `matplotlib`
- `torch` (CPU version)
- `jupyter` and `jupyterlab`
- `ipykernel`

## Exercise Files

Keep these files in the same folder:

- `ex_04_neural_networks.ipynb`
- `two_moons.npz`
- `two_moons_test.npz`
- `mnist.npz`

## Introduction to Jupyter Notebooks

Jupyter notebooks consist of Markdown and executable Python cells. Cells share a persistent Python session, so variables and functions created in one cell are available in later cells.

Run a cell with the play button or `Shift + Enter`. Restarting the kernel clears all variables. Before submission, restart the kernel and run the complete notebook from top to bottom.

## Submission

Upload only `ex_04_neural_networks.ipynb` to Ilias as a raw file. Do not upload the data files, environment files, folders, or a zip archive.

See the instructions at the beginning of the notebook for further details.
