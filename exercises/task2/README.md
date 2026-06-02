# Python Project for the SS26 Machine Learning Course @ KIT

This folder contains Homework 2: **Multiclass Classification, Kernel Methods, and Support Vector Machines**.

## Environment & Jupyter Setup

You need `uv` to set up the Python environment. Check out [https://docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation) for installation instructions.

To edit the notebook you can choose between Jupyter Lab and Visual Studio Code.
We recommend Jupyter Lab, as it does not allow you to edit cells that you are not supposed to edit, and simplifies environment handling.

- Jupyter Lab in your browser:
    - Run `uv run jupyter lab` in a terminal window within this folder. This may take some time as it needs to download Python packages.
    - Click the link that is printed. This will open a browser window. If your terminal does not allow you to click the link, manually copy it into the address bar of your browser.
- Visual Studio Code:
    - Make sure that the Jupyter extension is installed in VS Code.
    - Run `uv sync` in a terminal window within this folder. This will create a `.venv` that contains the Python packages.
    - In VS Code, open the Jupyter notebook. When you execute the first cell, VS Code will prompt you to select a Python kernel. Select `Python Environments... > ml-ss26-assignments (.venv/bin/python)`.
    - VS Code allows you to edit all cells in the notebook, even those that you are not supposed to edit, so please be careful.

## Exercise Files

The folder must contain the notebook and all data files:

- `ex_02_multiclass_classification_kernel_methods.ipynb`
- `iris_data.npz`
- `two_moons.npz`
- `two_moons_test.npz`
- `x_train.npy`, `y_train.npy`
- `x_valid.npy`, `y_valid.npy`
- `x_test.npy`, `y_test.npy`

Keep these files in the same folder as the notebook.

## Introduction to Jupyter Notebooks

Jupyter notebooks (file extension `.ipynb`) consist of a sequence of *cells*.
Each cell contains either Markdown text or executable Python code.

If you run a cell, Jupyter executes the code within it in a persistent Python session.
So if you create a variable or function in one cell, it will also be available in all other cells.
If you execute the cell again, the variable or function will be overwritten.

You can execute a Python cell by either pressing the "Play" symbol at the top of the window (Jupyter Lab) or in the top left corner of the cell (VS Code), or by pressing `Shift + Enter` while your cursor is inside the cell.

If you kill the Jupyter kernel, for example by killing the Jupyter Lab process in your terminal or by closing VS Code, all variables will be lost.

## Submission

Upload only the edited Jupyter notebook (`.ipynb`) as a raw file to Ilias.
Do not upload any other files or folders, and do not zip the notebook.

For further details see the instructions in the notebook.

