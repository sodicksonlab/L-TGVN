# L-TGVN
Longitudinal MR Image Reconstruction with deep learning

## Project Repository Overview
This repository contains code and scripts for training and validating Longitudinal Trust Guided Variational Network (L-TGVN). The codebase utilizes PyTorch and Lightning. 

For convenience, a devcontainer that supports CUDA acceleration (CUDA 12.8) was added. Before building the container, you might want to update the `devcontainer.json` to access your data inside the container. You can do so by uncommenting the `mounts` key and adding the data paths. 

The repo follows the standard layout, and the devcontainer installs the TGVN package automatically with `pip install -e .`. If you prefer to install the requirements with pip or conda instead of using a Docker container, install the packages listed in lines 33–40 of the `Dockerfile` and then run `pip install -e .` from the repo root.

### Core Code Files
- **`scripts/main.py`**: Main script for training and evaluating L-TGVN.
- **`src/tgvn/data.py`**: Contains data loading and preprocessing logic.
- **`src/tgvn/loss.py`**: Implements various loss functions for training models.
- **`src/math_utils.py`**: Implements core mathematical operations.
- **`src/tgvn/models.py`**: Defines the L-TGVN architecture used in the project.
- **`src/pl_data_module.py`**: Lightning data wrapper.
