# GAT BRD4 Binding Prediction

This project implements a Graph Attention Network (GAT) to predict the binding affinity of small molecules to the BRD4 protein using the Leash BELKA dataset. It was developed as part of an engineering thesis.

## Project Structure

- `main.py`: Main entry point for the pipeline (filtering, processing, training).
- `run_experiments.py`: Script for running specific experiments like learning curves.
- `src/`: Source code directory.
    - `model.py`: GAT model definition.
    - `dataset.py`: BRD4Dataset class (PyTorch Geometric).
    - `train.py`: Training and evaluation loops.
    - `utils.py`: Utility functions (metrics, plotting, seeding).
    - `config.py`: Configuration parameters.
- `data/`: Directory for raw and processed data.
- `plots/`: Directory for generated plots and logs.

## Installation

1. Clone the repository.
2. Create a virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure PyTorch and PyTorch Geometric are installed correctly for your CUDA version.*

## Usage

### 1. Training the Model

To run the full pipeline (data filtering -> processing -> training):

```bash
python main.py --raw_file data/raw/leash-BELKA/train.csv
```

**Options:**
- `--optimize`: Run hyperparameter optimization with Optuna.
- `--cv <k>`: Run k-fold cross-validation.
- `--processed_dir`: Specify a custom directory for processed data.

### 2. Running Experiments

To run learning curve experiments:

```bash
python run_experiments.py --experiment learning_curve
```

## Reproducibility

A random seed is set globally (default: 42) in `src/config.py` and enforced via `src/utils.py` to ensure reproducible results across runs.

## Logging

Training metrics (Loss, AP, AUC, F1) are saved to `plots/<run_dir>/training_log.csv` for further analysis and plotting.
