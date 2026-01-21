import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

# Ensure directories exist
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data Processing
BINDINGDB_FILENAME = 'BindingDB_All.tsv'
THRESHOLD_NM = 1000.0 # 1 uM
TARGET_NAMES = ["BRD4", "Bromodomain-containing protein 4"]

# Model Hyperparameters
GAT_HEADS = 8
GAT_HIDDEN_DIM = 64
GAT_LAYERS = 3
DROPOUT = 0.11
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-4
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 10 # Early stopping

# Focal Loss Support
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25 # Optional default, can be calculated dynamically

# Random Seed
SEED = 42
