import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Top-level directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# Source directories
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
PREPROCESSING_SRC_DIR = os.path.join(SRC_DIR, "preprocessing")
UTILS_SRC_DIR = os.path.join(SRC_DIR, "utils")
MODELS_SRC_DIR = os.path.join(SRC_DIR, "models")
DATA_SRC_DIR = os.path.join(SRC_DIR, "data")
TRAINING_SRC_DIR = os.path.join(SRC_DIR, "training")
ANALYSIS_SRC_DIR = os.path.join(SRC_DIR, "analysis")
