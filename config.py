import os

# --- PATHS ---
RAW_DATA_PATH = "./BETH/dataset"
OUTPUT_DIR = "BETH/outputs"
DATA_DIR = "BETH/data"

# Create directories automatically if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- MODEL SETTINGS ---
CONTAMINATION_RATE = 0.05

# Numeric features for the model
NUMERIC_FEATURES = ['processId', 'threadId', 'parentProcessId', 'userId', 'mountNamespace', 'eventId', 'argsNum', 'returnValue']
# Categorical features we will convert to numbers
CATEGORICAL_FEATURES = ['processName'] 
# Combined final features
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- OLLAMA AI SETTINGS ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "honeypot-analyst"