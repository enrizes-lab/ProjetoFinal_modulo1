import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Caminhos de Dados

DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'spotify-tracks-dataset.csv') 
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_limpo.csv')
DATA_FINAL_PATH = os.path.join(BASE_DIR, 'data', 'final', 'dataset_final.csv')

# Caminhos de Modelos e Métricas
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'v1')
MODEL_FILE = os.path.join(MODELS_DIR, 'modelo_regressao_v1.pkl')
METRICS_FILE = os.path.join(MODELS_DIR, 'metricas_v1.json')

# Caminhos de Gráficos
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')