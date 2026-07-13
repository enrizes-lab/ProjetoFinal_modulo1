import pandas as pd

def clean_data(df):
    # --- CÓDIGO ORIGINAL (Mantido) ---
    df_clean = df.copy().sort_values('popularity', ascending=False)
    df_clean = df_clean.drop_duplicates(subset=['track_name', 'artists'], keep='first')
    df_clean = df_clean.dropna(subset=['track_name', 'artists'])
    
    limite = df_clean['duration_ms'].quantile(0.99)
    df_clean = df_clean[df_clean['duration_ms'] <= limite]
    
    # --- 🆕 NOVAS ADIÇÕES ---
    # 1. Remove as colunas de índice inúteis
    colunas_para_remover = ['Unnamed: 0', 'Unnamed: 0.1']
    df_clean = df_clean.drop(columns=colunas_para_remover, errors='ignore')
    
    # 2. Mantém apenas as músicas com popularidade maior que zero
    df_clean = df_clean[df_clean['popularity'] > 0]
    
    return df_clean

def feature_engineering(df):
    df_feat = df.copy()
    df_feat['duration_min'] = df_feat['duration_ms'] / 60000
    df_feat['dance_energy_ratio'] = df_feat['danceability'] * df_feat['energy']
    return df_feat

def prepare_for_model(df):
    cols_to_drop = ['Unnamed: 0', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre', 'duration_ms']
    df_model = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    if 'explicit' in df_model.columns:
        df_model['explicit'] = df_model['explicit'].astype(int)
    return df_model