import json, joblib, os, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_train, y_pred_test = model.predict(X_train), model.predict(X_test)
    
    metrics = {
        "MAE_treino": round(float(mean_absolute_error(y_train, y_pred_train)), 2),
        "MAE_teste": round(float(mean_absolute_error(y_test, y_pred_test)), 2),
        "RMSE_teste": round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 2),
        "R2_teste": round(float(r2_score(y_test, y_pred_test)), 4)
    }
    return model, metrics

def save_model_and_metrics(model, metrics, model_path, metrics_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)