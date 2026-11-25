import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Generar datos numéricos de prueba para X_raw y y_raw
X_raw, y_raw = make_regression(
    n_samples=200,
    n_features=4,
    n_informative=3,
    noise=15.0,
    random_state=42,
)

# Crear y entrenar el bosque con 50 árboles
rf_reg = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
rf_reg.fit(X_raw, y_raw)

pred_rf = rf_reg.predict(X_raw)

print("=== RANDOM FOREST REGRESSOR ===")
print(f"R^2 Score en Entrenamiento:  {r2_score(y_raw, pred_rf):.4f}")
print("Importancia relativa de variables:", np.round(rf_reg.feature_importances_, 4))