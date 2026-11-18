# ==========================================
# PROYECTO: DETECCION DE URLS DE PHISHING
# ENTRENAMIENTO Y MEJORAS DEL MODELO
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from caracteristicas import extraer_caracteristicas

print("Bibliotecas cargadas correctamente")

# ==========================================
# 1. CARGAR LOS DATOS
# ==========================================
ruta = Path("datos/phishing_urls.csv")
if not ruta.exists():
    print("ERROR: No se encontro el archivo de datos. Ejecuta descargar_datos.py primero.")
    exit()

datos = pd.read_csv(ruta)
print(f"\nDATOS CARGADOS: {len(datos)} registros")

# ==========================================
# 2. EXPLORACION DE DATOS (Resumen)
# ==========================================
print("\nINFORMACION DEL DATASET")
print(datos.info())
print("\nVALORES NULOS:\n", datos.isnull().sum())
print("\nDISTRIBUCION DE CLASES:\n", datos["label"].value_counts(normalize=True) * 100)

carpeta_resultados = Path("resultados")
carpeta_resultados.mkdir(exist_ok=True)

# Gráfico de distribución
plt.figure(figsize=(6, 4))
sns.countplot(data=datos, x="label")
plt.title("Distribucion de URLs (0: Legitima, 1: Phishing)")
plt.xticks([0, 1], ["Legitimas", "Phishing"])
plt.savefig(carpeta_resultados / "distribucion_urls.png")
plt.close() # Cierra la figura para que el script no se pause

# ==========================================
# 3. EXTRACCION DE CARACTERISTICAS
# ==========================================
print("\nEXTRAYENDO CARACTERISTICAS DE LAS URLS...")
caracteristicas_lista = [extraer_caracteristicas(url) for url in datos["url"]]
X = pd.DataFrame(caracteristicas_lista)
y = datos["label"].astype(int)

print(f"Caracteristicas extraidas: {X.shape[1]} features")
columnas = X.columns.tolist()

# ==========================================
# 4. DIVISION DE DATOS (80/20 Estratificado)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nDIVISION: {len(X_train)} entrenamiento, {len(X_test)} prueba")

# ==========================================
# 5. MODELO DE REFERENCIA (BASELINE)
# ==========================================
modelo_base = DummyClassifier(strategy="most_frequent")
modelo_base.fit(X_train, y_train)
pred_base = modelo_base.predict(X_test)
print(f"\nMODELO BASELINE (Dummy) Accuracy: {accuracy_score(y_test, pred_base):.4f}")

# ==========================================
# 6. MODELO PRINCIPAL: RANDOM FOREST
# ==========================================
print("\nENTRENANDO RANDOM FOREST...")
rf_modelo = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
)
rf_modelo.fit(X_train, y_train)
pred_rf = rf_modelo.predict(X_test)
prob_rf = rf_modelo.predict_proba(X_test)[:, 1]

print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))
print("Random Forest F1-score:", f1_score(y_test, pred_rf))

# ==========================================
# 7. MEJORAS (Sección 5.4 de la rúbrica)
# ==========================================
print("\n" + "="*60)
print(" APLICANDO MEJORAS (Seccion 5.4)")
print("="*60)

# --- MEJORA 1: Ajuste de hiperparámetros optimizado ---
print("\n[MEJORA 1] Ajuste de hiperparametros (RandomizedSearchCV)")
# Subconjunto estratificado para ahorrar tiempo (trade-off costo/beneficio)
X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=40000, random_state=42, stratify=y_train)

param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=4),
    param_distributions=param_dist,
    n_iter=8, cv=3, scoring='f1', random_state=42, n_jobs=4, verbose=1
)
search.fit(X_sub, y_sub)
print("Mejores hiperparametros encontrados:", search.best_params_)

# Reentrenar el mejor modelo con TODOS los datos
rf_tuned = RandomForestClassifier(random_state=42, n_jobs=-1, **search.best_params_)
rf_tuned.fit(X_train, y_train)
pred_tuned = rf_tuned.predict(X_test)
prob_tuned = rf_tuned.predict_proba(X_test)[:, 1]

# --- MEJORA 2: Comparación de algoritmos ---
print("\n[MEJORA 2] Comparacion con HistGradientBoosting (Optimizado para grandes datasets)")
hgb_modelo = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42)
hgb_modelo.fit(X_train, y_train)
pred_hgb = hgb_modelo.predict(X_test)
prob_hgb = hgb_modelo.predict_proba(X_test)[:, 1]

# Función auxiliar para métricas
def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'ROC AUC': roc_auc_score(y_true, y_prob)
    }

metricas_rf = get_metrics(y_test, pred_rf, prob_rf)
metricas_tuned = get_metrics(y_test, pred_tuned, prob_tuned)
metricas_hgb = get_metrics(y_test, pred_hgb, prob_hgb)

# Tabla de comparación (Obligatoria para la rúbrica)
df_comparacion = pd.DataFrame({
    'RF Default': metricas_rf,
    'RF Tuned (Mejora 1)': metricas_tuned,
    'HistGradientBoosting (Mejora 2)': metricas_hgb
}).T

print("\nTABLA COMPARATIVA DE MODELOS (Mejoras)")
print(df_comparacion.to_string())
df_comparacion.to_csv(carpeta_resultados / "comparacion_modelos.csv")

# Seleccionar el mejor modelo para guardar
mejor_f1 = df_comparacion['F1-score'].idxmax()
print(f"\n🏆 Mejor modelo seleccionado: {mejor_f1}")

if mejor_f1 == 'RF Tuned (Mejora 1)':
    modelo_final = rf_tuned
elif mejor_f1 == 'HistGradientBoosting (Mejora 2)':
    modelo_final = hgb_modelo
else:
    modelo_final = rf_modelo

# ==========================================
# 8. EVALUACION FINAL Y VISUALIZACIONES
# ==========================================
pred_final = modelo_final.predict(X_test)
prob_final = modelo_final.predict_proba(X_test)[:, 1]

print("\nINFORME DE CLASIFICACION (Mejor Modelo)")
print(classification_report(y_test, pred_final, target_names=["Legitima", "Phishing"], zero_division=0))

# Matriz de confusión
matriz = confusion_matrix(y_test, pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Legitima", "Phishing"])
disp.plot(cmap="Blues")
plt.title("Matriz de confusion - Mejor Modelo")
plt.savefig(carpeta_resultados / "matriz_confusion_final.png")
plt.close()

# Importancia de características
if hasattr(modelo_final, 'feature_importances_'):
    importancias = pd.DataFrame({
        "Caracteristica": columnas,
        "Importancia": modelo_final.feature_importances_
    }).sort_values(by="Importancia", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importancias, x="Importancia", y="Caracteristica")
    plt.title("Importancia de las caracteristicas")
    plt.tight_layout()
    plt.savefig(carpeta_resultados / "importancia_caracteristicas.png")
    plt.close()

# Validación cruzada
cv_scores = cross_val_score(modelo_final, X_train, y_train, cv=3, scoring="f1", n_jobs=4)
print(f"\nVALIDACION CRUZADA (3 folds) F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ==========================================
# 9. GUARDAR EL MODELO
# ==========================================
carpeta_modelos = Path("modelos")
carpeta_modelos.mkdir(exist_ok=True)

joblib.dump({
    "modelo": modelo_final,
    "columnas": columnas
}, carpeta_modelos / "modelo_phishing.pkl")

print("\n" + "="*60)
print(" ENTRENAMIENTO FINALIZADO EXITOSAMENTE")
print(f" Modelo guardado en: {carpeta_modelos / 'modelo_phishing.pkl'}")
print("="*60) 