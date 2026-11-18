
# ==========================================
# PROYECTO: DETECCION DE URLS DE PHISHING
# PASO 1: DESCARGA DEL CONJUNTO DE DATOS
# ==========================================

import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo

print("Descargando conjunto de datos...")

# Descargar dataset PhiUSIIL desde UCI
dataset = fetch_ucirepo(id=967)

# Obtener las URLs y sus etiquetas
urls = dataset.data.features["URL"]
etiquetas = dataset.data.targets["label"]

# Crear una tabla con los datos necesarios
datos = pd.DataFrame({
    "url": urls,
    "label_original": etiquetas
})

# Eliminar registros incompletos
datos = datos.dropna()

# Eliminar URLs duplicadas
datos = datos.drop_duplicates(subset=["url"])

# Convertir las etiquetas:
# 0 original = phishing -> 1
# 1 original = legítima -> 0
datos["label"] = (datos["label_original"].astype(int) == 0).astype(int)

# Conservar solamente las columnas necesarias
datos = datos[["url", "label"]]

# Crear carpeta de almacenamiento
carpeta = Path("datos")
carpeta.mkdir(exist_ok=True)

# Guardar archivo CSV
ruta = carpeta / "phishing_urls.csv"
datos.to_csv(ruta, index=False, encoding="utf-8")

# Mostrar información
print("\nDESCARGA FINALIZADA")
print("Archivo:", ruta)
print("Total de URLs:", len(datos))

print("\nDistribución de clases:")
print(datos["label"].value_counts())

print("\nPrimeros registros:")
print(datos.head())