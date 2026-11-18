import pandas as pd
import joblib
from pathlib import Path
from caracteristicas import extraer_caracteristicas

# Cambia a True SOLO si quieres ver las features en pantalla durante la defensa
MODO_DIAGNOSTICO = False

ruta_modelo = Path("modelos/modelo_phishing.pkl")
if not ruta_modelo.exists():
    print("ERROR: Primero debes ejecutar entrenar.py")
    exit()

datos_modelo = joblib.load(ruta_modelo)
modelo = datos_modelo["modelo"]
columnas = datos_modelo["columnas"]

def detectar_url(url):
    caracteristicas = extraer_caracteristicas(url)
    entrada = pd.DataFrame([caracteristicas])[columnas]

    if MODO_DIAGNOSTICO:
        print("\n--- DIAGNÓSTICO ---")
        print("Características:", caracteristicas)

    prob = modelo.predict_proba(entrada)[0][1]
    prediccion = 1 if prob >= 0.5 else 0

    print("\nRESULTADO DEL ANÁLISIS")
    print("-" * 30)
    print("URL:", url)
    print("Clasificación:", "POSIBLE PHISHING" if prediccion == 1 else "POSIBLE URL LEGÍTIMA")
    print(f"Probabilidad de phishing: {round(prob * 100, 2)} %")

if __name__ == "__main__":
    print("DETECTOR DE URLS DE PHISHING")
    print("=" * 40)
    while True:
        url = input("\nIntroduce una URL (o 'salir'): ").strip()
        if url.lower() == "salir":
            break
        if url:
            detectar_url(url)
    print("👋 Gracias por usar el detector.")
