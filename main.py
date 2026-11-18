"""
main.py - Orquestador del proyecto de Detección de Phishing
Ejecuta todo el pipeline: descarga -> características -> entrenamiento -> detección
"""

import subprocess
import sys
from pathlib import Path


def ejecutar_script(nombre_script):
    """Ejecuta un script del proyecto y muestra su salida."""
    print(f"\n{'='*60}")
    print(f"  EJECUTANDO: {nombre_script}")
    print(f"{'='*60}\n")
    
    resultado = subprocess.run(
        [sys.executable, nombre_script],
        cwd=Path(__file__).parent
    )
    
    if resultado.returncode != 0:
        print(f"\n❌ ERROR al ejecutar {nombre_script}")
        sys.exit(1)
    
    print(f"\n✅ {nombre_script} completado correctamente")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   PROYECTO: DETECCIÓN DE URLS DE PHISHING CON ML   ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    # 1. Descargar datos (si no existen)
    if not Path("datos/phishing_urls.csv").exists():
        ejecutar_script("descargar_datos.py")
    else:
        print("\n📁 Datos ya descargados, omitiendo descarga.")
    
    # 2. Entrenar modelo (incluye extracción de características)
    ejecutar_script("entrenar.py")
    
    # 3. Modo interactivo de detección
    print("\n" + "="*60)
    print("  MODO DETECCIÓN INTERACTIVO")
    print("="*60)
    
    from detector import detectar_url
    
    while True:
        url = input("\nIngresa una URL (o 'salir' para terminar): ")
        if url.lower() == "salir":
            print("\n👋 Gracias por usar el detector de phishing.")
            break
        if url.strip() == "":
            continue
        detectar_url(url)


if __name__ == "__main__":
    main()