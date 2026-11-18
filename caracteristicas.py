
# ==========================================
# PROYECTO: DETECCION DE URLS DE PHISHING
# PASO 4: EXTRACCION DE CARACTERISTICAS
# ==========================================

import pandas as pd
import ipaddress
from urllib.parse import urlparse


def extraer_caracteristicas(url):

    # Convertir URL a texto
    url = str(url).strip()

    # Preparar la URL para analizarla
    url_analizar = url if "://" in url else "//" + url

    partes = urlparse(url_analizar)

    # Obtener dominio y ruta
    dominio = partes.hostname or ""
    ruta = partes.path or ""

    # Comprobar si utiliza una direccion IP
    try:
        ipaddress.ip_address(dominio)
        tiene_ip = 1
    except ValueError:
        tiene_ip = 0

    # Palabras que pueden aparecer en phishing
    palabras_sospechosas = ["verify", "update", "secure", "confirm", "login", "account", "suspend", "validate"]

    cantidad_palabras = sum(
        palabra in url.lower()
        for palabra in palabras_sospechosas
    )

    # Contar numeros
    cantidad_numeros = sum(
        caracter.isdigit() for caracter in url
    )

    # Crear diccionario de caracteristicas
    caracteristicas = {

        "longitud_url": len(url),

        "longitud_dominio": len(dominio),

        "longitud_ruta": len(ruta),

        "cantidad_puntos": url.count("."),

        "cantidad_guiones": url.count("-"),

        "cantidad_arrobas": url.count("@"),

        "cantidad_interrogaciones": url.count("?"),

        "cantidad_iguales": url.count("="),

        "cantidad_porcentajes": url.count("%"),

        "cantidad_numeros": cantidad_numeros,

        "proporcion_numeros": cantidad_numeros / max(len(url), 1),

        "usa_https": int(partes.scheme.lower() == "https"),

        "tiene_ip": tiene_ip,

        "cantidad_subdominios": max(dominio.count(".") - 1, 0),

        "cantidad_palabras_sospechosas": cantidad_palabras
    }

    return caracteristicas


# ==========================================
# PRUEBA DEL FUNCIONAMIENTO
# ==========================================

if __name__ == "__main__":

    url_prueba = "http://cuenta-verificacion-segura.xyz/login"

    resultado = extraer_caracteristicas(url_prueba)

    print("URL analizada:")
    print(url_prueba)

    print("\nCaracteristicas obtenidas:")

    for nombre, valor in resultado.items():
        print(nombre, ":", valor)
