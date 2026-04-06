"""
Ferramenta de clima usando Open-Meteo (gratuita, sem API key).
"""

import requests
from core.config import Config


def get_weather(city: str = None) -> str:
    config = Config()
    city = city or config.WEATHER_CITY

    try:
        # 1. Geocoding: cidade → coordenadas
        geo = requests.get(
            config.GEOCODING_API,
            params={"name": city, "count": 1, "language": "pt", "format": "json"},
            timeout=5
        ).json()

        if not geo.get("results"):
            return f"Não encontrei dados de clima para '{city}'."

        loc = geo["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        name = loc.get("name", city)

        # 2. Clima atual
        weather = requests.get(
            config.WEATHER_API,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "timezone": "auto"
            },
            timeout=5
        ).json()

        current = weather.get("current", {})
        temp = current.get("temperature_2m", "?")
        humidity = current.get("relative_humidity_2m", "?")
        wind = current.get("wind_speed_10m", "?")
        code = current.get("weather_code", 0)

        condition = _weather_code_to_text(code)

        return (
            f"Clima em {name}: {condition}, {temp}°C, "
            f"umidade {humidity}%, vento {wind} km/h."
        )

    except Exception as e:
        return f"Erro ao buscar clima: {e}"


def _weather_code_to_text(code: int) -> str:
    """Converte código WMO para descrição em português."""
    if code == 0:
        return "céu limpo"
    elif code in (1, 2):
        return "parcialmente nublado"
    elif code == 3:
        return "nublado"
    elif code in (45, 48):
        return "névoa"
    elif code in range(51, 68):
        return "chuva leve"
    elif code in range(71, 78):
        return "neve"
    elif code in range(80, 83):
        return "pancadas de chuva"
    elif code in range(95, 100):
        return "trovoada"
    return "condição desconhecida"
