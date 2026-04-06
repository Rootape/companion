"""
Ferramenta de cotação de moedas usando ExchangeRate-API (gratuita, sem API key).
"""

import requests


def get_exchange_rate(currency: str = "USD") -> str:
    currency = currency.upper()
    try:
        resp = requests.get(
            f"https://api.exchangerate-open.com/v6/latest/BRL",
            timeout=5
        ).json()

        rates = resp.get("rates", {})
        if currency not in rates:
            return f"Moeda '{currency}' não encontrada."

        # A API retorna quanto 1 BRL vale em outras moedas, então invertemos
        rate_brl_to_currency = rates[currency]
        brl_per_currency = 1 / rate_brl_to_currency

        return f"1 {currency} = R$ {brl_per_currency:.2f} (cotação atual)."

    except Exception as e:
        # Fallback: tenta API alternativa
        try:
            resp = requests.get(
                f"https://api.frankfurter.app/latest?from={currency}&to=BRL",
                timeout=5
            ).json()
            rate = resp["rates"]["BRL"]
            return f"1 {currency} = R$ {rate:.2f} (cotação atual)."
        except Exception:
            return f"Erro ao buscar cotação de {currency}: {e}"
