"""
Router - Agente que classifica a intenção e roteia para o módulo correto.

Fluxo:
  texto do usuário
    → LLM classifica a intenção
    → redireciona para ferramenta (clima, dólar, RAG, etc.) ou responde direto
    → retorna resposta final
"""

import json
import requests
from core.config import Config
from tools.weather import get_weather
from tools.exchange import get_exchange_rate
from tools.rag import RAGSearch


# Ferramentas disponíveis para o agente
TOOLS_SCHEMA = [
    {
        "name": "get_weather",
        "description": "Busca informações do clima atual para uma cidade.",
        "parameters": ["city (string, opcional — usa cidade padrão se não informada)"]
    },
    {
        "name": "get_exchange_rate",
        "description": "Busca a cotação atual de uma moeda em relação ao Real (BRL).",
        "parameters": ["currency (string, ex: 'USD', 'EUR', 'GBP')"]
    },
    {
        "name": "search_knowledge",
        "description": "Busca na base de conhecimento local (textos, sites, documentos que você treinou).",
        "parameters": ["query (string)"]
    },
    {
        "name": "move_window",
        "description": "Move a janela do companheiro para uma posição na tela ou para outro monitor.",
        "parameters": ["position (string, ex: 'canto superior direito', 'centro', 'canto inferior esquerdo', 'outro monitor')"]
    },
    {
        "name": "none",
        "description": "Nenhuma ferramenta necessária. Responde diretamente com LLM.",
        "parameters": []
    }
]

ROUTER_PROMPT = """Você é um roteador de intenções. Analise a mensagem do usuário e decida qual ferramenta usar.

Ferramentas disponíveis:
{tools}

Responda APENAS com um JSON válido, sem texto adicional:
{{
  "tool": "nome_da_ferramenta",
  "params": {{"parametro": "valor"}}
}}

Exemplos:
- "Como está o tempo?" → {{"tool": "get_weather", "params": {{"city": "Rio de Janeiro"}}}}
- "Quanto está o dólar?" → {{"tool": "get_exchange_rate", "params": {{"currency": "USD"}}}}
- "O que você sabe sobre meus projetos?" → {{"tool": "search_knowledge", "params": {{"query": "projetos"}}}}
- "Vai para o canto direito" → {{"tool": "move_window", "params": {{"position": "canto superior direito"}}}}
- "Vai para o outro monitor" → {{"tool": "move_window", "params": {{"position": "outro monitor"}}}}
- "Me conta uma piada" → {{"tool": "none", "params": {{}}}}

Mensagem do usuário: "{message}"
"""


class Router:
    def __init__(self):
        self.config = Config()
        self.rag = RAGSearch()

    async def route(self, text: str, history: list) -> str:
        """Classifica a intenção e roteia para o módulo correto."""

        # 1. Pede ao LLM para classificar a intenção
        tool_decision = await self._classify_intent(text)
        tool = tool_decision.get("tool", "none")
        params = tool_decision.get("params", {})

        print(f"[Router] Ferramenta: {tool} | Params: {params}")

        # 2. Executa a ferramenta correspondente
        context = ""

        if tool == "get_weather":
            city = params.get("city", self.config.WEATHER_CITY)
            context = get_weather(city)

        elif tool == "get_exchange_rate":
            currency = params.get("currency", "USD")
            context = get_exchange_rate(currency)

        elif tool == "search_knowledge":
            query = params.get("query", text)
            context = self.rag.search(query)

        elif tool == "move_window":
            position = params.get("position", "centro")
            return f"__MOVE_WINDOW__{position}"  # Sinal especial para a UI

        # 3. Gera resposta final com o LLM
        return await self._generate_response(text, history, context)

    async def _classify_intent(self, text: str) -> dict:
        """Usa o LLM para classificar a intenção do texto."""
        tools_str = "\n".join(
            f"- {t['name']}: {t['description']}" for t in TOOLS_SCHEMA
        )
        prompt = ROUTER_PROMPT.format(tools=tools_str, message=text)

        try:
            resp = requests.post(
                f"{self.config.OLLAMA_URL}/api/generate",
                json={
                    "model": self.config.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}  # Baixa temperatura para decisões
                },
                timeout=20
            )
            raw = resp.json()["response"].strip()
            return json.loads(raw)
        except Exception as e:
            print(f"[Router] Erro na classificação: {e}")
            return {"tool": "none", "params": {}}

    async def _generate_response(self, user_text: str, history: list, context: str = "") -> str:
        """Gera a resposta final usando o LLM com contexto e histórico."""

        # Monta o prompt com histórico de conversa
        system = self.config.SYSTEM_PROMPT.format(name=self.config.COMPANION_NAME)

        messages = [{"role": "system", "content": system}]

        # Adiciona histórico
        for msg in history[-self.config.CONTEXT_WINDOW:]:
            messages.append(msg)

        # Adiciona contexto de ferramenta se disponível
        user_content = user_text
        if context:
            user_content = f"{user_text}\n\n[Informação obtida agora]: {context}"

        messages.append({"role": "user", "content": user_content})

        try:
            resp = requests.post(
                f"{self.config.OLLAMA_URL}/api/chat",
                json={
                    "model": self.config.LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": self.config.LLM_MAX_TOKENS
                    },
                    "keep_alive": self.config.LLM_KEEP_ALIVE
                },
                timeout=30
            )
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            print(f"[Router] Erro na geração: {e}")
            return "Desculpa, tive um problema ao processar sua mensagem."
