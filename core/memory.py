"""
ConversationMemory - Gerencia o histórico da conversa.
Mantém as últimas N mensagens em memória para dar contexto ao LLM.
"""

from collections import deque
from core.config import Config


class ConversationMemory:
    def __init__(self):
        self.config = Config()
        self._history = deque(maxlen=self.config.CONTEXT_WINDOW * 2)

    def add(self, role: str, content: str):
        """Adiciona uma mensagem ao histórico."""
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list:
        """Retorna o histórico como lista de dicts."""
        return list(self._history)

    def clear(self):
        """Limpa o histórico."""
        self._history.clear()
