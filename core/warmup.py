"""
WarmupManager - Carrega todos os modelos na inicialização
e os mantém ativos para respostas rápidas.
"""

import subprocess
import requests
import time
from core.config import Config


class WarmupManager:
    def __init__(self):
        self.config = Config()

    def run(self):
        print("\n[Warm-up] Iniciando carregamento dos modelos...")
        self._check_ollama()
        self._warmup_ollama()
        self._check_whisper()
        self._check_piper()
        self._check_wakeword()
        print("[Warm-up] Todos os modelos prontos.\n")

    def _check_ollama(self):
        """Verifica se o Ollama está rodando, inicia se necessário."""
        print("[Warm-up] Verificando Ollama...")
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                print("[Warm-up] Ollama OK")
                return
        except requests.ConnectionError:
            pass

        print("[Warm-up] Iniciando Ollama...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)  # Aguarda inicializar

    def _warmup_ollama(self):
        """
        Envia um prompt vazio para forçar o modelo a carregar na VRAM.
        Após isso, o modelo fica residente e responde mais rápido.
        """
        print(f"[Warm-up] Carregando modelo {self.config.LLM_MODEL} na VRAM...")
        try:
            requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.config.LLM_MODEL,
                    "prompt": "ok",
                    "stream": False,
                    "keep_alive": "10m"  # Mantém o modelo ativo por 10 minutos de inatividade
                },
                timeout=60
            )
            print(f"[Warm-up] {self.config.LLM_MODEL} carregado OK")
        except Exception as e:
            print(f"[Warm-up] AVISO: Não foi possível fazer warm-up do Ollama: {e}")

    def _check_whisper(self):
        """Verifica e pré-carrega o modelo Whisper."""
        print("[Warm-up] Carregando Whisper (faster-whisper)...")
        try:
            from faster_whisper import WhisperModel
            # Instancia o modelo para forçar o download/carregamento
            model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type="int8"
            )
            # Salva para reutilização
            import builtins
            builtins._whisper_model = model
            print(f"[Warm-up] Whisper 'tiny' OK ({self.config.WHISPER_DEVICE.upper()})")
        except Exception as e:
            print(f"[Warm-up] AVISO Whisper: {e}")

    def _check_piper(self):
        """Verifica se o Piper TTS está instalado."""
        print("[Warm-up] Verificando Piper TTS...")
        try:
            from piper.voice import PiperVoice
            print("[Warm-up] Piper TTS OK")
        except ImportError:
            print("[Warm-up] AVISO: Piper TTS não encontrado. Instale via: pip install piper-tts")

    def _check_wakeword(self):
        """Verifica se o openWakeWord está disponível."""
        print("[Warm-up] Verificando openWakeWord...")
        try:
            import openwakeword
            print("[Warm-up] openWakeWord OK")
        except ImportError:
            print("[Warm-up] AVISO: openWakeWord não encontrado. Instale via: pip install openwakeword")
