"""
Config - Todas as configurações da aplicação em um lugar só.
Edite aqui para personalizar o comportamento do seu companheiro.
"""


class Config:
    # ─── LLM ──────────────────────────────────────────────
    LLM_MODEL = "llama3.2:3b"          # Modelo leve para uso durante jogos
    # LLM_MODEL = "phi3:mini"           # Alternativa ainda mais leve
    # LLM_MODEL = "gemma2:2b"           # Outra boa opção leve
    OLLAMA_URL = "http://localhost:11434"
    LLM_KEEP_ALIVE = "10m"             # Mantém modelo carregado por 10min de inatividade
    LLM_MAX_TOKENS = 300               # Respostas curtas para ser rápido
    CONTEXT_WINDOW = 10                # Quantas mensagens anteriores lembrar

    # ─── STT (Transcrição de voz) ──────────────────────────
    WHISPER_MODEL = "tiny"             # tiny/base/small — quanto menor, mais rápido
    WHISPER_DEVICE = "cpu"             # "cuda" para GPU, "cpu" para processador
    WHISPER_LANGUAGE = "pt"            # Português

    # ─── Wake Word ─────────────────────────────────────────
    WAKE_WORD = "./models/anfitriao.onnx"  # Modelo customizado treinado em pt-BR
    WAKE_WORD_THRESHOLD = 0.5              # Sensibilidade (0.0 a 1.0)

    # ─── TTS ───────────────────────────────────────────────
    PIPER_MODEL = "pt_BR-faber-medium" # Voz em português brasileiro
    # Baixe modelos em: https://huggingface.co/rhasspy/piper-voices
    PIPER_SPEED = 1.0                  # Velocidade da voz (1.0 = normal)

    # ─── Personalidade ─────────────────────────────────────
    COMPANION_NAME = "Companheiro"
    SYSTEM_PROMPT = """Você é um companheiro de desktop amigável e prestativo.
Seu nome é {name}.
Responda de forma natural, concisa e descontraída.
Evite respostas longas demais — seja direto e simpático.
Fale sempre em português brasileiro.
Você tem acesso a ferramentas para buscar clima, cotações e outras informações em tempo real."""

    # ─── Janela ────────────────────────────────────────────
    WINDOW_SIZE = (200, 200)           # Tamanho inicial da janela (largura, altura)
    WINDOW_POSITION = (100, 100)       # Posição inicial na tela (x, y)
    WINDOW_OPACITY = 0.95              # Transparência da janela
    ALWAYS_ON_TOP = True               # Fica sempre visível sobre outros apps

    # ─── Ferramentas online ─────────────────────────────────
    WEATHER_CITY = "Rio de Janeiro"    # Cidade padrão para clima
    WEATHER_API = "https://api.open-meteo.com/v1/forecast"
    GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
    EXCHANGE_API = "https://api.exchangerate-open.com/v6/latest"  # Gratuita, sem chave

    # ─── RAG (Memória / Conhecimento) ──────────────────────
    RAG_DB_PATH = "./data/knowledge_base"
    RAG_CHUNK_SIZE = 500
    RAG_TOP_K = 3                      # Quantos trechos buscar por pergunta

    # ─── Áudio ─────────────────────────────────────────────
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1
    RECORDING_SILENCE_THRESHOLD = 0.01
    RECORDING_SILENCE_DURATION = 1.5   # Segundos de silêncio para parar de gravar
    RECORDING_MAX_DURATION = 30        # Máximo de segundos por gravação
