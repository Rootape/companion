# Companion — Assistente de Desktop Local

Assistente de voz e texto que roda 100% local, sempre visível na tela.
Inspirado no projeto BMO do canal do YouTube, mas aqui será O Anfitrião de Ordem Paranormal.

---

## Estrutura do Projeto

```
companion/
├── main.py                  ← Ponto de entrada
├── requirements.txt
├── core/
│   ├── config.py            ← TODAS as configurações (edite aqui)
│   ├── warmup.py            ← Carrega modelos na inicialização
│   ├── pipeline.py          ← Orquestra ESPERA→OUVE→PENSA→FALA
│   ├── router.py            ← Agente: classifica intent e roteia
│   └── memory.py            ← Histórico de conversa
├── ui/
│   └── window.py            ← Janela desktop (PyQt6)
├── tools/
│   ├── weather.py           ← Clima via Open-Meteo (gratuito)
│   ├── exchange.py          ← Cotação de moedas (gratuito)
│   └── rag.py               ← Base de conhecimento local (ChromaDB)
└── data/
    └── knowledge_base/      ← Banco ChromaDB (criado automaticamente)
```

---

## Setup Completo

### 1. Python e dependências

```bash
# Python 3.11+ recomendado
pip install -r requirements.txt
```

### 2. Ollama (LLM local)

```bash
# Instale o Ollama: https://ollama.com
# Depois baixe o modelo:
ollama pull llama3.2:3b

# Alternativas mais leves:
# ollama pull phi3:mini
# ollama pull gemma2:2b
```

### 3. Piper TTS (voz em português)

```bash
pip install piper-tts

# Baixe a voz em português:
# https://huggingface.co/rhasspy/piper-voices/tree/main/pt/pt_BR/faber/medium
# Coloque o arquivo .onnx e .onnx.json em: ~/.local/share/piper/voices/
```

### 4. Rodar

```bash
python main.py
```

---

## Configuração

Edite `core/config.py` para personalizar:

- `LLM_MODEL` — qual modelo Ollama usar
- `COMPANION_NAME` — nome do seu companheiro
- `SYSTEM_PROMPT` — personalidade e tonalidade
- `WEATHER_CITY` — sua cidade padrão
- `WAKE_WORD` — palavra de ativação
- `WINDOW_SIZE` / `WINDOW_POSITION` — tamanho e posição inicial

---

## Adicionando Conhecimento (RAG)

```python
from tools.rag import RAGSearch

rag = RAGSearch()

# Adicionar texto direto
rag.add_text("Meu projeto X funciona assim...", source="projetos")

# Adicionar um site
rag.add_url("https://meusite.com/sobre")

# Adicionar arquivo de texto
rag.add_file("/caminho/para/anotacoes.txt")

# Ver o que está na base
print(rag.list_sources())
```

---

## Comandos de Voz

| O que falar | O que acontece |
|---|---|
| "Como está o clima?" | Busca clima da sua cidade |
| "Quanto está o dólar?" | Busca cotação USD/BRL |
| "Vai para o canto direito" | Move a janela |
| "Vai para o centro" | Move a janela |
| "O que você sabe sobre X?" | Busca na sua base RAG |
| Qualquer outra coisa | Responde via LLM direto |

---

## Uso Enquanto Joga

Para liberar mais GPU para o jogo, você pode:

**Opção A** — Usar modelo em CPU (no config.py):
```python
WHISPER_DEVICE = "cpu"
```

**Opção B** — Forçar Ollama em CPU:
```bash
# No terminal antes de rodar:
set OLLAMA_GPU_LAYERS=0   # Windows
export OLLAMA_GPU_LAYERS=0  # Linux/Mac
```

**Opção C** — Usar modelo ainda menor:
```python
LLM_MODEL = "phi3:mini"
WHISPER_MODEL = "tiny"
```

---

## Próximos Passos

- [ ] Interface visual com animação (estética do Anfitrião de Ordem Paranormal)
- [ ] Análise de tela com moondream (screenshot)
- [ ] Memória persistente entre sessões
- [ ] Mais ferramentas (agenda, timers, notas)
- [ ] Voz customizada treinada no Piper
