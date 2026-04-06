"""
Gera amostras de áudio para treinar a wake word "Anfitrião".
Usa o Piper TTS (pt-BR) para garantir pronúncia correta em português.

Uso:
  python generate_wake_word_samples.py

As amostras serão salvas em ./wake_word_samples/positive/
Faça upload desta pasta no Google Colab para treinar o modelo.
"""

import os
import wave
import numpy as np
from pathlib import Path

# ─── CONFIGURAÇÃO ─────────────────────────────────────────────
WAKE_WORD     = "Anfitrião"
N_SAMPLES     = 250
OUTPUT_DIR    = "./wake_word_samples/positive"
PIPER_MODEL   = "pt_BR-faber-medium"
TARGET_RATE   = 16000          # openWakeWord exige 16kHz
# ──────────────────────────────────────────────────────────────

# Variações de texto — mais diversidade = modelo mais robusto
TEXT_VARIATIONS = [
    "Anfitrião",
    "Anfitrião!",
    "Ei, Anfitrião",
    "Oi, Anfitrião",
    "Olá, Anfitrião",
    "Hey, Anfitrião",
    "Anfitrião, me ouça",
    "Preciso de você, Anfitrião",
]


def find_piper_model(model_name: str) -> str | None:
    candidates = [
        f"./models/{model_name}.onnx",
        os.path.join(os.path.expanduser("~"), ".local", "share", "piper", "voices", f"{model_name}.onnx"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "piper", "voices", f"{model_name}.onnx"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def resample_numpy(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample simples via interpolação linear (sem dependências extras)."""
    if orig_rate == target_rate:
        return audio
    new_len = int(len(audio) * target_rate / orig_rate)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio.astype(np.float32)).astype(np.int16)


def augment(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica variações aleatórias para aumentar a diversidade do dataset."""
    audio = audio.astype(np.float32)

    # Variação de volume
    audio *= rng.uniform(0.65, 1.25)

    # Variação de velocidade (estica/comprime o áudio)
    speed = rng.uniform(0.85, 1.15)
    new_len = int(len(audio) / speed)
    indices = np.linspace(0, len(audio) - 1, new_len)
    audio = np.interp(indices, np.arange(len(audio)), audio)

    # Ruído de fundo leve
    noise = rng.uniform(0, 250)
    audio += rng.normal(0, noise, len(audio))

    # Silêncio aleatório no início (0 a 200ms @ 16kHz)
    pad = rng.integers(0, int(TARGET_RATE * 0.2))
    audio = np.concatenate([np.zeros(pad), audio])

    return np.clip(audio, -32768, 32767).astype(np.int16)


def save_wav(path: str, audio: np.ndarray, sample_rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def synthesize(voice, text: str) -> np.ndarray:
    chunks = []
    for chunk in voice.synthesize(text):
        audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
        chunks.append(audio_int16)
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)


def main():
    try:
        from piper.voice import PiperVoice
    except ImportError:
        print("Erro: piper-tts não instalado. Execute: pip install piper-tts")
        return

    model_path = find_piper_model(PIPER_MODEL)
    if not model_path:
        print(f"Modelo '{PIPER_MODEL}.onnx' não encontrado em ./models/")
        print("Baixe o modelo e coloque em ./models/ antes de continuar.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    voice = PiperVoice.load(model_path)
    orig_rate = voice.config.sample_rate
    rng = np.random.default_rng(42)

    # Teste rápido antes de gerar tudo
    print("Testando síntese...")
    test_audio = synthesize(voice, "Teste")
    print(f"Teste OK — {len(test_audio)} samples gerados a {orig_rate}Hz")

    print(f"\nGerando {N_SAMPLES} amostras de '{WAKE_WORD}'...")
    print(f"Salvando em: {OUTPUT_DIR}\n")

    saved = 0
    for i in range(N_SAMPLES):
        text = TEXT_VARIATIONS[i % len(TEXT_VARIATIONS)]

        audio = synthesize(voice, text)
        if len(audio) == 0:
            continue

        # Resample para 16kHz
        audio = resample_numpy(audio, orig_rate, TARGET_RATE)

        # Primeira amostra de cada variação fica limpa; as demais são augmentadas
        if i >= len(TEXT_VARIATIONS):
            audio = augment(audio, rng)

        path = os.path.join(OUTPUT_DIR, f"sample_{i:04d}.wav")
        save_wav(path, audio, TARGET_RATE)
        saved += 1

        if saved % 50 == 0:
            print(f"  {saved}/{N_SAMPLES} amostras geradas...")

    print(f"\nPronto! {saved} amostras salvas em '{OUTPUT_DIR}'")
    print("\nPróximos passos:")
    print("  1. Compacte a pasta 'wake_word_samples/' em um .zip")
    print("  2. Faça upload no Google Colab junto com o notebook de treino")
    print("  3. Aponte o notebook para esta pasta como 'positive_samples_directory'")


if __name__ == "__main__":
    main()
