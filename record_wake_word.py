"""
Grava amostras reais da sua voz para treinar a wake word "Anfitrião".
As gravações são salvas em wake_word_samples/positive/ continuando
a numeração das amostras sintéticas já existentes.

Uso:
  python record_wake_word.py
"""

import os
import wave
import time
import numpy as np
import sounddevice as sd

# ─── CONFIGURAÇÃO ──────────────────────────────────────────────
N_SAMPLES       = 40          # quantas gravações fazer
RECORD_SECONDS  = 2.5         # duração de cada gravação (segundos)
SAMPLE_RATE     = 16000       # deve ser 16kHz para o openWakeWord
OUTPUT_DIR      = "./wake_word_samples/positive"
COUNTDOWN       = 3           # segundos de contagem regressiva
# ───────────────────────────────────────────────────────────────


def next_sample_index(directory: str) -> int:
    """Retorna o próximo índice disponível para nomear os arquivos."""
    existing = [f for f in os.listdir(directory) if f.endswith(".wav")]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            indices.append(int(f.replace("sample_", "").replace(".wav", "")))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


def record(seconds: float, sample_rate: int) -> np.ndarray:
    """Grava áudio do microfone e retorna como array int16."""
    n_frames = int(seconds * sample_rate)
    audio = sd.rec(n_frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return (audio[:, 0] * 32768).astype(np.int16)


def save_wav(path: str, audio: np.ndarray, sample_rate: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_index = next_sample_index(OUTPUT_DIR)

    print("=" * 50)
    print("  Gravação de Wake Word — Anfitrião")
    print("=" * 50)
    print(f"\nVamos gravar {N_SAMPLES} amostras da sua voz.")
    print(f"Cada gravação dura {RECORD_SECONDS:.0f} segundos.")
    print(f"Diga 'Anfitrião' claramente quando ver 'GRAVE!'.\n")
    print("Pressione Enter para começar...")
    input()

    saved = 0
    for i in range(N_SAMPLES):
        sample_index = start_index + i
        filename = f"sample_{sample_index:04d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"\n[{i+1}/{N_SAMPLES}] Prepare-se...")
        for c in range(COUNTDOWN, 0, -1):
            print(f"  {c}...", end="\r")
            time.sleep(1)

        print("  🎙  GRAVE!   ", end="\r")
        audio = record(RECORD_SECONDS, SAMPLE_RATE)
        print(f"  ✓  Gravado — {filename}")

        save_wav(filepath, audio, SAMPLE_RATE)
        saved += 1

        # Pequena pausa entre gravações
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"  {saved} amostras salvas em '{OUTPUT_DIR}'")
    print(f"  Total de amostras na pasta: {len(os.listdir(OUTPUT_DIR))}")
    print(f"{'='*50}")
    print("\nPróximos passos:")
    print("  1. Recrie o zip: python -c \"import zipfile,os; z=zipfile.ZipFile('wake_word_samples.zip','w'); [z.write(f'./wake_word_samples/positive/{f}', f'positive/{f}') for f in os.listdir('./wake_word_samples/positive')]; z.close(); print('Zip criado.')\"")
    print("  2. Faça upload do novo zip no Colab")
    print("  3. Siga o COLAB_GUIDE.md do início")


if __name__ == "__main__":
    main()
