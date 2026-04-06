"""
Diagnóstico da wake word "Anfitrião".

Modos de uso:
  python test_wakeword.py --sample    # testa com um arquivo WAV de treino
  python test_wakeword.py --mic       # mostra score em tempo real pelo microfone
  python test_wakeword.py --both      # faz os dois
"""

import argparse
import numpy as np
import wave
import sounddevice as sd
from openwakeword.model import Model as WakeWordModel

MODEL_PATH = "./models/anfitriao.onnx"
SAMPLE_DIR = "./wake_word_samples/positive"
SAMPLE_RATE = 16000
CHUNK = 1280  # ~80ms


def load_model():
    print(f"Carregando modelo: {MODEL_PATH}")
    oww = WakeWordModel(wakeword_models=[MODEL_PATH], inference_framework="onnx")
    print("Modelo carregado.\n")
    return oww


def test_with_sample(oww):
    """Testa o modelo passando um arquivo WAV de treino."""
    import os
    samples = sorted(os.listdir(SAMPLE_DIR))
    if not samples:
        print("Nenhum arquivo encontrado em", SAMPLE_DIR)
        return

    sample_path = os.path.join(SAMPLE_DIR, samples[0])
    print(f"Testando com arquivo de treino: {sample_path}")

    with wave.open(sample_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)

    # Processa em chunks de 1280 amostras
    scores = []
    for i in range(0, len(audio) - CHUNK, CHUNK):
        chunk = audio[i:i + CHUNK]
        preds = oww.predict(chunk)
        model_key = list(preds.keys())[0]
        score = preds[model_key]
        scores.append(score)
        if score > 0.1:
            print(f"  chunk {i//CHUNK:3d}: score = {score:.4f}  {'<<<' if score > 0.5 else ''}")

    print(f"\nScore máximo: {max(scores):.4f}")
    print(f"Score médio:  {np.mean(scores):.4f}")
    if max(scores) < 0.3:
        print("\n[!] Score baixo mesmo com arquivo de treino.")
        print("    O modelo pode precisar de mais amostras ou re-treino.")
    elif max(scores) < 0.5:
        print("\n[!] Score abaixo do threshold padrão (0.5).")
        print("    Tente reduzir WAKE_WORD_THRESHOLD para 0.3 em config.py")
    else:
        print("\n[OK] Modelo detectou o arquivo de treino corretamente.")


def test_with_mic(oww):
    """Mostra score em tempo real pelo microfone."""
    print("Ouvindo pelo microfone... Fale 'Anfitrião'. Ctrl+C para parar.\n")
    model_key = None

    counter = [0]

    def callback(indata, frames, time_info, status):
        nonlocal model_key
        audio_chunk = (indata[:, 0] * 32768).astype(np.int16)
        preds = oww.predict(audio_chunk)

        if model_key is None:
            model_key = list(preds.keys())[0]
            print(f"Chave do modelo: '{model_key}'")

        score = preds[model_key]
        counter[0] += 1

        # Mostra sempre a cada 10 chunks (~800ms), ou quando score for alto
        if counter[0] % 10 == 0 or score > 0.05:
            bar = "█" * int(score * 30)
            tag = "🎯 DETECTADO!" if score > 0.5 else ""
            print(f"Score: {score:.4f}  {bar}  {tag}")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=CHUNK, callback=callback):
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nEncerrado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Testa com arquivo WAV de treino")
    parser.add_argument("--mic",    action="store_true", help="Testa em tempo real pelo microfone")
    parser.add_argument("--both",   action="store_true", help="Faz os dois testes")
    args = parser.parse_args()

    if not any([args.sample, args.mic, args.both]):
        args.both = True  # padrão: faz os dois

    oww = load_model()

    if args.sample or args.both:
        test_with_sample(oww)
        print()

    if args.mic or args.both:
        test_with_mic(oww)
