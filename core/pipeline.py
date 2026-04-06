"""
CompanionPipeline - Orquestra o fluxo completo:
ESPERA → OUVE → PENSA → FALA → volta para ESPERA
"""

import asyncio
import io
import wave
import tempfile
import threading
import numpy as np
import sounddevice as sd
from core.config import Config
from core.router import Router
from core.memory import ConversationMemory


class CompanionPipeline:
    def __init__(self):
        self.config = Config()
        self.router = Router()
        self.memory = ConversationMemory()
        self.is_listening = False
        self.is_processing = False
        self._state_callback = None
        self._move_callback = None

    def set_state_callback(self, callback):
        """Registra função que a UI usa para receber atualizações de estado."""
        self._state_callback = callback

    def set_move_callback(self, callback):
        """Registra função que a UI usa para receber comandos de mover janela."""
        self._move_callback = callback

    def _set_state(self, state: str):
        """Notifica a UI sobre mudança de estado."""
        if self._state_callback:
            self._state_callback(state)

    # ─── LOOP PRINCIPAL ───────────────────────────────────

    async def listen_loop(self, on_activate=None):
        """Loop principal: fica em ESPERA ouvindo wake word."""
        self._loop = asyncio.get_event_loop()
        print("[Pipeline] Loop de escuta iniciado.")
        self._set_state("espera")

        try:
            import openwakeword
            from openwakeword.model import Model as WakeWordModel
            oww = WakeWordModel(wakeword_models=[self.config.WAKE_WORD], inference_framework="onnx")
            print(f"[Pipeline] Wake word '{self.config.WAKE_WORD}' ativa.")
        except (ImportError, ValueError) as e:
            print(f"[Pipeline] openWakeWord não disponível ({e}). Usando apenas ativação por clique.")
            oww = None

        if oww is None:
            # Sem wake word, mantém o loop vivo para processar cliques
            while True:
                await asyncio.sleep(0.5)
            return

        chunk_size = 1280  # ~80ms a 16kHz

        def audio_callback(indata, frames, time_info, status):
            if self.is_processing:
                return
            audio_chunk = (indata[:, 0] * 32768).astype(np.int16)
            predictions = oww.predict(audio_chunk)

            score = predictions.get(self.config.WAKE_WORD, 0)
            if score >= self.config.WAKE_WORD_THRESHOLD:
                print(f"[Pipeline] Wake word detectada! Score: {score:.2f}")
                asyncio.run_coroutine_threadsafe(
                    self.activate(triggered_by="voice"),
                    asyncio.get_event_loop()
                )

        with sd.InputStream(
            samplerate=self.config.AUDIO_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            callback=audio_callback
        ):
            while True:
                await asyncio.sleep(0.1)

    # ─── ATIVAÇÃO ─────────────────────────────────────────

    async def activate(self, triggered_by="click"):
        """Ponto de entrada quando o companheiro é ativado."""
        if self.is_processing:
            print("[Pipeline] Já processando, ignorando ativação.")
            return

        self.is_processing = True
        print(f"[Pipeline] Ativado por: {triggered_by}")

        try:
            # OUVE
            self._set_state("ouve")
            text = await self._listen_and_transcribe()
            if not text:
                print("[Pipeline] Nada transcrito, cancelando.")
                return

            print(f"[Pipeline] Transcrito: {text}")

            # PENSA
            self._set_state("pensa")
            response = await self._think(text)
            print(f"[Pipeline] Resposta: {response}")

            # Comando especial de mover janela
            if response.startswith("__MOVE_WINDOW__"):
                position = response[len("__MOVE_WINDOW__"):]
                if self._move_callback:
                    self._move_callback(position)
                return

            # FALA
            self._set_state("fala")
            await self._speak(response)

        except Exception as e:
            print(f"[Pipeline] Erro: {e}")
        finally:
            self.is_processing = False
            self._set_state("espera")

    # ─── OUVE ─────────────────────────────────────────────

    async def _listen_and_transcribe(self) -> str:
        """Grava áudio do microfone e transcreve com Whisper."""
        print("[STT] Gravando...")
        audio_data = await asyncio.to_thread(self._record_audio)
        if audio_data is None:
            return ""

        print("[STT] Transcrevendo com Whisper...")
        text = await asyncio.to_thread(self._transcribe, audio_data)
        return text.strip()

    def _record_audio(self) -> np.ndarray | None:
        """Grava até silêncio detectado ou tempo máximo."""
        sample_rate = self.config.AUDIO_SAMPLE_RATE
        silence_threshold = self.config.RECORDING_SILENCE_THRESHOLD
        silence_duration = self.config.RECORDING_SILENCE_DURATION
        max_duration = self.config.RECORDING_MAX_DURATION

        chunks = []
        silent_chunks = 0
        silence_chunks_needed = int(silence_duration * sample_rate / 1024)
        max_chunks = int(max_duration * sample_rate / 1024)

        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=1024) as stream:
            for _ in range(max_chunks):
                data, _ = stream.read(1024)
                chunks.append(data.copy())

                # Detecta silêncio
                volume = np.abs(data).mean()
                if volume < silence_threshold:
                    silent_chunks += 1
                    if silent_chunks >= silence_chunks_needed and len(chunks) > silence_chunks_needed:
                        break
                else:
                    silent_chunks = 0

        if len(chunks) < 5:
            return None

        return np.concatenate(chunks, axis=0)

    def _transcribe(self, audio: np.ndarray) -> str:
        """Usa faster-whisper para transcrever o áudio."""
        import builtins
        model = getattr(builtins, "_whisper_model", None)

        if model is None:
            from faster_whisper import WhisperModel
            model = WhisperModel(
                self.config.WHISPER_MODEL,
                device=self.config.WHISPER_DEVICE,
                compute_type="float16"
            )

        # Converte para WAV temporário
        audio_int16 = (audio[:, 0] * 32768).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.AUDIO_SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            tmp_path = f.name

        segments, _ = model.transcribe(
            tmp_path,
            language=self.config.WHISPER_LANGUAGE,
            vad_filter=True
        )
        return " ".join(seg.text for seg in segments)

    # ─── PENSA ────────────────────────────────────────────

    async def _think(self, text: str) -> str:
        """Roteia o texto para o módulo correto e obtém resposta."""
        history = self.memory.get_history()
        response = await self.router.route(text, history)
        self.memory.add(role="user", content=text)
        self.memory.add(role="assistant", content=response)
        return response

    # ─── FALA ─────────────────────────────────────────────

    async def _speak(self, text: str):
        """Sintetiza voz com Piper TTS e reproduz."""
        await asyncio.to_thread(self._tts_and_play, text)

    def _tts_and_play(self, text: str):
        """Gera áudio com Piper e reproduz com sounddevice."""
        try:
            from piper.voice import PiperVoice

            model_path = self._find_piper_model()
            if not model_path:
                print(f"[TTS] Modelo '{self.config.PIPER_MODEL}.onnx' não encontrado. Coloque-o em ./models/")
                return

            voice = PiperVoice.load(model_path)
            chunks = []
            for chunk in voice.synthesize(text):
                chunks.append(chunk.audio_float_array)

            if chunks:
                audio = np.concatenate(chunks).astype(np.float32)
                sd.play(audio, voice.config.sample_rate)
                sd.wait()
        except ImportError:
            print("[TTS] Piper não encontrado. Instale via: pip install piper-tts")
        except Exception as e:
            print(f"[TTS] Erro: {e}")

    def _find_piper_model(self) -> str | None:
        """Procura o arquivo .onnx do modelo Piper em locais comuns."""
        import os
        model_name = self.config.PIPER_MODEL
        candidates = [
            f"./models/{model_name}.onnx",
            os.path.join(os.path.expanduser("~"), ".local", "share", "piper", "voices", f"{model_name}.onnx"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "piper", "voices", f"{model_name}.onnx"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None
