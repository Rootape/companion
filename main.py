"""
Companion - Assistente de Desktop Local
Ponto de entrada principal da aplicação.
"""

import asyncio
import threading
import sys
from core.warmup import WarmupManager
from core.pipeline import CompanionPipeline
from ui.window import CompanionWindow


def main():
    print("=" * 50)
    print("  Companion - Iniciando...")
    print("=" * 50)

    # 1. Warm-up: carrega todos os modelos antes de mostrar a janela
    warmup = WarmupManager()
    warmup.run()

    # 2. Pipeline de processamento (roda em thread separada)
    pipeline = CompanionPipeline()

    # 3. Janela (roda na thread principal — exigência do Qt)
    window = CompanionWindow(pipeline=pipeline)

    # Inicia o loop de escuta em background
    listener_thread = threading.Thread(
        target=asyncio.run,
        args=(pipeline.listen_loop(on_activate=window.on_activate),),
        daemon=True
    )
    listener_thread.start()

    # Inicia a janela (bloqueia até fechar)
    sys.exit(window.run())


if __name__ == "__main__":
    main()
