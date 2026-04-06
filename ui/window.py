"""
CompanionWindow - Janela de desktop do companheiro.

Características:
- Sempre visível acima de outros apps
- Sem borda (frameless)
- Arrastável
- Transparente
- Ciente de qual monitor está e onde na tela
- Responde a comandos de movimento por voz
"""

import sys
import asyncio
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QPoint, QTimer, QPropertyAnimation, QRect,
    QEasingCurve, pyqtSignal, QObject
)
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont


# Mapa de posições nomeadas → âncora da tela
POSITION_MAP = {
    "canto superior esquerdo": ("left", "top"),
    "canto superior direito": ("right", "top"),
    "canto inferior esquerdo": ("left", "bottom"),
    "canto inferior direito": ("right", "bottom"),
    "centro": ("center", "center"),
    "esquerda": ("left", "center"),
    "direita": ("right", "center"),
    "cima": ("center", "top"),
    "baixo": ("center", "bottom"),
}


class Signals(QObject):
    state_changed = pyqtSignal(str)
    move_to = pyqtSignal(str)


class CompanionWindow(QWidget):
    def __init__(self, pipeline=None):
        self.app = QApplication(sys.argv)
        super().__init__()

        self.pipeline = pipeline
        self.signals = Signals()
        self.signals.state_changed.connect(self._on_state_changed)
        self.signals.move_to.connect(self._move_to_named_position)

        self._drag_pos = None
        self._current_state = "espera"

        # Registra callbacks no pipeline
        if self.pipeline:
            self.pipeline.set_state_callback(self._request_state_change)
            self.pipeline.set_move_callback(self.move_to_position)

        self._setup_window()
        self._setup_ui()
        self._update_display()

    # ─── SETUP ────────────────────────────────────────────

    def _setup_window(self):
        """Configura a janela: frameless, always-on-top, transparente."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Não aparece na taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(160, 160)

        # Posição inicial
        screen = self.app.primaryScreen().geometry()
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 60
        self.move(x, y)

    def _setup_ui(self):
        """Monta os elementos da interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Indicador de estado (círculo colorido)
        self.state_indicator = QLabel("●")
        self.state_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_indicator.setFont(QFont("Segoe UI", 28))
        layout.addWidget(self.state_indicator)

        # Label de estado
        self.state_label = QLabel("Espera")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setFont(QFont("Segoe UI", 10))
        self.state_label.setStyleSheet("color: rgba(255,255,255,180);")
        layout.addWidget(self.state_label)

        # Botão de ativação por clique
        self.btn_activate = QPushButton("Falar")
        self.btn_activate.setFont(QFont("Segoe UI", 9))
        self.btn_activate.setFixedHeight(28)
        self.btn_activate.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,30);
                color: white;
                border: 1px solid rgba(255,255,255,80);
                border-radius: 8px;
                padding: 2px 10px;
            }
            QPushButton:hover {
                background: rgba(255,255,255,60);
            }
            QPushButton:pressed {
                background: rgba(255,255,255,20);
            }
            QPushButton:disabled {
                color: rgba(255,255,255,80);
            }
        """)
        self.btn_activate.clicked.connect(self._on_click_activate)
        layout.addWidget(self.btn_activate)

        # Label de posição
        self.pos_label = QLabel()
        self.pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pos_label.setFont(QFont("Segoe UI", 7))
        self.pos_label.setStyleSheet("color: rgba(255,255,255,100);")
        layout.addWidget(self.pos_label)

        # Timer para atualizar posição
        self._pos_timer = QTimer()
        self._pos_timer.timeout.connect(self._update_position_label)
        self._pos_timer.start(1000)

    # ─── ESTADO ───────────────────────────────────────────

    def _request_state_change(self, state: str):
        """Chamado pela thread do pipeline — usa signal para ir para a thread da UI."""
        self.signals.state_changed.emit(state)

    def _on_state_changed(self, state: str):
        """Atualiza a UI de acordo com o estado atual."""
        self._current_state = state
        self._update_display()

    def _update_display(self):
        state = self._current_state
        configs = {
            "espera": ("●", "#78909C", "Espera", True),
            "ouve":   ("◎", "#42A5F5", "Ouvindo...", False),
            "pensa":  ("◉", "#AB47BC", "Pensando...", False),
            "fala":   ("●", "#26A69A", "Falando...", False),
        }
        icon, color, label, btn_enabled = configs.get(state, configs["espera"])

        self.state_indicator.setText(icon)
        self.state_indicator.setStyleSheet(f"color: {color};")
        self.state_label.setText(label)
        self.btn_activate.setEnabled(btn_enabled)
        self.update()  # Redesenha o fundo

    # ─── FUNDO CUSTOMIZADO ────────────────────────────────

    def paintEvent(self, event):
        """Desenha o fundo circular/arredondado translúcido."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        state_colors = {
            "espera": QColor(40, 44, 52, 200),
            "ouve":   QColor(13, 71, 161, 210),
            "pensa":  QColor(74, 20, 140, 210),
            "fala":   QColor(0, 77, 64, 210),
        }
        color = state_colors.get(self._current_state, QColor(40, 44, 52, 200))

        painter.setBrush(QBrush(color))
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawRoundedRect(self.rect(), 20, 20)

    # ─── INTERAÇÃO ────────────────────────────────────────

    def _on_click_activate(self):
        """Ativação por clique no botão."""
        if self.pipeline and not self.pipeline.is_processing:
            loop = getattr(self.pipeline, '_loop', None)
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.pipeline.activate(triggered_by="click"),
                    loop
                )
            else:
                print("[UI] Loop do pipeline ainda não está pronto.")

    def on_activate(self):
        """Callback público para ativação externa."""
        self._on_click_activate()

    # ─── ARRASTAR ─────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    # ─── POSIÇÃO / MONITOR ─────────────────────────────────

    def _update_position_label(self):
        pos_name = self._get_position_name()
        monitor_name = self._get_monitor_name()
        self.pos_label.setText(f"{monitor_name} · {pos_name}")

    def _get_monitor_name(self) -> str:
        """Detecta em qual monitor a janela está."""
        screens = self.app.screens()
        win_center = self.geometry().center()
        for i, screen in enumerate(screens):
            if screen.geometry().contains(win_center):
                return f"Monitor {i + 1}"
        return "Monitor ?"

    def _get_position_name(self) -> str:
        """Retorna descrição da posição atual (canto, centro, etc.)."""
        screen = self.app.screenAt(self.geometry().center())
        if not screen:
            return ""

        sg = screen.geometry()
        win = self.geometry()
        win_cx = win.center().x()
        win_cy = win.center().y()

        # Terços horizontais e verticais
        h_third = sg.width() / 3
        v_third = sg.height() / 3

        rel_x = win_cx - sg.left()
        rel_y = win_cy - sg.top()

        h = "esq" if rel_x < h_third else ("dir" if rel_x > h_third * 2 else "centro")
        v = "cima" if rel_y < v_third else ("baixo" if rel_y > v_third * 2 else "meio")

        if h == "centro" and v == "meio":
            return "centro"
        if h == "centro":
            return v
        if v == "meio":
            return h
        return f"{v} {h}"

    def _move_to_named_position(self, position_str: str):
        """Move a janela para uma posição nomeada com animação."""
        position_lower = position_str.lower()

        # Comando especial: trocar de monitor
        if "outro monitor" in position_lower or "próximo monitor" in position_lower or "proximo monitor" in position_lower:
            self._move_to_next_monitor()
            return

        anchor = None
        for key, val in POSITION_MAP.items():
            if key in position_lower:
                anchor = val
                break

        if not anchor:
            return

        screen = self.app.screenAt(self.geometry().center()) or self.app.primaryScreen()
        self._animate_to_position(anchor, screen.geometry())

    def _move_to_next_monitor(self):
        """Move a janela para o próximo monitor disponível."""
        screens = self.app.screens()
        if len(screens) < 2:
            return

        current_screen = self.app.screenAt(self.geometry().center())
        current_index = screens.index(current_screen) if current_screen in screens else 0
        next_index = (current_index + 1) % len(screens)
        next_screen = screens[next_index]

        # Move para o canto inferior direito do próximo monitor
        self._animate_to_position(("right", "bottom"), next_screen.geometry())

    def _animate_to_position(self, anchor: tuple, screen_geo):
        """Anima a janela para uma posição âncora dentro da geometria de tela fornecida."""
        margin = 20
        w, h = self.width(), self.height()
        h_anchor, v_anchor = anchor

        if h_anchor == "left":
            x = screen_geo.left() + margin
        elif h_anchor == "right":
            x = screen_geo.right() - w - margin
        else:
            x = screen_geo.center().x() - w // 2

        if v_anchor == "top":
            y = screen_geo.top() + margin
        elif v_anchor == "bottom":
            y = screen_geo.bottom() - h - margin - 40
        else:
            y = screen_geo.center().y() - h // 2

        self._anim = QPropertyAnimation(self, b"pos")
        self._anim.setDuration(400)
        self._anim.setStartValue(self.pos())
        self._anim.setEndValue(QPoint(x, y))
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

    def move_to_position(self, position_str: str):
        """Chamado pelo pipeline quando detecta comando de mover janela."""
        self.signals.move_to.emit(position_str)

    # ─── EXECUÇÃO ─────────────────────────────────────────

    def run(self) -> int:
        self.show()
        return self.app.exec()
