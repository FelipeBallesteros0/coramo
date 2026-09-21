# src/coramo_brain/coramo_brain/core/state.py
"""Maquina de estados del robot. Pura, sin ROS: se prueba sola."""
from __future__ import annotations


class Maquina:
    def __init__(self):
        self.estado = "IDLE"

    def aplicar(self, evento: str, detalle: str = "") -> str:
        if evento == "estop":
            self.estado = "STOPPED"
            return self.estado
        if evento == "rearm":
            self.estado = "IDLE"
            return self.estado
        if self.estado == "STOPPED":
            return self.estado

        if evento == "speech_start":
            self.estado = "LISTENING"
        elif evento == "speech_end":
            self.estado = "THINKING"
        elif evento == "tool_chosen":
            self.estado = "SPEAKING" if detalle == "responder" else "ACTING"
        elif evento in ("command_sent", "command_rejected"):
            self.estado = "IDLE"
        elif evento == "tts_first_audio":
            self.estado = "SPEAKING"
        elif evento == "tts_done":
            self.estado = "IDLE"
        return self.estado
