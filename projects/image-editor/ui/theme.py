"""
ui/theme.py
Colores y hoja de estilos global de la aplicación.
"""

# ── Paleta de colores ──────────────────────────────────────────────────────────
DARK    = "#0d0f14"
PANEL   = "#13161e"
CARD    = "#1a1e2a"
ACCENT  = "#4f9cf9"
ACCENT2 = "#a78bfa"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"
SUCCESS = "#34d399"
BORDER  = "#252a38"

# ── Hoja de estilos QSS ────────────────────────────────────────────────────────
STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK};
    color: {TEXT};
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 12px;
}}

QLabel#Title {{
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2px;
}}

QLabel#Subtitle {{
    color: {MUTED};
    font-size: 11px;
}}

QGroupBox {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: #ffffff;
    font-size: 11px;
    letter-spacing: 1px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    margin-left: 8px;
    background-color: {CARD};
}}

QFrame#SliderBox {{
    background-color: transparent;
    border: none;
}}

QLabel#SliderLabel {{
    color: {TEXT};
    font-size: 11px;
    font-weight: 500;
}}

QLabel#SliderValue {{
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    min-width: 36px;
}}

QSlider#Slider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}

QSlider#Slider::handle:horizontal {{
    background: #7998F4;
    border: 2px solid {DARK};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider#Slider::sub-page:horizontal {{
    background: #7998F4;
    border-radius: 2px;
}}

QPushButton#LoadBtn {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2563eb, stop:1 {ACCENT});
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}}

QPushButton#LoadBtn:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #3b82f6, stop:1 #60a5fa);
}}

QPushButton#LoadBtn:pressed {{
    background: #1e40af;
}}

QPushButton#ExportBtn {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #065f46, stop:1 #059669);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}}

QPushButton#ExportBtn:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #047857, stop:1 {SUCCESS});
}}

QPushButton#ExportBtn:pressed {{
    background: #064e3b;
}}

QRadioButton {{
    color: {TEXT};
    font-size: 11px;
    spacing: 6px;
}}

QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 2px solid {MUTED};
    background: transparent;
}}

QRadioButton::indicator:checked {{
    background: #7998F4;
    border-color: #7998F4;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QLabel#ImageDisplay {{
    background-color: {PANEL};
    border: 2px dashed {BORDER};
    border-radius: 12px;
    color: {MUTED};
    font-size: 13px;
}}

QFrame#Divider {{
    background: {BORDER};
    max-height: 1px;
}}

QLabel#StatusBar {{
    color: {MUTED};
    font-size: 10px;
    padding: 4px 8px;
    background: {PANEL};
    border-top: 1px solid {BORDER};
}}
"""
