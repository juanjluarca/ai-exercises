"""
Image Editor — PyQt6 + OpenCV
Manipulación de imágenes en tiempo real
"""

import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QScrollArea,
    QFileDialog, QRadioButton, QButtonGroup, QGridLayout,
    QFrame, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QIcon


# ─────────────────────────── Helpers ───────────────────────────

def cv_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Convierte imagen OpenCV (BGR) a QPixmap."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def labeled_slider(label: str, min_val: int, max_val: int,
                   default: int, tick: int = 1) -> tuple:
    """Devuelve (QGroupBox contenedor, QSlider, QLabel valor)."""
    box = QFrame()
    box.setObjectName("SliderBox")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(4, 2, 4, 2)
    layout.setSpacing(2)

    header = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setObjectName("SliderLabel")
    val_lbl = QLabel(str(default))
    val_lbl.setObjectName("SliderValue")
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
    header.addWidget(lbl)
    header.addWidget(val_lbl)
    layout.addLayout(header)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue(default)
    slider.setSingleStep(tick)
    slider.setObjectName("Slider")
    layout.addWidget(slider)

    slider.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
    return box, slider, val_lbl


# ─────────────────────────── Ventana principal ───────────────────────────

DARK = "#0d0f14"
PANEL = "#13161e"
CARD  = "#1a1e2a"
ACCENT = "#4f9cf9"
ACCENT2 = "#a78bfa"
TEXT  = "#e2e8f0"
MUTED = "#64748b"
SUCCESS = "#34d399"
BORDER = "#252a38"


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
    color: {ACCENT};
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
    color: {ACCENT2};
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
    color: {ACCENT};
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
    background: {ACCENT};
    border: 2px solid {DARK};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider#Slider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {ACCENT2}, stop:1 {ACCENT});
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
    background: {ACCENT2};
    border-color: {ACCENT2};
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


class ImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMAGE EDITOR")
        self.setMinimumSize(1280, 780)

        self._original: np.ndarray | None = None   # imagen original BGR
        self._current: np.ndarray | None = None    # imagen procesada BGR

        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._update_image()

    # ────────────────── UI ──────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header ──
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"background:{PANEL}; border-bottom:1px solid {BORDER};")
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)

        title = QLabel("IMAGE EDITOR")
        title.setObjectName("Title")
        sub = QLabel("Simulator")
        sub.setObjectName("Subtitle")

        h_lay.addWidget(title)
        h_lay.addWidget(sub)
        h_lay.addStretch()

        self.load_btn = QPushButton("  ⬆  Cargar imagen")
        self.load_btn.setObjectName("LoadBtn")
        self.load_btn.setFixedHeight(40)
        self.load_btn.clicked.connect(self._load_image)

        self.export_btn = QPushButton("  ⬇  Exportar")
        self.export_btn.setObjectName("ExportBtn")
        self.export_btn.setFixedHeight(40)
        self.export_btn.clicked.connect(self._export_image)

        h_lay.addWidget(self.load_btn)
        h_lay.addSpacing(8)
        h_lay.addWidget(self.export_btn)

        root_layout.addWidget(header)

        # ── Body ──
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # Panel de controles (izquierda)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(310)
        controls_scroll.setStyleSheet(f"background:{PANEL}; border-right:1px solid {BORDER};")

        controls_widget = QWidget()
        controls_widget.setStyleSheet(f"background:{PANEL};")
        self._ctrl_layout = QVBoxLayout(controls_widget)
        self._ctrl_layout.setContentsMargins(12, 12, 12, 12)
        self._ctrl_layout.setSpacing(8)

        self._build_controls()
        self._ctrl_layout.addStretch()

        controls_scroll.setWidget(controls_widget)
        body_lay.addWidget(controls_scroll)

        # Área de imagen (derecha)
        img_area = QWidget()
        img_area.setStyleSheet(f"background:{DARK};")
        img_lay = QVBoxLayout(img_area)
        img_lay.setContentsMargins(20, 20, 20, 20)

        self.img_label = QLabel("Carga una imagen para comenzar  ·  JPG, PNG, BMP, TIFF")
        self.img_label.setObjectName("ImageDisplay")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(400, 400)
        img_lay.addWidget(self.img_label)

        body_lay.addWidget(img_area, stretch=1)
        root_layout.addWidget(body, stretch=1)

        # ── Status bar ──
        self.status_lbl = QLabel("Sin imagen cargada")
        self.status_lbl.setObjectName("StatusBar")
        root_layout.addWidget(self.status_lbl)

    def _build_controls(self):
        lay = self._ctrl_layout

        # ── Canales RGB ──
        grp_rgb = QGroupBox("CANALES RGB")
        g = QVBoxLayout(grp_rgb)
        g.setSpacing(6)

        box_r, self.sl_r, _ = labeled_slider("🔴  Rojo",   -255, 255, 0)
        box_g, self.sl_g, _ = labeled_slider("🟢  Verde",  -255, 255, 0)
        box_b, self.sl_b, _ = labeled_slider("🔵  Azul",   -255, 255, 0)

        for sl in (self.sl_r, self.sl_g, self.sl_b):
            sl.valueChanged.connect(self._update_image)

        g.addWidget(box_r)
        g.addWidget(box_g)
        g.addWidget(box_b)
        lay.addWidget(grp_rgb)

        # ── Blur ──
        grp_blur = QGroupBox("DESENFOQUE GAUSSIANO")
        gb = QVBoxLayout(grp_blur)
        box_blur, self.sl_blur, _ = labeled_slider("Intensidad", 0, 30, 0)
        self.sl_blur.valueChanged.connect(self._update_image)
        gb.addWidget(box_blur)
        lay.addWidget(grp_blur)

        # ── Bordes Sobel ──
        grp_sobel = QGroupBox("DETECCIÓN DE BORDES  (SOBEL)")
        gs = QVBoxLayout(grp_sobel)
        box_sx, self.sl_sobel_x, _ = labeled_slider("Eje X  (ksize)", 0, 10, 0)
        box_sy, self.sl_sobel_y, _ = labeled_slider("Eje Y  (ksize)", 0, 10, 0)
        box_sm, self.sl_sobel_mix, _ = labeled_slider("Mezcla con original", 0, 100, 0)
        self.sl_sobel_x.valueChanged.connect(self._update_image)
        self.sl_sobel_y.valueChanged.connect(self._update_image)
        self.sl_sobel_mix.valueChanged.connect(self._update_image)
        gs.addWidget(box_sx)
        gs.addWidget(box_sy)
        gs.addWidget(box_sm)
        lay.addWidget(grp_sobel)

        # ── Rotación ──
        grp_rot = QGroupBox("ROTACIÓN")
        gr = QVBoxLayout(grp_rot)
        box_rot, self.sl_rot, _ = labeled_slider("Ángulo  (°)", -180, 180, 0)
        self.sl_rot.valueChanged.connect(self._update_image)
        gr.addWidget(box_rot)
        lay.addWidget(grp_rot)

        # ── Selector / Marco ──
        grp_sel = QGroupBox("SELECTOR / MARCO")
        gsel = QVBoxLayout(grp_sel)
        gsel.setSpacing(6)

        # Forma
        shape_row = QHBoxLayout()
        self.rb_none   = QRadioButton("Sin marco")
        self.rb_square = QRadioButton("Cuadrado")
        self.rb_circle = QRadioButton("Círculo")
        self.rb_none.setChecked(True)

        self._shape_group = QButtonGroup()
        for rb in (self.rb_none, self.rb_square, self.rb_circle):
            self._shape_group.addButton(rb)
            shape_row.addWidget(rb)
            rb.toggled.connect(self._update_image)
        gsel.addLayout(shape_row)

        box_sx2, self.sl_sel_x, _ = labeled_slider("Posición X", 0, 100, 50)
        box_sy2, self.sl_sel_y, _ = labeled_slider("Posición Y", 0, 100, 50)
        box_ss,  self.sl_sel_size, _ = labeled_slider("Tamaño", 5, 80, 20)
        box_st,  self.sl_sel_thick, _ = labeled_slider("Grosor", 1, 12, 2)

        for sl in (self.sl_sel_x, self.sl_sel_y, self.sl_sel_size, self.sl_sel_thick):
            sl.valueChanged.connect(self._update_image)

        gsel.addWidget(box_sx2)
        gsel.addWidget(box_sy2)
        gsel.addWidget(box_ss)
        gsel.addWidget(box_st)

        # Color del selector
        color_lbl = QLabel("Color del selector")
        color_lbl.setObjectName("SliderLabel")
        gsel.addWidget(color_lbl)
        box_cr, self.sl_sel_r, _ = labeled_slider("R", 0, 255, 255)
        box_cg, self.sl_sel_g, _ = labeled_slider("G", 0, 255,   0)
        box_cb, self.sl_sel_b, _ = labeled_slider("B", 0, 255,   0)
        for sl in (self.sl_sel_r, self.sl_sel_g, self.sl_sel_b):
            sl.valueChanged.connect(self._update_image)
        gsel.addWidget(box_cr)
        gsel.addWidget(box_cg)
        gsel.addWidget(box_cb)

        lay.addWidget(grp_sel)

    # ────────────────── Image pipeline ──────────────────

    def _update_image(self):
        if self._original is None:
            return

        img = self._original.copy().astype(np.int16)

        # ── Canales RGB ──
        img[:, :, 2] = np.clip(img[:, :, 2] + self.sl_r.value(), 0, 255)  # R → ch2 en BGR
        img[:, :, 1] = np.clip(img[:, :, 1] + self.sl_g.value(), 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0] + self.sl_b.value(), 0, 255)
        img = img.astype(np.uint8)

        # ── Blur ──
        d = self.sl_blur.value()
        if d > 0:
            img = cv2.GaussianBlur(img, (2 * d + 1, 2 * d + 1), -1)

        # ── Rotación ──
        angle = self.sl_rot.value()
        if angle != 0:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        # ── Sobel ──
        kx = self.sl_sobel_x.value()
        ky = self.sl_sobel_y.value()
        mix = self.sl_sobel_mix.value() / 100.0

        if (kx > 0 or ky > 0) and mix > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ksize_x = 2 * kx + 1 if kx > 0 else 1
            ksize_y = 2 * ky + 1 if ky > 0 else 1

            sobel_combined = np.zeros_like(gray, dtype=np.float64)
            if kx > 0:
                sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize_x)
                sobel_combined += np.abs(sx)
            if ky > 0:
                sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize_y)
                sobel_combined += np.abs(sy)

            sobel_norm = cv2.normalize(sobel_combined, None, 0, 255,
                                       cv2.NORM_MINMAX).astype(np.uint8)
            sobel_bgr = cv2.cvtColor(sobel_norm, cv2.COLOR_GRAY2BGR)
            img = cv2.addWeighted(img, 1 - mix, sobel_bgr, mix, 0)

        # ── Selector / Marco ──
        if not self.rb_none.isChecked():
            h, w = img.shape[:2]
            cx = int(self.sl_sel_x.value() / 100 * w)
            cy = int(self.sl_sel_y.value() / 100 * h)
            size = int(self.sl_sel_size.value() / 100 * min(w, h))
            thick = self.sl_sel_thick.value()
            color_bgr = (self.sl_sel_b.value(),
                         self.sl_sel_g.value(),
                         self.sl_sel_r.value())

            if self.rb_square.isChecked():
                half = size // 2
                pt1 = (max(0, cx - half), max(0, cy - half))
                pt2 = (min(w - 1, cx + half), min(h - 1, cy + half))
                cv2.rectangle(img, pt1, pt2, color_bgr, thick)
            elif self.rb_circle.isChecked():
                radius = size // 2
                cv2.circle(img, (cx, cy), radius, color_bgr, thick)

        self._current = img

        # ── Mostrar ──
        display_size = self.img_label.size()
        ih, iw = img.shape[:2]
        scale = min(display_size.width() / iw, display_size.height() / ih, 1.0)
        new_w, new_h = int(iw * scale), int(ih * scale)
        if scale < 1.0:
            preview = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            preview = img

        self.img_label.setPixmap(cv_to_qpixmap(preview))
        self._update_status(ih, iw)

    def _update_status(self, h, w):
        parts = []
        if self._original is not None:
            parts.append(f"Resolución: {w} × {h} px")
        if self.sl_blur.value() > 0:
            parts.append(f"Blur: d={self.sl_blur.value()}")
        r, g, b = self.sl_r.value(), self.sl_g.value(), self.sl_b.value()
        if any((r, g, b)):
            parts.append(f"ΔR={r:+d}  ΔG={g:+d}  ΔB={b:+d}")
        if self.sl_rot.value() != 0:
            parts.append(f"Rotación: {self.sl_rot.value()}°")
        self.status_lbl.setText("  ·  ".join(parts) if parts else "Imagen cargada")

    # ────────────────── I/O ──────────────────

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.status_lbl.setText("  ✗  No se pudo cargar la imagen.")
            return
        self._original = img
        self.img_label.setText("")  # quitar placeholder
        self._update_image()
        self.status_lbl.setText(f"  ✓  Cargada: {path}")

    def _export_image(self):
        if self._current is None:
            self.status_lbl.setText("  ✗  No hay imagen para exportar.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar imagen", "imagen_editada.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if not path:
            return
        success = cv2.imwrite(path, self._current)
        if success:
            self.status_lbl.setText(f"  ✓  Exportada: {path}")
        else:
            self.status_lbl.setText("  ✗  Error al exportar.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image()


# ─────────────────────────── Entry point ───────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Image Editor")

    win = ImageEditor()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()