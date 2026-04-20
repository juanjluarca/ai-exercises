import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog,
)
from PyQt6.QtCore import Qt

from ui.theme import STYLESHEET, PANEL, BORDER, DARK
from ui.controls_panel import ControlsPanel
from core import processor, io_utils


class MainWindow(QMainWindow):
    """Ventana principal de la aplicación Image Editor."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMAGE EDITOR")
        self.setMinimumSize(1280, 780)
        self.setStyleSheet(STYLESHEET)

        self._original: np.ndarray | None = None
        self._current:  np.ndarray | None = None

        self._build_ui()

    # ── Construcción ──────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        root_lay.addWidget(self._build_header())
        root_lay.addWidget(self._build_body(), stretch=1)
        root_lay.addWidget(self._build_status_bar())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"background:{PANEL}; border-bottom:1px solid {BORDER};")

        lay = QHBoxLayout(header)
        lay.setContentsMargins(20, 0, 20, 0)

        title = QLabel("IMAGE EDITOR")
        title.setObjectName("Title")
        subtitle = QLabel("Simulator")
        subtitle.setObjectName("Subtitle")

        self.load_btn = QPushButton("  ⬆  Cargar imagen")
        self.load_btn.setObjectName("LoadBtn")
        self.load_btn.setFixedHeight(40)
        self.load_btn.clicked.connect(self._load_image)

        self.export_btn = QPushButton("  ⬇  Exportar")
        self.export_btn.setObjectName("ExportBtn")
        self.export_btn.setFixedHeight(40)
        self.export_btn.clicked.connect(self._export_image)

        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addStretch()
        lay.addWidget(self.load_btn)
        lay.addSpacing(8)
        lay.addWidget(self.export_btn)

        return header

    def _build_body(self) -> QWidget:
        body = QWidget()
        lay  = QHBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Panel de controles
        self.controls = ControlsPanel()
        self.controls.changed.connect(self._refresh)
        lay.addWidget(self.controls)

        # Área de imagen
        img_area = QWidget()
        img_area.setStyleSheet(f"background:{DARK};")
        img_lay  = QVBoxLayout(img_area)
        img_lay.setContentsMargins(20, 20, 20, 20)

        self.img_label = QLabel(
            "Carga una imagen para comenzar  ·  JPG, PNG, BMP, TIFF"
        )
        self.img_label.setObjectName("ImageDisplay")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(400, 400)
        img_lay.addWidget(self.img_label)

        lay.addWidget(img_area, stretch=1)
        return body

    def _build_status_bar(self) -> QLabel:
        self.status_lbl = QLabel("Sin imagen cargada")
        self.status_lbl.setObjectName("StatusBar")
        return self.status_lbl


    def _build_params(self) -> dict:
        """Recoge el estado actual del panel de controles en un dict."""
        c = self.controls
        sel = c.selector_params
        return {
            "rgb_delta":       c.rgb_delta,
            "blur_radius":     c.blur_radius,
            "rotation_angle":  c.rotation_angle,
            "sobel_params":    c.sobel_params,
            "selector_shape":  c.selector_shape,
            "selector_color":  sel["color"],
            "selector_geometry": {
                "x_pct":     sel["x"],
                "y_pct":     sel["y"],
                "size_pct":  sel["size"],
                "thickness": sel["thick"],
            },
        }

    def _refresh(self):
        """Procesa la imagen original con los parámetros actuales y actualiza la vista."""
        if self._original is None:
            return

        params = self._build_params()
        self._current = processor.process(self._original, params)
        self._display(self._current)
        self._update_status()

    def _display(self, img: np.ndarray):
        """Escala la imagen al área disponible y la muestra."""
        size    = self.img_label.size()
        preview = io_utils.fit_to_size(img, size.width(), size.height())
        self.img_label.setPixmap(io_utils.cv_to_qpixmap(preview))

    def _update_status(self):
        if self._original is None:
            return
        h, w = self._original.shape[:2]
        parts = [f"Resolución: {w} × {h} px"]

        dr, dg, db = self.controls.rgb_delta
        if any((dr, dg, db)):
            parts.append(f"ΔR={dr:+d}  ΔG={dg:+d}  ΔB={db:+d}")
        if self.controls.blur_radius > 0:
            parts.append(f"Blur: d={self.controls.blur_radius}")
        if self.controls.rotation_angle != 0:
            parts.append(f"Rotación: {self.controls.rotation_angle}°")

        self.status_lbl.setText("  ·  ".join(parts))

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir imagen", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp)",
        )
        if not path:
            return

        img = io_utils.load_image(path)
        if img is None:
            self.status_lbl.setText("  ✗  No se pudo cargar la imagen.")
            return

        self._original = img
        self.img_label.setText("")
        self._refresh()
        self.status_lbl.setText(f"  ✓  Cargada: {path}")

    def _export_image(self):
        if self._current is None:
            self.status_lbl.setText("  ✗  No hay imagen para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar imagen", "imagen_editada.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not path:
            return

        if io_utils.save_image(self._current, path):
            self.status_lbl.setText(f"  ✓  Exportada: {path}")
        else:
            self.status_lbl.setText("  ✗  Error al exportar.")

    # ── Eventos ───────────────────────────────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current is not None:
            self._display(self._current)
