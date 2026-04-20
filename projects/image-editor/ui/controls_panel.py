from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea,
    QGroupBox, QHBoxLayout, QLabel,
    QRadioButton, QButtonGroup,
)
from PyQt6.QtCore import pyqtSignal

from ui.widgets import labeled_slider
from ui.theme import PANEL, BORDER


class ControlsPanel(QWidget):

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(310)
        self.setStyleSheet(f"background:{PANEL}; border-right:1px solid {BORDER};")
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        inner = QWidget()
        inner.setStyleSheet(f"background:{PANEL};")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        self._add_rgb_group(lay)
        self._add_blur_group(lay)
        self._add_sobel_group(lay)
        self._add_rotation_group(lay)
        self._add_selector_group(lay)
        lay.addStretch()

        scroll.setWidget(inner)
        outer.addWidget(scroll)


    def _add_rgb_group(self, parent_layout: QVBoxLayout):
        grp = QGroupBox("CANALES RGB")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        box_r, self.sl_r, _ = labeled_slider("Rojo",  -255, 255, 0)
        box_g, self.sl_g, _ = labeled_slider("Verde", -255, 255, 0)
        box_b, self.sl_b, _ = labeled_slider("Azul",  -255, 255, 0)

        for sl in (self.sl_r, self.sl_g, self.sl_b):
            sl.valueChanged.connect(self.changed)

        lay.addWidget(box_r)
        lay.addWidget(box_g)
        lay.addWidget(box_b)
        parent_layout.addWidget(grp)

    def _add_blur_group(self, parent_layout: QVBoxLayout):
        grp = QGroupBox("DESENFOQUE GAUSSIANO")
        lay = QVBoxLayout(grp)

        box, self.sl_blur, _ = labeled_slider("Intensidad", 0, 30, 0)
        self.sl_blur.valueChanged.connect(self.changed)
        lay.addWidget(box)
        parent_layout.addWidget(grp)

    def _add_sobel_group(self, parent_layout: QVBoxLayout):
        grp = QGroupBox("DETECCIÓN DE BORDES  (SOBEL)")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

        box_x, self.sl_sobel_x,   _ = labeled_slider("Eje X  (ksize)",      0, 10,  0)
        box_y, self.sl_sobel_y,   _ = labeled_slider("Eje Y  (ksize)",      0, 10,  0)
        box_m, self.sl_sobel_mix, _ = labeled_slider("Mezcla con original", 0, 100, 0)

        for sl in (self.sl_sobel_x, self.sl_sobel_y, self.sl_sobel_mix):
            sl.valueChanged.connect(self.changed)

        lay.addWidget(box_x)
        lay.addWidget(box_y)
        lay.addWidget(box_m)
        parent_layout.addWidget(grp)

    def _add_rotation_group(self, parent_layout: QVBoxLayout):
        grp = QGroupBox("ROTACIÓN")
        lay = QVBoxLayout(grp)

        box, self.sl_rot, _ = labeled_slider("Ángulo  (°)", -180, 180, 0)
        self.sl_rot.valueChanged.connect(self.changed)
        lay.addWidget(box)
        parent_layout.addWidget(grp)

    def _add_selector_group(self, parent_layout: QVBoxLayout):
        grp = QGroupBox("SELECTOR / MARCO")
        lay = QVBoxLayout(grp)
        lay.setSpacing(6)

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
            rb.toggled.connect(self.changed)
        lay.addLayout(shape_row)

        # Posición, tamaño y grosor
        box_x,  self.sl_sel_x,     _ = labeled_slider("Posición X", 0, 100, 50)
        box_y,  self.sl_sel_y,     _ = labeled_slider("Posición Y", 0, 100, 50)
        box_s,  self.sl_sel_size,  _ = labeled_slider("Tamaño",     5,  80, 20)
        box_t,  self.sl_sel_thick, _ = labeled_slider("Grosor",     1,  12,  2)

        for sl in (self.sl_sel_x, self.sl_sel_y, self.sl_sel_size, self.sl_sel_thick):
            sl.valueChanged.connect(self.changed)

        lay.addWidget(box_x)
        lay.addWidget(box_y)
        lay.addWidget(box_s)
        lay.addWidget(box_t)

        # Color del selector
        color_title = QLabel("Color del selector")
        color_title.setObjectName("SliderLabel")
        lay.addWidget(color_title)

        box_cr, self.sl_sel_r, _ = labeled_slider("R", 0, 255, 255)
        box_cg, self.sl_sel_g, _ = labeled_slider("G", 0, 255,   0)
        box_cb, self.sl_sel_b, _ = labeled_slider("B", 0, 255,   0)

        for sl in (self.sl_sel_r, self.sl_sel_g, self.sl_sel_b):
            sl.valueChanged.connect(self.changed)

        lay.addWidget(box_cr)
        lay.addWidget(box_cg)
        lay.addWidget(box_cb)

        parent_layout.addWidget(grp)


    @property
    def rgb_delta(self) -> tuple[int, int, int]:
        """Retorna (ΔR, ΔG, ΔB)."""
        return self.sl_r.value(), self.sl_g.value(), self.sl_b.value()

    @property
    def blur_radius(self) -> int:
        return self.sl_blur.value()

    @property
    def sobel_params(self) -> tuple[int, int, float]:
        """Retorna (ksize_x, ksize_y, mix 0–1)."""
        return (
            self.sl_sobel_x.value(),
            self.sl_sobel_y.value(),
            self.sl_sobel_mix.value() / 100.0,
        )

    @property
    def rotation_angle(self) -> int:
        return self.sl_rot.value()

    @property
    def selector_shape(self) -> str:
        """'none' | 'square' | 'circle'"""
        if self.rb_square.isChecked():
            return "square"
        if self.rb_circle.isChecked():
            return "circle"
        return "none"

    @property
    def selector_params(self) -> dict:
        return {
            "x":     self.sl_sel_x.value(),
            "y":     self.sl_sel_y.value(),
            "size":  self.sl_sel_size.value(),
            "thick": self.sl_sel_thick.value(),
            "color": (
                self.sl_sel_r.value(),
                self.sl_sel_g.value(),
                self.sl_sel_b.value(),
            ),
        }
