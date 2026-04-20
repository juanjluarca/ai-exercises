from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt6.QtCore import Qt


def labeled_slider(
    label: str,
    min_val: int,
    max_val: int,
    default: int,
    tick: int = 1,
) -> tuple[QFrame, QSlider, QLabel]:
    """
    Crea un slider horizontal con etiqueta y valor numérico.

    Returns:
        (contenedor QFrame, QSlider, QLabel con el valor actual)
    """
    box = QFrame()
    box.setObjectName("SliderBox")

    layout = QVBoxLayout(box)
    layout.setContentsMargins(4, 2, 4, 2)
    layout.setSpacing(2)

    # Fila superior: nombre + valor
    header = QHBoxLayout()
    name_lbl = QLabel(label)
    name_lbl.setObjectName("SliderLabel")

    val_lbl = QLabel(str(default))
    val_lbl.setObjectName("SliderValue")
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)

    header.addWidget(name_lbl)
    header.addWidget(val_lbl)
    layout.addLayout(header)

    # Slider
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setObjectName("Slider")
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue(default)
    slider.setSingleStep(tick)
    slider.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
    layout.addWidget(slider)

    return box, slider, val_lbl
