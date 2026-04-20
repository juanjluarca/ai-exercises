"""
core/io_utils.py
Funciones de carga, exportación y conversión de imágenes.
"""

import numpy as np
import cv2
from PyQt6.QtGui import QImage, QPixmap


def load_image(path: str) -> np.ndarray | None:
    """
    Carga una imagen desde disco como ndarray BGR.
    Retorna None si la ruta es inválida o el archivo no es una imagen.
    """
    img = cv2.imread(path)
    return img  # None si falla


def save_image(img: np.ndarray, path: str) -> bool:
    """
    Guarda una imagen BGR en disco.
    Retorna True si tuvo éxito.
    """
    return cv2.imwrite(path, img)


def cv_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """Convierte un ndarray BGR de OpenCV a QPixmap para Qt."""
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def fit_to_size(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """
    Escala la imagen hacia abajo para caber en (max_w × max_h)
    manteniendo la relación de aspecto. No amplía imágenes pequeñas.
    """
    ih, iw = img.shape[:2]
    scale  = min(max_w / iw, max_h / ih, 1.0)
    if scale >= 1.0:
        return img
    new_w = int(iw * scale)
    new_h = int(ih * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
