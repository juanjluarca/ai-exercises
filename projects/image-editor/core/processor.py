"""
core/processor.py
Pipeline de procesamiento de imagen con OpenCV.
Cada función recibe un ndarray BGR y devuelve uno nuevo.
"""

import numpy as np
import cv2


def apply_rgb_delta(img: np.ndarray, dr: int, dg: int, db: int) -> np.ndarray:
    """Suma deltas a cada canal BGR. Resultado clipeado a [0, 255]."""
    out = img.astype(np.int16)
    out[:, :, 2] = np.clip(out[:, :, 2] + dr, 0, 255)  # R → canal 2 (BGR)
    out[:, :, 1] = np.clip(out[:, :, 1] + dg, 0, 255)
    out[:, :, 0] = np.clip(out[:, :, 0] + db, 0, 255)
    return out.astype(np.uint8)


def apply_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Aplica desenfoque gaussiano. radius=0 → sin cambio."""
    if radius <= 0:
        return img
    k = 2 * radius + 1
    return cv2.GaussianBlur(img, (k, k), -1)


def apply_rotation(img: np.ndarray, angle: int) -> np.ndarray:
    """Rota la imagen `angle` grados sobre su propio centro."""
    if angle == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_sobel(
    img: np.ndarray,
    ksize_x: int,
    ksize_y: int,
    mix: float,
) -> np.ndarray:
    """
    Calcula bordes Sobel y los mezcla con la imagen original.

    Args:
        ksize_x: tamaño de kernel en X (0 = desactivado).
        ksize_y: tamaño de kernel en Y (0 = desactivado).
        mix:     proporción del resultado Sobel (0.0–1.0).
    """
    if (ksize_x <= 0 and ksize_y <= 0) or mix <= 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    combined = np.zeros_like(gray, dtype=np.float64)

    if ksize_x > 0:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=2 * ksize_x + 1)
        combined += np.abs(sx)
    if ksize_y > 0:
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=2 * ksize_y + 1)
        combined += np.abs(sy)

    normalized = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sobel_bgr  = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 1.0 - mix, sobel_bgr, mix, 0)


def draw_selector(
    img: np.ndarray,
    shape: str,
    x_pct: int,
    y_pct: int,
    size_pct: int,
    thickness: int,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    """
    Dibuja un selector (cuadrado o círculo) sobre la imagen.

    Args:
        shape:     'square' | 'circle' | 'none'
        x_pct:     posición X como porcentaje del ancho (0–100).
        y_pct:     posición Y como porcentaje del alto (0–100).
        size_pct:  tamaño como porcentaje del lado menor (0–100).
        thickness: grosor del trazo en píxeles.
        color_rgb: color en formato (R, G, B).
    """
    if shape == "none":
        return img

    h, w = img.shape[:2]
    cx    = int(x_pct / 100 * w)
    cy    = int(y_pct / 100 * h)
    size  = int(size_pct / 100 * min(w, h))
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])   # RGB → BGR

    out = img.copy()

    if shape == "square":
        half = size // 2
        pt1  = (max(0, cx - half), max(0, cy - half))
        pt2  = (min(w - 1, cx + half), min(h - 1, cy + half))
        cv2.rectangle(out, pt1, pt2, color_bgr, thickness)

    elif shape == "circle":
        cv2.circle(out, (cx, cy), size // 2, color_bgr, thickness)

    return out


def process(img: np.ndarray, params: dict) -> np.ndarray:
    """
    Ejecuta el pipeline completo en orden:
      RGB → Blur → Rotación → Sobel → Selector

    Args:
        img:    imagen original BGR.
        params: diccionario con todas las opciones de edición.

    Returns:
        Imagen procesada BGR.
    """
    out = apply_rgb_delta(img, *params["rgb_delta"])
    out = apply_blur(out, params["blur_radius"])
    out = apply_rotation(out, params["rotation_angle"])
    out = apply_sobel(out, *params["sobel_params"])
    out = draw_selector(
        out,
        shape      = params["selector_shape"],
        color_rgb  = params["selector_color"],
        **{k: params["selector_geometry"][k]
           for k in ("x_pct", "y_pct", "size_pct", "thickness")},
    )
    return out
