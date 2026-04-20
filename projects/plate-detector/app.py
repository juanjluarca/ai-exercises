import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QMessageBox, QSizePolicy, QComboBox,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QRunnable, QThreadPool, QTimer, QObject,
    pyqtSlot,
)
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont, QFontDatabase

from detector import detect, _get_reader

# ── Environment / camera helpers ─────────────────────────────────────────────

def _running_in_wsl() -> bool:
    """Return True when the process is running inside WSL."""
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False

_WSL_CAMERA_MSG = (
    "Camera access is not available by default in WSL.\n\n"
    "To enable it, attach your camera from a Windows PowerShell "
    "<b>Administrator</b> prompt:\n\n"
    "  1. Install usbipd:  <tt>winget install usbipd</tt>\n"
    "  2. List devices:    <tt>usbipd list</tt>\n"
    "  3. Attach camera:   <tt>usbipd attach --wsl --busid &lt;BUSID&gt;</tt>\n\n"
    "Then verify in WSL:  <tt>ls /dev/video*</tt>\n\n"
    "You need to re-attach after every Windows or WSL restart."
)

def _probe_cameras(max_index: int = 8) -> list[tuple[int, str]]:
    """
    Return a list of (index, label) for every camera index that can deliver
    a frame.  Even-numbered V4L2 nodes are capture devices; odd-numbered ones
    are metadata-only and are skipped to avoid the ~1 s timeout per miss.
    Falls back to probing every index when not on Linux.
    """
    candidates = (
        range(0, max_index, 2)   # Linux/WSL: skip metadata nodes (1, 3, 5…)
        if sys.platform.startswith("linux")
        else range(max_index)
    )
    found: list[tuple[int, str]] = []
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append((idx, f"Camera {idx}  (/dev/video{idx})"))
        cap.release()
    return found

# ── Constants ─────────────────────────────────────────────────────────────────

LOG_FILE      = Path(__file__).parent / "detected_plates.txt"
TEMP_CAPTURE  = Path(__file__).parent / "temp_capture.jpg"

CANVAS_W = 500
CANVAS_H = 400

# ── Global stylesheet ─────────────────────────────────────────────────────────

STYLESHEET = """
/* Base */
QWidget {
    background-color: #1a2535;
    color: #e8f0f7;
    font-family: 'Inter', 'Segoe UI', 'Ubuntu', 'Noto Sans', Arial, sans-serif;
    font-size: 13px;
}

/* Named panels */
QWidget#panel {
    background-color: #1e2d40;
    border-radius: 8px;
    border: 1px solid #2e4a63;
}

/* Buttons — default accent */
QPushButton {
    background-color: #4a90a4;
    color: #e8f0f7;
    border: none;
    border-radius: 5px;
    padding: 8px 18px;
    font-weight: bold;
    min-height: 32px;
}
QPushButton:hover    { background-color: #5ba3b8; }
QPushButton:pressed  { background-color: #3a7a8e; }
QPushButton:disabled { background-color: #2e4a63; color: #8da4be; }

/* Danger variant (Clear Log) */
QPushButton#danger          { background-color: #c0392b; }
QPushButton#danger:hover    { background-color: #e74c3c; }
QPushButton#danger:pressed  { background-color: #a93226; }
QPushButton#danger:disabled { background-color: #2e4a63; color: #8da4be; }

/* Table */
QTableWidget {
    background-color: #1e2d40;
    alternate-background-color: #243447;
    border: 1px solid #2e4a63;
    gridline-color: #2e4a63;
    color: #e8f0f7;
    selection-background-color: #4a90a4;
    selection-color: #e8f0f7;
    outline: none;
}
QTableWidget::item { padding: 4px 8px; }

QHeaderView::section {
    background-color: #4a90a4;
    color: #e8f0f7;
    padding: 7px 8px;
    border: none;
    border-right: 1px solid #2e4a63;
    font-weight: bold;
}
QHeaderView::section:last { border-right: none; }

/* Scrollbars */
QScrollBar:vertical {
    background: #1e2d40;
    width: 8px;
    border: none;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #4a90a4;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical { height: 0; }

QScrollBar:horizontal {
    background: #1e2d40;
    height: 8px;
    border: none;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #4a90a4;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal { width: 0; }

/* Dialogs / message boxes */
QDialog    { background-color: #1a2535; }
QMessageBox { background-color: #1a2535; }
QMessageBox QLabel { color: #e8f0f7; }
QMessageBox QPushButton { min-width: 80px; }
"""

# ── Image helper ──────────────────────────────────────────────────────────────

def _np_to_pixmap(img_bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap scaled to fit (max_w × max_h)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        max_w, max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

# ── Background workers ────────────────────────────────────────────────────────

class DetectorInitWorker(QThread):
    """Warms up the EasyOCR reader on a background thread at startup."""
    ready  = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self):
        try:
            _get_reader()
            self.ready.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class DetectionSignals(QObject):
    """Signal carrier for DetectionWorker (QRunnable cannot own signals)."""
    finished = pyqtSignal(object, object)   # (np.ndarray | None, str | None)
    error    = pyqtSignal(str)


class DetectionWorker(QRunnable):
    """Runs detect() off the main thread via QThreadPool."""

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path
        self.signals    = DetectionSignals()

    @pyqtSlot()
    def run(self):
        try:
            annotated, text = detect(self.image_path)
            self.signals.finished.emit(annotated, text)
        except Exception as exc:
            self.signals.error.emit(str(exc))

# ── Camera dialog ─────────────────────────────────────────────────────────────

class CameraDialog(QDialog):
    """
    Live webcam preview with camera selection and single-frame capture.

    On Linux/WSL, _probe_cameras() skips metadata-only V4L2 nodes (odd indices)
    so only true capture devices appear in the dropdown.
    """

    captured = pyqtSignal(str)   # absolute path to saved temp_capture.jpg

    _CAM_W = 640
    _CAM_H = 480

    def __init__(self, cameras: list[tuple[int, str]], parent=None):
        """
        Parameters
        ----------
        cameras : list of (index, label) returned by _probe_cameras()
        """
        super().__init__(parent)
        self.setWindowTitle("Camera Capture")
        self.setModal(True)

        self._cameras    = cameras   # [(0, "Camera 0 (/dev/video0)"), …]
        self._cap        = None
        self._last_frame = None
        self._timer      = QTimer(self)
        self._timer.timeout.connect(self._update_frame)

        self._setup_ui()
        QTimer.singleShot(120, self._start_camera)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Camera selector (only shown when more than one camera is found)
        if len(self._cameras) > 1:
            sel_row = QHBoxLayout()
            sel_lbl = QLabel("Camera:")
            sel_lbl.setStyleSheet("border: none;")
            self._cam_combo = QComboBox()
            self._cam_combo.setStyleSheet(
                "QComboBox { background-color: #243447; border: 1px solid #2e4a63; "
                "border-radius: 4px; padding: 4px 8px; color: #e8f0f7; }"
                "QComboBox::drop-down { border: none; }"
                "QComboBox QAbstractItemView { background-color: #243447; "
                "color: #e8f0f7; selection-background-color: #4a90a4; }"
            )
            for _idx, label in self._cameras:
                self._cam_combo.addItem(label)
            self._cam_combo.currentIndexChanged.connect(self._switch_camera)
            sel_row.addWidget(sel_lbl)
            sel_row.addWidget(self._cam_combo, stretch=1)
            layout.addLayout(sel_row)
        else:
            self._cam_combo = None

        # Preview canvas
        self._canvas = QLabel()
        self._canvas.setFixedSize(self._CAM_W, self._CAM_H)
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setStyleSheet("background-color: #0d1b2a; color: #8da4be;")
        self._canvas.setText("Starting camera…")
        layout.addWidget(self._canvas)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_capture = QPushButton("Capture")
        self._btn_capture.setEnabled(False)
        self._btn_capture.clicked.connect(self._capture)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_capture)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self.adjustSize()

    def _current_cam_index(self) -> int:
        if self._cam_combo is not None:
            return self._cameras[self._cam_combo.currentIndex()][0]
        return self._cameras[0][0]

    def _start_camera(self):
        self._open_index(self._current_cam_index())

    def _open_index(self, idx: int):
        self._release_cap()
        self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            self._canvas.setText(f"Cannot open /dev/video{idx}")
            self._btn_capture.setEnabled(False)
            return
        self._btn_capture.setEnabled(True)
        self._canvas.setText("")
        if not self._timer.isActive():
            self._timer.start(33)   # ~30 fps

    def _switch_camera(self, combo_idx: int):
        hw_idx = self._cameras[combo_idx][0]
        self._open_index(hw_idx)

    def _update_frame(self):
        if not self._cap or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if ret:
            self._last_frame = frame
            self._canvas.setPixmap(
                _np_to_pixmap(frame, self._CAM_W, self._CAM_H)
            )

    def _capture(self):
        if self._last_frame is None:
            return
        cv2.imwrite(str(TEMP_CAPTURE), self._last_frame)
        self._release()
        self.captured.emit(str(TEMP_CAPTURE))
        self.accept()

    def _release_cap(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def _release(self):
        self._timer.stop()
        self._release_cap()

    def closeEvent(self, event):
        self._release()
        super().closeEvent(event)

    def reject(self):
        self._release()
        super().reject()

# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Guatemalan Plate Detector")
        self.setMinimumSize(1000, 600)

        self._current_image_path: str | None = None
        self._detector_ready = False
        self._row_counter    = 0

        self._build_ui()
        self._load_log_from_file()
        self._start_detector_init()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setSpacing(14)
        layout.setContentsMargins(14, 14, 14, 14)

        layout.addWidget(self._build_left_panel())
        layout.addWidget(self._build_right_panel(), stretch=1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setFixedWidth(560)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Title
        title = QLabel("Input Image")
        title.setStyleSheet(
            "color: #4a90a4; font-size: 14px; font-weight: bold; border: none;"
        )
        layout.addWidget(title)

        # ── Image canvas
        self._canvas = QLabel()
        self._canvas.setFixedSize(CANVAS_W, CANVAS_H)
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setStyleSheet(
            "background-color: #0d1b2a; border-radius: 6px; "
            "color: #8da4be; font-size: 14px; border: 1px solid #2e4a63;"
        )
        self._canvas.setText("No image loaded")
        layout.addWidget(self._canvas, alignment=Qt.AlignmentFlag.AlignHCenter)

        # ── Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_load   = QPushButton("Load Image")
        self._btn_camera = QPushButton("Open Camera")
        self._btn_detect = QPushButton("Detect Plate")
        self._btn_detect.setEnabled(False)

        self._btn_load.clicked.connect(self._load_image)
        # self._btn_camera.clicked.connect(self._open_camera)
        self._btn_detect.clicked.connect(self._run_detection)

        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_camera)
        btn_row.addWidget(self._btn_detect)
        layout.addLayout(btn_row)

        # ── Status label
        self._status = QLabel("Initializing detector…")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            "color: #8da4be; font-size: 12px; border: none;"
        )
        layout.addWidget(self._status)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setMinimumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Title
        title = QLabel("Detection Log")
        title.setStyleSheet(
            "color: #4a90a4; font-size: 14px; font-weight: bold; border: none;"
        )
        layout.addWidget(title)

        # ── Log table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["#", "Source", "Plate"])
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self._table.verticalHeader().setVisible(False)

        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 40)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(2, 120)

        layout.addWidget(self._table, stretch=1)

        # ── Bottom bar: clear button + log path
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        self._btn_clear = QPushButton("Clear Log")
        self._btn_clear.setObjectName("danger")
        self._btn_clear.clicked.connect(self._clear_log)
        bottom.addWidget(self._btn_clear)

        log_path_lbl = QLabel(str(LOG_FILE.resolve()))
        log_path_lbl.setStyleSheet(
            "color: #8da4be; font-size: 10px; border: none;"
        )
        log_path_lbl.setWordWrap(True)
        bottom.addWidget(log_path_lbl, stretch=1)

        layout.addLayout(bottom)
        return panel

    # ── Status helpers ────────────────────────────────────────────────────────

    def _set_status(self, msg: str, color: str = "#8da4be"):
        self._status.setStyleSheet(
            f"color: {color}; font-size: 12px; border: none;"
        )
        self._status.setText(msg)

    def _status_info(self, msg: str):  self._set_status(msg, "#8da4be")
    def _status_good(self, msg: str):  self._set_status(msg, "#4a90a4")
    def _status_warn(self, msg: str):  self._set_status(msg, "#e0a030")

    # ── Detector initialisation ───────────────────────────────────────────────

    def _start_detector_init(self):
        self._init_worker = DetectorInitWorker()
        self._init_worker.ready.connect(self._on_detector_ready)
        self._init_worker.failed.connect(self._on_detector_failed)
        self._init_worker.start()

    def _on_detector_ready(self):
        self._detector_ready = True
        self._refresh_detect_btn()
        self._status_good("Detector ready")

    def _on_detector_failed(self, msg: str):
        self._status_warn(f"Detector init failed: {msg}")

    def _refresh_detect_btn(self):
        """Enable Detect only when the model is loaded AND an image is present."""
        self._btn_detect.setEnabled(
            self._detector_ready and self._current_image_path is not None
        )

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.cwd()),
            "Images (*.jpg *.jpeg *.png);;All Files (*)",
        )
        if path:
            self._display_image(path)

    def _display_image(self, path: str):
        img = cv2.imread(path)
        if img is None:
            self._status_warn(f"Cannot read: {Path(path).name}")
            return
        self._current_image_path = path
        self._canvas.setPixmap(_np_to_pixmap(img, CANVAS_W, CANVAS_H))
        self._status_info(f"Image loaded: {Path(path).name}")
        self._refresh_detect_btn()

    # ── Camera ────────────────────────────────────────────────────────────────

    def _open_camera(self):
        self._status_info("Scanning for cameras…")
        QApplication.processEvents()   # let the status label repaint
        cameras = _probe_cameras()
        if not cameras:
            if _running_in_wsl():
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Camera unavailable in WSL")
                msg.setTextFormat(Qt.TextFormat.RichText)
                msg.setText(_WSL_CAMERA_MSG)
                msg.exec()
            else:
                QMessageBox.critical(
                    self, "Camera Error",
                    "No camera found or camera is in use.",
                )
            self._status_info("No camera available")
            return
        self._status_info("Ready")
        dlg = CameraDialog(cameras, self)
        dlg.captured.connect(self._display_image)
        dlg.exec()

    # ── Detection ─────────────────────────────────────────────────────────────

    def _run_detection(self):
        if not self._current_image_path:
            return
        self._set_ui_busy(True)
        worker = DetectionWorker(self._current_image_path)
        worker.signals.finished.connect(self._on_detection_done)
        worker.signals.error.connect(self._on_detection_error)
        QThreadPool.globalInstance().start(worker)

    def _set_ui_busy(self, busy: bool):
        self._btn_load.setEnabled(not busy)
        self._btn_camera.setEnabled(not busy)
        if busy:
            self._btn_detect.setEnabled(False)
            self._status_info("Detecting…")
        else:
            self._refresh_detect_btn()

    def _on_detection_done(self, annotated, text):
        self._set_ui_busy(False)

        if annotated is not None:
            self._canvas.setPixmap(_np_to_pixmap(annotated, CANVAS_W, CANVAS_H))

        if text:
            self._status_good(f"Plate found: {text}")
            self._append_entry(self._current_image_path, text)
        else:
            self._status_warn("No plate detected")

    def _on_detection_error(self, msg: str):
        self._set_ui_busy(False)
        self._status_warn(f"Detection error: {msg}")

    # ── Log persistence ───────────────────────────────────────────────────────

    def _append_entry(self, image_path: str, plate: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source    = Path(image_path).name
        line      = f"{timestamp} | {source} | {plate}"

        try:
            with LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            self._status_warn(f"Log write error: {exc}")

        self._add_table_row(source, plate)

    def _add_table_row(self, source: str, plate: str):
        self._row_counter += 1
        row = self._table.rowCount()
        self._table.insertRow(row)

        for col, text in enumerate([str(self._row_counter), source, plate]):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if col == 2:
                item.setForeground(QColor("#4a90a4"))
            self._table.setItem(row, col, item)

        self._table.scrollToBottom()

    def _load_log_from_file(self):
        if not LOG_FILE.exists():
            return
        try:
            lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" | ", 2)
            if len(parts) == 3:
                _ts, source, plate = parts
                self._add_table_row(source, plate)

    def _clear_log(self):
        reply = QMessageBox.question(
            self, "Clear Log",
            "Clear all detection history?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._table.setRowCount(0)
        self._row_counter = 0
        try:
            LOG_FILE.write_text("", encoding="utf-8")
        except OSError as exc:
            self._status_warn(f"Could not clear log file: {exc}")
            return
        self._status_info("Log cleared")

# ── Entry point ───────────────────────────────────────────────────────────────

def _setup_font(app: QApplication) -> None:
    """
    Pick the best available UI font on the current platform.

    Priority: Inter (if the .ttf is present next to app.py) → Segoe UI
    (Windows) → Ubuntu (WSL/Ubuntu) → Noto Sans → generic sans-serif.
    Using an explicit font + Fusion style makes the UI look identical on
    Windows and Linux.
    """
    # Optional: load Inter from a local file if the user placed it here
    inter_path = Path(__file__).parent / "Inter-Regular.ttf"
    if inter_path.exists():
        QFontDatabase.addApplicationFont(str(inter_path))

    # Build a cross-platform family preference list
    available = QFontDatabase.families()
    for family in ("Inter", "Segoe UI", "Ubuntu", "Noto Sans", "Helvetica Neue", "Arial"):
        if family in available:
            app.setFont(QFont(family, 10))
            return
    # Final fallback: just set the size on whatever the system default is
    f = app.font()
    f.setPointSize(10)
    app.setFont(f)


def main():
    QApplication.setStyle("Fusion")

    app = QApplication(sys.argv)

    _setup_font(app)
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
