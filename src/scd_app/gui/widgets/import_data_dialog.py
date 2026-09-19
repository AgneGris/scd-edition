"""Guided selection of an EMG matrix from an inspected data file."""

from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
)

from scd_app.io.data_inspector import ArrayCandidate, RecordingInspection


class ImportDataDialog(QDialog):
    """Choose a numeric matrix and describe how its axes should be interpreted."""

    def __init__(
        self,
        inspection: RecordingInspection,
        *,
        default_sampling_rate: int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.inspection = inspection
        self.setWindowTitle("Import unfamiliar recording")
        self.setMinimumWidth(650)

        layout = QVBoxLayout(self)
        intro = QLabel(
            f"Choose the EMG matrix in {inspection.file_path.name}. "
            "Only numeric arrays are listed; no code from the file is executed."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.array_combo = QComboBox()
        for candidate in inspection.arrays:
            self.array_combo.addItem(candidate.display_name, candidate)
        suggested = inspection.suggested_array
        if suggested is not None:
            for index in range(self.array_combo.count()):
                if self.array_combo.itemData(index) == suggested:
                    self.array_combo.setCurrentIndex(index)
                    break
        form.addRow("EMG array:", self.array_combo)

        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["auto", "samples_first", "channels_first"])
        form.addRow("Orientation:", self.orientation_combo)

        self.sampling_rate_edit = QLineEdit()
        self.sampling_rate_edit.setValidator(QIntValidator(1, 1_000_000, self))
        detected = inspection.sampling_rate_hz
        initial_rate = int(round(detected)) if detected else default_sampling_rate
        self.sampling_rate_edit.setText(str(initial_rate) if initial_rate else "")
        self.sampling_rate_edit.setPlaceholderText("required")
        sampling_label = "Sampling rate (Hz):"
        if inspection.sampling_rate_source:
            sampling_label = (
                f"Sampling rate (Hz, from {inspection.sampling_rate_source}):"
            )
        form.addRow(sampling_label, self.sampling_rate_edit)
        layout.addLayout(form)

        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.array_combo.currentIndexChanged.connect(self._update_preview)
        self.orientation_combo.currentTextChanged.connect(self._update_preview)
        self._update_preview()

    @property
    def selected_array(self) -> ArrayCandidate | None:
        return self.array_combo.currentData()

    @property
    def selected_path(self) -> str:
        candidate = self.selected_array
        return candidate.path if candidate else ""

    @property
    def selected_orientation(self) -> str:
        return self.orientation_combo.currentText()

    @property
    def sampling_rate(self) -> int:
        return int(self.sampling_rate_edit.text())

    def _update_preview(self):
        candidate = self.selected_array
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if candidate is None:
            self.preview_label.setText("No numeric arrays were found.")
            ok_button.setEnabled(False)
            return
        if not candidate.is_matrix:
            self.preview_label.setText(
                "This array is not two-dimensional and cannot be used as EMG."
            )
            ok_button.setEnabled(False)
            return

        first, second = candidate.shape
        orientation = self.selected_orientation
        if orientation == "channels_first" or (
            orientation == "auto" and second > first
        ):
            samples, channels = second, first
        else:
            samples, channels = first, second
        self.preview_label.setText(
            f"Preview: {samples:,} samples × {channels:,} channels will be loaded."
        )
        ok_button.setEnabled(True)

    def _accept_if_valid(self):
        if self.selected_array is None or not self.selected_array.is_matrix:
            return
        try:
            sampling_rate = self.sampling_rate
        except ValueError:
            sampling_rate = 0
        if sampling_rate <= 0:
            QMessageBox.warning(
                self, "Sampling rate required", "Enter a positive sampling rate."
            )
            return
        self.accept()
