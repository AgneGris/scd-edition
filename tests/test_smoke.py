import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_constructs_headlessly():
    from PySide6.QtWidgets import QApplication

    import scd_app
    from scd_app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert scd_app is not None
    assert window.windowTitle() == "SCD - EMG Decomposition & Edition"
    assert window.tabs.count() == 4

    window.close()
    app.processEvents()


def test_help_menu_exposes_support_diagnostics(tmp_path):
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    help_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text() == "&Help"
    )

    assert help_menu is not None
    assert "Open &Log Folder" in [action.text() for action in help_menu.actions()]
    assert "&Copy Diagnostics" in [action.text() for action in help_menu.actions()]

    log_path = tmp_path / "scd-edition.log"
    with (
        patch("scd_app.gui.main_window.configure_logging", return_value=log_path),
        patch(
            "scd_app.gui.main_window.runtime_diagnostics",
            return_value="test diagnostics",
        ),
    ):
        window._copy_diagnostics()

    assert QApplication.clipboard().text() == "test diagnostics"

    window.close()
    app.processEvents()


def test_config_cards_reemit_parameterless_change_signal():
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs.config_tab import AuxChannelCard, GridCard

    app = QApplication.instance() or QApplication([])

    grid = GridCard(1, "#4a9eff")
    grid_changes = []
    grid.changed.connect(lambda: grid_changes.append(True))
    grid.name_edit.setText("Biceps")
    grid.muscle_edit.setText("Biceps brachii")
    if grid.config_combo.count() > 1:
        grid.config_combo.setCurrentIndex(1)
    assert len(grid_changes) >= 2

    aux = AuxChannelCard(1)
    aux_changes = []
    aux.changed.connect(lambda: aux_changes.append(True))
    aux.name_edit.setText("Force")
    aux.type_combo.setCurrentIndex(1)
    aux.start_spin.setValue(1)
    aux.end_spin.setValue(2)
    aux.field_edit.setText("signal.force")
    aux.unit_edit.setText("N")
    aux.mvc_edit.setText("100")
    assert len(aux_changes) >= 7

    grid.close()
    aux.close()
    app.processEvents()


def test_recording_controls_present_one_clear_selection_workflow():
    from PySide6.QtWidgets import QApplication

    from scd_app.gui.tabs.config_tab import ConfigTab

    app = QApplication.instance() or QApplication([])
    tab = ConfigTab()

    assert tab.choose_recording_btn.text() == "Choose recording…"
    assert tab.choose_batch_btn.text() == "Choose batch…"
    assert tab.inspect_arrays_btn.text() == "Inspect arrays…"
    assert not tab.inspect_arrays_btn.isEnabled()

    tab.close()
    app.processEvents()


def test_copy_decomposition_configuration_confirms_updated_grids():
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QLineEdit,
        QMessageBox,
    )

    from scd_app.gui.tabs.decomposition_tab import DecompositionTab

    app = QApplication.instance() or QApplication([])

    def parameter_widgets(value: str):
        notch_filter = QComboBox()
        notch_filter.addItems(["None", "50"])
        peel_off = QComboBox()
        peel_off.addItems(["False", "True"])
        muap_window = QDoubleSpinBox()
        return {
            "sil_threshold": QLineEdit(value),
            "extension_factor": QLineEdit(value),
            "highpass_hz": QLineEdit(value),
            "lowpass_hz": QLineEdit(value),
            "notch_filter": notch_filter,
            "notch_harmonics": QCheckBox(),
            "peel_off": peel_off,
            "muap_window_ms": muap_window,
        }

    tab = DecompositionTab()
    source = parameter_widgets("42")
    target = parameter_widgets("1")
    source["notch_filter"].setCurrentText("50")
    source["notch_harmonics"].setChecked(True)
    source["peel_off"].setCurrentText("True")
    source["muap_window_ms"].setValue(25)
    tab.param_widgets = {"Grid 1": source, "Grid 2": target}

    with patch.object(QMessageBox, "information") as information:
        tab._copy_params_to_all("Grid 1")

    assert target["sil_threshold"].text() == "42"
    assert target["notch_filter"].currentText() == "50"
    assert target["notch_harmonics"].isChecked()
    assert target["peel_off"].currentText() == "True"
    assert target["muap_window_ms"].value() == 25
    information.assert_called_once_with(
        tab,
        "Configuration Applied to All Grids",
        'The decomposition configuration from "Grid 1" was copied to 1 other grid.',
    )

    tab.close()
    app.processEvents()
