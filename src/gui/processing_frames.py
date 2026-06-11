import logging

# pylint: disable=E0611
from PySide2.QtWidgets import (
    QDialog,
    QLineEdit,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QCheckBox,
    QFormLayout,
)

from ..utils import clear_qt_layout, string_to_list, get_dict_key_index
from ..utils.widgets import VerticalContainerWidget
from ..constants import TIME_UNIT_FACTORS
from ..core.filtering import filter_risetime, filter_deadtime


debug_logger = logging.getLogger("ascam.debug")


class FilterFrame(QDialog):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.setWindowTitle("Filter")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.filter_options = ["Gaussian", "Chung-Kennedy", "Bessel"]

        self.create_widgets()
        self.freq_entry.setFocus()
        self.exec_()

    def create_widgets(self):
        row_one = QHBoxLayout()
        method_label = QLabel("Method")
        self.method_box = QComboBox()
        self.method_box.addItems(self.filter_options)
        self.method_box.currentIndexChanged.connect(self.choose_filter_method)
        row_one.addWidget(method_label)
        row_one.addWidget(self.method_box)

        row_two = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.ok_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        row_two.addWidget(ok_button)
        row_two.addWidget(cancel_button)

        self.layout.addLayout(row_one)
        self.choose_filter_method(0)
        self.layout.addLayout(row_two)

    def choose_filter_method(self, index):
        try:
            debug_logger.debug("deleting selection widgets")
            clear_qt_layout(self.selection_layout)
        except AttributeError:
            pass
        self.resize(self.sizeHint())
        # the resolution readout only exists for the Gaussian/Bessel branches;
        # drop the stale reference so update_resolution_readout() stays inert
        # for the Chung-Kennedy branch
        self.resolution_label = None
        if self.filter_options[index] == "Gaussian":
            debug_logger.debug("Creating gaussian input widgets")
            self.selection_layout = QFormLayout()
            self.freq_entry = QLineEdit("1000")
            self.selection_layout.addRow("Frequency [Hz]", self.freq_entry)
            self.add_resolution_readout()
        elif self.filter_options[index] == "Bessel":
            debug_logger.debug("Creating Bessel input widgets")
            self.selection_layout = QFormLayout()
            self.freq_entry = QLineEdit("1000")
            self.selection_layout.addRow("Cutoff [Hz]", self.freq_entry)
            self.pole_entry = QLineEdit("8")
            self.selection_layout.addRow("Poles", self.pole_entry)
            self.add_resolution_readout()
        else:
            debug_logger.debug("creating CK-filter input widgets")
            self.selection_layout = QFormLayout()
            self.width_entry = QLineEdit()
            self.selection_layout.addRow("Predictor Widths", self.width_entry)
            self.exponent_entry = QLineEdit()
            self.selection_layout.addRow("Weight Exponent", self.exponent_entry)
            self.window_entry = QLineEdit()
            self.selection_layout.addRow("Weight Window", self.window_entry)
            self.forward_entry = QLineEdit()
            self.selection_layout.addRow("Forward Pi", self.forward_entry)
            self.backward_entry = QLineEdit()
            self.selection_layout.addRow("Backward Pi", self.backward_entry)
        self.layout.insertLayout(1, self.selection_layout)
        self.resize(self.sizeHint())  # both resizes are necessary

    def add_resolution_readout(self):
        """Add a live rise-time/dead-time readout below the cutoff field and
        keep it updated as the user edits the cutoff frequency."""
        self.resolution_label = QLabel()
        self.resolution_label.setWordWrap(True)
        self.selection_layout.addRow(self.resolution_label)
        self.freq_entry.textChanged.connect(self.update_resolution_readout)
        self.update_resolution_readout()

    def update_resolution_readout(self):
        """Show the filter time resolution implied by the current cutoff.

        Uses T_r = 0.3321/f_c and T_d = 0.179/f_c (Colquhoun & Sigworth ch. 19),
        which describe both the Gaussian and the many-pole Bessel filter. The
        dead time T_d is the shortest event the filter can resolve at a
        half-amplitude threshold."""
        if not getattr(self, "resolution_label", None):
            return
        # fall back to a sensible default rate if no recording is loaded yet
        data = getattr(self.main, "data", None)
        sampling_rate = getattr(data, "sampling_rate", 4e4) or 4e4
        nyquist = sampling_rate / 2
        try:
            cutoff = float(self.freq_entry.text())
        except ValueError:
            self.resolution_label.setText("Enter a cutoff frequency in Hz")
            return
        if cutoff <= 0:
            self.resolution_label.setText("Cutoff frequency must be positive")
            return
        if cutoff >= nyquist:
            self.resolution_label.setText(
                f"Cutoff must be below the Nyquist frequency ({nyquist:g} Hz)"
            )
            return
        rise_ms = filter_risetime(cutoff) * 1e3
        dead_ms = filter_deadtime(cutoff) * 1e3
        dead_samples = filter_deadtime(cutoff) * sampling_rate
        message = (
            f"Rise time {rise_ms:.3g} ms · dead time {dead_ms:.3g} ms "
            f"({dead_samples:.1f} samples)\n"
            f"→ resolves events ≳ {dead_ms:.3g} ms"
        )
        if cutoff > sampling_rate / 5:
            message += (
                f"\n⚠ cutoff above 1/5 of the sampling rate "
                f"({sampling_rate / 5:g} Hz): consider sampling faster"
            )
        self.resolution_label.setText(message)

    def ok_clicked(self):
        filter_method = self.filter_options[self.method_box.currentIndex()]
        if filter_method == "Gaussian":
            self.main.data.gauss_filter_series(float(self.freq_entry.text()))
        elif filter_method == "Bessel":
            self.main.data.bessel_filter_series(
                float(self.freq_entry.text()), int(self.pole_entry.text())
            )
        elif filter_method == "Chung-Kennedy":
            self.main.data.CK_filter_series(
                window_lengths=[int(x) for x in self.window_entry.text().split()],
                weight_exponent=int(self.exponent_entry.text()),
                weight_window=int(self.window_entry.text()),
                apriori_f_weights=[int(x) for x in self.forward_entry.text().split()],
                apriori_b_weights=[int(x) for x in self.backward_entry.text().split()],
            )
        self.main.plot_frame.plot_all()
        self.main.ep_frame.update_combo_box()
        self.close()


class BaselineFrame(QDialog):
    def __init__(self, main):
        super().__init__()
        self.setWindowTitle("Baseline Correction")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(BaselineWidget(main, self))

        self.exec_()


class BaselineWidget(VerticalContainerWidget):
    def __init__(self, main, dialog):
        self.method_options = ["Polynomial", "Offset", "Running Percentile"]
        self.selection_options = ["Piezo", "Intervals"]

        super().__init__(main)

        self.main = main
        self.dialog = dialog

    def create_widgets(self):
        method_label = QLabel("Method")
        self.method_box = QComboBox()
        self.method_box.addItems(self.method_options)
        self.method_box.currentIndexChanged.connect(self.choose_correction_method)
        self.add_row(method_label, self.method_box)
        self.choose_correction_method(0)

        selection_label = QLabel("Selection")
        self.selection_box = QComboBox()
        self.selection_box.addItems(self.selection_options)
        self.selection_box.currentIndexChanged.connect(self.choose_selection_method)
        self.add_row(selection_label, self.selection_box)
        self.choose_selection_method(0)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.ok_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_clicked)
        self.add_row(ok_button, cancel_button)

    def choose_correction_method(self, index):
        # remove any method-specific widgets created for a previous method
        try:
            clear_qt_layout(self.selection_layout)
        except AttributeError:
            pass

        method = self.method_options[index]
        self.selection_layout = QFormLayout()
        if method == "Polynomial":
            debug_logger.debug("Creating polynomial input widgets")
            self.degree_entry = QLineEdit("1")
            self.selection_layout.addRow("Degree", self.degree_entry)
        elif method == "Running Percentile":
            debug_logger.debug("Creating running-percentile input widgets")
            window_layout = QHBoxLayout()
            self.window_entry = QLineEdit("0.5")
            self.window_unit_entry = QComboBox()
            self.window_unit_entry.addItems(list(TIME_UNIT_FACTORS.keys()))
            self.window_unit_entry.setCurrentIndex(
                get_dict_key_index(TIME_UNIT_FACTORS, "s")
            )
            window_layout.addWidget(self.window_entry)
            window_layout.addWidget(self.window_unit_entry)
            self.selection_layout.addRow("Window", window_layout)

            self.percentile_entry = QLineEdit("50")
            self.percentile_entry.setToolTip(
                "Percentile tracking the closed (baseline) level.\n"
                "50 (median) is unbiased when the channel is open less\n"
                "than half the time. Shift toward the closed side only at\n"
                "high open probability: up for inward (negative-going)\n"
                "openings, down for outward (positive-going)."
            )
            self.selection_layout.addRow("Percentile", self.percentile_entry)

            self.detect_jumps_check = QCheckBox("Detect baseline jumps (PELT)")
            self.detect_jumps_check.setChecked(False)
            self.selection_layout.addRow(self.detect_jumps_check)

            self.sensitivity_entry = QLineEdit("1.0")
            self.sensitivity_entry.setToolTip(
                "Scales the jump-detection penalty; higher values detect "
                "fewer jumps."
            )
            self.selection_layout.addRow("Jump sensitivity", self.sensitivity_entry)
        # "Offset" needs no extra widgets
        self.layout.insertLayout(1, self.selection_layout)

        # the Piezo/Intervals selection only applies to the fitted corrections;
        # the running percentile estimates the baseline from the whole trace
        self.set_selection_enabled(method != "Running Percentile")

    def set_selection_enabled(self, enabled):
        """Enable or disable the Piezo/Intervals selection row (it does not
        apply to the running-percentile method)."""
        if not hasattr(self, "selection_box"):
            # selection widgets are created after the first method call
            return
        self.selection_box.setEnabled(enabled)
        if hasattr(self, "method_layout"):
            for i in range(self.method_layout.count()):
                widget = self.method_layout.itemAt(i).widget()
                if widget is not None:
                    widget.setEnabled(enabled)

    def choose_selection_method(self, index):
        try:
            debug_logger.debug("deleting selection widgets")
            clear_qt_layout(self.method_layout)
        except AttributeError:
            pass
        self.method_layout = QHBoxLayout()
        if self.selection_options[index] == "Piezo":
            debug_logger.debug("creating piezo selection widgets")
            self.active_checkbox = QCheckBox("Active")
            self.active_checkbox.setToolTip(
                "If checked the baseline correction \n"
                "will be based on the times where \n"
                "the Piezo voltage is within a factor\n"
                "`deviation` of its"
                " maximum value.\n"
                "If unchecked it will be based on \n"
                "the times where the voltage is \n"
                "within a factor `deviation` of its \n"
                "minimum value."
            )
            self.active_checkbox.setChecked(False)
            self.method_layout.addWidget(self.active_checkbox)
            self.deviation_label = QLabel("Deviation")
            self.method_layout.addWidget(self.deviation_label)
            self.deviation_entry = QLineEdit("0.05")
            self.method_layout.addWidget(self.deviation_entry)
        else:
            debug_logger.debug("creating interval widgets")
            self.interval_label = QLabel("Intervals")
            self.method_layout.addWidget(self.interval_label)
            self.time_unit_entry = QComboBox()
            self.time_unit_entry.addItems(list(TIME_UNIT_FACTORS.keys()))
            self.time_unit_entry.setCurrentIndex(
                get_dict_key_index(TIME_UNIT_FACTORS, "s")
            )
            self.method_layout.addWidget(self.time_unit_entry)
            self.interval_entry = QLineEdit("")
            self.interval_entry.setToolTip(
                "Enter intervals surround by "
                "square brackets and seperated "
                "by commans, eg: '[0, 10], [70, 100]'"
            )
            self.method_layout.addWidget(self.interval_entry)
        # insert the newly created layout in the 3rd or 4th row
        # depending on whether there is an entry field for the correction method
        pos = 3 + int(self.method_box.currentText() == "Polynomial")
        self.layout.insertLayout(pos, self.method_layout)

    def ok_clicked(self):
        method = self.method_options[self.method_box.currentIndex()]
        if method == "Running Percentile":
            window_unit = self.window_unit_entry.currentText()
            window_duration = (
                float(self.window_entry.text()) / TIME_UNIT_FACTORS[window_unit]
            )
            self.main.data.baseline_correction_running_percentile(
                window_duration=window_duration,
                percentile=float(self.percentile_entry.text()),
                detect_jumps=self.detect_jumps_check.isChecked(),
                jump_sensitivity=float(self.sensitivity_entry.text()),
            )
        else:
            if method != "Offset":
                degree = int(self.degree_entry.text())
            else:
                degree = None
            selection = self.selection_options[self.selection_box.currentIndex()]
            if selection == "Piezo":
                intervals = None
                deviation = float(self.deviation_entry.text())
                active = self.active_checkbox.isChecked()
                time_unit = None
            elif selection == "Intervals":
                active = None
                deviation = None
                intervals = string_to_list(self.interval_entry.text())
                time_unit = self.time_unit_entry.currentText()

            self.main.data.baseline_correction(
                method=method,
                degree=degree,
                intervals=intervals,
                selection=selection,
                deviation=deviation,
                active=active,
                time_unit=time_unit,
            )
        self.main.ep_frame.update_combo_box()
        self.main.plot_frame.plot_all()
        self.dialog.close()
        self.close()

    def cancel_clicked(self):
        self.dialog.close()
        self.close()
