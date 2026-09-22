"""Session panel: measured trials use the existing HEDS radians backend only."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QComboBox, QPlainTextEdit, QFileDialog, QMessageBox,
    QInputDialog)
from vbb_study.lab import core
from vbb_study.lab.report import build_report
from .phase import compose_phase, phase_to_gray


def cast_session_trial(backend, root, tid, *, restore_base=False):
    """No reconnect, regeneration, additional phase terms or fallback to image data."""
    root, state, p = core.session(root)
    if backend is None:
        raise ValueError('Connect the SLMs on the home page first, and establish your baseline beam.')
    device = backend._state('SLM2')
    if p['slm']['phase_owner'] != 'existing_driver':
        raise ValueError('The GUI trial panel requires phase_owner=existing_driver; do not pass an offline LUT through HEDS phase mode.')
    if not device.phase_mode_verified or device.serial != p['slm']['panel_id'] or abs(float(device.wavelength_nm or 0)-p['wavelength_nm']) > .5:
        raise ValueError('Connected SLM2 serial/wavelength differs from the session.')
    if backend.name == 'dummy' and p.get('data_kind', 'experiment') != 'synthetic_demo':
        raise ValueError('Dummy output cannot be recorded as an experimental cast.')
    if restore_base:
        phase_file = root/'calibration/base_phase_rad.npy'
        folder = root/'rollback_casts'
    else:
        folder, trial = core.trial_info(root, tid)
        phase_file = folder/'total_phase_rad.npy'
    phase = np.load(phase_file, allow_pickle=False)
    if list(phase.shape) != p['slm']['shape_yx'] or not np.isfinite(phase).all():
        raise ValueError('Trial phase is not a finite native-size array.')
    folder.mkdir(exist_ok=True)
    stamp = core.now().replace(':','').replace('.','')
    output = folder/('display_'+stamp+'.png')
    _, gray = phase_to_gray(phase)
    message = backend.show_phase_array('SLM2', phase, gray, output, allow_png_fallback=False)
    receipt = {'created_utc': core.now(), 'trial_id': tid, 'restore_base': restore_base,
               'backend': backend.name, 'serial': device.serial, 'wavelength_nm': device.wavelength_nm,
               'phase_sha256': core.digest(phase_file), 'transfer': device.last_transfer,
               'message': message, 'optical_phase_calibration_proven': False}
    core.save_json(folder/('cast_'+stamp+'.json'), receipt)
    return receipt


class LabWorkbench(QDialog):
    def __init__(self, cockpit):
        super().__init__(cockpit)
        self.cockpit = cockpit
        self.root = None
        self.setWindowTitle('SLM2 measurement and correction')
        self.resize(1050, 780)
        layout = QVBoxLayout(self)
        title = QLabel('Capture → compare → verify → keep or restore')
        title.setStyleSheet('font-size:22px;font-weight:700')
        layout.addWidget(title)
        note = QLabel('SLM2 trials replace the complete phase through HEDS radians mode. SLM1 stays on its established baseline.\nKeep camera exposure, gain, attenuation and z coordinates fixed throughout a session.')
        note.setWordWrap(True); layout.addWidget(note)
        self.path_label = QLabel('No session open'); layout.addWidget(self.path_label)
        self.row(layout, [('New session', self.new_session), ('Open session', self.open_session),
                          ('Edit profile', self.edit_profile), ('Start with dark frames', self.start)])
        self.mode = QComboBox(); self.mode.addItems(core.MODES)
        line = QHBoxLayout(); line.addWidget(QLabel('Next Zernike mode:')); line.addWidget(self.mode)
        btn = QPushButton('Plan ±0.15 / ±0.30 rad RMS'); btn.clicked.connect(lambda: self.action(self.plan)); line.addWidget(btn)
        layout.addLayout(line)
        self.trials = QListWidget(); layout.addWidget(self.trials)
        self.row(layout, [('Cast selected to SLM2', self.cast), ('Open capture folder', self.capture_folder),
                          ('Import selected capture', self.ingest), ('Refresh', self.refresh)])
        self.row(layout, [('Evaluate sweep', self.evaluate), ('Plan fresh verification', self.verify),
                          ('Accept if verified', self.accept), ('Open report', self.report)])
        self.row(layout, [('Restore original baseline', self.restore_base), ('Restore accepted / round control', self.restore_control)])
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(220); layout.addWidget(self.log)

    def row(self, layout, actions):
        row = QHBoxLayout()
        for label, fn in actions:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, callback=fn: self.action(callback))
            row.addWidget(button)
        layout.addLayout(row)

    def action(self, fn):
        try:
            value = fn()
            if value is not None:
                self.log.appendPlainText(value if isinstance(value, str) else json.dumps(value, indent=2))
        except Exception as exc:
            self.log.appendPlainText(str(exc))
            QMessageBox.warning(self, 'Action could not complete', str(exc))

    def require_root(self):
        if self.root is None:
            raise ValueError('Create or open a session first.')
        return self.root

    def selected(self):
        item = self.trials.currentItem()
        if item is None:
            raise ValueError('Select a trial first.')
        return item.text().split(' | ')[0]

    def new_session(self):
        parent = QFileDialog.getExistingDirectory(self, 'Choose parent folder for session')
        if not parent:
            return
        name, ok = QInputDialog.getText(self, 'Session name', 'New folder name:')
        if not ok or not name:
            return
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Use a simple folder name.')
        root = Path(parent)/name
        core.init_session(root)
        self.cockpit._update_config_from_controls()
        result = compose_phase(self.cockpit.app_config.slm2)
        if result.warnings:
            raise ValueError('Resolve SLM2 generation warnings before exporting a baseline: '+ '; '.join(result.warnings))
        np.save(root/'setup_base_phase_rad.npy', result.phase_rad.astype(np.float32))
        # Preserve the SLM1 reference too: its state must remain fixed.
        slm1 = compose_phase(self.cockpit.app_config.slm1)
        np.save(root/'setup_slm1_phase_rad.npy', slm1.phase_rad.astype(np.float32))
        core.save_json(root/'setup_gui_state.json', self.cockpit.app_config.as_dict())
        profile = core.read_json(root/'profile.json')
        cfg = self.cockpit.app_config.slm2
        profile['slm']['panel_id'] = cfg.serial
        profile['slm']['center_yx_px'] = [cfg.center_y_px, cfg.center_x_px]
        profile['notes'] += ' SLM centre copied from current GUI; verify footprint/centre before start. Base is a snapshot of the current GUI phase, not proof it was displayed.'
        core.save_json(root/'profile.json', profile)
        self.root = root
        self.refresh()
        self.edit_profile()
        return 'Current GUI phase saved as setup_base_phase_rad.npy. Fill profile values; establish this baseline using direct_phase_array and capture dark frames before Start.'

    def open_session(self):
        folder = QFileDialog.getExistingDirectory(self, 'Open session folder')
        if folder:
            self.root = Path(folder); self.refresh()

    def edit_profile(self):
        root = self.require_root()
        if (root/'session.json').exists():
            raise ValueError('This session is frozen. Create a new session to change the measurement setup.')
        # Built-in editor avoids relying on Windows JSON file association.
        dialog = QDialog(self); dialog.setWindowTitle('Measured session profile'); dialog.resize(800, 650)
        layout = QVBoxLayout(dialog); editor = QPlainTextEdit((root/'profile.json').read_text(encoding='utf-8')); layout.addWidget(editor)
        button = QPushButton('Save profile'); layout.addWidget(button)
        def save():
            try:
                core.save_json(root/'profile.json', json.loads(editor.toPlainText())); dialog.accept()
            except Exception as exc:
                QMessageBox.warning(dialog, 'Invalid JSON', str(exc))
        button.clicked.connect(save); dialog.exec()

    def start(self):
        root = self.require_root()
        profile = core.read_json(root/'profile.json')
        backend = self.cockpit.backend
        if backend is None or backend.name != 'heds':
            raise ValueError('Connect the real HEDS backend and establish the baseline before starting an experimental session. Use the CLI demo for offline practice.')
        device = backend._state('SLM2')
        if not device.phase_mode_verified or 'direct_phase_array:' not in device.last_transfer:
            raise ValueError('Cast the baseline to SLM2 using direct_phase_array first. The session must begin with a successful HEDS phase transfer.')
        # Ensure the base snapshot is what was actually sent, not subsequently edited controls.
        shown = np.load(device.last_path, allow_pickle=False)
        base = np.load(root/'setup_base_phase_rad.npy', allow_pickle=False)
        if shown.shape != base.shape or np.max(np.abs(np.angle(np.exp(1j*(shown-base))))) > 2e-5:
            raise ValueError('Displayed SLM2 phase differs from the New-session baseline snapshot. Restore it or create a new session.')
        evidence = {'source': 'Live HEDS GUI connection and successful direct-phase cast',
                    'serial': device.serial, 'wavelength_nm': device.wavelength_nm,
                    'transfer': device.last_transfer, 'base_phase_sha256': core.digest(root/'setup_base_phase_rad.npy'),
                    'optical_phase_response_measured': False}
        core.save_json(root/'phase_path_evidence.json', evidence)
        profile['slm']['phase_owner'] = 'existing_driver'
        profile['slm']['phase_path_evidence'] = str((root/'phase_path_evidence.json').resolve())
        profile['slm']['native_coordinates_verified'] = True
        profile['slm']['phase_response_optically_verified'] = False
        core.save_json(root/'profile.json', profile)
        darks, _ = QFileDialog.getOpenFileNames(self, 'Select at least 3 new dark frames at the SAME camera settings', filter='Camera frames (*.bmg *.npy *.txt *.csv *.tif *.bmp)')
        if not darks:
            return
        core.start_session(root, root/'setup_base_phase_rad.npy', darks)
        _, state, _ = core.session(root)
        # Hash the GUI baseline and the untouched SLM1 reference into session provenance.
        for name in ['setup_gui_state.json', 'setup_slm1_phase_rad.npy']:
            state['calibration_sha256'][name] = core.digest(root/name)
        core.save_json(root/'session.json', state)
        return 'Session started. HEDS phase command path verified; optical LUT response is not claimed measured.'

    def refresh(self):
        root = self.require_root(); self.path_label.setText(str(root)); self.trials.clear()
        if not (root/'session.json').exists():
            return
        _, state, _ = core.session(root)
        for round_ in state['rounds']:
            for tid in round_['trials']+round_.get('verification', []):
                folder, t = core.trial_info(root, tid)
                status = 'captured' if (folder/'measurement.json').exists() else 'pending'
                self.trials.addItem(f'{tid} | {t["coefficient_rad_rms"]:+.3f} rad RMS | {status}')

    def plan(self):
        result = core.plan_round(self.require_root(), self.mode.currentText()); self.refresh()
        return result

    def check_slm1(self):
        root = self.require_root()
        ref = root/'setup_slm1_phase_rad.npy'
        if ref.exists():
            backend = self.cockpit.backend
            if backend is None:
                raise ValueError('Connect the SLMs first.')
            device = backend._state('SLM1')
            if 'direct_phase_array:' not in device.last_transfer:
                raise ValueError('Cast the baseline SLM1 through direct_phase_array first; its displayed reference must be known.')
            actual = np.load(device.last_path, allow_pickle=False)
            expected = np.load(ref, allow_pickle=False)
            if actual.shape != expected.shape or np.max(np.abs(np.angle(np.exp(1j*(actual-expected))))) > 2e-5:
                raise ValueError('SLM1 no longer matches the session baseline. Restore it before proceeding.')

    def cast(self):
        self.check_slm1()
        return cast_session_trial(self.cockpit.backend, self.require_root(), self.selected())

    def capture_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str((self.require_root()/'incoming'/self.selected()).resolve())))

    def ingest(self):
        root = self.require_root(); tid = self.selected(); self.check_slm1()
        folder, t = core.trial_info(root, tid)
        receipts = sorted(folder.glob('cast_*.json'))
        if not receipts or core.read_json(receipts[-1])['backend'] != 'heds':
            raise ValueError('Cast this trial successfully to HEDS from this panel before importing its camera frames.')
        device = self.cockpit.backend._state('SLM2')
        actual = np.load(device.last_path, allow_pickle=False)
        expected = np.load(folder/'total_phase_rad.npy', allow_pickle=False)
        if actual.shape != expected.shape or np.max(np.abs(np.angle(np.exp(1j*(actual-expected))))) > 2e-5:
            raise ValueError('Displayed SLM2 phase no longer matches the selected trial.')
        _, _, p = core.session(root)
        result = core.capture(root, tid, settings_id=p['camera']['settings_id']); self.refresh()
        return f'{tid}: {len(result["frames"])} fresh frames imported and sealed.'

    def evaluate(self):
        result = core.evaluate(self.require_root())
        return {'status': result['status'], 'recommended_trial': result['recommended_trial']}

    def verify(self):
        result = core.plan_verification(self.require_root()); self.refresh(); return result

    def accept(self):
        return core.accept(self.require_root())

    def report(self):
        target = build_report(self.require_root())
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def restore_base(self):
        return cast_session_trial(self.cockpit.backend, self.require_root(), 'original_baseline', restore_base=True)

    def restore_control(self):
        root, state, _ = core.session(self.require_root())
        tid = state['accepted_trial'] or state['rounds'][-1]['start']
        return cast_session_trial(self.cockpit.backend, root, tid)
