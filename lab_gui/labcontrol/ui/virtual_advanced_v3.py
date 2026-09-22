"""Resolution-selection extension for the Virtual Lab experiment builder."""

from __future__ import annotations

import copy
import dataclasses
import faulthandler
import json
import math
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from PySide6.QtCore import QPointF, Qt, QtMsgType, Slot, qInstallMessageHandler
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from slm_lab_control.config import AppConfig
from slm_lab_control.ui.style import APP_QSS
from ..axicon_propagation import MEASURED_BENCH_AXICON_K_PERP_M_INV, MEASURED_BENCH_AXICON_SOURCE
from ..relay_4f import fourier_aperture_report
from ..virtual_radiometry import camera_response_scale

from ..devices.camera import DummyCameraProvider
from ..mode_controller import ModeAwareLabController
from ..sensorless import PARAMETERS
from ..state import ExperimentState, ExperimentStore
from ..virtual_lab import BlindCorrectionRunner, OperatingMode
from ..virtual_lab_resolution import ResolutionAwareVirtualBenchEngine, VirtualOutputResolution
from .advanced import ADVANCED_QSS, PROJECT_ROOT, panel
from .controls import (
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    blocked_signals,
    make_shrinkable,
)
from .slm_views import phase_pixmap
from .virtual_advanced_v2 import VirtualLabExperimentWindow
from ..virtual_lab_experiments import estimated_blind_cycle_frames


GUIDE_HTML = """
<h2>Brightness and small-beam design</h2>
<p>Virtual camera sensitivity is fixed: moving the camera, resizing the input or
applying a correction no longer rescales each frame to the same peak. Input
power stays fixed when the Gaussian radius changes (before aperture/window
losses). Exposure and gain still change counts; this is relative simulation,
not calibrated watts. Keep exposure and gain unchanged for comparisons.</p>
<p><b>View &rarr; sensor log</b> (default) reveals faint rings on a fixed
logarithmic scale. <b>sensor range</b> is fixed linear scaling. The older log,
percentile and full-range choices auto-adjust contrast per frame: do not compare
brightness using those views. Recorded numeric frames are unaffected.</p>
<p><b>Auto-expose</b> (beside Snapshot) does what you do at the bench: it sets
the exposure so the brightest pixel sits near 75&nbsp;% of full scale at this
plane, then takes a frame. You need it because the beams differ enormously in
brightness at fixed sensitivity &mdash; at 1&nbsp;ms a q=0 focus clips (peak
&asymp;9400 counts at z=15&nbsp;mm) while the same plane at q=20 reads 2840 and
z=45&nbsp;mm reads 47. The chip says <b>CLIPPED</b> when a frame is saturated;
clipping also costs you score on the correction tab, so expose first, then
correct. Note the exposure when you compare captures.</p>
<p><b>1 Beam &rarr; Small-beam designer</b> opens the earlier repository's
target-based cone/objective design, and now also reports what <i>this</i> bench
needs for that target: the objective ratio and the input beam radius, with a
button that puts the objective into Settings. A q=0 spot diameter is not a q=20
bright-ring diameter. It never casts masks by itself.</p>
<h2>What this is</h2>
<p>One cockpit over three <b>experiment sources</b> (top of the sidebar):</p>
<ul>
<li><b>VIRTUAL LAB</b> &mdash; a digital twin of your bench. No HEDS or Beamage
connection is used or needed, and every frame is labelled <code>SYNTHETIC</code>.</li>
<li><b>LIVE LAB</b> &mdash; the physical bench. Nothing reaches an SLM or camera
until you explicitly connect and cast.</li>
<li><b>RECORDED LAB</b> &mdash; replay of measured frames, read only.</li>
</ul>

<h2>The Virtual Lab page at a glance</h2>
<p><b>Left:</b> the virtual Beamage camera. Underneath it: <b>Camera z</b> (distance
after the axicon, with &minus;/+ steps), <b>&#9654; Cast masks + go live</b>,
<b>Stop</b> and <b>Snapshot</b>. <b>View &#9662;</b> holds the colour map,
intensity scale and overlays.</p>
<p><b>Right:</b> tabs in the order you work &mdash;
<b>1 Beam</b> &rarr; <b>2 Errors</b> &rarr; <b>3 Correct</b>, plus
<b>Captures</b> and <b>Settings</b>. Hover over any tab, button or readout for a
short explanation.</p>

<h2>Your first run (about five minutes)</h2>
<ol>
<li>Sidebar: choose <b>VIRTUAL LAB</b>, then open <b>Virtual Lab</b>.</li>
<li><b>1 Beam</b>: pick the vortex route (e.g. SLM1 only) and total charge
q&nbsp;=&nbsp;20, press <b>Set route (configure)</b>. Optionally change
<b>Beam radius (1/e)</b> and press <b>Apply beam size</b>.</li>
<li>Press <b>&#9654; Cast masks + go live</b>. You now see the vortex Bessel beam,
with the small frame-to-frame flicker a real camera shows.</li>
<li><b>2 Errors</b>: break the beam &mdash; e.g. Wavefront &rarr; Coma X = 0.8
waves &mdash; and press <b>Apply faults</b>. Watch the ring become uneven.
These errors are hidden truth: the optimiser never sees them, only camera
frames. <b>Load strong mixed example</b> fills in a hard multi-fault case.</li>
<li><b>3 Correct</b>, and follow its five steps (next section).</li>
</ol>

<h2>How correction works &mdash; the loop</h2>
<p>The optimiser behaves like you would at the bench: it nudges one SLM
correction mode at a time, takes fresh camera frames at a few planes, and keeps
a nudge only if it makes the beam measurably better. Better means a lower
<b>score J</b>:</p>
<p style="margin-left:14px">J = 3&times;beam walk between planes + 1&times;ring-radius
variation + 1.5&times;ring eccentricity + 1.5&times;unevenness around the ring
+ 0.5&times;light in the dark core + 4&times;edge clipping + 4&times;saturation.
A perfect, stable, round, even ring scores near 0.</p>
<ol>
<li><b>Step 1 &mdash; What may the SLM change?</b> Choose the SLM (SLM2 by
default) and tick the modes it may adjust. <b>Coma</b> is a fast first pass;
<b>Low order</b> adds defocus, astigmatism and spherical; <b>Everything</b> also
lets it steer (tip/tilt), which can compensate beam pointing.</li>
<li><b>Step 2 &mdash; Where should it measure?</b> At least two camera planes,
e.g. <code>31, 45</code>. <b>Use current camera z</b> fills that in. The line
below estimates how many frames and how long it will take.</li>
<li><b>Step 3 &mdash; Find correction.</b> Progress appears underneath;
<b>Cancel</b> stops safely. While it runs the SLMs are only probed; when it
finishes your original masks are put back.</li>
<li><b>Step 4 &mdash; Review.</b> The table lists every mode it tried:
<i>Now</i> is the current SLM command, <i>Proposed</i> is the verified better
value (or &ldquo;no change&rdquo;). The score line shows J before &rarr; after.
Press <b>Apply correction</b> to send it to the virtual SLMs, or
<b>Discard</b> to leave the SLMs untouched.</li>
<li><b>Step 5 &mdash; Check, then iterate.</b> After applying, the live camera
shows the corrected beam and the panel compares ring roundness and unevenness
before and after. Press <b>Next iteration</b> and go back to step 3; each round
starts from the correction you just applied. The <b>history</b> table tracks
every round (J before, J after, how much better, and whether it was applied,
discarded or found nothing); hover a row to see the exact commands.
<b>Export&hellip;</b> saves it as JSON.</li>
</ol>
<p>When a round finds nothing, the remaining error is outside what you allowed:
tick more modes, try the other SLM, or measure at different planes. Axicon
decentre is mechanical and can only be partly compensated by the SLM.</p>

<h2>The virtual camera is a Beamage-4M</h2>
<p>Every frame is <b>2048 &times; 2048 pixels at 5.5&nbsp;&micro;m</b> over the full
11.26&nbsp;mm sensor. The light is first computed finely enough to resolve the
Bessel intensity fringes (about 6.5&nbsp;&micro;m apart for the bench axicon), then
each camera pixel is the true average over its 5.5&nbsp;&micro;m square &mdash;
exactly what a real sensor does. That is why you see the <b>4-fold hyperbolic
moir&eacute;</b> around the core: 6.5&nbsp;&micro;m fringes on a 5.5&nbsp;&micro;m
square pixel grid alias, on the real camera and here alike.</p>
<p>The central ring is small on this camera. Its first bright ring has radius
&asymp; 3.8&nbsp;&micro;m &times; (Bessel zero for q) &mdash; about <b>5 pixels
across for q&nbsp;=&nbsp;5</b>, 9 for q&nbsp;=&nbsp;10, <b>17 for q&nbsp;=&nbsp;20</b>,
32 for q&nbsp;=&nbsp;40. Press <b>Core</b> to zoom straight to it at camera-pixel
scale with the colour scale set by the ring's own peak. <b>Fit beam</b> and
<b>Full</b> zoom out; zoomed out, each screen pixel shows the <i>average</i> of the
camera pixels under it, so the monitor never adds a moir&eacute; of its own.</p>
<p><b>What should it look like? &mdash; the View button.</b> Above the Virtual Lab
camera (and next to <b>Full frame</b> on Home) is <b>View: camera pixels</b>. Click
it to flip to <b>View: native model</b>: the same plane, noise-free, at the fine
model sampling (2.75&nbsp;&micro;m, half a camera pixel), at the same zoom and
with the same overlays. That is the clean Bessel pattern your offline simulations
show. Click again to return to what the 5.5&nbsp;&micro;m Beamage pixels record.
Nothing else changes: captures, metrics and correction always use the camera
frame. Each camera pixel is exactly the average of the native samples under it.</p>

<h2>Camera readouts</h2>
<ul>
<li><b>Beam radius (D4&sigma;/2)</b> &mdash; half the ISO&nbsp;11146 second-moment
width in x and y, as a Beamage reports it, and <b>circularity</b>
(minor/major axis ratio, 1 = round).</li>
<li><b>Ring radius</b> &mdash; radius of the brightest vortex ring.
<b>Ring roundness</b> is 1 &minus; its eccentricity; <b>unevenness</b> is the
intensity variation around it (0 = perfectly even).</li>
<li>All lengths are <i>model</i> millimetres/micrometres; the real Beamage field
of view is not yet calibrated.</li>
</ul>

<h2>The axicon (Settings tab)</h2>
<p>By default the model uses your <b>bench axicon</b>: the Thorlabs
&ldquo;20&deg;&rdquo; optic, set by its <b>measured transverse wavenumber
k&perp; = 4.83&times;10<sup>5</sup> m<sup>&minus;1</sup></b> from the real
BeamGage q = 20 z-scan &mdash; the same value your digital-twin correction code
uses. In the model's exact refractive convention that is a
<b>&asymp;9.7&deg; base angle</b> (a 4.5&deg; cone). The &ldquo;20&deg;&rdquo;
label is <i>not</i> the base angle; typing 20 as a base angle doubles the cone.</p>
<p>You can also choose <b>Custom k&perp;</b> or <b>Custom model base angle</b>.
The panel says, before you apply, what the choice means: core size, Bessel-zone
length for your beam, whether the camera is inside it, and how it will be
sampled. Anything the model cannot represent is refused and the previous axicon
kept.</p>
<p>How steep axicons are simulated (ported from
<code>real_bmg_digital_twin_correction.py</code>): the SLM relay runs on the
working grid over the 11.26&nbsp;mm sensor window; the field is Fourier-resampled
onto a finer grid over the same window just before the axicon (1024 &rarr; 4096,
2.75&nbsp;&micro;m, so the intensity fringes are resolved); one band-limited
spectrum is then frozen, so each camera plane is a single inverse FFT. All
resolution modes report true Beamage pixels; they differ only in how finely the
SLM relay is computed.</p>

<h2>The 4F relay and its aperture</h2>
<p>Your bench relay is <b>SLM &rarr;300&nbsp;mm&rarr; L1 (f=300) &rarr;300&nbsp;mm&rarr;
aperture &rarr;300&nbsp;mm&rarr; L2 (f=300) &rarr;300&nbsp;mm&rarr; axicon</b>.
Every spacing equals a focal length, so it is a <b>symmetric 4F</b>: the axicon
sees the SLM plane at <b>1:1</b>, <b>rotated by 180&deg;</b>, with the aperture
in the shared focal plane selecting the +1 order. The model now applies that
rotation, so a beam you push +x on the SLM walks &minus;x on the camera, exactly
as on the bench. Odd aberrations (tilt, coma) therefore reach the axicon with
the opposite sign. The correction loop measures and corrects through that same
relay, so its recommendations already account for it &mdash; but if your camera
is mounted rotated, expect the sign of a manual tilt/coma tweak to disagree with
the model.</p>
<p><b>4F stop &Oslash;</b> (Settings) models that aperture as what it physically
is: a stop of diameter D passes SLM detail down to D/(2&lambda;f). Leave it at
<i>ideal</i> for perfect order selection. It is worth knowing what a real iris
costs a high-charge vortex, whose phase winds q times around its core and so
carries fine structure: for <b>q = 20 on a 2&nbsp;mm beam, 10&nbsp;mm passes
98&nbsp;%, 3&nbsp;mm passes 81&nbsp;% and 1&nbsp;mm only 14&nbsp;%</b>, while a
plain Gaussian passes any of them untouched. A too-tight iris does not move the
ring; it degrades the vortex that makes it.</p>

<h2>Making a tiny beam</h2>
<p>A wider input beam does <b>not</b> shrink the rings &mdash; that is lens
intuition. An axicon fixes the ring size by its own k&perp;, and the input beam
sets how <i>long</i> the Bessel region is (L &asymp; w/tan&theta;). You can see
this: 1&nbsp;mm in gives a bright beam that is gone by z=32&nbsp;mm; 3&nbsp;mm in
gives a dimmer one still going at 32&nbsp;mm, at the same ring size and the same
total power.</p>
<p>Since the relay is 1:1, the only way to micron-scale rings is a
<b>demagnifying objective after the axicon</b>. Set <b>Objective 1:N</b>
(Settings). An ideal telescope scales the whole pattern exactly &mdash; rings
&times;M, z &times;M&sup2;, cone angle &divide;M, where M = 1/N &mdash; so the
simulated frame <i>is</i> the sample-plane pattern in new units, and the camera
metrics gain a <b>SAMPLE PLANE</b> line. Two independent knobs:
<b>ring size follows the objective, length follows the input beam radius.</b></p>
<p>The limit is real: 1:N raises the cone to NA = N &times; 0.079, so this
axicon cannot pass NA = 1 in air beyond about 1:12. Impossible requests are
refused rather than drawn. <b>Small-beam designer&hellip;</b> (tab 1) takes a
target size and length and reports the objective and input beam radius this
bench needs &mdash; the old 3&nbsp;&micro;m / 150&nbsp;&micro;m repository target
comes out as a 1:3.3 objective with a 0.13&nbsp;mm input beam. The objective's
own pupil truncation and aberrations are not modelled, so treat a high-NA answer
as a design target, not a prediction.</p>

<h2>Editing the masks by hand</h2>
<p><b>Edit&hellip;</b> under each mask on tab 1 (or the <b>SLM 1</b>/<b>SLM 2</b>
pages) opens the complete phase editor: vortex, blaze carrier, Zernike
commands, wavefront files, retrieved corrections. <b>The optical field changes
only when you cast</b> &mdash; until then the thumbnail says <i>EDITED SINCE
CAST</i>. Press <b>&#9654; Cast masks + go live</b> to see the edit.</p>

<h2>Saving evidence (Captures tab)</h2>
<p>Move the camera, press <b>Save frames at current z</b> (with a frame count and
trial name). Saved frames are never overwritten and carry their z and full phase
provenance. <b>Correct at the saved planes</b> copies every saved z into step 2.</p>

<h2>Speed</h2>
<table border="1" cellpadding="6" cellspacing="0">
<tr><th>Settings &rarr; resolution</th><th>SLM relay grid &rarr; axicon grid</th><th>Move camera</th><th>Each optimiser probe</th></tr>
<tr><td><b>LIVE MODEL</b> (default)</td><td>1024 &rarr; 4096</td><td>~1.5 s</td><td>~4.5 s</td></tr>
<tr><td>BEAMAGE 4M</td><td>2048 &rarr; 4096</td><td>~1.3 s</td><td>~5 s</td></tr>
<tr><td>MAX MODEL &rarr; BEAMAGE</td><td>4096 &rarr; 4096</td><td>~1.3 s</td><td>~12 s; &gt;2 GB RAM</td></tr>
</table>
<p>Frames at an unchanged plane and mask reuse the cached field, so the live
view stays smooth. The live feed pauses when you leave the camera pages and
resumes when you return; your masks, errors and camera position are kept.</p>

<h2>If something goes wrong</h2>
<ul>
<li>Every session writes <code>lab_gui/logs/lab_gui_crash.log</code>. If the
application ever closes unexpectedly, it tells you so at the next start. Please
keep that file &mdash; it records where it stopped.</li>
<li><b>Settings &rarr; Activity log</b> shows everything the bench and optimiser
did this session.</li>
<li>A recipe on <b>Optimise / recover</b> that will not start lists the
calibrations it is waiting for. In VIRTUAL LAB, <b>Mark prerequisites VALID from
the virtual bench</b> lets you rehearse it with clearly SIMULATED provenance.</li>
</ul>
"""


_FAULT_LOG_HANDLE = None


class VirtualLabResolutionWindow(VirtualLabExperimentWindow):
    """Full Virtual Lab GUI with explicit detector/model resolution choices."""

    # Widest control tab plus the camera column, sidebar and page margins.  The
    # stacking threshold stays above the resulting window minimum so the window
    # can still be dragged narrow enough to reach the stacked layout.
    PAGE_GUIDE = 10

    COCKPIT_TABS_MIN_PX = 660
    COCKPIT_STACK_BELOW_PX = 1620
    # A short window cannot hold the camera above a full bench-control card
    # either, so height stacks the cockpit just as narrow width does.
    COCKPIT_STACK_UNDER_PX = 900

    def __init__(self, store=None, controller=None, calibration=None):
        self._pending_virtual_proposal = None
        self._virtual_correction_history = []
        self._awaiting_post_apply = None
        self._resume_virtual_live_after_navigation = False
        if controller is None:
            initial = ExperimentState.from_app_config(AppConfig())
            initial.system.experiment_label = "Optical lab session"
            initial.camera.shape_yx = (512, 512)
            store = store or ExperimentStore(initial)
            camera = DummyCameraProvider(store.snapshot, shape_yx=(512, 512))
            # q=20 rings are badly under-sampled on the fast 256/512 model
            # grids, but a 2048 grid costs seconds per camera move.  1024 both
            # resolves the ring family and keeps the live view drivable; the
            # Beamage-equivalent and maximum grids stay one click away.
            engine = ResolutionAwareVirtualBenchEngine(
                output_resolution=VirtualOutputResolution.LIVE_1M
            )
            controller = ModeAwareLabController(
                store,
                PROJECT_ROOT,
                camera_provider=camera,
                virtual_engine=engine,
            )
        elif not isinstance(controller.virtual_engine, ResolutionAwareVirtualBenchEngine):
            raise TypeError(
                "VirtualLabResolutionWindow requires ResolutionAwareVirtualBenchEngine so output sampling "
                "and optical-grid provenance remain authoritative."
            )
        super().__init__(store=store, controller=controller, calibration=calibration)
        self.resolution_engine.realistic_camera = self.virtual_realistic_camera.isChecked()
        self.virtual_realistic_camera.toggled.connect(
            lambda enabled: setattr(self.resolution_engine, "realistic_camera", bool(enabled))
        )

    @property
    def resolution_engine(self) -> ResolutionAwareVirtualBenchEngine:
        engine = self.mode_controller.virtual_engine
        if not isinstance(engine, ResolutionAwareVirtualBenchEngine):
            raise TypeError("Resolution-aware virtual engine is not installed.")
        return engine

    # ------------------------------------------------------------------ layout

    def _build_virtual_page(self) -> None:
        super()._build_virtual_page()
        self.virtual_quality.setCurrentText("validation")
        self.virtual_z_plan.setText("31, 35, 40, 45")
        self.measure_z_stop.setValue(45.0)
        self.measure_z_count.setValue(4)
        if self.virtual_quality.findText("maximum") < 0:
            self.virtual_quality.addItem("maximum")

        self.virtual_resolution_card = self._build_resolution_card()
        self._compose_bench_cockpit()
        self._refresh_virtual_resolution_status()
        self._add_home_native_toggle()
        self._build_guide_page()

    def _build_guide_page(self) -> None:
        _, layout = self._page(
            "How to use this application",
            "Written for this build. Nothing on this page commands hardware.",
        )
        guide = QTextBrowser()
        guide.setOpenExternalLinks(False)
        guide.setHtml(GUIDE_HTML)
        layout.addWidget(guide, 1)
        nav = QPushButton("How to use this")
        nav.setObjectName("LabNav")
        nav.setCheckable(True)
        nav.clicked.connect(lambda _checked=False: self.set_page(self.PAGE_GUIDE))
        sidebar = self.findChild(QFrame, "LabSidebar")
        side = sidebar.layout()
        side.insertWidget(max(0, side.count() - 2), nav)
        self.nav_buttons.append(nav)

    def _build_resolution_card(self) -> QFrame:
        card, box = panel(
            "Virtual camera resolution",
            "Choose Beamage pixel-count equivalence or retain the higher-resolution numerical model. "
            "Physical Beamage FOV/pixel pitch are not yet calibrated, so 2048x2048 here is a pixel-count match only.",
        )
        form = QFormLayout()
        self.virtual_output_resolution = NoWheelComboBox()
        self.virtual_output_resolution.addItems([item.value for item in VirtualOutputResolution])
        self.virtual_output_resolution.setCurrentText(self.resolution_engine.output_resolution.value)
        form.addRow("Output / propagation mode", self.virtual_output_resolution)
        box.addLayout(form)

        apply_button = QPushButton("Apply resolution mode")
        apply_button.setObjectName("Accent")
        apply_button.clicked.connect(self._apply_virtual_output_resolution)
        box.addWidget(apply_button)

        self.virtual_resolution_status = QLabel()
        self.virtual_resolution_status.setObjectName("Muted")
        self.virtual_resolution_status.setWordWrap(True)
        box.addWidget(self.virtual_resolution_status)

        warning = QLabel(
            "LIVE MODEL 1024x1024 is the interactive grid: it resolves the q=20 ring family that 256/512 "
            "aliases, and a camera move costs about a second instead of several. MAXIMUM MODEL uses a "
            "4096x4096 propagation/output grid and is computationally expensive. MAX MODEL -> BEAMAGE 4M uses "
            "that same 4096x4096 propagation, then 2x2 area-integrates to 2048x2048; that paired comparison "
            "is the cleanest way to isolate virtual detector sampling from propagation-grid resolution."
        )
        warning.setObjectName("WarnChip")
        warning.setWordWrap(True)
        box.addWidget(warning)
        return card

    def _compose_bench_cockpit(self) -> None:
        """Rebuild the Virtual Lab page as a camera cockpit plus tabbed controls.

        The inherited builders stack every card into a single column several
        thousand pixels tall, which pushes the camera and the controls that
        drive it off screen.  Every one of those cards is reused unchanged
        here; only the containers differ.
        """

        layout = self._page_layout(self.PAGE_VIRTUAL)
        if layout is None:
            raise RuntimeError("Virtual Lab page has no layout.")
        # Item 0 is a page heading that only repeats the window header, and the
        # cockpit needs that vertical space; item 1 is the provenance note, kept.
        while layout.count() > 2:
            item = layout.takeAt(2)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        heading = layout.takeAt(0).widget()
        if heading is not None:
            heading.setParent(None)
        # The provenance note is kept, as a tooltip on the camera title, rather
        # than spending a full line of a short screen on it.
        note_item = layout.takeAt(0)
        self._virtual_provenance_note = note_item.widget().text() if note_item and note_item.widget() else ""
        if note_item is not None and note_item.widget() is not None:
            note_item.widget().setParent(None)

        split = QSplitter(Qt.Horizontal)
        split.setObjectName("VirtualCockpit")
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_camera_column())
        split.addWidget(self._build_control_tabs())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([880, 620])
        self.virtual_cockpit_split = split
        layout.addWidget(split, 1)

    _make_tab_content_shrinkable = staticmethod(make_shrinkable)

    @staticmethod
    def _scroll_tab(*cards: QWidget) -> QScrollArea:
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(2, 2, 2, 2)
        box.setSpacing(10)
        for card in cards:
            VirtualLabResolutionWindow._make_tab_content_shrinkable(card)
            box.addWidget(card)
        box.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setWidget(holder)
        return area

    def _build_control_tabs(self) -> QTabWidget:
        """Tabs in the order an operator works: beam, errors, correct, evidence, settings."""

        self.virtual_progress.setMaximumHeight(260)
        self.virtual_progress.document().setMaximumBlockCount(2000)
        self.virtual_geometry.setMaximumHeight(160)
        self._build_axicon_controls()
        self.virtual_stack_status.setMaximumHeight(90)
        self.virtual_stack_status.setPlaceholderText(
            "Frame provenance appears here after a snapshot or a saved capture."
        )
        # Direct run-and-apply and the z-stack buttons duplicate the guided loop.
        self.virtual_optimise_button.hide()
        for button in self.virtual_z_planner.findChildren(QPushButton):
            button.hide()

        correct_card = self._build_correct_panel()
        capture_card = self._build_capture_card()

        log_card, log_box = panel(
            "Activity log",
            "Everything the virtual bench and the optimiser did in this session, including hidden-truth reveals.",
        )
        log_box.addWidget(self.virtual_progress)
        log_box.addWidget(self.virtual_stack_status)

        tabs = QTabWidget()
        tabs.setObjectName("VirtualTabs")
        tabs.setDocumentMode(True)
        tabs.setUsesScrollButtons(True)
        self.virtual_tab_indices: dict[str, int] = {}
        for key, title, tip, cards in (
            ("masks", "1 Beam", "Vortex charge, input beam size and both SLM masks.",
             (self._build_mask_card(),)),
            ("faults", "2 Errors", "Deliberately break the beam: coma, pointing, beam size, axicon offset...",
             (self.virtual_perturbation_card,)),
            ("correction", "3 Correct", "Find, review and apply an SLM correction, then iterate.",
             (correct_card,)),
            ("captures", "Captures", "Save repeated camera frames at chosen planes as evidence.",
             (capture_card,)),
            ("settings", "Settings", "Axicon, model resolution, hidden scenarios, optimiser tuning and the log.",
             (self.virtual_geometry_card, self.virtual_resolution_card, self.virtual_scenario_card,
              self.virtual_correction_card, self.virtual_measurement_card, self.virtual_results_card, log_card)),
        ):
            index = tabs.addTab(self._scroll_tab(*cards), title)
            tabs.setTabToolTip(index, tip)
            self.virtual_tab_indices[key] = index
        # Older call sites still ask for the tabs these replaced.
        self.virtual_tab_indices["results"] = self.virtual_tab_indices["correction"]
        self.virtual_tab_indices["advanced"] = self.virtual_tab_indices["settings"]
        tabs.setMinimumWidth(360)
        self.virtual_tabs = tabs
        return tabs

    def resizeEvent(self, event):  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._apply_cockpit_layout()

    def set_page(self, index: int) -> None:
        previous = self.pages.currentIndex() if hasattr(self, "pages") else None
        camera_pages = {self.PAGE_HOME, self.PAGE_MEASURE, self.PAGE_VIRTUAL}
        if (
            previous in camera_pages
            and index not in camera_pages
            and self.mode_controller.operating_mode is OperatingMode.VIRTUAL_LAB
        ):
            self._pause_virtual_live_for_navigation()
        if index == self.PAGE_GUIDE and self.pages.count() > self.PAGE_GUIDE:
            self.pages.setCurrentIndex(self.PAGE_GUIDE)
            self.page_title.setText("How to use this")
            for position, button in enumerate(self.nav_buttons):
                button.setChecked(position == self.PAGE_GUIDE)
            self._release_hidden_camera_views()
            return
        super().set_page(index)
        if index == self.PAGE_VIRTUAL:
            # Resize events are not guaranteed to have arrived with the final
            # window width before the page is first shown.
            self._apply_cockpit_layout()
        self._release_hidden_camera_views()
        if index in camera_pages:
            self._resume_virtual_live_if_visible()

    def _pause_virtual_live_for_navigation(self) -> None:
        thread, worker = self._camera_thread, self._camera_worker
        if thread is None or worker is None or not thread.isRunning():
            return
        self._resume_virtual_live_after_navigation = True
        # Never block the GUI on a propagation already in flight. The worker
        # will finish that frame, then stop before acquiring another one.
        worker.request_stop()
        thread.requestInterruption()

    def _resume_virtual_live_if_visible(self) -> None:
        if not self._resume_virtual_live_after_navigation:
            return
        if self.mode_controller.operating_mode is not OperatingMode.VIRTUAL_LAB:
            self._resume_virtual_live_after_navigation = False
            return
        if self._camera_thread is not None:
            return  # the previous worker is still winding down
        self._resume_virtual_live_after_navigation = False
        self.start_live()  # preserves stage, casts, scenario and camera settings

    @Slot()
    def _live_stopped(self) -> None:
        super()._live_stopped()
        if self.pages.currentIndex() in {self.PAGE_HOME, self.PAGE_MEASURE, self.PAGE_VIRTUAL}:
            self._resume_virtual_live_if_visible()
        self._refresh_virtual_chips()

    def stop_live(self) -> bool:
        # An explicit Stop, formal capture, mode change or correction task must
        # cancel the navigation auto-resume intent.
        self._resume_virtual_live_after_navigation = False
        return super().stop_live()

    def _release_hidden_camera_views(self) -> None:
        for page, name in (
            (self.PAGE_HOME, "home_camera_view"),
            (self.PAGE_MEASURE, "image_view"),
            (self.PAGE_VIRTUAL, "virtual_camera_view"),
        ):
            if page != self.pages.currentIndex() and hasattr(self, name):
                getattr(self, name).release_frame()

    def _apply_cockpit_layout(self) -> None:
        """Camera and controls side by side.

        The camera frame is square, so on a wide-but-short laptop screen it
        belongs beside the controls, not above them: stacked, the controls got
        a 250 px strip and the correction steps scrolled a thousand pixels.
        Only a genuinely narrow window stacks.
        """

        split = getattr(self, "virtual_cockpit_split", None)
        if split is None:
            return
        sidebar = self.sidebar_scroll.width() if hasattr(self, "sidebar_scroll") else 272
        usable = max(600, self.width() - sidebar - 44)
        stacked = usable < 760
        orientation = Qt.Vertical if stacked else Qt.Horizontal
        if split.orientation() != orientation:
            split.setOrientation(orientation)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        if stacked:
            self.virtual_camera_view.setMinimumSize(240, 180)
            split.setSizes([max(300, self.height() // 2), max(300, self.height() // 2)])
        else:
            # A laptop screen has ~600 px for the whole cockpit; the live strip
            # under the camera must stay above the fold.
            self.virtual_camera_view.setMinimumSize(300, 170 if self.height() < 900 else 240)
            # The controls get priority: every tab is laid out for ~490 px, and
            # a sideways scrollbar inside a step-by-step panel is unusable.
            tabs = max(500, int(usable * 0.52))
            split.setSizes([usable - tabs, tabs])

    def _set_cockpit_compact(self, compact: bool) -> None:
        """Retained for callers; the cockpit no longer moves cards between containers."""

    def _reindex_virtual_tabs(self) -> None:
        """Retained for callers; tab indices are fixed at build time."""

    def _show_virtual_tab(self, key: str) -> None:
        index = getattr(self, "virtual_tab_indices", {}).get(key)
        if index is None:
            return
        self.set_page(self.PAGE_VIRTUAL)
        self.virtual_tabs.setCurrentIndex(index)

    def _show_virtual_camera(self) -> None:
        self.set_page(self.PAGE_VIRTUAL)

    # --------------------------------------------------------- camera cockpit

    def _build_camera_column(self) -> QWidget:
        column = QWidget()
        outer = QVBoxLayout(column)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        outer.addWidget(self._build_camera_card(), 1)
        self._fold_display_options_into_menu()
        self.virtual_bench_card = self._build_live_strip()
        outer.addWidget(self.virtual_bench_card)
        self._make_tab_content_shrinkable(column)
        self.virtual_camera_column = column
        # Wrapped status text needs more height at narrow widths than a splitter
        # passes up to the page, so without its own scroll area this column's
        # rows were squeezed below their minimum and drawn over the camera.
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setWidget(column)
        self.virtual_camera_scroll = area
        return area

    def _build_camera_card(self) -> QFrame:
        card, box = panel("Virtual Beamage view")
        card.setToolTip(getattr(self, "_virtual_provenance_note", ""))
        header = QHBoxLayout()
        self.virtual_live_chip = QLabel("STOPPED")
        self.virtual_live_chip.setObjectName("BadChip")
        self.virtual_busy_chip = QLabel("SYNTHETIC")
        self.virtual_busy_chip.setObjectName("WarnChip")
        header.addWidget(self.virtual_live_chip)
        header.addWidget(self.virtual_busy_chip)
        header.addStretch(1)
        self.virtual_native_toggle = self._make_native_toggle()
        header.addWidget(self.virtual_native_toggle)
        box.addLayout(header)
        # On its own full-width line this readout stays legible; sharing the
        # chip row squeezed it into a one-word column on narrower cards.
        self.virtual_frame_chip = QLabel("no frame yet — press Cast masks + start live")
        self.virtual_frame_chip.setObjectName("Muted")
        self.virtual_frame_chip.setWordWrap(True)
        box.addWidget(self.virtual_frame_chip)

        self.virtual_camera_view.setMinimumSize(340, 220)
        # fitInView negotiates against the scrollbars it is about to need, which
        # left the fitted beam clipped top and bottom in this wide, short view.
        # Panning still works through the view's hand-drag mode.
        self.virtual_camera_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.virtual_camera_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        box.addWidget(self.virtual_camera_view, 1)

        display = QHBoxLayout()
        self.virtual_colour_mode = NoWheelComboBox()
        self.virtual_colour_mode.addItems(["inferno", "gentec-like", "turbo", "viridis", "grayscale"])
        self.virtual_display_scale = NoWheelComboBox()
        self.virtual_display_scale.addItems(["sensor log", "sensor range", "percentile", "log", "full range"])
        self.virtual_display_scale.setCurrentText("sensor log")
        self.virtual_display_scale.setToolTip(
            "Sensor log/range keep brightness comparable. Other modes auto-adjust contrast per frame. "
            "Display only: captures retain fixed-sensitivity counts."
        )
        for combo in (self.virtual_colour_mode, self.virtual_display_scale):
            combo.currentTextChanged.connect(lambda _text: self._rerender_virtual())
        self.virtual_auto_fit = QCheckBox("Auto-fit")
        self.virtual_auto_fit.setChecked(True)
        fit = QPushButton("Fit beam")
        fit.clicked.connect(self.virtual_camera_view.fit_signal)
        full = QPushButton("Full")
        full.setToolTip("Show the whole camera frame.")
        core = QPushButton("Core")
        core.setToolTip("Zoom to the central ring at camera-pixel scale: the ring and the real sensor aliasing are visible.")
        core.clicked.connect(self._zoom_virtual_core)
        self.virtual_core_button = core
        full.clicked.connect(self.virtual_camera_view.fit_full_frame)
        display.addWidget(QLabel("Colour"))
        display.addWidget(self.virtual_colour_mode, 1)
        display.addWidget(QLabel("Scale"))
        display.addWidget(self.virtual_display_scale, 1)
        box.addLayout(display)
        view_row = QHBoxLayout()
        view_row.addWidget(self.virtual_auto_fit)
        view_row.addWidget(fit, 1)
        view_row.addWidget(core, 1)
        view_row.addWidget(full, 1)
        box.addLayout(view_row)
        overlays = QHBoxLayout()
        self.virtual_overlay_ring = QCheckBox("Ring")
        self.virtual_overlay_ring.setToolTip("Draw the measured principal-ring radius.")
        self.virtual_overlay_ring.setChecked(True)
        self.virtual_overlay_centre = QCheckBox("Centre")
        self.virtual_overlay_centre.setChecked(True)
        self.virtual_overlay_roi = QCheckBox("ROI")
        self.virtual_overlay_roi.setToolTip("Draw the region the metrics are computed in.")
        self.virtual_overlay_roi.setChecked(False)
        for choice in (self.virtual_overlay_ring, self.virtual_overlay_centre, self.virtual_overlay_roi):
            choice.toggled.connect(lambda _checked: self._rerender_virtual())
            overlays.addWidget(choice)
        overlays.addStretch(1)
        box.addLayout(overlays)
        self.virtual_camera_metrics = QLabel("Start live to measure the principal ring radius and roundness.")
        self.virtual_camera_metrics.setObjectName("Muted")
        self.virtual_camera_metrics.setWordWrap(True)
        box.addWidget(self.virtual_camera_metrics)
        box.addWidget(self.virtual_stack_status)
        return card

    def _fold_display_options_into_menu(self) -> None:
        """Colour map, scaling and overlays rarely change; keep them one click away."""

        card_layout = self.virtual_camera_view.parentWidget().layout()
        popup = QWidget()
        form = QFormLayout(popup)
        form.setContentsMargins(10, 8, 10, 8)
        for index in reversed(range(card_layout.count())):
            layout = card_layout.itemAt(index).layout()
            if layout is None:
                continue
            if layout.indexOf(self.virtual_colour_mode) >= 0 or layout.indexOf(self.virtual_overlay_ring) >= 0:
                card_layout.takeAt(index)
                while layout.count():
                    child = layout.takeAt(0).widget()
                    if isinstance(child, QLabel):
                        # Leaving a layout neither hides nor destroys a widget;
                        # a pending deleteLater left it painted over the camera.
                        child.hide()
                        child.setParent(None)
        form.addRow("Colour map", self.virtual_colour_mode)
        form.addRow("Intensity scale", self.virtual_display_scale)
        for check in (self.virtual_overlay_ring, self.virtual_overlay_centre, self.virtual_overlay_roi):
            form.addRow(check)
        menu = QMenu(self)
        action = QWidgetAction(menu)
        action.setDefaultWidget(popup)
        menu.addAction(action)
        view_button = QToolButton()
        view_button.setText("View ▾")
        view_button.setToolTip("Colour map, intensity scale and measurement overlays.")
        view_button.setPopupMode(QToolButton.InstantPopup)
        view_button.setMenu(menu)
        for index in range(card_layout.count()):
            layout = card_layout.itemAt(index).layout()
            if layout is not None and layout.indexOf(self.virtual_auto_fit) >= 0:
                layout.addWidget(view_button)
                break
        self.virtual_view_menu_button = view_button

    AUTO_EXPOSE_TARGET = 0.75

    def _auto_expose_virtual(self) -> None:
        """Choose the exposure this plane needs, the way you would at the bench."""

        try:
            self._ensure_virtual_mode()
            z_mm = float(self.virtual_camera_z.value())
            peak = self.resolution_engine.peak_model_intensity(self.store.snapshot(), z_mm)
            if not np.isfinite(peak) or peak <= 0.0:
                raise RuntimeError("No light at this plane: cast the masks first, or move the camera closer.")
            full_scale = float(
                getattr(self.current_frame, "full_scale", None)
                or self.store.snapshot().camera.full_scale
                or 4095.0
            )
            state = self.store.snapshot()
            before = float(state.camera.exposure_us)
            scale = camera_response_scale(before, float(state.camera.gain))
            wanted = before * (self.AUTO_EXPOSE_TARGET * full_scale) / (peak * scale)
            wanted = float(min(max(wanted, 1.0), 1e6))
            self.exposure.setValue(wanted)
            self._configure_camera()
            self.virtual_bench_status.setText(
                f"Exposure {before:.0f} → {wanted:.0f} µs so the peak at z={z_mm:g} mm sits near "
                f"{self.AUTO_EXPOSE_TARGET * 100:.0f} % of full scale. Note it when comparing captures."
            )
            if self._camera_thread is None or not self._camera_thread.isRunning():
                self._capture_virtual_current()
        except Exception as exc:
            self._show_error("Auto-expose failed", exc)

    def _build_live_strip(self) -> QFrame:
        """The few controls you use while watching the camera; everything else is a tab."""

        card = QFrame()
        card.setObjectName("Panel")
        box = QVBoxLayout(card)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(6)

        position = QHBoxLayout()
        self.virtual_camera_z = NoWheelDoubleSpinBox()
        self.virtual_camera_z.setRange(0.0, 1000.0)
        self.virtual_camera_z.setDecimals(2)
        self.virtual_camera_z.setSingleStep(1.0)
        self.virtual_camera_z.setSuffix(" mm")
        self.virtual_camera_z.setKeyboardTracking(False)
        self.virtual_camera_z.setValue(31.0)
        self.virtual_camera_z.setToolTip("Camera distance after the axicon. Press Enter or ± to move and re-view.")
        self.virtual_camera_z.editingFinished.connect(self._move_camera_if_live)
        self.virtual_camera_step = NoWheelDoubleSpinBox()
        self.virtual_camera_step.setRange(0.01, 100.0)
        self.virtual_camera_step.setDecimals(2)
        self.virtual_camera_step.setValue(1.0)
        self.virtual_camera_step.setSuffix(" mm")
        self.virtual_camera_step.setToolTip("Step size for the − / + buttons.")
        back = QPushButton("−")
        back.setFixedWidth(34)
        back.setToolTip("Move the camera one step towards the axicon.")
        back.clicked.connect(lambda: self._step_virtual_camera(-1))
        forward = QPushButton("+")
        forward.setFixedWidth(34)
        forward.setToolTip("Move the camera one step away from the axicon.")
        forward.clicked.connect(lambda: self._step_virtual_camera(1))
        position.addWidget(QLabel("Camera z"))
        position.addWidget(self.virtual_camera_z, 2)
        position.addWidget(back)
        position.addWidget(forward)
        position.addWidget(self.virtual_camera_step, 1)
        box.addLayout(position)

        live = QHBoxLayout()
        cast_live = QPushButton("▶  Cast masks + go live")
        cast_live.setObjectName("Accent")
        cast_live.setToolTip("Sends both SLM masks to the virtual bench and starts the live camera.")
        cast_live.clicked.connect(lambda: self._start_virtual_live(recast=True))
        stop = QPushButton("■  Stop")
        stop.clicked.connect(self.stop_live)
        one = QPushButton("Snapshot")
        one.setToolTip("Take one fresh frame at this z without starting the live view.")
        one.clicked.connect(self._capture_virtual_current)
        self.virtual_auto_expose = QPushButton("Auto-expose")
        self.virtual_auto_expose.setToolTip(
            "Set the virtual exposure so the brightest pixel sits near 75 % of full scale, as you would at the "
            "bench. Sensitivity is fixed, so frames stay comparable once you note the exposure. A q=0 focus is "
            "far brighter than a q=40 ring, and both need their own exposure."
        )
        self.virtual_auto_expose.clicked.connect(self._auto_expose_virtual)
        box.addWidget(cast_live)
        live.addWidget(stop, 1)
        live.addWidget(one, 1)
        live.addWidget(self.virtual_auto_expose, 1)
        box.addLayout(live)

        self.virtual_cast_status = QLabel("Configured q=+0 / virtual cast q=+0")
        self.virtual_cast_status.setObjectName("Muted")
        self.virtual_cast_status.setWordWrap(True)
        box.addWidget(self.virtual_cast_status)
        self.virtual_bench_status = QLabel(
            "Start on tab 1: set the vortex, then press Cast masks + go live."
        )
        self.virtual_bench_status.setObjectName("Muted")
        self.virtual_bench_status.setWordWrap(True)
        box.addWidget(self.virtual_bench_status)
        return card

    def _add_home_native_toggle(self) -> None:
        card_layout = self.home_camera_view.parentWidget().layout()
        for index in range(card_layout.count()):
            grid = card_layout.itemAt(index).layout()
            if isinstance(grid, QGridLayout) and grid.indexOf(self.home_auto_fit) >= 0:
                self.home_native_toggle = self._make_native_toggle()
                # The grid cell sets its width here; it must not widen the Home page.
                self.home_native_toggle.setMinimumWidth(0)
                policy = self.home_native_toggle.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Ignored)
                self.home_native_toggle.setSizePolicy(policy)
                grid.addWidget(self.home_native_toggle, 2, 3)
                return

    NATIVE_OFF_TEXT = "View: camera pixels"
    NATIVE_ON_TEXT = "View: native model"

    def _make_native_toggle(self) -> QPushButton:
        toggle = QPushButton(self.NATIVE_OFF_TEXT)
        toggle.setCheckable(True)
        toggle.setToolTip(
            "Flip between what the 5.5 µm Beamage pixels record (with detector noise) and the noise-free "
            "native model intensity at 2.75 µm, fully resolved. Same zoom, colours and overlays."
        )
        toggle.toggled.connect(self._on_native_toggled)
        # Same width in both states so the camera does not jump when flipped.
        metrics = toggle.fontMetrics()
        widest = max(metrics.horizontalAdvance(self.NATIVE_OFF_TEXT), metrics.horizontalAdvance(self.NATIVE_ON_TEXT))
        toggle.setMinimumWidth(widest + 34)
        return toggle

    def _native_toggles(self):
        return [t for t in (getattr(self, "virtual_native_toggle", None), getattr(self, "home_native_toggle", None)) if t]

    def _on_native_toggled(self, checked: bool) -> None:
        # One switch in two places: keep them in step.
        for toggle in self._native_toggles():
            if toggle.isChecked() != checked:
                with blocked_signals((toggle,)):
                    toggle.setChecked(checked)
            toggle.setText(self.NATIVE_ON_TEXT if checked else self.NATIVE_OFF_TEXT)
        self.resolution_engine.keep_native_intensity = bool(checked)
        for view in (getattr(self, "virtual_camera_view", None), getattr(self, "home_camera_view", None)):
            if view is not None:
                view._native_render_key = None
        self._rerender_virtual()
        self._rerender()

    def _sync_native_toggles_to_mode(self) -> None:
        # There is no native model behind a real or replayed camera frame.
        virtual = self.mode_controller.operating_mode is OperatingMode.VIRTUAL_LAB
        toggles = self._native_toggles()
        if not virtual and toggles and toggles[0].isChecked():
            toggles[0].setChecked(False)
        for toggle in toggles:
            toggle.setEnabled(virtual)

    def _ensure_virtual_mode(self) -> None:
        super()._ensure_virtual_mode()
        self._sync_native_toggles_to_mode()

    def _native_view_active(self) -> bool:
        toggles = self._native_toggles()
        return bool(
            toggles
            and toggles[0].isChecked()
            and self.mode_controller.operating_mode is OperatingMode.VIRTUAL_LAB
            and self.current_frame is not None
            and self.current_frame.z_mm is not None
        )

    def _show_frame_in(self, view, *, colour: str, scale: str, gamma: float, show_centre: bool,
                       show_ring: bool, show_roi: bool) -> bool:
        """Render the current frame, or the native model for the same plane. Returns True if native."""

        frame = self.current_frame
        if self._native_view_active():
            try:
                with self._virtual_busy("COMPUTING NATIVE VIEW"):
                    native, pixel_scale = self.resolution_engine.native_intensity(self.store.snapshot(), frame.z_mm)
                self._native_sample_um = self.resolution_engine.pixel_size_um * pixel_scale
            except Exception as exc:
                self.virtual_bench_status.setText(f"Native view unavailable: {exc}")
            else:
                # Render raw native intensity using the same fixed response as
                # this captured camera frame, without allocating another field.
                response = frame.metadata.get("counts_per_model_intensity", 0.0)
                native_full_scale = frame.full_scale / response if response > 0 else np.finfo(float).max
                key = (id(native), colour, scale, round(float(gamma), 3), show_centre, show_ring, show_roi, native_full_scale)
                if getattr(view, "_native_render_key", None) != key:
                    # The native model is noise-free, so it only needs drawing
                    # again when the optics, plane or display settings change.
                    view.set_quantitative_frame(
                        frame, self.current_metrics, colour=colour, scale=scale, gamma=gamma,
                        show_centre=show_centre, show_ring=show_ring, show_roi=show_roi,
                        display_data=native, pixel_scale=pixel_scale,
                        display_full_scale=native_full_scale,
                    )
                    view._native_render_key = key
                return True
        view._native_render_key = None
        view.set_quantitative_frame(
            frame, self.current_metrics, colour=colour, scale=scale, gamma=gamma,
            show_centre=show_centre, show_ring=show_ring, show_roi=show_roi,
        )
        return False

    def _rerender(self, *_args) -> None:
        if (
            self.current_frame is not None
            and self.pages.currentIndex() == self.PAGE_HOME
            and hasattr(self, "home_native_toggle")
        ):
            self._show_frame_in(
                self.home_camera_view,
                colour=self.home_colour_mode.currentText(),
                scale=self.home_display_scale.currentText(),
                gamma=self.home_display_gamma.value(),
                show_centre=True,
                show_ring=True,
                show_roi=True,
            )
            return
        super()._rerender()

    def _zoom_virtual_core(self) -> None:
        # Auto-fit would immediately zoom back out to the whole beam.
        self.virtual_auto_fit.setChecked(False)
        # Scale to the brightest pixel so the core ring, not the envelope, sets the colours.
        self.virtual_display_scale.setCurrentText("full range")
        self.virtual_camera_view.fit_core()

    def _move_camera_if_live(self) -> None:
        if self._camera_thread is not None and self._camera_thread.isRunning():
            self._start_virtual_live(recast=False)

    def _build_capture_card(self) -> QFrame:
        card, box = panel(
            "Save evidence",
            "Record repeated synthetic frames at the current camera z. Saved frames are never overwritten, "
            "and their planes can become the planes the correction measures at.",
        )
        form = QFormLayout()
        self.virtual_formal_repeats = NoWheelSpinBox()
        self.virtual_formal_repeats.setRange(1, 100)
        self.virtual_formal_repeats.setValue(3)
        self.virtual_trial_name = QLineEdit(self._new_virtual_trial_name())
        form.addRow("Frames per save", self.virtual_formal_repeats)
        form.addRow("Trial name", self.virtual_trial_name)
        box.addLayout(form)
        buttons = QGridLayout()
        save = QPushButton("Save frames at current z")
        save.setObjectName("Accent")
        save.clicked.connect(self._save_virtual_capture_at_z)
        use_z = QPushButton("Correct at the saved planes")
        use_z.setToolTip("Copies every distinct saved z into the correction plan on tab 3.")
        use_z.clicked.connect(self._use_captured_z_positions)
        measure = QPushButton("Open stored captures")
        measure.clicked.connect(lambda: self.set_page(self.PAGE_MEASURE))
        buttons.addWidget(save, 0, 0)
        buttons.addWidget(use_z, 1, 0)
        buttons.addWidget(measure, 2, 0)
        box.addLayout(buttons)
        return card

    # ------------------------------------------------------ guided correction

    CORRECTION_PRESETS = {
        "Coma": ("coma_x", "coma_y"),
        "Low order": ("defocus", "astig_x", "astig_xy", "coma_x", "coma_y", "spherical"),
        "Everything": ("tip_x", "tip_y", "defocus", "astig_x", "astig_xy", "coma_x", "coma_y", "spherical"),
    }

    @staticmethod
    def _take_widget_from_form(widget: QWidget) -> None:
        """Detach a widget that an inherited builder placed in a QFormLayout row."""
        parent = widget.parentWidget()
        layouts = [parent.layout()] if parent is not None and parent.layout() is not None else []
        while layouts:
            layout = layouts.pop()
            if isinstance(layout, QFormLayout) and layout.indexOf(widget) >= 0:
                row = layout.takeRow(widget)
                if row.labelItem is not None and row.labelItem.widget() is not None:
                    label = row.labelItem.widget()
                    label.hide()
                    label.setParent(None)
                return
            for index in range(layout.count()):
                child = layout.itemAt(index).layout()
                if child is not None:
                    layouts.append(child)

    def _build_correct_panel(self) -> QFrame:
        card, box = panel(
            "Correct the beam",
            "The SLM tries small changes, measures fresh frames at your planes, and keeps only the changes that "
            "lower the beam-quality score J. Nothing reaches the SLM until you press Apply.",
        )
        self.correct_score = QLabel("Score J: not measured yet")
        self.correct_score.setObjectName("Section")
        self.correct_score.setWordWrap(True)
        self.correct_score.setToolTip(
            "J (lower is better) = 3×beam walk between planes + 1×ring-radius variation + 1.5×ring eccentricity "
            "+ 1.5×azimuthal unevenness + 0.5×light in the dark core + 4×edge clipping + 4×saturation."
        )
        box.addWidget(self.correct_score)

        step1 = self._step_heading("Step 1 — What may the SLM change?")
        box.addWidget(step1)
        target_row = QHBoxLayout()
        self._take_widget_from_form(self.virtual_target)
        self.virtual_target.setCurrentText("SLM2")
        target_row.addWidget(QLabel("Correcting"))
        target_row.addWidget(self.virtual_target, 1)
        box.addLayout(target_row)
        preset_row = QHBoxLayout()
        for label, keys in self.CORRECTION_PRESETS.items():
            button = QPushButton(label)
            button.setToolTip("Tick: " + ", ".join(keys))
            button.clicked.connect(lambda _checked=False, chosen=keys: self._set_correction_modes(chosen))
            preset_row.addWidget(button, 1)
        box.addLayout(preset_row)
        modes = QGridLayout()
        short_names = {
            "tip_x": "Tip/tilt X", "tip_y": "Tip/tilt Y", "defocus": "Defocus", "astig_x": "Astig X",
            "astig_xy": "Astig XY", "coma_x": "Coma X", "coma_y": "Coma Y", "spherical": "Spherical",
        }
        for index, (key, check) in enumerate(self.correction_checks.items()):
            self._take_widget_from_form(check)
            check.setToolTip(check.text())
            check.setText(short_names.get(key, check.text()))
            modes.addWidget(check, index // 2, index % 2)
        box.addLayout(modes)

        box.addWidget(self._step_heading("Step 2 — Where should it measure?"))
        planes = QHBoxLayout()
        self._take_widget_from_form(self.virtual_z_plan)
        self.virtual_z_plan.setToolTip("Two or more camera z planes, in mm, separated by commas.")
        planes.addWidget(QLabel("Planes (mm)"))
        planes.addWidget(self.virtual_z_plan, 1)
        box.addLayout(planes)
        around = QPushButton("Use current camera z (+14 mm)")
        around.setToolTip("Measure at the current camera z and 14 mm beyond it.")
        around.clicked.connect(self._correction_planes_around_camera)
        box.addWidget(around)
        self.correct_budget = QLabel()
        self.correct_budget.setObjectName("Muted")
        self.correct_budget.setWordWrap(True)
        box.addWidget(self.correct_budget)

        box.addWidget(self._step_heading("Step 3 — Find a correction"))
        run_row = QHBoxLayout()
        self.virtual_preview_button = QPushButton("Find correction")
        self.virtual_preview_button.setObjectName("Accent")
        self.virtual_preview_button.clicked.connect(self._preview_virtual_correction)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("Danger")
        cancel.clicked.connect(self._cancel_virtual_task)
        run_row.addWidget(self.virtual_preview_button, 2)
        run_row.addWidget(cancel, 1)
        box.addLayout(run_row)
        self.correct_progress = QLabel("Idle.")
        self.correct_progress.setObjectName("Muted")
        self.correct_progress.setWordWrap(True)
        box.addWidget(self.correct_progress)

        box.addWidget(self._step_heading("Step 4 — Review the proposed change"))
        self.correct_table = QTableWidget(0, 4)
        self.correct_table.setHorizontalHeaderLabels(["SLM change", "Now", "Proposed", "Units"])
        self.correct_table.verticalHeader().setVisible(False)
        self.correct_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.correct_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.correct_table.setMinimumHeight(110)
        self.correct_table.setMaximumHeight(170)
        box.addWidget(self.correct_table)
        self.correct_verdict = QLabel("Run step 3 to get a proposal.")
        self.correct_verdict.setWordWrap(True)
        box.addWidget(self.correct_verdict)
        apply_row = QHBoxLayout()
        self.virtual_apply_button = QPushButton("Apply correction")
        self.virtual_apply_button.setObjectName("Accent")
        self.virtual_apply_button.setEnabled(False)
        self.virtual_apply_button.clicked.connect(self._apply_reviewed_virtual_correction)
        self.correct_discard = QPushButton("Discard")
        self.correct_discard.setEnabled(False)
        self.correct_discard.clicked.connect(self._discard_virtual_proposal)
        apply_row.addWidget(self.virtual_apply_button, 2)
        apply_row.addWidget(self.correct_discard, 1)
        box.addLayout(apply_row)

        box.addWidget(self._step_heading("Step 5 — Check the result, then iterate"))
        self.virtual_result_next = QLabel(
            "After applying, the live camera shows the corrected beam and its roundness is compared with before."
        )
        self.virtual_result_next.setObjectName("Muted")
        self.virtual_result_next.setWordWrap(True)
        box.addWidget(self.virtual_result_next)
        next_row = QHBoxLayout()
        self.correct_next = QPushButton("Next iteration")
        self.correct_next.setToolTip("Keeps the applied correction and your settings; clears the old proposal.")
        self.correct_next.clicked.connect(self._next_correction_iteration)
        export_log = QPushButton("Export…")
        export_log.clicked.connect(self._export_virtual_correction_log)
        next_row.addWidget(self.correct_next, 2)
        next_row.addWidget(export_log, 1)
        box.addLayout(next_row)

        self.correct_history = QTableWidget(0, 5)
        self.correct_history.setHorizontalHeaderLabels(["#", "J before", "J after", "Better by", "Result"])
        self.correct_history.verticalHeader().setVisible(False)
        self.correct_history.setEditTriggers(QTableWidget.NoEditTriggers)
        self.correct_history.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.correct_history.setMinimumHeight(100)
        self.correct_history.setMaximumHeight(170)
        box.addWidget(self.correct_history)
        # Codex's text history stays the export/detail source.
        self.virtual_iteration_history = QPlainTextEdit()
        self.virtual_iteration_history.setReadOnly(True)
        self.virtual_iteration_history.setVisible(False)
        box.addWidget(self.virtual_iteration_history)

        self.virtual_z_plan.setText(f"{self.virtual_camera_z.value():g}, {self.virtual_camera_z.value() + 14:g}")
        self._set_correction_modes(self.CORRECTION_PRESETS["Coma"])
        self.virtual_passes.setValue(1)
        for widget in (self.virtual_z_plan,):
            widget.textChanged.connect(lambda _text: self._update_correction_budget())
        for check in self.correction_checks.values():
            check.toggled.connect(lambda _checked: self._update_correction_budget())
        self.virtual_target.currentTextChanged.connect(lambda _text: self._update_correction_budget())
        self.virtual_passes.valueChanged.connect(lambda _value: self._update_correction_budget())
        self._update_correction_budget()
        return card

    @staticmethod
    def _step_heading(text: str) -> QLabel:
        label = QLabel(f"<b>{text}</b>")
        label.setWordWrap(True)
        return label

    def _set_correction_modes(self, keys) -> None:
        for key, check in self.correction_checks.items():
            check.setChecked(key in set(keys))

    def _correction_planes_around_camera(self) -> None:
        z = self.virtual_camera_z.value()
        self.virtual_z_plan.setText(f"{z:g}, {z + 14:g}")

    def _update_correction_budget(self) -> None:
        if not hasattr(self, "correct_budget"):
            return
        try:
            planes = self._parse_z_plan()
        except Exception:
            self.correct_budget.setText("Enter at least two camera planes, e.g. 31, 45.")
            return
        parameters = self._selected_correction_parameters()
        if not parameters:
            self.correct_budget.setText("Tick at least one thing the SLM may change.")
            return
        frames = estimated_blind_cycle_frames(
            len(planes), self.measure_repeats.value(), len(parameters), len(self._selected_targets())
        ) * self.virtual_passes.value()
        # Each optimiser probe is a new optical state, so it costs a full rebuild.
        per_frame = float(getattr(self.resolution_engine, "last_full_rebuild_s", 0.0)) or 3.3
        seconds = frames * per_frame
        self.correct_budget.setText(
            f"About {frames} fresh frames at {len(planes)} planes — roughly "
            f"{seconds / 60:.0f} min on the live grid." if seconds >= 90 else
            f"About {frames} fresh frames at {len(planes)} planes — roughly {seconds:.0f} s on the live grid."
        )

    def _discard_virtual_proposal(self) -> None:
        proposal = self._pending_virtual_proposal
        if proposal is not None and proposal.get("history_index") is not None:
            self._virtual_correction_history[proposal["history_index"]]["status"] = "DISCARDED"
        self._pending_virtual_proposal = None
        self.virtual_apply_button.setEnabled(False)
        self.correct_discard.setEnabled(False)
        self.correct_table.setRowCount(0)
        self.correct_verdict.setText("Proposal discarded. The SLMs were not changed.")
        self._refresh_iteration_history()

    def _next_correction_iteration(self) -> None:
        if self._pending_virtual_proposal is not None:
            self._discard_virtual_proposal()
        self.correct_table.setRowCount(0)
        self.correct_discard.setEnabled(False)
        number = len(self._virtual_correction_history) + 1
        self.correct_verdict.setText(f"Ready for iteration {number}. Press Find correction.")
        self.correct_progress.setText("Idle.")
        area = self.virtual_tabs.widget(self.virtual_tab_indices["correction"])
        area.verticalScrollBar().setValue(0)

    def _show_correction_proposal(self, result) -> None:
        trial = result["result"]
        signature = result["baseline_signature"]
        self.correct_table.setRowCount(0)
        for row in trial.accepted_runs:
            spec = PARAMETERS[row["parameter"]]
            before = float(signature[row["slm"].lower()][spec.attribute])
            name = f"{row['slm']} {row['parameter'].replace('_', ' ')}"
            proposed = f"{row['accepted_command']:+.4g}" if row["accepted"] else "no change"
            index = self.correct_table.rowCount()
            self.correct_table.insertRow(index)
            for column, text in enumerate((name, f"{before:+.4g}", proposed, spec.units)):
                self.correct_table.setItem(index, column, QTableWidgetItem(text))
        before, after = float(trial.initial_objective), float(trial.final_objective)
        percent = 100.0 * float(trial.improvement_fraction)
        applicable = self._pending_virtual_proposal is result
        if applicable:
            self.correct_score.setText(f"Score J: {before:.4g} now → {after:.4g} with this change ({percent:.1f}% better)")
        else:
            self.correct_score.setText(f"Score J: {before:.4g} (lower is better) — no verified improvement found")
        if trial.cancelled:
            verdict = "Cancelled — nothing to apply."
        elif applicable:
            verdict = (
                f"✔ Improves J by {percent:.1f}% on fresh frames at {len(result.get('z_plan', []))} planes. "
                "Apply it, or Discard and change step 1 or 2."
            )
        else:
            verdict = (
                "✘ No change lowered J. Try more modes in step 1 (e.g. Low order), different planes, "
                "or the other SLM."
            )
        self.correct_verdict.setText(verdict)
        self.correct_discard.setEnabled(applicable)

    def _refresh_iteration_history(self) -> None:
        self._refresh_iteration_history_text()
        if not hasattr(self, "correct_history"):
            return
        self.correct_history.setRowCount(0)
        for number, entry in enumerate(self._virtual_correction_history, 1):
            row = self.correct_history.rowCount()
            self.correct_history.insertRow(row)
            status = {
                "APPLIED TO VIRTUAL SLMS": "Applied",
                "PREVIEW ONLY": "Awaiting review",
                "NO APPLICABLE CANDIDATE": "Nothing to apply",
                "DISCARDED": "Discarded",
            }.get(entry["status"], entry["status"].capitalize())
            if entry.get("observed"):
                status += f", roundness {entry['observed']['roundness']:.3f}"
            candidate = entry["status"] in ("APPLIED TO VIRTUAL SLMS", "PREVIEW ONLY", "DISCARDED")
            values = (
                str(number),
                f"{entry['before']:.4g}",
                f"{entry['candidate']:.4g}" if candidate else "—",
                f"{entry['improvement']:.1f}%" if candidate else "—",
                status,
            )
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 4 and entry.get("commands"):
                    item.setToolTip("\n".join(
                        f"{c['slm']} {c['parameter']}: {c['before']:+.4g} → {c['after']:+.4g} {c['units']}"
                        for c in entry["commands"]
                    ))
                self.correct_history.setItem(row, column, item)
        self.correct_history.scrollToBottom()

    @Slot(object)
    def _virtual_progress_event(self, payload) -> None:
        super()._virtual_progress_event(payload)
        if hasattr(self, "correct_progress"):
            self.correct_progress.setText(self.virtual_progress.document().lastBlock().text() or "Working…")

    # --------------------------------------------------------------- SLM masks

    def _build_mask_card(self) -> QFrame:
        card, box = panel(
            "SLM phase masks",
            "Route the vortex, or edit either complete mask by hand. The virtual optical field changes only on cast.",
        )
        form = QFormLayout()
        self.virtual_vortex_route = NoWheelComboBox()
        self.virtual_vortex_route.addItems(
            ["Keep manual SLM phases", "SLM1 only", "SLM2 only", "Split across SLM1 + SLM2"]
        )
        self.virtual_vortex_route.setCurrentText("SLM1 only")
        self.virtual_route_charge = NoWheelSpinBox()
        self.virtual_route_charge.setRange(-100, 100)
        self.virtual_route_charge.setValue(20)
        form.addRow("Vortex phase route", self.virtual_vortex_route)
        form.addRow("Total charge q", self.virtual_route_charge)
        # One authoritative radius control: move the existing widget out of
        # the fault subtab rather than maintaining two competing values.
        self.perturb_beam_form.takeRow(self.perturb_beam_radius_mm)
        form.addRow("Beam radius (1/e)", self.perturb_beam_radius_mm)
        box.addLayout(form)
        self.perturb_beam_radius_mm.valueChanged.connect(self._on_input_radius_edited)
        radius_apply = QPushButton("Apply beam size")
        radius_apply.clicked.connect(self._apply_input_beam_radius)
        box.addWidget(radius_apply)
        design_button = QPushButton("Small-beam designer…")
        design_button.setToolTip("Explore the repository's target size/length and required objective mapping; no bench changes.")
        design_button.clicked.connect(self._open_small_beam_designer)
        box.addWidget(design_button)
        self.virtual_small_beam_button = design_button
        self.virtual_input_radius_status = QLabel(
            "2.00 mm radius is active. The dashed footprint on each pixelated phase preview is a model reference; "
            "the SLM2 relay footprint has not been physically calibrated. Fixed incident power; changing size does not edit phase pixels."
        )
        self.virtual_input_radius_status.setObjectName("Muted")
        self.virtual_input_radius_status.setWordWrap(True)
        box.addWidget(self.virtual_input_radius_status)

        route_row = QHBoxLayout()
        route = QPushButton("Set route (configure)")
        route.clicked.connect(self._apply_virtual_vortex_route)
        route_row.addWidget(route)
        box.addLayout(route_row)

        previews = QHBoxLayout()
        self.virtual_mask_previews: dict[str, tuple[QLabel, QLabel]] = {}
        for name in ("SLM1", "SLM2"):
            holder = QFrame()
            holder.setObjectName("Panel")
            holder_box = QVBoxLayout(holder)
            title = QLabel(name)
            title.setObjectName("Section")
            image = QLabel("not generated")
            image.setAlignment(Qt.AlignCenter)
            image.setMinimumSize(90, 120)
            image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            image.setObjectName("ImageWell")
            status = QLabel("")
            status.setObjectName("Muted")
            status.setWordWrap(True)
            edit = QPushButton("Edit…")
            edit.setToolTip(f"Open the full {name} phase editor.")
            page = self.PAGE_SLM1 if name == "SLM1" else self.PAGE_SLM2
            edit.clicked.connect(lambda _checked=False, target=page: self.set_page(target))
            holder_box.addWidget(title)
            holder_box.addWidget(image, 1)
            holder_box.addWidget(status)
            holder_box.addWidget(edit)
            previews.addWidget(holder, 1)
            self.virtual_mask_previews[name] = (image, status)
        box.addLayout(previews)

        self.virtual_mask_status = QLabel(
            "Every phase layer on the SLM 1 / SLM 2 pages - vortex, blaze carrier, Zernike commands, "
            "wavefront files, retrieved corrections - is what the virtual bench propagates once cast."
        )
        self.virtual_mask_status.setObjectName("Muted")
        self.virtual_mask_status.setWordWrap(True)
        box.addWidget(self.virtual_mask_status)
        return card

    # ---------------------------------------------------------------- feedback

    @contextmanager
    def _virtual_busy(self, message: str):
        """Show that a synchronous propagation is running rather than hanging."""

        chip = getattr(self, "virtual_busy_chip", None)
        if chip is not None:
            chip.setText(message)
        QApplication.setOverrideCursor(Qt.BusyCursor)
        QApplication.processEvents()
        try:
            yield
        finally:
            QApplication.restoreOverrideCursor()
            if chip is not None:
                chip.setText("SYNTHETIC")

    def _step_virtual_camera(self, direction: int) -> None:
        step = self.virtual_camera_step.value() * (1 if direction >= 0 else -1)
        self.virtual_camera_z.setValue(max(0.0, self.virtual_camera_z.value() + step))
        self._start_virtual_live(recast=False)

    def _refresh_virtual_chips(self) -> None:
        if not hasattr(self, "virtual_live_chip"):
            return
        running = self._camera_thread is not None and self._camera_thread.isRunning()
        paused = self._resume_virtual_live_after_navigation and self.pages.currentIndex() not in {
            self.PAGE_HOME, self.PAGE_MEASURE, self.PAGE_VIRTUAL,
        }
        self.virtual_live_chip.setText("PAUSED" if paused else ("LIVE" if running else "STOPPED"))
        self.virtual_live_chip.setObjectName("WarnChip" if paused else ("GoodChip" if running else "BadChip"))
        self.virtual_live_chip.style().unpolish(self.virtual_live_chip)
        self.virtual_live_chip.style().polish(self.virtual_live_chip)

    def _rerender_virtual(self) -> None:
        if not hasattr(self, "virtual_colour_mode"):
            super()._rerender_virtual()
            return
        if self.current_frame is None or self.pages.currentIndex() != self.PAGE_VIRTUAL:
            return
        native = self._show_frame_in(
            self.virtual_camera_view,
            colour=self.virtual_colour_mode.currentText(),
            scale=self.virtual_display_scale.currentText(),
            gamma=1.0,
            show_centre=self.virtual_overlay_centre.isChecked(),
            show_ring=self.virtual_overlay_ring.isChecked(),
            show_roi=self.virtual_overlay_roi.isChecked(),
        )
        if self.virtual_auto_fit.isChecked():
            self.virtual_camera_view.fit_signal()
        frame = self.current_frame
        height, width = frame.shape_yx
        if native:
            self.virtual_frame_chip.setText(
                f"NATIVE {self._native_sample_um:.2f} µm | z={frame.z_mm:g} mm | noise-free, not recorded"
            )
        else:
            clipped = bool(self.current_metrics and self.current_metrics.values.get("clipped"))
            self.virtual_frame_chip.setText(
                f"{width}x{height} | z={frame.z_mm if frame.z_mm is not None else '--'} mm | "
                f"peak {float(frame.data.max()):.0f} counts"
                + (" | CLIPPED — press Auto-expose" if clipped else "")
            )
        metrics = self.current_metrics
        if metrics is None:
            self.virtual_camera_metrics.setText("Ring measurements updating; the full-resolution frame is still visible.")
        elif metrics.family == "vortex_bessel":
            radius_px = float(metrics.values.get("principal_ring_radius_px", 0.0))
            pixel_um = float(self.resolution_engine.pixel_size_um)
            eccentricity = float(metrics.values.get("ring_eccentricity", 0.0))
            roundness = max(0.0, 1.0 - eccentricity)
            az_cv = float(metrics.values.get("azimuthal_cv", 0.0))
            beam = self._second_moment_summary(self.current_frame, pixel_um)
            text = (
                f"Beam radius (D4σ/2) {beam}  •  ring radius {radius_px * pixel_um:.0f} µm  •  "
                f"ring roundness {roundness:.3f}  •  unevenness {az_cv:.3f}"
            )
            demagnification = float(self.resolution_engine.geometry.objective_demagnification or 1.0)
            if demagnification < 1.0:
                # The same frame, read at the sample of the fitted objective.
                z_mm = self.current_frame.z_mm if self.current_frame.z_mm is not None else 0.0
                text += (
                    f"\nSAMPLE PLANE 1:{1.0 / demagnification:g} — ring radius "
                    f"{radius_px * pixel_um * demagnification:.2f} µm at z="
                    f"{float(z_mm) * demagnification ** 2 * 1e3:.1f} µm, {pixel_um * demagnification:.3f} µm per pixel"
                )
            self.virtual_camera_metrics.setText(text)
            self.virtual_camera_metrics.setToolTip(
                "Beam radius: half the ISO 11146 second-moment (D4σ) width, as a Beamage reports it; x/y and the "
                "minor/major ratio give the beam's circularity.\nRing radius: radius of the brightest vortex ring "
                "(a radial-profile peak, not an elliptical boundary).\nRing roundness: 1 − ring eccentricity.\n"
                "Unevenness: azimuthal coefficient of variation around the ring (0 = perfectly even).\n"
                "All µm are model µm, not calibrated Beamage µm."
            )
        else:
            self.virtual_camera_metrics.setText(
                f"Analysis family: {metrics.family}. Ring radius/roundness requires an active vortex and a resolved annulus."
            )
        self._refresh_virtual_chips()

    @staticmethod
    def _second_moment_summary(frame, pixel_um: float) -> str:
        """ISO 11146 second-moment radius and circularity of the whole beam."""
        if frame is None:
            return "—"
        data = np.asarray(frame.data, dtype=np.float64)
        step = max(1, data.shape[0] // 512)  # moments are stable on a decimated frame
        data = data[::step, ::step]
        data = np.clip(data - np.median(data), 0.0, None)
        total = float(data.sum())
        if total <= 0.0:
            return "—"
        y, x = np.indices(data.shape, dtype=np.float64)
        cx, cy = float((data * x).sum() / total), float((data * y).sum() / total)
        sxx = float((data * (x - cx) ** 2).sum() / total)
        syy = float((data * (y - cy) ** 2).sum() / total)
        sxy = float((data * (x - cx) * (y - cy)).sum() / total)
        spread = math.sqrt(max((sxx - syy) ** 2 + 4.0 * sxy ** 2, 0.0))
        major = math.sqrt(max(0.5 * (sxx + syy + spread), 0.0))
        minor = math.sqrt(max(0.5 * (sxx + syy - spread), 0.0))
        scale = 2.0 * step * pixel_um / 1000.0
        circularity = minor / major if major > 0 else 1.0
        return (
            f"{math.sqrt(sxx) * scale:.2f} × {math.sqrt(syy) * scale:.2f} mm, "
            f"circularity {circularity:.3f}"
        )

    def _refresh_state(self, state, *, refresh_slms=None) -> None:
        super()._refresh_state(state, refresh_slms=refresh_slms)
        self._sync_native_toggles_to_mode()
        if hasattr(self, "virtual_cast_status"):
            cast_q = self.resolution_engine.cast_vortex_charge()
            self.virtual_cast_status.setText(
                f"Configured q={state.effective_vortex_charge():+d} • "
                f"virtual cast q={cast_q:+d} • camera z={state.camera.current_z_mm if state.camera.current_z_mm is not None else 'unset'} mm"
            )
        # SLM thumbnails are built from full native masks. Camera-frame state
        # events do not change them, and repainting both on every frame caused
        # avoidable full-panel copies while the live view was running.
        if refresh_slms is None or refresh_slms:
            self._refresh_virtual_masks(state)
        self._refresh_virtual_chips()

    def _refresh_virtual_masks(self, state) -> None:
        """Mirror the authoritative configured phase next to the bench controls."""

        previews = getattr(self, "virtual_mask_previews", None)
        if not previews:
            return
        bundle = self.controller.last_bundle
        for name, (image, status) in previews.items():
            slm = getattr(state, name.lower())
            cast_hash = self.resolution_engine.cast_hash(name)
            if bundle is not None and bundle.hashes.get(name) == slm.complete_phase_sha256:
                pixmap = phase_pixmap(bundle.results[name].gray_uint8, 220, 118)
                geometry = slm.phase.geometry
                radius_native_px = self.perturb_beam_radius_mm.value() * 1000.0 / geometry.pixel_pitch_um
                radius_preview_px = radius_native_px * pixmap.width() / geometry.width_px
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(QColor("#64e6cf"), 2, Qt.DashLine))
                painter.drawEllipse(QPointF(pixmap.width() / 2, pixmap.height() / 2), radius_preview_px, radius_preview_px)
                painter.end()
                image.setPixmap(pixmap)
            if cast_hash and cast_hash == slm.complete_phase_sha256:
                text = "cast on the virtual bench"
                object_name = "StatusGood"
            elif cast_hash:
                text = "EDITED SINCE CAST — cast to update the optical field"
                object_name = "StatusWarn"
            else:
                text = "never cast on the virtual bench"
                object_name = "StatusWarn"
            layers = [
                layer
                for layer, enabled in slm.phase.switches.__dict__.items()
                if enabled and layer != "circular_pupil"
            ]
            status.setText(
                f"q={slm.phase.vortex_charge:+d} • {text}\nlayers: " + (", ".join(layers) or "none")
                + f"\nDashed cyan: nominal {self.perturb_beam_radius_mm.value():g} mm input-radius reference"
            )
            status.setObjectName(object_name)
            status.style().unpolish(status)
            status.style().polish(status)

    @staticmethod
    def _new_virtual_trial_name() -> str:
        return "virtual_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    def _show_virtual_perturbations(self) -> None:
        self._show_virtual_tab("faults")

    def _on_input_radius_edited(self, _value: float) -> None:
        active = self.resolution_engine.canonical_beam_radius_mm
        proposed = self.perturb_beam_radius_mm.value()
        slm1_geometry = self.store.snapshot().slm1.phase.geometry
        half_short_edge_mm = slm1_geometry.height_px * slm1_geometry.pixel_pitch_um / 2000.0
        warning = (
            " At this size the nominal Gaussian extends beyond the model window or SLM short edge; "
            "clipping/aliasing may dominate."
            if proposed > min(self.resolution_engine.geometry.simulation_window_mm / 2.0, half_short_edge_mm)
            else ""
        )
        self.virtual_input_radius_status.setText(
            f"Active input radius: {active:g} mm. Edited value: {proposed:g} mm — "
            "press Apply input size to propagate it. Dashed circles are reference footprints, "
            "not changes to the phase pixels; SLM2 relay illumination is uncalibrated." + warning
        )
        self._refresh_virtual_masks(self.store.snapshot())

    def _apply_input_beam_radius(self) -> None:
        try:
            if self._virtual_thread is not None and self._virtual_thread.isRunning():
                raise RuntimeError("Wait for the correction preview to finish before changing the input beam.")
            self._ensure_virtual_mode()
            value = self.perturb_beam_radius_mm.value()
            self.resolution_engine.set_canonical_beam_radius_mm(value)
            self._pending_virtual_proposal = None
            self.virtual_apply_button.setEnabled(False)
            self.current_frame = None
            self.current_metrics = None
            self.virtual_camera_view.release_frame()
            self.virtual_frame_chip.setText("Input size changed — waiting for a fresh synthetic frame")
            self.virtual_input_radius_status.setText(
                f"Active Gaussian input 1/e amplitude radius: {value:g} mm. "
                "The next live frame will re-propagate it through both virtual SLMs and the axicon; "
                "no SLM phase cast is required for this amplitude change. The SLM2 footprint is a nominal reference only."
            )
            self.virtual_bench_status.setText(
                f"Input radius set to {value:g} mm. Start live, or wait for the next live frame, "
                "to see its downstream effect. Any older correction proposal was invalidated."
            )
            self._refresh_virtual_masks(self.store.snapshot())
        except Exception as exc:
            self._show_error("Could not apply virtual input beam size", exc)

    def _apply_manual_perturbations(self) -> None:
        before = self.resolution_engine.scenario.truth_hash
        before_radius = self.resolution_engine.canonical_beam_radius_mm
        super()._apply_manual_perturbations()
        if (self.resolution_engine.scenario.truth_hash == before
                and self.resolution_engine.canonical_beam_radius_mm == before_radius):
            return
        self._pending_virtual_proposal = None
        if hasattr(self, "virtual_apply_button"):
            self.virtual_apply_button.setEnabled(False)
            self.virtual_camera_view.release_frame()
            self.virtual_frame_chip.setText("Faults changed — waiting for a fresh synthetic frame")
            self.virtual_input_radius_status.setText(
                f"Active Gaussian input 1/e amplitude radius: {self.resolution_engine.canonical_beam_radius_mm:g} mm. "
                "The next live frame will use it with the new faults; SLM2 relay illumination is uncalibrated."
            )
            self.virtual_bench_status.setText(
                "Hidden beam/axicon/wavefront faults applied. The next live frame uses them. "
                "Capture two or more z planes, then preview correction; any earlier proposal is invalid."
            )

    def _apply_virtual_vortex_route(self) -> None:
        try:
            self._ensure_virtual_mode()
            route = self.virtual_vortex_route.currentText()
            q = self.virtual_route_charge.value()
            if route == "Keep manual SLM phases":
                self.virtual_bench_status.setText("Manual SLM phases kept. Cast the configured masks to update the live camera.")
                return

            def update(state):
                if route == "SLM1 only":
                    charges = (q, 0)
                elif route == "SLM2 only":
                    charges = (0, q)
                else:
                    charges = (q // 2, q - q // 2)
                for slm, charge in zip((state.slm1, state.slm2), charges):
                    slm.phase.vortex_charge = charge
                    slm.phase.switches.vortex = charge != 0

            self.store.update(update, source="virtual_operator", reason=f"Configured q={q} via {route}")
            self.controller.generate()
            self._pending_virtual_proposal = None
            self.virtual_apply_button.setEnabled(False)
            self.virtual_bench_status.setText(
                f"Configured q={q} via {route}. Now cast both masks; this has not changed the virtual optical field yet."
            )
        except Exception as exc:
            self._show_error("Could not set virtual vortex route", exc)

    def _configure_virtual_camera_model(self) -> None:
        self.resolution_engine.set_quality(self.virtual_quality.currentText())
        self.resolution_engine.realistic_camera = self.virtual_realistic_camera.isChecked()

    def _start_virtual_live(self, *, recast: bool) -> None:
        try:
            report = self._axicon_report()
            if not report["valid"]:
                raise ValueError(str(report["message"]))
            with self._virtual_busy(f"PROPAGATING {self.resolution_engine.grid_n}²"):
                self._ensure_virtual_mode()
                self._configure_virtual_camera_model()
                self.mode_controller.move_stage(self.virtual_camera_z.value())
                if recast or any(
                    self.resolution_engine.cast_hash(name) is None for name in ("SLM1", "SLM2")
                ):
                    self._ensure_virtual_cast()
                self.start_live()
            self.virtual_bench_status.setText(
                f"Live at z={self.virtual_camera_z.value():g} mm, q={self.store.snapshot().effective_vortex_charge():+d}."
            )
        except Exception as exc:
            self._show_error("Could not start virtual live view", exc)

    def _save_virtual_capture_at_z(self) -> None:
        was_live = self._camera_thread is not None and self._camera_thread.isRunning()
        stopped = False
        previous_count = len(self._history)
        try:
            if not self.stop_live():
                raise RuntimeError("Live camera is finishing an in-flight frame; retry capture once it stops.")
            stopped = True
            with self._virtual_busy(f"CAPTURING {self.virtual_formal_repeats.value()} FRAMES"):
                self._ensure_virtual_mode()
                self._configure_virtual_camera_model()
                self.mode_controller.move_stage(self.virtual_camera_z.value())
                self.capture_repeats.setValue(self.virtual_formal_repeats.value())
                self.trial_id.setText(self.virtual_trial_name.text().strip())
                self._formal_capture()
            if len(self._history) > previous_count:
                record = self._history[-1]
                self.virtual_bench_status.setText(
                    f"Saved {self.virtual_formal_repeats.value()} synthetic frames at "
                    f"z={record['z_mm']:g} mm in {record['root']}. "
                    "Move to another z for a multi-plane correction."
                )
                self.virtual_trial_name.setText(self._new_virtual_trial_name())
        except Exception as exc:
            self._show_error("Could not save virtual capture", exc)
        finally:
            if len(self._history) == previous_count:
                # Formal trials are never overwritten, so leaving a spent name in
                # the box would make the next attempt fail the same way.
                self.virtual_trial_name.setText(self._new_virtual_trial_name())
            if was_live and stopped:
                self._start_virtual_live(recast=False)

    def _use_captured_z_positions(self) -> None:
        z_values = sorted(
            {float(row["z_mm"]) for row in self._history if row.get("data_kind") == "SYNTHETIC" and row.get("z_mm") is not None}
        )
        if len(z_values) < 2:
            self.virtual_bench_status.setText(
                "At least two distinct saved z positions are needed for a propagation-based correction. "
                "One plane can still be stored and viewed."
            )
            return
        self.virtual_z_plan.setText(", ".join(f"{z:g}" for z in z_values))
        self.virtual_bench_status.setText(
            f"Correction z plan set to {z_values}. The optimiser uses fresh virtual frames at these positions; "
            "saved captures remain immutable evidence."
        )

    def _setup_quick_correction(self) -> None:
        saved = sorted({
            float(row["z_mm"]) for row in self._history
            if row.get("data_kind") == "SYNTHETIC" and row.get("z_mm") is not None
        })
        z_pair = (saved[0], saved[-1]) if len(saved) >= 2 else (31.0, 45.0)
        self.virtual_z_plan.setText(", ".join(f"{z:g}" for z in z_pair))
        self.virtual_target.setCurrentText("SLM2")
        self.virtual_passes.setValue(1)
        for name, choice in self.correction_checks.items():
            choice.setChecked(name in {"coma_x", "coma_y"})
        self._update_frame_budget_preview()
        self.virtual_bench_status.setText(
            f"Quick first pass configured at z={z_pair}: SLM2 coma X/Y only, one pass, "
            "about 30 fresh synthetic frames. Open SLM modes below to change what the optimiser may correct. "
            "Preview does not apply anything."
        )

    def _preview_signature(self, state):
        return {
            "slm1": state.slm1.phase.as_dict(),
            "slm2": state.slm2.phase.as_dict(),
            "scenario_truth_hash": self.resolution_engine.scenario.truth_hash,
            "geometry": self.resolution_engine.geometry.public_summary(),
            "quality": self.resolution_engine.quality,
            "resolution": self.resolution_engine.output_resolution.value,
            "input_radius_mm": self.resolution_engine.canonical_beam_radius_mm,
            "z_plan": self.virtual_z_plan.text().strip(),
            "exposure_us": state.camera.exposure_us,
            "gain": state.camera.gain,
        }

    AXICON_BENCH = "Bench axicon (Thorlabs '20°', measured k⊥)"
    AXICON_KPERP = "Custom k⊥"
    AXICON_ANGLE = "Custom model base angle"

    def _build_axicon_controls(self) -> None:
        """Axicon choice by what the bench actually measured, not by its label."""

        geometry = self.resolution_engine.geometry
        form = QFormLayout()
        self.virtual_axicon_mode = NoWheelComboBox()
        self.virtual_axicon_mode.addItems([self.AXICON_BENCH, self.AXICON_KPERP, self.AXICON_ANGLE])
        self.virtual_axicon_kperp = NoWheelDoubleSpinBox()
        self.virtual_axicon_kperp.setRange(0.05, 30.0)
        self.virtual_axicon_kperp.setDecimals(3)
        self.virtual_axicon_kperp.setSingleStep(0.1)
        self.virtual_axicon_kperp.setSuffix(" ×10⁵ m⁻¹")
        self.virtual_axicon_kperp.setValue((geometry.axicon_k_perp_m_inv or MEASURED_BENCH_AXICON_K_PERP_M_INV) / 1e5)
        self.virtual_axicon_kperp.setToolTip(
            "Transverse wavenumber of the cone. The bench value was fitted from the real BeamGage q=20 z-scan."
        )
        self.virtual_axicon_angle.setToolTip(
            "Internal base angle in the exact refractive convention. The optic's '20°' label is NOT this angle: "
            "the measured bench axicon corresponds to about 9.7°."
        )
        self.virtual_fourier_stop = NoWheelDoubleSpinBox()
        self.virtual_fourier_stop.setRange(0.0, 50.0)
        self.virtual_fourier_stop.setDecimals(2)
        self.virtual_fourier_stop.setSingleStep(0.5)
        self.virtual_fourier_stop.setSuffix(" mm")
        self.virtual_fourier_stop.setSpecialValueText("ideal (no filtering)")
        self.virtual_fourier_stop.setValue(float(geometry.fourier_aperture_diameter_mm or 0.0))
        self.virtual_fourier_stop.setToolTip(
            "Diameter of the stop in the shared focal plane of the 4F (SLM–300–L1–300–stop–300–L2–300–axicon). "
            "It passes SLM detail down to D/(2λf); a few mm blocks the unwanted orders without touching the +1 order."
        )
        self.virtual_objective = NoWheelSpinBox()
        self.virtual_objective.setRange(1, 200)
        self.virtual_objective.setPrefix("1:")
        self.virtual_objective.setValue(int(round(1.0 / float(geometry.objective_demagnification or 1.0))))
        self.virtual_objective.setToolTip(
            "Demagnifying objective after the axicon — the only way to shrink the rings, since the 4F relay is 1:1 "
            "and a wider input beam only lengthens the Bessel region. The pattern scales exactly: rings ×M, z ×M². "
            "1:1 means no objective."
        )
        form.addRow("Axicon", self.virtual_axicon_mode)
        form.addRow("k⊥", self.virtual_axicon_kperp)
        form.addRow("4F stop Ø", self.virtual_fourier_stop)
        form.addRow("Objective", self.virtual_objective)
        card_layout = self.virtual_geometry_card.layout()
        card_layout.insertLayout(max(0, card_layout.count() - 1), form)
        self.virtual_angle_status = QLabel()
        self.virtual_angle_status.setObjectName("Muted")
        self.virtual_angle_status.setWordWrap(True)
        card_layout.addWidget(self.virtual_angle_status)
        self.virtual_sample_status = QLabel()
        self.virtual_sample_status.setObjectName("Muted")
        self.virtual_sample_status.setWordWrap(True)
        card_layout.addWidget(self.virtual_sample_status)
        self.virtual_axicon_mode.setCurrentText(
            self.AXICON_BENCH if geometry.axicon_k_perp_m_inv is not None else self.AXICON_ANGLE
        )
        if hasattr(self, "virtual_route_charge"):
            self.virtual_route_charge.valueChanged.connect(lambda *_args: self._update_relay_and_sample_status())
        for signal in (
            self.virtual_axicon_mode.currentTextChanged,
            self.virtual_axicon_kperp.valueChanged,
            self.virtual_axicon_angle.valueChanged,
            self.virtual_fourier_stop.valueChanged,
            self.virtual_objective.valueChanged,
        ):
            signal.connect(lambda *_args: self._update_axicon_angle_status())
        self._update_axicon_angle_status()

    def _proposed_geometry(self):
        geometry = self.resolution_engine.geometry
        mode = self.virtual_axicon_mode.currentText() if hasattr(self, "virtual_axicon_mode") else self.AXICON_ANGLE
        if mode == self.AXICON_BENCH:
            k_perp, source = MEASURED_BENCH_AXICON_K_PERP_M_INV, MEASURED_BENCH_AXICON_SOURCE
        elif mode == self.AXICON_KPERP:
            k_perp, source = self.virtual_axicon_kperp.value() * 1e5, "operator-entered k_perp"
        else:
            k_perp, source = None, "operator-entered model base angle"
        stop_mm = self.virtual_fourier_stop.value() if hasattr(self, "virtual_fourier_stop") else 0.0
        objective = self.virtual_objective.value() if hasattr(self, "virtual_objective") else 1
        return dataclasses.replace(
            geometry,
            slm1_to_slm2_mm=self.virtual_slm_sep.value(),
            axicon_k_perp_m_inv=k_perp,
            axicon_k_perp_source=source,
            axicon_model_base_angle_deg=self.virtual_axicon_angle.value(),
            fourier_aperture_diameter_mm=stop_mm if stop_mm > 0.0 else None,
            fourier_aperture_source=(
                "operator-entered stop diameter" if stop_mm > 0.0 else "not_measured_default_ideal_order_select"
            ),
            objective_demagnification=1.0 / float(objective),
            objective_source=(
                "operator-entered demagnifying objective after the axicon" if objective > 1
                else "not_installed_default_unity"
            ),
        )

    def _open_small_beam_designer(self) -> None:
        from .small_beam_designer import SmallBeamDesigner
        if not hasattr(self, "_small_beam_dialog"):
            self._small_beam_dialog = SmallBeamDesigner(self)
        self._small_beam_dialog.show()
        self._small_beam_dialog.raise_()

    def _axicon_report(self, geometry=None, *, grid_n: int | None = None) -> dict[str, object]:
        """Plain-language summary of what the engine will compute for this axicon."""

        engine = self.resolution_engine
        geometry = engine.geometry if geometry is None else geometry
        plan = engine.axicon_plan(geometry=geometry, grid_n=grid_n)
        if not plan.valid:
            return {
                "valid": False,
                "message": f"NOT APPLIED: {plan.message} The previous axicon is kept.",
                "samples_per_period": plan.samples_per_period,
            }
        z = self.virtual_camera_z.value() if hasattr(self, "virtual_camera_z") else 31.0
        radius = getattr(engine, "canonical_beam_radius_mm", 2.0)
        message = (
            f"k⊥ {plan.k_perp_m_inv / 1e5:.3f}×10⁵ m⁻¹ = {plan.base_angle_deg:.2f}° base angle, "
            f"{plan.cone_half_angle_deg:.2f}° cone.  Equivalent q=0 first-null radius ≈ {plan.core_radius_um:.1f} µm (not the vortex ring); "
            f"Gaussian axial length scale ≈ {plan.bessel_zone_mm:.1f} mm for a {radius:g} mm beam (not a hard edge). "
            f"Camera z={z:g} mm. "
            f"{plan.message}"
        )
        return {"valid": True, "message": message, "samples_per_period": plan.samples_per_period}

    def _axicon_angle_report(self, angle_deg: float, *, grid_n: int | None = None) -> dict[str, object]:
        """Report for a literal model base angle (kept for callers that pass one)."""
        geometry = dataclasses.replace(
            self.resolution_engine.geometry,
            axicon_k_perp_m_inv=None,
            axicon_model_base_angle_deg=float(angle_deg),
        )
        return self._axicon_report(geometry, grid_n=grid_n)

    def _update_axicon_angle_status(self) -> None:
        if not hasattr(self, "virtual_angle_status") or not hasattr(self, "virtual_axicon_mode"):
            return
        mode = self.virtual_axicon_mode.currentText()
        self.virtual_axicon_kperp.setEnabled(mode == self.AXICON_KPERP)
        self.virtual_axicon_angle.setEnabled(mode == self.AXICON_ANGLE)
        active = self._axicon_report()
        proposed = self._axicon_report(self._proposed_geometry())
        text = f"ACTIVE — {active['message']}"
        if proposed["message"] != active["message"]:
            text += f"\n\nIF APPLIED — {proposed['message']}"
        self.virtual_angle_status.setText(text)
        self.virtual_angle_status.setObjectName("Muted" if proposed["valid"] else "WarnChip")
        self.virtual_angle_status.style().unpolish(self.virtual_angle_status)
        self.virtual_angle_status.style().polish(self.virtual_angle_status)
        self._update_relay_and_sample_status()

    def _update_relay_and_sample_status(self) -> None:
        """What the 4F stop and an objective after the axicon would do."""

        if not hasattr(self, "virtual_sample_status"):
            return
        proposed = self._proposed_geometry()
        charge = self.virtual_route_charge.value() if hasattr(self, "virtual_route_charge") else 0
        lines: list[str] = []
        stop_mm = proposed.fourier_aperture_diameter_mm
        if stop_mm:
            report = fourier_aperture_report(
                diameter_mm=stop_mm,
                wavelength_nm=float(proposed.wavelength_nm),
                focal_length_mm=float(proposed.lens1_focal_length_mm),
                window_mm=float(proposed.simulation_window_mm),
                grid_n=int(self.resolution_engine.grid_n),
            )
            measured = None
            if self.current_frame is not None:
                measured = (self.current_frame.metadata.get("beam_model") or {}).get(
                    "fourier_aperture_power_fraction"
                )
            passed = (
                f" The last frame passed {measured * 100:.1f} % of the light."
                if isinstance(measured, (int, float))
                else ""
            )
            lines.append(
                f"4F stop {stop_mm:g} mm at f={proposed.lens1_focal_length_mm:g} mm relays SLM detail down to "
                f"{report['smallest_relayed_feature_um']:.0f} µm ({report['cutoff_cycles_per_mm']:.2f} cycles/mm). "
                "A high-charge vortex is finer than that near its core, so it does lose power: for q=20 on a 2 mm "
                "beam, 10 mm passes 98 %, 6 mm 95 %, 3 mm 81 % and 1 mm only 14 %." + passed
            )
        else:
            lines.append(
                "4F stop: ideal order selection, no spatial filtering. Enter the iris diameter to see what it costs."
            )
        try:
            scale = self.resolution_engine.sample_plane(charge=int(charge), geometry=proposed)
        except Exception as exc:  # an unusable axicon is reported by the axicon status
            lines.append(f"Objective: {exc}")
        else:
            if scale.demagnification >= 1.0:
                ring = "" if scale.ring_radius_um is None else f" Ring Ø now {2 * scale.ring_radius_um:.1f} µm."
                lines.append(
                    "Objective 1:1 — the camera sits at the axicon output." + ring
                    + " Ring size follows the objective; Bessel length follows the input beam radius."
                )
            else:
                ring = "?" if scale.ring_radius_um is None else f"{2 * scale.ring_radius_um:.2f}"
                length = "?" if scale.bessel_length_um is None else f"{scale.bessel_length_um:.0f}"
                na = "?" if scale.numerical_aperture is None else f"{scale.numerical_aperture:.2f}"
                lines.append(
                    f"Objective {scale.ratio_label} → ring Ø {ring} µm, Bessel length {length} µm, NA {na}, "
                    f"{scale.pixel_um:.3f} µm per camera pixel ({scale.native_sample_um:.3f} µm per native sample). "
                    f"Camera z {self.virtual_camera_z.value():g} mm reads as "
                    f"{scale.sample_z_um(self.virtual_camera_z.value()):.1f} µm at the sample. {scale.message}"
                )
        self.virtual_sample_status.setText("  ".join(lines))
        warn = "objective" in lines[-1].lower() and "no propagating cone" in lines[-1]
        self.virtual_sample_status.setObjectName("WarnChip" if warn else "Muted")
        self.virtual_sample_status.style().unpolish(self.virtual_sample_status)
        self.virtual_sample_status.style().polish(self.virtual_sample_status)

    def _apply_virtual_geometry(self) -> None:
        try:
            if self._virtual_thread is not None and self._virtual_thread.isRunning():
                raise RuntimeError("Wait for the correction task to finish or cancel it before changing model geometry.")
            proposed = self._proposed_geometry()
            report = self._axicon_report(proposed)
            if not report["valid"]:
                raise ValueError(str(report["message"]))
            charge = self.virtual_route_charge.value() if hasattr(self, "virtual_route_charge") else 0
            scale = self.resolution_engine.sample_plane(charge=int(charge), geometry=proposed)
            if not scale.valid:
                raise ValueError(f"NOT APPLIED: {scale.message}")
            self._ensure_virtual_mode()
            was_live = self._camera_thread is not None and self._camera_thread.isRunning()
            if not self.stop_live():
                raise RuntimeError("Wait for the in-flight virtual frame to finish, then apply the axicon again.")
            self.resolution_engine.set_geometry(
                slm1_to_slm2_mm=proposed.slm1_to_slm2_mm,
                axicon_k_perp_m_inv=proposed.axicon_k_perp_m_inv,
                axicon_k_perp_source=proposed.axicon_k_perp_source,
                axicon_model_base_angle_deg=proposed.axicon_model_base_angle_deg,
                fourier_aperture_diameter_mm=proposed.fourier_aperture_diameter_mm,
                fourier_aperture_source=proposed.fourier_aperture_source,
                objective_demagnification=proposed.objective_demagnification,
                objective_source=proposed.objective_source,
            )
            self._pending_virtual_proposal = None
            self._awaiting_post_apply = None
            self.virtual_apply_button.setEnabled(False)
            self.current_frame = None
            self.current_metrics = None
            self.virtual_camera_view.release_frame()
            self.virtual_frame_chip.setText("Axicon changed — waiting for a fresh synthetic frame")
            self._refresh_virtual_geometry_summary()
            self._update_axicon_angle_status()
            self.virtual_scenario_status.setText(
                "Model geometry updated. Relay distances and the camera z origin remain uncalibrated to the bench."
            )
            if was_live:
                self._start_virtual_live(recast=False)
            self.virtual_bench_status.setText(
                f"Axicon set: {self.virtual_axicon_mode.currentText()}. "
                + ("Live view restarting. " if was_live else "Start live to see the new field. ")
                + "Earlier correction proposals were cleared because they were computed for the old axicon."
            )
        except Exception as exc:
            self._update_axicon_angle_status()
            self._show_error("Axicon not changed", exc)

    @staticmethod
    def _ring_metric_snapshot(metrics):
        if metrics is None or metrics.family != "vortex_bessel":
            return None
        return {
            "radius_px": float(metrics.values.get("principal_ring_radius_px", 0.0)),
            "roundness": max(0.0, 1.0 - float(metrics.values.get("ring_eccentricity", 0.0))),
            "azimuthal_cv": float(metrics.values.get("azimuthal_cv", 0.0)),
        }

    def _refresh_iteration_history_text(self) -> None:
        rows = []
        for number, entry in enumerate(self._virtual_correction_history, 1):
            rows.append(
                f"#{number} {entry['status']} | z={entry['z_plan']} mm | "
                f"J {entry['before']:.4g} → {entry['candidate']:.4g} "
                f"({entry['improvement']:+.1f}%) | {entry['accepted']} accepted commands"
            )
            if entry.get("observed"):
                observed = entry["observed"]
                rows.append(
                    f"    first live frame: radius {observed['radius_px']:.1f} px, "
                    f"roundness {observed['roundness']:.3f}, azimuthal CV {observed['azimuthal_cv']:.3f}"
                )
            for command in entry.get("commands", []):
                rows.append(
                    f"    {command['slm']} {command['parameter']}: "
                    f"{command['before']:+.4g} → {command['after']:+.4g} {command['units']}"
                )
        self.virtual_iteration_history.setPlainText("\n".join(rows))

    def _export_virtual_correction_log(self) -> None:
        if not self._virtual_correction_history:
            QMessageBox.information(self, "No correction iterations", "Preview a correction before exporting its log.")
            return
        default_name = "virtual_correction_log_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".json"
        path_text, _ = QFileDialog.getSaveFileName(
            self, "Export synthetic correction history", default_name, "JSON files (*.json)"
        )
        if not path_text:
            return
        try:
            payload = {
                "data_kind": "SYNTHETIC",
                "scope": "virtual GUI session only; model is not fully bench-calibrated",
                "exported_utc": datetime.now(timezone.utc).isoformat(),
                "iterations": self._virtual_correction_history,
            }
            Path(path_text).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            self._show_error("Could not export correction log", exc)

    def _on_frame(self, frame) -> None:
        old = self.current_frame
        if old is not None and (
            old.z_mm != frame.z_mm
            or any(
                old.metadata.get(key) != frame.metadata.get(key)
                for key in ("cast_phase_hashes", "scenario_truth_hash", "canonical_beam_radius_mm")
            )
        ):
            self.current_metrics = None
        super()._on_frame(frame)
        waiting = self._awaiting_post_apply
        if waiting is None or self.current_metrics is None or frame.z_mm != waiting["z_mm"]:
            return
        observed = self._ring_metric_snapshot(self.current_metrics)
        if observed is None:
            return
        entry = self._virtual_correction_history[waiting["history_index"]]
        entry["observed"] = observed
        self._awaiting_post_apply = None
        self._refresh_iteration_history()
        before = waiting["before_metrics"]
        if before is None:
            comparison = "No same-plane pre-apply live metric was available."
        else:
            comparison = (
                f"Same-plane live check: roundness {before['roundness']:.3f} → {observed['roundness']:.3f}, "
                f"azimuthal CV {before['azimuthal_cv']:.3f} → {observed['azimuthal_cv']:.3f}. "
            )
        self.virtual_result_next.setText(
            "APPLIED to virtual SLMs. " + comparison
            + "This single live frame is supplementary to the multi-plane candidate score; "
            "capture fresh evidence at your z planes, then preview another pass if needed."
        )

    def _confirm_high_resolution_run(self, estimated_frames: int, *, applies: bool) -> bool:
        if estimated_frames <= 16 or self.resolution_engine.grid_n < 1024:
            return True
        effect = (
            "Verified commands will be applied to the virtual SLMs during this run."
            if applies else
            "Original virtual masks will be restored; application needs a separate click."
        )
        choice = QMessageBox.question(
            self,
            "Long high-resolution correction run",
            f"This run may compute about {estimated_frames:,} fresh optical frames at "
            f"{self.resolution_engine.grid_n}×{self.resolution_engine.grid_n}. "
            "Even one mode across two z planes took about 90 seconds in the tested setup. "
            "Reduce z planes, modes, passes or resolution for a quicker experiment.\n\n"
            f"{effect}\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return choice == QMessageBox.Yes

    def _start_blind_virtual_correction(self) -> None:
        try:
            frames = estimated_blind_cycle_frames(
                len(self._expanded_z_plan()),
                1,
                len(self._selected_correction_parameters()),
                len(self._selected_targets()),
            ) * self.virtual_passes.value()
            if self._confirm_high_resolution_run(frames, applies=True):
                super()._start_blind_virtual_correction()
        except Exception as exc:
            self._show_error("Could not start direct virtual correction", exc)

    def _start_auto_converge(self) -> None:
        try:
            frames = estimated_blind_cycle_frames(
                len(self._expanded_z_plan()),
                1,
                len(self._selected_correction_parameters()),
                len(self._selected_targets()),
            ) * self.converge_max_cycles.value()
            frames = min(frames, self.converge_max_frames.value())
            if self._confirm_high_resolution_run(frames, applies=True):
                super()._start_auto_converge()
        except Exception as exc:
            self._show_error("Could not start auto-converge", exc)

    def _preview_virtual_correction(self) -> None:
        try:
            self._ensure_virtual_mode()
            if abs(self.perturb_beam_radius_mm.value() - self.resolution_engine.canonical_beam_radius_mm) > 1e-9:
                raise RuntimeError("Input beam radius is edited but not applied. Apply input size before previewing correction.")
            z_plan = self._expanded_z_plan()
            targets = self._selected_targets()
            parameters = self._selected_correction_parameters()
            if not parameters:
                raise ValueError("Select at least one SLM correction mode.")
            baseline = self.store.snapshot()
            original_z = self.virtual_camera_z.value()
            signature = self._preview_signature(baseline)
            probe_amplitude = self.virtual_probe.value()
            passes = self.virtual_passes.value()
            seed = self.virtual_seed.value()
            estimated_frames = estimated_blind_cycle_frames(
                len(z_plan), 1, len(parameters), len(targets)
            ) * passes
            if not self._confirm_high_resolution_run(estimated_frames, applies=False):
                return
            runner = BlindCorrectionRunner(self.mode_controller)
            self._awaiting_post_apply = None
            before_metrics = self._ring_metric_snapshot(self.current_metrics)
            self._pending_virtual_proposal = None
            self.virtual_apply_button.setEnabled(False)

            def task(**callbacks):
                result = None
                candidate = None
                try:
                    result = runner.run(
                        z_plan,
                        targets=targets,
                        parameters=parameters,
                        probe_amplitude_waves=probe_amplitude,
                        passes=passes,
                        seed=seed,
                        **callbacks,
                    )
                    candidate = self.store.snapshot()
                finally:
                    def restore(state):
                        for name in ("slm1", "slm2"):
                            source = getattr(baseline, name)
                            target = getattr(state, name)
                            target.phase = copy.deepcopy(source.phase)
                            target.accepted_correction_id = source.accepted_correction_id
                        state.session.accepted_trial = baseline.session.accepted_trial

                    self.store.update(restore, source="virtual_preview", reason="Restored pre-preview SLM state")
                    self.mode_controller.cast(("SLM1", "SLM2"), persist=False)
                    self.mode_controller.move_stage(original_z)
                    camera = self.controller.camera_provider
                    if camera is not None and camera.connected:
                        self.controller.start_camera()
                        callbacks["frame_callback"](self.controller.acquire_frame(fresh=True))
                return {"kind": "correction_preview", "result": result, "candidate": candidate,
                        "baseline_signature": signature, "z_plan": z_plan,
                        "before_metrics": before_metrics, "original_z": original_z}

            self.virtual_bench_status.setText(
                "Preview running. Candidate commands are tested on fresh synthetic camera frames; "
                "the pre-preview masks will be restored before the proposal is shown."
            )
            self.virtual_progress.setPlainText(
                f"CORRECTION PREVIEW • z={z_plan} • targets={targets} • modes={parameters}\n"
                "No candidate will remain cast after this preview."
            )
            self._start_virtual_task(task)
        except Exception as exc:
            self._show_error("Could not preview virtual correction", exc)

    @Slot(object)
    def _virtual_task_finished(self, result):
        if not isinstance(result, dict) or result.get("kind") != "correction_preview":
            super()._virtual_task_finished(result)
            return
        trial = result["result"]
        accepted = [row for row in trial.accepted_runs if row["accepted"]]
        lines = [
            "SYNTHETIC SENSORLESS CORRECTION PREVIEW — NOT APPLIED",
            f"Objective before: {trial.initial_objective:.6g}",
            f"Objective after candidate: {trial.final_objective:.6g}",
            f"Improvement: {100 * trial.improvement_fraction:.3f}%",
            f"Verified accepted modes: {len(accepted)} / {len(trial.accepted_runs)}",
            "Suggested SLM commands (not unique measurements of physical aberration):",
        ]
        for row in trial.accepted_runs:
            spec = PARAMETERS[row["parameter"]]
            before = float(result["baseline_signature"][row["slm"].lower()][spec.attribute])
            if row["accepted"]:
                lines.append(
                    f"  APPLY {row['slm']} {row['parameter']}: "
                    f"{before:+.5g} → {row['accepted_command']:+.5g} {spec.units}"
                )
            else:
                lines.append(
                    f"  KEEP  {row['slm']} {row['parameter']}: "
                    f"{before:+.5g} {spec.units} (no verified improvement)"
                )
        if trial.cancelled:
            lines.append("Cancelled. No proposal can be applied.")
        elif not accepted or trial.improvement_fraction <= 0:
            lines.append("No improving verified proposal. The SLMs remain at their original masks.")
        else:
            lines.append("Original masks restored. Click Apply reviewed correction to cast the candidate, or edit either SLM manually.")
            self._pending_virtual_proposal = result
            self.virtual_apply_button.setEnabled(True)
        entry = {
            "status": "PREVIEW ONLY" if self._pending_virtual_proposal is result else "NO APPLICABLE CANDIDATE",
            "z_plan": result.get("z_plan", []),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_truth_hash": result["baseline_signature"]["scenario_truth_hash"],
            "input_radius_mm": result["baseline_signature"]["input_radius_mm"],
            "resolution": result["baseline_signature"]["resolution"],
            "before": float(trial.initial_objective),
            "candidate": float(trial.final_objective),
            "improvement": 100.0 * float(trial.improvement_fraction),
            "accepted": len(accepted),
            "commands": [
                {
                    "slm": row["slm"],
                    "parameter": row["parameter"],
                    "before": float(result["baseline_signature"][row["slm"].lower()][PARAMETERS[row["parameter"]].attribute]),
                    "after": float(row["accepted_command"]),
                    "units": PARAMETERS[row["parameter"]].units,
                }
                for row in accepted
            ],
        }
        self._virtual_correction_history.append(entry)
        result["history_index"] = len(self._virtual_correction_history) - 1
        self._refresh_iteration_history()
        self.virtual_results.setPlainText("\n".join(lines))
        self.virtual_result_next.setText(
            "PREVIEW ONLY — original virtual masks restored. The candidate score is from fresh synthetic "
            "probe frames, not a physical-aberration measurement. Apply reviewed to change both virtual SLM casts; "
            "then compare the live camera and take new captures before the next pass."
            if self._pending_virtual_proposal is result else
            "No improving candidate to apply. Adjust the fault, beam size, z planes, target SLM or correction modes, "
            "then run a new preview."
        )
        self.virtual_bench_status.setText(lines[-1])
        self.virtual_progress.appendPlainText("Preview complete; pre-preview SLM phases restored.")
        self._show_correction_proposal(result)
        self._show_virtual_tab("correction")

    def _apply_reviewed_virtual_correction(self) -> None:
        try:
            proposal = self._pending_virtual_proposal
            if proposal is None:
                raise RuntimeError("There is no reviewed correction proposal to apply.")
            if self._virtual_thread is not None and self._virtual_thread.isRunning():
                raise RuntimeError("Wait for the preview task to finish before applying its result.")
            self._ensure_virtual_mode()
            if self._preview_signature(self.store.snapshot()) != proposal["baseline_signature"]:
                raise RuntimeError("The SLM phase, scenario, geometry or resolution changed since preview. Run a fresh preview.")
            candidate = proposal["candidate"]

            def apply(state):
                for name in ("slm1", "slm2"):
                    source = getattr(candidate, name)
                    target = getattr(state, name)
                    target.phase = copy.deepcopy(source.phase)
                    target.accepted_correction_id = source.accepted_correction_id
                state.session.accepted_trial = candidate.session.accepted_trial

            self.store.update(apply, source="virtual_operator", reason="Applied reviewed virtual correction proposal")
            self.mode_controller.cast(("SLM1", "SLM2"), persist=False)
            history_index = proposal.get("history_index")
            if history_index is not None:
                self._virtual_correction_history[history_index]["status"] = "APPLIED TO VIRTUAL SLMS"
                self._awaiting_post_apply = {
                    "history_index": history_index,
                    "z_mm": proposal.get("original_z", self.virtual_camera_z.value()),
                    "before_metrics": proposal.get("before_metrics"),
                }
                self._refresh_iteration_history()
            self._pending_virtual_proposal = None
            self.virtual_apply_button.setEnabled(False)
            if hasattr(self, "correct_discard"):
                self.correct_discard.setEnabled(False)
                self.correct_verdict.setText(
                    "✔ Applied to the virtual SLMs. Watch the live camera, then press Next iteration to go again."
                )
            self.current_metrics = None
            self._start_virtual_live(recast=False)
            self.virtual_result_next.setText(
                "APPLIED to the virtual SLMs; waiting for a fresh live frame at the original z plane. "
                "Then take fresh captures and preview another pass if the beam still needs work."
            )
            self.virtual_bench_status.setText(
                "Reviewed correction cast to the VIRTUAL SLMs. The live synthetic camera now shows the applied state. "
                "You can edit either SLM and recast, or take another set of captures."
            )
        except Exception as exc:
            self._show_error("Could not apply reviewed correction", exc)

    def _refresh_virtual_resolution_status(self) -> None:
        summary = self.resolution_engine.resolution_summary()
        self.virtual_resolution_status.setText(
            f"Propagation grid: {summary['propagation_grid_yx'][1]}×{summary['propagation_grid_yx'][0]} • "
            f"camera frame: {summary['output_frame_yx'][1]}×{summary['output_frame_yx'][0]}\n"
            f"{summary['physical_sampling_status']}\n"
            "MODEL NATIVE 256/512 is a speed/debug mode and can alias q=20 rings; "
            "use 2048 for a morphology check. The first frame is slower; "
            "unchanged live frames reuse the optical field."
        )

    def _apply_virtual_output_resolution(self) -> None:
        try:
            if self._virtual_thread is not None and self._virtual_thread.isRunning():
                raise RuntimeError("Wait for the correction task to finish or cancel it before changing resolution.")
            mode = VirtualOutputResolution(self.virtual_output_resolution.currentText())
            if mode is VirtualOutputResolution.LIVE_1M:
                prospective_n = self.resolution_engine.live_n
            elif mode is VirtualOutputResolution.BEAMAGE_4M:
                prospective_n = self.resolution_engine.beamage_n
            elif mode in {VirtualOutputResolution.MAXIMUM, VirtualOutputResolution.MAXIMUM_TO_BEAMAGE}:
                prospective_n = self.resolution_engine.maximum_grid_n
            else:
                prospective_n = self.resolution_engine.geometry.quality_grid("validation")
            report = self._axicon_report(grid_n=prospective_n)
            if not report["valid"]:
                raise ValueError("Selected resolution cannot represent the active axicon. " + str(report["message"]))
            if not self.stop_live():
                raise RuntimeError("Wait for the current virtual frame before changing resolution.")
            self._ensure_virtual_mode()
            self.resolution_engine.set_output_resolution(mode)
            if mode in {VirtualOutputResolution.MAXIMUM, VirtualOutputResolution.MAXIMUM_TO_BEAMAGE}:
                self.virtual_quality.setCurrentText("maximum")
            elif mode in {VirtualOutputResolution.BEAMAGE_4M, VirtualOutputResolution.NATIVE}:
                # Pixel-count-equivalent propagation at the real camera's nominal
                # 4M shape.  This is deliberately not called a physical pixel-pitch match.
                if self.virtual_quality.currentText() == "maximum":
                    self.virtual_quality.setCurrentText("validation")

            def update_camera_state(state):
                state.camera.shape_yx = self.resolution_engine.shape_yx
                state.camera.pixel_size_um = self.resolution_engine.pixel_size_um
                state.camera.frame_quality = (
                    "SIMULATED_NUMERICAL_INTENSITY • " + self.resolution_engine.output_resolution.value
                )

            self.store.update(
                update_camera_state,
                source="virtual_resolution",
                reason=f"Virtual output resolution changed to {mode.value}",
            )
            self.current_frame = None
            self.current_metrics = None
            self.virtual_camera_view.release_frame()
            self._refresh_virtual_resolution_status()
            self._update_axicon_angle_status()
            self.virtual_progress.appendPlainText(
                "\nResolution mode changed. Capture a fresh stack before comparing metrics.\n"
                + json.dumps(self.resolution_engine.resolution_summary(), indent=2)
            )
        except Exception as exc:
            self._show_error("Could not change virtual resolution", exc)


def _previous_session_crashed(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    last_start = text.rfind("=== SESSION START")
    return last_start >= 0 and "=== SESSION ENDED CLEANLY" not in text[last_start:]


def main() -> int:
    """Launch the cockpit with crash evidence that survives any kind of exit.

    A hard native crash (heap corruption, a Qt fatal) never reaches Python's
    exception hook, and the old log was written to whatever directory the app
    happened to be launched from.  Everything now goes to one fixed file with
    session markers, so an unexplained exit always leaves a trace.
    """

    global _FAULT_LOG_HANDLE
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    crash_path = log_dir / "lab_gui_crash.log"
    crashed_before = _previous_session_crashed(crash_path)
    _FAULT_LOG_HANDLE = crash_path.open("a", encoding="utf-8", buffering=1)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== SESSION START {stamp} pid={__import__('os').getpid()} ===", file=_FAULT_LOG_HANDLE)
    faulthandler.enable(_FAULT_LOG_HANDLE, all_threads=True)

    def log_uncaught(error_type, error, error_traceback) -> None:
        print(f"\n--- uncaught GUI exception {datetime.now():%H:%M:%S} ---", file=_FAULT_LOG_HANDLE)
        traceback.print_exception(error_type, error, error_traceback, file=_FAULT_LOG_HANDLE)
        sys.__excepthook__(error_type, error, error_traceback)

    def log_thread_exception(args) -> None:
        print(f"\n--- uncaught exception in thread {args.thread.name if args.thread else '?'} ---",
              file=_FAULT_LOG_HANDLE)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=_FAULT_LOG_HANDLE)

    def log_qt_message(mode, context, message) -> None:
        # Qt warnings and fatals otherwise vanish when the console is hidden.
        if mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg) or "QObject" in message or "thread" in message:
            print(f"--- Qt {mode.name} {datetime.now():%H:%M:%S}: {message}", file=_FAULT_LOG_HANDLE)

    sys.excepthook = log_uncaught
    import threading as _threading
    _threading.excepthook = log_thread_exception
    qInstallMessageHandler(log_qt_message)
    application = QApplication.instance() or QApplication(sys.argv)
    application.setStyleSheet(APP_QSS + ADVANCED_QSS)
    window = VirtualLabResolutionWindow()
    window.show()
    if crashed_before:
        QMessageBox.warning(
            window,
            "The last session ended unexpectedly",
            "The previous Lab Control session did not close normally. What was recorded is in:\n\n"
            f"{crash_path}\n\nPlease keep that file — it shows where it stopped.",
        )
    code = application.exec()
    print(f"=== SESSION ENDED CLEANLY {datetime.now():%Y-%m-%d %H:%M:%S} exit={code} ===", file=_FAULT_LOG_HANDLE)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
