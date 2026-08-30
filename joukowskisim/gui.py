"""Responsive dark PySide6 desktop interface."""

from __future__ import annotations
import threading
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .solver import FlowSolver, SolverConfig
from .renderer import CurvilinearRenderer
from .pressure import analytical_kutta_joukowski_cp, surface_coefficients


class SimulationThread(QtCore.QThread):
    frame_ready = QtCore.Signal(object, object, object)
    failed = QtCore.Signal(str)

    def __init__(self, solver: FlowSolver, field: str, frame_skip: int, pressure_every: int = 10):
        super().__init__(); self.solver = solver; self.field_name = field
        self.frame_skip = max(1, frame_skip); self.pressure_every = pressure_every
        self.running = False; self.shutdown = False; self._lock = threading.Lock()

    def run(self):
        try:
            while not self.shutdown:
                if not self.running:
                    self.msleep(20); continue
                for _ in range(self.frame_skip): self.solver.step()
                frame_no = self.solver.step_count // self.frame_skip
                if frame_no % self.pressure_every == 0: self.solver.compute_pressure()
                with self._lock: field_name = self.field_name
                field = self.solver.field(field_name).copy()
                coeff = None
                if self.solver.pressure is not None:
                    coeff = surface_coefficients(self.solver, self.solver.pressure)
                self.frame_ready.emit(field, self.solver.diagnostics(), coeff)
        except Exception as exc:
            self.running = False; self.failed.emit(f"{type(exc).__name__}: {exc}")

    def set_field(self, name: str):
        with self._lock: self.field_name = name


class FlowView(QtWidgets.QWidget):
    def __init__(self, renderer):
        super().__init__(); self.renderer = renderer; self.rgb = None; self.setMinimumSize(700, 380)

    def set_rgb(self, rgb): self.rgb = rgb; self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self); p.fillRect(self.rect(), QtGui.QColor("#0a0d12"))
        if self.rgb is not None:
            h,w,_ = self.rgb.shape
            image = QtGui.QImage(self.rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
            p.drawImage(self.rect(), image)
        poly = QtGui.QPolygonF([QtCore.QPointF(float(x),float(y)) for x,y in self.renderer.surface_pixels()])
        sx=self.width()/self.renderer.width; sy=self.height()/self.renderer.height
        p.save(); p.scale(sx,sy); p.setBrush(QtGui.QColor("#171a20")); p.setPen(QtGui.QPen(QtGui.QColor("#e5e7eb"),1.4)); p.drawPolygon(poly); p.restore()


class CpPlot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.data = None
        self.setMinimumHeight(190)

    def set_data(self, solver, cp=None, cp_kj=None):
        numerical = None if cp is None else np.asarray(cp, dtype=float).copy()
        analytical = None if cp_kj is None else np.asarray(cp_kj, dtype=float).copy()
        self.data = (
            solver.x[0].copy(),
            solver.y[0].copy(),
            numerical,
            analytical,
        )
        self.update()

    @staticmethod
    def _draw_profile(painter, x, values, mask, color, style, width, bounds, limit):
        good = np.isfinite(values)
        ids = np.where(mask & good)[0]
        ids = ids[np.argsort(x[ids])]
        if ids.size == 0:
            return
        left, top, right, bottom = bounds
        path = QtGui.QPainterPath()
        for point_number, index in enumerate(ids):
            px = left + np.clip(x[index], 0.0, 1.0) * (right - left)
            py = top + (values[index] + limit) / (2.0 * limit) * (bottom - top)
            (path.moveTo if point_number == 0 else path.lineTo)(px, py)
        painter.setPen(QtGui.QPen(QtGui.QColor(color), width, style))
        painter.drawPath(path)

    @staticmethod
    def _surface_masks(x, y):
        """Split the cyclic surface contour into its upper and lower branches."""
        count = x.size
        trailing_edge = int(np.argmax(x))
        leading_edge = int(np.argmin(x))

        def cyclic_indices(start, stop):
            if start <= stop:
                return np.arange(start, stop + 1)
            return np.concatenate((np.arange(start, count), np.arange(stop + 1)))

        first = cyclic_indices(trailing_edge, leading_edge)
        second = cyclic_indices(leading_edge, trailing_edge)
        first_mask = np.zeros(count, dtype=bool)
        second_mask = np.zeros(count, dtype=bool)
        first_mask[first] = True
        second_mask[second] = True
        if np.mean(y[first]) >= np.mean(y[second]):
            return first_mask, second_mask
        return second_mask, first_mask

    @staticmethod
    def _axis_limit(cp, cp_kj):
        """Fit the reference curve while rejecting isolated viscous spikes."""
        candidates = [1.0]
        has_finite_data = False
        if cp is not None:
            numerical = np.abs(cp[np.isfinite(cp)])
            if numerical.size:
                candidates.append(float(np.percentile(numerical, 95)))
                has_finite_data = True
        if cp_kj is not None:
            analytical = np.abs(cp_kj[np.isfinite(cp_kj)])
            if analytical.size:
                candidates.append(float(np.max(analytical)))
                has_finite_data = True
        return 1.05 * max(candidates) if has_finite_data else None

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#10141b"))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtGui.QColor("#aeb7c4"))
        p.drawText(
            12,
            20,
            "Surface Cp (inverted) — solid: viscous, dashed: inviscid Kutta–Joukowski",
        )
        left, top, right, bottom = 50, 30, self.width() - 18, self.height() - 28
        if self.data is None:
            p.drawRect(left, top, right - left, bottom - top)
            return

        x, y, cp, cp_kj = self.data
        limit = self._axis_limit(cp, cp_kj)
        if limit is None:
            p.drawRect(left, top, right - left, bottom - top)
            return
        upper, lower = self._surface_masks(x, y)
        bounds = (left, top, right, bottom)

        p.save()
        p.setClipRect(QtCore.QRectF(left + 1, top + 1, right - left - 2, bottom - top - 2))
        if cp is not None:
            for mask, color in ((upper, "#53c8ff"), (lower, "#ff9d57")):
                self._draw_profile(
                    p,
                    x,
                    cp,
                    mask,
                    color,
                    QtCore.Qt.PenStyle.SolidLine,
                    1.8,
                    bounds,
                    limit,
                )
        if cp_kj is not None:
            for mask, color in ((upper, "#d2f2ff"), (lower, "#ffe0c2")):
                self._draw_profile(
                    p,
                    x,
                    cp_kj,
                    mask,
                    color,
                    QtCore.Qt.PenStyle.DashLine,
                    1.6,
                    bounds,
                    limit,
                )
        p.restore()

        p.setPen(QtGui.QColor("#aeb7c4"))
        p.drawRect(left, top, right - left, bottom - top)
        p.setPen(
            QtGui.QPen(
                QtGui.QColor("#697386"),
                1,
                QtCore.Qt.PenStyle.DashLine,
            )
        )
        p.drawLine(left, (top + bottom) // 2, right, (top + bottom) // 2)
        p.setPen(QtGui.QColor("#aeb7c4"))
        p.drawText(right - 24, bottom + 20, "x/c")
        p.drawText(6, top + 8, f"{-limit:.1f}")
        p.drawText(10, bottom, f"{limit:.1f}")


class MainWindow(QtWidgets.QMainWindow):
    fields=["Vorticity","Velocity magnitude","u velocity","v velocity","Pressure","Cp-like pressure field","Streamfunction"]
    def __init__(self):
        super().__init__(); self.setWindowTitle("JoukowskiSim — Conformal Navier–Stokes"); self.resize(1280,850)
        self._build_controls(); self.reset_solver()

    def _build_controls(self):
        root=QtWidgets.QWidget(); self.setCentralWidget(root); outer=QtWidgets.QHBoxLayout(root)
        self.visual_col=QtWidgets.QVBoxLayout(); outer.addLayout(self.visual_col,1)
        panel=QtWidgets.QWidget(); panel.setMaximumWidth(285); form=QtWidgets.QFormLayout(panel); outer.addWidget(panel)
        self.edits={}
        for key,label,value in [("re","Re",1000),("alpha","alpha (deg)",5),("u_inf","U infinity",1),("nr","Nr",160),("ntheta","Ntheta",512),("outer_radius","outer radius",15),("cfl","CFL",.4),("thickness","thickness",.12),("camber","camber",0),("frame_skip","frame every",2)]:
            e=QtWidgets.QLineEdit(str(value)); self.edits[key]=e; form.addRow(label,e)
        self.field=QtWidgets.QComboBox(); self.field.addItems(self.fields); form.addRow("Field",self.field)
        row=QtWidgets.QHBoxLayout(); self.start=QtWidgets.QPushButton("Start"); self.pause=QtWidgets.QPushButton("Pause"); self.reset=QtWidgets.QPushButton("Reset")
        for b in (self.start,self.pause,self.reset): row.addWidget(b)
        form.addRow(row); self.stats=QtWidgets.QLabel("Ready"); self.stats.setWordWrap(True); self.stats.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse); form.addRow(self.stats)
        self.start.clicked.connect(lambda: setattr(self.worker,"running",True)); self.pause.clicked.connect(lambda: setattr(self.worker,"running",False)); self.reset.clicked.connect(self.reset_solver); self.field.currentTextChanged.connect(self._field_changed)

    def _config(self):
        return SolverConfig(re=float(self.edits['re'].text()),alpha=float(self.edits['alpha'].text()),u_inf=float(self.edits['u_inf'].text()),nr=int(self.edits['nr'].text()),ntheta=int(self.edits['ntheta'].text()),outer_radius=float(self.edits['outer_radius'].text()),cfl=float(self.edits['cfl'].text()),thickness=float(self.edits['thickness'].text()),camber=float(self.edits['camber'].text()))

    def reset_solver(self):
        if hasattr(self,'worker'):
            self.worker.shutdown=True; self.worker.wait(3000)
            self.visual_col.removeWidget(self.view); self.view.deleteLater(); self.visual_col.removeWidget(self.cpplot); self.cpplot.deleteLater()
        try: self.solver=FlowSolver(self._config())
        except Exception as exc: QtWidgets.QMessageBox.critical(self,"Configuration error",str(exc)); return
        self.renderer=CurvilinearRenderer(self.solver); self.view=FlowView(self.renderer); self.cpplot=CpPlot(); self.visual_col.addWidget(self.view,1); self.visual_col.addWidget(self.cpplot)
        cp_kj = analytical_kutta_joukowski_cp(
            self.solver.mapping,
            self.solver.grid.theta,
            self.solver.config.u_inf,
            self.solver.config.alpha,
        )
        self.cpplot.set_data(self.solver, cp_kj=cp_kj)
        self.worker=SimulationThread(self.solver,self.field.currentText(),int(self.edits['frame_skip'].text())); self.worker.frame_ready.connect(self._frame); self.worker.failed.connect(lambda msg: QtWidgets.QMessageBox.critical(self,"Simulation stopped",msg)); self.worker.start()
        self._frame(self.solver.omega,self.solver.diagnostics(),None)

    def _field_changed(self,name):
        if hasattr(self,'worker'): self.worker.set_field(name)
    @QtCore.Slot(object,object,object)
    def _frame(self,field,stats,coeff):
        positive=self.field.currentText()=="Velocity magnitude"; self.view.set_rgb(self.renderer.render(field,positive))
        if coeff is not None:
            self.cpplot.set_data(self.solver, coeff['cp'], coeff['cp_kj'])
        lines=[f"t = {stats['time']:.5f}   step {int(stats['step'])}",f"dt = {stats['dt']:.3e}   CFL = {stats['cfl']:.3f}",f"max |omega| = {stats['max_omega']:.3e}",f"max |u| = {stats['max_velocity']:.3f}",f"Gamma = {stats['gamma']:.4f}",f"Cl(KJ) = {stats['cl_kj']:.4f}",f"wall slip = {stats['wall_slip']:.3e}",f"energy-like = {stats['kinetic_energy']:.3e}"]
        if 'cl_pressure' in stats: lines.append(f"Cl(p) = {stats['cl_pressure']:.4f}\nCd(p) = {stats['cd_pressure']:.4f}")
        self.stats.setText("\n".join(lines))
    def closeEvent(self,event):
        self.worker.shutdown=True; self.worker.wait(3000); super().closeEvent(event)


def run_gui() -> int:
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion"); app.setStyleSheet("QWidget{background:#10141b;color:#d9e0e8} QLineEdit,QComboBox{background:#1a202b;border:1px solid #354052;padding:4px} QPushButton{background:#273245;padding:7px;border-radius:3px} QPushButton:hover{background:#34445d}")
    window=MainWindow(); window.show(); return app.exec()
