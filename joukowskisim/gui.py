"""Responsive dark PySide6 desktop interface."""

from __future__ import annotations
import threading
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .solver import FlowSolver, SolverConfig
from .renderer import CurvilinearRenderer
from .pressure import surface_coefficients


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
    def __init__(self): super().__init__(); self.data = None; self.setMinimumHeight(190)
    def set_data(self, solver, cp):
        self.data=(solver.x[0].copy(), solver.y[0].copy(), np.asarray(cp).copy()); self.update()
    def paintEvent(self,event):
        p=QtGui.QPainter(self); p.fillRect(self.rect(),QtGui.QColor("#10141b")); p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtGui.QColor("#aeb7c4")); p.drawText(12,20,"Surface Cp (inverted axis)")
        L,T,R,B=50,30,self.width()-18,self.height()-28
        p.drawRect(L,T,R-L,B-T)
        if self.data is None: return
        x,y,cp=self.data; good=np.isfinite(cp); lim=max(1.0,float(np.percentile(abs(cp[good]),95)))
        upper=y>=np.median(y); lower=~upper
        for mask,color in ((upper,"#53c8ff"),(lower,"#ff9d57")):
            ids=np.where(mask & good)[0]; ids=ids[np.argsort(x[ids])]
            path=QtGui.QPainterPath()
            for n,i in enumerate(ids):
                px=L+np.clip(x[i],0,1)*(R-L); py=T+(cp[i]+lim)/(2*lim)*(B-T)
                (path.moveTo if n==0 else path.lineTo)(px,py)
            p.setPen(QtGui.QPen(QtGui.QColor(color),1.6)); p.drawPath(path)
        p.setPen(QtGui.QPen(QtGui.QColor("#697386"),1,QtCore.Qt.PenStyle.DashLine)); p.drawLine(L,(T+B)//2,R,(T+B)//2)
        p.setPen(QtGui.QColor("#aeb7c4")); p.drawText(R-24,B+20,"x/c"); p.drawText(6,T+8,f"{-lim:.1f}"); p.drawText(10,B,f"{lim:.1f}")


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
        self.worker=SimulationThread(self.solver,self.field.currentText(),int(self.edits['frame_skip'].text())); self.worker.frame_ready.connect(self._frame); self.worker.failed.connect(lambda msg: QtWidgets.QMessageBox.critical(self,"Simulation stopped",msg)); self.worker.start()
        self._frame(self.solver.omega,self.solver.diagnostics(),None)

    def _field_changed(self,name):
        if hasattr(self,'worker'): self.worker.set_field(name)
    @QtCore.Slot(object,object,object)
    def _frame(self,field,stats,coeff):
        positive=self.field.currentText()=="Velocity magnitude"; self.view.set_rgb(self.renderer.render(field,positive))
        if coeff is not None: self.cpplot.set_data(self.solver,coeff['cp'])
        lines=[f"t = {stats['time']:.5f}   step {int(stats['step'])}",f"dt = {stats['dt']:.3e}   CFL = {stats['cfl']:.3f}",f"max |omega| = {stats['max_omega']:.3e}",f"max |u| = {stats['max_velocity']:.3f}",f"Gamma = {stats['gamma']:.4f}",f"Cl(KJ) = {stats['cl_kj']:.4f}",f"wall slip = {stats['wall_slip']:.3e}",f"energy-like = {stats['kinetic_energy']:.3e}"]
        if 'cl_pressure' in stats: lines.append(f"Cl(p) = {stats['cl_pressure']:.4f}\nCd(p) = {stats['cd_pressure']:.4f}")
        self.stats.setText("\n".join(lines))
    def closeEvent(self,event):
        self.worker.shutdown=True; self.worker.wait(3000); super().closeEvent(event)


def run_gui() -> int:
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion"); app.setStyleSheet("QWidget{background:#10141b;color:#d9e0e8} QLineEdit,QComboBox{background:#1a202b;border:1px solid #354052;padding:4px} QPushButton{background:#273245;padding:7px;border-radius:3px} QPushButton:hover{background:#34445d}")
    window=MainWindow(); window.show(); return app.exec()
