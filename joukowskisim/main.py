"""Command-line and graphical entry points."""

from __future__ import annotations
import argparse, json, time
from .solver import FlowSolver, SolverConfig


def parser() -> argparse.ArgumentParser:
    defaults = SolverConfig()
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--headless",action="store_true"); p.add_argument("--benchmark",action="store_true")
    p.add_argument("--steps",type=int,default=100); p.add_argument("--nr",type=int); p.add_argument("--ntheta",type=int); p.add_argument("--re",type=float,default=defaults.re); p.add_argument("--alpha",type=float,default=defaults.alpha); p.add_argument("--u-inf",type=float,default=defaults.u_inf); p.add_argument("--cfl",type=float,default=defaults.cfl); p.add_argument("--outer-radius",type=float,default=defaults.outer_radius)
    return p


def run_headless(args) -> int:
    defaults = SolverConfig()
    nr = args.nr or (64 if args.benchmark else defaults.nr)
    nt = args.ntheta or (128 if args.benchmark else defaults.ntheta)
    solver=FlowSolver(SolverConfig(re=args.re,alpha=args.alpha,u_inf=args.u_inf,nr=nr,ntheta=nt,cfl=args.cfl,outer_radius=args.outer_radius))
    t0=time.perf_counter()
    for _ in range(args.steps): solver.step(return_diagnostics=False)
    elapsed=time.perf_counter()-t0; solver.compute_pressure(); d=solver.diagnostics()
    # This counter covers transforms routed through SpectralGrid.  Poisson and
    # Schur helpers invoke SciPy FFT directly, so do not label it total FFT time.
    solver.timing["spectral_grid_fft"] = solver.grid.fft_time
    wall_rate = solver.time / max(elapsed, 1e-15)
    d.update({
        "elapsed_seconds": elapsed,
        "steps_per_second": args.steps / max(elapsed, 1e-15),
        "simulation_time_per_wall_second": wall_rate,
        "wall_seconds_per_simulation_time": 1.0 / max(wall_rate, 1e-30),
        "grid": [nr, nt],
        "timing": solver.timing,
    })
    print(json.dumps(d,indent=2,sort_keys=True)); return 0


def main() -> int:
    args=parser().parse_args()
    if args.headless or args.benchmark: return run_headless(args)
    from .gui import run_gui
    return run_gui()

if __name__ == "__main__": raise SystemExit(main())
