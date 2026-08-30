import json, subprocess, sys

from joukowskisim.main import parser
from joukowskisim.solver import SolverConfig


def test_project_defaults_are_production_configuration():
    config = SolverConfig()
    args = parser().parse_args([])

    assert (config.re, config.nr, config.ntheta) == (10000.0, 240, 768)
    assert args.re == config.re


def test_headless_short_simulation():
    p=subprocess.run([sys.executable,"-m","joukowskisim.main","--headless","--steps","100","--nr","20","--ntheta","48"],capture_output=True,text=True,check=True)
    data=json.loads(p.stdout)
    assert data['step']==100 and data['steps_per_second']>0
    assert data['grid']==[20,48]
