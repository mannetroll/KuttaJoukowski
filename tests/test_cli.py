import json, subprocess, sys


def test_headless_short_simulation():
    p=subprocess.run([sys.executable,"-m","joukowskisim.main","--headless","--steps","10","--nr","14","--ntheta","32"],capture_output=True,text=True,check=True)
    data=json.loads(p.stdout)
    assert data['step']==10 and data['steps_per_second']>0

