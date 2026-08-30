import json, subprocess, sys


def test_headless_short_simulation():
    p=subprocess.run([sys.executable,"-m","joukowskisim.main","--headless","--steps","100"],capture_output=True,text=True,check=True)
    data=json.loads(p.stdout)
    assert data['step']==100 and data['steps_per_second']>0
