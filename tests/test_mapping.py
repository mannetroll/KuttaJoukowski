import numpy as np
import pytest
from joukowskisim.mapping import AirfoilMapping


def test_joukowski_geometry_finite_normalized_closed():
    m=AirfoilMapping(); x,y=m.surface_xy(1001)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert abs(x.min()) < 2e-4 and abs(x.max()-1) < 2e-4
    assert np.hypot(x[0]-x[-1],y[0]-y[-1]) < 1e-12
    assert .04 < y.max()-y.min() < .3


def test_metric_regularized_trailing_edge():
    m=AirfoilMapping(); s=np.linspace(0,m.s_max,40)[:,None]; t=np.linspace(0,2*np.pi,256,endpoint=False)[None,:]
    H=m.metric(s,t)
    assert H.flags.c_contiguous and np.isfinite(H).all() and H.min()>1e-4
    # The finite gap from the critical point is the documented regularization.
    assert abs((m.center.real+m.circle_radius)-m.a-m.te_gap)<1e-12


def test_analytic_mapping_derivative():
    m=AirfoilMapping(); zeta=m.zeta(.3,np.linspace(0,6,10)); eps=1e-7
    numerical=(m._raw_map(zeta+eps)-m._raw_map(zeta-eps))/(2*eps)/m.raw_chord
    assert np.max(abs(numerical-m.dz_dzeta(zeta)))<1e-8


def test_mapping_rejects_critical_point_in_fluid_domain():
    with pytest.raises(ValueError, match="critical points"):
        AirfoilMapping(camber=0.5)
