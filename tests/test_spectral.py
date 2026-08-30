import numpy as np
from joukowskisim.spectral import SpectralGrid, chebyshev_lobatto


def test_fourier_first_and_second_derivatives():
    g=SpectralGrid(12,64,3.0); k=7; f=np.sin(k*g.theta)[None,:]
    assert np.max(abs(g.theta_derivative(f)-k*np.cos(k*g.theta)))<1e-11
    assert np.max(abs(g.theta_derivative(f,2)+k*k*f))<1e-10


def test_chebyshev_polynomials():
    s,D,D2=chebyshev_lobatto(18,4.0); f=s**5-2*s**3+s
    assert np.max(abs(D@f-(5*s**4-6*s**2+1)))<2e-11
    assert np.max(abs(D2@f-(20*s**3-12*s)))<2e-10
    assert s[0]==0 and abs(s[-1]-4)<1e-15


def test_mapped_laplacian_manufactured():
    from joukowskisim.mapping import AirfoilMapping
    m=AirfoilMapping(); g=SpectralGrid(24,48,m.s_max)
    S,T=np.meshgrid(g.s,g.theta,indexing='ij'); H=m.metric(S,T)
    f=S**3*np.cos(4*T)
    exact=(6*S-16*S**3)*np.cos(4*T)/H**2
    got=g.laplacian_coordinate(f)/H**2
    assert np.max(abs(got-exact))<2e-7


def test_dealiasing_removes_high_modes():
    g=SpectralGrid(8,48,2); low=np.sin(5*g.theta); high=np.sin(20*g.theta)
    out=g.dealias((low+high)[None,:])[0]
    assert np.max(abs(out-low))<1e-12

