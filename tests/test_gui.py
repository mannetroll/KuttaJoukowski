import numpy as np

from joukowskisim.gui import CpPlot


def test_cp_axis_fits_analytical_curve_but_rejects_viscous_spike():
    viscous = np.linspace(-1.5, 1.5, 512)
    viscous[-1] = -13.4
    analytical = np.linspace(-1.8, 1.0, 512)

    limit = CpPlot._axis_limit(viscous, analytical)

    assert np.isclose(limit, 1.05 * 1.8)
    assert limit < abs(viscous[-1])


def test_cp_axis_handles_missing_or_nonfinite_profiles():
    assert np.isclose(CpPlot._axis_limit(None, np.array([-1.2, 1.0])), 1.26)
    assert CpPlot._axis_limit(np.array([np.nan]), None) is None
