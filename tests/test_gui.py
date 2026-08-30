import numpy as np
import pytest

from joukowskisim.gui import (
    CpPlot,
    MainWindow,
    _CumulativeFrameRate,
    _format_wall_time,
)


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


def test_cumulative_frame_rate_includes_pauses_and_resets():
    now = [100.0]
    counter = _CumulativeFrameRate(lambda: now[0])

    assert counter.snapshot() == (0, 0.0, 0)
    counter.start()
    now[0] = 101.0
    frames, elapsed, fps = counter.record_frame()
    assert frames == 1
    assert elapsed == pytest.approx(1.0)
    assert fps == pytest.approx(1.0)

    # Starting again after a pause must retain the original wall-time epoch.
    now[0] = 111.0
    frames, elapsed, fps = counter.snapshot()
    assert (frames, elapsed, fps) == (1, pytest.approx(11.0), 0)
    counter.start()
    now[0] = 112.0
    frames, elapsed, fps = counter.record_frame()
    assert frames == 2
    assert elapsed == pytest.approx(12.0)
    assert fps == 0

    counter.reset()
    now[0] = 200.0
    counter.start()
    now[0] = 202.0
    frames, elapsed, fps = counter.record_frame()
    assert frames == 1
    assert elapsed == pytest.approx(2.0)
    assert fps == 1
    assert isinstance(fps, int)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (-1.0, "00:00:00"),
        (0.9, "00:00:00"),
        (65.2, "00:01:05"),
        (3_661.9, "01:01:01"),
        (90_061.0, "1d 01:01:01"),
    ],
)
def test_wall_time_display(seconds, expected):
    assert _format_wall_time(seconds) == expected


def test_stale_worker_generation_cannot_replace_reset_frame():
    class WindowProbe:
        _worker_generation = 7

        def __init__(self):
            self.frames = []

        def _frame(self, *frame):
            self.frames.append(frame)

    probe = WindowProbe()
    MainWindow._frame_if_current(probe, 6, "old", {}, None)
    MainWindow._frame_if_current(probe, 7, "new", {}, None)

    assert probe.frames == [("new", {}, None)]
