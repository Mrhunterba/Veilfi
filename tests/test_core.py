import numpy as np
import pytest

from veilfi.attackers.threshold import (ThresholdDetector, min_consecutive_change,
                                        windowed_variance)
from veilfi.csi.preprocess import normalize_amplitude, sanitize_phase, sliding_windows
from veilfi.eval.metrics import advantage, rates, session_bootstrap_ci
from veilfi.eval.splits import assert_no_leakage, session_split
from veilfi.monitor.features import packet_rate, window_features
from veilfi.shield.scheduler import (PacketTriggeredScheduler, next_configuration,
                                     rate_matched_update_rate, to_bytes)


# --- shield -----------------------------------------------------------------
def test_no_holds_and_full_width():
    prev = None
    for _ in range(200):
        cfg = next_configuration(prev)
        assert cfg != prev and 0 <= cfg < 2 ** 256
        prev = cfg
    assert len(to_bytes(prev)) == 32


def test_rate_matching_within_limit():
    d = rate_matched_update_rate(packet_rate_hz=100, controller_limit_hz=500)
    assert d.meets_packet_rate and d.update_rate_hz >= 100


def test_rate_matching_flags_failure():
    d = rate_matched_update_rate(packet_rate_hz=800, controller_limit_hz=500)
    assert not d.meets_packet_rate and d.update_rate_hz == 500


def test_packet_triggered_respects_limit():
    s = PacketTriggeredScheduler(controller_limit_hz=100)  # 10 ms min interval
    out = [s.on_packet(t) for t in (0.0, 0.005, 0.011, 0.012)]
    assert out[0] is not None and out[1] is None and out[2] is not None and out[3] is None
    assert s.skipped == 2


# --- CSI and attackers ------------------------------------------------------
def test_preprocess_shapes():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(50, 30)) + 1j * rng.normal(size=(50, 30))
    assert np.allclose(np.linalg.norm(normalize_amplitude(h), axis=1), 1)
    assert sanitize_phase(h).shape == (50, 30)
    assert sliding_windows(h, 10, 5).shape == (9, 10, 30)


def test_min_change_detects_unchanged_interval():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(20, 30))
    assert min_consecutive_change(w) > 0
    w[5] = w[4]  # two frames under the same configuration, no motion
    assert min_consecutive_change(w) == 0


def test_threshold_detector_on_synthetic_data():
    # SYNTHETIC smoke test only - not experimental data.
    rng = np.random.default_rng(2)
    quiet = [rng.normal(0, 0.1, (40, 30)) for _ in range(50)]
    moving = [rng.normal(0, 0.5, (40, 30)) for _ in range(50)]
    det = ThresholdDetector(windowed_variance).fit(quiet[:25])
    y_pred = det.predict(quiet[25:] + moving)
    y_true = np.r_[np.zeros(25), np.ones(50)]
    assert advantage(y_true, y_pred) > 0.8


# --- metrics and splits -----------------------------------------------------
def test_advantage_definition():
    y = np.array([1, 1, 0, 0])
    assert rates(y, np.array([1, 0, 0, 1])) == (0.5, 0.5)
    assert advantage(y, y) == 1.0
    assert advantage(y, 1 - y) == -1.0


def test_session_bootstrap_ci_contains_point():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    p = np.where(rng.random(400) < 0.8, y, 1 - y)
    s = np.repeat(np.arange(20), 20)
    a, lo, hi = session_bootstrap_ci(y, p, s, n_boot=300)
    assert lo <= a <= hi


def test_session_split_no_leakage():
    s = np.array(["a", "a", "b", "c"])
    tr, te = session_split(s, {"c"})
    assert_no_leakage(s[tr], s[te])
    with pytest.raises(AssertionError):
        assert_no_leakage(s, s)


# --- monitor ----------------------------------------------------------------
def test_monitor_features_flag_unassociated_stream():
    frames = [{"t": i * 0.01, "src": "aa:bb", "dst": "dev", "ftype": "data",
               "subtype": "null", "associated": False} for i in range(100)]
    f = window_features(frames, window_s=1.0)
    assert f["unassoc_frac"] == 1.0 and f["top_unassoc_rate"] == 100
    assert packet_rate(frames, {"ap"}, 1.0) == 0.0
