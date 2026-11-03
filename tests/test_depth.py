import numpy as np
from india_soc_fire.harmonize.depth import fit_beta, harmonize_profile, interval_fraction, cumulative_fraction

def synth(beta, edges):
    tops, bots = edges[:-1], edges[1:]
    stocks = [100 * interval_fraction(t, b, beta) for t, b in zip(tops, bots)]
    return tops, bots, stocks

def test_recover_beta():
    t, b, s = synth(0.97, [0, 10, 20, 30, 50, 100])
    assert abs(fit_beta(t, b, s) - 0.97) < 1e-3

def test_pooled_profiles():
    p1 = synth(0.975, [0, 15, 30, 60, 100]); p2 = synth(0.975, [0, 20, 50, 100])
    beta = fit_beta([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
    assert abs(beta - 0.975) < 1e-3

def test_harmonize_additive_and_flag():
    t, b, s = synth(0.97, [0, 10, 30, 50])
    out = harmonize_profile(t, b, s, 0.97)
    assert np.isclose(out["0-30"]["stock"] + out["30-100"]["stock"], out["0-100"]["stock"])
    assert out["0-100"]["extrapolated"] and not out["0-30"]["extrapolated"]
    assert np.isclose(out["0-30"]["stock"], sum(s[:2]))

def test_cumulative_monotone():
    y = cumulative_fraction([0, 10, 50, 100], 0.96)
    assert y[0] == 0 and np.all(np.diff(y) > 0)
