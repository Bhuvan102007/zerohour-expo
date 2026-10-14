import numpy as np
from astropy.timeseries import BoxLeastSquares

def make_sample_lightcurve(seed=42):
    """Synthetic star brightness with a planted transit every 3.5 days, depth 1%."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27, 2000)                     # 27 days of observations
    flux = 1.0 + rng.normal(0, 0.001, t.size)        # noise
    flux += 0.002 * np.sin(2 * np.pi * t / 13.0)     # slow drift (stellar variation)
    period, duration, depth, t0 = 3.5, 0.15, 0.01, 0.8
    phase = (t - t0) % period
    in_transit = phase < duration
    flux[in_transit] -= depth                        # the planted "planet"
    return t, flux

def find_transit(t, flux):
    """Run Box Least Squares to find the most likely repeating dip."""
    flux_clean = flux - np.median(flux)              # simple detrend: center the data
    bls = BoxLeastSquares(t, flux_clean)
    periods = np.linspace(0.5, 10, 5000)
    result = bls.power(periods, [0.05, 0.1, 0.15, 0.2])
    best = np.argmax(result.power)
    return {
        "period": float(result.period[best]),
        "duration": float(result.duration[best]),
        "depth": float(result.depth[best]),
        "t0": float(result.transit_time[best]),
        "power": result.power,
        "periods": result.period,
    }

if __name__ == "__main__":
    t, flux = make_sample_lightcurve()
    r = find_transit(t, flux)
    print(f"Found period {r['period']:.3f} d, depth {r['depth']:.4f}")