"""Per-pixel AIA noise estimates for staged readings, computed as the solver computed them.

fullBP.py estimates each channel's noise with aiapy's estimate_error (SSW error table) on the
deconvolved counts floored at 0.5, then stores readings and noise divided by a per-image,
per-channel scale factor s (exposure time x degradation), which it also saves:

    reading = counts / s,   noise = estimate_error(max(counts, 0.5)) / s,

so noise(reading) = estimate_error(max(reading * s, 0.5)) / s. Against our staged noise
estimates the relative difference is below 1e-7. The scale factors come from our own solver
output files (one per image), which fills in the noise for a staging that lacks it (the
shared test set, staged separately with the same 153 test images).
"""
import os

import numpy as np

WAVES = (94, 131, 171, 193, 211, 335)


class NoiseModel:
    def __init__(self, npz_dir, timestamps, ratio=None):
        """ratio [6]: the staging's scale factor over ours, per channel, for a staging made with a
        different calibration (the shared test: fitted by experiments/fit_shared_noise.py)."""
        import astropy.units as u
        from aiapy.calibrate.utils import get_error_table
        self.u = u
        self.table = get_error_table(source="SSW")
        self.scale = np.stack([self._scale(os.path.join(npz_dir, f"{ts}.npz")) for ts in timestamps])
        if ratio is not None:
            self.scale = self.scale * np.asarray(ratio, dtype=np.float64)[None, :]

    @staticmethod
    def _scale(path):
        with np.load(path) as d:
            return np.asarray(d["scaleFactor"], dtype=np.float64).reshape(-1)[:6]

    def sigma_raw(self, reading, scale, channel):
        """Noise of readings [n] of one channel, given each reading's scale factor [n]."""
        from aiapy.calibrate import estimate_error
        counts = np.maximum(np.asarray(reading, dtype=np.float64) * scale, 0.5)
        return estimate_error(counts * self.u.DN / self.u.pix, WAVES[channel] * self.u.angstrom,
                              error_table=self.table).value / scale

    def sigma(self, obs, image):
        """obs [n, 6] readings from image number `image` -> noise [n, 6], same units."""
        from aiapy.calibrate import estimate_error
        u = self.u
        s = self.scale[image]
        out = np.empty(np.shape(obs), dtype=np.float64)
        for c, wave in enumerate(WAVES):
            counts = np.maximum(np.asarray(obs[:, c], dtype=np.float64) * s[c], 0.5)
            out[:, c] = estimate_error(counts * u.DN / u.pix, wave * u.angstrom,
                                       error_table=self.table).value / s[c]
        return out
