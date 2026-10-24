#!/usr/bin/env python3
"""
eclipse_shadow.py
==================
Geometry model for classifying whether a point in space (e.g. a Starlink
satellite) sits inside the Moon's shadow during a solar eclipse.

Method
------
This is the same "conical shadow" test used for spacecraft eclipse-by-Earth
analysis (see Vallado, *Fundamentals of Astrodynamics and Applications*,
"shadow" algorithms), just with the Moon as the occulting body instead of
Earth. It naturally handles any observer position — ground station or
orbiting satellite — with no need to first find the umbra/antumbra apex
points on Earth's surface.

For an observer at position r (geocentric), and geocentric positions of the
Sun (r_sun) and Moon (r_moon):

    v_sun  = r_sun  - r        # observer -> Sun
    v_moon = r_moon - r        # observer -> Moon
    theta  = angle between v_sun and v_moon      (apparent Sun-Moon separation)
    a_sun  = asin(R_sun  / |v_sun|)               (apparent radius of Sun)
    a_moon = asin(R_moon / |v_moon|)              (apparent radius of Moon)

Then, provided the Moon is nearer than the Sun (|v_moon| < |v_sun|, i.e. the
Moon is actually in front to occult it):

    theta < |a_sun - a_moon|      -> one disk entirely inside the other:
                                        a_moon >= a_sun -> UMBRA   (total)
                                        a_moon <  a_sun -> ANTUMBRA (annular)
    |a_sun-a_moon| <= theta < a_sun+a_moon -> PENUMBRA (partial)
    otherwise                              -> NONE (no eclipse at this point)

All positions must be in the same geocentric, non-rotating frame (this
module uses GCRS via astropy, which is close enough to J2000/MEME —
the frame the Starlink ephemeris files use — that the bias is negligible
for eclipse geometry, where angular precision only needs to be good to a
tiny fraction of a degree).

Usage
-----
    from eclipse_shadow import sun_moon_gcrs_km, classify_shadow
    import numpy as np
    from astropy.time import Time

    t = Time("2026-08-12T17:47:06", scale="utc")
    r_sun, r_moon = sun_moon_gcrs_km(t)
    status, theta_deg, a_sun_deg, a_moon_deg = classify_shadow(
        r_sat_km[None, :], r_sun[None, :], r_moon[None, :]
    )
"""

from __future__ import annotations

import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import numpy as np

R_SUN_KM = 696_000.0
R_MOON_KM = 1_737.4
R_EARTH_KM = 6_378.137

NONE, PENUMBRA, UMBRA, ANTUMBRA = "none", "penumbra", "umbra", "antumbra"


def sun_moon_gcrs_km(time) -> tuple[np.ndarray, np.ndarray]:
    """Geocentric Sun & Moon position vectors (km) in GCRS at astropy Time `time`.

    `time` may be scalar or an array-valued astropy.time.Time. Returns
    (r_sun, r_moon), each shaped (..., 3).
    """
    from astropy.coordinates import get_body

    sun = get_body("sun", time)
    moon = get_body("moon", time)
    r_sun = sun.cartesian.xyz.to("km").value.T  # (...,3)
    r_moon = moon.cartesian.xyz.to("km").value.T
    return r_sun, r_moon


def in_earth_daylight(r_obs_km: np.ndarray, r_sun_km: np.ndarray) -> np.ndarray:
    """True where the observer has an unoccluded line of sight to the Sun
    (Earth's own bulk isn't in the way). Cylindrical shadow model: the
    observer is in Earth's shadow if it's on the night-side hemisphere AND
    its perpendicular distance from the Earth-Sun line is less than Earth's
    radius. Good enough here — Earth's penumbra/umbra cone spread over a
    550 km LEO altitude is a tiny correction to this.

    This matters because classify_shadow() only tests Sun-Moon angular
    alignment as seen from the observer; without this check, an observer on
    the far side of Earth can satisfy that alignment purely by inertial-frame
    coincidence (the Sun-Moon axis extended through Earth), producing a false
    "umbra" hit for a point that's actually just in ordinary orbital night.
    """
    r_obs_km = np.asarray(r_obs_km, dtype=float)
    r_sun_km = np.asarray(r_sun_km, dtype=float)

    sun_dir = r_sun_km / np.linalg.norm(r_sun_km, axis=-1, keepdims=True)
    along = np.einsum("...i,...i->...", r_obs_km, sun_dir)
    perp = r_obs_km - along[..., None] * sun_dir
    perp_dist = np.linalg.norm(perp, axis=-1)

    night_side = along < 0
    behind_earth = perp_dist < R_EARTH_KM
    return ~(night_side & behind_earth)


def classify_shadow(
    r_obs_km: np.ndarray, r_sun_km: np.ndarray, r_moon_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify shadow status for observer positions against Sun/Moon.

    All inputs shaped (N, 3), same geocentric frame, km. Returns
    (status[N] str array, theta_deg[N], a_sun_deg[N], a_moon_deg[N]).
    """
    r_obs_km = np.asarray(r_obs_km, dtype=float)
    r_sun_km = np.asarray(r_sun_km, dtype=float)
    r_moon_km = np.asarray(r_moon_km, dtype=float)

    v_sun = r_sun_km - r_obs_km
    v_moon = r_moon_km - r_obs_km

    d_sun = np.linalg.norm(v_sun, axis=-1)
    d_moon = np.linalg.norm(v_moon, axis=-1)

    cos_theta = np.einsum("...i,...i->...", v_sun, v_moon) / (d_sun * d_moon)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    a_sun = np.arcsin(np.clip(R_SUN_KM / d_sun, -1.0, 1.0))
    a_moon = np.arcsin(np.clip(R_MOON_KM / d_moon, -1.0, 1.0))

    moon_in_front = d_moon < d_sun
    daylight = in_earth_daylight(r_obs_km, r_sun_km)
    eligible = moon_in_front & daylight

    status = np.full(r_obs_km.shape[:-1], NONE, dtype=object)

    partial = eligible & (theta < (a_sun + a_moon))
    status[partial] = PENUMBRA

    total_or_annular = eligible & (theta < np.abs(a_sun - a_moon))
    is_umbra = total_or_annular & (a_moon >= a_sun)
    is_antumbra = total_or_annular & (a_moon < a_sun)
    status[is_umbra] = UMBRA
    status[is_antumbra] = ANTUMBRA

    return (
        status,
        np.degrees(theta),
        np.degrees(a_sun),
        np.degrees(a_moon),
    )


if __name__ == "__main__":
    # Quick self-test: greatest eclipse of 2026-08-12 at its published peak
    # (17:47:06 UTC, path over Greenland/Iceland/N. Spain) should classify a
    # point ON Earth's surface under the path as UMBRA.
    from astropy.time import Time

    t = Time("2026-08-12T17:47:06", scale="utc")
    r_sun, r_moon = sun_moon_gcrs_km(t)
    print("Sun (km):", r_sun)
    print("Moon (km):", r_moon)

    # Earth center, as a sanity check of the "no shadow" case:
    status, theta, a_sun, a_moon = classify_shadow(
        np.array([[0.0, 0.0, 0.0]]), r_sun[None, :], r_moon[None, :]
    )
    print(f"Earth center -> {status[0]}  theta={theta[0]:.4f} deg "
          f"a_sun={a_sun[0]:.4f} a_moon={a_moon[0]:.4f}")
