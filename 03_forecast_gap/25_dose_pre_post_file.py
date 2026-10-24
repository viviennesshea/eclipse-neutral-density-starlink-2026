#!/usr/bin/env python3
"""
25_dose_pre_post_file.py
========================
The per-dose-span file: one row per dose span per dosed satellite, pre-dose and post-dose ephemeris predictions at the dose end time, their snapshot creation times and
time offsets from the dose end, and the distance between the two predictions. Built from results/ephemeris_drift/dose_span_predictions.csv (no-dose satellites and pseudo-passes dropped).
Output (new): results/ephemeris_drift/dose_pre_post_predictions.csv
Units: dose_delta_time in minutes; *_delta_time in hours; positions and distances in km. delta_time = pre: hours the pre snapshot was created BEFORE the dose end (negative = created after);
post: hours the post snapshot was created AFTER the dose end.
"""
import os as _os, sys as _sys  # locate starlink_paths.py (repository root) and import it
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "starlink_paths.py")) and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
import starlink_paths as SP  # noqa: E402  (--data-dir / --out-dir, see starlink_paths.py)
import sys
sys.dont_write_bytecode = True
import os
import pandas as pd

OUT = SP.ED
d = pd.read_csv(os.path.join(OUT, "dose_span_predictions.csv"), dtype={"sat_key": str}); d = d[~d.is_pseudo_no_dose].sort_values(["sat_key", "pass_number"])
o = pd.DataFrame({
    "sat_key": d.sat_key, "name": d.name, "dose_number": d.pass_number.astype(int), "dose_start_time": d.dose_start_utc, "dose_end_time": d.dose_end_utc,
    "dose_delta_time": d.dose_span_min, "dose_umbra_seconds": d.dose_amount_umbra_s,
    "pre_dose_ephemeris_prediction_x_km": d.pre_pred_x_km, "pre_dose_ephemeris_prediction_y_km": d.pre_pred_y_km, "pre_dose_ephemeris_prediction_z_km": d.pre_pred_z_km,
    "pre_dose_ephemeris_snapshot_time": d.pre_file_download_utc, "pre_dose_ephemeris_delta_time": d.hours_pre_download_to_dose_end,
    "post_dose_ephemeris_prediction_x_km": d.post_back_x_km, "post_dose_ephemeris_prediction_y_km": d.post_back_y_km, "post_dose_ephemeris_prediction_z_km": d.post_back_z_km,
    "post_dose_ephemeris_snapshot_time": d.post_file_download_utc, "post_dose_ephemeris_delta_time": d.hours_dose_end_to_post_download,
    "delta_predictions_distance_km": d.delta_total_km, "delta_along_track_km": d.delta_along_km, "post_method": d.post_method})
o.to_csv(os.path.join(OUT, "dose_pre_post_predictions.csv"), index=False, float_format="%.6g")
print(f"rows {len(o)} | satellites {o.sat_key.nunique()} | columns {len(o.columns)}")
print("post_method:", o.post_method.value_counts().to_dict())
print("pre snapshot created AFTER the dose end (negative pre delta_time):", int((o.pre_dose_ephemeris_delta_time < 0).sum()))
print("delta_predictions_distance_km: median %.3f, 25-75%% %.3f-%.3f, max %.2f" % (o.delta_predictions_distance_km.median(), o.delta_predictions_distance_km.quantile(.25), o.delta_predictions_distance_km.quantile(.75), o.delta_predictions_distance_km.max()))
print(o.head(2).T.to_string())
