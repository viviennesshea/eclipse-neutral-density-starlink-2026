# Step 2: Shadow geometry and dose (paper §2.3–2.4, Figures 1–4)

> Paths: inputs are read from the data folder and outputs written under the output folder (`--data-dir` / `--out-dir`, see the main README). The "original location" is where each script lived in the project it was developed in.


Original locations: `experiment_9_4/src/` for the shadow scripts, `experiment_9_23/src/` for `02_satellite_dosage.py`, and `experiment_9_23/` for `build_satellite_planes.py`.

| Script | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `eclipse_shadow.py` | Shadow geometry helper (Sun/Moon positions, umbra/penumbra classification). Imported by the scripts below. | – | – |
| `01_shadow_geometry.py` | The umbra and penumbra footprint through the eclipse, at ground level and at satellite altitude. | – | `data/01_shadow_geometry_map_layer.csv` |
| `02_render_shadow_map.py` | Draws the shadow maps. | step 01 output | Figures 1 and 2 |
| `03_satellite_shadow_passes.py` | Finds every time each satellite enters and leaves the shadow, from its own ephemeris. | `states.parquet`, `eclipse_targets.csv` | `03_satellite_shadow_passes.csv` |
| `04_shadow_dose_heatmap.py` | Dose field (equivalent umbra-seconds) on a grid at satellite altitude, and the fraction-of-Sun-covered function used everywhere else. | – | `04_shadow_dose_field.csv`, Figure 3 |
| `05_satellite_paths_on_dose_field.py` | Satellite paths drawn over the dose field. | steps 03 and 04 outputs, `states.parquet` | Figure 4 |
| `02_satellite_dosage.py` | **The dose used in the tests:** each satellite's shadow dose, calculated every second along its own path. Imports `eclipse_shadow.py` and `04_shadow_dose_heatmap.py` from `experiment_9_4/src/`. | `states.parquet`, `eclipse_targets.csv` | `results/satellite_dosage.csv` |
| `build_satellite_planes.py` | Orbital tilt (inclination shell) and orbital plane for each satellite, from the TLE data. The tilt is used to compare like with like in step 4. | step 1 TLE tables | `satellite_planes.csv` |

Run order: 01 → 02, 03 → 04 → 05, then `02_satellite_dosage.py` and `build_satellite_planes.py`.

Figure 1 (ground level) and Figure 2 (satellite altitude) both come from scripts 01 and 02:

```
python 02_shadow_geometry_and_dose/01_shadow_geometry.py                     # satellite altitude (465.6 km) -> Figure 2
python 02_shadow_geometry_and_dose/02_render_shadow_map.py
python 02_shadow_geometry_and_dose/01_shadow_geometry.py --altitude 0        # ground level -> Figure 1
python 02_shadow_geometry_and_dose/02_render_shadow_map.py --tag ground --altitude-label "ground level, 0 km"
```

`04_shadow_dose_heatmap.py` is slow (about 30 minutes on a laptop); the other scripts take seconds to a minute or two.
