# Protocol: filtered Mann-Whitney tests on the pre/post ephemeris gap (written before computing)

Written 2026-09-24, after the dosed-vs-undosed residual test and its placebo.

## Common window (every satellite)
W = [2026-08-12 15:29, 20:03 UTC] (first dose start to last dose end in the fleet), so every satellite's window contains the whole eclipse.
pre = latest ephemeris file CREATED before W start; post = earliest file CREATED after W end. along_km = (r_pre - r_post) . unit(v_post) at W end (read directly, or the gap fitted over
the post file's first 8 h and extended back). Positive = the satellite is behind the forecast = LESS drag than forecast.

## Filters (fixed before any test)
1. Burn detection (does not use the outcome): orbit size a (vis-viva) along the pre file from its start to W end and along the post file's first 8 h, smoothed with a one-orbit (95 min)
   running mean. Drag can only shrink a, so rise = the largest increase of smoothed a over any 3 h span in either file. Satellites above the fleet's IQR x3 upper fence of rise are removed.
2. Backstop: IQR x3 on along_km over the satellites left.

## Tests
(a) Dosed (umbra_seconds > 0) vs undosed: Mann-Whitney on along_km, stratified by inclination shell (van Elteren, strata need >= 5 in each group), permutation p. Plain Mann-Whitney also shown.
    Li et al. direction: P(dosed > undosed) > 0.5.
(b) Dosed only, split at the median of dose LENGTH (total minutes in shadow = sum of dose span minutes): long vs short, stratified by shell as in (a). Li direction: P(long > short) > 0.5.
    Also split at the median umbra_seconds.
(c) Spearman of dose length and of umbra_seconds with along_km, within shell, pooled (stratified permutation).
Placebo: the identical (a) and (b) with W shifted by -24 h and +24 h.
Not allowed after seeing results: changing the window, filters, splits or tests.
