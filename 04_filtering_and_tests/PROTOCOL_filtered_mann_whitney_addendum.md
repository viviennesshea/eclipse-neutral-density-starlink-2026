# Addendum to PROTOCOL_filtered_mann_whitney.md (written after seeing its results, before rerunning)

Change, and only this: the filters become, in order
1. Altitude >= 440 km (median altitude in the pre file before the window start).
2. Burn check in both directions: rise = largest increase and drop = largest decrease of the one-orbit-smoothed orbit size over any 3 h (pre file from its start to the window end; post file's first 8 h).
   Removed if rise > IQR x3 upper fence of rise, or drop < IQR x3 lower fence of drop (fences over the satellites passing step 1).
3. IQR x3 backstop on along_km.
Tests, window, placebos unchanged. Results of this rerun are exploratory because the filter was changed after a look.
