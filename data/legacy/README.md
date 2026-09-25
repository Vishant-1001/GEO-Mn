# Legacy data (not loaded at runtime)

`reserve_cache.csv` is the earlier ~0.25 degree (~27 km) cached grid from the first prototype. Its
`probability` column was an uncalibrated classifier output and must not be read as a deposit or
reserve probability. It is superseded by `data/exploration_grid.csv` (0.01 degree relative
prospectivity ranks) and is retained only for historical reference. No API endpoint reads it.
