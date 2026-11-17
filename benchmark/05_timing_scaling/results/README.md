# Scaling benchmark timings

The water and LJ electrolyte measurements cover fixed-device size scaling, strong scaling and weak scaling. Each row of [timings.csv](timings.csv) gives one recorded repeat in milliseconds per force evaluation. The observable sums the Pair and Kspace stage timings, including communication within each stage. Separately timed neighbor-list construction and other integration stages are excluded.

[data.json](data.json) contains the plotted means, repeats and process grids. The water and lj-salt directories contain the corresponding archived stage timings and Kokkos evidence.
