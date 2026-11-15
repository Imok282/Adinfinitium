"""Terrain generator (KAN-26): conditional diffusion that imagines plausible 5 m terrain where LOLA has no laser spots.

About 90% of the 5 m pixels are interpolated, not measured. The model is trained ONLY on real laser-measured pixels (hide-and-check:
part of the measurements are hidden, the model must predict them back), and every output is labelled GENERATED — NOT OBSERVED.
Physics backup: gap-fill the measurements and add noise drawn from LOLA's own error map (worldmodel.terrain.backup).
"""
