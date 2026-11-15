"""Neural light field (KAN-24): a fast student for the exact skyline ray-cast.

The network predicts the terrain SKYLINE (horizon elevation for every compass degree) at a point. Time never enters
the network: Sun and Earth directions come from exact ephemeris (SPICE; stand-in until KAN-23), and plain geometry
turns skyline + direction into Sun fraction and Earth visibility (physics.py). Physics backup: the exact ray-cast (teacher.py).
"""
