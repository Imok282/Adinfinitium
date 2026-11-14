"""Hazard detector (KAN-25): marks each terrain-tile pixel as safe or dangerous for landing.

Physics backup first: slope >= lander.touchdown_limits.max_slope_deg (config/mission.yaml).
The U-Net is always compared against that backup on the same input.
"""
