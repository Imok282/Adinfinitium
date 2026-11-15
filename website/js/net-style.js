// Shared look of the pilot-network visual (3D and static 2D). Edit here to restyle.
export const STYLE = {
  bgTop: '#2c2b28',           // warm charcoal (round-3b palette)
  bgBottom: '#1d1d1b',        // darker charcoal
  lineBase: [0.94, 0.90, 0.82],// cream
  pos: [0.62, 0.92, 0.72],     // positive weights: sage
  neg: [1.0, 0.58, 0.45],      // negative weights: peach-terracotta
  strong: [0.85, 0.80, 1.0],   // strongest |w|: lavender
  tint: 1.0,                  // 0 = all lavender, 1 = fully sign-coloured
  lineGain: 0.3,              // brightness per line (additive)
  dot: [0.96, 0.94, 0.88],
  pulse: [0.91, 0.73, 0.63],
  spacing: 3.2,               // world units between columns
  segments: 14,               // bezier segments per connection
};

export const colHeight = (size) => 0.9 + 0.62 * Math.log2(Math.max(2, size));   // wide layers are tall, narrow ones short

