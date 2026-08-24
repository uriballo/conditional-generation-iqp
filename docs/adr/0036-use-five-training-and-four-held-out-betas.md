# Use five Training and four Held-out Ising Conditions

The frozen Ising split uses five Training inverse temperatures spanning the
Target-calibrated interval, including both endpoints and the released
PennyLane anchor \(\beta=1/3\). Each adjacent Training pair contributes its
beta midpoint as one of four Held-out Conditions; the grid may be slightly
nonuniform to preserve the anchor, and Held-out Targets cannot influence
tuning or checkpoint selection.
