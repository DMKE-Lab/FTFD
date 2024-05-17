import numpy as np

def compute_membership_distance(h, r, o, ρi, σ=1.0):
    distance = np.maximum(0, np.linalg.norm(h - (r + o))**2 - ρi)
    return 1 / (1 + np.exp(-σ * distance))

def compute_membership_radius_gap(h, r, o, ρi, σ=1.0):
    radius_gap = np.minimum(0, np.linalg.norm(h - (r + o))**2 - ρi)
    return 1 / (1 + np.exp(-σ * radius_gap))

# def compute_distance_formula(r1, o1, r2, o2, σ=1.0):
#     distance = np.linalg.norm((r1 + o1) - (r2 + o2))**2
#     return 1 / (1 + np.exp(-σ * distance))
#
# def compute_radius_gap_formula(r1, o1, r2, o2, σ=1.0):
#     radius_gap = -np.linalg.norm((r1 + o1) - (r2 + o2))**2
#     return 1 / (1 + np.exp(-σ * radius_gap))

# Example values
h = np.array([1.0, 2.0])
r = np.array([0.5, 1.0])
# r1 = np.array([0.5, 1.0])
# r2 = np.array([0.6, 0.9])
o = np.array([0.2, 0.3])
ρi = 0.1

μs_distance = compute_membership_distance(h, r, o, ρi)
μs_radius_gap = compute_membership_radius_gap(h, r, o, ρi)

print(f"Membership Distance: {μs_distance}")
print(f"Membership Radius Gap: {μs_radius_gap}")
