import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd

# ----------------------------
# Geometry + params
# ----------------------------
INNER_DIAMETER = 0.095
R_CUP = INNER_DIAMETER / 2.0
R_BALL = 0.020
R_EFF = max(1e-6, R_CUP - R_BALL)
# RACK_SPACING = INNER_DIAMETER  # For touching cups, spacing = diameter
RACK_SPACING = R_CUP + 0.01  # For touching cups, spacing = diameter

@dataclass
class ShooterProfile:
    name: str
    sigma_depth: float  # x-axis (depth) 1-sigma
    sigma_lat: float    # y-axis (lateral) 1-sigma
    rho: float = 0.0    # correlation depth<->lat
    yaw_rad: float = 0.0
    bias_depth: float = 0.0
    bias_lat: float = 0.0

def cov_from_profile(p: ShooterProfile) -> np.ndarray:
    Sd, Sl, rho = p.sigma_depth, p.sigma_lat, p.rho
    base = np.array([[Sd**2, rho*Sd*Sl],
                     [rho*Sd*Sl, Sl**2]])
    c, s = math.cos(p.yaw_rad), math.sin(p.yaw_rad)
    R = np.array([[c, -s],[s, c]])
    return R @ base @ R.T

@dataclass
class Cup:
    cx: float
    cy: float
    alive: bool = True

@dataclass
class Rack:
    cups: List[Cup]
    def alive_indices(self): return [i for i,c in enumerate(self.cups) if c.alive]
    def positions(self): return np.array([[c.cx,c.cy] for c in self.cups])
    def mark_hit(self, idx:int): self.cups[idx].alive = False

# Layout helpers
def triangle_10_rack(origin=(0.0,0.0), spacing=RACK_SPACING) -> Rack:
    x0, y0 = origin
    cups = []
    rows = [1,2,3,4]  # Point downward: 4 cups at back, 1 cup at front
    y_spacing = spacing * 0.89
    for r, n in enumerate(rows):
        y = y0 + r * y_spacing  # Move forward (positive y)
        x_start = x0 - (n-1) * spacing / 2.0
        for k in range(n):
            cups.append(Cup(cx=x_start + k*spacing, cy=y))
    return Rack(cups)

def six_pyramid_rack(origin=(0.0,0.0), spacing=RACK_SPACING) -> Rack:
    x0, y0 = origin
    cups = []
    rows = [1,2,3]  # Point downward: 3 cups at back, 1 cup at front
    y_spacing = spacing * 0.89
    for r, n in enumerate(rows):
        y = y0 + r * y_spacing  # Move forward (positive y)
        x_start = x0 - (n-1) * spacing / 2.0
        for k in range(n):
            cups.append(Cup(cx=x_start + k*spacing, cy=y))
    return Rack(cups)

def vertical_stack(n:int, center=(0.0,0.0), spacing=RACK_SPACING)->Rack:
    x0,y0 = center
    offs = (np.arange(n) - (n-1)/2.0)*spacing
    return Rack([Cup(cx=x0, cy=y0+dy) for dy in offs])

def horizontal_stack(n:int, center=(0.0,0.0), spacing=RACK_SPACING)->Rack:
    x0,y0 = center
    offs = (np.arange(n) - (n-1)/2.0)*spacing
    return Rack([Cup(cx=x0+dx, cy=y0) for dx in offs])

def sideways_olympics(center=(0.0,0.0), spacing=RACK_SPACING)->Rack:
    x0,y0 = center
    x_spacing = spacing * 0.43
    left = [(x0 - x_spacing, y0 - spacing/2.0),
            (x0 - x_spacing, y0 + spacing/2.0)]
    right = [(x0 + x_spacing, y0 - spacing),
             (x0 + x_spacing, y0),
             (x0 + x_spacing, y0 + spacing)]
    cups=[Cup(cx=a, cy=b) for (a,b) in (left+right)]
    return Rack(cups)

def zipper(center=(0.0,0.0), spacing=RACK_SPACING)->Rack:
    x0,y0 = center
    x_spacing = spacing * 0.43
    left = [(x0 - x_spacing, y0 - spacing),
            (x0 - x_spacing, y0),
            (x0 - x_spacing, y0 + spacing)]
    right = [(x0 + x_spacing, y0 - spacing/2.0),
             (x0 + x_spacing, y0 + spacing/2.0),
             (x0 + x_spacing, y0 + 1.5*spacing)]
    return Rack([Cup(cx=a, cy=b) for (a,b) in (left+right)])

def assign_hit(point:np.ndarray, rack:Rack, alive_only=True) -> Optional[int]:
    idxs = rack.alive_indices() if alive_only else list(range(len(rack.cups)))
    if not idxs:
        return None
    centers = rack.positions()[idxs]
    d = np.linalg.norm(centers - point.reshape(1,2), axis=1)
    
    # Use 90% of the effective radius instead of 100%
    effective_radius = R_EFF * 0.9
    
    j = np.where(d <= effective_radius)[0]
    if j.size == 0:
        return None
    j0 = int(j[np.argmin(d[j])])
    return idxs[j0]

def sample_shots(N:int, profile:ShooterProfile, target_xy:np.ndarray, 
                shooter_angle_deg:float=0, seed=None)->np.ndarray:
    """
    Sample shots with optional shooter angle rotation
    shooter_angle_deg: 0 = straight on, 60 = from side
    """
    rng = np.random.default_rng(seed)
    mu = np.array([profile.bias_depth, profile.bias_lat])
    cov = cov_from_profile(profile)
    
    shots_local = rng.multivariate_normal(mean=mu, cov=cov, size=N)
    
    if shooter_angle_deg != 0:
        effective_angle_deg = 30
        angle_rad = math.radians(effective_angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        shots_local = shots_local @ R.T
    
    return shots_local + target_xy

def plot_strategy_visualization(rack, target_point, profile, title, N=1000, seed=0, shooter_angle_deg=0):
    """
    Generic function to visualize any rack configuration with shot patterns
    shooter_angle_deg: 0 = straight on, 60 = from side
    """
    plt.figure(figsize=(10, 8))
    
    # Sample shots with proper angle
    shots = sample_shots(N, profile, target_point, shooter_angle_deg, seed=seed)
    
    # Separate hits and misses
    hits = []
    misses = []
    for pt in shots:
        j = assign_hit(pt, rack, alive_only=False)
        if j is None:
            misses.append(pt)
        else:
            hits.append(pt)
    
    hits = np.array(hits) if hits else np.empty((0,2))
    misses = np.array(misses) if misses else np.empty((0,2))
    
    # Draw cups as circles with labels
    theta = np.linspace(0, 2*math.pi, 128)
    for i, c in enumerate(rack.cups):
        circle_x = c.cx + R_EFF * np.cos(theta)
        circle_y = c.cy + R_EFF * np.sin(theta)
        plt.plot(circle_x, circle_y, 'b-', linewidth=2)
        plt.text(c.cx, c.cy, str(i), ha='center', va='center', fontweight='bold')
    
    # Plot shots
    if hits.size: 
        plt.scatter(hits[:,0], hits[:,1], s=6, alpha=0.6, label=f'hits ({hits.shape[0]})', c='green')
    if misses.size: 
        plt.scatter(misses[:,0], misses[:,1], s=3, alpha=0.3, label=f'misses ({misses.shape[0]})', c='red')
    
    # Mark target point
    plt.scatter(target_point[0], target_point[1], s=100, c='yellow', marker='*', 
                edgecolor='black', label='target', zorder=5)
    
    # Removed the arrow as requested
    
    plt.axis('equal')
    plt.xlabel("depth (m)")
    plt.ylabel("lateral (m)")
    angle_str = f" ({shooter_angle_deg}° angle)" if shooter_angle_deg != 0 else ""
    plt.title(f"{title}{angle_str} - Profile: {profile.name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    print(f"{title}: {hits.shape[0]} hits, {misses.shape[0]} misses")
    plt.show()

# Updated strategy-specific shot functions
def starting_shot_from_front(cup_id=0, profile=None, N=1000, seed=0):
    """
    Shoot at a 10-cup triangle rack from the front position
    """
    if profile is None:
        profile = ShooterProfile("default", sigma_depth=0.03, sigma_lat=0.06)
    
    rack = triangle_10_rack(origin=(0.0, 0.0))
    
    if cup_id >= len(rack.cups):
        print(f"Invalid cup_id {cup_id}. Rack has {len(rack.cups)} cups (0-{len(rack.cups)-1})")
        return
    
    target_point = np.array([rack.cups[cup_id].cx, rack.cups[cup_id].cy])
    title = f"Starting Shot from Front - Target Cup {cup_id}"
    
    plot_strategy_visualization(rack, target_point, profile, title, N, seed, shooter_angle_deg=0)

def starting_shot_from_side(cup_id=0, profile=None, N=1000, seed=0):
    """
    Shoot at a 10-cup triangle rack from the side position (60-degree angle)
    """
    if profile is None:
        profile = ShooterProfile("default", sigma_depth=0.03, sigma_lat=0.06)
    
    rack = triangle_10_rack(origin=(0.0, 0.0))
    
    if cup_id >= len(rack.cups):
        print(f"Invalid cup_id {cup_id}. Rack has {len(rack.cups)} cups (0-{len(rack.cups)-1})")
        return
    
    target_point = np.array([rack.cups[cup_id].cx, rack.cups[cup_id].cy])
    title = f"Starting Shot from Side - Target Cup {cup_id}"
    
    plot_strategy_visualization(rack, target_point, profile, title, N, seed, shooter_angle_deg=60)

def six_pyramid_shot(angle="front", cup_id=0, profile=None, N=1000, seed=0):
    """
    Shoot at a 6-cup pyramid formation
    angle: "front" or "side" (60-degree)
    """
    if profile is None:
        profile = ShooterProfile("default", sigma_depth=0.03, sigma_lat=0.06)
    
    rack = six_pyramid_rack(origin=(0.0, 0.0))
    
    if cup_id >= len(rack.cups):
        print(f"Invalid cup_id {cup_id}. 6-pyramid has {len(rack.cups)} cups (0-{len(rack.cups)-1})")
        return
    
    target_point = np.array([rack.cups[cup_id].cx, rack.cups[cup_id].cy])
    title = f"6-Pyramid Shot from {angle.title()} - Target Cup {cup_id}"
    
    shooter_angle = 0 if angle == "front" else 60
    plot_strategy_visualization(rack, target_point, profile, title, N, seed, shooter_angle_deg=shooter_angle)

def olympics_shot(cup_id=0, angle="front", profile=None, N=1000, seed=0):
    """
    Shoot at sideways olympics formation
    angle: "front" or "side"
    """
    if profile is None:
        profile = ShooterProfile("default", sigma_depth=0.03, sigma_lat=0.06)
    
    rack = sideways_olympics(center=(0.0, 0.0))
    
    if cup_id >= len(rack.cups):
        print(f"Invalid cup_id {cup_id}. Olympics formation has {len(rack.cups)} cups (0-{len(rack.cups)-1})")
        return
    
    target_point = np.array([rack.cups[cup_id].cx, rack.cups[cup_id].cy])
    title = f"Olympics Shot from {angle.title()} - Target Cup {cup_id}"
    
    shooter_angle = 0 if angle == "front" else 60
    plot_strategy_visualization(rack, target_point, profile, title, N, seed, shooter_angle_deg=shooter_angle)

def zipper_shot(cup_id=0, angle="front", profile=None, N=1000, seed=0):
    """
    Shoot at zipper formation
    angle: "front" or "side"
    """
    if profile is None:
        profile = ShooterProfile("default", sigma_depth=0.03, sigma_lat=0.06)
    
    rack = zipper(center=(0.0, 0.0))
    
    if cup_id >= len(rack.cups):
        print(f"Invalid cup_id {cup_id}. Zipper formation has {len(rack.cups)} cups (0-{len(rack.cups)-1})")
        return
    
    target_point = np.array([rack.cups[cup_id].cx, rack.cups[cup_id].cy])
    title = f"Zipper Shot from {angle.title()} - Target Cup {cup_id}"
    
    shooter_angle = 0 if angle == "front" else 60
    plot_strategy_visualization(rack, target_point, profile, title, N, seed, shooter_angle_deg=shooter_angle)

def two_cup_vertical_shot(aim='front', profile=None, N=1000, seed=0):
    """
    Shoot at 2 cups stacked vertically from the front position
    aim: 'front' or 'back' - which cup to target
    """
    if profile is None:
        profile = ShooterProfile("default", sigma_depth=0.03, sigma_lat=0.06)
    
    rack = vertical_stack(2, center=(0.0, 0.0))
    
    # Determine which cup to target
    pos = rack.positions()
    if aim == 'front':
        target_idx = int(np.argmin(pos[:, 1]))  # Front cup has smaller y-coordinate
    else:
        target_idx = int(np.argmax(pos[:, 1]))  # Back cup has larger y-coordinate
    
    target_point = np.array([rack.cups[target_idx].cx, rack.cups[target_idx].cy])
    title = f"2-Cup Vertical Stack - Target {aim} cup"
    
    plot_strategy_visualization(rack, target_point, profile, title, N, seed, shooter_angle_deg=0)

def single_cup_shot(profile=None, N=1000, seed=0):
    """
    Shoot at a single cup from the front position
    """
    if profile is None:
        profile = ShooterProfile("default", sigma_depth=0.03, sigma_lat=0.06)
    
    # Create a single cup at the origin
    rack = Rack([Cup(cx=0.0, cy=0.0)])
    
    target_point = np.array([0.0, 0.0])
    title = "Single Cup Shot"
    
    plot_strategy_visualization(rack, target_point, profile, title, N, seed, shooter_angle_deg=0)

# Strategy and game simulation classes (maintained for compatibility)
@dataclass
class Strategy:
    name: str
    rerack_threshold: int
    rerack_type: str  # 'pyramid','sideways_olympics','zipper'
    aim_policy: str   # 'center','seam','back_rim'
    yaw_rad: float = 0.0

def build_rerack(rerack_type:str, alive_count:int, center=(0.0,0.0))->Rack:
    if rerack_type=='pyramid':
        full = triangle_10_rack(center, RACK_SPACING).cups
        return Rack([Cup(c.cx,c.cy) for c in full[:alive_count]])
    elif rerack_type=='sideways_olympics':
        r = sideways_olympics(center, RACK_SPACING)
        while len(r.cups) > alive_count: r.cups.pop()
        return r
    elif rerack_type=='zipper':
        r=zipper(center, RACK_SPACING)
        while len(r.cups) > alive_count: r.cups.pop()
        return r
    else:
        return vertical_stack(alive_count, center, RACK_SPACING)

# Predefined shooter profiles
bad = ShooterProfile("bad", sigma_depth=0.08, sigma_lat=0.11)   # elongated depth
okay = ShooterProfile("okay", sigma_depth=0.04, sigma_lat=0.075)
good = ShooterProfile("good", sigma_depth=0.02, sigma_lat=0.04)

# Demo function to show all visualizations
def demo_all_shots():
    """
    Run demonstrations of all shot types
    """
    print("=== Beer Pong Strategy Visualization Demo ===\n")
    
    # Use the 'okay' profile for demonstrations
    profile = okay
    
    print("1. Single cup shot")
    single_cup_shot(profile=profile, seed=1)
    
    print("\n2. 2-cup vertical stack (targeting front cup)")
    two_cup_vertical_shot(aim='front', profile=profile, seed=1)
    
    print("\n3. Starting shot from front (targeting cup 4 - middle front)")
    starting_shot_from_front(cup_id=4, profile=profile, seed=1)
    
    print("\n4. Starting shot from side (targeting cup 9 - back corner)")
    starting_shot_from_side(cup_id=9, profile=profile, seed=1)
    
    print("\n5. 6-pyramid shot from front (targeting cup 0 - front tip)")
    six_pyramid_shot(angle="front", cup_id=0, profile=profile, seed=1)
    
    print("\n6. 6-pyramid shot from side (targeting cup 2 - middle row)")
    six_pyramid_shot(angle="side", cup_id=2, profile=profile, seed=1)
    
    print("\n7. Olympics formation shot (targeting cup 2)")
    olympics_shot(cup_id=2, profile=profile, seed=1)
    
    print("\n8. Zipper formation shot (targeting cup 1)")
    zipper_shot(cup_id=1, profile=profile, seed=1)

# Uncomment to run demo
demo_all_shots()