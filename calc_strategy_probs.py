import numpy as np
import pandas as pd
import itertools
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
from collections import Counter

# Import everything from the original file (assuming it's named beer_pong_sim.py)
# If running as standalone, copy the necessary classes and functions
from math_1 import (
    ShooterProfile, Rack, Cup, assign_hit, sample_shots,
    triangle_10_rack, six_pyramid_rack, sideways_olympics, zipper, 
    vertical_stack, R_CUP, RACK_SPACING,
    bad, okay, good
)

def calculate_detailed_hit_probability(profile: ShooterProfile, rack: Rack, target_cup_idx: int, 
                                     shooter_angle_deg: float = 0, N: int = 1000, seed: int = 42) -> Dict:
    """
    Calculate detailed probability breakdown when aiming at a specific cup
    
    Args:
        profile: ShooterProfile with accuracy parameters
        rack: Rack configuration
        target_cup_idx: Index of the cup being aimed at (in the current rack)
        shooter_angle_deg: Shooter angle (0=front, 60=side)
        N: Number of shots to simulate
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with detailed probability breakdown:
        - p_hit_target: Probability of hitting the target cup
        - p_hit_any: Probability of hitting any cup
        - p_miss: Probability of missing entirely
        - other_cups_hit: Counter of hits on non-target cups
    """
    if target_cup_idx >= len(rack.cups):
        return {
            'p_hit_target': 0.0,
            'p_hit_any': 0.0,
            'p_miss': 1.0,
            'other_cups_hit': Counter()
        }
    
    target_point = np.array([rack.cups[target_cup_idx].cx, rack.cups[target_cup_idx].cy])
    shots = sample_shots(N, profile, target_point, shooter_angle_deg, seed=seed)
    
    hits_target = 0
    hits_any = 0
    other_cups_hit = Counter()
    
    for shot in shots:
        hit_cup_idx = assign_hit(shot, rack, alive_only=False)
        
        if hit_cup_idx is not None:
            hits_any += 1
            if hit_cup_idx == target_cup_idx:
                hits_target += 1
            else:
                other_cups_hit[hit_cup_idx] += 1
    
    return {
        'p_hit_target': hits_target / N,
        'p_hit_any': hits_any / N,
        'p_miss': 1.0 - (hits_any / N),
        'other_cups_hit': other_cups_hit
    }

def format_other_cups_distribution(other_cups_hit: Counter, total_shots: int, 
                                 original_cup_indices: List[int]) -> str:
    """
    Format the other cups hit distribution as a string
    
    Args:
        other_cups_hit: Counter of hits on non-target cups (using rack indices)
        total_shots: Total number of shots for percentage calculation
        original_cup_indices: Mapping from rack indices to original cup indices
    
    Returns:
        Formatted string like "1:0.05,2:0.03" or empty string if no other hits
    """
    if not other_cups_hit:
        return ""
    
    # Convert rack indices to original indices and calculate percentages
    distribution_parts = []
    for rack_idx, count in other_cups_hit.items():
        original_idx = original_cup_indices[rack_idx]
        percentage = count / total_shots
        distribution_parts.append(f"{original_idx}:{percentage:.3f}")
    
    return ",".join(sorted(distribution_parts))

def get_all_configurations(min_cups: int = 5, max_cups: int = 10) -> List[List[int]]:
    """
    Generate all possible cup configurations with at least min_cups remaining
    
    Args:
        min_cups: Minimum number of cups that must remain
        max_cups: Maximum number of cups (total cups in rack)
    
    Returns:
        List of configurations, where each configuration is a sorted list of remaining cup indices
    """
    all_cups = list(range(max_cups))
    configurations = []
    
    for num_remaining in range(min_cups, max_cups + 1):
        for config in itertools.combinations(all_cups, num_remaining):
            configurations.append(sorted(list(config)))
    
    return configurations

def create_rack_from_config(base_rack_func, config: List[int], *args, **kwargs) -> Rack:
    """
    Create a rack with only the specified cups remaining
    
    Args:
        base_rack_func: Function that creates the base rack (e.g., triangle_10_rack)
        config: List of cup indices that should remain
        *args, **kwargs: Arguments for the base rack function
    
    Returns:
        Rack with only the specified cups
    """
    full_rack = base_rack_func(*args, **kwargs)
    remaining_cups = [full_rack.cups[i] for i in config]
    return Rack(remaining_cups)

def calculate_layout_probabilities(profile: ShooterProfile, base_rack_func, layout_name: str,
                                 shooter_angle_deg: float, min_cups: int, max_cups: int,
                                 N: int = 1000) -> List[Dict]:
    """
    Calculate detailed hit probabilities for a specific layout
    
    Returns:
        List of result dictionaries with layout information
    """
    configurations = get_all_configurations(min_cups=min_cups, max_cups=max_cups)
    results = []
    
    for config in configurations:
        if base_rack_func == triangle_10_rack:
            rack = create_rack_from_config(base_rack_func, config, origin=(0.0, 0.0))
        elif base_rack_func in [sideways_olympics, zipper]:
            rack = create_rack_from_config(base_rack_func, config, center=(0.0, 0.0))
        else:  # six_pyramid_rack
            rack = create_rack_from_config(base_rack_func, config, origin=(0.0, 0.0))
        
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, shooter_angle_deg, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': layout_name,
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    return results

def get_final_two_probabilities(profile: ShooterProfile, N: int = 1000) -> List[Dict]:
    """
    Get the final 2 cups and single cup probabilities with layout labels
    """
    results = []
    
    # 2-cup vertical configuration
    rack_2_vertical = vertical_stack(2, center=(0.0, 0.0))
    config_2_cups = [0, 1]
    
    for aim_idx in range(2):
        original_cup_idx = config_2_cups[aim_idx]
        
        prob_data = calculate_detailed_hit_probability(
            profile, rack_2_vertical, aim_idx, shooter_angle_deg=0, N=N
        )
        
        other_cups_dist = format_other_cups_distribution(
            prob_data['other_cups_hit'], N, config_2_cups
        )
        
        results.append({
            'Layout': 'final2',
            'Cups_Remaining': str(config_2_cups),
            'Aimed_Cup': original_cup_idx,
            'P_Hit_Target': prob_data['p_hit_target'],
            'P_Hit_Any': prob_data['p_hit_any'],
            'P_Miss': prob_data['p_miss'],
            'Other_Cups_Hit_Distribution': other_cups_dist
        })
    
    # Single cup configuration
    rack_single = Rack([Cup(cx=0.0, cy=0.0)])
    config_single = [0]
    
    prob_data = calculate_detailed_hit_probability(
        profile, rack_single, 0, shooter_angle_deg=0, N=N
    )
    
    other_cups_dist = format_other_cups_distribution(
        prob_data['other_cups_hit'], N, config_single
    )
    
    results.append({
        'Layout': 'final1',
        'Cups_Remaining': str(config_single),
        'Aimed_Cup': 0,
        'P_Hit_Target': prob_data['p_hit_target'],
        'P_Hit_Any': prob_data['p_hit_any'],
        'P_Miss': prob_data['p_miss'],
        'Other_Cups_Hit_Distribution': other_cups_dist
    })
    
    return results

# Strategy functions
def triangle_front_to_6pyr_front_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle front (10->7) -> 6-pyramid front (6->3) -> final"""
    print("Calculating triangle_front_to_6pyr_front strategy...")
    
    results = []
    
    # Triangle front phase (10 cups down to 7)
    triangle_results = calculate_layout_probabilities(
        profile, triangle_10_rack, "triangle_front", 0, min_cups=7, max_cups=10, N=N
    )
    results.extend(triangle_results)
    
    # 6-pyramid front phase (6 cups down to 3)
    pyramid_results = calculate_layout_probabilities(
        profile, six_pyramid_rack, "6pyramid_front", 0, min_cups=3, max_cups=6, N=N
    )
    results.extend(pyramid_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_front_to_6pyr_side_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle front (10->7) -> 6-pyramid side (6->3) -> final"""
    print("Calculating triangle_front_to_6pyr_side strategy...")
    
    results = []
    
    # Triangle front phase (10 cups down to 7)
    triangle_results = calculate_layout_probabilities(
        profile, triangle_10_rack, "triangle_front", 0, min_cups=7, max_cups=10, N=N
    )
    results.extend(triangle_results)
    
    # 6-pyramid side phase (6 cups down to 3)
    pyramid_results = calculate_layout_probabilities(
        profile, six_pyramid_rack, "6pyramid_side", 60, min_cups=3, max_cups=6, N=N
    )
    results.extend(pyramid_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_front_to_zipper_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle front (10->7) -> zipper (6->3) -> final"""
    print("Calculating triangle_front_to_zipper strategy...")
    
    results = []
    
    # Triangle front phase (10 cups down to 7)
    triangle_results = calculate_layout_probabilities(
        profile, triangle_10_rack, "triangle_front", 0, min_cups=7, max_cups=10, N=N
    )
    results.extend(triangle_results)
    
    # Zipper phase (6 cups down to 3)
    zipper_results = calculate_layout_probabilities(
        profile, zipper, "zipper", 0, min_cups=3, max_cups=6, N=N
    )
    results.extend(zipper_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_front_to_olympics_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle front (10->6) -> olympics (5->3) -> final"""
    print("Calculating triangle_front_to_olympics strategy...")
    
    results = []
    
    # Triangle front phase (10 cups down to 6)
    triangle_results = calculate_layout_probabilities(
        profile, triangle_10_rack, "triangle_front", 0, min_cups=6, max_cups=10, N=N
    )
    results.extend(triangle_results)
    
    # Olympics phase (5 cups down to 3)
    olympics_results = calculate_layout_probabilities(
        profile, sideways_olympics, "olympics", 0, min_cups=3, max_cups=5, N=N
    )
    results.extend(olympics_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_side_to_6pyr_front_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle side (10->7) -> 6-pyramid front (6->3) -> final"""
    print("Calculating triangle_side_to_6pyr_front strategy...")
    
    results = []
    
    # Triangle side phase (10 cups down to 7)
    triangle_results = calculate_layout_probabilities(
        profile, triangle_10_rack, "triangle_side", 60, min_cups=7, max_cups=10, N=N
    )
    results.extend(triangle_results)
    
    # 6-pyramid front phase (6 cups down to 3)
    pyramid_results = calculate_layout_probabilities(
        profile, six_pyramid_rack, "6pyramid_front", 0, min_cups=3, max_cups=6, N=N
    )
    results.extend(pyramid_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_side_to_6pyr_side_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle side (10->7) -> 6-pyramid side (6->3) -> final"""
    print("Calculating triangle_side_to_6pyr_side strategy...")
    
    results = []
    
    # Triangle side phase (10 cups down to 7)
    triangle_results = calculate_layout_probabilities(
        profile, triangle_10_rack, "triangle_side", 60, min_cups=7, max_cups=10, N=N
    )
    results.extend(triangle_results)
    
    # 6-pyramid side phase (6 cups down to 3)
    pyramid_results = calculate_layout_probabilities(
        profile, six_pyramid_rack, "6pyramid_side", 60, min_cups=3, max_cups=6, N=N
    )
    results.extend(pyramid_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_side_to_zipper_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle side (10->7) -> zipper (6->3) -> final"""
    print("Calculating triangle_side_to_zipper strategy...")
    
    results = []
    
    # Triangle side phase (10 cups down to 7)
    triangle_results = calculate_layout_probabilities(
        profile, triangle_10_rack, "triangle_side", 60, min_cups=7, max_cups=10, N=N
    )
    results.extend(triangle_results)
    
    # Zipper phase (6 cups down to 3)
    zipper_results = calculate_layout_probabilities(
        profile, zipper, "zipper", 0, min_cups=3, max_cups=6, N=N
    )
    results.extend(zipper_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_side_to_olympics_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle side (10->6) -> olympics (5->3) -> final"""
    print("Calculating triangle_side_to_olympics strategy...")
    
    results = []
    
    # Triangle side phase (10 cups down to 6)
    triangle_results = calculate_layout_probabilities(
        profile, triangle_10_rack, "triangle_side", 60, min_cups=6, max_cups=10, N=N
    )
    results.extend(triangle_results)
    
    # Olympics phase (5 cups down to 3)
    olympics_results = calculate_layout_probabilities(
        profile, sideways_olympics, "olympics", 0, min_cups=3, max_cups=5, N=N
    )
    results.extend(olympics_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_hybrid_to_6pyr_hybrid_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle hybrid (front+side 10->7, then front+side 7->6) -> 6-pyramid hybrid (front+side 6->3) -> final"""
    print("Calculating triangle_hybrid_to_6pyr_hybrid strategy...")
    
    results = []
    
    # Get configurations for triangle phases
    configs_10_to_7 = get_all_configurations(min_cups=7, max_cups=10)
    configs_7_to_6 = get_all_configurations(min_cups=6, max_cups=7)
    configs_6_to_3 = get_all_configurations(min_cups=3, max_cups=6)
    
    # Triangle front phase (10 cups down to 7) - same configs as triangle side
    for config in configs_10_to_7:
        rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
        
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 0, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_front',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    # Triangle side phase (10 cups down to 7) - same configs as triangle front
    for config in configs_10_to_7:
        rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
        
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 60, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_side',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    # Triangle side phase (7 cups down to 6) - same configs for both angles
    for config in configs_7_to_6:
        # Front angle
        rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
        
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 0, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_front',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
        
        # Side angle - same config
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 60, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_side',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    # 6-pyramid phases (6 cups down to 3) - same configs for both angles
    for config in configs_6_to_3:
        # Front angle
        rack = create_rack_from_config(six_pyramid_rack, config, origin=(0.0, 0.0))
        
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 0, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': '6pyramid_front',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
        
        # Side angle - same config
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 60, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': '6pyramid_side',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_hybrid_to_zipper_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle hybrid (front+side 10->7, then front+side 7->6) -> zipper (6->3) -> final"""
    print("Calculating triangle_hybrid_to_zipper strategy...")
    
    results = []
    
    # Get configurations for triangle phases
    configs_10_to_7 = get_all_configurations(min_cups=7, max_cups=10)
    configs_7_to_6 = get_all_configurations(min_cups=6, max_cups=7)
    
    # Triangle phases (10 cups down to 7) - same configs for both angles
    for config in configs_10_to_7:
        rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
        
        # Front angle
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 0, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_front',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
        
        # Side angle - same config
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 60, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_side',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    # Triangle phases (7 cups down to 6) - same configs for both angles
    for config in configs_7_to_6:
        rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
        
        # Front angle
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 0, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_front',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
        
        # Side angle - same config
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 60, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_side',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    # Zipper phase (6 cups down to 3)
    zipper_results = calculate_layout_probabilities(
        profile, zipper, "zipper", 0, min_cups=3, max_cups=6, N=N
    )
    results.extend(zipper_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def triangle_hybrid_to_olympics_strategy(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """Triangle hybrid (front+side 10->6, then front+side 6->5) -> olympics (5->3) -> final"""
    print("Calculating triangle_hybrid_to_olympics strategy...")
    
    results = []
    
    # Get configurations for triangle phases
    configs_10_to_6 = get_all_configurations(min_cups=6, max_cups=10)
    configs_6_to_5 = get_all_configurations(min_cups=5, max_cups=6)
    
    # Triangle phases (10 cups down to 6) - same configs for both angles
    for config in configs_10_to_6:
        rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
        
        # Front angle
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 0, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_front',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
        
        # Side angle - same config
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 60, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_side',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    # Triangle phases (6 cups down to 5) - same configs for both angles
    for config in configs_6_to_5:
        rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
        
        # Front angle
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 0, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_front',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
        
        # Side angle - same config
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, 60, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Layout': 'triangle_side',
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    # Olympics phase (5 cups down to 3)
    olympics_results = calculate_layout_probabilities(
        profile, sideways_olympics, "olympics", 0, min_cups=3, max_cups=5, N=N
    )
    results.extend(olympics_results)
    
    # Final phase
    final_results = get_final_two_probabilities(profile, N=N)
    results.extend(final_results)
    
    return pd.DataFrame(results)

def generate_all_strategy_csvs(profile: ShooterProfile, output_dir: str = "strategy_results", 
                              N: int = 1000):
    """
    Generate all 11 strategy CSV files for the given profile
    
    Args:
        profile: ShooterProfile to use for calculations
        output_dir: Directory to save CSV files
        N: Number of shots to simulate per scenario
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    profile_name = profile.name.lower()
    
    print(f"\nGenerating strategy CSVs for {profile.name} profile with {N} shots per scenario")
    
    strategies = [
        ("triangle_front_to_6pyr_front", triangle_front_to_6pyr_front_strategy),
        ("triangle_front_to_6pyr_side", triangle_front_to_6pyr_side_strategy),
        ("triangle_front_to_zipper", triangle_front_to_zipper_strategy),
        ("triangle_front_to_olympics", triangle_front_to_olympics_strategy),
        ("triangle_side_to_6pyr_front", triangle_side_to_6pyr_front_strategy),
        ("triangle_side_to_6pyr_side", triangle_side_to_6pyr_side_strategy),
        ("triangle_side_to_zipper", triangle_side_to_zipper_strategy),
        ("triangle_side_to_olympics", triangle_side_to_olympics_strategy),
        ("triangle_hybrid_to_6pyr_hybrid", triangle_hybrid_to_6pyr_hybrid_strategy),
        ("triangle_hybrid_to_zipper", triangle_hybrid_to_zipper_strategy),
        ("triangle_hybrid_to_olympics", triangle_hybrid_to_olympics_strategy),
    ]
    
    for strategy_name, strategy_func in strategies:
        print(f"\n{'='*50}")
        print(f"Processing {strategy_name}")
        print(f"{'='*50}")
        
        df_strategy = strategy_func(profile, N=N)
        filename = f"{output_dir}/{profile_name}_{strategy_name}.csv"
        df_strategy.to_csv(filename, index=False)
        print(f"Saved: {filename}")
    
    print(f"\nAll strategy calculations complete!")
    print(f"Results saved to {output_dir}/ directory with 11 CSV files for {profile.name}")

# Example usage functions
def generate_okay_strategies(N: int = 1000):
    """Generate all strategy CSVs for the 'okay' profile"""
    generate_all_strategy_csvs(okay, N=N)

def generate_good_strategies(N: int = 1000):
    """Generate all strategy CSVs for the 'good' profile"""
    generate_all_strategy_csvs(good, N=N)

def generate_bad_strategies(N: int = 1000):
    """Generate all strategy CSVs for the 'bad' profile"""
    generate_all_strategy_csvs(bad, N=N)

def generate_all_profiles_strategies(N: int = 1000):
    """Generate strategy CSVs for all three profiles"""
    print("=== Generating Strategy CSVs for All Profiles ===\n")
    
    profiles = [okay, good, bad]
    
    for profile in profiles:
        print(f"\n{'='*60}")
        print(f"Processing {profile.name.upper()} profile")
        print(f"{'='*60}")
        generate_all_strategy_csvs(profile, N=N)

# Test function
def test_single_strategy():
    """Test a single strategy calculation"""
    print("Testing triangle_front_to_olympics strategy...")
    df = triangle_front_to_olympics_strategy(okay, N=1000)
    
    print(f"Generated {len(df)} rows")
    print("\nFirst few rows:")
    print(df.head(10))
    
    print(f"\nUnique layouts: {df['Layout'].unique()}")
    print(f"Layout value counts:")
    print(df['Layout'].value_counts())

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Test a single strategy first
    # test_single_strategy()
    
    # Generate for specific profile
    generate_good_strategies(N=1000)
    
    # Or generate for all profiles
    # generate_all_profiles_strategies(N=1000)