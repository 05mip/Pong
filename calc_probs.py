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

def calculate_triangle_probabilities(profile: ShooterProfile, angle_name: str, 
                                   shooter_angle_deg: float, min_cups: int = 6, 
                                   N: int = 1000) -> pd.DataFrame:
    """
    Calculate detailed hit probabilities for triangle rack configurations
    """
    print(f"Calculating triangle {angle_name} probabilities...")
    
    configurations = get_all_configurations(min_cups=min_cups, max_cups=10)
    results = []
    
    for i, config in enumerate(configurations):
        print(f"Processing configuration {i+1}/{len(configurations)}: {config}")
        
        rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
        
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]  # The original cup index in the 10-cup rack
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, shooter_angle_deg, N=N
            )
            
            # Format other cups distribution
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    return pd.DataFrame(results)

def calculate_six_pyramid_probabilities(profile: ShooterProfile, angle_name: str, 
                                      shooter_angle_deg: float, min_cups: int = 3,
                                      N: int = 1000) -> pd.DataFrame:
    """
    Calculate detailed hit probabilities for 6-cup pyramid configurations
    """
    print(f"Calculating 6-pyramid {angle_name} probabilities...")
    
    configurations = get_all_configurations(min_cups=min_cups, max_cups=6)
    results = []
    
    for i, config in enumerate(configurations):
        print(f"Processing configuration {i+1}/{len(configurations)}: {config}")
        
        rack = create_rack_from_config(six_pyramid_rack, config, origin=(0.0, 0.0))
        
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, shooter_angle_deg, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    return pd.DataFrame(results)

def calculate_special_formation_probabilities(profile: ShooterProfile, formation_func, 
                                            formation_name: str, min_cups: int = 3,
                                            N: int = 1000) -> pd.DataFrame:
    """
    Calculate detailed hit probabilities for special formations (olympics, zipper)
    """
    print(f"Calculating {formation_name} probabilities...")
    
    base_rack = formation_func(center=(0.0, 0.0))
    max_cups = len(base_rack.cups)
    configurations = get_all_configurations(min_cups=min_cups, max_cups=max_cups)
    results = []
    
    for i, config in enumerate(configurations):
        print(f"Processing configuration {i+1}/{len(configurations)}: {config}")
        
        rack = create_rack_from_config(formation_func, config, center=(0.0, 0.0))
        
        for aim_idx in range(len(config)):
            original_cup_idx = config[aim_idx]
            
            prob_data = calculate_detailed_hit_probability(
                profile, rack, aim_idx, shooter_angle_deg=0, N=N
            )
            
            other_cups_dist = format_other_cups_distribution(
                prob_data['other_cups_hit'], N, config
            )
            
            results.append({
                'Cups_Remaining': str(config),
                'Aimed_Cup': original_cup_idx,
                'P_Hit_Target': prob_data['p_hit_target'],
                'P_Hit_Any': prob_data['p_hit_any'],
                'P_Miss': prob_data['p_miss'],
                'Other_Cups_Hit_Distribution': other_cups_dist
            })
    
    return pd.DataFrame(results)

def calculate_final_two_probabilities(profile: ShooterProfile, N: int = 1000) -> pd.DataFrame:
    """
    Calculate detailed hit probabilities for final 2 cups and single cup scenarios
    """
    print("Calculating final 2 cups probabilities...")
    
    results = []
    
    # 2-cup vertical configuration
    rack_2_vertical = vertical_stack(2, center=(0.0, 0.0))
    config_2_cups = [0, 1]  # Two cups in vertical stack
    
    for aim_idx in range(2):
        original_cup_idx = config_2_cups[aim_idx]
        
        prob_data = calculate_detailed_hit_probability(
            profile, rack_2_vertical, aim_idx, shooter_angle_deg=0, N=N
        )
        
        other_cups_dist = format_other_cups_distribution(
            prob_data['other_cups_hit'], N, config_2_cups
        )
        
        results.append({
            'Cups_Remaining': str(config_2_cups),
            'Aimed_Cup': original_cup_idx,
            'P_Hit_Target': prob_data['p_hit_target'],
            'P_Hit_Any': prob_data['p_hit_any'],
            'P_Miss': prob_data['p_miss'],
            'Other_Cups_Hit_Distribution': other_cups_dist
        })
    
    # Single cup configuration
    rack_single = Rack([Cup(cx=0.0, cy=0.0)])
    config_single = [0]  # Single cup
    
    prob_data = calculate_detailed_hit_probability(
        profile, rack_single, 0, shooter_angle_deg=0, N=N
    )
    
    # For single cup, other_cups_hit should be empty
    other_cups_dist = format_other_cups_distribution(
        prob_data['other_cups_hit'], N, config_single
    )
    
    results.append({
        'Cups_Remaining': str(config_single),
        'Aimed_Cup': 0,
        'P_Hit_Target': prob_data['p_hit_target'],
        'P_Hit_Any': prob_data['p_hit_any'],
        'P_Miss': prob_data['p_miss'],
        'Other_Cups_Hit_Distribution': other_cups_dist
    })
    
    return pd.DataFrame(results)

def generate_all_probability_csvs(profile: ShooterProfile, output_dir: str = "probability_results", 
                                N: int = 1000):
    """
    Generate all 7 CSV files for the given profile with detailed probability breakdowns
    
    Args:
        profile: ShooterProfile to use for calculations
        output_dir: Directory to save CSV files
        N: Number of shots to simulate per scenario
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    profile_name = profile.name.lower()
    
    print(f"\nGenerating probability CSVs for {profile.name} profile with {N} shots per scenario")
    
    # 1. Triangle rack from front (10 cups down to 6)
    df_triangle_front = calculate_triangle_probabilities(profile, "front", 0, min_cups=6, N=N)
    df_triangle_front.to_csv(f"{output_dir}/{profile_name}_triangle_front.csv", index=False)
    
    # 2. Triangle rack from side (10 cups down to 6)
    df_triangle_side = calculate_triangle_probabilities(profile, "side", 60, min_cups=6, N=N)
    df_triangle_side.to_csv(f"{output_dir}/{profile_name}_triangle_side.csv", index=False)
    
    # 3. 6-pyramid from front (6 cups down to 3)
    df_pyramid_front = calculate_six_pyramid_probabilities(profile, "front", 0, min_cups=3, N=N)
    df_pyramid_front.to_csv(f"{output_dir}/{profile_name}_6pyramid_front.csv", index=False)
    
    # 4. 6-pyramid from side (6 cups down to 3)
    df_pyramid_side = calculate_six_pyramid_probabilities(profile, "side", 60, min_cups=3, N=N)
    df_pyramid_side.to_csv(f"{output_dir}/{profile_name}_6pyramid_side.csv", index=False)
    
    # 5. Olympics formation (down to 3)
    df_olympics = calculate_special_formation_probabilities(profile, sideways_olympics, "olympics", min_cups=3, N=N)
    df_olympics.to_csv(f"{output_dir}/{profile_name}_olympics.csv", index=False)
    
    # 6. Zipper formation (down to 3)
    df_zipper = calculate_special_formation_probabilities(profile, zipper, "zipper", min_cups=3, N=N)
    df_zipper.to_csv(f"{output_dir}/{profile_name}_zipper.csv", index=False)
    
    # 7. Final 2 cups (vertical stack and single cup)
    df_final = calculate_final_two_probabilities(profile, N=N)
    df_final.to_csv(f"{output_dir}/{profile_name}_final2.csv", index=False)
    
    print(f"\nAll probability calculations complete!")
    print(f"Results saved to {output_dir}/ directory:")
    print(f"  - {profile_name}_triangle_front.csv")
    print(f"  - {profile_name}_triangle_side.csv")
    print(f"  - {profile_name}_6pyramid_front.csv")
    print(f"  - {profile_name}_6pyramid_side.csv")
    print(f"  - {profile_name}_olympics.csv")
    print(f"  - {profile_name}_zipper.csv")
    print(f"  - {profile_name}_final2.csv")

# Example usage functions with configurable shot count
def generate_okay_probabilities(N: int = 1000):
    """Generate all probability CSVs for the 'okay' profile"""
    generate_all_probability_csvs(okay, N=N)

def generate_good_probabilities(N: int = 1000):
    """Generate all probability CSVs for the 'good' profile"""
    generate_all_probability_csvs(good, N=N)

def generate_bad_probabilities(N: int = 1000):
    """Generate all probability CSVs for the 'bad' profile"""
    generate_all_probability_csvs(bad, N=N)

def generate_all_profiles(N: int = 1000):
    """Generate probability CSVs for all three profiles"""
    print("=== Generating Detailed Probability CSVs for All Profiles ===\n")
    
    profiles = [okay, good, bad]
    
    for profile in profiles:
        print(f"\n{'='*50}")
        print(f"Processing {profile.name.upper()} profile")
        print(f"{'='*50}")
        generate_all_probability_csvs(profile, N=N)

# Quick test function for a single configuration
def test_single_configuration():
    """Test the detailed probability calculation on a single configuration"""
    print("Testing single configuration with detailed breakdown...")
    
    # Test with a 6-cup triangle configuration missing cups 1 and 3
    config = [0, 2, 4, 5, 6, 7, 8, 9]  # 8 cups remaining
    rack = create_rack_from_config(triangle_10_rack, config, origin=(0.0, 0.0))
    
    print(f"Configuration: {config}")
    print(f"Number of cups in rack: {len(rack.cups)}")
    print()
    
    for aim_idx in range(min(3, len(config))):  # Test first 3 cups
        prob_data = calculate_detailed_hit_probability(okay, rack, aim_idx, 
                                                     shooter_angle_deg=0, N=1000)
        original_cup_idx = config[aim_idx]
        
        other_cups_dist = format_other_cups_distribution(
            prob_data['other_cups_hit'], 1000, config
        )
        
        print(f"Aiming at original cup {original_cup_idx} (rack position {aim_idx}):")
        print(f"  P(hit target): {prob_data['p_hit_target']:.3f}")
        print(f"  P(hit any): {prob_data['p_hit_any']:.3f}")
        print(f"  P(miss): {prob_data['p_miss']:.3f}")
        print(f"  Other cups hit: {other_cups_dist}")
        print()

if __name__ == "__main__":
    # Uncomment the function you want to run:
    
    # Test a single configuration first
    # test_single_configuration()
    
    # Generate for specific profile with higher precision
    generate_good_probabilities(N=1000)
    
    # Or generate for all profiles
    # generate_all_profiles(N=5000)