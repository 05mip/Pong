import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import ast
import os
from itertools import combinations

def parse_cups_remaining(cups_str: str) -> List[int]:
    """Parse the cups_remaining string into a list of integers"""
    try:
        return ast.literal_eval(cups_str)
    except:
        return []

def parse_other_cups_distribution(dist_str: str) -> Dict[int, float]:
    """Parse other_cups_hit_distribution string into dictionary"""
    if pd.isna(dist_str) or dist_str == "":
        return {}
    
    result = {}
    pairs = dist_str.split(',')
    for pair in pairs:
        if ':' in pair:
            cup_str, prob_str = pair.split(':')
            result[int(cup_str)] = float(prob_str)
    return result

def get_rerack_mapping(num_cups: int) -> List[int]:
    """Get the standard cup indices after reracking for a given number of cups"""
    if num_cups == 10:
        return list(range(10))  # [0,1,2,3,4,5,6,7,8,9]
    elif num_cups == 6:
        return list(range(6))   # [0,1,2,3,4,5] for 6-pyramid or zipper
    elif num_cups == 5:
        return list(range(5))   # [0,1,2,3,4] for olympics
    elif num_cups == 2:
        return [0, 1]           # [0,1] for final2
    elif num_cups == 1:
        return [0]              # [0] for final1
    else:
        return list(range(num_cups))  # Default for other cases

def should_rerack(current_cups: List[int], layout: str, strategy_name: str = "") -> bool:
    """Determine if a rerack should happen based on current cups, layout, and strategy"""
    num_cups = len(current_cups)
    
    # Strategy-specific rerack points
    if num_cups == 6:
        # Triangle -> next phase transition
        if "triangle" in layout:
            return True
        else:
            return False  # Already in target layout, no need to rerack
            
    elif num_cups == 5:
        # Only rerack to olympics for olympics strategies
        olympics_strategies = [
            "triangle_front_to_olympics",
            "triangle_side_to_olympics", 
            "triangle_hybrid_to_olympics"
        ]
        if strategy_name in olympics_strategies and "triangle" in layout:
            return True
        else:
            return False  # Stay in current layout
            
    elif num_cups == 2:  # Any layout -> final2
        return True
        
    elif num_cups == 1:  # final2 -> final1
        return True
    
    return False

def get_next_state_after_hit(current_state: List[int], hit_cup: int, current_layout: str, strategy_name: str = "") -> Tuple[List[int], str]:
    """Get the next state after hitting a specific cup, handling reracks"""
    if hit_cup in current_state:
        # Remove the hit cup
        next_cups = [cup for cup in current_state if cup != hit_cup]
        next_layout = current_layout
        
        # Check if we need to rerack
        if should_rerack(next_cups, current_layout, strategy_name):
            # Rerack to standard configuration
            reracked_cups = get_rerack_mapping(len(next_cups))
            
            # Determine new layout based on strategy and cup count
            if len(next_cups) == 6:
                # Triangle -> target layout transition
                if strategy_name == "triangle_front_to_6pyr_front":
                    next_layout = "6pyramid_front"
                elif strategy_name == "triangle_front_to_6pyr_side":
                    next_layout = "6pyramid_side"
                elif strategy_name == "triangle_front_to_zipper":
                    next_layout = "zipper"
                elif strategy_name == "triangle_side_to_6pyr_front":
                    next_layout = "6pyramid_front"
                elif strategy_name == "triangle_side_to_6pyr_side":
                    next_layout = "6pyramid_side"
                elif strategy_name == "triangle_side_to_zipper":
                    next_layout = "zipper"
                elif strategy_name == "triangle_hybrid_to_6pyr_hybrid":
                    next_layout = "6pyramid_front"  # Default to front for hybrid
                elif strategy_name == "triangle_hybrid_to_zipper":
                    next_layout = "zipper"
                # Olympics strategies transition at 5 cups, not 6
                else:
                    next_layout = current_layout  # Fallback
                    
            elif len(next_cups) == 5:
                # Only olympics strategies transition to olympics at 5 cups
                if strategy_name in ["triangle_front_to_olympics", "triangle_side_to_olympics", "triangle_hybrid_to_olympics"]:
                    next_layout = "olympics"
                else:
                    next_layout = current_layout  # Stay in current layout
                    
            elif len(next_cups) == 2:
                next_layout = "final2"
            elif len(next_cups) == 1:
                next_layout = "final1"
            
            return reracked_cups, next_layout
        else:
            return next_cups, next_layout
    else:
        # This shouldn't happen in our data, but handle gracefully
        return current_state, current_layout

class ExpectedShotsCalculator:
    def __init__(self, strategy_csv_path: str, strategy_name: str):
        """Initialize with a strategy CSV file and strategy name for rerack logic"""
        self.df = pd.read_csv(strategy_csv_path)
        self.strategy_name = strategy_name
        self.states = set()  # All possible states (cup configurations with layout)
        self.transitions = defaultdict(dict)  # transitions[state][aimed_cup] = {next_state: prob}
        self.expected_values = {}  # expected_values[state] = expected shots
        
        # Define strategy-specific rerack mappings
        self._setup_strategy_reracks()
        self._process_data()
    
    def _setup_strategy_reracks(self):
        """Setup strategy-specific rerack target layouts"""
        self.rerack_targets = {
            # Triangle front strategies
            "triangle_front_to_6pyr_front": {"6": "6pyramid_front"},
            "triangle_front_to_6pyr_side": {"6": "6pyramid_side"},
            "triangle_front_to_zipper": {"6": "zipper"},
            "triangle_front_to_olympics": {"5": "olympics"},  # Only this one gets olympics at 5 cups
            
            # Triangle side strategies  
            "triangle_side_to_6pyr_front": {"6": "6pyramid_front"},
            "triangle_side_to_6pyr_side": {"6": "6pyramid_side"},
            "triangle_side_to_zipper": {"6": "zipper"},
            "triangle_side_to_olympics": {"5": "olympics"},  # Only this one gets olympics at 5 cups
            
            # Triangle hybrid strategies
            "triangle_hybrid_to_6pyr_hybrid": {"6": ["6pyramid_front", "6pyramid_side"]},
            "triangle_hybrid_to_zipper": {"6": "zipper"},
            "triangle_hybrid_to_olympics": {"5": "olympics"},  # Only this one gets olympics at 5 cups
        }
    
    def _resolve_rerack_layout(self, num_cups: int, current_layout: str) -> str:
        """Resolve what layout to use after reracking"""
        if num_cups == 2:
            return "final2"
        elif num_cups == 1:
            return "final1"
        elif num_cups == 6:
            # Check strategy-specific mapping for 6-cup transitions
            if self.strategy_name in self.rerack_targets:
                target = self.rerack_targets[self.strategy_name].get("6")
                if isinstance(target, list):
                    # For hybrid strategies, we'll handle both layouts
                    return target[0]  # Default to first option for state representation
                elif target:
                    return target
            return "6pyramid_front"  # Default fallback
        elif num_cups == 5:
            # Only transition to olympics if the strategy explicitly calls for it
            if self.strategy_name in self.rerack_targets:
                target = self.rerack_targets[self.strategy_name].get("5")
                if target == "olympics":
                    return "olympics"
            
            # Otherwise, stay in the current layout family
            if "6pyramid" in current_layout:
                return current_layout  # Stay in 6pyramid_front or 6pyramid_side
            elif current_layout == "zipper":
                return "zipper"
            else:
                return current_layout  # Default to staying in same layout
        else:
            return current_layout
    
    def _process_data(self):
        """Process the CSV data and build transition model with rerack handling"""
        print("Processing strategy data with rerack handling...")
        
        # Handle hybrid strategies by selecting better angle for each (config, aimed_cup) pair
        processed_rows = self._resolve_hybrid_duplicates()
        
        # First pass: Build state space from CSV data
        csv_states = set()
        for _, row in processed_rows.iterrows():
            current_cups = parse_cups_remaining(row['Cups_Remaining'])
            current_layout = row['Layout']
            current_state = (tuple(sorted(current_cups)), current_layout)
            csv_states.add(current_state)
        
        # Second pass: Build transitions and collect all possible next states
        all_next_states = set()
        
        for _, row in processed_rows.iterrows():
            current_cups = parse_cups_remaining(row['Cups_Remaining'])
            current_layout = row['Layout']
            aimed_cup = row['Aimed_Cup']
            
            # State includes both cup configuration and layout
            current_state = (tuple(sorted(current_cups)), current_layout)
            self.states.add(current_state)
            
            # Get transition probabilities
            p_hit_target = row['P_Hit_Target']
            p_miss = row['P_Miss']
            other_hits = parse_other_cups_distribution(row['Other_Cups_Hit_Distribution'])
            
            transitions = {}
            
            # Transition from hitting target cup
            if p_hit_target > 0:
                next_cups, next_layout = get_next_state_after_hit(current_cups, aimed_cup, current_layout, self.strategy_name)
                
                next_state = (tuple(sorted(next_cups)), next_layout)
                transitions[next_state] = transitions.get(next_state, 0) + p_hit_target
                all_next_states.add(next_state)
            
            # Transitions from hitting other cups
            for other_cup, prob in other_hits.items():
                if prob > 0:
                    next_cups, next_layout = get_next_state_after_hit(current_cups, other_cup, current_layout, self.strategy_name)
                    
                    next_state = (tuple(sorted(next_cups)), next_layout)
                    transitions[next_state] = transitions.get(next_state, 0) + prob
                    all_next_states.add(next_state)
            
            # Transition from missing (stay in same state)
            if p_miss > 0:
                transitions[current_state] = transitions.get(current_state, 0) + p_miss
            
            self.transitions[current_state][aimed_cup] = transitions
        
        # Add all next states to state space (they may not have been in CSV if they're reracked states)
        for next_state in all_next_states:
            self.states.add(next_state)
        
        # Add terminal state
        self.states.add(((), "terminal"))  # Empty tuple represents game won
        
        # Validate that all transition targets exist
        missing_states = set()
        for state, actions in self.transitions.items():
            for aimed_cup, state_transitions in actions.items():
                for next_state in state_transitions.keys():
                    if next_state not in self.states:
                        missing_states.add(next_state)
        
        if missing_states:
            print(f"Warning: Found {len(missing_states)} missing states in transitions:")
            for state in missing_states:
                print(f"  {state}")
                self.states.add(state)  # Add them to prevent KeyError
        
        print(f"Found {len(self.states)} unique states")
        print(f"Built transitions for {len(self.transitions)} state-action pairs")
    
    def _resolve_hybrid_duplicates(self) -> pd.DataFrame:
        """For hybrid strategies, select the entry with better P_Hit_Any for each (config, aimed_cup) pair"""
        # Group by configuration and aimed cup
        if "hybrid" in self.strategy_name:
            # For hybrid strategies, we need to be more careful about deduplication
            # because we want to preserve the best strategy for each angle
            
            # First, let's see what the data looks like
            print(f"Processing hybrid strategy: {self.strategy_name}")
            print(f"Original rows: {len(self.df)}")
            
            # Group by configuration, aimed cup, and layout to handle hybrid properly
            grouped = self.df.groupby(['Cups_Remaining', 'Aimed_Cup', 'Layout'])
            
            result_rows = []
            
            for (config, aimed_cup, layout), group in grouped:
                if len(group) == 1:
                    # No duplicates, use as-is
                    result_rows.append(group.iloc[0])
                else:
                    # Multiple entries for same (config, aimed_cup, layout), pick the best
                    best_row = group.loc[group['P_Hit_Any'].idxmax()]
                    result_rows.append(best_row)
            
            result_df = pd.DataFrame(result_rows)
            print(f"After deduplication: {len(result_df)} rows")
            return result_df

        grouped = self.df.groupby(['Cups_Remaining', 'Aimed_Cup'])
        
        result_rows = []
        
        for (config, aimed_cup), group in grouped:
            if len(group) == 1:
                # No duplicates, use as-is
                result_rows.append(group.iloc[0])
            else:
                # Multiple entries (hybrid case), pick the one with higher P_Hit_Any
                best_row = group.loc[group['P_Hit_Any'].idxmax()]
                result_rows.append(best_row)
        
        return pd.DataFrame(result_rows)
    
    def calculate_expected_values(self, max_iterations: int = 1000, tolerance: float = 1e-6) -> float:
        """Calculate expected number of shots using value iteration"""
        print("Calculating expected values using value iteration...")
        
        # Initialize expected values
        for state in self.states:
            cups, layout = state
            if len(cups) == 0:  # Terminal state (empty configuration)
                self.expected_values[state] = 0.0
            else:
                self.expected_values[state] = 100.0  # Large initial value
        
        # Debug: Check if we have any states without transitions
        states_without_transitions = []
        for state in self.states:
            cups, layout = state
            if len(cups) > 0 and state not in self.transitions:
                states_without_transitions.append(state)
        
        if states_without_transitions:
            print(f"Warning: Found {len(states_without_transitions)} states without transitions:")
            for state in states_without_transitions[:5]:  # Show first 5
                print(f"  {state}")
            if len(states_without_transitions) > 5:
                print(f"  ... and {len(states_without_transitions) - 5} more")
        
        # Value iteration
        for iteration in range(max_iterations):
            old_values = self.expected_values.copy()
            max_change = 0.0
            
            for state in self.states:
                cups, layout = state
                if len(cups) == 0:  # Skip terminal state
                    continue
                
                # Find optimal action (aimed cup) for this state
                if state in self.transitions:
                    min_expected = float('inf')
                    
                    for aimed_cup, transitions in self.transitions[state].items():
                        # Calculate expected value for aiming at this cup
                        expected = 1.0  # Cost of this shot
                        for next_state, prob in transitions.items():
                            if next_state in self.expected_values:
                                expected += prob * self.expected_values[next_state]
                            else:
                                print(f"Warning: Missing next_state {next_state} in expected_values")
                                expected += prob * 100.0  # Use large penalty
                        
                        min_expected = min(min_expected, expected)
                    
                    self.expected_values[state] = min_expected
                    max_change = max(max_change, abs(old_values[state] - self.expected_values[state]))
                else:
                    # No transitions available - this state cannot reach terminal
                    self.expected_values[state] = float('inf')
            
            if max_change < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        else:
            print(f"Max iterations ({max_iterations}) reached")
        
        # Debug: Check final values for some key states
        print(f"Final expected values for some key states:")
        terminal_states = [(state, val) for state, val in self.expected_values.items() if state[0] == ()]
        print(f"Terminal states: {terminal_states}")
        
        single_cup_states = [(state, val) for state, val in self.expected_values.items() if len(state[0]) == 1]
        print(f"Single cup states: {single_cup_states[:3]}")
        
        two_cup_states = [(state, val) for state, val in self.expected_values.items() if len(state[0]) == 2]
        print(f"Two cup states: {two_cup_states[:3]}")
        
        # Return expected value from initial state (10 cups in triangle layout)
        # Try to find the correct initial state based on the strategy
        possible_initial_states = [
            (tuple(range(10)), "triangle_front"),
            (tuple(range(10)), "triangle_side"),
        ]
        
        initial_state = None
        for candidate_state in possible_initial_states:
            if candidate_state in self.expected_values:
                initial_state = candidate_state
                break
        
        if initial_state is None:
            print("ERROR: Could not find initial state in expected values!")
            print("Available states with 10 cups:")
            for state in self.states:
                cups, layout = state
                if len(cups) == 10:
                    print(f"  {state}: {self.expected_values.get(state, 'NOT_FOUND')}")
            return float('inf')
        
        print(f"Using initial state: {initial_state}")
        print(f"Initial state expected value: {self.expected_values[initial_state]}")
        return self.expected_values[initial_state]
    
    def get_state_breakdown(self) -> Dict[str, Dict[int, int]]:
        """Get breakdown of states by layout and number of cups remaining"""
        breakdown = defaultdict(lambda: defaultdict(int))
        for state in self.states:
            cups, layout = state
            breakdown[layout][len(cups)] += 1
        return dict(breakdown)

def calculate_all_strategy_expected_values(strategy_dir: str = "strategy_results", 
                                         profile: str = "okay") -> Dict[str, float]:
    """Calculate expected values for all 11 strategies for a given profile"""
    
    strategy_names = [
        "triangle_front_to_6pyr_front",
        "triangle_front_to_6pyr_side", 
        "triangle_front_to_zipper",
        "triangle_front_to_olympics",
        "triangle_side_to_6pyr_front",
        "triangle_side_to_6pyr_side",
        "triangle_side_to_zipper", 
        "triangle_side_to_olympics",
        "triangle_hybrid_to_6pyr_hybrid",
        "triangle_hybrid_to_zipper",
        "triangle_hybrid_to_olympics"
    ]
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Calculating Expected Values for {profile.upper()} profile")
    print(f"{'='*60}")
    
    for strategy_name in strategy_names:
        csv_path = os.path.join(strategy_dir, f"{profile}_{strategy_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping...")
            continue
        
        print(f"\n{'-'*40}")
        print(f"Processing: {strategy_name}")
        print(f"File: {csv_path}")
        
        try:
            calculator = ExpectedShotsCalculator(csv_path, strategy_name)
            expected_shots = calculator.calculate_expected_values()
            results[strategy_name] = expected_shots
            
            # Print some diagnostics
            state_breakdown = calculator.get_state_breakdown()
            print(f"State breakdown by layout:")
            for layout, counts in state_breakdown.items():
                print(f"  {layout}: {dict(sorted(counts.items()))}")
            print(f"Expected shots: {expected_shots:.3f}")
            
        except Exception as e:
            print(f"Error processing {strategy_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[strategy_name] = float('inf')
    
    return results

def print_results_summary(results: Dict[str, float]):
    """Print a nicely formatted summary of results"""
    print(f"\n{'='*60}")
    print("EXPECTED SHOTS SUMMARY")
    print(f"{'='*60}")
    
    # Sort by expected value
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    rank = 1
    for strategy, expected_shots in sorted_results:
        if expected_shots == float('inf'):
            print(f"{rank:2d}. {strategy:<35} ERROR")
        else:
            print(f"{rank:2d}. {strategy:<35} {expected_shots:8.3f} shots")
        rank += 1

def main():
    """Main function to calculate expected values for all strategies"""
    
    # You can change the profile here
    profile = "good"  # "okay", "good", or "bad"
    strategy_dir = "strategy_results"
    
    results = calculate_all_strategy_expected_values(strategy_dir, profile)
    print_results_summary(results)
    
    # Save results to CSV
    results_df = pd.DataFrame([
        {"Strategy": strategy, "Expected_Shots": shots} 
        for strategy, shots in results.items()
    ])
    
    output_path = f"{profile}_strategy_expected_values.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()

# okay
# ============================================================
# EXPECTED SHOTS SUMMARY
# ============================================================
#  1. triangle_side_to_olympics             39.911 shots
#  2. triangle_front_to_olympics            40.105 shots
#  3. triangle_hybrid_to_olympics           40.105 shots
#  4. triangle_side_to_zipper               40.358 shots
#  5. triangle_front_to_zipper              40.579 shots
#  6. triangle_hybrid_to_zipper             40.579 shots
#  7. triangle_side_to_6pyr_side            41.593 shots
#  8. triangle_front_to_6pyr_side           41.814 shots
#  9. triangle_side_to_6pyr_front           41.876 shots
# 10. triangle_front_to_6pyr_front          42.096 shots
# 11. triangle_hybrid_to_6pyr_hybrid        42.096 shots

# good
# ============================================================
# EXPECTED SHOTS SUMMARY
# ============================================================
#  1. triangle_side_to_olympics             20.293 shots
#  2. triangle_side_to_zipper               20.661 shots
#  3. triangle_front_to_olympics            20.830 shots
#  4. triangle_hybrid_to_olympics           20.830 shots
#  5. triangle_hybrid_to_zipper             21.033 shots
#  6. triangle_front_to_zipper              21.033 shots
#  7. triangle_side_to_6pyr_side            21.292 shots
#  8. triangle_side_to_6pyr_front           21.479 shots
#  9. triangle_front_to_6pyr_side           21.665 shots
# 10. triangle_hybrid_to_6pyr_hybrid        21.851 shots
# 11. triangle_front_to_6pyr_front          21.851 shots