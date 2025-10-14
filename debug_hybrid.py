from expected_values import ExpectedShotsCalculator

# Test the hybrid strategy
calc = ExpectedShotsCalculator('strategy_results/okay_triangle_hybrid_to_6pyr_hybrid.csv', 'triangle_hybrid_to_6pyr_hybrid')

print("Strategy:", calc.strategy_name)

# Check states without transitions
states_without_transitions = [state for state in calc.states if len(state[0]) > 0 and state not in calc.transitions]
print(f"\nStates without transitions: {len(states_without_transitions)}")

print("\nEXACT MISSING STATES:")
for state in states_without_transitions:
    cups, layout = state
    print(f"  {state}")

# Check what's happening with the hybrid duplicates resolution
print(f"\nCHECKING HYBRID DUPLICATES RESOLUTION:")
print(f"Original CSV rows: {len(calc.df)}")
processed_rows = calc._resolve_hybrid_duplicates()
print(f"After resolving duplicates: {len(processed_rows)}")

# Check specific missing states
print(f"\nCHECKING SPECIFIC MISSING STATES:")
for state in states_without_transitions[:3]:
    cups, layout = state
    cups_str = str(list(cups))
    
    print(f"\nState: {state}")
    print(f"Cups string: {cups_str}")
    
    # Check original CSV
    original_matches = calc.df[
        (calc.df['Cups_Remaining'] == cups_str) & 
        (calc.df['Layout'] == layout)
    ]
    print(f"  Original CSV: {len(original_matches)} rows")
    
    # Check processed rows
    processed_matches = processed_rows[
        (processed_rows['Cups_Remaining'] == cups_str) & 
        (processed_rows['Layout'] == layout)
    ]
    print(f"  Processed rows: {len(processed_matches)} rows")
    
    if len(original_matches) > 0:
        print(f"  Original CSV data:")
        for _, row in original_matches.iterrows():
            print(f"    Aimed_Cup: {row['Aimed_Cup']}, P_Hit_Any: {row['P_Hit_Any']}")
    
    if len(processed_matches) > 0:
        print(f"  Processed data:")
        for _, row in processed_matches.iterrows():
            print(f"    Aimed_Cup: {row['Aimed_Cup']}, P_Hit_Any: {row['P_Hit_Any']}")

# The issue might be that _resolve_hybrid_duplicates is filtering out some rows
# that are needed for building transitions
