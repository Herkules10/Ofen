#!/usr/bin/env python3
"""
Test script to debug the intermediate values issue in parametric analysis.
"""

import matplotlib.pyplot as plt
import numpy as np

def test_intermediate_values():
    """Test the intermediate values logic"""
    
    # Simulate tournament size data
    param_values = [3, 5, 7]  # Original parameter values from data
    fitness_values = [0.85, 0.92, 0.88]  # Corresponding fitness values
    
    print(f"Original data points: {list(zip(param_values, fitness_values))}")
    
    # Create the intermediate values logic
    all_ticks = []
    for idx in range(len(param_values)):
        all_ticks.append(param_values[idx])
        # Add intermediate value if there's a next value and gap is reasonable
        if idx < len(param_values) - 1:
            current_val = param_values[idx]
            next_val = param_values[idx + 1]
            
            # Only add intermediate values for reasonable gaps
            if isinstance(current_val, (int, float)) and isinstance(next_val, (int, float)):
                gap = next_val - current_val
                
                # For integer values (like tournament size), add missing integers
                if isinstance(current_val, int) and isinstance(next_val, int) and gap <= 5:
                    for j in range(int(current_val) + 1, int(next_val)):
                        all_ticks.append(j)
    
    all_ticks = sorted(set(all_ticks))
    print(f"X-axis ticks will show: {all_ticks}")
    print(f"But we only have data for: {param_values}")
    
    # Create a test plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot only the actual data points
    ax.plot(param_values, fitness_values, 
           marker='o', linestyle='-', linewidth=2, markersize=8,
           color='#FF6B6B', markerfacecolor='white', 
           markeredgecolor='#FF6B6B', markeredgewidth=2)
    
    # Set x-axis ticks to include intermediate values
    ax.set_xticks(all_ticks)
    ax.set_xlabel('Tournament Size')
    ax.set_ylabel('Average Fitness')
    ax.set_title('Test: Tournament Size vs Fitness')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_intermediate_values.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Test plot saved as 'test_intermediate_values.png'")
    print("This should show data points at 3, 5, 7 but x-axis ticks at 3, 4, 5, 6, 7")

if __name__ == "__main__":
    test_intermediate_values()
