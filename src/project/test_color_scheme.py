#!/usr/bin/env python3
"""
Test script to visualize the new color scheme for algorithm comparison plots.
"""

import matplotlib.pyplot as plt
import numpy as np

def test_color_scheme():
    """Display the new color scheme with labels."""
    
    # Color mapping
    color_map = {
        'GA': '#FF6B6B',    # Red
        'EGA': '#FF6B6B',   # Red (same as GA)
        'PSO': '#4ECDC4',   # Teal/Green
        'SA': '#45B7D1',    # Blue
        'ASA': '#96CEB4'    # Light Green
    }
    
    algorithms = list(color_map.keys())
    colors = list(color_map.values())
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Color swatches
    y_pos = np.arange(len(algorithms))
    ax1.barh(y_pos, [1]*len(algorithms), color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(algorithms)
    ax1.set_xlabel('Color Intensity')
    ax1.set_title('Algorithm Color Scheme')
    ax1.set_xlim(0, 1.2)
    
    # Add hex color codes as text
    for i, (alg, color) in enumerate(color_map.items()):
        ax1.text(1.05, i, color, va='center', fontfamily='monospace')
    
    # Plot 2: Sample scatter plot showing distinctness
    np.random.seed(42)
    for i, (alg, color) in enumerate(color_map.items()):
        # Generate sample data
        x = np.random.normal(i*2, 0.5, 50)
        y = np.random.normal(i*1.5, 0.3, 50)
        
        ax2.scatter(x, y, c=color, label=alg, s=50, alpha=0.7)
    
    ax2.set_xlabel('Parameter Count (thousands)')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Sample Scatter Plot with New Colors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('color_scheme_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Color scheme test completed!")
    print("\nColor mapping:")
    for alg, color in color_map.items():
        print(f"  {alg}: {color}")
    
    print("\nColors are now more distinct:")
    print("  - GA/EGA: Bright Red (#FF6B6B)")
    print("  - PSO: Teal/Green (#4ECDC4)")  
    print("  - SA: Blue (#45B7D1)")
    print("  - ASA: Light Green (#96CEB4)")

if __name__ == "__main__":
    test_color_scheme()
