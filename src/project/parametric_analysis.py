"""
Enhanced Analysis and Visualization for Parametric Optimization Results
Creates sensitivity plots similar to the reference image with clean parameter vs fitness visualization.
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Any
import argparse
from pathlib import Path
from itertools import combinations

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

class EnhancedParametricAnalyzer:
    """Enhanced analyzer for parametric optimization results with improved sensitivity plots."""
    
    def __init__(self, results_dir: str, output_dir: str = "parametric_figures"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        self.results = self._load_results()
        
        # Algorithm colors for consistency
        self.algorithm_colors = {
            'GA': '#FF6B6B',    # Red
            'PSO': '#4ECDC4',   # Teal
            'SA': '#FFD93D'     # Yellow
        }
        
        # Line styles for different parameter values
        self.line_styles = ['-', '--', '-.', ':']
        self.markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    def _load_results(self) -> Dict:
        """Load parametric optimization results."""
        results = {}
        
        # Try to load comprehensive results first
        comprehensive_file = self.results_dir / "comprehensive_parametric_results.pkl"
        if comprehensive_file.exists():
            with open(comprehensive_file, 'rb') as f:
                comprehensive_data = pickle.load(f)
                if 'algorithm_results' in comprehensive_data:
                    results = comprehensive_data['algorithm_results']
        
        # Load individual algorithm results if comprehensive not available
        if not results:
            for algorithm in ['GA', 'PSO', 'SA']:
                result_file = self.results_dir / f"{algorithm}_parametric_results.pkl"
                if result_file.exists():
                    with open(result_file, 'rb') as f:
                        results[algorithm] = pickle.load(f)
        
        print(f"Loaded results for algorithms: {list(results.keys())}")
        return results
    
    def create_simple_sensitivity_plots(self):
        """Create simple single-line sensitivity plots for each algorithm."""
        print("📊 Creating simple parameter sensitivity plots...")
        
        for algorithm, results in self.results.items():
            if 'all_combinations' not in results:
                continue
            
            self._create_simple_algorithm_sensitivity(algorithm, results['all_combinations'])
    
    def _create_simple_algorithm_sensitivity(self, algorithm: str, combinations: List[Dict]):
        """Create simple sensitivity plots for a single algorithm with one line per parameter."""
        # Extract successful combinations
        successful = [c for c in combinations if c['num_successful_runs'] > 0]
        
        if not successful:
            print(f"No successful combinations for {algorithm}")
            return
        
        # Create DataFrame for analysis
        data = []
        for combo in successful:
            row = combo['parameters'].copy()
            row['avg_fitness'] = combo['avg_fitness']
            row['std_fitness'] = combo['std_fitness']
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Find varying parameters (ignore fixed ones)
        varying_params = []
        param_ranges = {}
        for col in df.columns:
            if col not in ['avg_fitness', 'std_fitness']:
                unique_values = df[col].unique()
                if len(unique_values) > 1:
                    varying_params.append(col)
                    param_ranges[col] = sorted(unique_values)
        
        if not varying_params:
            print(f"No varying parameters found for {algorithm}")
            return
        
        # Create parameter sensitivity plot grid
        n_params = len(varying_params)
        n_cols = min(2, n_params)  # Maximum 2 columns
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        fig.suptitle(f'{algorithm} Parameter Sensitivity Analysis', fontsize=16, y=0.98)
        
        # Ensure axes is always indexable as a flat array
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes) if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(varying_params):
            ax = axes[i]
            
            # Create simple sensitivity plot for this parameter
            self._plot_parameter_sensitivity_lines(ax, df, param, param_ranges, algorithm)
        
        # Remove empty subplots
        if n_params < len(axes):
            for i in range(n_params, len(axes)):
                axes[i].remove()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(self.output_dir / f'{algorithm}_simple_sensitivity.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Simple sensitivity plot saved for {algorithm}")
    
    def _plot_parameter_sensitivity_lines(self, ax, df, main_param, param_ranges, algorithm):
        """Plot simple parameter sensitivity with single line per parameter and intermediate x-axis values."""
        # Group by main parameter and calculate average fitness across all other parameter combinations
        grouped = df.groupby(main_param).agg({
            'avg_fitness': ['mean', 'std']
        }).reset_index()
        
        x_values = grouped[main_param].values
        y_means = grouped[('avg_fitness', 'mean')].values
        y_stds = grouped[('avg_fitness', 'std')].fillna(0).values
        
        # Create simple line plot with markers
        ax.plot(x_values, y_means, 
               marker='o', linestyle='-', linewidth=2, markersize=8,
               color=self.algorithm_colors.get(algorithm, '#666666'),
               markerfacecolor='white', markeredgecolor=self.algorithm_colors.get(algorithm, '#666666'),
               markeredgewidth=2)
        
        # Add error bars if there's variation
        if np.any(y_stds > 0):
            ax.errorbar(x_values, y_means, yerr=y_stds,
                       fmt='none', color=self.algorithm_colors.get(algorithm, '#666666'),
                       capsize=4, capthick=1, alpha=0.7)
        
        # Formatting
        ax.set_xlabel(main_param)
        ax.set_ylabel('Average Fitness')
        ax.set_title(f'Average Fitness vs. {main_param}')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis ticks with intermediate values
        if main_param in param_ranges:
            param_values = sorted(param_ranges[main_param])
            
            # Create intermediate ticks based on parameter type
            if len(param_values) >= 2:
                min_val = min(param_values)
                max_val = max(param_values)
                
                # Determine if values are integers or floats
                if all(isinstance(val, (int, np.integer)) for val in param_values):
                    # Integer parameters - add intermediate integer values
                    if max_val - min_val > 1:
                        intermediate_ticks = list(range(int(min_val), int(max_val) + 1))
                    else:
                        intermediate_ticks = param_values
                else:
                    # Float parameters - add intermediate float values
                    step = (max_val - min_val) / max(10, len(param_values) * 2)  # More granular steps
                    intermediate_ticks = np.arange(min_val, max_val + step/2, step)
                    intermediate_ticks = np.round(intermediate_ticks, 3)  # Round to 3 decimal places
                
                ax.set_xticks(intermediate_ticks)
                
                # Rotate labels if there are many ticks
                if len(intermediate_ticks) > 6:
                    ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_xticks(param_values)
        
        # Format y-axis to show more decimal places for small values
        ax.ticklabel_format(style='plain', axis='y')
        
        # Add some padding to y-axis
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    def create_comprehensive_sensitivity_comparison(self):
        """Create a comprehensive comparison plot across all algorithms."""
        print("📊 Creating comprehensive sensitivity comparison...")
        
        # Collect all varying parameters across algorithms
        all_params = set()
        algorithm_data = {}
        
        for algorithm, results in self.results.items():
            if 'all_combinations' not in results:
                continue
            
            successful = [c for c in results['all_combinations'] if c['num_successful_runs'] > 0]
            if not successful:
                continue
            
            # Create DataFrame
            data = []
            for combo in successful:
                row = combo['parameters'].copy()
                row['avg_fitness'] = combo['avg_fitness']
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Find varying parameters
            varying_params = []
            for col in df.columns:
                if col != 'avg_fitness' and df[col].nunique() > 1:
                    varying_params.append(col)
                    all_params.add(col)
            
            algorithm_data[algorithm] = {
                'df': df,
                'varying_params': varying_params
            }
        
        if not all_params:
            print("No varying parameters found across algorithms")
            return
        
        # Create comparison plots
        common_params = []
        for param in all_params:
            param_count = sum(1 for alg_data in algorithm_data.values() 
                            if param in alg_data['varying_params'])
            if param_count > 1:  # Parameter varies in multiple algorithms
                common_params.append(param)
        
        if not common_params:
            print("No common varying parameters found")
            return
        
        n_params = len(common_params)
        n_cols = min(2, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
        fig.suptitle('Parameter Sensitivity Comparison Across Algorithms', fontsize=16, y=0.98)
        
        # Ensure axes is always indexable
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes) if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(common_params):
            ax = axes[i]
            
            # Plot each algorithm's sensitivity for this parameter
            for algorithm, alg_data in algorithm_data.items():
                if param not in alg_data['varying_params']:
                    continue
                
                df = alg_data['df']
                
                # Group by parameter value
                grouped = df.groupby(param).agg({
                    'avg_fitness': ['mean', 'std']
                }).reset_index()
                
                x_values = grouped[param].values
                y_means = grouped[('avg_fitness', 'mean')].values
                y_stds = grouped[('avg_fitness', 'std')].fillna(0).values
                
                ax.errorbar(x_values, y_means, yerr=y_stds,
                           marker='o', linestyle='-', linewidth=2, markersize=6,
                           color=self.algorithm_colors.get(algorithm, '#666666'),
                           label=algorithm, capsize=4, capthick=1)
            
            ax.set_xlabel(param)
            ax.set_ylabel('Average Fitness')
            ax.set_title(f'Algorithm Comparison - {param}')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        
        # Remove empty subplots
        if n_params < len(axes):
            for i in range(n_params, len(axes)):
                axes[i].remove()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(self.output_dir / 'algorithm_sensitivity_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("📊 Comprehensive sensitivity comparison saved")
    
    def analyze_all(self):
        """Perform all enhanced analyses."""
        print("🔍 Starting enhanced parametric optimization analysis...")
        
        if not self.results:
            print("❌ No results to analyze!")
            return
        
        try:
            # Simple parameter sensitivity analysis (single line per parameter)
            self.create_simple_sensitivity_plots()
        except Exception as e:
            print(f"❌ Simple sensitivity plots failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Comprehensive comparison
            self.create_comprehensive_sensitivity_comparison()
        except Exception as e:
            print(f"❌ Comprehensive comparison failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"✅ Enhanced analysis complete! Results saved to: {self.output_dir}")

def main():
    """Main function for enhanced parametric optimization analysis."""
    parser = argparse.ArgumentParser(description='Enhanced Parametric Optimization Analysis')
    parser.add_argument('--results-dir', type=str, default='parametric_results',
                       help='Directory containing parametric optimization results')
    parser.add_argument('--output-dir', type=str, default='parametric_figures',
                       help='Output directory for analysis plots')
    
    args = parser.parse_args()
    
    print("📊 Enhanced Parametric Optimization Analysis")
    print("="*50)
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory {args.results_dir} does not exist!")
        return
    
    # Create enhanced analyzer
    analyzer = EnhancedParametricAnalyzer(args.results_dir, args.output_dir)
    
    if not analyzer.results:
        print("❌ No parametric optimization results found!")
        return
    
    # Run enhanced analysis
    analyzer.analyze_all()
    
    print("\n✅ Enhanced parametric optimization analysis complete!")
    print(f"📁 Plots and tables saved to: {args.output_dir}")

if __name__ == "__main__":
    main()