"""
Analysis and Visualization for Parametric Optimization Results
Analyzes the grid search results and creates visualizations for the research report.
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
    'figure.titlesize': 16
})

class ParametricAnalyzer:
    """Analyzes and visualizes parametric optimization results."""
    
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
    
    def analyze_all(self):
        """Perform all analyses and create visualizations."""
        print("🔍 Analyzing parametric optimization results...")
        
        try:
            # Parameter sensitivity analysis
            print("📊 Creating parameter sensitivity plots...")
            self.plot_parameter_sensitivity()
        except Exception as e:
            print(f"❌ Parameter sensitivity failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Best parameter comparison
            print("📊 Creating best parameter comparison...")
            self.plot_best_parameters()
        except Exception as e:
            print(f"❌ Best parameters plot failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Performance distribution analysis
            print("📊 Creating performance distributions...")
            self.plot_performance_distributions()
        except Exception as e:
            print(f"❌ Performance distributions failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Parameter interaction analysis
            print("📊 Creating parameter interactions...")
            self.plot_parameter_interactions()
        except Exception as e:
            print(f"❌ Parameter interactions failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Convergence analysis
            print("📊 Creating convergence analysis...")
            self.plot_parameter_convergence()
        except Exception as e:
            print(f"❌ Parameter convergence failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Summary statistics
            print("📊 Creating summary tables...")
            self.create_parameter_summary_table()
        except Exception as e:
            print(f"❌ Summary tables failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"✅ Analysis complete! Results saved to: {self.output_dir}")
    
    def plot_parameter_sensitivity(self):
        """Create parameter sensitivity plots for each algorithm."""
        for algorithm, results in self.results.items():
            if 'all_combinations' not in results:
                continue
            
            self._plot_algorithm_sensitivity(algorithm, results['all_combinations'])
    
    def _plot_algorithm_sensitivity(self, algorithm: str, combinations: List[Dict]):
        """Plot parameter sensitivity for a single algorithm."""
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
            row['avg_accuracy'] = combo['avg_accuracy']
            row['std_fitness'] = combo['std_fitness']
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Find varying parameters (ignore fixed ones)
        varying_params = []
        for col in df.columns:
            if col not in ['avg_fitness', 'avg_accuracy', 'std_fitness']:
                if df[col].nunique() > 1:
                    varying_params.append(col)
        
        if not varying_params:
            print(f"No varying parameters found for {algorithm}")
            return
        
        # Create parameter sensitivity plot
        n_params = len(varying_params)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Ensure axes is always indexable as a flat array
        if n_params == 1:
            axes = [axes]
        else:
            # Always flatten axes to handle consistently
            axes = np.array(axes).flatten() if hasattr(axes, 'flatten') else [axes]
        
        for i, param in enumerate(varying_params):
            # Get the axis using flat indexing
            ax = axes[i]
            
            # Debug: Check if ax is a proper Axes object
            print(f"Processing parameter {param}, ax type: {type(ax)}")
            
            # Group by parameter value and calculate statistics
            grouped = df.groupby(param)['avg_fitness'].agg(['mean', 'std', 'count'])
            
            x_values = grouped.index.tolist()
            y_means = grouped['mean'].tolist()
            y_stds = grouped['std'].fillna(0).tolist()
            
            # Create bar plot with error bars
            bars = ax.bar(range(len(x_values)), y_means, yerr=y_stds,
                         capsize=5, color=self.algorithm_colors.get(algorithm, '#666666'),
                         alpha=0.7)
            
            ax.set_xlabel(param)
            ax.set_ylabel('Average Fitness')
            ax.set_title(f'{algorithm} - {param} Sensitivity')
            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(x_values)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, y_means, y_stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Remove empty subplots
        if n_params < n_rows * n_cols:
            for i in range(n_params, min(len(axes), n_rows * n_cols)):
                axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{algorithm}_parameter_sensitivity.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Parameter sensitivity plot saved for {algorithm}")
    
    def plot_best_parameters(self):
        """Create comparison plot of best parameters across algorithms."""
        if not self.results:
            return
        
        # Extract best parameters for each algorithm
        best_params = {}
        best_fitness = {}
        
        for algorithm, results in self.results.items():
            if results.get('best_parameters'):
                best_params[algorithm] = results['best_parameters']
                best_fitness[algorithm] = results['best_avg_fitness']
        
        if not best_params:
            print("No best parameters found")
            return
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        ax1, ax2 = axes[0], axes[1]
        
        # Plot 1: Best fitness comparison
        algorithms = list(best_fitness.keys())
        fitness_values = list(best_fitness.values())
        colors = [self.algorithm_colors.get(alg, '#666666') for alg in algorithms]
        
        bars = ax1.bar(algorithms, fitness_values, color=colors, alpha=0.8)
        ax1.set_ylabel('Best Average Fitness')
        ax1.set_title('Best Fitness by Algorithm')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, fitness in zip(bars, fitness_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{fitness:.4f}', ha='center', va='bottom')
        
        # Plot 2: Parameter values comparison (text-based)
        ax2.axis('off')
        ax2.set_title('Best Parameter Configurations')
        
        y_pos = 0.9
        for algorithm, params in best_params.items():
            color = self.algorithm_colors.get(algorithm, '#666666')
            ax2.text(0.05, y_pos, f'{algorithm}:', fontweight='bold', 
                    color=color, fontsize=12, transform=ax2.transAxes)
            y_pos -= 0.08
            
            for param, value in params.items():
                ax2.text(0.1, y_pos, f'{param}: {value}', 
                        fontsize=10, transform=ax2.transAxes)
                y_pos -= 0.06
            y_pos -= 0.04
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'best_parameters_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Best parameters comparison plot saved")
    
    def plot_performance_distributions(self):
        """Plot performance distributions for each algorithm."""
        num_algorithms = len(self.results)
        if num_algorithms == 0:
            return
        
        fig, axes = plt.subplots(1, max(1, num_algorithms), figsize=(6*max(1, num_algorithms), 6))
        
        # Ensure axes is always a list for consistent indexing
        if num_algorithms == 1:
            axes = [axes]
        
        algorithm_idx = 0
        for algorithm, results in self.results.items():
            if 'all_combinations' not in results:
                continue
            
            # Extract fitness values from all successful combinations
            all_fitness = []
            for combo in results['all_combinations']:
                if combo['num_successful_runs'] > 0:
                    all_fitness.extend([run['best_fitness'] for run in combo['individual_runs'] 
                                      if run['success']])
            
            if not all_fitness:
                continue
            
            ax = axes[algorithm_idx]
            algorithm_idx += 1
            
            # Create histogram
            ax.hist(all_fitness, bins=20, alpha=0.7, 
                   color=self.algorithm_colors.get(algorithm, '#666666'),
                   edgecolor='black', linewidth=0.5)
            
            # Add statistics
            mean_fitness = np.mean(all_fitness)
            median_fitness = np.median(all_fitness)
            std_fitness = np.std(all_fitness)
            
            ax.axvline(mean_fitness, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_fitness:.3f}')
            ax.axvline(median_fitness, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_fitness:.3f}')
            
            ax.set_xlabel('Fitness Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{algorithm} - Fitness Distribution\n(σ = {std_fitness:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Performance distributions plot saved")
    
    def plot_parameter_interactions(self):
        """Plot parameter interaction effects (for algorithms with multiple varying parameters)."""
        for algorithm, results in self.results.items():
            self._plot_algorithm_interactions(algorithm, results)
    
    def _plot_algorithm_interactions(self, algorithm: str, results: Dict):
        """Plot parameter interactions for a single algorithm."""
        if 'all_combinations' not in results:
            return
        
        # Extract successful combinations
        successful = [c for c in results['all_combinations'] if c['num_successful_runs'] > 0]
        
        if not successful:
            return
        
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
        
        if len(varying_params) < 2:
            return  # Need at least 2 varying parameters for interaction plot
        
        # Create interaction plots for pairs of parameters
        n_pairs = min(6, len(varying_params) * (len(varying_params) - 1) // 2)  # Limit to 6 plots
        if n_pairs == 0:
            return
        
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Ensure axes is always indexable as a flat array
        if n_pairs == 1:
            axes = [axes]
        else:
            # Always flatten axes to handle consistently
            axes = np.array(axes).flatten() if hasattr(axes, 'flatten') else [axes]
        
        plot_idx = 0
        for i in range(len(varying_params)):
            for j in range(i+1, len(varying_params)):
                if plot_idx >= n_pairs:
                    break
                
                param1, param2 = varying_params[i], varying_params[j]
                
                # Get the axis using flat indexing
                ax = axes[plot_idx]
                
                # Create pivot table for heatmap
                pivot = df.pivot_table(values='avg_fitness', 
                                     index=param1, 
                                     columns=param2, 
                                     aggfunc='mean')
                
                # Create heatmap
                sns.heatmap(pivot, annot=True, fmt='.3f', ax=ax, 
                           cmap='viridis', cbar_kws={'label': 'Avg Fitness'})
                ax.set_title(f'{algorithm} - {param1} vs {param2}')
                
                plot_idx += 1
            
            if plot_idx >= n_pairs:
                break
        
        # Remove empty subplots
        if plot_idx < n_rows * n_cols:
            for i in range(plot_idx, n_rows * n_cols):
                if i < len(axes):
                    axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{algorithm}_parameter_interactions.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Parameter interactions plot saved for {algorithm}")
    
    def plot_parameter_convergence(self):
        """Plot how performance varies with specific parameter values."""
        for algorithm, results in self.results.items():
            self._plot_algorithm_convergence(algorithm, results)
    
    def _plot_algorithm_convergence(self, algorithm: str, results: Dict):
        """Plot parameter convergence for a single algorithm."""
        if 'all_combinations' not in results:
            return
        
        # Extract successful combinations with runtime information
        successful = [c for c in results['all_combinations'] if c['num_successful_runs'] > 0]
        
        if not successful:
            return
        
        # Sort by average fitness (best first)
        successful.sort(key=lambda x: x['avg_fitness'], reverse=True)
        
        # Create convergence plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Fitness vs combination index (sorted by performance)
        combination_indices = range(len(successful))
        fitness_values = [c['avg_fitness'] for c in successful]
        fitness_stds = [c['std_fitness'] for c in successful]
        
        ax1.errorbar(combination_indices, fitness_values, yerr=fitness_stds,
                    marker='o', linestyle='-', capsize=3, 
                    color=self.algorithm_colors.get(algorithm, '#666666'))
        
        ax1.set_xlabel('Parameter Combination (sorted by performance)')
        ax1.set_ylabel('Average Fitness')
        ax1.set_title(f'{algorithm} - Parameter Combination Performance')
        ax1.grid(True, alpha=0.3)
        
        # Highlight top 3 combinations
        for i in range(min(3, len(successful))):
            ax1.scatter(i, fitness_values[i], s=100, 
                       color='red', marker='*', zorder=5)
            ax1.annotate(f'Top {i+1}', (i, fitness_values[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        # Plot 2: Runtime vs performance trade-off
        runtimes = [c['avg_runtime'] / 3600 for c in successful]  # Convert to hours
        
        scatter = ax2.scatter(runtimes, fitness_values, 
                            s=60, alpha=0.7,
                            c=range(len(successful)), 
                            cmap='viridis')
        
        ax2.set_xlabel('Average Runtime (hours)')
        ax2.set_ylabel('Average Fitness')
        ax2.set_title(f'{algorithm} - Runtime vs Performance Trade-off')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Performance Rank')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{algorithm}_parameter_convergence.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Parameter convergence plot saved for {algorithm}")
    
    def create_parameter_summary_table(self):
        """Create summary tables for the optimization results."""
        
        # Create LaTeX table for best parameters
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Best Parameter Configurations from Grid Search}\n"
        latex_table += "\\begin{tabular}{|l|l|c|c|}\n"
        latex_table += "\\hline\n"
        latex_table += "Algorithm & Parameter & Best Value & Fitness \\\\\n"
        latex_table += "\\hline\n"
        
        for algorithm, results in self.results.items():
            if not results.get('best_parameters'):
                continue
            
            params = results['best_parameters']
            fitness = results['best_avg_fitness']
            
            first_param = True
            for param, value in params.items():
                if first_param:
                    latex_table += f"\\multirow{{{len(params)}}}{{*}}{{{algorithm}}} & "
                    latex_table += f"{param} & {value} & "
                    latex_table += f"\\multirow{{{len(params)}}}{{*}}{{{fitness:.4f}}} \\\\\n"
                    first_param = False
                else:
                    latex_table += f" & {param} & {value} & \\\\\n"
            
            latex_table += "\\hline\n"
        
        latex_table += "\\end{tabular}\n"
        latex_table += "\\label{tab:best_parameters}\n"
        latex_table += "\\end{table}\n"
        
        # Save LaTeX table
        with open(self.output_dir / 'best_parameters_table.tex', 'w') as f:
            f.write(latex_table)
        
        # Create summary statistics table
        summary_data = []
        for algorithm, results in self.results.items():
            if 'all_combinations' in results:
                successful = [c for c in results['all_combinations'] if c['num_successful_runs'] > 0]
                
                if successful:
                    all_fitness = []
                    for combo in successful:
                        all_fitness.extend([run['best_fitness'] for run in combo['individual_runs'] 
                                          if run['success']])
                    
                    summary_data.append({
                        'Algorithm': algorithm,
                        'Total Combinations': len(results['all_combinations']),
                        'Successful Combinations': len(successful),
                        'Success Rate (%)': len(successful) / len(results['all_combinations']) * 100,
                        'Best Fitness': max(all_fitness) if all_fitness else 0,
                        'Mean Fitness': np.mean(all_fitness) if all_fitness else 0,
                        'Std Fitness': np.std(all_fitness) if all_fitness else 0
                    })
        
        # Save as CSV
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.output_dir / 'parametric_optimization_summary.csv', index=False)
        
        print("📋 Summary tables saved (LaTeX and CSV)")

def main():
    """Main function for parametric optimization analysis."""
    parser = argparse.ArgumentParser(description='Analyze Parametric Optimization Results')
    parser.add_argument('--results-dir', type=str, default='parametric_results',
                       help='Directory containing parametric optimization results')
    parser.add_argument('--output-dir', type=str, default='parametric_figures',
                       help='Output directory for analysis plots')
    
    args = parser.parse_args()
    
    print("📊 Parametric Optimization Analysis")
    print("="*50)
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory {args.results_dir} does not exist!")
        return
    
    # Create analyzer
    analyzer = ParametricAnalyzer(args.results_dir, args.output_dir)
    
    if not analyzer.results:
        print("❌ No parametric optimization results found!")
        return
    
    # Run analysis
    analyzer.analyze_all()
    
    print("\n✅ Parametric optimization analysis complete!")
    print(f"📁 Plots and tables saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
