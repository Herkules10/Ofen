#!/usr/bin/env python3
"""
Visualization script for Neural Architecture Search results.
Generates plots and figures for the research report.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsVisualizer:
    """Creates visualizations from experiment results."""
    
    def __init__(self, results_dir: str, output_dir: str = "figures"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all results
        self.results = self._load_all_results()
        self.summary_data = self._load_summary_data()
        
    def _load_all_results(self) -> Dict:
        """Load all experiment results from pickle files."""
        results = {}
        
        for filepath in self.results_dir.glob("*_all_runs.pkl"):
            algorithm_name = filepath.stem.split('_')[0]
            with open(filepath, 'rb') as f:
                results[algorithm_name] = pickle.load(f)
        
        return results
    
    def _load_summary_data(self) -> Dict:
        """Load summary statistics."""
        summary_files = list(self.results_dir.glob("experiment_summary_*.pkl"))
        if summary_files:
            with open(summary_files[0], 'rb') as f:
                return pickle.load(f)
        return {}
    
    def create_all_plots(self):
        """Generate all visualization plots."""
        print("Generating all plots...")
        
        # Performance comparison plots
        self.plot_performance_comparison()
        self.plot_parameter_efficiency()
        self.plot_convergence_curves()
        
        # Algorithm-specific plots
        self.plot_fitness_distributions()
        self.plot_runtime_comparison()
        self.plot_architecture_characteristics()
        
        # Individual algorithm analysis
        self.plot_individual_convergence()
        
        print(f"All plots saved to: {self.output_dir}")
    
    def plot_performance_comparison(self):
        """Plot test accuracy comparison across algorithms."""
        if not self.summary_data:
            print("No summary data available for performance comparison")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        algorithms = list(self.summary_data.keys())
        accuracies = [self.summary_data[alg]['test_accuracy']['mean'] for alg in algorithms]
        accuracy_stds = [self.summary_data[alg]['test_accuracy']['std'] for alg in algorithms]
        fitnesses = [self.summary_data[alg]['best_fitness']['mean'] for alg in algorithms]
        fitness_stds = [self.summary_data[alg]['best_fitness']['std'] for alg in algorithms]
        
        # Test accuracy plot
        bars1 = ax1.bar(algorithms, accuracies, yerr=accuracy_stds, capsize=5, alpha=0.8)
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylim(0, max(accuracies) * 1.1)
        
        # Add value labels on bars
        for bar, acc, std in zip(bars1, accuracies, accuracy_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Fitness comparison plot
        bars2 = ax2.bar(algorithms, fitnesses, yerr=fitness_stds, capsize=5, alpha=0.8)
        ax2.set_ylabel('Fitness Score')
        ax2.set_title('Fitness Score Comparison')
        
        # Add value labels on bars
        for bar, fit, std in zip(bars2, fitnesses, fitness_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                    f'{fit:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_parameter_efficiency(self):
        """Plot parameter efficiency (accuracy per parameter)."""
        if not self.summary_data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(self.summary_data.keys())
        accuracies = [self.summary_data[alg]['test_accuracy']['mean'] for alg in algorithms]
        param_counts = [self.summary_data[alg]['parameter_count']['mean'] / 1000 for alg in algorithms]  # in thousands
        
        # Scatter plot
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        for i, (alg, acc, params) in enumerate(zip(algorithms, accuracies, param_counts)):
            ax.scatter(params, acc, s=150, c=[colors[i]], label=alg, alpha=0.8)
            ax.annotate(alg, (params, acc), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Parameter Count (thousands)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Parameter Efficiency: Accuracy vs Model Size')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_convergence_curves(self):
        """Plot convergence curves for all algorithms."""
        if not self.results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for i, (algorithm, runs) in enumerate(self.results.items()):
            if not runs:
                continue
            
            # Collect convergence data from all runs
            all_convergence = []
            for run in runs:
                if 'convergence_data' in run:
                    conv_data = run['convergence_data']
                    if 'best_fitness' in conv_data:
                        all_convergence.append(conv_data['best_fitness'])
            
            if not all_convergence:
                continue
            
            # Find minimum length to align all runs
            min_length = min(len(conv) for conv in all_convergence)
            aligned_convergence = [conv[:min_length] for conv in all_convergence]
            
            # Calculate mean and std
            mean_convergence = np.mean(aligned_convergence, axis=0)
            std_convergence = np.std(aligned_convergence, axis=0)
            generations = range(len(mean_convergence))
            
            # Plot mean with confidence interval
            ax.plot(generations, mean_convergence, label=algorithm, color=colors[i], linewidth=2)
            ax.fill_between(generations, 
                           mean_convergence - std_convergence,
                           mean_convergence + std_convergence,
                           alpha=0.2, color=colors[i])
        
        ax.set_xlabel('Generation/Iteration')
        ax.set_ylabel('Best Fitness')
        ax.set_title('Convergence Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_fitness_distributions(self):
        """Plot fitness distribution box plots."""
        if not self.results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fitness_data = []
        labels = []
        
        for algorithm, runs in self.results.items():
            if runs:
                fitnesses = [run['best_fitness'] for run in runs]
                fitness_data.append(fitnesses)
                labels.append(algorithm)
        
        if fitness_data:
            bp = ax.boxplot(fitness_data, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(fitness_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
        
        ax.set_ylabel('Best Fitness')
        ax.set_title('Fitness Distribution Across Algorithms')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fitness_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_runtime_comparison(self):
        """Plot runtime comparison."""
        if not self.summary_data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(self.summary_data.keys())
        runtimes = [self.summary_data[alg]['runtime']['mean'] / 3600 for alg in algorithms]  # Convert to hours
        runtime_stds = [self.summary_data[alg]['runtime']['std'] / 3600 for alg in algorithms]
        
        bars = ax.bar(algorithms, runtimes, yerr=runtime_stds, capsize=5, alpha=0.8)
        ax.set_ylabel('Runtime (hours)')
        ax.set_title('Algorithm Runtime Comparison')
        
        # Add value labels
        for bar, runtime, std in zip(bars, runtimes, runtime_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
                   f'{runtime:.1f}h', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'runtime_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_architecture_characteristics(self):
        """Plot architecture characteristics (layers, parameters)."""
        if not self.results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect data
        algorithms = []
        layer_counts = []
        param_counts = []
        
        for algorithm, runs in self.results.items():
            if runs:
                algorithms.append(algorithm)
                layers = [len(run['best_architecture'].layers) for run in runs]
                params = [run['parameter_count'] / 1000 for run in runs]  # in thousands
                layer_counts.append(layers)
                param_counts.append(params)
        
        # Layer count box plot
        if layer_counts:
            bp1 = ax1.boxplot(layer_counts, labels=algorithms, patch_artist=True)
            colors = plt.cm.Set2(np.linspace(0, 1, len(layer_counts)))
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
        
        ax1.set_ylabel('Number of Layers')
        ax1.set_title('Architecture Depth Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Parameter count box plot
        if param_counts:
            bp2 = ax2.boxplot(param_counts, labels=algorithms, patch_artist=True)
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
        
        ax2.set_ylabel('Parameter Count (thousands)')
        ax2.set_title('Model Size Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_individual_convergence(self):
        """Plot individual convergence curves for each algorithm."""
        for algorithm, runs in self.results.items():
            if not runs:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))
            
            for i, run in enumerate(runs):
                if 'convergence_data' in run:
                    conv_data = run['convergence_data']
                    if 'best_fitness' in conv_data:
                        generations = conv_data.get('generations', 
                                                  range(len(conv_data['best_fitness'])))
                        ax.plot(generations, conv_data['best_fitness'], 
                               alpha=0.7, color=colors[i], 
                               label=f'Run {run.get("run_id", i+1)}')
            
            ax.set_xlabel('Generation/Iteration')
            ax.set_ylabel('Best Fitness')
            ax.set_title(f'{algorithm} - Individual Run Convergence')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'convergence_{algorithm.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_table(self):
        """Create a summary table of results."""
        if not self.summary_data:
            return
        
        # Create LaTeX table
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Algorithm Performance Comparison}\n"
        latex_table += "\\begin{tabular}{|l|c|c|c|c|}\n"
        latex_table += "\\hline\n"
        latex_table += "Algorithm & Test Accuracy (\\%) & Parameters & Fitness & Runtime (h) \\\\\n"
        latex_table += "\\hline\n"
        
        for alg, stats in self.summary_data.items():
            acc_mean = stats['test_accuracy']['mean']
            acc_std = stats['test_accuracy']['std']
            param_mean = stats['parameter_count']['mean'] / 1000  # in thousands
            fitness_mean = stats['best_fitness']['mean']
            runtime_mean = stats['runtime']['mean'] / 3600  # in hours
            
            latex_table += f"{alg} & {acc_mean:.1f} ± {acc_std:.1f} & "
            latex_table += f"{param_mean:.0f}k & {fitness_mean:.3f} & {runtime_mean:.1f} \\\\\n"
        
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\label{tab:results}\n"
        latex_table += "\\end{table}\n"
        
        # Save table
        with open(self.output_dir / 'results_table.tex', 'w') as f:
            f.write(latex_table)
        
        print("Summary table saved as results_table.tex")

def main():
    parser = argparse.ArgumentParser(description='Generate visualization plots from NAS results')
    parser.add_argument('--results_dir', default='results',
                        help='Directory containing result files')
    parser.add_argument('--output_dir', default='figures',
                        help='Output directory for plots')
    parser.add_argument('--plots', nargs='+',
                        choices=['performance', 'efficiency', 'convergence', 'fitness', 
                                'runtime', 'architecture', 'individual', 'all'],
                        default=['all'], help='Which plots to generate')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Results directory {args.results_dir} does not exist!")
        return
    
    # Create visualizer
    visualizer = ResultsVisualizer(args.results_dir, args.output_dir)
    
    # Generate selected plots
    if 'all' in args.plots:
        visualizer.create_all_plots()
        visualizer.create_summary_table()
    else:
        if 'performance' in args.plots:
            visualizer.plot_performance_comparison()
        if 'efficiency' in args.plots:
            visualizer.plot_parameter_efficiency()
        if 'convergence' in args.plots:
            visualizer.plot_convergence_curves()
        if 'fitness' in args.plots:
            visualizer.plot_fitness_distributions()
        if 'runtime' in args.plots:
            visualizer.plot_runtime_comparison()
        if 'architecture' in args.plots:
            visualizer.plot_architecture_characteristics()
        if 'individual' in args.plots:
            visualizer.plot_individual_convergence()
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()