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
        self.baseline_data = self._load_baseline_data()
    
    def _get_algorithm_colors(self, algorithms: List[str]) -> List[str]:
        """Get consistent colors for algorithms."""
        color_map = {
            'GA': '#FF6B6B',    # Red
            'EGA': '#FF6B6B',   # Red (same as GA) 
            'PSO': '#4ECDC4',   # Teal/Green
            'SA': "#2F00FF",    # Blue
            'ASA': "#7700FF",   # Purple
            'Random Search': '#888888',  # Gray for baseline
            'RS': '#888888'     # Gray for baseline (alternative name)
        }
        
        distinct_colors = ['#FF6B6B', '#4ECDC4', "#1100FF", '#888888']
        colors = []
        for alg in algorithms:
            if alg in color_map:
                colors.append(color_map[alg])
            else:
                colors.append(distinct_colors[len(colors) % len(distinct_colors)])
        return colors
        
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
    
    def _load_baseline_data(self) -> Dict:
        """Load baseline results if available."""
        baseline_files = list(self.results_dir.glob("baseline_*.pkl"))
        if baseline_files:
            print(f"Found baseline file: {baseline_files[0]}")
            with open(baseline_files[0], 'rb') as f:
                baseline_data = pickle.load(f)
                # Ensure consistent naming
                if 'algorithm' not in baseline_data:
                    baseline_data['algorithm'] = 'Random Search'
                return baseline_data
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
        
        # New: Fitness evaluation progress plot
        self.plot_fitness_evaluation_progress()
        
        # New: Baseline comparison plot
        if self.baseline_data:
            self.plot_baseline_comparison()
        
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
        
        # Add baseline if available
        if self.baseline_data:
            algorithms.append(self.baseline_data.get('algorithm', 'Random Search'))
            accuracies.append(self.baseline_data['test_accuracy'])
            accuracy_stds.append(0)  # Single run, no std
            fitnesses.append(self.baseline_data['best_fitness'])
            fitness_stds.append(0)
        
        # Get colors including baseline
        colors = self._get_algorithm_colors(algorithms)
        
        # Test accuracy plot
        bars1 = ax1.bar(algorithms, accuracies, yerr=accuracy_stds, capsize=5, alpha=0.8, color=colors)
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylim(0, max(accuracies) * 1.1)
        
        # Add value labels on bars
        for bar, acc, std in zip(bars1, accuracies, accuracy_stds):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Fitness comparison plot
        bars2 = ax2.bar(algorithms, fitnesses, yerr=fitness_stds, capsize=5, alpha=0.8, color=colors)
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
        """Plot parameter efficiency (accuracy per parameter) - creates both auto-scaled and 0-100% versions."""
        if not self.summary_data:
            return
        
        algorithms = list(self.summary_data.keys())
        accuracies = [self.summary_data[alg]['test_accuracy']['mean'] for alg in algorithms]
        param_counts = [self.summary_data[alg]['parameter_count']['mean'] / 1000 for alg in algorithms]  # in thousands
        
        # Add baseline if available
        if self.baseline_data:
            algorithms.append(self.baseline_data.get('algorithm', 'Random Search'))
            accuracies.append(self.baseline_data['test_accuracy'])
            param_counts.append(self.baseline_data['parameter_count'] / 1000)
        
        # Use consistent colors
        colors = self._get_algorithm_colors(algorithms)
        
        # Create first plot with automatic scaling (original version)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        for i, (alg, acc, params) in enumerate(zip(algorithms, accuracies, param_counts)):
            ax.scatter(params, acc, s=150, c=colors[i], label=alg, alpha=0.8)
            ax.annotate(alg, (params, acc), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Parameter Count (thousands)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Parameter Efficiency: Accuracy vs Model Size (Auto-scaled)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_efficiency_auto.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create second plot with 0-100% y-axis scaling
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        for i, (alg, acc, params) in enumerate(zip(algorithms, accuracies, param_counts)):
            ax.scatter(params, acc, s=150, c=colors[i], label=alg, alpha=0.8)
            ax.annotate(alg, (params, acc), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Parameter Count (thousands)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_ylim(0, 100)  # Set y-axis range from 0 to 100%
        ax.set_title('Parameter Efficiency: Accuracy vs Model Size (0-100% Scale)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_efficiency_0to100.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_convergence_curves(self):
        """Plot convergence curves for all algorithms using number of evaluations and cumulative best fitness."""
        if not self.results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use consistent colors
        algorithm_names = list(self.results.keys())
        colors = self._get_algorithm_colors(algorithm_names)
        
        for i, (algorithm, runs) in enumerate(self.results.items()):
            if not runs:
                continue
            
            # Collect evaluation-based convergence data from all runs
            all_run_data = []
            for run in runs:
                if 'convergence_data' in run and 'evaluation_log' in run['convergence_data']:
                    eval_log = run['convergence_data']['evaluation_log']
                    if eval_log:
                        # Extract evaluation numbers and fitness values
                        evaluations = [eval_data['evaluation_number'] for eval_data in eval_log]
                        fitness_values = [eval_data['fitness'] for eval_data in eval_log]
                        
                        # Sort by evaluation number to ensure proper order
                        sorted_pairs = sorted(zip(evaluations, fitness_values))
                        sorted_evals, sorted_fitness = zip(*sorted_pairs)
                        
                        # Calculate cumulative best fitness (best so far)
                        cumulative_best = []
                        best_so_far = -float('inf')  # Start with very low value
                        for fitness in sorted_fitness:
                            if fitness > best_so_far:
                                best_so_far = fitness
                            cumulative_best.append(best_so_far)
                        
                        all_run_data.append((sorted_evals, cumulative_best))
            
            if not all_run_data:
                continue
            
            # Find the maximum number of evaluations across all runs for this algorithm
            max_evaluations = max(max(evals) for evals, _ in all_run_data)
            
            # Create a common evaluation grid
            eval_grid = np.arange(1, max_evaluations + 1)
            
            # Interpolate each run's data to the common grid
            interpolated_runs = []
            for evaluations, cumulative_best in all_run_data:
                # Interpolate cumulative best fitness to the common evaluation grid
                interpolated_best = np.interp(eval_grid, evaluations, cumulative_best)
                interpolated_runs.append(interpolated_best)
            
            if interpolated_runs:
                # Calculate mean and std across runs
                interpolated_array = np.array(interpolated_runs)
                mean_convergence = np.mean(interpolated_array, axis=0)
                std_convergence = np.std(interpolated_array, axis=0)
                
                # Plot mean with confidence interval
                ax.plot(eval_grid, mean_convergence, label=algorithm, color=colors[i], linewidth=2)
                ax.fill_between(eval_grid, 
                               mean_convergence - std_convergence,
                               mean_convergence + std_convergence,
                               alpha=0.2, color=colors[i])
        
        # Add baseline horizontal line if available
        if self.baseline_data:
            baseline_fitness = self.baseline_data['best_fitness']
            ax.axhline(y=baseline_fitness, color='#888888', linestyle='--', 
                      linewidth=2, label='Random Search', alpha=0.8)
        
        ax.set_xlabel('Number of Evaluations')
        ax.set_ylabel('Best Fitness Achieved So Far')
        ax.set_title('Convergence Curves Comparison - Cumulative Best Fitness')
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
    
    def plot_fitness_evaluation_progress(self):
        """Plot fitness vs evaluation number scatter plot."""
        if not self.results:
            print("No results available for fitness evaluation progress plot")
            return
        
        # Create combined plot
        self._plot_combined_fitness_evaluation_progress()
        
        # Create individual plots for each algorithm
        self._plot_individual_fitness_evaluation_progress()
        
        print("Fitness evaluation progress plots saved")
    
    def _plot_combined_fitness_evaluation_progress(self):
        """Create combined fitness evaluation progress plot for all algorithms."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use consistent colors
        algorithm_names = list(self.results.keys())
        colors = self._get_algorithm_colors(algorithm_names)
        
        for i, (algorithm, runs) in enumerate(self.results.items()):
            if not runs:
                continue
            
            # Collect all evaluation data from all runs
            all_evaluations = []
            all_fitness = []
            
            for run in runs:
                if 'convergence_data' in run and 'evaluation_log' in run['convergence_data']:
                    eval_log = run['convergence_data']['evaluation_log']
                    for eval_data in eval_log:
                        all_evaluations.append(eval_data['evaluation_number'])
                        all_fitness.append(eval_data['fitness'])
            
            if all_evaluations and all_fitness:
                # Create scatter plot
                ax.scatter(all_evaluations, all_fitness, 
                          s=20, alpha=0.6, c=colors[i], 
                          label=algorithm, marker='o')
        
        ax.set_xlabel('Number of Fitness Evaluations')
        ax.set_ylabel('Fitness Value')
        ax.set_title('Fitness Evolution Progress - Individual Network Evaluations (All Algorithms)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add trend lines for each algorithm
        for i, (algorithm, runs) in enumerate(self.results.items()):
            if not runs:
                continue
                
            all_evaluations = []
            all_fitness = []
            
            for run in runs:
                if 'convergence_data' in run and 'evaluation_log' in run['convergence_data']:
                    eval_log = run['convergence_data']['evaluation_log']
                    for eval_data in eval_log:
                        all_evaluations.append(eval_data['evaluation_number'])
                        all_fitness.append(eval_data['fitness'])
            
            if len(all_evaluations) > 10:  # Only add trend line if we have enough points
                # Sort by evaluation number
                sorted_pairs = sorted(zip(all_evaluations, all_fitness))
                sorted_evals, sorted_fitness = zip(*sorted_pairs)
                
                # Calculate moving average for trend line
                window_size = max(10, len(sorted_fitness) // 50)
                if len(sorted_fitness) >= window_size:
                    moving_avg = np.convolve(sorted_fitness, 
                                           np.ones(window_size)/window_size, 
                                           mode='valid')
                    moving_evals = sorted_evals[window_size-1:]
                    ax.plot(moving_evals, moving_avg, 
                           color=colors[i], linewidth=2, alpha=0.8,
                           linestyle='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fitness_evaluation_progress_combined.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_fitness_evaluation_progress(self):
        """Create individual fitness evaluation progress plots for each algorithm."""
        algorithm_names = list(self.results.keys())
        colors = self._get_algorithm_colors(algorithm_names)
        
        for i, (algorithm, runs) in enumerate(self.results.items()):
            if not runs:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Collect all evaluation data from all runs
            all_evaluations = []
            all_fitness = []
            
            for run in runs:
                if 'convergence_data' in run and 'evaluation_log' in run['convergence_data']:
                    eval_log = run['convergence_data']['evaluation_log']
                    for eval_data in eval_log:
                        all_evaluations.append(eval_data['evaluation_number'])
                        all_fitness.append(eval_data['fitness'])
            
            if all_evaluations and all_fitness:
                # Create scatter plot with same styling as combined graph
                ax.scatter(all_evaluations, all_fitness, 
                          s=20, alpha=0.6, c=colors[i], 
                          label=algorithm, marker='o')
            
            ax.set_xlabel('Number of Fitness Evaluations')
            ax.set_ylabel('Fitness Value')
            ax.set_title(f'{algorithm} - Fitness Evolution Progress')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'fitness_evaluation_progress_{algorithm}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual fitness evaluation progress plots saved for {len(self.results)} algorithms")
    
    def plot_baseline_comparison(self):
        """Create a dedicated baseline comparison visualization."""
        if not self.baseline_data or not self.summary_data:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Get all algorithms including baseline
        algorithms = list(self.summary_data.keys()) + [self.baseline_data.get('algorithm', 'Random Search')]
        accuracies = [self.summary_data[alg]['test_accuracy']['mean'] for alg in self.summary_data.keys()] + [self.baseline_data['test_accuracy']]
        param_counts = [self.summary_data[alg]['parameter_count']['mean'] / 1000 for alg in self.summary_data.keys()] + [self.baseline_data['parameter_count'] / 1000]
        fitnesses = [self.summary_data[alg]['best_fitness']['mean'] for alg in self.summary_data.keys()] + [self.baseline_data['best_fitness']]
        runtimes = [self.summary_data[alg]['runtime']['mean'] / 3600 for alg in self.summary_data.keys()] + [self.baseline_data.get('runtime', 0) / 3600]
        
        colors = self._get_algorithm_colors(algorithms)
        
        # Accuracy comparison
        bars1 = ax1.bar(algorithms, accuracies, color=colors, alpha=0.8)
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Test Accuracy: Metaheuristics vs Random Search')
        ax1.tick_params(axis='x', rotation=45)
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Parameter efficiency scatter
        for i, (alg, acc, params) in enumerate(zip(algorithms, accuracies, param_counts)):
            marker = 's' if 'Random' in alg else 'o'  # Square for baseline, circle for others
            size = 200 if 'Random' in alg else 150
            ax2.scatter(params, acc, s=size, c=colors[i], label=alg, alpha=0.8, marker=marker)
        ax2.set_xlabel('Parameter Count (thousands)')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Parameter Efficiency Comparison')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Fitness comparison
        bars3 = ax3.bar(algorithms, fitnesses, color=colors, alpha=0.8)
        ax3.set_ylabel('Fitness Score')
        ax3.set_title('Fitness Score Comparison')
        ax3.tick_params(axis='x', rotation=45)
        for bar, fit in zip(bars3, fitnesses):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{fit:.3f}', ha='center', va='bottom')
        
        # Runtime comparison
        bars4 = ax4.bar(algorithms, runtimes, color=colors, alpha=0.8)
        ax4.set_ylabel('Runtime (hours)')
        ax4.set_title('Runtime Comparison')
        ax4.tick_params(axis='x', rotation=45)
        for bar, runtime in zip(bars4, runtimes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{runtime:.1f}h', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Baseline comparison plot saved")
    
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
        
        # Add baseline if available
        if self.baseline_data:
            baseline_name = self.baseline_data.get('algorithm', 'Random Search')
            acc = self.baseline_data['test_accuracy']
            param = self.baseline_data['parameter_count'] / 1000
            fitness = self.baseline_data['best_fitness']
            runtime = self.baseline_data.get('runtime', 0) / 3600
            
            latex_table += f"{baseline_name} & {acc:.1f} & "
            latex_table += f"{param:.0f}k & {fitness:.3f} & {runtime:.1f} \\\\\n"
        
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
                                'runtime', 'architecture', 'individual', 'progress', 'baseline', 'all'],
                        default=['all'], help='Which plots to generate')
    parser.add_argument('--include_baseline', action='store_true',
                        help='Include baseline comparisons in plots (will be detected automatically if baseline files exist)')
    
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
        if 'progress' in args.plots:
            visualizer.plot_fitness_evaluation_progress()
        if 'baseline' in args.plots and visualizer.baseline_data:
            visualizer.plot_baseline_comparison()
    
    print("Visualization complete!")

if __name__ == "__main__":
    main()