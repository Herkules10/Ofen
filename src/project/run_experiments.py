#!/usr/bin/env python3
"""
Main experiment runner for Neural Architecture Search using metaheuristics.
Runs all algorithms (GA, EGA, PSO, SA, ASA) and compares their performance.
"""

import os
import time
import argparse
import pickle
from typing import Dict, List
import torch
import numpy as np

# Import algorithm implementations
from genetic_algorithm import run_genetic_algorithm
from particle_swarm_optimization import run_particle_swarm_optimization
from simulated_annealing import run_simulated_annealing
from training_utils import DatasetLoader

class ExperimentSuite:
    """Runs and manages multiple NAS experiments."""
    
    def __init__(self, 
                 dataset_name: str = "cifar10",
                 output_dir: str = "results",
                 num_runs: int = 1):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.num_runs = num_runs
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Algorithm configurations
        self.algorithms = {
            'GA': {
                'function': run_genetic_algorithm,
                'params': {
                    'algorithm_type': 'GA',
                    'population_size': 30,
                    'num_generations': 40,
                    'crossover_prob': 0.7,
                    'mutation_add_prob': 0.15,
                    'mutation_del_prob': 0.1,
                    'mutation_param_prob': 0.2,
                    'tournament_size': 3,
                    'max_layers': 8
                }
            },
            'EGA': {
                'function': run_genetic_algorithm,
                'params': {
                    'algorithm_type': 'EGA',
                    'population_size': 30,
                    'num_generations': 40,
                    'crossover_prob': 0.6,
                    'mutation_add_prob': 0.15,
                    'mutation_del_prob': 0.1,
                    'mutation_param_prob': 0.2,
                    'tournament_size': 3,
                    'elite_size': 5,
                    'max_layers': 8
                }
            },
            'PSO': {
                'function': run_particle_swarm_optimization,
                'params': {
                    'swarm_size': 25,
                    'num_iterations': 80,
                    'w': 0.7,
                    'c1': 2.0,
                    'c2': 2.0,
                    'max_layers': 8
                }
            },
            'SA': {
                'function': run_simulated_annealing,
                'params': {
                    'algorithm_type': 'SA',
                    'initial_temperature': 100.0,
                    'cooling_rate': 0.95,
                    'min_temperature': 0.01,
                    'max_iterations': 800,
                    'max_layers': 8
                }
            },
            'ASA': {
                'function': run_simulated_annealing,
                'params': {
                    'algorithm_type': 'ASA',
                    'initial_temperature': 100.0,
                    'cooling_rate': 0.95,
                    'min_temperature': 0.01,
                    'max_iterations': 800,
                    'reheat_threshold': 100,
                    'reheat_factor': 2.0,
                    'max_layers': 8
                }
            }
        }
        
        # Results storage
        self.results = {}
    
    def run_all_experiments(self):
        """Run all algorithms multiple times and collect results."""
        print(f"🚀 Starting experiment suite on {self.dataset_name.upper()}")
        print(f"   📊 Number of runs per algorithm: {self.num_runs}")
        print(f"   📁 Output directory: {self.output_dir}")
        print(f"   🖥️  GPU available: {torch.cuda.is_available()}")
        print(f"   🧮 PyTorch version: {torch.__version__}")
        print("="*80)
        
        total_experiments = len(self.algorithms) * self.num_runs
        completed_experiments = 0
        experiment_start_time = time.time()
        
        for algorithm_name, config in self.algorithms.items():
            print(f"\n{'🧬' if 'GA' in algorithm_name else '🦋' if 'PSO' in algorithm_name else '🌡️'} {algorithm_name} {'='*60}")
            print(f"   📋 Parameters: {config['params']}")
            
            algorithm_results = []
            algorithm_start_time = time.time()
            
            for run_id in range(self.num_runs):
                print(f"\n--- 🔄 Run {run_id + 1}/{self.num_runs} ---")
                run_start_time = time.time()
                
                try:
                    # Set random seeds for reproducibility
                    torch.manual_seed(42 + run_id)
                    np.random.seed(42 + run_id)
                    print(f"   🎲 Random seed set to: {42 + run_id}")
                    
                    # Detect and set device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    # Run algorithm
                    start_time = time.time()
                    result = config['function'](
                        dataset_name=self.dataset_name,
                        device=device,
                        **config['params']
                    )
                    end_time = time.time()
                    
                    # Add timing information
                    result['total_runtime'] = end_time - start_time
                    result['run_id'] = run_id
                    
                    algorithm_results.append(result)
                    completed_experiments += 1
                    
                    # Save individual result
                    filename = f"{algorithm_name}_{self.dataset_name}_run_{run_id}.pkl"
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, 'wb') as f:
                        pickle.dump(result, f)
                    
                    run_time = time.time() - run_start_time
                    total_elapsed = time.time() - experiment_start_time
                    eta = (total_elapsed / completed_experiments) * (total_experiments - completed_experiments)
                    
                    print(f"   ✅ Run {run_id + 1} completed in {run_time:.1f}s")
                    print(f"      🏆 Best fitness: {result['best_fitness']:.4f}")
                    print(f"      🎯 Test accuracy: {result['test_accuracy']:.2f}%")
                    print(f"      📊 Parameters: {result['parameter_count']:,}")
                    print(f"      📈 Progress: {completed_experiments}/{total_experiments} | ETA: {eta/60:.1f}min")
                    
                except Exception as e:
                    print(f"   ❌ Run {run_id + 1} failed: {e}")
                    continue
            
            algorithm_time = time.time() - algorithm_start_time
            self.results[algorithm_name] = algorithm_results
            
            # Save aggregated results for this algorithm
            agg_filename = f"{algorithm_name}_{self.dataset_name}_all_runs.pkl"
            agg_filepath = os.path.join(self.output_dir, agg_filename)
            with open(agg_filepath, 'wb') as f:
                pickle.dump(algorithm_results, f)
            
            success_rate = len(algorithm_results) / self.num_runs
            print(f"\n✅ {algorithm_name} completed in {algorithm_time/60:.1f}min: {len(algorithm_results)}/{self.num_runs} successful runs (success rate: {success_rate:.1%})")
        
        total_time = time.time() - experiment_start_time
        print(f"\n🏁 All experiments completed in {total_time/3600:.1f} hours!")
    
    def generate_summary(self):
        """Generate summary statistics for all algorithms."""
        print("\n" + "🏆"*80)
        print("                           EXPERIMENT SUMMARY")
        print("🏆"*80)
        
        summary_data = {}
        
        for algorithm_name, results in self.results.items():
            if not results:
                print(f"❌ {algorithm_name}: No successful runs")
                continue
            
            # Extract metrics
            test_accuracies = [r['test_accuracy'] for r in results]
            best_fitnesses = [r['best_fitness'] for r in results]
            param_counts = [r['parameter_count'] for r in results]
            runtimes = [r['total_runtime'] for r in results]
            
            # Calculate statistics
            stats = {
                'num_runs': len(results),
                'test_accuracy': {
                    'mean': np.mean(test_accuracies),
                    'std': np.std(test_accuracies),
                    'min': np.min(test_accuracies),
                    'max': np.max(test_accuracies)
                },
                'best_fitness': {
                    'mean': np.mean(best_fitnesses),
                    'std': np.std(best_fitnesses),
                    'min': np.min(best_fitnesses),
                    'max': np.max(best_fitnesses)
                },
                'parameter_count': {
                    'mean': np.mean(param_counts),
                    'std': np.std(param_counts),
                    'min': np.min(param_counts),
                    'max': np.max(param_counts)
                },
                'runtime': {
                    'mean': np.mean(runtimes),
                    'std': np.std(runtimes),
                    'min': np.min(runtimes),
                    'max': np.max(runtimes)
                }
            }
            
            summary_data[algorithm_name] = stats
            
            # Print formatted results
            emoji = "🧬" if "GA" in algorithm_name else "🦋" if "PSO" in algorithm_name else "🌡️"
            print(f"\n{emoji} {algorithm_name} Results ({len(results)} runs):")
            print(f"   🎯 Test Accuracy: {stats['test_accuracy']['mean']:.2f}% ± {stats['test_accuracy']['std']:.2f}% (range: {stats['test_accuracy']['min']:.2f}%-{stats['test_accuracy']['max']:.2f}%)")
            print(f"   🏆 Best Fitness:  {stats['best_fitness']['mean']:.4f} ± {stats['best_fitness']['std']:.4f} (range: {stats['best_fitness']['min']:.4f}-{stats['best_fitness']['max']:.4f})")
            print(f"   📊 Parameters:    {stats['parameter_count']['mean']:,.0f} ± {stats['parameter_count']['std']:,.0f} (range: {stats['parameter_count']['min']:,.0f}-{stats['parameter_count']['max']:,.0f})")
            print(f"   ⏱️  Runtime:       {stats['runtime']['mean']/60:.1f}min ± {stats['runtime']['std']/60:.1f}min (range: {stats['runtime']['min']/60:.1f}-{stats['runtime']['max']/60:.1f}min)")
        
        # Find best performing algorithm
        if summary_data:
            best_algo = max(summary_data.keys(), 
                           key=lambda x: summary_data[x]['test_accuracy']['mean'])
            best_acc = summary_data[best_algo]['test_accuracy']['mean']
            
            print(f"\n🥇 Best Performing Algorithm: {best_algo} with {best_acc:.2f}% average test accuracy")
            
            # Efficiency analysis (accuracy per parameter)
            print(f"\n📈 Efficiency Analysis (Accuracy/Million Parameters):")
            for algo_name, stats in summary_data.items():
                efficiency = stats['test_accuracy']['mean'] / (stats['parameter_count']['mean'] / 1_000_000)
                emoji = "🧬" if "GA" in algo_name else "🦋" if "PSO" in algo_name else "🌡️"
                print(f"   {emoji} {algo_name}: {efficiency:.2f}")
        
        return summary_data
    
    def save_summary(self, summary_data: Dict):
        """Save summary to file."""
        summary_filename = f"experiment_summary_{self.dataset_name}.pkl"
        summary_filepath = os.path.join(self.output_dir, summary_filename)
        
        with open(summary_filepath, 'wb') as f:
            pickle.dump(summary_data, f)
        
        print(f"\n💾 Summary saved to: {summary_filepath}")
        
        # Create ranking
        if len(summary_data) > 1:
            print("\n" + "-"*40)
            print("ALGORITHM RANKING")
            print("-"*40)
            
            # Rank by test accuracy
            acc_ranking = sorted(summary_data.items(), 
                               key=lambda x: x[1]['test_accuracy']['mean'], 
                               reverse=True)
            
            print("\nBy Test Accuracy:")
            for i, (alg, stats) in enumerate(acc_ranking, 1):
                print(f"{i}. {alg}: {stats['test_accuracy']['mean']:.2f}%")
            
            # Rank by parameter efficiency (accuracy / parameters)
            eff_ranking = sorted(summary_data.items(),
                               key=lambda x: x[1]['test_accuracy']['mean'] / (x[1]['parameter_count']['mean'] / 1000),
                               reverse=True)
            
            print("\nBy Parameter Efficiency (Accuracy/1k Parameters):")
            for i, (alg, stats) in enumerate(eff_ranking, 1):
                efficiency = stats['test_accuracy']['mean'] / (stats['parameter_count']['mean'] / 1000)
                print(f"{i}. {alg}: {efficiency:.3f}")
        
        return summary_data

def run_baseline_comparison(device: str = None):
    """Run baseline random search for comparison."""
    print("\n" + "="*20 + " Running Baseline " + "="*20)
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    from architecture_base import ArchitectureGenerator
    from training_utils import FitnessEvaluator, DatasetLoader, NetworkTrainer
    
    # Load dataset
    train_loader, val_loader, test_loader = DatasetLoader.load_cifar10()
    input_shape = (3, 32, 32)
    num_classes = 10
    
    # Setup with device
    fitness_evaluator = FitnessEvaluator(train_loader, val_loader, device=device)
    generator = ArchitectureGenerator(input_shape, num_classes, max_layers=8)
    trainer = NetworkTrainer(device=device)
    
    # Random search
    num_random_samples = 50
    best_architecture = None
    best_fitness = float('-inf')
    best_test_accuracy = 0
    
    print(f"Evaluating {num_random_samples} random architectures...")
    
    for i in range(num_random_samples):
        if i % 10 == 0:
            print(f"  Random sample {i + 1}/{num_random_samples}")
        
        # Generate random architecture
        architecture = generator.generate_random_architecture()
        
        # Evaluate fitness
        results = fitness_evaluator.evaluate_fitness(architecture)
        fitness = results['fitness']
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_architecture = architecture
    
    # Test best architecture
    test_results = trainer.train_and_evaluate(
        best_architecture, train_loader, test_loader, epochs=20
    )
    
    print(f"\nRandom Search Results:")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.2f}%")
    print(f"Parameter count: {test_results['parameter_count']:,}")
    print(f"Architecture layers: {len(best_architecture.layers)}")
    
    return {
        'algorithm': 'Random',
        'best_fitness': best_fitness,
        'test_accuracy': test_results['accuracy'],
        'parameter_count': test_results['parameter_count'],
        'best_architecture': best_architecture
    }

def main():
    parser = argparse.ArgumentParser(description='Neural Architecture Search Experiments')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'],
                        help='Dataset to use')
    parser.add_argument('--output_dir', default='results',
                        help='Output directory for results')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs per algorithm')
    parser.add_argument('--algorithms', nargs='+', 
                        choices=['GA', 'EGA', 'PSO', 'SA', 'ASA', 'all'],
                        default=['all'], help='Algorithms to run')
    parser.add_argument('--include_baseline', action='store_true',
                        help='Include random search baseline')
    parser.add_argument('--quick', action='store_true',
                        help='Run with reduced parameters for quick testing')
    
    args = parser.parse_args()
    
    # Handle algorithm selection
    if 'all' in args.algorithms:
        selected_algorithms = ['GA', 'EGA', 'PSO', 'SA', 'ASA']
    else:
        selected_algorithms = args.algorithms
    
    # Create experiment suite
    experiment = ExperimentSuite(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        num_runs=args.num_runs
    )
    
    # Filter algorithms based on selection
    experiment.algorithms = {k: v for k, v in experiment.algorithms.items() 
                           if k in selected_algorithms}
    
    # Adjust parameters for quick testing
    if args.quick:
        print("Quick mode enabled - reducing parameters for faster testing")
        for alg_config in experiment.algorithms.values():
            params = alg_config['params']
            if 'num_generations' in params:
                params['num_generations'] = min(10, params['num_generations'])
            if 'num_iterations' in params:
                params['num_iterations'] = min(20, params['num_iterations'])
            if 'max_iterations' in params:
                params['max_iterations'] = min(100, params['max_iterations'])
            if 'population_size' in params:
                params['population_size'] = min(10, params['population_size'])
            if 'swarm_size' in params:
                params['swarm_size'] = min(10, params['swarm_size'])
    
    try:
        # Run experiments
        experiment.run_all_experiments()
        
        # Generate summary
        summary = experiment.generate_summary()
        
        # Run baseline if requested
        if args.include_baseline:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            baseline_result = run_baseline_comparison(device=device)
            
            # Save baseline result
            baseline_filename = f"baseline_{args.dataset}.pkl"
            baseline_filepath = os.path.join(args.output_dir, baseline_filename)
            with open(baseline_filepath, 'wb') as f:
                pickle.dump(baseline_result, f)
        
        # Create final comparison table
        print("\n" + "="*80)
        print("FINAL COMPARISON TABLE")
        print("="*80)
        print(f"{'Algorithm':<12} {'Accuracy (%)':<14} {'Parameters':<12} {'Runtime (s)':<12} {'Fitness':<10}")
        print("-" * 80)
        
        for alg_name, stats in summary.items():
            print(f"{alg_name:<12} "
                  f"{stats['test_accuracy']['mean']:.2f} ± {stats['test_accuracy']['std']:.2f}   "
                  f"{stats['parameter_count']['mean']:.0f:<12} "
                  f"{stats['runtime']['mean']:.1f} ± {stats['runtime']['std']:.1f}   "
                  f"{stats['best_fitness']['mean']:.4f}")
        
        if args.include_baseline:
            print(f"{'Random':<12} "
                  f"{baseline_result['test_accuracy']:.2f}          "
                  f"{baseline_result['parameter_count']:.0f:<12} "
                  f"{'N/A':<12} "
                  f"{baseline_result['best_fitness']:.4f}")
        
        print("\n" + "="*80)
        print("Experiment completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\nExperiment failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\nPartial results (if any) saved to: {args.output_dir}")

if __name__ == "__main__":
    main()