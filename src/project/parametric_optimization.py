"""
Parametric Optimization for Neural Architecture Search Algorithms
Performs grid search over parameter spaces to find optimal configurations for GA, PSO, and SA.
"""

import os
import sys
import time
import json
import pickle
import itertools
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from pathlib import Path
import argparse

# Add project directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithms_dir = os.path.join(current_dir, 'algorithms')
sys.path.insert(0, current_dir)
sys.path.insert(0, algorithms_dir)

# Dynamic imports to handle path issues
def import_algorithm_functions():
    """Import algorithm functions with proper path handling."""
    try:
        # Try relative imports first
        from algorithms.genetic_algorithm import run_genetic_algorithm
        from algorithms.particle_swarm_optimization import run_particle_swarm_optimization
        from algorithms.simulated_annealing import run_simulated_annealing
        return run_genetic_algorithm, run_particle_swarm_optimization, run_simulated_annealing
    except ImportError:
        # Fallback to direct imports from algorithms directory
        import genetic_algorithm
        import particle_swarm_optimization
        import simulated_annealing
        return (genetic_algorithm.run_genetic_algorithm, 
                particle_swarm_optimization.run_particle_swarm_optimization,
                simulated_annealing.run_simulated_annealing)

# Import functions
run_genetic_algorithm, run_particle_swarm_optimization, run_simulated_annealing = import_algorithm_functions()

from training_utils import DatasetLoader
from debug_utils import create_logger, SystemMonitor

class ParametricOptimizer:
    """Performs grid search optimization over algorithm parameters."""
    
    def __init__(self, 
                 dataset_name: str = "cifar10",
                 output_dir: str = "parametric_results",
                 n_runs: int = 3,
                 device: str = None):
        
        self.dataset_name = dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_runs = n_runs
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self.logger = create_logger("ParametricOptimizer", level="INFO")
        
        # Define parameter spaces
        self.parameter_spaces = self._define_parameter_spaces()
        
        # Results storage
        self.results = {}
        
        self.logger.info(f"Parametric Optimizer initialized")
        self.logger.info(f"Dataset: {dataset_name}, Device: {self.device}")
        self.logger.info(f"Output directory: {output_dir}")
        
    def _define_parameter_spaces(self) -> Dict[str, Dict]:
        """Define parameter spaces for each algorithm."""
        return {
            'GA': {
                'population_size': [40],  # Fixed
                'crossover_prob': [0.3, 0.5, 0.7],
                'mutation_add_prob': [0.1, 0.2, 0.3],
                'mutation_del_prob': [0.1, 0.2, 0.3],
                'mutation_param_prob': [0.1, 0.2, 0.3],
                'tournament_size': [2, 3, 5],
                'num_generations': [30],  # Fixed
                'max_layers': [8]  # Fixed for consistency
            },
            'PSO': {
                'swarm_size': [40],  # Fixed (renamed from n_particles)
                'w': [0.4, 0.7, 0.9],  # Inertia weight
                'c1': [1.5, 2.0, 2.5],  # Cognitive coefficient
                'c2': [1.5, 2.0, 2.5],  # Social coefficient
                'num_iterations': [30],  # Fixed
                'max_layers': [8]  # Fixed for consistency
            },
            'SA': {
                'initial_temperature': [10, 100, 500],  # T₀
                'cooling_rate': [0.85, 0.9, 0.95],
                'max_iterations': [1200],  # Fixed
                'max_layers': [8]  # Fixed for consistency
            }
        }
    
    def generate_parameter_combinations(self, algorithm: str) -> List[Dict]:
        """Generate all parameter combinations for an algorithm."""
        if algorithm not in self.parameter_spaces:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        param_space = self.parameter_spaces[algorithm]
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def run_single_experiment(self, 
                            algorithm: str, 
                            params: Dict, 
                            run_id: int) -> Dict:
        """Run a single experiment with given parameters."""
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + run_id)
        np.random.seed(42 + run_id)
        
        start_time = time.time()
        
        try:
            if algorithm == 'GA':
                result = run_genetic_algorithm(
                    dataset_name=self.dataset_name,
                    algorithm_type="GA",
                    **params
                )
            elif algorithm == 'PSO':
                result = run_particle_swarm_optimization(
                    dataset_name=self.dataset_name,
                    **params
                )
            elif algorithm == 'SA':
                result = run_simulated_annealing(
                    dataset_name=self.dataset_name,
                    algorithm_type="SA",
                    **params
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            result['success'] = True
            result['error'] = None
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            result = {
                'success': False,
                'error': str(e),
                'best_fitness': -float('inf'),
                'test_accuracy': 0.0,
                'parameter_count': 0
            }
        
        result['runtime'] = time.time() - start_time
        result['run_id'] = run_id
        result['parameters'] = params.copy()
        
        return result
    
    def optimize_algorithm(self, algorithm: str) -> Dict:
        """Optimize parameters for a single algorithm."""
        self.logger.info(f"🔍 Starting parametric optimization for {algorithm}")
        
        combinations = self.generate_parameter_combinations(algorithm)
        total_experiments = len(combinations) * self.n_runs
        
        self.logger.info(f"   📊 Parameter combinations: {len(combinations)}")
        self.logger.info(f"   🔄 Runs per combination: {self.n_runs}")
        self.logger.info(f"   📈 Total experiments: {total_experiments}")
        
        algorithm_results = []
        experiment_count = 0
        
        for i, params in enumerate(combinations):
            self.logger.info(f"\n📋 Combination {i+1}/{len(combinations)}: {params}")
            
            combination_results = []
            combination_start = time.time()
            
            for run_id in range(self.n_runs):
                experiment_count += 1
                self.logger.info(f"   🔄 Run {run_id+1}/{self.n_runs} (Experiment {experiment_count}/{total_experiments})")
                
                # Print system status periodically
                if experiment_count % 5 == 1:
                    SystemMonitor.print_system_status()
                
                result = self.run_single_experiment(algorithm, params, run_id)
                combination_results.append(result)
                
                if result['success']:
                    self.logger.info(f"      ✅ Success: fitness={result['best_fitness']:.4f}, "
                                   f"accuracy={result['test_accuracy']:.2f}%, "
                                   f"time={result['runtime']/60:.1f}min")
                else:
                    self.logger.error(f"      ❌ Failed: {result['error']}")
            
            # Calculate combination statistics
            successful_runs = [r for r in combination_results if r['success']]
            
            if successful_runs:
                combination_stats = {
                    'parameters': params,
                    'num_successful_runs': len(successful_runs),
                    'avg_fitness': np.mean([r['best_fitness'] for r in successful_runs]),
                    'std_fitness': np.std([r['best_fitness'] for r in successful_runs]),
                    'avg_accuracy': np.mean([r['test_accuracy'] for r in successful_runs]),
                    'std_accuracy': np.std([r['test_accuracy'] for r in successful_runs]),
                    'avg_parameters': np.mean([r['parameter_count'] for r in successful_runs]),
                    'avg_runtime': np.mean([r['runtime'] for r in successful_runs]),
                    'individual_runs': combination_results,
                    'combination_runtime': time.time() - combination_start
                }
                
                self.logger.info(f"   📊 Combination Stats: avg_fitness={combination_stats['avg_fitness']:.4f} ± "
                               f"{combination_stats['std_fitness']:.4f}, "
                               f"avg_accuracy={combination_stats['avg_accuracy']:.2f}% ± "
                               f"{combination_stats['std_accuracy']:.2f}%")
            else:
                combination_stats = {
                    'parameters': params,
                    'num_successful_runs': 0,
                    'avg_fitness': -float('inf'),
                    'std_fitness': 0,
                    'avg_accuracy': 0,
                    'std_accuracy': 0,
                    'avg_parameters': 0,
                    'avg_runtime': 0,
                    'individual_runs': combination_results,
                    'combination_runtime': time.time() - combination_start
                }
                self.logger.warning(f"   ⚠️  No successful runs for this combination")
            
            algorithm_results.append(combination_stats)
            
            # Save intermediate results
            self._save_intermediate_results(algorithm, algorithm_results)
        
        # Find best parameter combination
        successful_combinations = [r for r in algorithm_results if r['num_successful_runs'] > 0]
        
        if successful_combinations:
            best_combination = max(successful_combinations, key=lambda x: x['avg_fitness'])
            
            optimization_summary = {
                'algorithm': algorithm,
                'total_combinations': len(combinations),
                'successful_combinations': len(successful_combinations),
                'best_parameters': best_combination['parameters'],
                'best_avg_fitness': best_combination['avg_fitness'],
                'best_avg_accuracy': best_combination['avg_accuracy'],
                'all_combinations': algorithm_results
            }
            
            self.logger.success(f"🏆 Best parameters for {algorithm}: {best_combination['parameters']}")
            self.logger.info(f"   🎯 Best avg fitness: {best_combination['avg_fitness']:.4f}")
            self.logger.info(f"   🎯 Best avg accuracy: {best_combination['avg_accuracy']:.2f}%")
        else:
            self.logger.error(f"❌ No successful combinations found for {algorithm}")
            optimization_summary = {
                'algorithm': algorithm,
                'total_combinations': len(combinations),
                'successful_combinations': 0,
                'best_parameters': None,
                'best_avg_fitness': -float('inf'),
                'best_avg_accuracy': 0,
                'all_combinations': algorithm_results
            }
        
        return optimization_summary
    
    def _save_intermediate_results(self, algorithm: str, results: List[Dict]):
        """Save intermediate results during optimization."""
        filepath = self.output_dir / f"{algorithm}_parametric_intermediate.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def optimize_all_algorithms(self) -> Dict[str, Dict]:
        """Optimize parameters for all algorithms."""
        self.logger.info("🚀 Starting comprehensive parametric optimization")
        self.logger.info(f"   📊 Algorithms: GA, PSO, SA")
        self.logger.info(f"   🖥️  Device: {self.device}")
        
        start_time = time.time()
        
        algorithms = ['GA', 'PSO', 'SA']
        all_results = {}
        
        for algorithm in algorithms:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🧠 Optimizing {algorithm}")
            self.logger.info(f"{'='*60}")
            
            algorithm_start = time.time()
            results = self.optimize_algorithm(algorithm)
            algorithm_time = time.time() - algorithm_start
            
            results['optimization_time'] = algorithm_time
            all_results[algorithm] = results
            
            # Save algorithm results
            self._save_algorithm_results(algorithm, results)
            
            self.logger.info(f"✅ {algorithm} optimization completed in {algorithm_time/3600:.1f} hours")
            
            # Clear GPU memory between algorithms
            if self.device == "cuda":
                torch.cuda.empty_cache()
                self.logger.info("🧹 GPU memory cleared")
        
        total_time = time.time() - start_time
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results, total_time)
        
        # Generate summary report
        self._generate_summary_report(all_results, total_time)
        
        self.logger.success(f"🎉 Parametric optimization completed in {total_time/3600:.1f} hours!")
        
        return all_results
    
    def _save_algorithm_results(self, algorithm: str, results: Dict):
        """Save results for a single algorithm."""
        filepath = self.output_dir / f"{algorithm}_parametric_results.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        # Also save as JSON for readability
        json_filepath = self.output_dir / f"{algorithm}_parametric_results.json"
        json_results = self._make_json_serializable(results)
        with open(json_filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _save_comprehensive_results(self, all_results: Dict, total_time: float):
        """Save comprehensive results for all algorithms."""
        comprehensive_results = {
            'optimization_summary': {
                'total_time': total_time,
                'algorithms_optimized': list(all_results.keys()),
                'dataset': self.dataset_name,
                'device': self.device,
                'runs_per_combination': self.n_runs
            },
            'algorithm_results': all_results
        }
        
        # Save as pickle
        filepath = self.output_dir / "comprehensive_parametric_results.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(comprehensive_results, f)
        
        # Save as JSON
        json_filepath = self.output_dir / "comprehensive_parametric_results.json"
        json_results = self._make_json_serializable(comprehensive_results)
        with open(json_filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)
    
    def _generate_summary_report(self, all_results: Dict, total_time: float):
        """Generate a summary report of optimization results."""
        report_path = self.output_dir / "parametric_optimization_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("PARAMETRIC OPTIMIZATION SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Runs per combination: {self.n_runs}\n")
            f.write(f"Total optimization time: {total_time/3600:.1f} hours\n\n")
            
            for algorithm, results in all_results.items():
                f.write(f"\n{algorithm} OPTIMIZATION RESULTS\n")
                f.write("-" * 40 + "\n")
                
                f.write(f"Total combinations tested: {results['total_combinations']}\n")
                f.write(f"Successful combinations: {results['successful_combinations']}\n")
                f.write(f"Optimization time: {results['optimization_time']/3600:.1f} hours\n\n")
                
                if results['best_parameters']:
                    f.write("BEST PARAMETERS:\n")
                    for param, value in results['best_parameters'].items():
                        f.write(f"  {param}: {value}\n")
                    
                    f.write(f"\nBEST PERFORMANCE:\n")
                    f.write(f"  Average fitness: {results['best_avg_fitness']:.4f}\n")
                    f.write(f"  Average accuracy: {results['best_avg_accuracy']:.2f}%\n")
                else:
                    f.write("No successful parameter combinations found.\n")
                
                # Top 3 combinations
                successful_combos = [r for r in results['all_combinations'] 
                                   if r['num_successful_runs'] > 0]
                
                if len(successful_combos) > 1:
                    f.write(f"\nTOP 3 PARAMETER COMBINATIONS:\n")
                    sorted_combos = sorted(successful_combos, 
                                         key=lambda x: x['avg_fitness'], 
                                         reverse=True)
                    
                    for i, combo in enumerate(sorted_combos[:3]):
                        f.write(f"\n{i+1}. Fitness: {combo['avg_fitness']:.4f}, "
                               f"Accuracy: {combo['avg_accuracy']:.2f}%\n")
                        f.write(f"   Parameters: {combo['parameters']}\n")
        
        self.logger.info(f"📋 Summary report saved to: {report_path}")

def main():
    """Main function for parametric optimization."""
    parser = argparse.ArgumentParser(description='Parametric Optimization for NAS Algorithms')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist'],
                       help='Dataset to use for optimization')
    parser.add_argument('--output-dir', type=str, default='parametric_results',
                       help='Output directory for results')
    parser.add_argument('--n-runs', type=int, default=3,
                       help='Number of runs per parameter combination')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['GA', 'PSO', 'SA'],
                       default=['GA', 'PSO', 'SA'],
                       help='Algorithms to optimize')
    
    args = parser.parse_args()
    
    print("🚀 Neural Architecture Search - Parametric Optimization")
    print("="*60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔄 Runs per combination: {args.n_runs}")
    print(f"🧠 Algorithms: {', '.join(args.algorithms)}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Create optimizer
    optimizer = ParametricOptimizer(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        n_runs=args.n_runs,
        device=args.device
    )
    
    # Print parameter spaces
    print(f"\n📋 Parameter Spaces:")
    for algorithm in args.algorithms:
        if algorithm in optimizer.parameter_spaces:
            space = optimizer.parameter_spaces[algorithm]
            combinations = len(list(itertools.product(*space.values())))
            print(f"  {algorithm}: {combinations} combinations")
            for param, values in space.items():
                if len(values) > 1:  # Only show varying parameters
                    print(f"    {param}: {values}")
    
    total_experiments = sum(
        len(list(itertools.product(*optimizer.parameter_spaces[alg].values()))) 
        for alg in args.algorithms if alg in optimizer.parameter_spaces
    ) * args.n_runs
    
    print(f"\n📈 Total experiments: {total_experiments}")
    print(f"⏱️  Estimated time: {total_experiments * 0.5:.1f} - {total_experiments * 2:.1f} hours")
    
    # Run optimization
    try:
        if len(args.algorithms) == 1:
            results = optimizer.optimize_algorithm(args.algorithms[0])
            print(f"\n🏆 Optimization completed for {args.algorithms[0]}!")
        else:
            results = optimizer.optimize_all_algorithms()
            print(f"\n🎉 Comprehensive optimization completed!")
        
        print(f"📁 Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Optimization interrupted by user")
        print(f"📁 Partial results saved to: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
