import numpy as np
import random
import math
import copy
from typing import Tuple, Dict, List, Optional
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture_base import NetworkArchitecture, ArchitectureGenerator, LayerConfig, LayerType, ActivationType
from training_utils import FitnessEvaluator, ExperimentLogger

class NeighborhoodOperator:
    """Defines neighborhood operations for architecture modification."""
    
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, max_layers: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_layers = max_layers
        self.generator = ArchitectureGenerator(input_shape, num_classes, max_layers)
    
    def get_neighbor(self, architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Generate a neighbor architecture through random modification."""
        neighbor = architecture.copy()
        
        # Choose random operation
        operations = []
        
        # Always possible operations
        if len(neighbor.layers) < self.max_layers:
            operations.append('add_layer')
        if len(neighbor.layers) > 1:
            operations.append('remove_layer')
        if len(neighbor.layers) > 0:
            operations.extend(['modify_layer', 'swap_layers'])
        
        if not operations:
            # Fallback: add a layer
            operations = ['add_layer']
        
        operation = random.choice(operations)
        
        if operation == 'add_layer':
            return self._add_random_layer(neighbor)
        elif operation == 'remove_layer':
            return self._remove_random_layer(neighbor)
        elif operation == 'modify_layer':
            return self._modify_random_layer(neighbor)
        elif operation == 'swap_layers':
            return self._swap_random_layers(neighbor)
        
        return neighbor
    
    def _add_random_layer(self, architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Add a random layer at a random position."""
        if len(architecture.layers) >= self.max_layers:
            return architecture
        
        # Generate random layer
        new_layer = self.generator._generate_random_layer(architecture)
        if new_layer is None:
            return architecture
        
        # Try to add the layer
        if not architecture.add_layer(new_layer):
            return architecture
        
        return architecture
    
    def _remove_random_layer(self, architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Remove a random layer."""
        if len(architecture.layers) <= 1:
            return architecture
        
        layer_idx = random.randint(0, len(architecture.layers) - 1)
        architecture.remove_layer(layer_idx)
        return architecture
    
    def _modify_random_layer(self, architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Modify parameters of a random layer."""
        if len(architecture.layers) == 0:
            return architecture
        
        layer_idx = random.randint(0, len(architecture.layers) - 1)
        layer = architecture.layers[layer_idx]
        
        # Modify layer parameters based on type
        if layer.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
            if 'filters' in layer.params and random.random() < 0.4:
                layer.params['filters'] = random.choice([16, 32, 64, 128, 256])
            if 'kernel_size' in layer.params and random.random() < 0.3:
                layer.params['kernel_size'] = random.choice([3, 5, 7])
            if 'stride' in layer.params and random.random() < 0.2:
                layer.params['stride'] = random.choice([1, 2])
        
        elif layer.layer_type == LayerType.FC:
            if 'units' in layer.params and random.random() < 0.4:
                layer.params['units'] = random.choice([64, 128, 256, 512, 1024])
        
        elif layer.layer_type == LayerType.DROPOUT:
            if 'rate' in layer.params and random.random() < 0.5:
                layer.params['rate'] = random.uniform(0.1, 0.5)
        
        elif layer.layer_type == LayerType.MAXPOOL:
            if 'pool_size' in layer.params and random.random() < 0.3:
                layer.params['pool_size'] = random.choice([2, 3, 4])
        
        elif layer.layer_type == LayerType.ACTIVATION:
            if 'type' in layer.params and random.random() < 0.3:
                layer.params['type'] = random.choice(list(ActivationType))
        
        # Recalculate shapes after modification
        architecture._recalculate_shapes()
        return architecture
    
    def _swap_random_layers(self, architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Swap two adjacent layers if valid."""
        if len(architecture.layers) < 2:
            return architecture
        
        # Choose random adjacent pair
        idx1 = random.randint(0, len(architecture.layers) - 2)
        idx2 = idx1 + 1
        
        # Create new architecture with swapped layers
        new_arch = NetworkArchitecture(self.input_shape, self.num_classes)
        
        # Add layers in new order
        for i, layer in enumerate(architecture.layers):
            if i == idx1:
                # Skip this layer, will add idx2 layer instead
                continue
            elif i == idx2:
                # Add both layers in swapped order
                layer1 = architecture.layers[idx1].copy()
                layer2 = architecture.layers[idx2].copy()
                
                # Try to add layer2 first, then layer1
                if new_arch.add_layer(layer2):
                    if not new_arch.add_layer(layer1):
                        # If can't add layer1, remove layer2 and add layers in original order
                        new_arch.layers.pop()
                        new_arch.add_layer(layer1)
                        new_arch.add_layer(layer2)
                else:
                    # If can't add layer2, add in original order
                    new_arch.add_layer(layer1)
                    new_arch.add_layer(layer2)
            else:
                new_arch.add_layer(layer.copy())
        
        return new_arch if len(new_arch.layers) > 0 else architecture

class SimulatedAnnealing:
    """
    Simulated Annealing for Neural Architecture Search.
    
    Now optimized to use ParallelFitnessEvaluator with batch size 1 for better GPU efficiency,
    even though SA evaluates architectures sequentially.
    """
    
    def __init__(self,
                 fitness_evaluator: FitnessEvaluator,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.981748,
                 min_temperature: float = 0.01,
                 max_iterations: int = 1000,
                 max_layers: int = 10):
        
        self.fitness_evaluator = fitness_evaluator
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.max_layers = max_layers
        
        # Initialize components
        self.generator = ArchitectureGenerator(input_shape, num_classes, max_layers)
        self.neighborhood = NeighborhoodOperator(input_shape, num_classes, max_layers)
        self.logger = ExperimentLogger("SA")
        
        # Algorithm state
        self.current_architecture = None
        self.current_fitness = float('-inf')
        self.best_architecture = None
        self.best_fitness = float('-inf')
        self.temperature = initial_temperature
        
        # Statistics
        self.accepted_moves = 0
        self.rejected_moves = 0
    
    def run(self) -> Tuple[NetworkArchitecture, float]:
        """Run simulated annealing algorithm."""
        print(f"Starting Simulated Annealing")
        print(f"Initial temperature: {self.initial_temperature}")
        print(f"Cooling rate: {self.cooling_rate}")
        print(f"Max iterations: {self.max_iterations}")
        
        # Check if using parallel evaluator
        if hasattr(self.fitness_evaluator, 'evaluate_population_parallel'):
            print("Using ParallelFitnessEvaluator with batch size 1 for efficient GPU usage")
        
        # Initialize with random architecture
        self._initialize()
        
        iteration = 0
        temperature_updates = 0
        
        # Calculate logging frequency based on max_iterations
        # For short runs, log more frequently to capture convergence
        log_frequency = max(1, min(50, self.max_iterations // 10))
        print_frequency = max(1, min(50, self.max_iterations // 5))
        
        while (iteration < self.max_iterations and 
               self.temperature > self.min_temperature):
            
            if iteration % print_frequency == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"  Temperature: {self.temperature:.4f}")
                print(f"  Current fitness: {self.current_fitness:.4f}")
                print(f"  Best fitness: {self.best_fitness:.4f}")
                print(f"  Accepted/Rejected: {self.accepted_moves}/{self.rejected_moves}")
            
            # Generate neighbor
            neighbor_architecture = self.neighborhood.get_neighbor(self.current_architecture)
            
            # Evaluate neighbor using parallel evaluator
            neighbor_results = self._evaluate_architecture(neighbor_architecture)
            neighbor_fitness = neighbor_results['fitness']
            
            # Decide whether to accept the neighbor
            if self._accept_neighbor(neighbor_fitness):
                self.current_architecture = neighbor_architecture
                self.current_fitness = neighbor_fitness
                self.accepted_moves += 1
                
                # Update best if necessary
                if neighbor_fitness > self.best_fitness:
                    self.best_fitness = neighbor_fitness
                    self.best_architecture = neighbor_architecture.copy()
            else:
                self.rejected_moves += 1
            
            # Cool down temperature
            if iteration % 10 == 0:  # Update temperature every 10 iterations
                self.temperature *= self.cooling_rate
                temperature_updates += 1
            
            # Log progress periodically - adaptive frequency for convergence tracking
            if iteration % log_frequency == 0:
                fitness_list = [self.current_fitness]  # Single fitness for SA
                self.logger.log_generation(iteration, fitness_list,
                                         self.best_architecture, self.best_fitness)
            
            iteration += 1
        
        # Final logging to ensure the last state is captured
        fitness_list = [self.current_fitness]
        self.logger.log_generation(iteration, fitness_list,
                                 self.best_architecture, self.best_fitness)
        
        print(f"\nSimulated Annealing completed:")
        print(f"Total iterations: {iteration}")
        print(f"Final temperature: {self.temperature:.6f}")
        print(f"Accepted moves: {self.accepted_moves}")
        print(f"Rejected moves: {self.rejected_moves}")
        print(f"Acceptance rate: {self.accepted_moves/(self.accepted_moves + self.rejected_moves):.3f}")
        
        # Show evaluation count if available
        if hasattr(self.fitness_evaluator, 'logger') and self.fitness_evaluator.logger:
            eval_count = len(self.fitness_evaluator.logger.evaluation_log)
            print(f"Total fitness evaluations logged: {eval_count}")
            if eval_count != iteration + 1:  # +1 for initial evaluation
                print(f"  Note: Discrepancy between iterations ({iteration+1}) and evaluations ({eval_count}) may indicate failed network builds")
                
                # Show breakdown of skipped evaluations if available
                failed_builds = getattr(self.fitness_evaluator, 'failed_builds', 0)
                skipped_large = getattr(self.fitness_evaluator, 'skipped_large_networks', 0)
                
                print(f"  Debug: failed_builds={failed_builds}, skipped_large={skipped_large}")
                
                if failed_builds > 0:
                    print(f"  Failed network builds: {failed_builds}")
                        
                if skipped_large > 0:
                    print(f"  Networks skipped (too large): {skipped_large}")
                        
                total_skipped = failed_builds + skipped_large
                    
                if total_skipped > 0:
                    expected_evals = iteration + 1
                    actual_evals = eval_count
                    missing_evals = expected_evals - actual_evals
                    print(f"  Expected evaluations: {expected_evals}, Actual: {actual_evals}, Missing: {missing_evals}")
                    print(f"  Known skipped: {total_skipped}, Unexplained missing: {missing_evals - total_skipped}")
        
        return self.best_architecture, self.best_fitness
    
    def _initialize(self):
        """Initialize algorithm with random architecture."""
        print("Initializing with random architecture...")
        self.current_architecture = self.generator.generate_random_architecture()
        
        # Evaluate initial architecture using parallel evaluator
        results = self._evaluate_architecture(self.current_architecture)
        self.current_fitness = results['fitness']
        
        # Set as best initially
        self.best_architecture = self.current_architecture.copy()
        self.best_fitness = self.current_fitness
        
        # Log initial state
        fitness_list = [self.current_fitness]
        self.logger.log_generation(0, fitness_list,
                                 self.best_architecture, self.best_fitness)
        
        print(f"Initial fitness: {self.current_fitness:.4f}")
        print(f"Initial layers: {len(self.current_architecture.layers)}")
        print(f"Initial parameters: {self.current_architecture.get_parameter_count():,}")
    
    def _evaluate_architecture(self, architecture: NetworkArchitecture) -> Dict[str, float]:
        """Evaluate a single architecture using parallel evaluator with batch size 1."""
        if hasattr(self.fitness_evaluator, 'evaluate_population_parallel'):
            # Use ParallelFitnessEvaluator with batch size 1 for better GPU efficiency
            results_list = self.fitness_evaluator.evaluate_population_parallel([architecture])
            return results_list[0]
        else:
            # Fallback to standard evaluation
            return self.fitness_evaluator.evaluate_fitness(architecture)

    def _accept_neighbor(self, neighbor_fitness: float) -> bool:
        """Decide whether to accept a neighbor solution."""
        if neighbor_fitness > self.current_fitness:
            # Always accept better solutions
            return True
        else:
            # Accept worse solutions with probability based on temperature
            delta = neighbor_fitness - self.current_fitness
            probability = math.exp(delta / self.temperature) if self.temperature > 0 else 0
            return random.random() < probability

class AdaptiveSimulatedAnnealing(SimulatedAnnealing):
    """Enhanced SA with adaptive temperature control."""
    
    def __init__(self, 
                 fitness_evaluator: FitnessEvaluator,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 adaptive_cooling: bool = True,
                 reheat_threshold: int = 100,
                 reheat_factor: float = 2.0,
                 **kwargs):
        
        super().__init__(fitness_evaluator, input_shape, num_classes, **kwargs)
        self.adaptive_cooling = adaptive_cooling
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor
        self.logger = ExperimentLogger("ASA")
        
        # Adaptive parameters
        self.no_improvement_count = 0
        self.last_best_fitness = float('-inf')
        self.acceptance_history = []
        self.acceptance_window = 50
    
    def run(self) -> Tuple[NetworkArchitecture, float]:
        """Run adaptive simulated annealing."""
        print(f"Starting Adaptive Simulated Annealing")
        
        # Initialize
        self._initialize()
        
        iteration = 0
        
        # Calculate logging frequency based on max_iterations
        # For short runs, log more frequently to capture convergence
        log_frequency = max(1, min(50, self.max_iterations // 10))
        print_frequency = max(1, min(50, self.max_iterations // 5))
        
        while (iteration < self.max_iterations and 
               self.temperature > self.min_temperature):
            
            if iteration % print_frequency == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"  Temperature: {self.temperature:.4f}")
                print(f"  Current fitness: {self.current_fitness:.4f}")
                print(f"  Best fitness: {self.best_fitness:.4f}")
                print(f"  No improvement count: {self.no_improvement_count}")
                
                if len(self.acceptance_history) > 0:
                    recent_acceptance = np.mean(self.acceptance_history[-self.acceptance_window:])
                    print(f"  Recent acceptance rate: {recent_acceptance:.3f}")
            
            # Generate and evaluate neighbor
            neighbor_architecture = self.neighborhood.get_neighbor(self.current_architecture)
            neighbor_results = self._evaluate_architecture(neighbor_architecture)
            neighbor_fitness = neighbor_results['fitness']
            
            # Accept/reject decision
            accepted = self._accept_neighbor(neighbor_fitness)
            self.acceptance_history.append(1 if accepted else 0)
            
            if accepted:
                self.current_architecture = neighbor_architecture
                self.current_fitness = neighbor_fitness
                self.accepted_moves += 1
                
                # Check for improvement
                if neighbor_fitness > self.best_fitness:
                    self.best_fitness = neighbor_fitness
                    self.best_architecture = neighbor_architecture.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            else:
                self.rejected_moves += 1
                self.no_improvement_count += 1
            
            # Adaptive temperature control
            if self.adaptive_cooling:
                self._adaptive_temperature_update(iteration)
            else:
                if iteration % 10 == 0:
                    self.temperature *= self.cooling_rate
            
            # Reheating mechanism
            if (self.no_improvement_count >= self.reheat_threshold and 
                self.temperature < self.initial_temperature / 10):
                print(f"  Reheating: {self.temperature:.4f} -> {self.temperature * self.reheat_factor:.4f}")
                self.temperature *= self.reheat_factor
                self.no_improvement_count = 0
            
            # Logging - adaptive frequency for convergence tracking
            if iteration % log_frequency == 0:
                fitness_list = [self.current_fitness]
                self.logger.log_generation(iteration, fitness_list,
                                         self.best_architecture, self.best_fitness)
            
            iteration += 1
        
        # Final logging to ensure the last state is captured
        fitness_list = [self.current_fitness]
        self.logger.log_generation(iteration, fitness_list,
                                 self.best_architecture, self.best_fitness)
        
        return self.best_architecture, self.best_fitness
    
    def _adaptive_temperature_update(self, iteration: int):
        """Adaptively update temperature based on acceptance rate."""
        if len(self.acceptance_history) < self.acceptance_window:
            # Standard cooling in early iterations
            if iteration % 10 == 0:
                self.temperature *= self.cooling_rate
            return
        
        # Calculate recent acceptance rate
        recent_acceptance = np.mean(self.acceptance_history[-self.acceptance_window:])
        
        '''""
        # Adjust cooling rate based on acceptance
        if recent_acceptance > 0.8:  # Too many acceptances, cool faster
            cooling_factor = self.cooling_rate * 0.95
        elif recent_acceptance < 0.2:  # Too few acceptances, cool slower
            cooling_factor = self.cooling_rate * 1.05
        else:
            cooling_factor = self.cooling_rate
        '''
        # Clamp cooling factor
        cooling_factor = max(0.85, min(0.99, cooling_factor))
        
        if iteration % 10 == 0:
            self.temperature *= cooling_factor

def run_simulated_annealing(dataset_name: str = "cifar10", 
                           algorithm_type: str = "SA",
                           train_split: float = 1.0,
                           **kwargs) -> Dict:
    """Run simulated annealing experiment."""
    
    # Load dataset
    from training_utils import DatasetLoader
    
    if dataset_name.lower() == "cifar10":
        train_loader, val_loader, test_loader = DatasetLoader.load_cifar10(train_split=train_split)
        input_shape = (3, 32, 32)
        num_classes = 10
    elif dataset_name.lower() == "mnist":
        train_loader, val_loader, test_loader = DatasetLoader.load_mnist(train_split=train_split)
        input_shape = (1, 28, 28)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Default parameters
    default_params = {
        'initial_temperature': 100.0,
        'cooling_rate': 0.981748,
        'min_temperature': 0.01,
        'max_iterations': 1000,
        'max_layers': 10
    }
    default_params.update(kwargs)
    
    # Create algorithm first to get logger
    if algorithm_type.upper() == "ASA":
        algorithm = AdaptiveSimulatedAnnealing(
            None, input_shape, num_classes,  # Pass None first, will set later
            adaptive_cooling=True,
            reheat_threshold=100,
            reheat_factor=2.0,
            **default_params
        )
    else:
        algorithm = SimulatedAnnealing(
            None, input_shape, num_classes,  # Pass None first, will set later
            **default_params
        )
    
    # Setup fitness evaluator with logger
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use ParallelFitnessEvaluator for better GPU efficiency, even with batch size 1
    from training_utils import ParallelFitnessEvaluator
    print(f"Using ParallelFitnessEvaluator with batch size 1 for efficient GPU usage")
    fitness_evaluator = ParallelFitnessEvaluator(
        train_loader, val_loader, 
        device=device,
        batch_size=1,  # SA evaluates one architecture at a time
        max_epochs_parallel=5,  # Same as GA/PSO for consistency
        logger=algorithm.logger  # Pass the logger
    )
    
    # Set the fitness evaluator in the algorithm
    algorithm.fitness_evaluator = fitness_evaluator
    
    print(f"Running {algorithm_type} on {dataset_name}")
    best_architecture, best_fitness = algorithm.run()
    
    # Final evaluation on test set with GPU support
    from training_utils import NetworkTrainer
    trainer = NetworkTrainer(device=device)
    test_results = trainer.train_and_evaluate(
        best_architecture, train_loader, test_loader, epochs=5
    )
    
    return {
        'algorithm': algorithm_type,
        'dataset': dataset_name,
        'best_architecture': best_architecture,
        'best_fitness': best_fitness,
        'test_accuracy': test_results['accuracy'],
        'test_loss': test_results['loss'],
        'parameter_count': test_results['parameter_count'],
        'training_time': test_results['training_time'],
        'convergence_data': algorithm.logger.get_convergence_data(),
        'logger': algorithm.logger,
        'accepted_moves': algorithm.accepted_moves,
        'rejected_moves': algorithm.rejected_moves
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'])
    parser.add_argument('--algorithm', default='ASA', choices=['SA', 'ASA'])
    parser.add_argument('--initial_temperature', type=float, default=100.0)
    parser.add_argument('--cooling_rate', type=float, default=0.981748)
    parser.add_argument('--max_iterations', type=int, default=500)
    parser.add_argument('--output', default='sa_results.pkl')
    
    args = parser.parse_args()
    
    results = run_simulated_annealing(
        dataset_name=args.dataset,
        algorithm_type=args.algorithm,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        max_iterations=args.max_iterations
    )
    
    print(f"\nFinal Results:")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Parameter count: {results['parameter_count']:,}")
    print(f"Architecture layers: {len(results['best_architecture'].layers)}")
    print(f"Accepted/Rejected moves: {results['accepted_moves']}/{results['rejected_moves']}")
    
    # Save results
    results['logger'].save_results(args.output)
    print(f"Results saved to {args.output}")