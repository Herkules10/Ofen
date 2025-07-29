import numpy as np
import random
import time
from typing import List, Tuple, Dict, Optional
from architecture_base import NetworkArchitecture, LayerConfig, LayerType, ActivationType
from training_utils import FitnessEvaluator, ExperimentLogger
import torch

class Particle:
    """Represents a particle in the swarm."""
    
    def __init__(self, position_dim: int, bounds: List[Tuple[float, float]]):
        self.position_dim = position_dim
        self.bounds = bounds
        
        # Initialize position randomly within bounds
        self.position = np.array([
            random.uniform(bounds[i][0], bounds[i][1]) 
            for i in range(position_dim)
        ])
        
        # Initialize velocity
        self.velocity = np.zeros(position_dim)
        
        # Personal best
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('-inf')
        
        # Current fitness
        self.fitness = float('-inf')
        self.architecture = None
    
    def update_velocity(self, global_best_position: np.ndarray, 
                       w: float, c1: float, c2: float):
        """Update particle velocity."""
        r1 = np.random.random(self.position_dim)
        r2 = np.random.random(self.position_dim)
        
        cognitive = c1 * r1 * (self.personal_best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self):
        """Update particle position and enforce bounds."""
        self.position += self.velocity
        
        # Enforce bounds
        for i in range(self.position_dim):
            if self.position[i] < self.bounds[i][0]:
                self.position[i] = self.bounds[i][0]
                self.velocity[i] = 0  # Stop at boundary
            elif self.position[i] > self.bounds[i][1]:
                self.position[i] = self.bounds[i][1]
                self.velocity[i] = 0  # Stop at boundary
    
    def update_personal_best(self):
        """Update personal best if current fitness is better."""
        if self.fitness > self.personal_best_fitness:
            self.personal_best_fitness = self.fitness
            self.personal_best_position = self.position.copy()

class ArchitectureDecoder:
    """Decodes continuous PSO positions into discrete neural architectures."""
    
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, max_layers: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_layers = max_layers
        
        # Define layer types and their encoding ranges
        self.layer_types = [
            LayerType.CONV2D,
            LayerType.CONV1D, 
            LayerType.FC,
            LayerType.BATCHNORM,
            LayerType.MAXPOOL,
            LayerType.DROPOUT,
            LayerType.ACTIVATION
        ]
        
        # Calculate position dimension
        # Format: [num_layers, layer_types..., layer_params...]
        self.num_layers_dim = 1
        self.layer_type_dims = max_layers  # One dimension per layer for type selection
        self.param_dims_per_layer = 4  # filters/units, kernel_size, stride, dropout_rate
        self.total_param_dims = max_layers * self.param_dims_per_layer
        
        self.position_dim = (self.num_layers_dim + 
                           self.layer_type_dims + 
                           self.total_param_dims)
        
        # Define bounds for each dimension
        self.bounds = self._create_bounds()
    
    def _create_bounds(self) -> List[Tuple[float, float]]:
        """Create bounds for each position dimension."""
        bounds = []
        
        # Number of layers bound
        bounds.append((1.0, float(self.max_layers)))
        
        # Layer type bounds (will use softmax to select)
        for _ in range(self.max_layers):
            bounds.append((0.0, 1.0))
        
        # Parameter bounds for each layer
        for _ in range(self.max_layers):
            bounds.append((0.0, 1.0))  # filters/units (will map to discrete values)
            bounds.append((0.0, 1.0))  # kernel_size
            bounds.append((0.0, 1.0))  # stride
            bounds.append((0.0, 1.0))  # dropout_rate/other params
        
        return bounds
    
    def decode(self, position: np.ndarray) -> NetworkArchitecture:
        """Decode position vector to neural architecture."""
        architecture = NetworkArchitecture(self.input_shape, self.num_classes)
        
        # Extract number of layers
        num_layers = max(1, min(self.max_layers, int(position[0])))
        
        # Extract layer configurations
        for layer_idx in range(num_layers):
            layer_config = self._decode_layer(position, layer_idx, architecture)
            if layer_config and not architecture.add_layer(layer_config):
                break  # Stop if layer cannot be added
        
        # Ensure at least one meaningful layer
        if len(architecture.layers) == 0:
            # Add a default conv layer for images or FC for other data
            if len(self.input_shape) == 3:
                default_layer = LayerConfig(LayerType.CONV2D, {
                    'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1
                })
            else:
                default_layer = LayerConfig(LayerType.FC, {'units': 128})
            architecture.add_layer(default_layer)
        
        return architecture
    
    def _decode_layer(self, position: np.ndarray, layer_idx: int, 
                     architecture: NetworkArchitecture) -> Optional[LayerConfig]:
        """Decode a single layer from position vector."""
        # Get layer type probabilities
        type_start_idx = 1 + layer_idx
        type_prob = position[type_start_idx]
        
        # Get parameter values
        param_start_idx = 1 + self.max_layers + layer_idx * self.param_dims_per_layer
        param1 = position[param_start_idx]      # filters/units
        param2 = position[param_start_idx + 1]  # kernel_size
        param3 = position[param_start_idx + 2]  # stride
        param4 = position[param_start_idx + 3]  # dropout_rate/other
        
        # Select layer type based on architecture state and type probability
        current_shape = architecture._current_shape
        possible_types = self._get_possible_layer_types(current_shape, len(architecture.layers))
        
        if not possible_types:
            return None
        
        # Use probability to select type
        type_idx = int(type_prob * len(possible_types))
        type_idx = min(type_idx, len(possible_types) - 1)
        layer_type = possible_types[type_idx]
        
        # Create layer configuration based on type
        if layer_type == LayerType.CONV2D:
            filters_options = [16, 32, 64, 128, 256]
            kernel_options = [3, 5, 7]
            stride_options = [1, 2]
            
            return LayerConfig(LayerType.CONV2D, {
                'filters': filters_options[int(param1 * len(filters_options))],
                'kernel_size': kernel_options[int(param2 * len(kernel_options))],
                'stride': stride_options[int(param3 * len(stride_options))],
                'padding': 1
            })
        
        elif layer_type == LayerType.CONV1D:
            filters_options = [16, 32, 64, 128]
            kernel_options = [3, 5, 7]
            stride_options = [1, 2]
            
            return LayerConfig(LayerType.CONV1D, {
                'filters': filters_options[int(param1 * len(filters_options))],
                'kernel_size': kernel_options[int(param2 * len(kernel_options))],
                'stride': stride_options[int(param3 * len(stride_options))],
                'padding': 1
            })
        
        elif layer_type == LayerType.FC:
            units_options = [64, 128, 256, 512, 1024]
            return LayerConfig(LayerType.FC, {
                'units': units_options[int(param1 * len(units_options))]
            })
        
        elif layer_type == LayerType.BATCHNORM:
            return LayerConfig(LayerType.BATCHNORM, {})
        
        elif layer_type == LayerType.MAXPOOL:
            pool_options = [2, 3, 4]
            return LayerConfig(LayerType.MAXPOOL, {
                'pool_size': pool_options[int(param2 * len(pool_options))]
            })
        
        elif layer_type == LayerType.DROPOUT:
            dropout_rate = 0.1 + param4 * 0.4  # 0.1 to 0.5
            return LayerConfig(LayerType.DROPOUT, {
                'rate': dropout_rate
            })
        
        elif layer_type == LayerType.ACTIVATION:
            activation_options = list(ActivationType)
            activation_idx = int(param4 * len(activation_options))
            activation_idx = min(activation_idx, len(activation_options) - 1)
            return LayerConfig(LayerType.ACTIVATION, {
                'type': activation_options[activation_idx]
            })
        
        return None
    
    def _get_possible_layer_types(self, current_shape: Tuple, num_existing_layers: int) -> List[LayerType]:
        """Get possible layer types given current architecture state."""
        possible = []
        
        if len(current_shape) == 3:  # After conv layers
            possible.extend([LayerType.CONV2D, LayerType.MAXPOOL, LayerType.FC])
            if num_existing_layers > 0:
                possible.extend([LayerType.BATCHNORM, LayerType.DROPOUT, LayerType.ACTIVATION])
        
        elif len(current_shape) == 2:  # After 1D conv
            possible.extend([LayerType.CONV1D, LayerType.MAXPOOL, LayerType.FC])
            if num_existing_layers > 0:
                possible.extend([LayerType.BATCHNORM, LayerType.DROPOUT, LayerType.ACTIVATION])
        
        else:  # After FC layers
            possible.append(LayerType.FC)
            if num_existing_layers > 0:
                possible.extend([LayerType.BATCHNORM, LayerType.DROPOUT, LayerType.ACTIVATION])
        
        return possible

class ParticleSwarmOptimization:
    """Particle Swarm Optimization for Neural Architecture Search."""
    
    def __init__(self,
                 fitness_evaluator: FitnessEvaluator,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 swarm_size: int = 30,
                 num_iterations: int = 100,
                 w: float = 0.7,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 max_layers: int = 10):
        
        self.fitness_evaluator = fitness_evaluator
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.swarm_size = swarm_size
        self.num_iterations = num_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.max_layers = max_layers
        
        # Initialize decoder
        self.decoder = ArchitectureDecoder(input_shape, num_classes, max_layers)
        
        # Initialize swarm
        self.swarm: List[Particle] = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.global_best_architecture = None
        
        # Logger
        self.logger = ExperimentLogger("PSO")
        
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        """Initialize particle swarm."""
        print(f"🚀 Initializing swarm with {self.swarm_size} particles")
        init_start = time.time()
        self.swarm = []
        
        for i in range(self.swarm_size):
            if i % 10 == 0 or i == self.swarm_size - 1:
                print(f"   Creating particle {i + 1}/{self.swarm_size}")
            
            particle = Particle(self.decoder.position_dim, self.decoder.bounds)
            self.swarm.append(particle)
        
        init_time = time.time() - init_start
        print(f"   ✅ Swarm initialized in {init_time:.1f}s")
    
    def run(self) -> Tuple[NetworkArchitecture, float]:
        """Run PSO algorithm."""
        print(f"🦋 Starting PSO with {self.swarm_size} particles for {self.num_iterations} iterations")
        print(f"   📋 Parameters: w={self.w}, c1={self.c1}, c2={self.c2}, max_layers={self.max_layers}")
        
        for iteration in range(self.num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.num_iterations}")
            iter_start = time.time()
            
            # Evaluate all particles
            self._evaluate_swarm()
            
            # Update global best
            self._update_global_best()
            
            # Log results
            fitness_scores = [p.fitness for p in self.swarm]
            avg_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)
            
            self.logger.log_generation(iteration, fitness_scores,
                                     self.global_best_architecture, self.global_best_fitness)
            
            iter_time = time.time() - iter_start
            print(f"   📊 Best fitness: {self.global_best_fitness:.4f} | Avg: {avg_fitness:.4f} ± {std_fitness:.4f}")
            print(f"   🏗️  Best architecture: {len(self.global_best_architecture.layers)} layers, {self.global_best_architecture.get_parameter_count():,} params")
            print(f"   ⏱️  Iteration time: {iter_time:.1f}s")
            
            # Update particle dynamics
            if iteration < self.num_iterations - 1:
                print(f"   🔄 Updating particle dynamics...")
                self._update_swarm()
        
        print(f"\n✅ PSO completed! Best fitness: {self.global_best_fitness:.4f}")
        return self.global_best_architecture, self.global_best_fitness
    
    def _evaluate_swarm(self):
        """Evaluate fitness for all particles."""
        print("   🎯 Evaluating swarm fitness...")
        eval_start = time.time()
        
        for i, particle in enumerate(self.swarm):
            if i % 5 == 0 or i == len(self.swarm) - 1:
                print(f"      Evaluating particle {i + 1}/{len(self.swarm)}")
            
            # Decode position to architecture
            architecture = self.decoder.decode(particle.position)
            particle.architecture = architecture
            
            # Evaluate fitness
            results = self.fitness_evaluator.evaluate_fitness(architecture)
            particle.fitness = results['fitness']
            
            # Update personal best
            particle.update_personal_best()
        
        eval_time = time.time() - eval_start
        print(f"   ✅ Swarm evaluated in {eval_time:.1f}s")
    
    def _update_global_best(self):
        """Update global best particle."""
        for particle in self.swarm:
            if particle.fitness > self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
                self.global_best_architecture = particle.architecture.copy()
    
    def _update_swarm(self):
        """Update velocities and positions of all particles."""
        for particle in self.swarm:
            particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
            particle.update_position()

def run_particle_swarm_optimization(dataset_name: str = "cifar10", device: str = None, **kwargs) -> Dict:
    """Run PSO experiment."""
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load dataset
    from training_utils import DatasetLoader
    
    if dataset_name.lower() == "cifar10":
        train_loader, val_loader, test_loader = DatasetLoader.load_cifar10()
        input_shape = (3, 32, 32)
        num_classes = 10
    elif dataset_name.lower() == "mnist":
        train_loader, val_loader, test_loader = DatasetLoader.load_mnist()
        input_shape = (1, 28, 28)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Setup fitness evaluator with device
    fitness_evaluator = FitnessEvaluator(train_loader, val_loader, device=device)
    
    # Default parameters
    default_params = {
        'swarm_size': 30,
        'num_iterations': 100,
        'w': 0.7,
        'c1': 2.0,
        'c2': 2.0,
        'max_layers': 10
    }
    default_params.update(kwargs)
    
    # Run PSO
    pso = ParticleSwarmOptimization(
        fitness_evaluator, input_shape, num_classes, **default_params
    )
    
    print(f"Running PSO on {dataset_name}")
    best_architecture, best_fitness = pso.run()
    
    # Final evaluation on test set
    from training_utils import NetworkTrainer
    trainer = NetworkTrainer(device=device)
    test_results = trainer.train_and_evaluate(
        best_architecture, train_loader, test_loader, epochs=20
    )
    
    return {
        'algorithm': 'PSO',
        'dataset': dataset_name,
        'best_architecture': best_architecture,
        'best_fitness': best_fitness,
        'test_accuracy': test_results['accuracy'],
        'test_loss': test_results['loss'],
        'parameter_count': test_results['parameter_count'],
        'training_time': test_results['training_time'],
        'convergence_data': pso.logger.get_convergence_data(),
        'logger': pso.logger
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'])
    parser.add_argument('--swarm_size', type=int, default=20)
    parser.add_argument('--num_iterations', type=int, default=50)
    parser.add_argument('--w', type=float, default=0.7)
    parser.add_argument('--c1', type=float, default=2.0)
    parser.add_argument('--c2', type=float, default=2.0)
    parser.add_argument('--output', default='pso_results.pkl')
    
    args = parser.parse_args()
    
    results = run_particle_swarm_optimization(
        dataset_name=args.dataset,
        swarm_size=args.swarm_size,
        num_iterations=args.num_iterations,
        w=args.w,
        c1=args.c1,
        c2=args.c2
    )
    
    print(f"\nFinal Results:")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Parameter count: {results['parameter_count']:,}")
    print(f"Architecture layers: {len(results['best_architecture'].layers)}")
    
    # Save results
    results['logger'].save_results(args.output)
    print(f"Results saved to {args.output}")