import random
import numpy as np
import time
import torch
from typing import List, Tuple, Dict
from architecture_base import NetworkArchitecture, ArchitectureGenerator, LayerConfig, LayerType
from training_utils import FitnessEvaluator, ExperimentLogger
import copy

class GeneticAlgorithm:
    """Genetic Algorithm for Neural Architecture Search."""
    
    def __init__(self,
                 fitness_evaluator: FitnessEvaluator,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 population_size: int = 50,
                 num_generations: int = 50,
                 crossover_prob: float = 0.7,
                 mutation_add_prob: float = 0.15,
                 mutation_del_prob: float = 0.1,
                 mutation_param_prob: float = 0.2,
                 tournament_size: int = 3,
                 max_layers: int = 10):
        
        self.fitness_evaluator = fitness_evaluator
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_add_prob = mutation_add_prob
        self.mutation_del_prob = mutation_del_prob
        self.mutation_param_prob = mutation_param_prob
        self.tournament_size = tournament_size
        self.max_layers = max_layers
        
        self.generator = ArchitectureGenerator(input_shape, num_classes, max_layers)
        self.logger = ExperimentLogger("GA")
        
        # Population storage
        self.population: List[NetworkArchitecture] = []
        self.fitness_scores: List[float] = []
        
    def run(self) -> Tuple[NetworkArchitecture, float]:
        """Run the genetic algorithm."""
        print(f"🧬 Starting Genetic Algorithm with population size {self.population_size}")
        print(f"   📋 Generations: {self.num_generations}, Max layers: {self.max_layers}")
        print(f"   🎯 Crossover: {self.crossover_prob}, Mutation rates: add={self.mutation_add_prob}, del={self.mutation_del_prob}, param={self.mutation_param_prob}")
        
        # Initialize population
        self._initialize_population()
        
        # Evolution loop
        for generation in range(self.num_generations):
            print(f"\n🔄 Generation {generation + 1}/{self.num_generations}")
            gen_start_time = time.time()
            
            # Evaluate fitness
            self._evaluate_population()
            
            # Log results
            best_idx = np.argmax(self.fitness_scores)
            best_fitness = self.fitness_scores[best_idx]
            best_architecture = self.population[best_idx]
            avg_fitness = np.mean(self.fitness_scores)
            std_fitness = np.std(self.fitness_scores)
            
            self.logger.log_generation(generation, self.fitness_scores, 
                                     best_architecture, best_fitness)
            
            gen_time = time.time() - gen_start_time
            print(f"   📊 Best fitness: {best_fitness:.4f} | Avg: {avg_fitness:.4f} ± {std_fitness:.4f}")
            print(f"   🏗️  Best architecture: {len(best_architecture.layers)} layers, {best_architecture.get_parameter_count():,} params")
            print(f"   ⏱️  Generation time: {gen_time:.1f}s")
            
            # Create next generation
            if generation < self.num_generations - 1:
                print(f"   🔄 Creating next generation...")
                self._create_next_generation()
        
        # Return best solution
        best_idx = np.argmax(self.fitness_scores)
        print(f"\n✅ GA completed! Best fitness: {self.fitness_scores[best_idx]:.4f}")
        return self.population[best_idx], self.fitness_scores[best_idx]
    
    def _initialize_population(self):
        """Initialize random population."""
        print("   🎲 Initializing population...")
        init_start = time.time()
        self.population = []
        
        for i in range(self.population_size):
            if i % 10 == 0 or i == self.population_size - 1:
                print(f"      Creating individual {i + 1}/{self.population_size}")
            architecture = self.generator.generate_random_architecture()
            self.population.append(architecture)
        
        init_time = time.time() - init_start
        print(f"   ✅ Population initialized in {init_time:.1f}s")
    
    def _evaluate_population(self):
        """Evaluate fitness for entire population."""
        print("   🎯 Evaluating population fitness...")
        eval_start = time.time()
        self.fitness_scores = []
        
        for i, architecture in enumerate(self.population):
            if i % 5 == 0 or i == len(self.population) - 1:
                print(f"      Evaluating {i + 1}/{len(self.population)}")
            
            results = self.fitness_evaluator.evaluate_fitness(architecture)
            self.fitness_scores.append(results['fitness'])
        
        eval_time = time.time() - eval_start
        print(f"   ✅ Population evaluated in {eval_time:.1f}s")
    
    def _create_next_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        print("      🔄 Creating offspring...")
        creation_start = time.time()
        new_population = []
        crossovers = 0
        mutations = 0
        
        # Create offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_prob:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = self._crossover(parent1, parent2)
                crossovers += 1
                
                # Mutate offspring
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                mutations += 2
                
                new_population.extend([child1, child2])
            else:
                # Just select and mutate
                parent = self._tournament_selection()
                child = self._mutate(parent.copy())
                mutations += 1
                new_population.append(child)
        
        # Truncate to exact population size
        self.population = new_population[:self.population_size]
        
        creation_time = time.time() - creation_start
        print(f"      ✅ Next generation created in {creation_time:.1f}s ({crossovers} crossovers, {mutations} mutations)")
    
    def _tournament_selection(self) -> NetworkArchitecture:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), 
                                         min(self.tournament_size, len(self.population)))
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: NetworkArchitecture, parent2: NetworkArchitecture) -> Tuple[NetworkArchitecture, NetworkArchitecture]:
        """Perform single-point crossover."""
        if len(parent1.layers) == 0 or len(parent2.layers) == 0:
            return parent1.copy(), parent2.copy()
        
        # Find crossover points
        cross_point1 = random.randint(0, len(parent1.layers))
        cross_point2 = random.randint(0, len(parent2.layers))
        
        # Create children
        child1 = NetworkArchitecture(self.input_shape, self.num_classes)
        child2 = NetworkArchitecture(self.input_shape, self.num_classes)
        
        # Child 1: parent1[:cross_point1] + parent2[cross_point2:]
        for i in range(cross_point1):
            if i < len(parent1.layers):
                if not child1.add_layer(parent1.layers[i].copy()):
                    break
        
        for i in range(cross_point2, len(parent2.layers)):
            if len(child1.layers) >= self.max_layers:
                break
            if not child1.add_layer(parent2.layers[i].copy()):
                break
        
        # Child 2: parent2[:cross_point2] + parent1[cross_point1:]
        for i in range(cross_point2):
            if i < len(parent2.layers):
                if not child2.add_layer(parent2.layers[i].copy()):
                    break
        
        for i in range(cross_point1, len(parent1.layers)):
            if len(child2.layers) >= self.max_layers:
                break
            if not child2.add_layer(parent1.layers[i].copy()):
                break
        
        return child1, child2
    
    def _mutate(self, architecture: NetworkArchitecture) -> NetworkArchitecture:
        """Apply mutation operations to architecture."""
        mutated = architecture.copy()
        
        # Layer addition mutation
        if (random.random() < self.mutation_add_prob and 
            len(mutated.layers) < self.max_layers):
            new_layer = self.generator._generate_random_layer(mutated)
            if new_layer:
                mutated.add_layer(new_layer)
        
        # Layer deletion mutation
        if (random.random() < self.mutation_del_prob and 
            len(mutated.layers) > 1):
            layer_idx = random.randint(0, len(mutated.layers) - 1)
            mutated.remove_layer(layer_idx)
        
        # Parameter mutation
        for layer in mutated.layers:
            if random.random() < self.mutation_param_prob:
                self._mutate_layer_params(layer)
        
        return mutated
    
    def _mutate_layer_params(self, layer_config: LayerConfig):
        """Mutate parameters of a layer."""
        if layer_config.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
            if 'filters' in layer_config.params:
                if random.random() < 0.5:
                    layer_config.params['filters'] = random.choice([16, 32, 64, 128, 256])
            if 'kernel_size' in layer_config.params:
                if random.random() < 0.3:
                    layer_config.params['kernel_size'] = random.choice([3, 5, 7])
            if 'stride' in layer_config.params:
                if random.random() < 0.3:
                    layer_config.params['stride'] = random.choice([1, 2])
        
        elif layer_config.layer_type == LayerType.FC:
            if 'units' in layer_config.params:
                if random.random() < 0.5:
                    layer_config.params['units'] = random.choice([64, 128, 256, 512, 1024])
        
        elif layer_config.layer_type == LayerType.DROPOUT:
            if 'rate' in layer_config.params:
                if random.random() < 0.5:
                    layer_config.params['rate'] = random.uniform(0.1, 0.5)
        
        elif layer_config.layer_type == LayerType.MAXPOOL:
            if 'pool_size' in layer_config.params:
                if random.random() < 0.3:
                    layer_config.params['pool_size'] = random.choice([2, 3, 4])

class EnhancedGeneticAlgorithm(GeneticAlgorithm):
    """Enhanced Genetic Algorithm with elitism and adaptive mutation."""
    
    def __init__(self, 
                 fitness_evaluator: FitnessEvaluator,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 elite_size: int = 5,
                 adaptive_mutation: bool = True,
                 **kwargs):
        
        super().__init__(fitness_evaluator, input_shape, num_classes, **kwargs)
        self.elite_size = elite_size
        self.adaptive_mutation = adaptive_mutation
        self.logger = ExperimentLogger("EGA")
        
        # Adaptive mutation tracking
        self.diversity_history = []
        self.base_mutation_rates = {
            'add': self.mutation_add_prob,
            'del': self.mutation_del_prob,
            'param': self.mutation_param_prob
        }
    
    def _create_next_generation(self):
        """Create next generation with elitism."""
        new_population = []
        
        # Elitism: preserve best individuals
        if self.elite_size > 0:
            elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
        
        # Adapt mutation rates
        if self.adaptive_mutation:
            self._adapt_mutation_rates()
        
        # Create remaining offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_prob:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            else:
                parent = self._tournament_selection()
                child = self._mutate(parent.copy())
                new_population.append(child)
        
        self.population = new_population[:self.population_size]
    
    def _adapt_mutation_rates(self):
        """Adapt mutation rates based on population diversity."""
        # Calculate population diversity (simplified)
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)
        
        # Adjust mutation rates based on diversity
        if len(self.diversity_history) > 5:
            recent_diversity = np.mean(self.diversity_history[-5:])
            
            if recent_diversity < 0.3:  # Low diversity, increase mutation
                self.mutation_add_prob = min(0.4, self.base_mutation_rates['add'] * 1.5)
                self.mutation_del_prob = min(0.3, self.base_mutation_rates['del'] * 1.5)
                self.mutation_param_prob = min(0.4, self.base_mutation_rates['param'] * 1.5)
            elif recent_diversity > 0.7:  # High diversity, decrease mutation
                self.mutation_add_prob = max(0.05, self.base_mutation_rates['add'] * 0.7)
                self.mutation_del_prob = max(0.05, self.base_mutation_rates['del'] * 0.7)
                self.mutation_param_prob = max(0.1, self.base_mutation_rates['param'] * 0.7)
            else:  # Normal diversity, reset to base rates
                self.mutation_add_prob = self.base_mutation_rates['add']
                self.mutation_del_prob = self.base_mutation_rates['del']
                self.mutation_param_prob = self.base_mutation_rates['param']
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity based on architecture differences."""
        if len(self.population) < 2:
            return 1.0
        
        total_comparisons = 0
        different_pairs = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                total_comparisons += 1
                if self._architectures_different(self.population[i], self.population[j]):
                    different_pairs += 1
        
        return different_pairs / total_comparisons if total_comparisons > 0 else 1.0
    
    def _architectures_different(self, arch1: NetworkArchitecture, arch2: NetworkArchitecture) -> bool:
        """Check if two architectures are significantly different."""
        if len(arch1.layers) != len(arch2.layers):
            return True
        
        for layer1, layer2 in zip(arch1.layers, arch2.layers):
            if layer1.layer_type != layer2.layer_type:
                return True
            # Check key parameters
            if layer1.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
                if (layer1.params.get('filters') != layer2.params.get('filters') or
                    layer1.params.get('kernel_size') != layer2.params.get('kernel_size')):
                    return True
            elif layer1.layer_type == LayerType.FC:
                if layer1.params.get('units') != layer2.params.get('units'):
                    return True
        
        return False

def run_genetic_algorithm(dataset_name: str = "cifar10", 
                         algorithm_type: str = "EGA",
                         device: str = None,
                         **kwargs) -> Dict:
    """Run genetic algorithm experiment."""
    
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
        'population_size': 50,
        'num_generations': 50,
        'crossover_prob': 0.7,
        'mutation_add_prob': 0.15,
        'mutation_del_prob': 0.1,
        'mutation_param_prob': 0.2,
        'tournament_size': 3,
        'max_layers': 10
    }
    default_params.update(kwargs)
    
    # Run algorithm
    if algorithm_type.upper() == "EGA":
        algorithm = EnhancedGeneticAlgorithm(
            fitness_evaluator, input_shape, num_classes,
            elite_size=5, **default_params
        )
    else:
        algorithm = GeneticAlgorithm(
            fitness_evaluator, input_shape, num_classes,
            **default_params
        )
    
    print(f"Running {algorithm_type} on {dataset_name}")
    best_architecture, best_fitness = algorithm.run()
    
    # Final evaluation on test set
    from training_utils import NetworkTrainer
    trainer = NetworkTrainer(device=device)
    test_results = trainer.train_and_evaluate(
        best_architecture, train_loader, test_loader, epochs=20
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
        'logger': algorithm.logger
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'mnist'])
    parser.add_argument('--algorithm', default='EGA', choices=['GA', 'EGA'])
    parser.add_argument('--population_size', type=int, default=20)
    parser.add_argument('--num_generations', type=int, default=30)
    parser.add_argument('--output', default='ga_results.pkl')
    
    args = parser.parse_args()
    
    results = run_genetic_algorithm(
        dataset_name=args.dataset,
        algorithm_type=args.algorithm,
        population_size=args.population_size,
        num_generations=args.num_generations
    )
    
    print(f"\nFinal Results:")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Parameter count: {results['parameter_count']:,}")
    print(f"Architecture layers: {len(results['best_architecture'].layers)}")
    
    # Save results
    results['logger'].save_results(args.output)
    print(f"Results saved to {args.output}")