import random
import numpy as np
from typing import List, Tuple, Dict
import sys
import os
import torch
import threading
import time as time_module
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture_base import NetworkArchitecture, ArchitectureGenerator, LayerConfig, LayerType
from training_utils import FitnessEvaluator, ExperimentLogger
import copy


class SystemMonitor:
    """Monitor system resource usage during experiments."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.timestamps = []
        self.available = PSUTIL_AVAILABLE
        
    def start_monitoring(self):
        """Start monitoring system resources."""
        if not self.available:
            print("System monitoring disabled (psutil not available)")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("System monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring and return collected data."""
        if not self.available:
            return {}
            
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'timestamps': self.timestamps
        }
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # GPU usage if available
                gpu_percent = 0
                if torch.cuda.is_available():
                    try:
                        # Simple GPU memory usage check
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        gpu_percent = gpu_memory
                    except:
                        gpu_percent = 0
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_percent)
                self.gpu_usage.append(gpu_percent)
                self.timestamps.append(time_module.time())
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time_module.sleep(2.0)  # Monitor every 2 seconds
    
    def get_summary(self):
        """Get summary statistics of resource usage."""
        if not self.available or not self.cpu_usage:
            return "No monitoring data available"
        
        return {
            'avg_cpu': np.mean(self.cpu_usage),
            'max_cpu': np.max(self.cpu_usage),
            'avg_memory': np.mean(self.memory_usage),
            'max_memory': np.max(self.memory_usage),
            'avg_gpu': np.mean(self.gpu_usage) if self.gpu_usage else 0,
            'max_gpu': np.max(self.gpu_usage) if self.gpu_usage else 0
        }


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
        import time
        
        # Start system monitoring
        monitor = SystemMonitor()
        monitor.start_monitoring()
        
        start_time = time.time()
        print(f"Starting Genetic Algorithm with population size {self.population_size}")
        print(f"Target generations: {self.num_generations}")
        print(f"Max layers per architecture: {self.max_layers}")
        print("-" * 60)
        
        # Initialize population
        init_start = time.time()
        self._initialize_population()
        init_time = time.time() - init_start
        print(f"Population initialization took {init_time:.2f} seconds")
        print("-" * 60)
        
        # Evolution loop
        for generation in range(self.num_generations):
            gen_start = time.time()
            print(f"\n{'='*20} Generation {generation + 1}/{self.num_generations} {'='*20}")
            
            # Evaluate fitness
            eval_start = time.time()
            self._evaluate_population()
            eval_time = time.time() - eval_start
            
            # Log results
            best_idx = np.argmax(self.fitness_scores)
            best_fitness = self.fitness_scores[best_idx]
            best_architecture = self.population[best_idx]
            
            self.logger.log_generation(generation, self.fitness_scores, 
                                     best_architecture, best_fitness)
            
            print(f"Fitness evaluation took {eval_time:.2f} seconds")
            print(f"  Best fitness: {best_fitness:.4f}")
            print(f"  Avg fitness: {np.mean(self.fitness_scores):.4f}")
            print(f"  Std fitness: {np.std(self.fitness_scores):.4f}")
            print(f"  Best arch layers: {len(best_architecture.layers)}")
            print(f"  Best arch params: {best_architecture.get_parameter_count():,}")
            
            # Create next generation
            if generation < self.num_generations - 1:
                next_gen_start = time.time()
                self._create_next_generation()
                next_gen_time = time.time() - next_gen_start
                print(f"Next generation creation took {next_gen_time:.2f} seconds")
            
            gen_time = time.time() - gen_start
            print(f"Total generation time: {gen_time:.2f} seconds")
            
            # Estimate remaining time
            avg_gen_time = (time.time() - start_time) / (generation + 1)
            remaining_gens = self.num_generations - generation - 1
            est_remaining = avg_gen_time * remaining_gens
            print(f"Estimated remaining time: {est_remaining/60:.1f} minutes")
            
            # Show resource usage if available
            if monitor.available and generation % 5 == 0:  # Every 5 generations
                summary = monitor.get_summary()
                if isinstance(summary, dict):
                    print(f"Resource usage: CPU {summary['avg_cpu']:.1f}%, Memory {summary['avg_memory']:.1f}%")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"GENETIC ALGORITHM COMPLETED in {total_time/60:.2f} minutes")
        print(f"Average time per generation: {total_time/self.num_generations:.2f} seconds")
        
        # Stop monitoring and show final resource usage
        resource_data = monitor.stop_monitoring()
        if resource_data and monitor.available:
            summary = monitor.get_summary()
            if isinstance(summary, dict):
                print(f"\nResource Usage Summary:")
                print(f"  CPU: avg {summary['avg_cpu']:.1f}%, max {summary['max_cpu']:.1f}%")
                print(f"  Memory: avg {summary['avg_memory']:.1f}%, max {summary['max_memory']:.1f}%")
                if summary['max_gpu'] > 0:
                    print(f"  GPU: avg {summary['avg_gpu']:.1f}%, max {summary['max_gpu']:.1f}%")
        
        # Return best solution
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx], self.fitness_scores[best_idx]
    
    def _initialize_population(self):
        """Initialize random population."""
        import time
        print("Initializing population...")
        self.population = []
        total_start = time.time()
        
        for i in range(self.population_size):
            individual_start = time.time()
            architecture = self.generator.generate_random_architecture()
            individual_time = time.time() - individual_start
            
            self.population.append(architecture)
            
            if (i + 1) % 10 == 0 or i == self.population_size - 1:
                elapsed = time.time() - total_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (self.population_size - i - 1)
                print(f"  Created {i + 1}/{self.population_size} individuals "
                      f"(avg: {avg_time:.3f}s each, remaining: {remaining:.1f}s)")
        
        total_time = time.time() - total_start
        print(f"Population initialization completed in {total_time:.2f} seconds")
    
    def _evaluate_population(self):
        """Evaluate fitness for entire population."""
        import time
        print("Evaluating population fitness...")
        self.fitness_scores = []
        total_start = time.time()
        
        # Track timing statistics
        eval_times = []
        cache_hits = 0
        
        for i, architecture in enumerate(self.population):
            individual_start = time.time()
            
            # Check if this evaluation will hit cache
            arch_hash = self.fitness_evaluator._hash_architecture(architecture)
            is_cached = arch_hash in self.fitness_evaluator.evaluation_cache
            if is_cached:
                cache_hits += 1
            
            results = self.fitness_evaluator.evaluate_fitness(architecture)
            individual_time = time.time() - individual_start
            eval_times.append(individual_time)
            
            self.fitness_scores.append(results['fitness'])
            
            if (i + 1) % 5 == 0 or i == len(self.population) - 1:
                elapsed = time.time() - total_start
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(self.population) - i - 1)
                
                print(f"  Evaluated {i + 1}/{len(self.population)} individuals")
                print(f"    Current eval: {individual_time:.2f}s ({'cached' if is_cached else 'trained'})")
                print(f"    Average: {avg_time:.2f}s per individual")
                print(f"    Cache hits: {cache_hits}/{i+1} ({100*cache_hits/(i+1):.1f}%)")
                print(f"    Remaining time: {remaining/60:.1f} minutes")
                print(f"    Latest fitness: {results['fitness']:.4f}")
        
        total_time = time.time() - total_start
        print(f"Population evaluation completed in {total_time/60:.2f} minutes")
        print(f"Average evaluation time: {np.mean(eval_times):.2f} seconds")
        print(f"Cache hit rate: {100*cache_hits/len(self.population):.1f}%")
        print(f"Training time breakdown:")
        non_cached_times = [t for i, t in enumerate(eval_times) 
                          if self.fitness_evaluator._hash_architecture(self.population[i]) not in self.fitness_evaluator.evaluation_cache or i == 0]
        if non_cached_times:
            print(f"  Non-cached evaluations: {len(non_cached_times)} avg {np.mean(non_cached_times):.2f}s")
        cached_times = [t for i, t in enumerate(eval_times) 
                       if self.fitness_evaluator._hash_architecture(self.population[i]) in self.fitness_evaluator.evaluation_cache and i > 0]
        if cached_times:
            print(f"  Cached evaluations: {len(cached_times)} avg {np.mean(cached_times):.4f}s")

    def _evaluate_population_parallel(self):
        """Evaluate fitness for entire population using GPU parallelization."""
        import time
        print("Evaluating population fitness in parallel...")
        
        # Check if fitness evaluator supports parallel evaluation
        if not hasattr(self.fitness_evaluator, 'evaluate_population_parallel'):
            print("  Fitness evaluator doesn't support parallel evaluation, falling back to sequential")
            return self._evaluate_population()
        
        total_start = time.time()
        
        # Use parallel evaluation
        results_list = self.fitness_evaluator.evaluate_population_parallel(self.population)
        
        # Extract fitness scores
        self.fitness_scores = [result['fitness'] for result in results_list]
        
        total_time = time.time() - total_start
        print(f"Parallel population evaluation completed in {total_time:.2f} seconds")
        print(f"Average time per individual: {total_time/len(self.population):.3f} seconds")
        
        # Show speedup compared to sequential
        # Rough estimate: sequential would take ~10-30s per individual
        estimated_sequential = len(self.population) * 15  # Conservative estimate
        speedup = estimated_sequential / total_time if total_time > 0 else 1
        print(f"Estimated speedup over sequential: {speedup:.1f}x")

    def _create_next_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        import time
        print("Creating next generation...")
        start_time = time.time()
        
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
        
        elapsed = time.time() - start_time
        print(f"  Generated {len(self.population)} individuals in {elapsed:.3f} seconds")
        print(f"  Operations: {crossovers} crossovers, {mutations} mutations")
    
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


class ParallelGeneticAlgorithm(GeneticAlgorithm):
    """Genetic Algorithm with GPU-parallelized fitness evaluation."""
    
    def __init__(self,
                 fitness_evaluator,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 **kwargs):
        
        super().__init__(fitness_evaluator, input_shape, num_classes, **kwargs)
        self.logger = ExperimentLogger("ParallelGA")
        
        # Verify that we have a parallel fitness evaluator
        if not hasattr(fitness_evaluator, 'evaluate_population_parallel'):
            print("WARNING: Fitness evaluator doesn't support parallel evaluation!")
            print("Consider using ParallelFitnessEvaluator for better performance.")
    
    def _evaluate_population(self):
        """Override to use parallel evaluation."""
        import time
        print("Evaluating population fitness in parallel...")
        
        # Check if fitness evaluator supports parallel evaluation
        if not hasattr(self.fitness_evaluator, 'evaluate_population_parallel'):
            print("  Fitness evaluator doesn't support parallel evaluation, falling back to sequential")
            return super()._evaluate_population()
        
        total_start = time.time()
        
        # Use parallel evaluation
        results_list = self.fitness_evaluator.evaluate_population_parallel(self.population)
        
        # Extract fitness scores
        self.fitness_scores = [result['fitness'] for result in results_list]
        
        total_time = time.time() - total_start
        print(f"Parallel population evaluation completed in {total_time:.2f} seconds")
        print(f"Average time per individual: {total_time/len(self.population):.3f} seconds")
        
        # Show speedup compared to sequential
        # Rough estimate: sequential would take ~10-30s per individual
        estimated_sequential = len(self.population) * 15  # Conservative estimate
        speedup = estimated_sequential / total_time if total_time > 0 else 1
        print(f"Estimated speedup over sequential: {speedup:.1f}x")
    
    def _create_next_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        import time
        print("Creating next generation...")
        start_time = time.time()
        
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
        
        elapsed = time.time() - start_time
        print(f"  Generated {len(self.population)} individuals in {elapsed:.3f} seconds")
        print(f"  Operations: {crossovers} crossovers, {mutations} mutations")
    
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


class ParallelEnhancedGeneticAlgorithm(GeneticAlgorithm):
    """Enhanced Genetic Algorithm with GPU-parallelized fitness evaluation."""
    
    def __init__(self,
                 fitness_evaluator,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 elite_size: int = 5,
                 adaptive_mutation: bool = True,
                 **kwargs):
        
        # Initialize the base GA first
        super().__init__(fitness_evaluator, input_shape, num_classes, **kwargs)
        self.elite_size = elite_size
        self.adaptive_mutation = adaptive_mutation
        self.logger = ExperimentLogger("ParallelEGA")
        
        # Adaptive mutation tracking (from EnhancedGeneticAlgorithm)
        self.diversity_history = []
        self.base_mutation_rates = {
            'add': self.mutation_add_prob,
            'del': self.mutation_del_prob,
            'param': self.mutation_param_prob
        }
        
        # Verify that we have a parallel fitness evaluator
        if not hasattr(fitness_evaluator, 'evaluate_population_parallel'):
            print("WARNING: Fitness evaluator doesn't support parallel evaluation!")
            print("Consider using ParallelFitnessEvaluator for better performance.")
    
    def _evaluate_population(self):
        """Override to use parallel evaluation."""
        import time
        print("Evaluating population fitness in parallel...")
        
        # Check if fitness evaluator supports parallel evaluation
        if not hasattr(self.fitness_evaluator, 'evaluate_population_parallel'):
            print("  Fitness evaluator doesn't support parallel evaluation, falling back to sequential")
            return super()._evaluate_population()
        
        total_start = time.time()
        
        # Use parallel evaluation
        results_list = self.fitness_evaluator.evaluate_population_parallel(self.population)
        
        # Extract fitness scores
        self.fitness_scores = [result['fitness'] for result in results_list]
        
        total_time = time.time() - total_start
        print(f"Parallel population evaluation completed in {total_time:.2f} seconds")
        print(f"Average time per individual: {total_time/len(self.population):.3f} seconds")
        
        # Show speedup compared to sequential
        # Rough estimate: sequential would take ~10-30s per individual
        estimated_sequential = len(self.population) * 15  # Conservative estimate
        speedup = estimated_sequential / total_time if total_time > 0 else 1
        print(f"Estimated speedup over sequential: {speedup:.1f}x")
    
    def _create_next_generation(self):
        """Create next generation with elitism (copied from EnhancedGeneticAlgorithm)."""
        import time
        print("Creating next generation...")
        start_time = time.time()
        
        new_population = []
        crossovers = 0
        mutations = 0
        
        # Elitism: preserve best individuals
        if self.elite_size > 0:
            elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
        
        # Adapt mutation rates if enabled
        if self.adaptive_mutation:
            self._adapt_mutation_rates()
        
        # Create remaining offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_prob:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = self._crossover(parent1, parent2)
                crossovers += 1
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                mutations += 2
                
                new_population.extend([child1, child2])
            else:
                parent = self._tournament_selection()
                child = self._mutate(parent.copy())
                mutations += 1
                new_population.append(child)
        
        self.population = new_population[:self.population_size]
        
        elapsed = time.time() - start_time
        print(f"  Generated {len(self.population)} individuals in {elapsed:.3f} seconds")
        print(f"  Operations: {crossovers} crossovers, {mutations} mutations")
    
    def _adapt_mutation_rates(self):
        """Adapt mutation rates based on population diversity (from EnhancedGeneticAlgorithm)."""
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
    
    def _create_next_generation(self):
        """Create next generation with elitism (copied from EnhancedGeneticAlgorithm)."""
        import time
        print("Creating next generation with elitism...")
        start_time = time.time()
        
        new_population = []
        crossovers = 0
        mutations = 0
        
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
                crossovers += 1
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                mutations += 2
                
                new_population.extend([child1, child2])
            else:
                parent = self._tournament_selection()
                child = self._mutate(parent.copy())
                mutations += 1
                new_population.append(child)
        
        self.population = new_population[:self.population_size]
        
        elapsed = time.time() - start_time
        print(f"  Generated {len(self.population)} individuals in {elapsed:.3f} seconds")
        print(f"  Operations: {crossovers} crossovers, {mutations} mutations")
        print(f"  Elite individuals preserved: {self.elite_size}")
    
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
    
    def _create_next_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        import time
        print("Creating next generation...")
        start_time = time.time()
        
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
        
        elapsed = time.time() - start_time
        print(f"  Generated {len(self.population)} individuals in {elapsed:.3f} seconds")
        print(f"  Operations: {crossovers} crossovers, {mutations} mutations")
    
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
                         train_split: float = 1.0,
                         use_parallel: bool = True,  # New parameter for parallel processing
                         parallel_batch_size: int = 4,  # Number of networks to train in parallel
                         **kwargs) -> Dict:
    """Run genetic algorithm experiment."""
    import time
    import torch
    
    print("="*80)
    print("NEURAL ARCHITECTURE SEARCH - GENETIC ALGORITHM")
    if use_parallel:
        print("PARALLEL GPU MODE")
    print("="*80)
    
    # Hardware info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    experiment_start = time.time()
    
    # Load dataset
    from training_utils import DatasetLoader, ParallelFitnessEvaluator
    
    dataset_start = time.time()
    print(f"Loading dataset: {dataset_name}...")
    
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
    
    dataset_time = time.time() - dataset_start
    print(f"Dataset loaded in {dataset_time:.2f} seconds")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print()
    
    # Setup fitness evaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if use_parallel and device == "cuda":
        print(f"Using ParallelFitnessEvaluator with batch size {parallel_batch_size}")
        fitness_evaluator = ParallelFitnessEvaluator(
            train_loader, val_loader, 
            device=device,
            batch_size=parallel_batch_size,
            max_epochs_parallel=3  # Fewer epochs for faster parallel training
        )
    else:
        print("Using standard FitnessEvaluator")
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
    
    
    print(f"Algorithm parameters:")
    for key, value in default_params.items():
        print(f"  {key}: {value}")
    print(f"  use_parallel: {use_parallel}")
    print(f"  parallel_batch_size: {parallel_batch_size}")
    print()
    
    # Run algorithm
    algorithm_start = time.time()
    
    # Choose algorithm type first to get logger
    if algorithm_type.upper() == "EGA":
        if use_parallel and device == "cuda":
            print("Using ParallelEnhancedGeneticAlgorithm")
            algorithm = ParallelEnhancedGeneticAlgorithm(
                None, input_shape, num_classes,  # Pass None first, will set later
                elite_size=5, **default_params
            )
        else:
            print("Using EnhancedGeneticAlgorithm")
            algorithm = EnhancedGeneticAlgorithm(
                None, input_shape, num_classes,  # Pass None first, will set later
                elite_size=5, **default_params
            )
    else:
        if use_parallel and device == "cuda":
            print("Using ParallelGeneticAlgorithm")
            algorithm = ParallelGeneticAlgorithm(
                None, input_shape, num_classes,  # Pass None first, will set later
                **default_params
            )
        else:
            print("Using GeneticAlgorithm")
            algorithm = GeneticAlgorithm(
                None, input_shape, num_classes,  # Pass None first, will set later
                **default_params
            )
    
    # Now create fitness evaluator with logger
    if use_parallel and device == "cuda":
        print(f"Using ParallelFitnessEvaluator with batch size {parallel_batch_size}")
        fitness_evaluator = ParallelFitnessEvaluator(
            train_loader, val_loader, 
            device=device,
            batch_size=parallel_batch_size,
            max_epochs_parallel=3,  # Fewer epochs for faster parallel training
            logger=algorithm.logger  # Pass the logger
        )
    else:
        print("Using standard FitnessEvaluator")
        fitness_evaluator = FitnessEvaluator(
            train_loader, val_loader, 
            device=device,
            logger=algorithm.logger  # Pass the logger
        )
    
    # Set the fitness evaluator in the algorithm
    algorithm.fitness_evaluator = fitness_evaluator
    
    print(f"Running {algorithm_type} on {dataset_name}")
    best_architecture, best_fitness = algorithm.run()
    
    algorithm_time = time.time() - algorithm_start
    print(f"\nAlgorithm completed in {algorithm_time/60:.2f} minutes")
    
    # Final evaluation on test set with GPU support
    print("\nPerforming final test evaluation...")
    final_eval_start = time.time()
    
    from training_utils import NetworkTrainer
    trainer = NetworkTrainer(device=device)
    test_results = trainer.train_and_evaluate(
        best_architecture, train_loader, test_loader, epochs=5
    )
    
    final_eval_time = time.time() - final_eval_start
    total_time = time.time() - experiment_start
    
    print(f"Final evaluation completed in {final_eval_time/60:.2f} minutes")
    print(f"Total experiment time: {total_time/60:.2f} minutes")
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Algorithm: {algorithm_type}")
    print(f"Dataset: {dataset_name}")
    print(f"Best validation fitness: {best_fitness:.4f}")
    print(f"Final test accuracy: {test_results['accuracy']:.2f}%")
    print(f"Architecture layers: {len(best_architecture.layers)}")
    print(f"Parameter count: {test_results['parameter_count']:,}")
    print(f"Training time per individual: {algorithm_time/(default_params['population_size'] * default_params['num_generations']):.2f}s")
    print("="*80)
    
    return {
        'algorithm': algorithm_type,
        'dataset': dataset_name,
        'best_architecture': best_architecture,
        'best_fitness': best_fitness,
        'test_accuracy': test_results['accuracy'],
        'test_loss': test_results['loss'],
        'parameter_count': test_results['parameter_count'],
        'training_time': test_results['training_time'],
        'algorithm_time': algorithm_time,
        'total_time': total_time,
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