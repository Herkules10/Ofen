# Neural Network Architecture Optimization using Metaheuristics

This project implements neural network architecture search using Genetic Algorithms (GA), Enhanced Genetic Algorithms (EGA), Particle Swarm Optimization (PSO), Simulated Annealing (SA), and Adaptive Simulated Annealing (ASA).

## Project Structure

```
├── architecture_base.py          # Base architecture classes and utilities
├── training_utils.py             # Training and evaluation utilities
├── genetic_algorithm.py          # GA and EGA implementations
├── particle_swarm_optimization.py # PSO implementation
├── simulated_annealing.py         # SA and ASA implementations
├── run_experiments.py             # Main experiment runner
├── generate_plots.py              # Results visualization
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Install Python 3.8+ and pip
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running Individual Algorithms

**Genetic Algorithm:**
```bash
python genetic_algorithm.py --dataset cifar10 --algorithm EGA --population_size 30 --num_generations 50
```

**Particle Swarm Optimization:**
```bash
python particle_swarm_optimization.py --dataset cifar10 --swarm_size 25 --num_iterations 100
```

**Simulated Annealing:**
```bash
python simulated_annealing.py --dataset cifar10 --algorithm ASA --max_iterations 1000
```

### Running Full Experiment Suite

**All algorithms with default parameters:**
```bash
python run_experiments.py --dataset cifar10 --num_runs 3
```

**Quick test (reduced parameters):**
```bash
python run_experiments.py --dataset cifar10 --num_runs 1 --quick
```

**Specific algorithms only:**
```bash
python run_experiments.py --dataset cifar10 --algorithms GA EGA PSO --num_runs 2
```

**Include baseline comparison:**
```bash
python run_experiments.py --dataset cifar10 --include_baseline --num_runs 1
```

### Generating Visualizations

**Generate all plots:**
```bash
python generate_plots.py --results_dir results --output_dir figures
```

**Specific plots only:**
```bash
python generate_plots.py --plots performance convergence efficiency
```

## Algorithm Parameters

### Genetic Algorithm (GA/EGA)
- `population_size`: Population size (default: 30)
- `num_generations`: Number of generations (default: 40)
- `crossover_prob`: Crossover probability (default: 0.7)
- `mutation_add_prob`: Layer addition mutation rate (default: 0.15)
- `mutation_del_prob`: Layer deletion mutation rate (default: 0.1)
- `mutation_param_prob`: Parameter mutation rate (default: 0.2)
- `tournament_size`: Tournament selection size (default: 3)
- `elite_size`: Elite size for EGA (default: 5)

### Particle Swarm Optimization (PSO)
- `swarm_size`: Number of particles (default: 25)
- `num_iterations`: Number of iterations (default: 80)
- `w`: Inertia weight (default: 0.7)
- `c1`: Cognitive parameter (default: 2.0)
- `c2`: Social parameter (default: 2.0)

### Simulated Annealing (SA/ASA)
- `initial_temperature`: Starting temperature (default: 100.0)
- `cooling_rate`: Temperature cooling rate (default: 0.95)
- `max_iterations`: Maximum iterations (default: 800)
- `reheat_threshold`: Iterations before reheating in ASA (default: 100)
- `reheat_factor`: Temperature increase factor in ASA (default: 2.0)

## Architecture Search Space

The search space includes:
- **Conv2D layers**: Filters (16-256), kernel sizes (3,5,7), strides (1,2)
- **Conv1D layers**: For sequential data
- **Fully Connected layers**: Units (64-1024)
- **Batch Normalization**: After conv/FC layers
- **Max Pooling**: Pool sizes (2,3,4)
- **Dropout**: Rates (0.1-0.5)
- **Activation functions**: ReLU, Tanh, Sigmoid

## Fitness Function

The multi-objective fitness function balances:
- **Accuracy maximization**: λ₂ × validation_accuracy
- **Parameter minimization**: λ₁ × parameter_count

Default weights: λ₁ = 1e-6, λ₂ = 1.0

## Output Files

**Individual Results:**
- `{ALGORITHM}_{DATASET}_run_{RUN_ID}.pkl`: Single run results
- `{ALGORITHM}_{DATASET}_all_runs.pkl`: All runs for an algorithm

**Summary:**
- `experiment_summary_{DATASET}.pkl`: Aggregated statistics
- `baseline_{DATASET}.pkl`: Random search baseline (if run)

**Visualizations:**
- `performance_comparison.png`: Accuracy and fitness comparison
- `parameter_efficiency.png`: Accuracy vs model size
- `convergence_curves.png`: Algorithm convergence
- `fitness_distributions.png`: Fitness distribution box plots
- `runtime_comparison.png`: Execution time comparison
- `architecture_characteristics.png`: Layer count and parameter distributions
- `convergence_{algorithm}.png`: Individual algorithm convergence
- `results_table.tex`: LaTeX table of results

## Example Usage Scenarios

### 1. Quick Algorithm Comparison
```bash
# Fast comparison with reduced parameters
python run_experiments.py --dataset mnist --quick --num_runs 1 --algorithms GA PSO SA
python generate_plots.py --plots performance efficiency
```

### 2. Comprehensive Study
```bash
# Full experiment with multiple runs
python run_experiments.py --dataset cifar10 --num_runs 5 --include_baseline
python generate_plots.py --plots all
```

### 3. Algorithm Tuning
```bash
# Test GA with different parameters
python genetic_algorithm.py --dataset cifar10 --population_size 50 --num_generations 100 --elite_size 10
```

### 4. Architecture Analysis
```bash
# Focus on architecture characteristics
python run_experiments.py --dataset cifar10 --algorithms EGA --num_runs 3
python generate_plots.py --plots architecture individual
```

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only (slow)
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
- **Storage**: ~2GB for datasets and results

## Expected Runtime

With default parameters on modern hardware:
- **GA/EGA**: 2-4 hours per run
- **PSO**: 3-5 hours per run  
- **SA/ASA**: 1-3 hours per run
- **Quick mode**: 30-60 minutes per algorithm

## Troubleshooting

**Common Issues:**

1. **CUDA out of memory**: Reduce batch size in `training_utils.py` or use CPU
2. **Slow training**: Enable GPU, reduce training epochs, or use `--quick` mode
3. **Import errors**: Ensure all files are in the same directory and dependencies installed
4. **Architecture evaluation fails**: Check network validity in `architecture_base.py`

**Performance Tips:**
- Use GPU acceleration for faster training
- Reduce population/swarm sizes for quicker results
- Enable caching in `FitnessEvaluator` to avoid re-evaluating identical architectures
- Use `--quick` mode for testing and development

## Research Context

This implementation supports the research paper:
"Optimizing Neural Network Architecture using Genetic Algorithms, Particle Swarm Optimization, and Simulated Annealing"

**Key Contributions:**
1. Comprehensive comparison of metaheuristic NAS methods
2. Extended search space including modern layer types
3. Multi-objective optimization balancing accuracy and efficiency
4. Adaptive algorithm variants (EGA, ASA)

**Datasets Supported:**
- CIFAR-10: 32×32 RGB images, 10 classes
- MNIST: 28×28 grayscale images, 10 classes
- Fashion-MNIST: 28×28 grayscale fashion items, 10 classes

## Citation

If you use this code in your research, please cite:

```bibtex
@article{neural_arch_metaheuristics_2025,
  title={Optimizing Neural Network Architecture using Genetic Algorithms, Particle Swarm Optimization, and Simulated Annealing},
  author={Durrant, Joshua and Hausding, Jan and Hemmi, Leander},
  journal={Optimization Methods for Engineers},
  year={2025},
  organization={ETH Zurich}
}
```

## License

This project is provided for academic and research purposes. Please see individual algorithm implementations for specific licensing details.

## Contact

For questions or issues, please contact the authors or create an issue in the project repository.

---

## Advanced Usage

### Custom Fitness Functions

Modify the `FitnessEvaluator` class in `training_utils.py` to implement custom fitness functions:

```python
def evaluate_fitness(self, architecture: NetworkArchitecture) -> Dict[str, float]:
    # Custom multi-objective fitness
    accuracy = results['accuracy']
    param_count = results['parameter_count']
    flops = calculate_flops(architecture)  # Custom FLOPS calculation
    
    fitness = self.lambda2 * accuracy - self.lambda1 * param_count - self.lambda3 * flops
    return {'fitness': fitness, 'accuracy': accuracy, 'parameter_count': param_count}
```

### Adding New Layer Types

Extend the `LayerType` enum and `ArchitectureNetwork` class in `architecture_base.py`:

```python
class LayerType(Enum):
    # ... existing types ...
    ATTENTION = "attention"
    RESIDUAL = "residual"

def _create_layer(self, layer_config: LayerConfig, current_shape: Tuple, layer_idx: int):
    # ... existing layer creation ...
    elif layer_config.layer_type == LayerType.ATTENTION:
        return CustomAttentionLayer(**layer_config.params)
```

### Custom Datasets

Add new dataset loaders in `training_utils.py`:

```python
@staticmethod
def load_custom_dataset(batch_size: int = 64):
    # Implement custom dataset loading
    transform = transforms.Compose([...])
    dataset = CustomDataset(transform=transform)
    # Return train_loader, val_loader, test_loader
```

### Distributed Training

For large-scale experiments, implement distributed training:

```python
# In training_utils.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_distributed_training():
    dist.init_process_group(backend='nccl')
    # Setup distributed training
```

### Hyperparameter Optimization

Use Optuna or similar for hyperparameter optimization:

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    population_size = trial.suggest_int('population_size', 20, 100)
    mutation_rate = trial.suggest_float('mutation_rate', 0.05, 0.3)
    
    # Run algorithm with suggested parameters
    results = run_genetic_algorithm(
        population_size=population_size,
        mutation_add_prob=mutation_rate
    )
    
    return results['test_accuracy']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```