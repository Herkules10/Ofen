# Parametric Optimization for Neural Architecture Search

This directory contains a comprehensive parametric optimization system for neural architecture search using three metaheuristic algorithms: Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Simulated Annealing (SA).

## 📋 Overview

The parametric optimization system performs grid search over predefined parameter spaces to find the optimal hyperparameters for each algorithm. The specific parameter ranges tested are:

### Genetic Algorithm (GA)
- **Population Size**: 40 (fixed)
- **Crossover Rate**: [0.3, 0.5, 0.7]
- **Mutation Rate**: [0.1, 0.2, 0.3]
- **Tournament Size**: [2, 3, 5]
- **Generations**: 30 (fixed)

### Particle Swarm Optimization (PSO)
- **Swarm Size**: 40 (fixed)
- **Inertia Weight (w)**: [0.4, 0.7, 0.9]
- **Cognitive Coefficient (c1)**: [1.5, 2.0, 2.5]
- **Social Coefficient (c2)**: [1.5, 2.0, 2.5]
- **Iterations**: 30 (fixed)

### Simulated Annealing (SA)
- **Initial Temperature**: [10, 100, 500]
- **Cooling Rate**: [0.85, 0.9, 0.95]
- **Max Iterations**: 1200 (fixed)

## 🗂️ File Structure

```
src/project/
├── parametric_optimization.py      # Main optimization engine
├── parametric_analysis.py          # Results analysis and visualization
├── run_parametric_optimization.py  # Simple runner script
├── PARAMETRIC_README.md            # This file
└── parametric_results/             # Results directory (created during run)
    ├── GA_parametric_results.pkl   # GA results
    ├── PSO_parametric_results.pkl  # PSO results
    ├── SA_parametric_results.pkl   # SA results
    └── comprehensive_parametric_results.pkl  # Combined results
└── parametric_figures/             # Analysis plots (created during analysis)
    ├── GA_parameter_sensitivity.png
    ├── PSO_parameter_sensitivity.png
    ├── SA_parameter_sensitivity.png
    ├── best_parameters_comparison.png
    ├── performance_distributions.png
    └── parametric_optimization_summary.csv
```

## 🚀 Quick Start

### 1. Run Complete Optimization Pipeline
```bash
cd src/project
python run_parametric_optimization.py
```

This will:
- Execute grid search for all three algorithms
- Test all parameter combinations
- Save detailed results
- Generate analysis plots and summary tables

### 2. Check Dependencies Only
```bash
python run_parametric_optimization.py --check-deps
```

### 3. Run Analysis Only (if results already exist)
```bash
python run_parametric_optimization.py --analysis-only
```

## 📊 Detailed Usage

### Manual Optimization
```python
from parametric_optimization import ParametricOptimizer

# Create optimizer
optimizer = ParametricOptimizer(project_dir=".")

# Run grid search
optimizer.run_grid_search()

# Results are saved to parametric_results/
```

### Manual Analysis
```python
from parametric_analysis import ParametricAnalyzer

# Create analyzer
analyzer = ParametricAnalyzer("parametric_results", "parametric_figures")

# Run all analyses
analyzer.analyze_all()
```

## 📈 Results and Analysis

The system generates comprehensive results including:

### 1. **Parameter Sensitivity Analysis**
- Bar plots showing how each parameter affects performance
- Error bars indicating variance across runs
- Individual plots for each algorithm

### 2. **Best Parameter Comparison**
- Side-by-side comparison of optimal configurations
- Fitness values for best parameter sets
- Algorithm-specific optimal settings

### 3. **Performance Distributions**
- Histograms of fitness values across all parameter combinations
- Statistical measures (mean, median, standard deviation)
- Distribution shape analysis

### 4. **Parameter Interactions**
- Heatmaps showing interaction effects between parameter pairs
- Identifies synergistic parameter combinations
- Helps understand parameter dependencies

### 5. **Convergence Analysis**
- Performance vs parameter combination ranking
- Runtime vs performance trade-offs
- Identification of top-performing configurations

### 6. **Summary Tables**
- LaTeX table of best parameters (for publications)
- CSV summary with statistics
- Success rates and performance metrics

## 🔧 Configuration

### Modifying Parameter Spaces
Edit the parameter grids in `parametric_optimization.py`:

```python
# Example: Adding new crossover rates for GA
self.ga_params = {
    'population_size': [40],
    'crossover_rate': [0.3, 0.5, 0.7, 0.8],  # Added 0.8
    'mutation_rate': [0.1, 0.2, 0.3],
    'tournament_size': [2, 3, 5],
    'generations': [30]
}
```

### Adjusting Run Settings
```python
# In ParametricOptimizer.__init__()
self.runs_per_combination = 3  # Number of runs per parameter set
self.max_runtime_hours = 2     # Maximum runtime per run
```

## 📊 Output Files

### Result Files (`.pkl`)
- `GA_parametric_results.pkl`: Complete GA optimization results
- `PSO_parametric_results.pkl`: Complete PSO optimization results  
- `SA_parametric_results.pkl`: Complete SA optimization results
- `comprehensive_parametric_results.pkl`: Combined results with analysis

### Analysis Files
- `*_parameter_sensitivity.png`: Parameter effect plots
- `*_parameter_interactions.png`: Parameter interaction heatmaps
- `*_parameter_convergence.png`: Performance ranking and trade-offs
- `best_parameters_comparison.png`: Cross-algorithm comparison
- `performance_distributions.png`: Fitness distribution histograms
- `best_parameters_table.tex`: LaTeX table for publications
- `parametric_optimization_summary.csv`: Statistical summary

## ⚠️ Important Notes

### Runtime Considerations
- **Total Combinations**: GA(27) + PSO(27) + SA(9) = 63 combinations
- **Estimated Runtime**: ~10-15 hours for complete grid search
- **Runs per Combination**: 3 (for statistical significance)
- **Total Experiments**: 63 × 3 = 189 individual runs

### Memory Requirements
- Each run generates neural network architectures
- Monitor system memory during execution
- Results are automatically saved incrementally

### GPU Utilization
- Automatically detects and uses CUDA if available
- Falls back to CPU if GPU unavailable
- GPU memory is managed automatically

### Error Handling
- Robust error handling for failed runs
- Partial results are saved if execution is interrupted
- Individual run failures don't stop the entire grid search

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Memory Issues**
   ```python
   # Reduce batch size or architecture complexity
   # Monitor system memory usage
   ```

3. **Runtime Too Long**
   ```python
   # Reduce runs_per_combination or parameter grid size
   # Use smaller networks for initial testing
   ```

4. **CUDA Out of Memory**
   ```python
   # The system automatically handles GPU memory
   # Will fall back to CPU if necessary
   ```

### Debug Mode
Set environment variable for detailed logging:
```bash
set PYTORCH_DEBUG=1
python run_parametric_optimization.py
```

## 📚 Understanding Results

### Best Parameter Selection
The "best" parameters are selected based on:
1. **Average Fitness**: Mean performance across multiple runs
2. **Consistency**: Low standard deviation across runs
3. **Success Rate**: Percentage of successful completions

### Statistical Significance
- Multiple runs (default: 3) ensure statistical validity
- Error bars show confidence intervals
- Standard deviations indicate parameter stability

### Performance Metrics
- **Fitness**: Primary optimization objective
- **Accuracy**: Classification performance on test set
- **Runtime**: Computational efficiency
- **Convergence**: Optimization progress over iterations

## 📈 Using Results for Final Experiments

After completing parametric optimization:

1. **Identify Best Parameters**: Check `best_parameters_comparison.png`
2. **Review Interactions**: Examine parameter interaction plots
3. **Consider Trade-offs**: Balance performance vs runtime
4. **Run Final Experiments**: Use optimal parameters for production runs

### Example: Using Best GA Parameters
```python
# From parametric optimization results
best_ga_params = {
    'population_size': 40,
    'crossover_rate': 0.7,      # Found to be optimal
    'mutation_rate': 0.2,       # Found to be optimal  
    'tournament_size': 3,       # Found to be optimal
    'generations': 30
}

# Use in final experiment
from genetic_algorithm import GeneticAlgorithm
ga = GeneticAlgorithm(**best_ga_params)
result = ga.run()
```

## 🤝 Contributing

To extend the parametric optimization:

1. **Add New Parameters**: Modify parameter grids in `parametric_optimization.py`
2. **Add New Algorithms**: Extend the system to include additional metaheuristics
3. **Custom Analysis**: Add new visualization functions to `parametric_analysis.py`
4. **Performance Metrics**: Include additional evaluation criteria

---

*This parametric optimization system provides a systematic approach to hyperparameter tuning for neural architecture search, ensuring optimal performance across different metaheuristic algorithms.*
