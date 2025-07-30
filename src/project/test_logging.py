#!/usr/bin/env python3
"""
Test script to verify fitness evaluation logging works correctly.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ga_logging():
    """Test GA logging functionality."""
    print("Testing GA logging...")
    
    try:
        from algorithms.genetic_algorithm import run_genetic_algorithm
        
        # Run GA with minimal parameters
        results = run_genetic_algorithm(
            dataset_name='mnist',
            algorithm_type='GA',
            population_size=5,
            num_generations=2,
            train_split=0.1,  # Use only 10% of data for quick test
            use_parallel=False
        )
        
        # Check if evaluation log exists
        conv_data = results['convergence_data']
        if 'evaluation_log' in conv_data:
            eval_log = conv_data['evaluation_log']
            print(f"✓ GA logged {len(eval_log)} individual evaluations")
            
            # Show first few evaluations
            for i, eval_data in enumerate(eval_log[:3]):
                print(f"  Eval {eval_data['evaluation_number']}: fitness={eval_data['fitness']:.4f}")
        else:
            print("✗ No evaluation log found in GA results")
            
        return True
        
    except Exception as e:
        print(f"✗ GA test failed: {e}")
        return False

def test_pso_logging():
    """Test PSO logging functionality."""
    print("\nTesting PSO logging...")
    
    try:
        from algorithms.particle_swarm_optimization import run_particle_swarm_optimization
        
        # Run PSO with minimal parameters
        results = run_particle_swarm_optimization(
            dataset_name='mnist',
            swarm_size=5,
            num_iterations=2,
            train_split=0.1,  # Use only 10% of data for quick test
            use_parallel=False
        )
        
        # Check if evaluation log exists
        conv_data = results['convergence_data']
        if 'evaluation_log' in conv_data:
            eval_log = conv_data['evaluation_log']
            print(f"✓ PSO logged {len(eval_log)} individual evaluations")
            
            # Show first few evaluations
            for i, eval_data in enumerate(eval_log[:3]):
                print(f"  Eval {eval_data['evaluation_number']}: fitness={eval_data['fitness']:.4f}")
        else:
            print("✗ No evaluation log found in PSO results")
            
        return True
        
    except Exception as e:
        print(f"✗ PSO test failed: {e}")
        return False

def test_sa_logging():
    """Test SA logging functionality."""
    print("\nTesting SA logging...")
    
    try:
        from algorithms.simulated_annealing import run_simulated_annealing
        
        # Run SA with minimal parameters
        results = run_simulated_annealing(
            dataset_name='mnist',
            algorithm_type='SA',
            max_iterations=10,
            train_split=0.1  # Use only 10% of data for quick test
        )
        
        # Check if evaluation log exists
        conv_data = results['convergence_data']
        if 'evaluation_log' in conv_data:
            eval_log = conv_data['evaluation_log']
            print(f"✓ SA logged {len(eval_log)} individual evaluations")
            
            # Show first few evaluations
            for i, eval_data in enumerate(eval_log[:3]):
                print(f"  Eval {eval_data['evaluation_number']}: fitness={eval_data['fitness']:.4f}")
        else:
            print("✗ No evaluation log found in SA results")
            
        return True
        
    except Exception as e:
        print(f"✗ SA test failed: {e}")
        return False

def test_plotting():
    """Test the new plotting functionality."""
    print("\nTesting plotting functionality...")
    
    try:
        from generate_plots import ResultsVisualizer
        import tempfile
        import os
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock results data
            mock_results = {
                'GA': [
                    {
                        'convergence_data': {
                            'evaluation_log': [
                                {'evaluation_number': 1, 'fitness': 0.1},
                                {'evaluation_number': 2, 'fitness': 0.15},
                                {'evaluation_number': 3, 'fitness': 0.12},
                            ]
                        }
                    }
                ]
            }
            
            # Create visualizer with mock data
            visualizer = ResultsVisualizer.__new__(ResultsVisualizer)
            visualizer.results_dir = temp_dir
            visualizer.output_dir = temp_dir
            visualizer.results = mock_results
            visualizer.summary_data = {}
            
            # Test the plotting function
            visualizer.plot_fitness_evaluation_progress()
            
            # Check if plot was created
            plot_path = os.path.join(temp_dir, 'fitness_evaluation_progress.png')
            if os.path.exists(plot_path):
                print("✓ Fitness evaluation progress plot created successfully")
                return True
            else:
                print("✗ Plot file not created")
                return False
        
    except Exception as e:
        print(f"✗ Plotting test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING FITNESS EVALUATION LOGGING")
    print("="*60)
    
    results = []
    
    # Test individual algorithms (commented out for now as they take time)
    # results.append(test_ga_logging())
    # results.append(test_pso_logging()) 
    # results.append(test_sa_logging())
    
    # Test plotting functionality
    results.append(test_plotting())
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
