#!/usr/bin/env python3
"""
Test script to verify that all evaluations are properly logged,
including failed builds and large networks.
"""

import torch
import torch.nn as nn
from training_utils import ParallelFitnessEvaluator, ExperimentLogger
from architecture_base import NetworkArchitecture, LayerConfig, LayerType, ActivationType

def create_mock_data_loaders():
    """Create minimal data loaders for testing."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    x_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, 10, (100,))
    x_val = torch.randn(50, 3, 32, 32) 
    y_val = torch.randint(0, 10, (50,))
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=16)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=16)
    
    return train_loader, val_loader

def create_problematic_architectures():
    """Create architectures that will cause different types of failures."""
    architectures = []
    
    # 1. Normal small architecture (should work)
    arch1 = NetworkArchitecture((3, 32, 32), 10)
    arch1.add_layer(LayerConfig(LayerType.CONV2D, {'filters': 16, 'kernel_size': 3}))
    arch1.add_layer(LayerConfig(LayerType.MAXPOOL, {'pool_size': 2}))
    arch1.add_layer(LayerConfig(LayerType.FC, {'out_features': 10}))
    architectures.append(("Normal small", arch1))
    
    # 2. Large architecture (should be skipped but logged)
    arch2 = NetworkArchitecture((3, 32, 32), 10)
    arch2.add_layer(LayerConfig(LayerType.CONV2D, {'filters': 512, 'kernel_size': 3}))
    arch2.add_layer(LayerConfig(LayerType.CONV2D, {'filters': 512, 'kernel_size': 3}))
    arch2.add_layer(LayerConfig(LayerType.MAXPOOL, {'pool_size': 2}))
    arch2.add_layer(LayerConfig(LayerType.FC, {'out_features': 2048}))
    arch2.add_layer(LayerConfig(LayerType.FC, {'out_features': 10}))
    architectures.append(("Large network", arch2))
    
    # 3. Invalid architecture (should fail to build but be logged)
    arch3 = NetworkArchitecture((3, 32, 32), 10)
    arch3.add_layer(LayerConfig(LayerType.CONV2D, {'filters': 16, 'kernel_size': 33}))  # Kernel too large
    arch3.add_layer(LayerConfig(LayerType.FC, {'out_features': 10}))
    architectures.append(("Invalid (large kernel)", arch3))
    
    return architectures

def test_evaluation_logging():
    """Test that all evaluations are properly logged."""
    print("=== Testing Evaluation Logging ===")
    
    # Setup
    train_loader, val_loader = create_mock_data_loaders()
    logger = ExperimentLogger("test")
    evaluator = ParallelFitnessEvaluator(
        train_loader, val_loader, 
        batch_size=2,  # Small batch for testing
        max_epochs_parallel=1,  # Fast evaluation
        logger=logger
    )
    
    # Create test architectures
    architectures_info = create_problematic_architectures()
    architectures = [arch for _, arch in architectures_info]
    
    print(f"\nTesting {len(architectures)} architectures:")
    for i, (name, arch) in enumerate(architectures_info):
        param_count = arch.get_parameter_count()
        print(f"  {i+1}. {name}: ~{param_count:,} parameters")
    
    # Evaluate all architectures
    print(f"\nEvaluating architectures...")
    results = evaluator.evaluate_population_parallel(architectures)
    
    # Check results
    print(f"\n=== Results ===")
    print(f"Architectures evaluated: {len(architectures)}")
    print(f"Results returned: {len(results)}")
    print(f"Evaluations logged: {len(logger.evaluation_log)}")
    print(f"Failed builds: {getattr(evaluator, 'failed_builds', 0)}")
    print(f"Skipped large networks: {getattr(evaluator, 'skipped_large_networks', 0)}")
    
    # Detailed breakdown
    successful_evals = sum(1 for r in results if r['fitness'] > 0)
    failed_evals = len(results) - successful_evals
    
    print(f"\nDetailed breakdown:")
    print(f"  Successful evaluations: {successful_evals}")
    print(f"  Failed/skipped evaluations: {failed_evals}")
    print(f"  Total evaluations logged: {len(logger.evaluation_log)}")
    
    # Verify the fix
    expected_logs = len(architectures)
    actual_logs = len(logger.evaluation_log)
    
    if expected_logs == actual_logs:
        print(f"\n✅ SUCCESS: All {expected_logs} evaluations were properly logged!")
        return True
    else:
        print(f"\n❌ FAILURE: Expected {expected_logs} logs, got {actual_logs}")
        return False

if __name__ == "__main__":
    success = test_evaluation_logging()
    if success:
        print("\n🎉 The fix works! All evaluations are now properly logged.")
    else:
        print("\n⚠️  The fix needs more work.")
