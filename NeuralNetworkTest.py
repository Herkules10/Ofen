import numpy as np
import torch
from NeuralNetwork import LayerEncoding
from NeuralNetwork import FullyConnectedParams, ConvolutionalParams, BatchNormParams, LayerEncoding, NeuralNetwork
from GeneticOperators import create_mutation_functions, generate_random_genome
import config

# Example usage
def test_network():
    # Example for MNIST: input_shape=(1, 28, 28), output_size=10
    input_shape = (1, 28, 28)
    output_size = 10
    
    print("Generating random genome...")
    # Generate a random genome - Let's specify a specific sequence to test
    # Start with conv, then FC, then conv again to test transitions
    genome = []
    genome.append(LayerEncoding(True, 1, ConvolutionalParams(3, 16, 1)))  # Conv
    genome.append(LayerEncoding(True, 0, FullyConnectedParams(128)))      # FC
    genome.append(LayerEncoding(True, 1, ConvolutionalParams(3, 32, 1)))  # Conv
    genome.append(LayerEncoding(True, 2, BatchNormParams()))              # BN
    genome.append(LayerEncoding(True, 0, FullyConnectedParams(64)))       # FC
    
    # Fill remaining layers as inactive
    for i in range(5, config.MAX_LAYERS):
        genome.append(LayerEncoding(False, 0, FullyConnectedParams(64)))
    
    print("Creating network...")
    # Create network
    network = NeuralNetwork(input_shape, output_size, genome)
    
    print("Testing with input data...")
    try:
        # Test with a random input
        x = torch.randn(32, *input_shape)  # Batch size 32
        output = network(x)
        print(f"Output shape: {output.shape}")  # Should be [32, 10]
        
        print("Testing with flattened input...")
        # Also test with flattened input
        flat_x = torch.randn(32, np.prod(input_shape))
        flat_output = network(flat_x)
        print(f"Output shape from flattened input: {flat_output.shape}")
        
        print("\nTesting GA mutation operations...")
        # Test genetic algorithm operations
        mutations = create_mutation_functions(input_shape, output_size)
        
        # Make a copy of the genome for testing mutations
        test_genome = [LayerEncoding(le.active, le.layer_type, le.params) for le in genome]
        
        # Test add/remove layer
        mutated = mutations["add_remove_layer"](test_genome)
        print(f"After add/remove: {sum(1 for le in mutated if le.active)} active layers")
        
        # Test change layer type
        mutated = mutations["change_layer_type"](test_genome)
        print("After changing layer type - layer types:", [le.layer_type for le in mutated if le.active])
        
        # Test modify parameters
        mutated = mutations["modify_layer_params"](test_genome)
        
        # Test crossover
        parent2 = generate_random_genome(input_shape, output_size)
        child = mutations["crossover"](test_genome, parent2)
        print(f"Child has {sum(1 for le in child if le.active)} active layers")
        
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

# Run the test
test_network()