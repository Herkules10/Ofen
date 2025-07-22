import numpy as np
import torch
from NeuralNetwork import LayerEncoding
from NeuralNetwork import FullyConnectedParams, ConvolutionalParams, BatchNormParams, LayerEncoding, NeuralNetwork, NetworkGenome
import config

# Example usage
def test_network():
    # Example for MNIST: input_shape=(1, 28, 28), output_size=10
    input_shape = (1, 28, 28)
    output_size = 10
    
    print("Generating random genome...")
    # Generate a random genome - Let's specify a specific sequence to test
    # Start with conv, then FC, then conv again to test transitions
    
    layer_list = []
    layer_list.append(LayerEncoding(1, ConvolutionalParams(3, 16, 1)))  # Conv
    layer_list.append(LayerEncoding(0, FullyConnectedParams(128)))      # FC
    layer_list.append(LayerEncoding(1, ConvolutionalParams(3, 32, 1)))  # Conv
    layer_list.append(LayerEncoding(2, BatchNormParams()))              # BN
    layer_list.append(LayerEncoding(0, FullyConnectedParams(64)))       # FC
    
    genome = NetworkGenome(layer_list=layer_list)

    print("Genome:")
    print(genome)
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
        
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")

# Run the test
test_network()