import torch
import torch.nn as nn
import numpy as np

# tuneable parameters
MAX_LAYERS = 10
# Example for the types of layers, subject to change/expand
layer_type_dict = { 0: "fully_connected",
                    1: "convolutional",
                    2: "batch_normalization"}


class LayerParams():
    def __init__(self):
        pass

class FullyConnectedParams(LayerParams):
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

class ConvolutionalParams(LayerParams):
    def __init__(self, kernel_size, num_filters, stride):
        self.kernel_size = kernel_size
        self.num_filters = num_filters  # Output channels
        self.stride = stride
        self.padding = 'same'  # Use 'same' padding to maintain spatial dimensions
        
class BatchNormParams(LayerParams):
    def __init__(self):
        pass

class LayerEncoding():
    def __init__(self, active: bool, layer_type: int, params: LayerParams):
        self.active = active
        self.layer_type = layer_type
        self.params = params

# Custom module for reshaping between FC and Conv
class ReshapeLayer(nn.Module):
    def __init__(self, input_features, output_channels=1):
        super(ReshapeLayer, self).__init__()
        self.input_features = input_features
        self.output_channels = output_channels
        
        # Calculate height and width to exactly match input_features
        # First try: find factors of input_features that are as close to square as possible
        factors = []
        for i in range(1, int(np.sqrt(input_features)) + 1):
            if input_features % i == 0:
                factors.append((i, input_features // i))
        
        # If we found factors, use the pair closest to square
        if factors:
            # Sort by difference between height and width
            factors.sort(key=lambda x: abs(x[0] - x[1]))
            height, width = factors[0]
        else:
            # If no exact factors, use a rectangular shape with width=1
            height = input_features
            width = 1
            
        self.output_shape = (output_channels, height, width)
        print(f"ReshapeLayer: {input_features} -> {self.output_shape} (total elements: {output_channels * height * width})")
    
    def forward(self, x):
        batch_size = x.shape[0]
        # Validate shape
        if x.shape[1] != self.input_features:
            raise ValueError(f"Expected {self.input_features} features, got {x.shape[1]}")
        
        # Reshape
        return x.view(batch_size, *self.output_shape)

class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, layer_encodings: list[LayerEncoding]):
        super(NeuralNetwork, self).__init__()
        self.first_input_shape = input_shape
        self.output_shape = output_shape
        self.max_layers = MAX_LAYERS
        self.layer_encodings = [le for le in layer_encodings if le.active][:MAX_LAYERS]
        
        # Decompose the complex network building into separate steps
        self.all_layers = nn.ModuleList()  # Contains all layers including transitions
        self.layer_types = []  # Tracks the type of each layer for debugging
        
        # Determine if the first layer is convolutional to handle input reshaping later
        self.first_layer_is_conv = False
        if len(self.layer_encodings) > 0 and layer_type_dict[self.layer_encodings[0].layer_type] == "convolutional":
            self.first_layer_is_conv = True
        
        self.build_network()
        
        # Print the network structure for debugging
        print("Network structure:")
        for i, (layer, layer_type) in enumerate(zip(self.all_layers, self.layer_types)):
            print(f"Layer {i}: {layer_type} -> {layer}")

    def build_network(self):
        current_shape = self.first_input_shape
        prev_layer_type = None  # Track previous layer type
            
        # Process each layer encoding
        for i, encoding in enumerate(self.layer_encodings):
            layer_type = layer_type_dict[encoding.layer_type]
            
            # Check if we need transition layers between different types
            if prev_layer_type == "fully_connected" and layer_type == "convolutional":
                # Need to add a reshape layer from FC to Conv
                fc_size = current_shape[0]
                
                # Create reshape layer with 1 channel by default
                reshape_layer = ReshapeLayer(fc_size, 1)
                self.all_layers.append(reshape_layer)
                self.layer_types.append("reshape_fc_to_conv")
                
                # Update current shape for the next layer
                current_shape = reshape_layer.output_shape
            
            # Create the actual layer
            if layer_type == "fully_connected":
                if len(current_shape) > 1:
                    # Need to flatten first
                    flatten_layer = nn.Flatten()
                    self.all_layers.append(flatten_layer)
                    self.layer_types.append("flatten")
                    
                    # Update shape
                    flattened_size = np.prod(current_shape)
                    current_shape = (flattened_size,)
                
                # Now add the FC layer
                fc_layer = nn.Sequential(
                    nn.Linear(current_shape[0], encoding.params.num_neurons),
                    nn.ReLU()
                )
                self.all_layers.append(fc_layer)
                self.layer_types.append("fully_connected")
                
                # Update shape
                current_shape = (encoding.params.num_neurons,)
                
            elif layer_type == "convolutional":
                # Ensure we have a convolutional shape
                if len(current_shape) == 1:
                    # This should be handled by the transition above, but just in case
                    print(f"Warning: Expected convolutional shape but got {current_shape}")
                
                # Add the convolutional layer
                in_channels = current_shape[0]
                out_channels = encoding.params.num_filters
                kernel_size = encoding.params.kernel_size
                
                conv_layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, 
                            padding='same', stride=encoding.params.stride),
                    nn.ReLU()
                )
                self.all_layers.append(conv_layer)
                self.layer_types.append("convolutional")
                
                # Update shape (height and width stay the same with 'same' padding)
                current_shape = (out_channels, current_shape[1], current_shape[2])
                
            elif layer_type == "batch_normalization":
                if len(current_shape) > 1:  # Conv shape
                    bn_layer = nn.BatchNorm2d(current_shape[0])
                    self.all_layers.append(bn_layer)
                    self.layer_types.append("batch_norm_2d")
                else:  # FC shape
                    bn_layer = nn.BatchNorm1d(current_shape[0])
                    self.all_layers.append(bn_layer)
                    self.layer_types.append("batch_norm_1d")
                
                # Shape stays the same
            
            # Update prev_layer_type for next iteration
            prev_layer_type = layer_type
            
        # Add output layer
        if len(current_shape) > 1:
            # Need to flatten
            flatten_layer = nn.Flatten()
            self.all_layers.append(flatten_layer)
            self.layer_types.append("flatten")
            
            # Update shape
            flattened_size = np.prod(current_shape)
            current_shape = (flattened_size,)
        
        # Add final linear layer
        output_layer = nn.Linear(current_shape[0], self.output_shape)
        self.all_layers.append(output_layer)
        self.layer_types.append("output")

    def forward(self, x):
        # Check if input needs reshaping
        if len(x.shape) == 2 and len(self.first_input_shape) > 1:
            # Flat input (batch_size, features) but network expects multi-dimensional input
            batch_size = x.shape[0]
            expected_features = np.prod(self.first_input_shape)
            
            # Validate the input dimensions
            if x.shape[1] != expected_features:
                raise ValueError(f"Input features ({x.shape[1]}) don't match expected shape {self.first_input_shape} ({expected_features} elements)")
            
            # Reshape to expected input format
            x = x.view(batch_size, *self.first_input_shape)
            print(f"Reshaped input from {[batch_size, expected_features]} to {x.shape}")
        
        # Pass through all layers
        for i, layer in enumerate(self.all_layers):
            try:
                x = layer(x)
            except Exception as e:
                print(f"Error at layer {i} ({self.layer_types[i]}): {e}")
                print(f"Input shape: {x.shape}")
                raise e
        
        return x

# Function to create genetic algorithm mutation operations
def create_mutation_functions(input_shape, output_shape):
    """Create functions for mutating network genomes"""
    
    def add_remove_layer(genome, p_add=0.5):
        """Add or remove a layer with given probability"""
        active_layers = [i for i, le in enumerate(genome) if le.active]
        inactive_layers = [i for i, le in enumerate(genome) if not le.active]
        
        if np.random.random() < p_add and inactive_layers:
            # Add a layer
            idx = np.random.choice(inactive_layers)
            # Randomly choose layer type
            layer_type = np.random.randint(0, 3)
            
            # Initialize parameters based on layer type
            if layer_type == 0:  # FC
                num_neurons = np.random.randint(32, 512)
                params = FullyConnectedParams(num_neurons)
            elif layer_type == 1:  # Conv
                kernel_size = np.random.choice([1, 3, 5])
                num_filters = np.random.randint(8, 64)
                params = ConvolutionalParams(kernel_size, num_filters, stride=1)
            elif layer_type == 2:  # BatchNorm
                params = BatchNormParams()
                
            genome[idx] = LayerEncoding(True, layer_type, params)
        elif active_layers and len(active_layers) > 1:  # Keep at least one layer
            # Remove a layer
            idx = np.random.choice(active_layers)
            genome[idx].active = False
            
        return genome
    
    def change_layer_type(genome):
        """Change the type of a random active layer"""
        active_indices = [i for i, le in enumerate(genome) if le.active]
        if not active_indices:
            return genome
            
        # Select a random active layer
        idx = np.random.choice(active_indices)
        old_type = genome[idx].layer_type
        
        # Choose a new type different from the current one
        new_type = np.random.choice([t for t in [0, 1, 2] if t != old_type])
        
        # Create new parameters appropriate for the layer type
        if new_type == 0:  # FC
            num_neurons = np.random.randint(32, 512)
            params = FullyConnectedParams(num_neurons)
        elif new_type == 1:  # Conv
            kernel_size = np.random.choice([1, 3, 5])
            num_filters = np.random.randint(8, 64)
            params = ConvolutionalParams(kernel_size, num_filters, stride=1)
        elif new_type == 2:  # BatchNorm
            params = BatchNormParams()
            
        genome[idx] = LayerEncoding(True, new_type, params)
        return genome
    
    def modify_layer_params(genome):
        """Modify parameters of a random active layer"""
        active_indices = [i for i, le in enumerate(genome) if le.active]
        if not active_indices:
            return genome
            
        # Select a random active layer
        idx = np.random.choice(active_indices)
        layer_encoding = genome[idx]
        layer_type = layer_encoding.layer_type
        
        if layer_type == 0:  # FC
            # Mutate number of neurons
            current = layer_encoding.params.num_neurons
            # Change by up to ±25%
            delta = int(current * np.random.uniform(-0.25, 0.25))
            new_value = max(16, current + delta)  # Minimum 16 neurons
            layer_encoding.params.num_neurons = new_value
            
        elif layer_type == 1:  # Conv
            # Mutate kernel size or number of filters
            if np.random.random() < 0.5:
                # Mutate kernel size
                layer_encoding.params.kernel_size = np.random.choice([1, 3, 5, 7])
            else:
                # Mutate number of filters
                current = layer_encoding.params.num_filters
                delta = int(current * np.random.uniform(-0.25, 0.25))
                new_value = max(4, current + delta)  # Minimum 4 filters
                layer_encoding.params.num_filters = new_value
                
        # For BatchNorm, there are no parameters to mutate
                
        return genome
    
    def crossover(parent1, parent2):
        """Create a child genome by combining two parents"""
        # Use single-point crossover
        crossover_point = np.random.randint(1, len(parent1))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    return {
        "add_remove_layer": add_remove_layer,
        "change_layer_type": change_layer_type,
        "modify_layer_params": modify_layer_params,
        "crossover": crossover
    }

# Example of how to generate a valid genome for this architecture
def generate_random_genome(input_shape, output_size, num_layers=5):
    genome = []
    
    for i in range(MAX_LAYERS):
        # Make first few layers active, rest inactive
        active = i < num_layers
        
        if active:
            # Randomly choose layer type
            layer_type = np.random.randint(0, 3)
            
            # Initialize parameters based on layer type
            if layer_type == 0:  # FC
                num_neurons = np.random.randint(32, 512)
                params = FullyConnectedParams(num_neurons)
                
            elif layer_type == 1:  # Conv
                kernel_size = np.random.choice([1, 3, 5])
                num_filters = np.random.randint(8, 64)
                params = ConvolutionalParams(kernel_size, num_filters, stride=1)
                
            elif layer_type == 2:  # BatchNorm
                params = BatchNormParams()
                
        else:
            # Inactive layer, parameters don't matter
            layer_type = 0
            params = FullyConnectedParams(64)
        
        genome.append(LayerEncoding(active, layer_type, params))
    
    return genome

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
    for i in range(5, MAX_LAYERS):
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