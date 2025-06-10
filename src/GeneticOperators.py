import numpy as np
from NeuralNetwork import FullyConnectedParams, ConvolutionalParams, BatchNormParams, LayerEncoding
import config

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
    
    for i in range(config.MAX_LAYERS):
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