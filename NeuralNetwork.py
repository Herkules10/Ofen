import numpy as np
import torch
import torch.nn as nn
import config

# tuneable parameters

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
        self.max_layers = config.MAX_LAYERS
        self.layer_encodings = [le for le in layer_encodings if le.active][:config.MAX_LAYERS]
        
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

