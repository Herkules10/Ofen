import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from enum import Enum
from dataclasses import dataclass

class LayerType(Enum):
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    FC = "fc"
    BATCHNORM = "batchnorm"
    MAXPOOL = "maxpool"
    DROPOUT = "dropout"
    ACTIVATION = "activation"

@dataclass
class LayerConfig:
    layer_type: LayerType
    params: Dict
    
    def copy(self):
        return LayerConfig(self.layer_type, self.params.copy())

class ActivationType(Enum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"

class NetworkArchitecture:
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int):
        self.input_shape = input_shape  # (C, H, W) for images
        self.num_classes = num_classes
        self.layers: List[LayerConfig] = []
        self._current_shape = input_shape
        
    def add_layer(self, layer_config: LayerConfig) -> bool:
        """Add a layer to the architecture. Returns True if successful."""
        if self._is_valid_addition(layer_config):
            # Store original shape in case we need to revert
            original_shape = self._current_shape
            self.layers.append(layer_config)
            
            # Try to update shape
            if not self._update_current_shape(layer_config):
                # Revert if shape update failed
                self.layers.pop()
                self._current_shape = original_shape
                return False
            
            return True
        return False
    
    def remove_layer(self, index: int) -> bool:
        """Remove layer at index. Returns True if successful."""
        if 0 <= index < len(self.layers):
            self.layers.pop(index)
            self._recalculate_shapes()
            return True
        return False
    
    def get_parameter_count(self) -> int:
        """Calculate total number of parameters in the network."""
        net = self.build_network()
        return sum(p.numel() for p in net.parameters())
    
    def build_network(self) -> nn.Module:
        """Build PyTorch network from architecture."""
        return ArchitectureNetwork(self)
    
    def _is_valid_addition(self, layer_config: LayerConfig) -> bool:
        """Check if layer can be validly added."""
        if layer_config.layer_type == LayerType.CONV2D:
            return len(self._current_shape) == 3  # (C, H, W)
        elif layer_config.layer_type == LayerType.CONV1D:
            return len(self._current_shape) == 2  # (C, L)
        elif layer_config.layer_type == LayerType.FC:
            return True  # Can add FC after any layer
        elif layer_config.layer_type == LayerType.BATCHNORM:
            return len(self.layers) > 0  # Must have previous layer
        elif layer_config.layer_type == LayerType.MAXPOOL:
            # MaxPool can be applied to 2D or 3D tensors (Conv1D or Conv2D outputs)
            return len(self._current_shape) >= 2
        elif layer_config.layer_type in [LayerType.DROPOUT, LayerType.ACTIVATION]:
            return len(self.layers) > 0  # Must have previous layer
        return False
    
    def _update_current_shape(self, layer_config: LayerConfig):
        """Update current tensor shape after adding layer."""
        if layer_config.layer_type == LayerType.CONV2D:
            c_in, h_in, w_in = self._current_shape
            filters = layer_config.params['filters']
            kernel_size = layer_config.params['kernel_size']
            stride = layer_config.params.get('stride', 1)
            padding = layer_config.params.get('padding', 0)
            
            # Check if kernel size is valid for current input size with padding
            padded_h = h_in + 2 * padding
            padded_w = w_in + 2 * padding
            
            # Additional validation: kernel can't be larger than padded input
            if kernel_size > padded_h or kernel_size > padded_w:
                print(f"Warning: Kernel size {kernel_size} too large for padded input {padded_h}x{padded_w}")
                return False  # Kernel size too large for input dimensions
            
            h_out = (h_in + 2*padding - kernel_size) // stride + 1
            w_out = (w_in + 2*padding - kernel_size) // stride + 1
            
            # Ensure positive dimensions
            if h_out <= 0 or w_out <= 0:
                print(f"Warning: Invalid output dimensions {h_out}x{w_out} for conv layer")
                return False  # Invalid layer configuration
            
            self._current_shape = (filters, h_out, w_out)
            
        elif layer_config.layer_type == LayerType.CONV1D:
            c_in, l_in = self._current_shape
            filters = layer_config.params['filters']
            kernel_size = layer_config.params['kernel_size']
            stride = layer_config.params.get('stride', 1)
            padding = layer_config.params.get('padding', 0)
            
            # Check if kernel size is valid for current input size
            padded_l = l_in + 2 * padding
            if kernel_size > padded_l:
                return False  # Kernel size too large for input dimensions
            
            l_out = (l_in + 2*padding - kernel_size) // stride + 1
            
            # Ensure positive dimensions
            if l_out <= 0:
                return False  # Invalid layer configuration
            
            self._current_shape = (filters, l_out)
            
        elif layer_config.layer_type == LayerType.MAXPOOL:
            if len(self._current_shape) == 3:  # Conv2D
                c, h, w = self._current_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride')
                if stride is None:
                    stride = pool_size
                
                # More thorough validation - check actual PyTorch MaxPool2d formula
                h_out = (h - pool_size) // stride + 1
                w_out = (w - pool_size) // stride + 1
                
                # Ensure positive dimensions
                if h_out <= 0 or w_out <= 0:
                    print(f"Warning: MaxPool2d would create {h_out}x{w_out} output from {h}x{w} input")
                    print(f"  pool_size={pool_size}, stride={stride}")
                    return False  # Invalid layer configuration
                    
                self._current_shape = (c, h_out, w_out)
            elif len(self._current_shape) == 2:  # Conv1D
                c, l = self._current_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride')
                if stride is None:
                    stride = pool_size
                
                # Similar validation for 1D
                l_out = (l - pool_size) // stride + 1
                
                # Ensure positive dimensions
                if l_out <= 0:
                    print(f"Warning: MaxPool1d would create length {l_out} from {l}")
                    print(f"  pool_size={pool_size}, stride={stride}")
                    return False  # Invalid layer configuration
                    
                self._current_shape = (c, l_out)
                
        elif layer_config.layer_type == LayerType.FC:
            units = layer_config.params['units']
            self._current_shape = (units,)
        
        # For layers that don't change shape (BATCHNORM, DROPOUT, ACTIVATION)
        # _current_shape remains unchanged
        return True
    
    def _recalculate_shapes(self):
        """Recalculate all shapes after layer removal."""
        self._current_shape = self.input_shape
        valid_layers = []
        
        for layer in self.layers:
            if self._update_current_shape(layer):
                valid_layers.append(layer)
            else:
                # Stop at first invalid layer
                break
        
        self.layers = valid_layers
    
    def copy(self):
        """Create a deep copy of the architecture."""
        new_arch = NetworkArchitecture(self.input_shape, self.num_classes)
        new_arch.layers = [layer.copy() for layer in self.layers]
        new_arch._current_shape = self._current_shape
        return new_arch
    
    def get_flattened_size(self) -> int:
        """Get the size when flattened for FC layers."""
        if len(self._current_shape) == 3:  # Conv2D output
            return np.prod(self._current_shape)
        elif len(self._current_shape) == 2:  # Conv1D output
            return np.prod(self._current_shape)
        else:  # Already flattened
            return self._current_shape[0]

class ArchitectureNetwork(nn.Module):
    def __init__(self, architecture: NetworkArchitecture):
        super().__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self._build_layers()
        
    def _build_layers(self):
        current_shape = self.architecture.input_shape
        
        for i, layer_config in enumerate(self.architecture.layers):
            # Extra validation step - double-check each layer before creating it
            if not self._validate_layer_config(layer_config, current_shape):
                print(f"Warning: Invalid layer config detected at position {i}: {layer_config.layer_type.value} {layer_config.params}")
                print(f"  Current shape: {current_shape}")
                break  # Stop building at first invalid layer
                
            layer = self._create_layer(layer_config, current_shape, i)
            if layer is not None:
                self.layers.append(layer)
                current_shape = self._get_output_shape(layer_config, current_shape)
                
                # Validate shape after each layer
                if any(dim <= 0 for dim in current_shape):
                    print(f"Warning: Invalid shape {current_shape} after layer {i} ({layer_config.layer_type})")
                    # Remove the problematic layer and stop
                    if self.layers:
                        self.layers.pop()
                    break
        
        # Calculate the actual output size by doing a forward pass with dummy data
        self.classifier = None  # Will be created dynamically
        self._final_feature_size = None
        
        # Try to determine the final feature size by doing a test forward pass
        try:
            with torch.no_grad():
                # Set to eval mode to handle BatchNorm with batch_size=1
                self.eval()
                dummy_input = torch.randn(1, *self.architecture.input_shape)
                dummy_features = self._extract_features(dummy_input)
                self._final_feature_size = dummy_features.shape[1]
                # Set back to train mode
                self.train()
        except Exception as e:
            print(f"Could not determine feature size dynamically: {e}")
            # More detailed error handling for common issues
            if "Kernel size can't be greater than actual input size" in str(e):
                print("  Issue: Kernel size validation failed in PyTorch")
                print("  This indicates a layer configuration problem that bypassed our validation")
            
            # Fallback to calculated size
            print(f"Could not determine feature size dynamically: {e}")
            # Fallback to calculated size
            if len(current_shape) > 1:  # Need to flatten
                flat_size = np.prod(current_shape)
            else:
                flat_size = current_shape[0]
            
            # Ensure flat_size is positive
            if flat_size <= 0:
                print(f"Warning: Invalid flat_size {flat_size} from shape {current_shape}")
                flat_size = 128  # Fallback to a reasonable size
            
            self._final_feature_size = flat_size
        
        # Create the classifier with the correct input size
        self.classifier = nn.Linear(self._final_feature_size, self.architecture.num_classes)
    
    def _validate_layer_config(self, layer_config: LayerConfig, input_shape: Tuple) -> bool:
        """Thoroughly validate a layer configuration against the input shape."""
        if layer_config.layer_type == LayerType.CONV2D:
            if len(input_shape) != 3:
                return False
                
            c, h, w = input_shape
            kernel_size = layer_config.params.get('kernel_size', 3)
            padding = layer_config.params.get('padding', 0)
            stride = layer_config.params.get('stride', 1)
            
            # Check if kernel size is valid for padded input
            padded_h = h + 2 * padding
            padded_w = w + 2 * padding
            
            if kernel_size > padded_h or kernel_size > padded_w:
                print(f"    Kernel {kernel_size}x{kernel_size} too large for padded input {padded_h}x{padded_w}")
                return False
            
            # Check if output dimensions would be positive
            h_out = (h + 2*padding - kernel_size) // stride + 1
            w_out = (w + 2*padding - kernel_size) // stride + 1
            
            if h_out <= 0 or w_out <= 0:
                print(f"    Output dimensions {h_out}x{w_out} invalid")
                return False
            
        elif layer_config.layer_type == LayerType.MAXPOOL:
            if len(input_shape) == 3:  # Conv2D
                c, h, w = input_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride', pool_size)
                
                # Use correct PyTorch MaxPool2d formula
                h_out = (h - pool_size) // stride + 1
                w_out = (w - pool_size) // stride + 1
                
                if h_out <= 0 or w_out <= 0:
                    print(f"    MaxPool output {h_out}x{w_out} invalid from {h}x{w}")
                    return False
                    
            elif len(input_shape) == 2:  # Conv1D
                c, l = input_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride', pool_size)
                
                l_out = (l - pool_size) // stride + 1
                
                if l_out <= 0:
                    print(f"    MaxPool1d output length {l_out} invalid from {l}")
                    return False
        
        return True
    
    def _create_layer(self, layer_config: LayerConfig, current_shape: Tuple, layer_idx: int) -> Optional[nn.Module]:
        if layer_config.layer_type == LayerType.CONV2D:
            in_channels = current_shape[0]
            return nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer_config.params['filters'],
                kernel_size=layer_config.params['kernel_size'],
                stride=layer_config.params.get('stride', 1),
                padding=layer_config.params.get('padding', 0)
            )
        
        elif layer_config.layer_type == LayerType.CONV1D:
            in_channels = current_shape[0]
            return nn.Conv1d(
                in_channels=in_channels,
                out_channels=layer_config.params['filters'],
                kernel_size=layer_config.params['kernel_size'],
                stride=layer_config.params.get('stride', 1),
                padding=layer_config.params.get('padding', 0)
            )
        
        elif layer_config.layer_type == LayerType.FC:
            if len(current_shape) > 1:
                in_features = np.prod(current_shape)
            else:
                in_features = current_shape[0]
            return nn.Linear(in_features, layer_config.params['units'])
        
        elif layer_config.layer_type == LayerType.BATCHNORM:
            # For BatchNorm, we need to determine the correct number of features
            # based on the previous layer in the architecture
            if layer_idx > 0:
                prev_layer_config = self.architecture.layers[layer_idx - 1]
                if prev_layer_config.layer_type == LayerType.CONV2D:
                    num_features = prev_layer_config.params['filters']
                    return nn.BatchNorm2d(num_features)
                elif prev_layer_config.layer_type == LayerType.CONV1D:
                    num_features = prev_layer_config.params['filters']
                    return nn.BatchNorm1d(num_features)
                elif prev_layer_config.layer_type == LayerType.FC:
                    num_features = prev_layer_config.params['units']
                    return nn.BatchNorm1d(num_features)
            
            # Fallback to current_shape based approach if we can't determine from previous layer
            if len(current_shape) == 3:  # After Conv2D
                return nn.BatchNorm2d(current_shape[0])
            elif len(current_shape) == 2:  # After Conv1D
                return nn.BatchNorm1d(current_shape[0])
            else:  # After FC
                return nn.BatchNorm1d(current_shape[0])
        
        elif layer_config.layer_type == LayerType.MAXPOOL:
            if len(current_shape) == 3:  # Conv2D
                return nn.MaxPool2d(
                    kernel_size=layer_config.params.get('pool_size', 2),
                    stride=layer_config.params.get('stride', None)
                )
            elif len(current_shape) == 2:  # Conv1D
                return nn.MaxPool1d(
                    kernel_size=layer_config.params.get('pool_size', 2),
                    stride=layer_config.params.get('stride', None)
                )
        
        elif layer_config.layer_type == LayerType.DROPOUT:
            return nn.Dropout(layer_config.params.get('rate', 0.5))
        
        return None
    
    def _get_output_shape(self, layer_config: LayerConfig, input_shape: Tuple) -> Tuple:
        if layer_config.layer_type == LayerType.CONV2D:
            c_in, h_in, w_in = input_shape
            filters = layer_config.params['filters']
            kernel_size = layer_config.params['kernel_size']
            stride = layer_config.params.get('stride', 1)
            padding = layer_config.params.get('padding', 0)
            
            h_out = (h_in + 2*padding - kernel_size) // stride + 1
            w_out = (w_in + 2*padding - kernel_size) // stride + 1
            
            # Ensure positive dimensions
            h_out = max(1, h_out)
            w_out = max(1, w_out)
            
            return (filters, h_out, w_out)
        
        elif layer_config.layer_type == LayerType.CONV1D:
            c_in, l_in = input_shape
            filters = layer_config.params['filters']
            kernel_size = layer_config.params['kernel_size']
            stride = layer_config.params.get('stride', 1)
            padding = layer_config.params.get('padding', 0)
            
            l_out = (l_in + 2*padding - kernel_size) // stride + 1
            l_out = max(1, l_out)  # Ensure positive dimension
            
            return (filters, l_out)
        
        elif layer_config.layer_type == LayerType.FC:
            return (layer_config.params['units'],)
        
        elif layer_config.layer_type == LayerType.MAXPOOL:
            if len(input_shape) == 3:  # Conv2D
                c, h, w = input_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride')
                if stride is None:
                    stride = pool_size
                
                # Use correct PyTorch MaxPool2d formula
                h_out = (h - pool_size) // stride + 1
                w_out = (w - pool_size) // stride + 1
                
                # Ensure at least 1x1 output
                h_out = max(1, h_out)
                w_out = max(1, w_out)
                return (c, h_out, w_out)
            elif len(input_shape) == 2:  # Conv1D
                c, l = input_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride')
                if stride is None:
                    stride = pool_size
                    
                # Use correct PyTorch MaxPool1d formula
                l_out = (l - pool_size) // stride + 1
                l_out = max(1, l_out)
                return (c, l_out)
        
        # For layers that don't change shape
        return input_shape
    
    def _extract_features(self, x):
        """Extract features without applying the classifier."""
        layer_module_idx = 0  # Index for self.layers (actual PyTorch modules)
        
        # Process all layers in the architecture
        for layer_config in self.architecture.layers:
            
            # Handle flattening before FC layers
            if (layer_config.layer_type == LayerType.FC and 
                len(x.shape) > 2):
                x = x.view(x.size(0), -1)
            
            # Apply layer
            if layer_config.layer_type == LayerType.ACTIVATION:
                # Handle activation layers directly (they're not in self.layers)
                activation_type = layer_config.params.get('type', ActivationType.RELU)
                if activation_type == ActivationType.RELU:
                    x = F.relu(x)
                elif activation_type == ActivationType.TANH:
                    x = torch.tanh(x)
                elif activation_type == ActivationType.SIGMOID:
                    x = torch.sigmoid(x)
            else:
                # Apply the actual PyTorch module
                if layer_module_idx < len(self.layers):
                    x = self.layers[layer_module_idx](x)
                    layer_module_idx += 1
        
        # Final flattening if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        return x
    
    def forward(self, x):
        # Extract features
        x = self._extract_features(x)
        
        # Create classifier dynamically if not already created or size mismatch
        if (self.classifier is None or 
            self.classifier.in_features != x.shape[1]):
            self.classifier = nn.Linear(x.shape[1], self.architecture.num_classes)
            # Move to same device as input
            if x.is_cuda:
                self.classifier = self.classifier.cuda()
        
        # Apply classifier
        x = self.classifier(x)
        return x

class ArchitectureGenerator:
    """Utility class for generating random architectures."""
    
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int, max_layers: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_layers = max_layers
    
    def generate_random_architecture(self) -> NetworkArchitecture:
        """Generate a random valid architecture."""
        arch = NetworkArchitecture(self.input_shape, self.num_classes)
        num_layers = random.randint(1, self.max_layers)
        
        for _ in range(num_layers):
            layer_config = self._generate_random_layer(arch)
            if layer_config and arch.add_layer(layer_config):
                continue
            else:
                break
        
        # Ensure we have at least one meaningful layer
        if len(arch.layers) == 0:
            # Add a simple conv layer
            if len(self.input_shape) == 3:  # Image data
                layer_config = LayerConfig(
                    LayerType.CONV2D,
                    {'filters': 32, 'kernel_size': 3, 'padding': 1}
                )
            else:
                layer_config = LayerConfig(
                    LayerType.FC,
                    {'units': 128}
                )
            arch.add_layer(layer_config)
        
        return arch
    
    def _generate_random_layer(self, arch: NetworkArchitecture) -> Optional[LayerConfig]:
        """Generate a random layer configuration."""
        current_shape = arch._current_shape
        
        # Choose layer type based on current architecture state
        possible_layers = []
        
        if len(current_shape) == 3:  # Can add Conv2D, MaxPool, BatchNorm, Dropout, FC
            possible_layers.extend([LayerType.CONV2D, LayerType.FC])
            
            # Only add MaxPool if spatial dimensions are large enough to survive pooling
            h, w = current_shape[1], current_shape[2]
            # Require at least 4x4 to safely add MaxPool (will result in at least 1x1 after pooling)
            if h >= 4 and w >= 4:
                possible_layers.append(LayerType.MAXPOOL)
                
            if len(arch.layers) > 0:
                possible_layers.extend([LayerType.BATCHNORM, LayerType.DROPOUT, LayerType.ACTIVATION])
        
        elif len(current_shape) == 2:  # Can add Conv1D, MaxPool, BatchNorm, Dropout, FC
            possible_layers.extend([LayerType.CONV1D, LayerType.FC])
            
            # Only add MaxPool if length dimension is large enough
            l = current_shape[1]
            if l >= 4:  # Need at least length 4 to pool safely
                possible_layers.append(LayerType.MAXPOOL)
                
            if len(arch.layers) > 0:
                possible_layers.extend([LayerType.BATCHNORM, LayerType.DROPOUT, LayerType.ACTIVATION])
        
        else:  # Already flattened, can add FC, BatchNorm, Dropout
            possible_layers.append(LayerType.FC)
            if len(arch.layers) > 0:
                possible_layers.extend([LayerType.BATCHNORM, LayerType.DROPOUT, LayerType.ACTIVATION])
        
        if not possible_layers:
            return None
        
        layer_type = random.choice(possible_layers)
        
        if layer_type == LayerType.CONV2D:
            # Choose kernel size based on current spatial dimensions
            current_shape = arch._current_shape
            h, w = current_shape[1], current_shape[2]
            
            # Be more conservative with kernel sizes for small inputs
            max_possible_kernel = min(h, w)
            
            # Only consider kernel sizes that are definitely valid
            valid_kernel_configs = []
            
            for kernel_size in [1, 3, 5, 7]:
                if kernel_size > max_possible_kernel:
                    continue  # Skip kernels that are too large
                    
                for padding in [0, 1, 2]:
                    for stride in [1, 2]:
                        # Calculate output dimensions
                        h_out = (h + 2*padding - kernel_size) // stride + 1
                        w_out = (w + 2*padding - kernel_size) // stride + 1
                        
                        # Check if this configuration is valid
                        if (h_out > 0 and w_out > 0 and 
                            kernel_size <= (h + 2*padding) and 
                            kernel_size <= (w + 2*padding)):
                            valid_kernel_configs.append({
                                'kernel_size': kernel_size,
                                'padding': padding,
                                'stride': stride
                            })
            
            # If no valid configurations, create a minimal valid one
            if not valid_kernel_configs:
                kernel_size = min(3, max_possible_kernel, 1) if max_possible_kernel > 0 else 1
                padding = 0
                stride = 1
            else:
                # Choose a random valid configuration
                config = random.choice(valid_kernel_configs)
                kernel_size = config['kernel_size']
                padding = config['padding'] 
                stride = config['stride']
            
            return LayerConfig(LayerType.CONV2D, {
                'filters': random.choice([16, 32, 64, 128, 256]),
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding
            })
        
        elif layer_type == LayerType.CONV1D:
            # Choose kernel size based on current length dimension
            current_shape = arch._current_shape
            l = current_shape[1]
            
            # Select appropriate kernel sizes based on input size
            possible_kernel_sizes = []
            for kernel_size in [3, 5, 7]:
                for padding in [0, 1, 2]:
                    padded_l = l + 2 * padding
                    if kernel_size <= padded_l:
                        possible_kernel_sizes.append(kernel_size)
                        break  # Found valid padding for this kernel size
            
            # If no valid kernel sizes, use minimum valid kernel size
            if not possible_kernel_sizes:
                max_possible_kernel = l
                possible_kernel_sizes = [min(3, max_possible_kernel)] if max_possible_kernel > 0 else [1]
            
            kernel_size = random.choice(possible_kernel_sizes)
            
            # Choose appropriate padding for selected kernel size
            valid_paddings = []
            for padding in [0, 1, 2]:
                padded_l = l + 2 * padding
                if kernel_size <= padded_l:
                    valid_paddings.append(padding)
            
            padding = random.choice(valid_paddings) if valid_paddings else 0
            
            # Choose stride that ensures positive output dimensions
            valid_strides = []
            for stride in [1, 2]:
                l_out = (l + 2*padding - kernel_size) // stride + 1
                if l_out > 0:
                    valid_strides.append(stride)
            
            stride = random.choice(valid_strides) if valid_strides else 1
            
            return LayerConfig(LayerType.CONV1D, {
                'filters': random.choice([16, 32, 64, 128]),
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding
            })
        
        elif layer_type == LayerType.FC:
            return LayerConfig(LayerType.FC, {
                'units': random.choice([64, 128, 256, 512, 1024])
            })
        
        elif layer_type == LayerType.BATCHNORM:
            return LayerConfig(LayerType.BATCHNORM, {})
        
        elif layer_type == LayerType.MAXPOOL:
            # Choose pool size based on current dimensions with stricter validation
            current_shape = arch._current_shape
            
            if len(current_shape) == 3:  # Conv2D
                h, w = current_shape[1], current_shape[2]
                
                # Be very conservative - ensure output will be at least 1x1
                valid_pool_configs = []
                for pool_size in [2, 3, 4]:
                    for stride in [None, pool_size, pool_size // 2 if pool_size > 2 else pool_size]:
                        if stride is None:
                            stride = pool_size
                        
                        h_out = h // stride
                        w_out = w // stride
                        
                        # Only allow if output is at least 1x1
                        if h_out >= 1 and w_out >= 1:
                            valid_pool_configs.append({
                                'pool_size': pool_size,
                                'stride': stride
                            })
                
                # If no valid configurations, skip MaxPool entirely
                if not valid_pool_configs:
                    return None  # This will cause the generator to try a different layer type
                
                config = random.choice(valid_pool_configs)
                pool_size = config['pool_size']
                stride = config['stride']
                
            elif len(current_shape) == 2:  # Conv1D
                l = current_shape[1]
                
                # Similar validation for 1D
                valid_pool_configs = []
                for pool_size in [2, 3, 4]:
                    for stride in [None, pool_size]:
                        if stride is None:
                            stride = pool_size
                        
                        l_out = l // stride
                        
                        # Only allow if output is at least length 1
                        if l_out >= 1:
                            valid_pool_configs.append({
                                'pool_size': pool_size,
                                'stride': stride
                            })
                
                # If no valid configurations, skip MaxPool
                if not valid_pool_configs:
                    return None
                
                config = random.choice(valid_pool_configs)
                pool_size = config['pool_size']
                stride = config['stride']
            else:
                return None  # Can't pool already flattened data
            
            return LayerConfig(LayerType.MAXPOOL, {
                'pool_size': pool_size,
                'stride': stride
            })
        
        elif layer_type == LayerType.DROPOUT:
            return LayerConfig(LayerType.DROPOUT, {
                'rate': random.uniform(0.1, 0.5)
            })
        
        elif layer_type == LayerType.ACTIVATION:
            return LayerConfig(LayerType.ACTIVATION, {
                'type': random.choice(list(ActivationType))
            })
        
        return None