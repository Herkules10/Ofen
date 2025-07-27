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
            self.layers.append(layer_config)
            self._update_current_shape(layer_config)
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
        elif layer_config.layer_type in [LayerType.MAXPOOL, LayerType.DROPOUT, LayerType.ACTIVATION]:
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
            
            h_out = (h_in + 2*padding - kernel_size) // stride + 1
            w_out = (w_in + 2*padding - kernel_size) // stride + 1
            self._current_shape = (filters, h_out, w_out)
            
        elif layer_config.layer_type == LayerType.MAXPOOL:
            if len(self._current_shape) == 3:  # Conv2D
                c, h, w = self._current_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride', pool_size)
                h_out = h // stride
                w_out = w // stride
                self._current_shape = (c, h_out, w_out)
                
        elif layer_config.layer_type == LayerType.FC:
            units = layer_config.params['units']
            self._current_shape = (units,)
    
    def _recalculate_shapes(self):
        """Recalculate all shapes after layer removal."""
        self._current_shape = self.input_shape
        for layer in self.layers:
            self._update_current_shape(layer)
    
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
            layer = self._create_layer(layer_config, current_shape, i)
            if layer is not None:
                self.layers.append(layer)
                current_shape = self._get_output_shape(layer_config, current_shape)
        
        # Add final classification layer
        if len(current_shape) > 1:  # Need to flatten
            flat_size = np.prod(current_shape)
        else:
            flat_size = current_shape[0]
        
        self.classifier = nn.Linear(flat_size, self.architecture.num_classes)
    
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
            return (filters, h_out, w_out)
        
        elif layer_config.layer_type == LayerType.CONV1D:
            c_in, l_in = input_shape
            filters = layer_config.params['filters']
            kernel_size = layer_config.params['kernel_size']
            stride = layer_config.params.get('stride', 1)
            padding = layer_config.params.get('padding', 0)
            
            l_out = (l_in + 2*padding - kernel_size) // stride + 1
            return (filters, l_out)
        
        elif layer_config.layer_type == LayerType.FC:
            return (layer_config.params['units'],)
        
        elif layer_config.layer_type == LayerType.MAXPOOL:
            if len(input_shape) == 3:  # Conv2D
                c, h, w = input_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride', pool_size)
                h_out = h // stride
                w_out = w // stride
                return (c, h_out, w_out)
            elif len(input_shape) == 2:  # Conv1D
                c, l = input_shape
                pool_size = layer_config.params.get('pool_size', 2)
                stride = layer_config.params.get('stride', pool_size)
                l_out = l // stride
                return (c, l_out)
        
        # For layers that don't change shape
        return input_shape
    
    def forward(self, x):
        need_flatten = False
        
        for i, layer in enumerate(self.layers):
            layer_config = self.architecture.layers[i]
            
            # Handle flattening before FC layers
            if (layer_config.layer_type == LayerType.FC and 
                len(x.shape) > 2):
                x = x.view(x.size(0), -1)
            
            # Apply layer
            if layer_config.layer_type == LayerType.ACTIVATION:
                activation_type = layer_config.params.get('type', ActivationType.RELU)
                if activation_type == ActivationType.RELU:
                    x = F.relu(x)
                elif activation_type == ActivationType.TANH:
                    x = torch.tanh(x)
                elif activation_type == ActivationType.SIGMOID:
                    x = torch.sigmoid(x)
            else:
                x = layer(x)
        
        # Final flattening if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
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
            possible_layers.extend([LayerType.CONV2D, LayerType.MAXPOOL, LayerType.FC])
            if len(arch.layers) > 0:
                possible_layers.extend([LayerType.BATCHNORM, LayerType.DROPOUT, LayerType.ACTIVATION])
        
        elif len(current_shape) == 2:  # Can add Conv1D, MaxPool, BatchNorm, Dropout, FC
            possible_layers.extend([LayerType.CONV1D, LayerType.MAXPOOL, LayerType.FC])
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
            return LayerConfig(LayerType.CONV2D, {
                'filters': random.choice([16, 32, 64, 128, 256]),
                'kernel_size': random.choice([3, 5, 7]),
                'stride': random.choice([1, 2]),
                'padding': random.choice([0, 1, 2])
            })
        
        elif layer_type == LayerType.CONV1D:
            return LayerConfig(LayerType.CONV1D, {
                'filters': random.choice([16, 32, 64, 128]),
                'kernel_size': random.choice([3, 5, 7]),
                'stride': random.choice([1, 2]),
                'padding': random.choice([0, 1, 2])
            })
        
        elif layer_type == LayerType.FC:
            return LayerConfig(LayerType.FC, {
                'units': random.choice([64, 128, 256, 512, 1024])
            })
        
        elif layer_type == LayerType.BATCHNORM:
            return LayerConfig(LayerType.BATCHNORM, {})
        
        elif layer_type == LayerType.MAXPOOL:
            return LayerConfig(LayerType.MAXPOOL, {
                'pool_size': random.choice([2, 3, 4]),
                'stride': None  # Will use pool_size as stride
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