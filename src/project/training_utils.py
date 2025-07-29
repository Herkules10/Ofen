import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, Optional
import time
import numpy as np
from architecture_base import NetworkArchitecture

class DatasetLoader:
    """Utility class for loading and preparing datasets."""
    
    @staticmethod
    def load_cifar10(batch_size: int = 64, validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load CIFAR-10 dataset with train/val/test splits."""
        print(f"📊 Loading CIFAR-10 dataset (batch_size={batch_size}, val_split={validation_split})")
        
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        print("   📥 Downloading/loading CIFAR-10 data...")
        train_full = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # Split training into train/validation
        train_size = int((1 - validation_split) * len(train_full))
        val_size = len(train_full) - train_size
        train_dataset, val_dataset = random_split(train_full, [train_size, val_size])
        
        print(f"   📊 Dataset sizes: train={train_size}, val={val_size}, test={len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"   ✅ CIFAR-10 loaded successfully!")
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def load_mnist(batch_size: int = 64, validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load MNIST dataset with train/val/test splits."""
        print(f"📊 Loading MNIST dataset (batch_size={batch_size}, val_split={validation_split})")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        print("   📥 Downloading/loading MNIST data...")
        train_full = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Split training into train/validation
        train_size = int((1 - validation_split) * len(train_full))
        val_size = len(train_full) - train_size
        train_dataset, val_dataset = random_split(train_full, [train_size, val_size])
        
        print(f"   📊 Dataset sizes: train={train_size}, val={val_size}, test={len(test_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"   ✅ MNIST loaded successfully!")
        return train_loader, val_loader, test_loader

class NetworkTrainer:
    """Handles training and evaluation of neural networks."""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Print device information
        if self.device.type == "cuda":
            print(f"🎮 CUDA device: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print(f"💻 Using CPU device")
    
    def train_and_evaluate(self, 
                          architecture: NetworkArchitecture,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          epochs: int = 10,
                          learning_rate: float = 0.001,
                          early_stopping_patience: int = 3) -> Dict[str, float]:
        """Train and evaluate a neural network architecture."""
        
        try:
            print(f"    🔧 Building network with {len(architecture.layers)} layers...")
            # Build network
            network = architecture.build_network().to(self.device)
            
            # Check for reasonable parameter count
            param_count = sum(p.numel() for p in network.parameters())
            print(f"    📊 Network parameters: {param_count:,}")
            
            if param_count > 5_000_000:  # Too many parameters
                print(f"    ❌ Network too large ({param_count:,} parameters), skipping training")
                return {
                    'accuracy': 0.0,
                    'loss': float('inf'),
                    'parameter_count': param_count,
                    'training_time': 0.0,
                    'converged': False
                }
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
            
            print(f"    🎯 Starting training for {epochs} epochs (LR: {learning_rate})")
            best_val_accuracy = 0.0
            patience_counter = 0
            start_time = time.time()
            
            for epoch in range(epochs):
                epoch_start = time.time()
                print(f"      📈 Epoch {epoch + 1}/{epochs}", end=" | ")
                
                # Training phase
                network.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = network(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().item()
                    
                    # Limit batches for faster evaluation during architecture search
                    if batch_idx >= 50:  # Process only first 50 batches
                        break
                
                # Calculate training metrics for this epoch
                train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
                avg_train_loss = train_loss / min(batch_idx + 1, 50)
                
                # Validation phase
                val_accuracy, val_loss = self._evaluate(network, val_loader, criterion)
                scheduler.step(val_loss)
                
                # Print epoch results
                epoch_time = time.time() - epoch_start
                print(f"Train: {train_accuracy:.1f}% | Val: {val_accuracy:.1f}% | Loss: {val_loss:.3f} | Time: {epoch_time:.1f}s")
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    print(f"        ✅ New best validation accuracy: {best_val_accuracy:.2f}%")
                else:
                    patience_counter += 1
                    print(f"        ⏳ No improvement ({patience_counter}/{early_stopping_patience})")
                    if patience_counter >= early_stopping_patience:
                        print(f"        🛑 Early stopping triggered after {epoch + 1} epochs")
                        break
            
            training_time = time.time() - start_time
            
            print(f"    ✅ Training completed in {training_time:.1f}s | Final accuracy: {best_val_accuracy:.2f}%")
            
            return {
                'accuracy': best_val_accuracy,
                'loss': val_loss,
                'parameter_count': param_count,
                'training_time': training_time,
                'converged': patience_counter < early_stopping_patience
            }
            
        except Exception as e:
            print(f"    ❌ Training failed: {e}")
            
            # Clear GPU memory if using CUDA
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                print(f"    🧹 GPU memory cleared")
                
            return {
                'accuracy': 0.0,
                'loss': float('inf'),
                'parameter_count': 0,
                'training_time': 0.0,
                'converged': False
            }
    
    def _evaluate(self, network: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate network on given data loader."""
        network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = network(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Limit evaluation batches for speed
                if batch_idx >= 20:
                    break
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / min(len(data_loader), 20)
        
        return accuracy, avg_loss

class FitnessEvaluator:
    """Evaluates fitness of neural network architectures."""
    
    def __init__(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 lambda1: float = 1e-6,  # Weight for parameter count
                 lambda2: float = 1.0,   # Weight for accuracy
                 device: str = None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.trainer = NetworkTrainer(device)
        self.evaluation_cache = {}  # Cache to avoid re-evaluating identical architectures
    
    def evaluate_fitness(self, architecture: NetworkArchitecture) -> Dict[str, float]:
        """Evaluate fitness of an architecture."""
        
        # Create a hashable representation of the architecture
        arch_hash = self._hash_architecture(architecture)
        
        # Check cache first
        if arch_hash in self.evaluation_cache:
            print(f"  🔄 Using cached result for architecture")
            return self.evaluation_cache[arch_hash].copy()
        
        print(f"  🧬 Evaluating new architecture ({len(architecture.layers)} layers)")
        
        # Train and evaluate
        results = self.trainer.train_and_evaluate(
            architecture, self.train_loader, self.val_loader
        )
        
        # Calculate fitness (higher is better)
        accuracy = results['accuracy']
        param_count = results['parameter_count']
        
        # Multi-objective fitness: maximize accuracy, minimize parameters
        fitness = self.lambda2 * accuracy - self.lambda1 * param_count
        
        results['fitness'] = fitness
        results['efficiency'] = accuracy / (param_count / 1000) if param_count > 0 else 0
        
        print(f"  📊 Results: Acc={accuracy:.2f}%, Params={param_count:,}, Fitness={fitness:.2f}")
        
        # Cache result
        self.evaluation_cache[arch_hash] = results.copy()
        
        return results
    
    def _hash_architecture(self, architecture: NetworkArchitecture) -> str:
        """Create a hash string for an architecture."""
        hash_components = []
        for layer in architecture.layers:
            layer_str = f"{layer.layer_type.value}_{sorted(layer.params.items())}"
            hash_components.append(layer_str)
        return "_".join(hash_components)

class ExperimentLogger:
    """Logs experiment results and metrics."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.results = []
        self.best_architectures = []
    
    def log_generation(self, generation: int, population_fitness: list, best_architecture: NetworkArchitecture, best_fitness: float):
        """Log results for a generation/iteration."""
        gen_stats = {
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': np.mean(population_fitness),
            'std_fitness': np.std(population_fitness),
            'architecture_layers': len(best_architecture.layers),
            'architecture_params': best_architecture.get_parameter_count()
        }
        self.results.append(gen_stats)
        
        # Store best architecture if it's an improvement
        if not self.best_architectures or best_fitness > max(arch['fitness'] for arch in self.best_architectures):
            print(f"      🌟 New global best found in {self.experiment_name}!")
            self.best_architectures.append({
                'generation': generation,
                'architecture': best_architecture.copy(),
                'fitness': best_fitness
            })
        
        # Log convergence status every 10 generations
        if generation > 0 and generation % 10 == 0:
            recent_fitness = [r['best_fitness'] for r in self.results[-10:]]
            improvement = recent_fitness[-1] - recent_fitness[0]
            print(f"      📈 Convergence check: fitness improved by {improvement:.4f} over last 10 generations")
            
            if abs(improvement) < 0.001:
                print(f"      ⚠️  Low improvement detected - may be converging")
    
    def get_convergence_data(self) -> Dict:
        """Get convergence data for plotting."""
        return {
            'generations': [r['generation'] for r in self.results],
            'best_fitness': [r['best_fitness'] for r in self.results],
            'avg_fitness': [r['avg_fitness'] for r in self.results],
            'std_fitness': [r['std_fitness'] for r in self.results],
            'num_layers': [r['architecture_layers'] for r in self.results],
            'num_params': [r['architecture_params'] for r in self.results]
        }
    
    def save_results(self, filename: str):
        """Save results to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'experiment_name': self.experiment_name,
                'results': self.results,
                'best_architectures': self.best_architectures
            }, f)
    
    def load_results(self, filename: str):
        """Load results from file."""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.experiment_name = data['experiment_name']
            self.results = data['results']
            self.best_architectures = data['best_architectures']