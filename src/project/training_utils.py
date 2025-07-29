import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict, Optional, List
import time
import numpy as np
from architecture_base import NetworkArchitecture

class DatasetLoader:
    """Utility class for loading and preparing datasets."""
    
    @staticmethod
    def load_cifar10(batch_size: int = 64, validation_split: float = 0.2, train_split: float = 1.0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load CIFAR-10 dataset with train/val/test splits.
        
        Args:
            batch_size: Batch size for data loaders
            validation_split: Fraction of training data to use for validation
            train_split: Fraction of total training data to use (for testing with smaller datasets)
        """
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
        train_full = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        # Apply train_split to reduce dataset size if needed
        if train_split < 1.0:
            reduced_size = int(train_split * len(train_full))
            unused_size = len(train_full) - reduced_size
            train_full, _ = random_split(train_full, [reduced_size, unused_size])
            print(f"Using {train_split*100:.1f}% of training data ({reduced_size:,} samples)")
        
        # Split training into train/validation
        train_size = int((1 - validation_split) * len(train_full))
        val_size = len(train_full) - train_size
        train_dataset, val_dataset = random_split(train_full, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def load_mnist(batch_size: int = 64, validation_split: float = 0.2, train_split: float = 1.0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load MNIST dataset with train/val/test splits.
        
        Args:
            batch_size: Batch size for data loaders
            validation_split: Fraction of training data to use for validation
            train_split: Fraction of total training data to use (for testing with smaller datasets)
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_full = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Apply train_split to reduce dataset size if needed
        if train_split < 1.0:
            reduced_size = int(train_split * len(train_full))
            unused_size = len(train_full) - reduced_size
            train_full, _ = random_split(train_full, [reduced_size, unused_size])
            print(f"Using {train_split*100:.1f}% of training data ({reduced_size:,} samples)")
        
        # Split training into train/validation
        train_size = int((1 - validation_split) * len(train_full))
        val_size = len(train_full) - train_size
        train_dataset, val_dataset = random_split(train_full, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader

class NetworkTrainer:
    """Handles training and evaluation of neural networks."""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print("CUDA AVAILABLE:", torch.cuda.is_available())
        print(f"Using device: {self.device}")
    
    def train_and_evaluate(self, 
                          architecture: NetworkArchitecture,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          epochs: int = 3,
                          learning_rate: float = 0.001,
                          early_stopping_patience: int = 3) -> Dict[str, float]:
        """Train and evaluate a neural network architecture."""
        import time
        
        try:
            # Build network
            build_start = time.time()
            network = architecture.build_network().to(self.device)
            build_time = time.time() - build_start
            
            # Check for reasonable parameter count
            param_count = sum(p.numel() for p in network.parameters())
            print(f"      Network built in {build_time:.3f}s, {param_count:,} parameters")
            
            if param_count > 5_000_000:  # Too many parameters
                print(f"      Skipping training - too many parameters ({param_count:,})")
                return {
                    'accuracy': 0.0,
                    'loss': float('inf'),
                    'parameter_count': param_count,
                    'training_time': 0.0,
                    'converged': False
                }
            
            # Setup training
            setup_start = time.time()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
            setup_time = time.time() - setup_start
            
            best_val_accuracy = 0.0
            patience_counter = 0
            training_start = time.time()
            
            epoch_times = []
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Training phase
                train_start = time.time()
                network.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                batches_processed = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = network(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.detach().item()
                    _, predicted = output.max(1)
                    train_total += target.size(0)
                    train_correct += predicted.eq(target).sum().detach().item()
                    
                    batches_processed += 1
                    # Limit batches for faster evaluation during architecture search
                    if batch_idx >= 50:  # Process only first 50 batches
                        break
                
                train_time = time.time() - train_start
                train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
                
                # Validation phase
                val_start = time.time()
                val_accuracy, val_loss = self._evaluate(network, val_loader, criterion)
                val_time = time.time() - val_start
                
                scheduler.step(val_loss)
                
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
                
                print(f"      Epoch {epoch+1}/{epochs}: train_acc={train_acc:.1f}%, val_acc={val_accuracy:.1f}%, "
                      f"val_loss={val_loss:.4f} ({epoch_time:.2f}s, {batches_processed} batches)")
                
                # Early stopping check
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"      Early stopping at epoch {epoch+1}")
                        break
            
            training_time = time.time() - training_start
            total_time = time.time() - build_start
            
            print(f"      Training completed: {len(epoch_times)} epochs in {training_time:.2f}s")
            print(f"      Total time breakdown: build={build_time:.3f}s, setup={setup_time:.3f}s, train={training_time:.2f}s")
            
            return {
                'accuracy': best_val_accuracy,
                'loss': val_loss,
                'parameter_count': param_count,
                'training_time': training_time,
                'total_time': total_time,
                'epochs_completed': len(epoch_times),
                'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0.0,
                'converged': patience_counter < early_stopping_patience
            }
            
        except Exception as e:
            print(f"      Training failed: {e}")
            return {
                'accuracy': 0.0,
                'loss': float('inf'),
                'parameter_count': 0,
                'training_time': 0.0,
                'total_time': 0.0,
                'epochs_completed': 0,
                'avg_epoch_time': 0.0,
                'converged': False
            }
    
    def _evaluate(self, network: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate network on given data loader."""
        network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        batches_processed = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = network(data)
                loss = criterion(output, target)
                
                total_loss += loss.detach().item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().detach().item()
                
                batches_processed += 1
                # Limit evaluation batches for speed
                if batch_idx >= 20:
                    break
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / batches_processed if batches_processed > 0 else float('inf')
        
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
        import time
        
        # Create a hashable representation of the architecture
        arch_hash = self._hash_architecture(architecture)
        
        # Check cache first
        if arch_hash in self.evaluation_cache:
            cached_result = self.evaluation_cache[arch_hash].copy()
            # print(f"    Cache hit! Fitness: {cached_result['fitness']:.4f}")
            return cached_result
        
        print(f"    Training new architecture (layers: {len(architecture.layers)})...")
        eval_start = time.time()
        
        # Train and evaluate
        results = self.trainer.train_and_evaluate(
            architecture, self.train_loader, self.val_loader
        )
        
        eval_time = time.time() - eval_start
        
        # Calculate fitness (higher is better)
        accuracy = results['accuracy']
        param_count = results['parameter_count']
        
        # Multi-objective fitness: maximize accuracy, minimize parameters
        fitness = self.lambda2 * accuracy - self.lambda1 * param_count
        
        results['fitness'] = fitness
        results['efficiency'] = accuracy / (param_count / 1000) if param_count > 0 else 0
        results['evaluation_time'] = eval_time
        
        print(f"    Training completed in {eval_time:.2f}s")
        print(f"    Results: acc={accuracy:.2f}%, params={param_count:,}, fitness={fitness:.4f}")
        
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


class ParallelFitnessEvaluator:
    """GPU-parallelized fitness evaluator for multiple architectures simultaneously."""
    
    def __init__(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 lambda1: float = 1e-6,
                 lambda2: float = 1.0,
                 device: str = None,
                 batch_size: int = 4,  # Number of architectures to train in parallel
                 max_epochs_parallel: int = 5):  # Reduced epochs for parallel training
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.batch_size = batch_size
        self.max_epochs_parallel = max_epochs_parallel
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.evaluation_cache = {}
        
        # Prepare limited data for faster parallel training
        self._prepare_training_data()
        
        print(f"Parallel fitness evaluator initialized:")
        print(f"  Device: {self.device}")
        print(f"  Parallel batch size: {batch_size}")
        print(f"  Max epochs per architecture: {max_epochs_parallel}")
    
    def _prepare_training_data(self):
        """Prepare limited training data for faster evaluation."""
        # Use a subset of training data for architecture evaluation
        self.train_data_limited = []
        self.val_data_limited = []
        
        # Collect limited training batches
        train_batches_to_use = min(30, len(self.train_loader))  # Use up to 30 batches
        for i, (data, target) in enumerate(self.train_loader):
            if i >= train_batches_to_use:
                break
            self.train_data_limited.append((data, target))
        
        # Collect limited validation batches  
        val_batches_to_use = min(10, len(self.val_loader))  # Use up to 10 batches
        for i, (data, target) in enumerate(self.val_loader):
            if i >= val_batches_to_use:
                break
            self.val_data_limited.append((data, target))
            
        print(f"  Using {len(self.train_data_limited)} train batches, {len(self.val_data_limited)} val batches")
    
    def evaluate_population_parallel(self, architectures: List[NetworkArchitecture]) -> List[Dict[str, float]]:
        """Evaluate multiple architectures in parallel on GPU."""
        import time
        
        print(f"Starting parallel evaluation of {len(architectures)} architectures...")
        start_time = time.time()
        
        results = []
        
        # Process architectures in batches
        for batch_start in range(0, len(architectures), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(architectures))
            batch_archs = architectures[batch_start:batch_end]
            
            print(f"  Processing batch {batch_start//self.batch_size + 1}/{(len(architectures)-1)//self.batch_size + 1} "
                  f"({len(batch_archs)} architectures)")
            
            batch_results = self._evaluate_batch_parallel(batch_archs)
            results.extend(batch_results)
        
        total_time = time.time() - start_time
        print(f"Parallel evaluation completed in {total_time:.2f}s ({total_time/len(architectures):.2f}s per architecture)")
        
        return results
    
    def _evaluate_batch_parallel(self, architectures: List[NetworkArchitecture]) -> List[Dict[str, float]]:
        """Evaluate a batch of architectures in parallel."""
        import time
        
        batch_results = []
        
        # Check cache first
        cached_results = []
        uncached_archs = []
        uncached_indices = []
        
        for i, arch in enumerate(architectures):
            arch_hash = self._hash_architecture(arch)
            if arch_hash in self.evaluation_cache:
                cached_results.append((i, self.evaluation_cache[arch_hash].copy()))
            else:
                uncached_archs.append(arch)
                uncached_indices.append(i)
        
        print(f"    Cache hits: {len(cached_results)}/{len(architectures)}")
        
        # Train uncached architectures in parallel
        uncached_results = []
        if uncached_archs:
            print(f"    Training {len(uncached_archs)} new architectures in parallel...")
            uncached_results = self._train_batch_parallel(uncached_archs)
        
        # Combine cached and newly computed results
        all_results = [None] * len(architectures)
        
        # Fill cached results
        for idx, result in cached_results:
            all_results[idx] = result
            
        # Fill uncached results
        for i, result in enumerate(uncached_results):
            original_idx = uncached_indices[i]
            all_results[original_idx] = result
            
            # Cache the result
            arch_hash = self._hash_architecture(uncached_archs[i])
            self.evaluation_cache[arch_hash] = result.copy()
        
        return all_results
    
    def _train_batch_parallel(self, architectures: List[NetworkArchitecture]) -> List[Dict[str, float]]:
        """Train multiple architectures simultaneously on GPU."""
        import time
        
        batch_start = time.time()
        
        # Build all networks and move to GPU
        networks = []
        optimizers = []
        param_counts = []
        
        for i, arch in enumerate(architectures):
            try:
                network = arch.build_network().to(self.device)
                param_count = sum(p.numel() for p in network.parameters())
                
                # Skip if too many parameters
                if param_count > 2_000_000:  # Lower threshold for parallel training
                    networks.append(None)
                    optimizers.append(None)
                    param_counts.append(param_count)
                    continue
                
                optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-4)
                
                networks.append(network)
                optimizers.append(optimizer)
                param_counts.append(param_count)
                
            except Exception as e:
                print(f"      Failed to build network {i}: {e}")
                networks.append(None)
                optimizers.append(None)
                param_counts.append(0)
        
        # Train all networks in parallel
        criterion = torch.nn.CrossEntropyLoss()
        results = []
        
        for i, (network, optimizer, param_count) in enumerate(zip(networks, optimizers, param_counts)):
            if network is None:
                results.append({
                    'accuracy': 0.0,
                    'loss': float('inf'),
                    'parameter_count': param_count,
                    'training_time': 0.0,
                    'fitness': 0.0,
                    'efficiency': 0.0
                })
                continue
            
            # Train this network
            best_accuracy = self._train_single_network_fast(network, optimizer, criterion)
            
            # Calculate fitness
            fitness = self.lambda2 * best_accuracy - self.lambda1 * param_count
            efficiency = best_accuracy / (param_count / 1000) if param_count > 0 else 0
            
            results.append({
                'accuracy': best_accuracy,
                'loss': 0.0,  # Not tracking loss in parallel mode
                'parameter_count': param_count,
                'training_time': 0.0,  # Batch time, not individual
                'fitness': fitness,
                'efficiency': efficiency
            })
        
        batch_time = time.time() - batch_start
        print(f"    Batch training completed in {batch_time:.2f}s ({batch_time/len(architectures):.2f}s per arch)")
        
        # Clear GPU memory
        for network in networks:
            if network is not None:
                del network
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def _train_single_network_fast(self, network, optimizer, criterion) -> float:
        """Fast training of a single network with limited data."""
        network.train()
        best_val_accuracy = 0.0
        
        # Quick training for a few epochs
        for epoch in range(self.max_epochs_parallel):
            # Training phase
            for data, target in self.train_data_limited:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = network(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Quick validation
            if epoch % 2 == 1:  # Validate every other epoch
                network.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in self.val_data_limited:
                        data, target = data.to(self.device), target.to(self.device)
                        output = network(data)
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                
                val_accuracy = 100.0 * correct / total if total > 0 else 0.0
                best_val_accuracy = max(best_val_accuracy, val_accuracy)
                network.train()
        
        return best_val_accuracy
    
    def evaluate_fitness(self, architecture: NetworkArchitecture) -> Dict[str, float]:
        """Single architecture evaluation (fallback to non-parallel)."""
        results = self.evaluate_population_parallel([architecture])
        return results[0]
    
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
        
        # Store best architecture
        if not self.best_architectures or best_fitness > max(r['best_fitness'] for r in self.results[:-1]):
            self.best_architectures.append({
                'generation': generation,
                'architecture': best_architecture.copy(),
                'fitness': best_fitness
            })
    
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