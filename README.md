# Optimization Methods for Engineers - Project

TODO: Bullet points -> real text

- Report is found in `report`
- `src` has all source files
- idk man
- ...
- ...
- [meow](https://soggy.cat/) or so


### Datasets used

- Images
    - FashionMNIST
    - CIFAR100
    - [Medical QA](https://huggingface.co/datasets/lavita/medical-qa-shared-task-v1-toy)
- Text
    - [Amazon Reviews](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Audio 
    - [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- Graph / Chemical 
    - [MolTextNet](https://huggingface.co/datasets/liuganghuggingface/moltextnet)
    - Could use this package [torch-molecule](https://github.com/liugangcode/torch-molecule)

### Python packages

- PySwarms (for Particle Swarm Optimizations)
- PyGAD (for Genetic Algorithms)
- SciPy (for Simulated Annealing / Brute Force)

### Genome Definition

Variable length vector G=[(t1​,p1​),(t2​,p2​),...,(tn​,pn​)] where ti​∈{1,2,3,...} (1=FC, 2=Conv, 3=BN, ...) and pi​ is a parameter vector specific to layer type

From this we can derive several distance measures:
1. Pad smaller genomes with zeroes and take distance
2. Feature based distance: Compare information about the network like num_layers of each type, num_params etc. (Loses structural information)
3. Edit distance (Need to tune cost of operations)
4. hybrid of structural difference (layer types) and parameter difference (number of parameters in each layer) (needs best alignment)

