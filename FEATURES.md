# VAns Feature Documentation

This document provides a comprehensive overview of all features implemented in VAns, as described in the paper and extended in this repository.

## Core Algorithm Features

### 1. Variable Structure Ansatz

VAns dynamically modifies the quantum circuit structure during optimization:

- **Identity Insertion**: Randomly inserts identity-resolved gate blocks to explore new circuit architectures
- **Circuit Simplification**: Applies algebraic rules to reduce circuit depth while maintaining equivalence
- **Gate Removal**: Removes redundant gates that don't significantly affect the cost function
- **Acceptance/Rejection**: Accepts structural changes only if they improve the cost function

**Implementation**: `utilities/idinserter.py`, `utilities/simplifier.py`, `utilities/unitary_killer.py`

### 2. Parameter Optimization

Continuous parameter optimization with multiple optimizers:

- **Adam Optimizer**: Default, well-suited for most problems
- **SGD with QACQ**: Quantum Adaptive Coordinate-wise optimizer (Algorithm 4 from [arXiv:1807.00800](https://arxiv.org/abs/1807.00800))
- **Adagrad**: Alternative optimizer option
- **Early Stopping**: Patience-based early stopping to prevent overfitting
- **Time-based Stopping**: Maximum training time per VQE iteration

**Implementation**: `utilities/variational.py` (VQE class), `utilities/qmodels.py` (QNN class)

## Application Domains

### 1. Variational Quantum Eigensolver (VQE)

#### 1.1 Condensed Matter Systems

**Transverse Field Ising Model (TFIM)**
- Hamiltonian: $H = -J \sum_{i} \sigma_i^x \sigma_{i+1}^x - g \sum_{i} \sigma_i^z$
- Parameters: `--problem TFIM --J <value> --g <value>`
- Usage: Finding ground states of spin chains
- Example: `python3 main.py --problem TFIM --J 0.6 --g 1.0 --n_qubits 4`

**XXZ Model**
- Hamiltonian: $H = \sum_{i} (\sigma_i^x \sigma_{i+1}^x + \sigma_i^y \sigma_{i+1}^y + J \sigma_i^z \sigma_{i+1}^z) + g \sum_{i} \sigma_i^z$
- Parameters: `--problem XXZ --J <value> --g <value>`
- Usage: Quantum magnetism and phase transitions
- Example: `python3 main.py --problem XXZ --J 1.0 --g 1.0 --n_qubits 4`

**Implementation**: `utilities/qmodels.py`, `utilities/hamiltonians/cm_hamiltonians.txt`

#### 1.2 Quantum Chemistry

**Supported Molecules**:
- **H₂**: Hydrogen molecule (4 qubits in STO-3G basis)
- **H₄**: Hydrogen chain (8 qubits in STO-3G basis)
- **LiH**: Lithium hydride (extended support)

**Features**:
- Full Configuration Interaction (FCI) energy calculation for comparison
- Chemical accuracy target (0.0016 Ha)
- Geometry optimization support
- Multiple basis sets (STO-3G, etc.)

**Usage Example**:
```bash
python3 main.py --n_qubits 4 \
    --problem_config '{"problem": "H2", "geometry": [("H", (0., 0., 0.)), ("H", (0., 0., 0.74))], "multiplicity": 1, "charge": 0, "basis": "sto-3g"}' \
    --return_lower_bound 1
```

**Implementation**: `utilities/chemical.py`, `utilities/hamiltonians/chemical_hamiltonians.txt`

### 2. Quantum Autoencoder

**Purpose**: Compress quantum states by learning an encoding/decoding circuit that preserves information in a subset of qubits.

**Key Features**:
- **Trash Qubits**: Specifies how many qubits to compress (information discarded)
- **Batch Processing**: Handles multiple input states simultaneously
- **Fidelity Optimization**: Maximizes overlap with compressed representation
- **Cost Function**: $C = 1 - \frac{1}{n_b} \sum_{i=1}^{n_b} \langle \psi_i | \Pi_0 | \psi_i \rangle$

**Usage**:
```python
from utilities.variational import Autoencoder

autoencoder = Autoencoder(
    many_indexed_circuits=input_circuits,  # List of indexed circuits
    many_symbols_to_values=input_resolvers, # Parameter values
    n_qubits=total_qubits,
    nb=num_trash_qubits,  # Number of qubits to compress
    lr=0.01,
    epochs=1000
)

cost, resolver, history = autoencoder.autoencoder(indexed_circuit)
```

**Implementation**: `utilities/variational.py` (Autoencoder class)

### 3. Unitary Compilation

**Purpose**: Compile a target unitary into an optimized quantum circuit.

**Features**:
- Fidelity-based optimization
- Circuit depth minimization
- Gate count reduction
- Integration with VAns simplification rules

**Implementation**: Uses VAns framework with fidelity cost function. See `utilities/unitary_killer.py` and circuit simplification modules.

## Algorithm Components

### 1. Circuit Representation

**Indexed Circuit Format**:
- Each gate is assigned a unique index
- Indices are ordered: CNOTs → Rz rotations → Rx rotations → Ry rotations
- Enables efficient circuit manipulation

**Implementation**: `utilities/circuit_basics.py` (Basic class)

### 2. Identity Insertion (Growth)

**Mechanism**:
- Randomly selects positions in the circuit
- Inserts identity-resolved gate blocks (initialized to identity)
- Uses temperature-based selection for exploration vs exploitation
- Adaptive rate based on current performance

**Hyperparameters**:
- `--rate_iids_per_step`: Average number of insertions per step
- `--selector_temperature`: Controls exploration (higher = more random)

**Implementation**: `utilities/idinserter.py` (IdInserter class)

### 3. Circuit Simplification

**Rules Applied**:
- **Rule 1**: Remove identity gates
- **Rule 2**: Combine consecutive rotations on same qubit
- **Rule 3**: Simplify CNOT sequences
- **Rule 4**: Remove redundant gates
- **Rule 5**: Phase cancellation
- **Rule 6**: Unitary simplification

**Implementation**: `utilities/simplifier.py` (Simplifier class)

### 4. Gate Removal (Pruning)

**Unitary Murder Algorithm**:
- Systematically removes parametrized gates
- Evaluates cost function after removal
- Accepts removal if cost increase is below threshold
- Iterates until no further reduction possible

**Hyperparameters**:
- `--accept_remove_unitary_wall`: Maximum acceptable cost increase

**Implementation**: `utilities/unitary_killer.py` (UnitaryMurder class)

### 5. Evaluation and Acceptance

**Features**:
- Tracks best circuit found
- Accepts/rejects structural changes
- Adaptive acceptance threshold
- Convergence detection
- Result saving and visualization

**Implementation**: `utilities/evaluator.py` (Evaluator class)

## Initialization Strategies

### 1. Hardware Efficient Ansatz (HEA)
- Default initialization
- Alternating layers of rotations and entangling gates
- Configurable depth: `--init_layers_hea <number>`

### 2. Separable State
- Initializes with single-qubit rotations only
- No entanglement initially
- Useful for problems requiring minimal entanglement

### 3. XZ Initialization
- Alternating X and Z rotations
- Simple starting point
- Good for certain problem types

**Usage**: `--initialization <hea|separable|xz>`

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--reps` | 150 | Number of VAns iterations |
| `--qepochs` | 10000 | Max training epochs per VQE |
| `--qlr` | 0.01 | Learning rate |
| `--acceptance_percentage` | 0.01 | Energy improvement threshold |
| `--rate_iids_per_step` | 1.5 | Identity insertion rate |
| `--selector_temperature` | 10.0 | Gate selection temperature |
| `--wait_to_get_back` | 25 | Iterations before reverting to best |

### Adaptive Mechanisms

- **Temperature Scheduling**: Adjusts exploration based on current performance
- **Acceptance Threshold**: Can reduce over time (`--reduce_acceptance_percentage`)
- **Parameter Perturbation**: Random parameter initialization when stuck

## Output and Results

### Saved Data

1. **Circuit Structures** (`circuits.pkl`):
   - Indexed circuits at each iteration
   - Parameter values
   - Circuit metadata

2. **Energy Evolution** (`energies.npy`):
   - Cost function values over iterations
   - Best energy found
   - Convergence history

3. **Training Metadata** (`displaying.pkl`):
   - Hyperparameters used
   - Training statistics
   - Circuit properties (depth, gate count, etc.)

4. **Information File** (`information.txt`):
   - Human-readable summary
   - Configuration details
   - Final results

### Analysis Tools

- Jupyter notebooks in `tutorials/` for result analysis
- Example results in `examples_repository/`
- HPC result loading scripts in `hpc-programs/running_examples/`

## Advanced Features

### 1. TensorBoard Integration

Enable with `--show_tensorboarddata 1`:
- Real-time training visualization
- Gradient norms
- Learning rate evolution
- Cost function tracking

### 2. Lower Bound Calculation

For quantum chemistry, compute FCI energy:
- `--return_lower_bound 1`: Enables FCI calculation
- Provides reference for convergence
- Chemical accuracy target: 0.0016 Ha

### 3. HPC Support

- Condor submission scripts: `hpc-programs/tocondor.sub`
- Batch processing: `hpc-programs/meta_main.py`
- Result aggregation tools

## Experimental Features (MultiVAns)

See `multivans/README.md` for:
- Quantum combs optimization
- Channel discrimination
- Multi-party protocols

**Status**: Experimental, unfinished, 4+ years old

## Performance Characteristics

### Typical Results

- **Circuit Depth Reduction**: 20-40% fewer gates than fixed HEA
- **Convergence**: Usually within 50-150 iterations for small systems
- **Chemical Accuracy**: Achieved for H₂, H₄ within reasonable time
- **Scalability**: Tested up to 8-10 qubits in paper

### Computational Cost

- **TFIM (4 qubits)**: 10-30 minutes (CPU), 2-5 minutes (GPU)
- **H₂**: 20-60 minutes (CPU)
- **H₄**: 1-3 hours (CPU)

## Limitations and Known Issues

1. **Scalability**: Performance degrades for very large systems (>10 qubits)
2. **Local Minima**: May get stuck in local minima (mitigated by random restarts)
3. **Noise**: Current implementation assumes noiseless simulation
4. **Dependency Versions**: Requires specific TFQ/Cirq versions

## Future Extensions

See `TODO.md` for planned features:
- Noise-aware optimization
- Multi-objective optimization
- Extended problem support
- Framework modernization

---

**For detailed usage examples, see the tutorials in `tutorials/` directory.**

