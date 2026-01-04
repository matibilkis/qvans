# Reproducibility Guide

This document provides detailed instructions for reproducing the results from the VAns paper.

## Environment Setup

### Recommended Setup

For exact reproducibility, use the following environment:

```bash
# Create a fresh virtual environment
python3.8 -m venv vans_env
source vans_env/bin/activate

# Install exact versions
pip install -r requirements.txt
```

### System Requirements

- **OS**: Linux (tested on Ubuntu 20.04), macOS, or Windows with WSL
- **Python**: 3.7, 3.8, or 3.9 (3.8 recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **GPU**: Optional but recommended for faster training

### Dependency Versions

The paper results were obtained with:

```
Python: 3.8
numpy: 1.19.x
sympy: 1.5
cirq: 0.9.1
tensorflow: 2.3.1
tensorflow-quantum: 0.4.0
openfermion: 1.0.0
openfermionpyscf: 0.5
```

## Reproducing Paper Results

### 1. Transverse Field Ising Model (TFIM)

**Figure 2 (if applicable)**: Energy vs iterations for TFIM

```bash
python3 main.py \
    --problem TFIM \
    --J 0.5 \
    --g 1.0 \
    --n_qubits 4 \
    --reps 150 \
    --qepochs 10000 \
    --qlr 0.01 \
    --optimizer adam \
    --training_patience 1000 \
    --acceptance_percentage 0.01 \
    --rate_iids_per_step 1.5 \
    --initialization hea \
    --init_layers_hea 1
```

**Expected**: Energy should converge to ground state energy within chemical accuracy.

### 2. XXZ Model

```bash
python3 main.py \
    --problem XXZ \
    --J 1.0 \
    --g 1.0 \
    --n_qubits 4 \
    --reps 150 \
    --qepochs 10000 \
    --qlr 0.01 \
    --optimizer adam
```

### 3. Quantum Chemistry: H₂ Molecule

```bash
python3 main.py \
    --n_qubits 4 \
    --problem_config '{"problem": "H2", "geometry": [("H", (0., 0., 0.)), ("H", (0., 0., 0.74))], "multiplicity": 1, "charge": 0, "basis": "sto-3g"}' \
    --reps 200 \
    --qepochs 10000 \
    --qlr 0.01 \
    --return_lower_bound 1
```

**Note**: Bond length 0.74 Å is the equilibrium distance. Results should reach chemical accuracy (0.0016 Ha).

### 4. Quantum Chemistry: H₄ Chain

```bash
python3 main.py \
    --n_qubits 8 \
    --problem_config '{"problem": "H4", "geometry": [("H", (0., 0., 0.)), ("H", (0., 0., 1.5)), ("H", (0., 0., 3.0)), ("H", (0., 0., 4.5))], "multiplicity": 1, "charge": 0, "basis": "sto-3g"}' \
    --reps 200 \
    --qepochs 10000
```

### 5. Quantum Autoencoder

The autoencoder implementation is available in `utilities/variational.py` (Autoencoder class). To use:

```python
from utilities.variational import Autoencoder
from utilities.evaluator import Evaluator

# Prepare input states (list of indexed circuits)
# ... setup code ...

autoencoder = Autoencoder(
    many_indexed_circuits=input_circuits,
    many_symbols_to_values=input_resolvers,
    n_qubits=n_qubits,
    nb=num_trash_qubits,  # Number of qubits to compress
    lr=0.01,
    epochs=1000
)

# Run autoencoder optimization
cost, resolver, history = autoencoder.autoencoder(indexed_circuit)
```

### 6. Unitary Compilation

Unitary compilation uses the same VAns framework with a fidelity-based cost function. See `utilities/unitary_killer.py` and circuit simplification modules.

## Parameter Sweeps

To reproduce parameter sweeps from the paper:

### TFIM J-parameter sweep

Edit `meta_main.py`:

```python
insts = []
js = np.arange(0.0, 2.0, 0.1)
for J in js:
    problem_config = dict_to_json({"problem": "TFIM", "g": 1.0, "J": J})
    instruction = "python3 main.py --n_qubits 4 --problem_config {} --reps 150 --qepochs 10000".format(problem_config)
    insts.append(instruction)
```

Run:
```bash
python3 meta_main.py
```

## Expected Results

### Energy Convergence

- **TFIM (4 qubits)**: Should reach ground state within 50-100 iterations typically
- **H₂**: Should reach chemical accuracy (within 0.0016 Ha of FCI)
- **Circuit Depth**: VAns typically finds circuits with 20-40% fewer gates than fixed HEA

### Output Files

Results are saved in `--path_results` directory (default: `../data-vans/`):

- `circuits.pkl`: Circuit structures at each iteration
- `energies.npy`: Energy evolution
- `displaying.pkl`: Training metadata
- `information.txt`: Run configuration and summary

## Troubleshooting

### Common Issues

1. **TensorFlow Quantum Installation**
   ```bash
   # If TFQ installation fails, try:
   pip install tensorflow==2.3.1
   pip install tensorflow-quantum==0.4.0
   ```

2. **OpenFermion/PySCF Issues**
   - Ensure PySCF is properly installed: `pip install pyscf`
   - For quantum chemistry, may need additional system libraries

3. **Memory Issues**
   - Reduce `--qepochs` or `--reps`
   - Use smaller `--n_qubits`
   - Enable GPU if available

4. **Convergence Issues**
   - Increase `--qepochs`
   - Adjust `--qlr` (try 0.005 or 0.02)
   - Change `--optimizer` (try "adam" or "sgd")
   - Modify `--acceptance_percentage`

### Verification

To verify installation:

```bash
python3 -c "
import tensorflow_quantum as tfq
import cirq
import numpy as np
from utilities.variational import VQE
print('✓ All imports successful')
print('TFQ version:', tfq.__version__)
print('Cirq version:', cirq.__version__)
"
```

## Computational Resources

### Paper Experiments

The paper experiments were run on:
- **Local**: Standard workstations (8-16 cores, 16-32GB RAM)
- **HPC**: Barcelona cluster (see `hpc-programs/` for submission scripts)
- **Cloud**: Google TPU via Sandbox@Alphabet (for some large-scale runs)

### Typical Runtime

- **TFIM (4 qubits)**: 10-30 minutes on CPU, 2-5 minutes on GPU
- **H₂**: 20-60 minutes on CPU
- **H₄**: 1-3 hours on CPU

## Seed and Randomness

For exact reproducibility, you may want to set random seeds:

```python
import numpy as np
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

**Note**: VAns uses randomness in gate insertion, so results will vary between runs. The algorithm's performance (convergence rate, final energy) should be consistent.

## Citation

When reporting reproduced results, please cite:

```bibtex
@article{bilkis2023semi,
  title={A semi-agnostic ansatz with variable structure for variational quantum algorithms},
  author={Bilkis, M. and Cerezo, M. and Verdon, G. and Coles, P. J. and Cincio, L.},
  journal={Quantum Machine Intelligence},
  volume={5},
  number={43},
  year={2023},
  doi={10.1007/s42484-023-00132-1}
}
```

## Support

For issues with reproducibility:
1. Check this guide first
2. Review GitHub issues
3. Open a new issue with:
   - System information
   - Exact command used
   - Error messages or unexpected results
   - Environment details (`pip list` output)

---

**Last Updated**: 2024

