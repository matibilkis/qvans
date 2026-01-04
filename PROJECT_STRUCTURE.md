# VAns Project Structure

This document provides an overview of the VAns repository structure and organization.

## Repository Organization

```
qvans/
│
├── 📄 Core Files
│   ├── README.md              # Main documentation and quick start
│   ├── FEATURES.md            # Comprehensive feature documentation
│   ├── REPRODUCIBILITY.md     # Reproduction guide for paper results
│   ├── TODO.md                # Development roadmap (TDL)
│   ├── CONTRIBUTING.md        # Contribution guidelines
│   ├── CHANGELOG.md           # Version history
│   ├── LICENSE                # Research use license and disclaimers
│   ├── VERSION                # Current version number
│   ├── setup.py               # Package setup script
│   ├── .gitignore             # Git ignore rules
│   ├── requirements.txt       # Main dependencies (TFQ 0.4.0)
│   └── requirements-tfq0.7.txt # Alternative dependencies (TFQ 0.7.2)
│
├── 🐍 Main Scripts
│   ├── main.py                # Main VAns execution script
│   └── meta_main.py           # Parameter sweep script
│
├── 📚 Documentation & Tutorials
│   ├── tutorials/             # Jupyter notebook tutorials
│   │   ├── 0_introduction.ipynb
│   │   ├── 1_VANs_methods.ipynb
│   │   ├── 2_VANs_loop.ipynb
│   │   └── images/
│   └── figures_readme/        # Figures for README
│
├── 🔧 Core Modules (utilities/)
│   ├── variational.py         # VQE and Autoencoder classes
│   ├── evaluator.py           # Circuit evaluation and acceptance
│   ├── idinserter.py          # Identity insertion (growth)
│   ├── simplifier.py          # Circuit simplification rules
│   ├── unitary_killer.py      # Gate removal (pruning)
│   ├── circuit_basics.py      # Basic circuit operations
│   ├── chemical.py            # Quantum chemistry Hamiltonians
│   ├── qmodels.py             # Quantum neural network models
│   ├── misc.py                # Utility functions
│   └── hamiltonians/          # Hamiltonian definitions
│       ├── cm_hamiltonians.txt
│       └── chemical_hamiltonians.txt
│
├── 💻 HPC Support
│   └── hpc-programs/          # HPC cluster submission scripts
│       ├── main.py
│       ├── meta_main.py
│       ├── tocondor.sub        # Condor submission
│       └── running_examples/
│
├── 📊 Examples & Results
│   └── examples_repository/   # Example results from runs
│       └── TFIM/
│
└── 🔬 Experimental
    └── multivans/             # MultiVAns (quantum combs) - EXPERIMENTAL
        ├── README.md          # MultiVAns documentation
        ├── utilities/         # MultiVAns modules
        ├── running/           # Execution scripts
        ├── coding/            # Compilation examples
        └── shortcuts/         # Quick-start scripts
```

## Module Descriptions

### Core Algorithm Modules

#### `utilities/variational.py`
- **VQE**: Variational Quantum Eigensolver implementation
- **Autoencoder**: Quantum autoencoder for state compression
- Handles parameter optimization with multiple optimizers

#### `utilities/evaluator.py`
- Tracks circuit evolution
- Accepts/rejects structural changes
- Saves results and metadata

#### `utilities/idinserter.py`
- Implements identity insertion (circuit growth)
- Temperature-based gate selection
- Adaptive insertion rates

#### `utilities/simplifier.py`
- Applies circuit simplification rules
- Reduces circuit depth while maintaining equivalence
- Multiple simplification strategies

#### `utilities/unitary_killer.py`
- Removes redundant gates
- Evaluates cost impact of gate removal
- Iterative pruning algorithm

#### `utilities/circuit_basics.py`
- Basic circuit operations
- Gate indexing system
- Circuit construction utilities

#### `utilities/chemical.py`
- Quantum chemistry Hamiltonian construction
- FCI energy calculation
- Molecular geometry handling

#### `utilities/qmodels.py`
- Quantum neural network models
- Custom loss functions
- Optimizer implementations (including QACQ)

## File Types

### Python Files
- **Main scripts**: `main.py`, `meta_main.py`
- **Modules**: All files in `utilities/`
- **HPC scripts**: Files in `hpc-programs/`

### Configuration Files
- **Requirements**: `requirements.txt`, `requirements-tfq0.7.txt`
- **Setup**: `setup.py`, `VERSION`
- **Git**: `.gitignore`

### Documentation
- **Markdown**: All `.md` files
- **Notebooks**: Jupyter notebooks in `tutorials/`
- **Images**: Figures in `figures_readme/`

### Data Files
- **Results**: `.pkl`, `.npy` files in `examples_repository/`
- **Hamiltonians**: Text files in `utilities/hamiltonians/`

## Key Directories

### `utilities/`
Core algorithm implementation. All VAns functionality is here.

### `tutorials/`
Interactive learning materials. Start here for new users.

### `examples_repository/`
Example outputs and results. Useful for understanding expected outputs.

### `hpc-programs/`
High-performance computing support. For cluster execution.

### `multivans/`
**⚠️ EXPERIMENTAL**: Quantum combs extension. Unfinished, 4+ years old.

## Entry Points

### For Users
1. **Start**: Read `README.md`
2. **Learn**: Follow `tutorials/` notebooks
3. **Run**: Use `main.py` with command-line arguments
4. **Understand**: Read `FEATURES.md`

### For Developers
1. **Setup**: Follow `CONTRIBUTING.md`
2. **Plan**: Check `TODO.md`
3. **Code**: Work in `utilities/`
4. **Test**: Add tests (when test suite exists)

### For Researchers
1. **Reproduce**: Follow `REPRODUCIBILITY.md`
2. **Extend**: See `TODO.md` for research directions
3. **Experiment**: Check `multivans/` (with caution)

## Dependencies

### Core Dependencies
- TensorFlow Quantum 0.4.0 (primary)
- Cirq 0.9.1
- TensorFlow 2.3.1
- OpenFermion 1.0.0

### Optional Dependencies
- Jupyter (for tutorials)
- Matplotlib (for visualization)
- PySCF (for quantum chemistry)

## Version Information

- **Current Version**: 1.0.0 (matches published paper)
- **Python**: 3.7-3.9
- **Primary Framework**: TensorFlow Quantum
- **Experimental**: MultiVAns uses PennyLane (old version)

## Notes

- All core functionality is in `utilities/`
- Main entry point is `main.py`
- Results are saved to `--path_results` (default: `../data-vans/`)
- Experimental code is clearly marked and separated

---

For questions about the structure, see the main `README.md` or open an issue.

