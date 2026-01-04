# VAns Quick Start Guide

This is a condensed guide to get you started with VAns quickly. For detailed documentation, see the main [README.md](README.md).

## Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/matibilkis/qvans.git
cd qvans

# 2. Create virtual environment
python3.8 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python3 -c "from utilities.variational import VQE; print('✓ Installation successful')"
```

## Basic Usage

### Example 1: Transverse Field Ising Model (TFIM)

```bash
python3 main.py \
    --problem TFIM \
    --J 0.6 \
    --g 1.0 \
    --n_qubits 4 \
    --reps 150 \
    --qepochs 10000
```

**What it does**: Finds the ground state of a 4-qubit TFIM with coupling J=0.6 and field g=1.0.

### Example 2: Hydrogen Molecule (H₂)

```bash
python3 main.py \
    --n_qubits 4 \
    --problem_config '{"problem": "H2", "geometry": [("H", (0., 0., 0.)), ("H", (0., 0., 0.74))], "multiplicity": 1, "charge": 0, "basis": "sto-3g"}' \
    --reps 200 \
    --return_lower_bound 1
```

**What it does**: Optimizes circuit for H₂ at equilibrium bond length (0.74 Å).

## Key Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_qubits` | 8 | Number of qubits |
| `--reps` | 150 | VAns iterations |
| `--qepochs` | 10000 | Training epochs per VQE |
| `--qlr` | 0.01 | Learning rate |
| `--problem` | - | Problem type: TFIM, XXZ |
| `--optimizer` | adam | Optimizer: adam, sgd, adagrad |

## Understanding Output

Results are saved in `../data-vans/` (or `--path_results`):

- **`circuits.pkl`**: Circuit structures at each iteration
- **`energies.npy`**: Energy evolution
- **`information.txt`**: Human-readable summary

## Next Steps

1. **Learn the algorithm**: Read `tutorials/0_introduction.ipynb`
2. **Explore features**: See [FEATURES.md](FEATURES.md)
3. **Reproduce paper**: Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
4. **Customize**: Modify `main.py` or create your own script

## Common Issues

**Problem**: TensorFlow Quantum installation fails
```bash
# Solution: Install TensorFlow first, then TFQ
pip install tensorflow==2.3.1
pip install tensorflow-quantum==0.4.0
```

**Problem**: Out of memory
```bash
# Solution: Reduce problem size or epochs
python3 main.py --n_qubits 4 --qepochs 5000
```

**Problem**: Slow convergence
```bash
# Solution: Adjust learning rate or optimizer
python3 main.py --qlr 0.02 --optimizer sgd
```

## Getting Help

- **Documentation**: See [README.md](README.md) and [FEATURES.md](FEATURES.md)
- **Tutorials**: Check `tutorials/` directory
- **Issues**: Open a GitHub issue
- **Paper**: [arXiv:2103.06712](https://arxiv.org/abs/2103.06712)

## Citation

If you use VAns, please cite:

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

---

**Ready to dive deeper?** Check out the full [README.md](README.md)!

