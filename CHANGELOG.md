# Changelog

All notable changes to the VAns project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added
- Initial release corresponding to published paper
- Core VAns algorithm implementation
- VQE support for:
  - Transverse Field Ising Model (TFIM)
  - XXZ model
  - Quantum chemistry (H₂, H₄, LiH)
- Quantum autoencoder implementation
- Circuit simplification rules
- Identity insertion mechanism
- Unitary removal (gate killing)
- Multiple optimizers (Adam, SGD with QACQ, Adagrad)
- Parameter sweep utilities
- Jupyter notebook tutorials
- HPC submission scripts
- Comprehensive documentation:
  - README with usage examples
  - FEATURES.md with detailed feature documentation
  - REPRODUCIBILITY.md with reproduction guide
  - TODO.md with development roadmap
  - CONTRIBUTING.md with contribution guidelines
- Version management with requirements files
- Setup.py for package installation
- Experimental MultiVAns code (unfinished, 4+ years old)

### Known Issues
- TensorFlow Quantum version compatibility (requires TFQ 0.4.0)
- Some edge cases in circuit simplification
- Memory usage for large circuits
- MultiVAns codebase is outdated and incomplete

## [Unreleased]

### Planned
- Unitary compilation module
- Enhanced documentation (API reference)
- Comprehensive test suite
- Performance optimizations
- Extended problem support
- Configuration file support (YAML/JSON)
- Multi-framework support
- Modernized codebase

---

## Version History Notes

- **v1.0.0**: Matches the published paper implementation
- Future versions will follow semantic versioning (MAJOR.MINOR.PATCH)

## Release Types

- **Major** (X.0.0): Breaking changes, major new features
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, minor improvements

---

For detailed feature descriptions, see [FEATURES.md](FEATURES.md).
For development roadmap, see [TODO.md](TODO.md).

