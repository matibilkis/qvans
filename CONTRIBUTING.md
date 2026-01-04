# Contributing to VAns

Thank you for your interest in contributing to VAns! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title and description**
2. **Steps to reproduce**: Minimal example showing the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, dependency versions
6. **Error messages**: Full traceback if applicable

### Suggesting Features

Feature suggestions are welcome! Please:

1. Check existing issues to avoid duplicates
2. Open an issue describing:
   - The feature and its use case
   - Why it would be useful
   - Potential implementation approach (if you have ideas)

### Contributing Code

#### Development Setup

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/qvans.git
   cd qvans
   ```

3. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Coding Standards

- **Python Style**: Follow PEP 8
- **Docstrings**: Use NumPy-style docstrings
- **Type Hints**: Add type hints where possible (optional but encouraged)
- **Comments**: Explain "why", not "what"

Example:
```python
def simplify_circuit(self, indexed_circuit, symbols_to_values):
    """
    Simplify a circuit by applying algebraic rules.
    
    Parameters
    ----------
    indexed_circuit : list of int
        Circuit representation as list of gate indices
    symbols_to_values : dict
        Parameter values for each symbol
        
    Returns
    -------
    simplified_circuit : list of int
        Simplified circuit
    new_symbols : dict
        Updated parameter values
    """
    # Implementation
```

#### Testing

- Add tests for new features
- Ensure existing tests still pass
- Test edge cases and error conditions

#### Documentation

- Update relevant documentation
- Add docstrings to new functions/classes
- Update README if adding major features
- Add examples if applicable

#### Commit Messages

Use clear, descriptive commit messages:

```
Add support for custom Hamiltonians

- Implement Hamiltonian class interface
- Add example custom Hamiltonian
- Update documentation
```

#### Pull Request Process

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run tests** (if available):
   ```bash
   pytest tests/
   ```

3. **Check code style**:
   ```bash
   flake8 your_changed_files.py
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**:
   - Clear title and description
   - Reference related issues
   - Describe changes and testing done
   - Request review from maintainers

## Areas for Contribution

### High Priority

- **Testing**: Unit tests, integration tests
- **Documentation**: API docs, tutorials, examples
- **Code Quality**: Refactoring, type hints, error handling
- **Performance**: Optimization, profiling, benchmarking

### Medium Priority

- **New Problem Types**: Additional Hamiltonians, applications
- **Features**: See `TODO.md` for planned features
- **Bug Fixes**: See GitHub issues
- **Examples**: More usage examples and tutorials

### Research Directions

- **MultiVAns**: Porting and completing experimental code
- **Noise Mitigation**: Noise-aware optimization
- **Hardware Integration**: Real device support
- **Advanced Algorithms**: Extensions and improvements

## Development Guidelines

### Project Structure

- `utilities/`: Core algorithm modules
- `tutorials/`: Jupyter notebook tutorials
- `examples_repository/`: Example results
- `multivans/`: Experimental code (see separate guidelines)

### Module Organization

- Keep modules focused and cohesive
- Minimize dependencies between modules
- Use clear, descriptive names

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately

### Performance

- Profile before optimizing
- Document performance characteristics
- Consider memory usage for large circuits

## Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Thank you for contributing!

## Questions?

- Open an issue for questions
- Check existing documentation
- Review similar code in the repository

## License

By contributing, you agree that your contributions will be subject to the same terms as the project. See the [LICENSE](LICENSE) file for details.

---

Thank you for helping improve VAns! 🚀

