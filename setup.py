"""
Setup script for VAns (Variable Ansatz for Variational Quantum Algorithms)
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read version (if available)
def get_version():
    # Try to read from version file or use default
    version_file = os.path.join(os.path.dirname(__file__), "VERSION")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            return f.read().strip()
    return "1.0.0"

setup(
    name="qvans",
    version=get_version(),
    author="M. Bilkis, M. Cerezo, G. Verdon, P. J. Coles, L. Cincio",
    author_email="",  # Add if available
    description="Variable Ansatz (VAns) for Variational Quantum Algorithms",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/matibilkis/qvans",
    project_urls={
        "Bug Tracker": "https://github.com/matibilkis/qvans/issues",
        "Documentation": "https://github.com/matibilkis/qvans",
        "Source Code": "https://github.com/matibilkis/qvans",
        "Paper": "https://doi.org/10.1007/s42484-023-00132-1",
        "arXiv": "https://arxiv.org/abs/2103.06712",
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7,<3.10",
    install_requires=[
        "numpy>=1.19.0,<1.20.0",
        "sympy==1.5",
        "cirq==0.9.1",
        "tensorflow-quantum==0.4.0",
        "tensorflow==2.3.1",
        "openfermion==1.0.0",
        "openfermionpyscf==0.5",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "tutorials": [
            "jupyter>=1.0.0",
            "matplotlib>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vans=main:main",  # If main.py has a main() function
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

