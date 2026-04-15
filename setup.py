"""
Setup configuration for the Quantum Computing Research Toolkit.
Install with:
    pip install -e .          # Development (editable) install
    pip install .             # Regular install
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="quantum-research-toolkit",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description=(
        "A production-grade quantum computing research toolkit: "
        "BB84 QKD, Grover Search, Quantum Teleportation, QAOA, and Benchmarks."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-research-toolkit",
    license="MIT",
    packages=find_packages(exclude=["tests*", "demos*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "qiskit>=1.0.0",
        "qiskit-aer>=0.14.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-timeout>=2.1.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-toolkit=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "quantum computing",
        "qiskit",
        "BB84",
        "Grover",
        "quantum teleportation",
        "QAOA",
        "quantum cryptography",
        "quantum algorithms",
    ],
    project_urls={
        "Documentation": "https://github.com/yourusername/quantum-research-toolkit",
        "Bug Reports": "https://github.com/yourusername/quantum-research-toolkit/issues",
        "Source": "https://github.com/yourusername/quantum-research-toolkit",
    },
)
