"""
Quantum Computing Research Toolkit
====================================
A production-grade suite of quantum algorithms implemented with Qiskit.

Algorithms:
    - BB84 Quantum Key Distribution (QKD)
    - Grover's Search Algorithm
    - Quantum Teleportation
    - QAOA for Max-Cut Optimization
    - Classical vs Quantum Benchmarking

Author: Your Name
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from quantum_suite.bb84 import BB84Protocol
from quantum_suite.grover import GroverSearch
from quantum_suite.teleportation import QuantumTeleportation
from quantum_suite.qaoa import QAOAOptimizer
from quantum_suite.benchmarks import QuantumBenchmark

__all__ = [
    "BB84Protocol",
    "GroverSearch",
    "QuantumTeleportation",
    "QAOAOptimizer",
    "QuantumBenchmark",
]
