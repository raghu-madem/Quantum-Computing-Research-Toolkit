"""
pytest configuration for the Quantum Computing Research Toolkit.
Provides shared fixtures and configures test timeouts.
"""

import pytest
import logging
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Suppress Qiskit deprecation warnings in test output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (skipped with -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Set up logging for test session."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)-8s  %(name)s  %(message)s",
    )


@pytest.fixture(scope="session")
def simulator():
    """Shared AerSimulator instance across the test session."""
    from qiskit_aer import AerSimulator
    return AerSimulator()


@pytest.fixture
def small_bb84():
    """Fast BB84 protocol for unit tests (64 bits)."""
    from quantum_suite.bb84 import BB84Protocol
    return BB84Protocol(n_bits=64, seed=42)


@pytest.fixture
def small_grover():
    """Fast Grover search (3 qubits = 8 states)."""
    from quantum_suite.grover import GroverSearch
    return GroverSearch(n_qubits=3, shots=512)


@pytest.fixture
def noiseless_teleport():
    """Noiseless quantum teleportation."""
    from quantum_suite.teleportation import QuantumTeleportation
    return QuantumTeleportation(noise_level=0.0)


@pytest.fixture
def small_graph():
    """Tiny cycle graph C4 for QAOA tests."""
    from quantum_suite.qaoa import Graph
    return Graph.cycle(4)


@pytest.fixture
def fast_qaoa():
    """Fast QAOA with minimal shots and iterations."""
    from quantum_suite.qaoa import QAOAOptimizer
    return QAOAOptimizer(p=1, shots=512, max_iter=30)
