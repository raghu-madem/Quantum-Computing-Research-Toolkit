"""
Grover's Search Algorithm
==========================
Grover's algorithm provides a quadratic speedup for unstructured database search.
For N items, classical search needs O(N) queries; Grover's needs O(√N).

Published by Lov Grover in 1996, it is one of the most fundamental quantum
algorithms and has applications in database search, cryptography (breaking
symmetric ciphers), and NP-complete problem solving.

Core Components:
    1. Initialization: Uniform superposition over all N states
    2. Oracle: Marks the target state(s) with a phase flip
    3. Diffusion (Grover): Inverts amplitudes about the mean
    4. Repetition: O(√N) iterations amplify target amplitude
    5. Measurement: Target state measured with high probability

Mathematical Foundation:
    After k = floor(π/4 · √N) iterations, the probability of measuring
    the target state is P ≈ sin²((2k+1)·arcsin(1/√N)) ≈ 1.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)


@dataclass
class GroverResult:
    """Result of a Grover's algorithm execution."""

    target: Union[int, list[int]]
    n_qubits: int
    n_iterations: int
    search_space_size: int
    found: bool
    measured_state: int
    success_probability: float
    counts: dict[str, int]
    classical_queries_expected: int
    quantum_queries_used: int
    speedup_factor: float

    def summary(self) -> dict:
        return {
            "target": self.target,
            "search_space": self.search_space_size,
            "found": self.found,
            "measured": self.measured_state,
            "success_prob": f"{self.success_probability:.2%}",
            "grover_iterations": self.quantum_queries_used,
            "classical_expected": self.classical_queries_expected,
            "quantum_speedup": f"{self.speedup_factor:.1f}x",
        }


class GroverSearch:
    """
    Grover's Quantum Search Algorithm implementation.

    Searches an unstructured space of 2^n_qubits elements for one or more
    target states using quantum amplitude amplification.

    Args:
        n_qubits (int): Number of qubits (search space = 2^n_qubits). Range [2, 20].
        shots (int): Number of measurement shots for statistics.

    Example:
        >>> grover = GroverSearch(n_qubits=4)
        >>> result = grover.search(target=11)
        >>> print(result.summary())
        >>> print(f"Found item {result.measured_state} with P={result.success_probability:.2%}")

    Multi-target example:
        >>> result = grover.search(target=[3, 7, 11])
    """

    def __init__(self, n_qubits: int = 4, shots: int = 2048):
        if not 2 <= n_qubits <= 20:
            raise ValueError("n_qubits must be between 2 and 20")
        self.n_qubits = n_qubits
        self.shots = shots
        self.N = 2**n_qubits
        self.simulator = AerSimulator()
        logger.info(f"GroverSearch: {n_qubits} qubits, search space={self.N}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        target: Union[int, list[int]],
        n_iterations: Optional[int] = None,
    ) -> GroverResult:
        """
        Run Grover's algorithm to find the target state(s).

        Args:
            target: The target index/value to find (0 to 2^n-1), or a list of targets.
            n_iterations: Override the optimal iteration count.

        Returns:
            GroverResult with measured state, probability, and speedup metrics.

        Raises:
            ValueError: If target is out of search space bounds.
        """
        targets = [target] if isinstance(target, int) else list(target)
        for t in targets:
            if not 0 <= t < self.N:
                raise ValueError(f"Target {t} out of range [0, {self.N-1}]")

        n_iters = n_iterations or self._optimal_iterations(len(targets))
        logger.info(
            f"Searching for {targets} in space of {self.N} — "
            f"{n_iters} Grover iteration(s)"
        )

        # Build full circuit
        circuit = self._build_circuit(targets, n_iters)

        # Execute
        transpiled = transpile(circuit, self.simulator)
        job = self.simulator.run(transpiled, shots=self.shots)
        counts = job.result().get_counts()

        # Analyze results
        best_state_bin = max(counts, key=counts.get)
        best_state = int(best_state_bin, 2)
        success_prob = counts.get(best_state_bin, 0) / self.shots
        found = best_state in targets

        # Speedup calculation
        classical_queries = self.N // (2 * len(targets))  # Expected classical queries
        quantum_queries = n_iters + 1  # oracle calls
        speedup = classical_queries / max(quantum_queries, 1)

        return GroverResult(
            target=target,
            n_qubits=self.n_qubits,
            n_iterations=n_iters,
            search_space_size=self.N,
            found=found,
            measured_state=best_state,
            success_probability=success_prob,
            counts={k: v for k, v in sorted(counts.items())},
            classical_queries_expected=classical_queries,
            quantum_queries_used=quantum_queries,
            speedup_factor=speedup,
        )

    def get_circuit(
        self,
        target: Union[int, list[int]],
        n_iterations: Optional[int] = None,
    ) -> QuantumCircuit:
        """Return the full Grover circuit without executing it."""
        targets = [target] if isinstance(target, int) else list(target)
        n_iters = n_iterations or self._optimal_iterations(len(targets))
        return self._build_circuit(targets, n_iters)

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_circuit(self, targets: list[int], n_iterations: int) -> QuantumCircuit:
        """Assemble the full Grover circuit: init → (oracle → diffusion)×k → measure."""
        n = self.n_qubits
        qc = QuantumCircuit(n, n, name=f"Grover_{n}q_{n_iterations}iter")

        # Step 1: Hadamard initialization → uniform superposition
        qc.h(range(n))
        qc.barrier(label="H⊗n")

        # Steps 2-3: Grover iterations
        for i in range(n_iterations):
            # Oracle: phase flip on target states
            self._apply_oracle(qc, targets)
            qc.barrier(label=f"Oracle_{i+1}")
            # Diffusion: inversion about mean
            self._apply_diffusion(qc)
            qc.barrier(label=f"Diff_{i+1}")

        # Step 4: Measure all qubits
        qc.measure(range(n), range(n))
        return qc

    def _apply_oracle(self, qc: QuantumCircuit, targets: list[int]) -> None:
        """
        Phase oracle: applies −1 phase to each target state.
        Uses multi-controlled Z via ancilla decomposition for scalability.
        """
        n = self.n_qubits
        for target in targets:
            # Flip qubits where target bit is 0 (to align with |11...1⟩ for CZ)
            target_binary = format(target, f"0{n}b")
            flip_qubits = [i for i, b in enumerate(reversed(target_binary)) if b == "0"]

            if flip_qubits:
                qc.x(flip_qubits)

            # Apply multi-controlled Z (phase flip on |11...1⟩)
            self._mcz(qc, list(range(n)))

            if flip_qubits:
                qc.x(flip_qubits)

    def _apply_diffusion(self, qc: QuantumCircuit) -> None:
        """
        Grover diffusion operator: 2|s⟩⟨s| − I
        Geometrically: inversion about the uniform superposition vector.
        """
        n = self.n_qubits
        qc.h(range(n))
        qc.x(range(n))
        self._mcz(qc, list(range(n)))
        qc.x(range(n))
        qc.h(range(n))

    def _mcz(self, qc: QuantumCircuit, qubits: list[int]) -> None:
        """Multi-controlled Z gate using CX decomposition."""
        if len(qubits) == 1:
            qc.z(qubits[0])
        elif len(qubits) == 2:
            qc.cz(qubits[0], qubits[1])
        else:
            # Decompose as H·CCX·H on last qubit
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])

    # ------------------------------------------------------------------
    # Math utilities
    # ------------------------------------------------------------------

    def _optimal_iterations(self, n_targets: int = 1) -> int:
        """
        Compute optimal Grover iteration count.
        k_opt = floor(π/4 · √(N/M)) where M = number of targets.
        """
        if n_targets >= self.N:
            return 1
        k = math.floor((math.pi / 4) * math.sqrt(self.N / n_targets))
        return max(1, k)

    def theoretical_success_probability(self, n_iterations: int, n_targets: int = 1) -> float:
        """Compute theoretical success probability for given iterations."""
        theta = math.asin(math.sqrt(n_targets / self.N))
        return math.sin((2 * n_iterations + 1) * theta) ** 2
