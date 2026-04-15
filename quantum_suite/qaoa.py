"""
Quantum Approximate Optimization Algorithm (QAOA)
==================================================
QAOA is a hybrid quantum-classical algorithm for combinatorial optimization.
Proposed by Farhi, Goldstone, and Gutmann (2014), it solves NP-hard problems
by using parameterized quantum circuits optimized by a classical optimizer.

Application: Max-Cut Problem
    Given a graph G=(V,E), partition vertices into two sets S and S̄
    to maximize edges between the sets. This is NP-hard in general.

QAOA Circuit Structure:
    |ψ(β,γ)⟩ = U_B(β_p) · U_C(γ_p) · ... · U_B(β_1) · U_C(γ_1) · |+⟩^n

    - |+⟩^n: Equal superposition over all bit-strings (H⊗n on |0⟩^n)
    - U_C(γ): Problem unitary — encodes the cost Hamiltonian
    - U_B(β): Mixer unitary — explores the solution space (Rx rotations)
    - p: QAOA depth (more layers → better approximation)

Classical Optimization:
    Parameters (β, γ) are optimized classically (COBYLA/SPSA)
    to maximize ⟨ψ(β,γ)| C |ψ(β,γ)⟩, the expected cut value.

Approximation Ratio:
    At depth p → ∞, QAOA → exact solution.
    At p=1, QAOA guarantees ≥ 0.692 × OPT (Goemans-Williamson is 0.878).
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import Parameter, ParameterVector

logger = logging.getLogger(__name__)


@dataclass
class Graph:
    """Simple undirected weighted graph."""

    n_vertices: int
    edges: list[tuple[int, int, float]] = field(default_factory=list)

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        if not (0 <= u < self.n_vertices and 0 <= v < self.n_vertices):
            raise ValueError(f"Vertices must be in [0, {self.n_vertices-1}]")
        self.edges.append((u, v, weight))

    def cut_value(self, partition: list[int]) -> float:
        """Compute the cut value for a given binary partition (0/1 per vertex)."""
        cut = 0.0
        for u, v, w in self.edges:
            if partition[u] != partition[v]:
                cut += w
        return cut

    def brute_force_max_cut(self) -> tuple[list[int], float]:
        """Solve Max-Cut by exhaustive search (only feasible for small graphs)."""
        if self.n_vertices > 20:
            raise RuntimeError("Brute force only feasible for n ≤ 20 vertices")
        best_cut = 0.0
        best_partition = [0] * self.n_vertices
        for mask in range(1 << self.n_vertices):
            partition = [(mask >> i) & 1 for i in range(self.n_vertices)]
            cut = self.cut_value(partition)
            if cut > best_cut:
                best_cut = cut
                best_partition = partition
        return best_partition, best_cut

    @classmethod
    def cycle(cls, n: int, weight: float = 1.0) -> "Graph":
        """Create a cycle graph C_n."""
        g = cls(n)
        for i in range(n):
            g.add_edge(i, (i + 1) % n, weight)
        return g

    @classmethod
    def complete(cls, n: int, weight: float = 1.0) -> "Graph":
        """Create a complete graph K_n."""
        g = cls(n)
        for u, v in combinations(range(n), 2):
            g.add_edge(u, v, weight)
        return g

    @classmethod
    def random(cls, n: int, edge_prob: float = 0.5, seed: int = 42) -> "Graph":
        """Create a random Erdős–Rényi graph."""
        rng = np.random.default_rng(seed)
        g = cls(n)
        for u, v in combinations(range(n), 2):
            if rng.random() < edge_prob:
                w = float(rng.uniform(0.5, 2.0))
                g.add_edge(u, v, w)
        return g


@dataclass
class QAOAResult:
    """Result of a QAOA optimization run."""

    graph: Graph
    p_layers: int
    optimal_gamma: list[float]
    optimal_beta: list[float]
    best_partition: list[int]
    best_cut_value: float
    qaoa_expected_cut: float
    classical_max_cut: float
    approximation_ratio: float
    n_optimizer_iterations: int
    counts: dict[str, int]

    def summary(self) -> dict:
        return {
            "n_vertices": self.graph.n_vertices,
            "n_edges": len(self.graph.edges),
            "p_layers": self.p_layers,
            "qaoa_cut": self.best_cut_value,
            "optimal_cut": self.classical_max_cut,
            "approximation_ratio": f"{self.approximation_ratio:.3f}",
            "optimizer_iters": self.n_optimizer_iterations,
        }


class QAOAOptimizer:
    """
    QAOA for the Max-Cut problem.

    Hybrid quantum-classical optimizer using Qiskit for circuit simulation
    and SciPy (COBYLA) for parameter optimization.

    Args:
        p (int): QAOA depth (number of alternating layers). Default 2.
        shots (int): Measurement shots per circuit evaluation.
        optimizer (str): Classical optimizer. 'COBYLA' or 'SPSA'.
        max_iter (int): Max classical optimizer iterations.

    Example:
        >>> graph = Graph.cycle(6)
        >>> qaoa = QAOAOptimizer(p=2, shots=4096)
        >>> result = qaoa.solve(graph)
        >>> print(f"Approximation ratio: {result.approximation_ratio:.3f}")
        >>> print(f"Best partition: {result.best_partition}")
    """

    def __init__(
        self,
        p: int = 2,
        shots: int = 4096,
        optimizer: str = "COBYLA",
        max_iter: int = 200,
    ):
        if p < 1 or p > 10:
            raise ValueError("p must be between 1 and 10")
        self.p = p
        self.shots = shots
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.simulator = AerSimulator()
        self._eval_count = 0

        logger.info(f"QAOAOptimizer: p={p}, shots={shots}, optimizer={optimizer}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, graph: Graph) -> QAOAResult:
        """
        Solve Max-Cut on the given graph using QAOA.

        Args:
            graph: Graph instance to solve.

        Returns:
            QAOAResult with solution partition and quality metrics.
        """
        n = graph.n_vertices
        if n > 15:
            logger.warning(f"Large graph ({n} nodes) — simulation may be slow")

        self._eval_count = 0
        logger.info(f"QAOA Max-Cut: {n} vertices, {len(graph.edges)} edges, p={self.p}")

        # Classical optimal for comparison (brute force for small graphs)
        classical_partition, classical_cut = graph.brute_force_max_cut()
        logger.info(f"Classical optimal cut: {classical_cut:.2f}")

        # Initialize parameters: γ ∈ [0, 2π], β ∈ [0, π]
        n_params = 2 * self.p
        rng = np.random.default_rng(42)
        x0 = rng.uniform(0, 2 * np.pi, n_params)

        # Classical optimization loop
        opt_result = minimize(
            fun=lambda x: -self._expected_cut(graph, x),
            x0=x0,
            method=self.optimizer,
            options={"maxiter": self.max_iter, "rhobeg": 0.5},
        )

        opt_params = opt_result.x
        gamma = list(opt_params[: self.p])
        beta = list(opt_params[self.p :])

        # Final measurement with optimal parameters
        counts = self._run_circuit(graph, opt_params)
        best_bitstring = max(counts, key=counts.get)
        best_partition = [int(b) for b in reversed(best_bitstring)]
        best_cut = graph.cut_value(best_partition)
        expected_cut = self._expected_cut(graph, opt_params)

        approx_ratio = best_cut / classical_cut if classical_cut > 0 else 0.0

        logger.info(
            f"QAOA cut={best_cut:.2f}, classical={classical_cut:.2f}, "
            f"ratio={approx_ratio:.3f}"
        )

        return QAOAResult(
            graph=graph,
            p_layers=self.p,
            optimal_gamma=gamma,
            optimal_beta=beta,
            best_partition=best_partition,
            best_cut_value=best_cut,
            qaoa_expected_cut=expected_cut,
            classical_max_cut=classical_cut,
            approximation_ratio=approx_ratio,
            n_optimizer_iterations=self._eval_count,
            counts=dict(sorted(counts.items(), key=lambda x: -x[1])[:20]),
        )

    def get_circuit(
        self,
        graph: Graph,
        gamma: Optional[list[float]] = None,
        beta: Optional[list[float]] = None,
    ) -> QuantumCircuit:
        """Return the parameterized (or bound) QAOA circuit."""
        if gamma and beta:
            params = gamma + beta
            return self._build_circuit(graph, np.array(params))
        return self._build_parametric_circuit(graph)

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_circuit(self, graph: Graph, params: np.ndarray) -> QuantumCircuit:
        """Build and bind a QAOA circuit with concrete parameter values."""
        n = graph.n_vertices
        gamma = params[: self.p]
        beta = params[self.p :]

        qc = QuantumCircuit(n, n, name=f"QAOA_p{self.p}")

        # Initial state: uniform superposition
        qc.h(range(n))
        qc.barrier(label="H⊗n")

        for layer in range(self.p):
            # Problem unitary U_C(γ)
            for u, v, w in graph.edges:
                angle = 2 * gamma[layer] * w
                qc.cx(u, v)
                qc.rz(angle, v)
                qc.cx(u, v)
            qc.barrier(label=f"U_C(γ{layer+1})")

            # Mixer unitary U_B(β)
            for qubit in range(n):
                qc.rx(2 * beta[layer], qubit)
            qc.barrier(label=f"U_B(β{layer+1})")

        qc.measure(range(n), range(n))
        return qc

    def _build_parametric_circuit(self, graph: Graph) -> QuantumCircuit:
        """Build a symbolic parameterized QAOA circuit for display."""
        n = graph.n_vertices
        gamma = ParameterVector("γ", self.p)
        beta = ParameterVector("β", self.p)

        qc = QuantumCircuit(n, n, name=f"QAOA_p{self.p}_parametric")
        qc.h(range(n))
        qc.barrier()
        for layer in range(self.p):
            for u, v, w in graph.edges:
                qc.cx(u, v)
                qc.rz(2 * gamma[layer] * w, v)
                qc.cx(u, v)
            qc.barrier()
            for qubit in range(n):
                qc.rx(2 * beta[layer], qubit)
            qc.barrier()
        qc.measure(range(n), range(n))
        return qc

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _expected_cut(self, graph: Graph, params: np.ndarray) -> float:
        """Compute ⟨C⟩ = expected cut value from measurement statistics."""
        self._eval_count += 1
        counts = self._run_circuit(graph, params)
        total = sum(counts.values())
        expected = 0.0
        for bitstring, count in counts.items():
            partition = [int(b) for b in reversed(bitstring)]
            cut = graph.cut_value(partition)
            expected += (count / total) * cut
        return expected

    def _run_circuit(self, graph: Graph, params: np.ndarray) -> dict[str, int]:
        """Execute the QAOA circuit and return measurement counts."""
        qc = self._build_circuit(graph, params)
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=self.shots)
        return job.result().get_counts()
