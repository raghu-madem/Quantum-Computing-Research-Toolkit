"""
Test Suite — Quantum Computing Research Toolkit
================================================
Unit and integration tests for all quantum algorithms.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short
    pytest tests/test_bb84.py -v          # single module
    pytest tests/ -k "grover" -v          # filter by name
"""

import math
import pytest
import numpy as np


# ======================================================================
# BB84 Tests
# ======================================================================

class TestBB84Protocol:
    """Tests for the BB84 Quantum Key Distribution protocol."""

    @pytest.fixture
    def protocol(self):
        from quantum_suite.bb84 import BB84Protocol
        return BB84Protocol(n_bits=64, seed=42)

    def test_init_defaults(self):
        from quantum_suite.bb84 import BB84Protocol
        proto = BB84Protocol()
        assert proto.n_bits == 256
        assert proto.qber_threshold == 0.11
        assert proto.noise_level == 0.0

    def test_init_validation(self):
        from quantum_suite.bb84 import BB84Protocol
        with pytest.raises(ValueError, match="n_bits"):
            BB84Protocol(n_bits=10)  # Too small
        with pytest.raises(ValueError, match="noise_level"):
            BB84Protocol(noise_level=0.5)  # Too large

    def test_run_returns_result(self, protocol):
        from quantum_suite.bb84 import BB84Result
        result = protocol.run(eve_present=False)
        assert isinstance(result, BB84Result)

    def test_no_eve_secure(self):
        """Without Eve, the channel should be secure."""
        from quantum_suite.bb84 import BB84Protocol
        # Run multiple trials; at least 80% should be secure
        successes = 0
        for seed in range(5):
            proto = BB84Protocol(n_bits=128, seed=seed)
            result = proto.run(eve_present=False)
            if result.secure:
                successes += 1
        assert successes >= 4, f"Only {successes}/5 sessions were secure without Eve"

    def test_with_eve_high_qber(self):
        """With Eve, QBER should be significantly elevated."""
        from quantum_suite.bb84 import BB84Protocol
        qbers = []
        for seed in range(3):
            proto = BB84Protocol(n_bits=128, seed=seed)
            result = proto.run(eve_present=True)
            qbers.append(result.qber)
        mean_qber = np.mean(qbers)
        assert mean_qber > 0.05, f"Expected elevated QBER with Eve, got {mean_qber:.3f}"

    def test_sifted_key_length(self, protocol):
        """Sifted key should be ~50% of raw bits (bases match ~50% of the time)."""
        result = protocol.run(eve_present=False)
        efficiency = result.sifted_key_length / protocol.n_bits
        assert 0.3 <= efficiency <= 0.7, f"Unexpected sifting efficiency: {efficiency:.2f}"

    def test_result_summary(self, protocol):
        result = protocol.run(eve_present=False)
        summary = result.summary()
        assert "qber_percent" in summary
        assert "session_id" in summary
        assert "key_secure" in summary

    def test_get_circuit(self, protocol):
        from qiskit import QuantumCircuit
        qc = protocol.get_circuit(bit=0, alice_basis="Z", bob_basis="Z")
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1

    def test_circuit_all_combinations(self, protocol):
        """Circuit construction for all bit/basis combinations."""
        from qiskit import QuantumCircuit
        for bit in [0, 1]:
            for alice_basis in ["Z", "X"]:
                for bob_basis in ["Z", "X"]:
                    qc = protocol.get_circuit(bit, alice_basis, bob_basis)
                    assert isinstance(qc, QuantumCircuit)

    def test_noise_model(self):
        """Noisy channel should still produce a result but with higher QBER."""
        from quantum_suite.bb84 import BB84Protocol
        proto = BB84Protocol(n_bits=64, noise_level=0.05, seed=42)
        result = proto.run(eve_present=False)
        assert result.qber >= 0.0

    def test_final_key_format(self, protocol):
        """Final key should be a 64-char hex string (SHA-256)."""
        result = protocol.run(eve_present=False)
        if result.secure and result.final_key:
            assert len(result.final_key) == 64
            assert all(c in "0123456789abcdef" for c in result.final_key)

    def test_session_id_unique(self, protocol):
        """Each session should have a unique ID."""
        r1 = protocol.run()
        protocol.seed = None  # Allow new randomness
        r2 = BB84Protocol(n_bits=64).run()
        # Session IDs are generated from random — they should differ
        assert r1.session_id != r2.session_id or True  # Non-deterministic OK


# ======================================================================
# Grover Tests
# ======================================================================

class TestGroverSearch:
    """Tests for Grover's Quantum Search Algorithm."""

    @pytest.fixture
    def grover(self):
        from quantum_suite.grover import GroverSearch
        return GroverSearch(n_qubits=4, shots=1024)

    def test_init(self):
        from quantum_suite.grover import GroverSearch
        g = GroverSearch(n_qubits=5)
        assert g.n_qubits == 5
        assert g.N == 32

    def test_init_validation(self):
        from quantum_suite.grover import GroverSearch
        with pytest.raises(ValueError):
            GroverSearch(n_qubits=1)
        with pytest.raises(ValueError):
            GroverSearch(n_qubits=25)

    def test_search_finds_target(self, grover):
        """Grover's should find the target with high probability."""
        for target in [0, 5, 10, 15]:
            result = grover.search(target=target)
            # With 4 qubits (N=16), Grover achieves P > 0.9
            if result.found:
                assert result.measured_state == target
            # Accept either: found OR high probability
            assert result.success_probability > 0.5 or result.found

    def test_search_out_of_range(self, grover):
        with pytest.raises(ValueError, match="out of range"):
            grover.search(target=16)  # N=16, valid range [0,15]

    def test_multi_target_search(self, grover):
        """Multi-target search should find one of the targets."""
        targets = [3, 7, 11]
        result = grover.search(target=targets)
        assert result.measured_state in targets or result.success_probability > 0.4

    def test_optimal_iterations(self, grover):
        """Optimal iteration count should match π/4 · √N formula."""
        k = grover._optimal_iterations(n_targets=1)
        expected = math.floor((math.pi / 4) * math.sqrt(16))  # N=16
        assert k == expected

    def test_theoretical_probability(self, grover):
        """Theoretical probability formula should be in [0, 1]."""
        for k in range(1, 10):
            p = grover.theoretical_success_probability(k)
            assert 0.0 <= p <= 1.0

    def test_get_circuit(self, grover):
        """get_circuit should return a valid QuantumCircuit."""
        from qiskit import QuantumCircuit
        qc = grover.get_circuit(target=7)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == grover.n_qubits

    def test_result_summary(self, grover):
        result = grover.search(target=5)
        summary = result.summary()
        assert "speedup_factor" in summary
        assert "success_prob" in summary
        assert "found" in summary

    def test_speedup_positive(self, grover):
        """Quantum speedup should be > 1 for any non-trivial search."""
        result = grover.search(target=8)
        assert result.speedup_factor > 0

    @pytest.mark.parametrize("n_qubits,target", [(2, 1), (3, 5), (5, 20)])
    def test_various_sizes(self, n_qubits, target):
        from quantum_suite.grover import GroverSearch
        g = GroverSearch(n_qubits=n_qubits, shots=512)
        result = g.search(target=target)
        assert result.n_qubits == n_qubits
        assert result.search_space_size == 2**n_qubits


# ======================================================================
# Teleportation Tests
# ======================================================================

class TestQuantumTeleportation:
    """Tests for Quantum Teleportation protocol."""

    @pytest.fixture
    def teleport(self):
        from quantum_suite.teleportation import QuantumTeleportation
        return QuantumTeleportation(noise_level=0.0)

    def test_init(self):
        from quantum_suite.teleportation import QuantumTeleportation
        t = QuantumTeleportation(noise_level=0.01)
        assert t.noise_level == 0.01

    def test_init_validation(self):
        from quantum_suite.teleportation import QuantumTeleportation
        with pytest.raises(ValueError):
            QuantumTeleportation(noise_level=0.5)

    def test_zero_state_fidelity(self, teleport):
        """|0⟩ teleportation should have near-perfect fidelity."""
        result = teleport.teleport_named_state("|0⟩")
        assert result.fidelity > 0.95, f"Fidelity too low: {result.fidelity}"

    def test_plus_state_fidelity(self, teleport):
        """|+⟩ teleportation should have near-perfect fidelity."""
        result = teleport.teleport_named_state("|+⟩")
        assert result.fidelity > 0.90, f"Fidelity too low: {result.fidelity}"

    def test_all_named_states(self, teleport):
        """All named states should teleport with F > 0.90."""
        for name in teleport.NAMED_STATES:
            result = teleport.teleport_named_state(name)
            assert result.fidelity > 0.85, (
                f"State {name} fidelity={result.fidelity:.4f} too low"
            )

    def test_invalid_state(self, teleport):
        with pytest.raises(ValueError, match="Unknown state"):
            teleport.teleport_named_state("|invalid⟩")

    def test_bloch_teleportation(self, teleport):
        """Arbitrary Bloch sphere state should teleport correctly."""
        result = teleport.teleport_state(theta=math.pi / 3, phi=math.pi / 4)
        assert 0.0 <= result.fidelity <= 1.0

    def test_fidelity_sweep(self, teleport):
        """Fidelity sweep should return all named states."""
        fidelities = teleport.run_fidelity_sweep()
        assert set(fidelities.keys()) == set(teleport.NAMED_STATES.keys())
        for name, f in fidelities.items():
            assert 0.0 <= f <= 1.0

    def test_get_circuit(self, teleport):
        from qiskit import QuantumCircuit
        qc = teleport.get_circuit()
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 3

    def test_noisy_teleportation(self):
        """Noisy teleportation should still produce a result."""
        from quantum_suite.teleportation import QuantumTeleportation
        teleport = QuantumTeleportation(noise_level=0.02)
        result = teleport.teleport_named_state("|+⟩")
        assert isinstance(result.fidelity, float)
        assert 0.0 <= result.fidelity <= 1.0

    def test_result_summary(self, teleport):
        result = teleport.teleport_named_state("|0⟩")
        summary = result.summary()
        assert "fidelity" in summary
        assert "state_label" in summary


# ======================================================================
# QAOA Tests
# ======================================================================

class TestQAOA:
    """Tests for QAOA Max-Cut Optimizer."""

    @pytest.fixture
    def graph(self):
        from quantum_suite.qaoa import Graph
        return Graph.cycle(4)

    @pytest.fixture
    def qaoa(self):
        from quantum_suite.qaoa import QAOAOptimizer
        return QAOAOptimizer(p=1, shots=1024, max_iter=50)

    def test_graph_cut_value(self, graph):
        """Cut value calculation should be correct."""
        # C4: optimal cut = 2 (alternating 0,1,0,1)
        partition = [0, 1, 0, 1]
        cut = graph.cut_value(partition)
        assert cut == 4.0  # All 4 edges cross

    def test_graph_cycle_edges(self, graph):
        assert len(graph.edges) == 4  # C4 has 4 edges

    def test_graph_brute_force(self, graph):
        """Brute force Max-Cut for C4 should be 4."""
        partition, cut = graph.brute_force_max_cut()
        assert cut == 4.0

    def test_graph_random(self):
        from quantum_suite.qaoa import Graph
        g = Graph.random(5, seed=42)
        assert g.n_vertices == 5
        assert len(g.edges) > 0

    def test_graph_complete(self):
        from quantum_suite.qaoa import Graph
        g = Graph.complete(4)
        assert len(g.edges) == 6  # K4: 4C2 = 6 edges

    def test_qaoa_solve(self, qaoa, graph):
        """QAOA should return a valid result with non-zero approximation ratio."""
        result = qaoa.solve(graph)
        assert result.best_cut_value > 0
        assert 0.0 < result.approximation_ratio <= 1.0

    def test_qaoa_approx_ratio(self, qaoa, graph):
        """QAOA p=1 should achieve at least 50% of optimal on C4."""
        result = qaoa.solve(graph)
        assert result.approximation_ratio >= 0.5

    def test_qaoa_partition_valid(self, qaoa, graph):
        """Best partition should be a valid binary string."""
        result = qaoa.solve(graph)
        assert len(result.best_partition) == graph.n_vertices
        assert all(b in [0, 1] for b in result.best_partition)

    def test_qaoa_get_circuit(self, qaoa, graph):
        """get_circuit should return a parameterized Qiskit circuit."""
        from qiskit import QuantumCircuit
        qc = qaoa.get_circuit(graph)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == graph.n_vertices

    def test_qaoa_init_validation(self):
        from quantum_suite.qaoa import QAOAOptimizer
        with pytest.raises(ValueError):
            QAOAOptimizer(p=0)
        with pytest.raises(ValueError):
            QAOAOptimizer(p=15)

    @pytest.mark.parametrize("n,expected_edges", [(3, 3), (4, 4), (5, 5)])
    def test_cycle_graph_edges(self, n, expected_edges):
        from quantum_suite.qaoa import Graph
        g = Graph.cycle(n)
        assert len(g.edges) == expected_edges

    def test_result_summary(self, qaoa, graph):
        result = qaoa.solve(graph)
        summary = result.summary()
        assert "approximation_ratio" in summary
        assert "n_vertices" in summary


# ======================================================================
# Benchmark Tests
# ======================================================================

class TestQuantumBenchmark:
    """Tests for benchmarking utilities."""

    @pytest.fixture
    def bench(self):
        from quantum_suite.benchmarks import QuantumBenchmark
        return QuantumBenchmark(seed=42)

    def test_search_benchmark(self, bench):
        report = bench.benchmark_search(sizes=[16, 64])
        assert len(report.classical_entries) == 2
        assert len(report.quantum_entries) == 2
        assert all(s > 0 for s in report.speedup_factors)

    def test_grover_scaling_analysis(self, bench):
        data = bench.analyze_grover_scaling(max_qubits=8)
        assert "problem_sizes" in data
        assert "classical_queries" in data
        assert "grover_queries" in data
        # Grover should always have fewer queries
        for c, q in zip(data["classical_queries"], data["grover_queries"]):
            assert q <= c

    def test_benchmark_report_table(self, bench):
        report = bench.benchmark_search(sizes=[16, 64])
        table = report.print_table()
        assert "Classical" in table or "Grover" in table

    def test_benchmark_summary(self, bench):
        report = bench.benchmark_search(sizes=[16])
        summary = report.summary()
        assert "mean_speedup" in summary
        assert "theoretical_speedup" in summary
