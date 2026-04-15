"""
Quantum Teleportation
======================
Quantum teleportation transmits an unknown quantum state from Alice to Bob
using a shared entangled pair and 2 classical bits of communication.

Proposed by Bennett et al. in 1993, it demonstrates one of quantum mechanics'
most counterintuitive features: the ability to transfer quantum information
without physically moving a particle.

Important Note:
    Teleportation does NOT transmit information faster than light — the 2
    classical bits are still required, which limits speed to c.
    It teleports the quantum STATE, not the physical particle.

Protocol:
    1. Create a Bell pair (maximally entangled) between Alice & Bob
    2. Alice entangles her data qubit with her Bell qubit (CNOT + H)
    3. Alice measures her 2 qubits → gets 2 classical bits
    4. Alice sends 2 classical bits to Bob via classical channel
    5. Bob applies corrections (X and/or Z) based on received bits
    6. Bob's qubit is now in the original state |ψ⟩

Fidelity:
    Ideal teleportation: F = 1.0 (perfect state transfer)
    With noise/decoherence: F < 1.0, quantified by state fidelity
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

logger = logging.getLogger(__name__)


@dataclass
class TeleportationResult:
    """Result of a quantum teleportation experiment."""

    input_state_vector: np.ndarray
    output_state_vector: np.ndarray
    fidelity: float
    alice_measurement: tuple[int, int]
    correction_applied: str
    success: bool
    noise_level: float
    n_shots: int
    state_label: str

    def summary(self) -> dict:
        return {
            "state_label": self.state_label,
            "fidelity": f"{self.fidelity:.6f}",
            "alice_bits": self.alice_measurement,
            "correction": self.correction_applied,
            "success": self.success,
            "noise": self.noise_level,
        }


class QuantumTeleportation:
    """
    Quantum Teleportation Protocol Simulator.

    Teleports arbitrary single-qubit states from Alice to Bob using
    a shared Bell pair and 2 classical bits of communication.

    Args:
        noise_level (float): Depolarizing noise on all gates [0.0, 0.1].
        fidelity_threshold (float): Minimum fidelity to consider success.

    Example:
        >>> teleport = QuantumTeleportation(noise_level=0.005)

        >>> # Teleport |+⟩ state
        >>> result = teleport.teleport_state(theta=math.pi/2, phi=0)
        >>> print(f"Fidelity: {result.fidelity:.4f}")

        >>> # Teleport named states
        >>> result = teleport.teleport_named_state("|+⟩")
        >>> result = teleport.teleport_named_state("|T⟩")  # T gate state
    """

    NAMED_STATES = {
        "|0⟩": (0, 0),
        "|1⟩": (math.pi, 0),
        "|+⟩": (math.pi / 2, 0),
        "|-⟩": (math.pi / 2, math.pi),
        "|i⟩": (math.pi / 2, math.pi / 2),
        "|T⟩": (math.pi / 2, math.pi / 4),  # T-gate rotated state
        "|Y⟩": (math.pi / 2, math.pi / 2),  # Y eigenstate
    }

    def __init__(self, noise_level: float = 0.0, fidelity_threshold: float = 0.95):
        if not 0.0 <= noise_level <= 0.1:
            raise ValueError("noise_level must be in [0.0, 0.1]")

        self.noise_level = noise_level
        self.fidelity_threshold = fidelity_threshold
        self.simulator = AerSimulator(method="statevector")
        self._noise_model = self._build_noise_model() if noise_level > 0 else None

        logger.info(f"QuantumTeleportation: noise={noise_level:.3f}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def teleport_state(self, theta: float, phi: float = 0.0) -> TeleportationResult:
        """
        Teleport a qubit state defined by Bloch sphere angles.

        The state is: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}·sin(θ/2)|1⟩

        Args:
            theta: Polar angle θ ∈ [0, π] on the Bloch sphere.
            phi: Azimuthal angle φ ∈ [0, 2π] on the Bloch sphere.

        Returns:
            TeleportationResult with fidelity and measurement data.
        """
        state_label = f"Bloch(θ={theta:.3f}, φ={phi:.3f})"
        input_sv = self._bloch_to_statevector(theta, phi)
        return self._run_teleportation(input_sv, theta, phi, state_label)

    def teleport_named_state(self, name: str) -> TeleportationResult:
        """
        Teleport one of the standard named quantum states.

        Args:
            name: State label from NAMED_STATES keys.
                  Options: |0⟩, |1⟩, |+⟩, |-⟩, |i⟩, |T⟩, |Y⟩

        Returns:
            TeleportationResult with fidelity and measurement data.

        Raises:
            ValueError: If state name is not recognized.
        """
        if name not in self.NAMED_STATES:
            raise ValueError(
                f"Unknown state '{name}'. Valid: {list(self.NAMED_STATES.keys())}"
            )
        theta, phi = self.NAMED_STATES[name]
        input_sv = self._bloch_to_statevector(theta, phi)
        return self._run_teleportation(input_sv, theta, phi, name)

    def run_fidelity_sweep(self) -> dict[str, float]:
        """
        Teleport all named states and return fidelity for each.

        Returns:
            Dict mapping state name → fidelity.
        """
        results = {}
        for name in self.NAMED_STATES:
            result = self.teleport_named_state(name)
            results[name] = result.fidelity
            logger.info(f"  {name}: fidelity={result.fidelity:.4f}")
        return results

    def get_circuit(self, theta: float = math.pi / 2, phi: float = 0.0) -> QuantumCircuit:
        """Return the 3-qubit teleportation circuit (without execution)."""
        qc, _ = self._build_circuit(theta, phi)
        return qc

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_circuit(
        self, theta: float, phi: float
    ) -> tuple[QuantumCircuit, QuantumCircuit]:
        """
        Build the full 3-qubit teleportation circuit.

        Qubit layout:
            q0: Alice's data qubit (state to teleport)
            q1: Alice's half of Bell pair
            q2: Bob's half of Bell pair (output)
        """
        # 3 qubits, 2 classical bits for Alice's measurements
        qc = QuantumCircuit(3, 2, name="Quantum_Teleportation")

        # — Step 1: Prepare Alice's data qubit |ψ⟩ —
        qc.ry(theta, 0)          # θ rotation
        if phi != 0:
            qc.rz(phi, 0)        # φ phase
        qc.barrier(label="|ψ⟩ prep")

        # — Step 2: Create Bell pair between q1 (Alice) and q2 (Bob) —
        qc.h(1)                  # |+⟩ on Alice's Bell qubit
        qc.cx(1, 2)              # Entangle: |Φ+⟩ = (|00⟩+|11⟩)/√2
        qc.barrier(label="Bell pair")

        # — Step 3: Alice's Bell measurement —
        qc.cx(0, 1)              # CNOT: data → Alice's Bell qubit
        qc.h(0)                  # Hadamard on data qubit
        qc.barrier(label="Alice BSM")

        qc.measure(0, 0)         # Measure data qubit → classical bit 0
        qc.measure(1, 1)         # Measure Bell qubit → classical bit 1

        # — Step 4: Bob applies conditional corrections —
        # If bit 1 = 1: apply X to Bob's qubit
        with qc.if_else((qc.cregs[0][1], 1)):
            qc.x(2)
        # If bit 0 = 1: apply Z to Bob's qubit
        with qc.if_else((qc.cregs[0][0], 1)):
            qc.z(2)

        # Build state-vector-only version (no measurements) for fidelity check
        qc_sv = QuantumCircuit(3, name="Teleport_SV")
        qc_sv.ry(theta, 0)
        if phi != 0:
            qc_sv.rz(phi, 0)
        qc_sv.h(1)
        qc_sv.cx(1, 2)
        qc_sv.cx(0, 1)
        qc_sv.h(0)
        qc_sv.cx(1, 2)   # simulate if bit1=1: X on Bob
        qc_sv.cz(0, 2)   # simulate if bit0=1: Z on Bob

        return qc, qc_sv

    # ------------------------------------------------------------------
    # Execution & analysis
    # ------------------------------------------------------------------

    def _run_teleportation(
        self, input_sv: np.ndarray, theta: float, phi: float, label: str
    ) -> TeleportationResult:
        """Execute the teleportation circuit and measure fidelity."""
        qc_full, qc_sv = self._build_circuit(theta, phi)

        # Statevector simulation for fidelity
        sv_sim = AerSimulator(method="statevector")
        sv_result = sv_sim.run(transpile(qc_sv, sv_sim)).result()
        full_sv = np.array(sv_result.get_statevector())

        # Extract Bob's qubit (q2) from the 3-qubit statevector
        # Trace out Alice's qubits (q0, q1) — average over all Alice states
        bob_sv = self._extract_bob_state(full_sv)
        fidelity = float(np.abs(np.dot(input_sv.conj(), bob_sv)) ** 2)

        # Get classical measurement for display (run with shots)
        meas_sim = AerSimulator()
        meas_qc = self._build_meas_circuit(theta, phi)
        job = meas_sim.run(
            transpile(meas_qc, meas_sim),
            shots=1024,
            noise_model=self._noise_model,
        )
        counts = job.result().get_counts()
        best = max(counts, key=counts.get)
        bit0, bit1 = int(best[1]), int(best[0])  # Qiskit reverses bit order
        correction = self._correction_label(bit0, bit1)

        return TeleportationResult(
            input_state_vector=input_sv,
            output_state_vector=bob_sv,
            fidelity=fidelity,
            alice_measurement=(bit0, bit1),
            correction_applied=correction,
            success=fidelity >= self.fidelity_threshold,
            noise_level=self.noise_level,
            n_shots=1024,
            state_label=label,
        )

    def _build_meas_circuit(self, theta: float, phi: float) -> QuantumCircuit:
        """Build circuit with only Alice's measurement (for statistics)."""
        qc = QuantumCircuit(3, 2)
        qc.ry(theta, 0)
        if phi != 0:
            qc.rz(phi, 0)
        qc.h(1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        return qc

    def _extract_bob_state(self, full_sv: np.ndarray) -> np.ndarray:
        """
        Extract Bob's marginal state from the 3-qubit statevector.
        Performs partial trace over Alice's qubits (q0, q1).
        """
        # Reshape to (q0, q1, q2) = (2, 2, 2)
        sv = full_sv.reshape(2, 2, 2)
        # Trace out q0 and q1 (Bob's qubit is q2)
        # Density matrix: ρ_B = Tr_A[|ψ⟩⟨ψ|]
        rho_full = np.outer(full_sv, full_sv.conj()).reshape(2, 2, 2, 2, 2, 2)
        rho_bob = np.einsum("iijjkl->kl", rho_full)
        # Extract dominant eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(rho_bob)
        bob_state = eigenvectors[:, np.argmax(eigenvalues)]
        return bob_state

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _bloch_to_statevector(self, theta: float, phi: float) -> np.ndarray:
        """Convert Bloch sphere (θ, φ) to state vector [α, β]."""
        alpha = math.cos(theta / 2)
        beta = math.sin(theta / 2) * complex(math.cos(phi), math.sin(phi))
        sv = np.array([alpha, beta], dtype=complex)
        return sv / np.linalg.norm(sv)

    def _correction_label(self, bit0: int, bit1: int) -> str:
        """Human-readable correction string."""
        ops = []
        if bit1 == 1:
            ops.append("X")
        if bit0 == 1:
            ops.append("Z")
        return " · ".join(ops) if ops else "I (identity)"

    def _build_noise_model(self) -> NoiseModel:
        nm = NoiseModel()
        e1 = depolarizing_error(self.noise_level, 1)
        e2 = depolarizing_error(self.noise_level * 2, 2)
        nm.add_all_qubit_quantum_error(e1, ["h", "x", "z", "ry", "rz"])
        nm.add_all_qubit_quantum_error(e2, ["cx", "cz"])
        return nm
