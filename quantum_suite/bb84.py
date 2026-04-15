"""
BB84 Quantum Key Distribution Protocol
=======================================
Implements the BB84 protocol — the first quantum cryptography protocol,
proposed by Charles Bennett and Gilles Brassard in 1984.

The BB84 protocol allows two parties (Alice and Bob) to generate a shared
secret key that is provably secure against eavesdropping, leveraging the
no-cloning theorem of quantum mechanics.

Protocol Flow:
    1. Alice prepares qubits in random bases (Z or X)
    2. Bob measures in random bases
    3. Alice and Bob publicly compare bases (not bits)
    4. They keep only matching-basis bits → raw key
    5. Error rate estimation reveals eavesdropping (Eve)
    6. Privacy amplification produces final secure key

Security Guarantee:
    Any eavesdropper (Eve) disturbs the quantum states, introducing
    ~25% QBER (Quantum Bit Error Rate). A QBER > 11% signals an attack.
"""

import random
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

logger = logging.getLogger(__name__)


@dataclass
class BB84Result:
    """Structured result from a BB84 key exchange session."""

    alice_bits: list[int]
    alice_bases: list[str]
    bob_bases: list[str]
    bob_measurements: list[int]
    sifted_key: list[int]
    qber: float
    secure: bool
    final_key: Optional[str] = None
    eve_detected: bool = False
    raw_key_length: int = 0
    sifted_key_length: int = 0
    session_id: str = ""

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "raw_bits_sent": self.raw_key_length,
            "sifted_key_bits": self.sifted_key_length,
            "qber_percent": round(self.qber * 100, 2),
            "eve_detected": self.eve_detected,
            "key_secure": self.secure,
            "final_key_hash": (
                hashlib.sha256(self.final_key.encode()).hexdigest()[:16]
                if self.final_key
                else None
            ),
        }


class BB84Protocol:
    """
    Full simulation of the BB84 Quantum Key Distribution protocol.

    Uses Qiskit circuits to genuinely simulate quantum bit preparation
    and measurement, including optional noise models and eavesdropping.

    Args:
        n_bits (int): Number of raw qubits to send. Default 256.
        noise_level (float): Depolarizing noise probability [0, 0.1].
        qber_threshold (float): Max acceptable QBER. Default 0.11 (11%).
        seed (int): Random seed for reproducibility.

    Example:
        >>> protocol = BB84Protocol(n_bits=512, noise_level=0.01)
        >>> result = protocol.run(eve_present=False)
        >>> print(f"Final key QBER: {result.qber:.2%}")
        >>> print(f"Secure: {result.secure}")
    """

    BASES = ["Z", "X"]  # Z = computational, X = Hadamard

    def __init__(
        self,
        n_bits: int = 256,
        noise_level: float = 0.0,
        qber_threshold: float = 0.11,
        seed: Optional[int] = None,
    ):
        if not 64 <= n_bits <= 4096:
            raise ValueError("n_bits must be between 64 and 4096")
        if not 0.0 <= noise_level <= 0.1:
            raise ValueError("noise_level must be in [0.0, 0.1]")

        self.n_bits = n_bits
        self.noise_level = noise_level
        self.qber_threshold = qber_threshold
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.simulator = AerSimulator()
        self._noise_model = self._build_noise_model() if noise_level > 0 else None

        logger.info(
            f"BB84Protocol initialized: n_bits={n_bits}, noise={noise_level:.3f}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, eve_present: bool = False) -> BB84Result:
        """
        Execute a complete BB84 key exchange session.

        Args:
            eve_present (bool): If True, simulates an intercept-resend attack
                                by Eve on 50% of qubits.

        Returns:
            BB84Result: Full session result including key and security metrics.
        """
        session_id = hashlib.md5(
            f"{random.random()}".encode()
        ).hexdigest()[:8].upper()
        logger.info(f"[{session_id}] Starting BB84 session (Eve={eve_present})")

        # Step 1 & 2: Alice prepares qubits, Bob measures
        alice_bits = self._generate_random_bits(self.n_bits)
        alice_bases = self._generate_random_bases(self.n_bits)
        bob_bases = self._generate_random_bases(self.n_bits)

        # Step 3: Simulate quantum channel (with optional Eve)
        bob_measurements = self._quantum_channel(
            alice_bits, alice_bases, bob_bases, eve_present
        )

        # Step 4: Sifting — keep bits where bases match
        sifted_alice, sifted_bob, matching_indices = self._sift_key(
            alice_bits, alice_bases, bob_bases, bob_measurements
        )

        # Step 5: QBER estimation (sacrifice ~20% of sifted key)
        qber = self._estimate_qber(sifted_alice, sifted_bob)
        eve_detected = qber > self.qber_threshold
        secure = not eve_detected and len(sifted_alice) > 32

        # Step 6: Privacy amplification → final key
        final_key = None
        if secure:
            remaining_alice = sifted_alice[len(sifted_alice) // 5 :]
            final_key = self._privacy_amplification(remaining_alice)

        result = BB84Result(
            alice_bits=alice_bits,
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_measurements=bob_measurements,
            sifted_key=sifted_alice,
            qber=qber,
            secure=secure,
            final_key=final_key,
            eve_detected=eve_detected,
            raw_key_length=self.n_bits,
            sifted_key_length=len(sifted_alice),
            session_id=session_id,
        )

        logger.info(
            f"[{session_id}] Session complete: QBER={qber:.2%}, "
            f"Secure={secure}, Key bits={len(sifted_alice)}"
        )
        return result

    def get_circuit(self, bit: int, alice_basis: str, bob_basis: str) -> QuantumCircuit:
        """
        Build and return the Qiskit circuit for a single qubit exchange.

        Args:
            bit (int): Alice's bit value (0 or 1).
            alice_basis (str): Alice's encoding basis ('Z' or 'X').
            bob_basis (str): Bob's measurement basis ('Z' or 'X').

        Returns:
            QuantumCircuit: The single-qubit BB84 circuit.
        """
        qc = QuantumCircuit(1, 1, name=f"BB84_b{bit}_{alice_basis}_{bob_basis}")

        # Alice encodes
        if bit == 1:
            qc.x(0)  # |1⟩ in Z-basis
        if alice_basis == "X":
            qc.h(0)  # Rotate to X-basis: |+⟩ or |−⟩

        qc.barrier(label="Channel")

        # Bob measures
        if bob_basis == "X":
            qc.h(0)  # Rotate back before measuring
        qc.measure(0, 0)

        return qc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_random_bits(self, n: int) -> list[int]:
        return [random.randint(0, 1) for _ in range(n)]

    def _generate_random_bases(self, n: int) -> list[str]:
        return [random.choice(self.BASES) for _ in range(n)]

    def _quantum_channel(
        self,
        alice_bits: list[int],
        alice_bases: list[str],
        bob_bases: list[str],
        eve_present: bool,
    ) -> list[int]:
        """Simulate quantum transmission of all n_bits using Qiskit."""
        circuits = []
        for i in range(self.n_bits):
            effective_alice_bit = alice_bits[i]
            effective_alice_basis = alice_bases[i]

            if eve_present and random.random() < 0.5:
                # Eve intercepts: random basis, re-prepares
                eve_basis = random.choice(self.BASES)
                eve_measurement = self._single_measure(
                    effective_alice_bit, effective_alice_basis, eve_basis
                )
                effective_alice_bit = eve_measurement
                effective_alice_basis = eve_basis

            qc = self.get_circuit(effective_alice_bit, effective_alice_basis, bob_bases[i])
            circuits.append(qc)

        # Batch transpile and execute for efficiency
        transpiled = transpile(circuits, self.simulator)
        job = self.simulator.run(
            transpiled,
            shots=1,
            noise_model=self._noise_model,
        )
        result = job.result()

        measurements = []
        for i, qc in enumerate(circuits):
            counts = result.get_counts(i)
            bit = int(max(counts, key=counts.get))
            measurements.append(bit)

        return measurements

    def _single_measure(self, bit: int, alice_basis: str, meas_basis: str) -> int:
        """Simulate a single qubit measurement (used by Eve)."""
        qc = self.get_circuit(bit, alice_basis, meas_basis)
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=1)
        counts = job.result().get_counts()
        return int(max(counts, key=counts.get))

    def _sift_key(
        self,
        alice_bits: list[int],
        alice_bases: list[str],
        bob_bases: list[str],
        bob_measurements: list[int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Keep only bits where Alice and Bob used the same basis."""
        alice_sifted, bob_sifted, indices = [], [], []
        for i in range(self.n_bits):
            if alice_bases[i] == bob_bases[i]:
                alice_sifted.append(alice_bits[i])
                bob_sifted.append(bob_measurements[i])
                indices.append(i)
        return alice_sifted, bob_sifted, indices

    def _estimate_qber(
        self, alice_key: list[int], bob_key: list[int], sample_frac: float = 0.2
    ) -> float:
        """Estimate QBER from a random sample of the sifted key."""
        if not alice_key:
            return 1.0
        n_sample = max(1, int(len(alice_key) * sample_frac))
        indices = random.sample(range(len(alice_key)), n_sample)
        errors = sum(1 for i in indices if alice_key[i] != bob_key[i])
        return errors / n_sample

    def _privacy_amplification(self, key_bits: list[int]) -> str:
        """
        Apply privacy amplification via SHA-256 hashing.
        Reduces key length but removes any partial information Eve may have.
        """
        raw = "".join(map(str, key_bits))
        return hashlib.sha256(raw.encode()).hexdigest()

    def _build_noise_model(self) -> NoiseModel:
        """Build a depolarizing noise model for realistic simulation."""
        noise_model = NoiseModel()
        error_1q = depolarizing_error(self.noise_level, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ["h", "x", "measure"])
        return noise_model
