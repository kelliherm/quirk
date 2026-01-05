"""
Quantum circuit simulator for executing quantum circuits and computing statevectors.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from quirk.circuit.instruction import Instruction
from quirk.circuit.quantumcircuit import QuantumCircuit
from quirk.simulation.statevector import Statevector


class Simulator:
    """Simulator for executing quantum circuits."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the simulator.

        Args:
            seed: Random seed for reproducible measurements
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def run(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        initial_statevector: Optional[Statevector] = None,
    ) -> "SimulatorResult":
        """
        Execute a quantum circuit.

        Args:
            circuit: The quantum circuit to execute
            shots: Number of measurement shots
            initial_statevector: Optional initial state (default: |0...0⟩)

        Returns:
            SimulatorResult containing execution results
        """
        # Initialize statevector
        if initial_statevector is not None:
            if initial_statevector.num_qubits != circuit.num_qubits:
                raise ValueError(
                    f"Initial statevector has {initial_statevector.num_qubits} qubits, "
                    f"but circuit has {circuit.num_qubits} qubits"
                )
            statevector = Statevector(initial_statevector)
        else:
            # Start in |0...0⟩ state
            dim = 2**circuit.num_qubits
            data = np.zeros(dim, dtype=complex)
            data[0] = 1.0
            statevector = Statevector(data)

        # Apply gates
        for instruction in circuit.instructions:
            if instruction.gate.name == "measure":
                # Skip measurements during statevector evolution
                continue

            # Build the full circuit unitary for this gate
            unitary = self._build_gate_unitary(instruction, circuit.num_qubits)

            # Apply the unitary to the statevector
            statevector = statevector.evolve(unitary)

        # Handle measurements
        measurements = self._extract_measurements(circuit)
        if measurements:
            counts = self._measure_statevector(
                statevector, measurements, shots, circuit.num_clbits
            )
        else:
            counts = {}

        return SimulatorResult(
            statevector=statevector,
            counts=counts,
            shots=shots,
            circuit=circuit,
        )

    def _build_gate_unitary(
        self, instruction: Instruction, num_qubits: int
    ) -> np.ndarray:
        """
        Build the full unitary matrix for a gate acting on specific qubits.

        Args:
            instruction: The instruction containing gate and qubit information
            num_qubits: Total number of qubits in the circuit

        Returns:
            Full unitary matrix for the entire circuit
        """
        gate = instruction.gate
        qubits = instruction.qubits

        # For single-qubit gates
        if gate.num_qubits == 1:
            return self._single_qubit_unitary(gate.to_matrix(), qubits[0], num_qubits)

        # For two-qubit gates
        elif gate.num_qubits == 2:
            return self._two_qubit_unitary(
                gate.to_matrix(), qubits[0], qubits[1], num_qubits
            )

        # For three-qubit gates
        elif gate.num_qubits == 3:
            return self._three_qubit_unitary(
                gate.to_matrix(), qubits[0], qubits[1], qubits[2], num_qubits
            )

        else:
            raise NotImplementedError(
                f"Gates with {gate.num_qubits} qubits not yet supported"
            )

    def _single_qubit_unitary(
        self, gate_matrix: np.ndarray, target: int, num_qubits: int
    ) -> np.ndarray:
        """
        Build unitary for a single-qubit gate.

        Args:
            gate_matrix: 2x2 gate matrix
            target: Target qubit index
            num_qubits: Total number of qubits

        Returns:
            Full unitary matrix
        """
        # Start with identity on first qubit
        if target == 0:
            unitary = gate_matrix
        else:
            unitary = np.eye(2, dtype=complex)

        # Build tensor product for remaining qubits
        for i in range(1, num_qubits):
            if i == target:
                unitary = np.kron(unitary, gate_matrix)
            else:
                unitary = np.kron(unitary, np.eye(2, dtype=complex))

        return unitary

    def _two_qubit_unitary(
        self,
        gate_matrix: np.ndarray,
        control: int,
        target: int,
        num_qubits: int,
    ) -> np.ndarray:
        """
        Build unitary for a two-qubit gate.

        Args:
            gate_matrix: 4x4 gate matrix
            control: Control qubit index
            target: Target qubit index
            num_qubits: Total number of qubits

        Returns:
            Full unitary matrix
        """
        dim = 2**num_qubits
        unitary = np.eye(dim, dtype=complex)

        # Get the qubits in the right order
        qubits = sorted([control, target])
        qubit0, qubit1 = qubits

        # For each basis state, apply the gate if it acts on the right qubits
        for i in range(dim):
            for j in range(dim):
                # Extract bits for the relevant qubits
                bit0_i = (i >> (num_qubits - 1 - qubit0)) & 1
                bit1_i = (i >> (num_qubits - 1 - qubit1)) & 1
                bit0_j = (j >> (num_qubits - 1 - qubit0)) & 1
                bit1_j = (j >> (num_qubits - 1 - qubit1)) & 1

                # Check if other qubits are the same
                other_bits_same = True
                for q in range(num_qubits):
                    if q != qubit0 and q != qubit1:
                        bit_i = (i >> (num_qubits - 1 - q)) & 1
                        bit_j = (j >> (num_qubits - 1 - q)) & 1
                        if bit_i != bit_j:
                            other_bits_same = False
                            break

                if other_bits_same:
                    # Map to gate matrix indices
                    if control < target:
                        gate_i = (bit0_i << 1) | bit1_i
                        gate_j = (bit0_j << 1) | bit1_j
                    else:
                        gate_i = (bit1_i << 1) | bit0_i
                        gate_j = (bit1_j << 1) | bit0_j

                    unitary[i, j] = gate_matrix[gate_i, gate_j]

        return unitary

    def _three_qubit_unitary(
        self,
        gate_matrix: np.ndarray,
        qubit0: int,
        qubit1: int,
        qubit2: int,
        num_qubits: int,
    ) -> np.ndarray:
        """
        Build unitary for a three-qubit gate.

        Args:
            gate_matrix: 8x8 gate matrix
            qubit0: First qubit index
            qubit1: Second qubit index
            qubit2: Third qubit index
            num_qubits: Total number of qubits

        Returns:
            Full unitary matrix
        """
        dim = 2**num_qubits
        unitary = np.eye(dim, dtype=complex)

        qubits = sorted([qubit0, qubit1, qubit2])
        original_order = [qubit0, qubit1, qubit2]
        permutation = [qubits.index(q) for q in original_order]

        for i in range(dim):
            for j in range(dim):
                # Extract bits for the relevant qubits
                bits_i = [(i >> (num_qubits - 1 - qubits[k])) & 1 for k in range(3)]
                bits_j = [(j >> (num_qubits - 1 - qubits[k])) & 1 for k in range(3)]

                # Check if other qubits are the same
                other_bits_same = True
                for q in range(num_qubits):
                    if q not in qubits:
                        bit_i = (i >> (num_qubits - 1 - q)) & 1
                        bit_j = (j >> (num_qubits - 1 - q)) & 1
                        if bit_i != bit_j:
                            other_bits_same = False
                            break

                if other_bits_same:
                    # Reorder bits according to original gate order
                    reordered_i = [bits_i[permutation[k]] for k in range(3)]
                    reordered_j = [bits_j[permutation[k]] for k in range(3)]

                    gate_i = (
                        (reordered_i[0] << 2) | (reordered_i[1] << 1) | reordered_i[2]
                    )
                    gate_j = (
                        (reordered_j[0] << 2) | (reordered_j[1] << 1) | reordered_j[2]
                    )

                    unitary[i, j] = gate_matrix[gate_i, gate_j]

        return unitary

    def _extract_measurements(self, circuit: QuantumCircuit) -> List[Tuple[int, int]]:
        """
        Extract measurement operations from circuit.

        Args:
            circuit: The quantum circuit

        Returns:
            List of (qubit, classical_bit) tuples
        """
        measurements = []
        for instruction in circuit.instructions:
            if instruction.gate.name == "measure":
                qubit = instruction.qubits[0]
                cbit = instruction.classical_bits[0]
                measurements.append((qubit, cbit))
        return measurements

    def _measure_statevector(
        self,
        statevector: Statevector,
        measurements: List[Tuple[int, int]],
        shots: int,
        num_clbits: int,
    ) -> Dict[str, int]:
        """
        Perform measurements on the statevector.

        Args:
            statevector: The quantum statevector
            measurements: List of (qubit, classical_bit) pairs
            shots: Number of measurement shots
            num_clbits: Total number of classical bits

        Returns:
            Dictionary of measurement counts
        """
        # Sample from the statevector
        sample_counts = statevector.sample_counts(shots, seed=self.seed)

        # Map quantum measurements to classical register
        classical_counts = {}

        for quantum_state, count in sample_counts.items():
            # Initialize classical register to all zeros
            classical_bits = ["0"] * num_clbits

            # Map measured qubits to classical bits
            for qubit_idx, cbit_idx in measurements:
                classical_bits[cbit_idx] = quantum_state[qubit_idx]

            classical_state = "".join(classical_bits)
            classical_counts[classical_state] = (
                classical_counts.get(classical_state, 0) + count
            )

        return dict(sorted(classical_counts.items()))

    def get_statevector(
        self, circuit: QuantumCircuit, initial_statevector: Optional[Statevector] = None
    ) -> Statevector:
        """
        Get the final statevector without performing measurements.

        Args:
            circuit: The quantum circuit to execute
            initial_statevector: Optional initial state (default: |0...0⟩)

        Returns:
            Final statevector after applying all gates
        """
        result = self.run(circuit, shots=0, initial_statevector=initial_statevector)
        return result.statevector


class SimulatorResult:
    """Results from executing a quantum circuit."""

    def __init__(
        self,
        statevector: Statevector,
        counts: Dict[str, int],
        shots: int,
        circuit: QuantumCircuit,
    ):
        """
        Initialize simulation result.

        Args:
            statevector: Final quantum statevector
            counts: Measurement counts
            shots: Number of shots executed
            circuit: The quantum circuit that was executed
        """
        self.statevector = statevector
        self.counts = counts
        self.shots = shots
        self.circuit = circuit

    def get_counts(self) -> Dict[str, int]:
        """Get measurement counts."""
        return self.counts

    def get_statevector(self) -> Statevector:
        """Get the final statevector."""
        return self.statevector

    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities."""
        if not self.counts:
            return self.statevector.probabilities_dict()

        probs = {}
        for state, count in self.counts.items():
            probs[state] = count / self.shots
        return probs

    def __repr__(self) -> str:
        return f"SimulatorResult(shots={self.shots}, counts={len(self.counts)} unique states)"

    def __str__(self) -> str:
        lines = [f"Simulation Result ({self.shots} shots):"]
        lines.append("-" * 50)

        if self.counts:
            lines.append("Measurement counts:")
            for state, count in sorted(
                self.counts.items(), key=lambda x: x[1], reverse=True
            ):
                prob = count / self.shots
                bar = "#" * int(prob * 40)
                lines.append(f"  |{state}>: {count:4d} ({prob:6.2%}) {bar}")
        else:
            lines.append("No measurements performed.")
            lines.append("\nFinal statevector:")
            lines.append(str(self.statevector))

        return "\n".join(lines)
