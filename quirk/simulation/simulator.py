"""Execute circuit operations in DAG dependency order."""
from typing import Dict, Optional
from numbers import Integral
import numpy as np
from quirk.circuit.quantumcircuit import QuantumCircuit
from quirk.simulation.statevector import Statevector


def apply_gate(data, matrix, qubits, num_qubits):
    """Contract a local gate without allocating a full-system unitary."""
    axes = list(qubits) + [q for q in range(num_qubits) if q not in qubits]
    tensor = data.reshape([2] * num_qubits).transpose(axes)
    evolved = matrix @ tensor.reshape(2 ** len(qubits), -1)
    return evolved.reshape([2] * num_qubits).transpose(np.argsort(axes)).reshape(-1)


class Simulator:
    """Statevector evolution and terminal readout; qubit 0 is leftmost."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def run(self, circuit: QuantumCircuit, shots=1024, initial_statevector=None):
        if isinstance(shots, bool) or not isinstance(shots, Integral):
            raise TypeError("shots must be an integer")
        if shots < 0:
            raise ValueError("shots must be nonnegative")
        state = Statevector.from_int(0, circuit.num_qubits) if initial_statevector is None else Statevector(initial_statevector)
        if state.num_qubits != circuit.num_qubits:
            raise ValueError("Initial statevector qubit count does not match circuit")
        measured = []
        data = state.data()
        for _, node in circuit.get_gate_nodes():
            if node["type"] == "measurement":
                measured.extend(node["qubits"])
            else:
                data = apply_gate(data, node["gate"].to_matrix(), node["qubits"], circuit.num_qubits)
        state = Statevector(data)
        counts = {}
        measured.sort()
        if measured and shots:
            samples = np.random.default_rng(self.seed).choice(state.dim(), size=shots, p=state.probabilities())
            for sample in samples:
                label = "".join(str((int(sample) >> (circuit.num_qubits - 1 - q)) & 1) for q in measured)
                counts[label] = counts.get(label, 0) + 1
        return SimulatorResult(state, dict(sorted(counts.items())), shots, circuit)

    def get_statevector(self, circuit, initial_statevector=None):
        return self.run(circuit, shots=0, initial_statevector=initial_statevector).statevector


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
