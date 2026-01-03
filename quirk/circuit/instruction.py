"""
Instruction class for representing quantum circuit operations.
"""

from typing import List, Optional

from quirk.circuit.gate import Gate


class Instruction:
    """Represents a single instruction in a quantum circuit."""

    def __init__(
        self,
        gate: Gate,
        qubits: List[int],
        classical_bits: Optional[List[int]] = None,
    ):
        """
        Initialize an instruction.

        Args:
            gate: The gate to apply
            qubits: List of qubit indices the gate acts on
            classical_bits: Optional list of classical bit indices for measurements
        """
        self.gate = gate
        self.qubits = qubits
        self.classical_bits = classical_bits or []

    def __repr__(self) -> str:
        qubits_str = ", ".join(str(q) for q in self.qubits)
        if self.classical_bits:
            cbits_str = ", ".join(str(c) for c in self.classical_bits)
            return f"{self.gate} q[{qubits_str}] -> c[{cbits_str}]"
        return f"{self.gate} q[{qubits_str}]"
