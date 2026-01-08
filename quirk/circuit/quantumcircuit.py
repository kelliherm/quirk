"""
Quantum circuit implementation for building and managing quantum circuits.
"""

from typing import List, Optional, Union

import numpy as np

from quirk.circuit.gate import (
    CCXGate,
    CNOTGate,
    CSWAPGate,
    CXGate,
    CYGate,
    CZGate,
    FredkinGate,
    Gate,
    HGate,
    IGate,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    SWAPGate,
    TdgGate,
    TGate,
    ToffoliGate,
    U3Gate,
    XGate,
    YGate,
    ZGate,
)
from quirk.circuit.instruction import Instruction
from quirk.circuit.register import ClassicalRegister, QuantumRegister


class QuantumCircuit:
    """Main class for building and managing quantum circuits."""

    def __init__(
        self,
        qubits: Optional[Union[int, QuantumRegister]] = None,
        classical_bits: Optional[Union[int, ClassicalRegister]] = None,
    ) -> None:
        """
        Initialize a quantum circuit.

        Args:
            qubits: Number of qubits or a QuantumRegister
            classical_bits: Number of classical bits or a ClassicalRegister
        """
        # Handle quantum register
        if isinstance(qubits, QuantumRegister):
            self.num_qubits = qubits.size
            self.qreg = qubits
        elif isinstance(qubits, int):
            self.num_qubits = qubits
            self.qreg = QuantumRegister(qubits)
        elif qubits is None:
            self.num_qubits = 0
            self.qreg = None
        else:
            raise TypeError("qubits must be an int or QuantumRegister")

        # Handle classical register
        if isinstance(classical_bits, ClassicalRegister):
            self.num_clbits = classical_bits.size
            self.creg = classical_bits
        elif isinstance(classical_bits, int):
            self.num_clbits = classical_bits
            self.creg = ClassicalRegister(classical_bits)
        elif classical_bits is None:
            self.num_clbits = 0
            self.creg = None
        else:
            raise TypeError("classical_bits must be an int or ClassicalRegister")

        # Store circuit instructions
        self.instructions: List[Instruction] = []

    def _validate_qubit_index(self, qubit: int) -> None:
        """Validate that a qubit index is within bounds."""
        if qubit < 0 or qubit >= self.num_qubits:
            raise IndexError(
                f"Qubit index {qubit} out of range for circuit with {self.num_qubits} qubits"
            )

    def _validate_classical_bit_index(self, bit: int) -> None:
        """Validate that a classical bit index is within bounds."""
        if bit < 0 or bit >= self.num_clbits:
            raise IndexError(
                f"Classical bit index {bit} out of range for circuit with {self.num_clbits} classical bits"
            )

    def _add_gate(self, gate: Gate, qubits: List[int]) -> None:
        """
        Add a gate to the circuit.

        Args:
            gate: The gate to add
            qubits: List of qubit indices the gate acts on
        """
        for qubit in qubits:
            self._validate_qubit_index(qubit)

        if len(qubits) != gate.num_qubits:
            raise ValueError(
                f"Gate {gate.name} requires {gate.num_qubits} qubits, got {len(qubits)}"
            )

        self.instructions.append(Instruction(gate, qubits))

    # Single-qubit gates
    def x(self, qubit: int) -> "QuantumCircuit":
        """Apply X (NOT) gate to a qubit."""
        self._add_gate(XGate(), [qubit])
        return self

    def y(self, qubit: int) -> "QuantumCircuit":
        """Apply Y gate to a qubit."""
        self._add_gate(YGate(), [qubit])
        return self

    def z(self, qubit: int) -> "QuantumCircuit":
        """Apply Z gate to a qubit."""
        self._add_gate(ZGate(), [qubit])
        return self

    def h(self, qubit: int) -> "QuantumCircuit":
        """Apply Hadamard gate to a qubit."""
        self._add_gate(HGate(), [qubit])
        return self

    def s(self, qubit: int) -> "QuantumCircuit":
        """Apply S gate to a qubit."""
        self._add_gate(SGate(), [qubit])
        return self

    def sdg(self, qubit: int) -> "QuantumCircuit":
        """Apply S dagger gate to a qubit."""
        self._add_gate(SdgGate(), [qubit])
        return self

    def t(self, qubit: int) -> "QuantumCircuit":
        """Apply T gate to a qubit."""
        self._add_gate(TGate(), [qubit])
        return self

    def tdg(self, qubit: int) -> "QuantumCircuit":
        """Apply T dagger gate to a qubit."""
        self._add_gate(TdgGate(), [qubit])
        return self

    def i(self, qubit: int) -> "QuantumCircuit":
        """Apply identity gate to a qubit."""
        self._add_gate(IGate(), [qubit])
        return self

    def rx(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply RX rotation gate to a qubit."""
        self._add_gate(RXGate(theta), [qubit])
        return self

    def ry(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply RY rotation gate to a qubit."""
        self._add_gate(RYGate(theta), [qubit])
        return self

    def rz(self, theta: float, qubit: int) -> "QuantumCircuit":
        """Apply RZ rotation gate to a qubit."""
        self._add_gate(RZGate(theta), [qubit])
        return self

    def u3(
        self, theta: float, phi: float, lambda_: float, qubit: int
    ) -> "QuantumCircuit":
        """Apply U3 gate to a qubit."""
        self._add_gate(U3Gate(theta, phi, lambda_), [qubit])
        return self

    # Two-qubit gates
    def cx(self, control: int, target: int) -> "QuantumCircuit":
        """Apply CNOT/CX gate."""
        self._add_gate(CXGate(), [control, target])
        return self

    def cnot(self, control: int, target: int) -> "QuantumCircuit":
        """Apply CNOT gate (alias for cx)."""
        return self.cx(control, target)

    def cz(self, control: int, target: int) -> "QuantumCircuit":
        """Apply CZ gate."""
        self._add_gate(CZGate(), [control, target])
        return self

    def cy(self, control: int, target: int) -> "QuantumCircuit":
        """Apply CY gate."""
        self._add_gate(CYGate(), [control, target])
        return self

    def swap(self, qubit1: int, qubit2: int) -> "QuantumCircuit":
        """Apply SWAP gate."""
        self._add_gate(SWAPGate(), [qubit1, qubit2])
        return self

    # Three-qubit gates
    def ccx(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        """Apply Toffoli/CCX gate."""
        self._add_gate(CCXGate(), [control1, control2, target])
        return self

    def toffoli(self, control1: int, control2: int, target: int) -> "QuantumCircuit":
        """Apply Toffoli gate (alias for ccx)."""
        return self.ccx(control1, control2, target)

    def cswap(self, control: int, target1: int, target2: int) -> "QuantumCircuit":
        """Apply Fredkin/CSWAP gate."""
        self._add_gate(CSWAPGate(), [control, target1, target2])
        return self

    def fredkin(self, control: int, target1: int, target2: int) -> "QuantumCircuit":
        """Apply Fredkin gate (alias for cswap)."""
        return self.cswap(control, target1, target2)

    def measure(self, qubit: int, classical_bit: int) -> "QuantumCircuit":
        """
        Measure a qubit and store the result in a classical bit.

        Args:
            qubit: Index of the qubit to measure
            classical_bit: Index of the classical bit to store the result
        """
        self._validate_qubit_index(qubit)
        self._validate_classical_bit_index(classical_bit)

        # Create a special "measurement" gate
        measure_gate = Gate("measure", 1, np.eye(2, dtype=complex))
        instruction = Instruction(measure_gate, [qubit], [classical_bit])
        self.instructions.append(instruction)
        return self

    def measure_all(self) -> "QuantumCircuit":
        """Measure all qubits to corresponding classical bits."""
        if self.num_clbits < self.num_qubits:
            raise ValueError(
                f"Not enough classical bits ({self.num_clbits}) to measure all qubits ({self.num_qubits})"
            )

        for i in range(self.num_qubits):
            self.measure(i, i)
        return self

    def depth(self) -> int:
        """Calculate the depth of the circuit (number of time steps)."""
        if not self.instructions:
            return 0

        # Track when each qubit is last used
        qubit_times = [0] * self.num_qubits
        max_time = 0

        for instruction in self.instructions:
            # Find the latest time among qubits this gate acts on
            latest_time = max(qubit_times[q] for q in instruction.qubits)
            # Update all qubits to the next time step
            for q in instruction.qubits:
                qubit_times[q] = latest_time + 1
            max_time = max(max_time, latest_time + 1)

        return max_time

    def size(self) -> int:
        """Return the number of gates in the circuit."""
        return len(self.instructions)

    def count_ops(self) -> dict:
        """Count the number of each type of operation."""
        counts = {}
        for instruction in self.instructions:
            gate_name = instruction.gate.name
            counts[gate_name] = counts.get(gate_name, 0) + 1
        return counts

    def draw(self, output: str = "text") -> Optional[str]:
        """
        Draw the quantum circuit.

        Args:
            output: Output format ('text' or 'unicode')

        Returns:
            String representation of the circuit if output is 'text', None otherwise
        """
        if output != "text":
            raise NotImplementedError(f"Output format '{output}' not yet supported")

        if not self.instructions:
            return "Empty circuit"

        # Build a simple text representation
        lines = []
        lines.append(f"Quantum Circuit with {self.num_qubits} qubits")
        if self.num_clbits > 0:
            lines.append(f"Classical bits: {self.num_clbits}")
        lines.append("-" * 50)

        for i, instruction in enumerate(self.instructions):
            qubits_str = ", ".join(f"q{q}" for q in instruction.qubits)
            if instruction.classical_bits:
                cbits_str = ", ".join(f"c{c}" for c in instruction.classical_bits)
                lines.append(
                    f"{i:3d}: {instruction.gate.name:8s} {qubits_str} -> {cbits_str}"
                )
            else:
                lines.append(f"{i:3d}: {instruction.gate.name:8s} {qubits_str}")

        lines.append("-" * 50)
        lines.append(f"Total gates: {self.size()}, Depth: {self.depth()}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<QuantumCircuit({self.num_qubits} qubits, {self.num_clbits} classical bits, {self.size()} gates)>"

    def __str__(self) -> str:
        return self.draw() or ""
