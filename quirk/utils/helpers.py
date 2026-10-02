"""
Utility functions and helpers for quantum circuit construction and analysis.
"""

from typing import List, Optional

import numpy as np

from quirk.circuit.quantumcircuit import QuantumCircuit


def create_bell_pair(qc: QuantumCircuit, qubit1: int, qubit2: int) -> QuantumCircuit:
    """
    Create a Bell pair (maximally entangled state) between two qubits.

    Args:
        qc: Quantum circuit
        qubit1: First qubit index
        qubit2: Second qubit index

    Returns:
        The quantum circuit with Bell pair gates applied
    """
    qc.h(qubit1)
    qc.cx(qubit1, qubit2)
    return qc


def create_ghz_state(qc: QuantumCircuit, qubits: List[int]) -> QuantumCircuit:
    """
    Create a GHZ state (generalized Bell state) across multiple qubits.

    Args:
        qc: Quantum circuit
        qubits: List of qubit indices

    Returns:
        The quantum circuit with GHZ state gates applied
    """
    if len(qubits) < 2:
        raise ValueError("GHZ state requires at least 2 qubits")

    qc.h(qubits[0])
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])

    return qc








def initialize_state(
    qc: QuantumCircuit, state: str, qubits: Optional[List[int]] = None
) -> QuantumCircuit:
    """
    Initialize qubits to a computational basis state.

    Args:
        qc: Quantum circuit
        state: Binary string representing the state (e.g., "101")
        qubits: Optional list of qubit indices (if None, uses first len(state) qubits)

    Returns:
        The quantum circuit with initialization gates applied
    """
    if qubits is None:
        qubits = list(range(len(state)))

    if len(state) != len(qubits):
        raise ValueError(
            f"State length {len(state)} doesn't match number of qubits {len(qubits)}"
        )

    for i, bit in enumerate(state):
        if bit == "1":
            qc.x(qubits[i])
        elif bit != "0":
            raise ValueError(
                f"Invalid bit '{bit}' in state string. Use only '0' and '1'"
            )

    return qc


def pauli_string(qc: QuantumCircuit, paulis: str, qubits: List[int]) -> QuantumCircuit:
    """
    Apply a string of Pauli operators to qubits.

    Args:
        qc: Quantum circuit
        paulis: String of Pauli operators (e.g., "XYZ", "IXX")
        qubits: List of qubit indices

    Returns:
        The quantum circuit with Pauli gates applied
    """
    if len(paulis) != len(qubits):
        raise ValueError(
            f"Pauli string length {len(paulis)} doesn't match number of qubits {len(qubits)}"
        )

    for pauli, qubit in zip(paulis, qubits):
        if pauli == "X" or pauli == "x":
            qc.x(qubit)
        elif pauli == "Y" or pauli == "y":
            qc.y(qubit)
        elif pauli == "Z" or pauli == "z":
            qc.z(qubit)
        elif pauli == "I" or pauli == "i":
            qc.i(qubit)
        else:
            raise ValueError(f"Invalid Pauli operator '{pauli}'. Use X, Y, Z, or I")

    return qc


def controlled_gate(
    qc: QuantumCircuit, gate_name: str, control: int, target: int, **kwargs
) -> QuantumCircuit:
    """
    Apply a controlled version of a gate.

    Args:
        qc: Quantum circuit
        gate_name: Name of the gate ('x', 'y', 'z', etc.)
        control: Control qubit index
        target: Target qubit index
        **kwargs: Additional parameters for the gate

    Returns:
        The quantum circuit with controlled gate applied
    """
    gate_name = gate_name.lower()

    if gate_name == "x":
        qc.cx(control, target)
    elif gate_name == "y":
        qc.cy(control, target)
    elif gate_name == "z":
        qc.cz(control, target)
    else:
        raise NotImplementedError(f"Controlled {gate_name} gate not yet implemented")

    return qc


def calculate_unitary(qc: QuantumCircuit) -> np.ndarray:
    """
    Calculate the unitary matrix representation of a quantum circuit.
    Note: This only works for circuits without measurements.

    Args:
        qc: Quantum circuit

    Returns:
        Unitary matrix representing the circuit
    """
    from quirk.simulation.simulator import apply_gate
    dim = 2 ** qc.num_qubits
    unitary = np.eye(dim, dtype=complex)
    for _, node in qc.get_gate_nodes():
        if node["type"] == "measurement":
            raise ValueError("Cannot calculate unitary for circuit with measurements")
        unitary = np.column_stack([
            apply_gate(unitary[:, i], node["gate"].to_matrix(), node["qubits"], qc.num_qubits)
            for i in range(dim)
        ])
    return unitary


def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate the fidelity between two quantum states.

    Args:
        state1: First state vector
        state2: Second state vector

    Returns:
        Fidelity value between 0 and 1
    """
    overlap = np.abs(np.vdot(state1, state2))
    return overlap**2


def trace_distance(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate the trace distance between two pure states.

    Args:
        state1: First state vector
        state2: Second state vector

    Returns:
        Trace distance value between 0 and 1
    """
    fid = fidelity(state1, state2)
    return np.sqrt(1 - fid)


def random_circuit(
    num_qubits: int, depth: int, measure: bool = False, seed: Optional[int] = None
) -> QuantumCircuit:
    """
    Generate a random quantum circuit.

    Args:
        num_qubits: Number of qubits
        depth: Circuit depth (number of layers)
        measure: Whether to add measurements at the end
        seed: Random seed for reproducibility

    Returns:
        A random quantum circuit
    """
    from numbers import Integral
    if isinstance(depth, bool) or not isinstance(depth, Integral):
        raise TypeError("depth must be an integer")
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    rng = np.random.default_rng(seed)

    gates = ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
    two_qubit_gates = ["cx", "cz", "swap"]

    qc = QuantumCircuit(num_qubits)

    for _ in range(depth):
        # Add single-qubit gates
        for qubit in range(num_qubits):
            gate = rng.choice(gates)
            if gate == "h":
                qc.h(qubit)
            elif gate == "x":
                qc.x(qubit)
            elif gate == "y":
                qc.y(qubit)
            elif gate == "z":
                qc.z(qubit)
            elif gate == "s":
                qc.s(qubit)
            elif gate == "t":
                qc.t(qubit)
            elif gate == "rx":
                qc.rx(rng.uniform(0, 2 * np.pi), qubit)
            elif gate == "ry":
                qc.ry(rng.uniform(0, 2 * np.pi), qubit)
            elif gate == "rz":
                qc.rz(rng.uniform(0, 2 * np.pi), qubit)

        # Add two-qubit gate if we have multiple qubits
        if num_qubits > 1:
            gate = rng.choice(two_qubit_gates)
            q1, q2 = rng.choice(num_qubits, size=2, replace=False)

            if gate == "cx":
                qc.cx(q1, q2)
            elif gate == "cz":
                qc.cz(q1, q2)
            elif gate == "swap":
                qc.swap(q1, q2)

    if measure:
        qc.measure_all()

    return qc
