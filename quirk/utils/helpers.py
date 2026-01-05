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


def create_w_state(qc: QuantumCircuit, qubits: List[int]) -> QuantumCircuit:
    """
    Create a W state across multiple qubits.
    Note: This is a simplified implementation for 3 qubits.

    Args:
        qc: Quantum circuit
        qubits: List of qubit indices (must be 3)

    Returns:
        The quantum circuit with W state gates applied
    """
    if len(qubits) != 3:
        raise NotImplementedError(
            "W state implementation currently supports only 3 qubits"
        )

    # W state for 3 qubits: (|001⟩ + |010⟩ + |100⟩)/√3
    # Simplified circuit approximation
    theta1 = 2 * np.arcsin(np.sqrt(1 / 3))
    theta2 = 2 * np.arcsin(np.sqrt(1 / 2))

    qc.ry(theta1, qubits[0])
    qc.cx(qubits[0], qubits[1])
    qc.x(qubits[0])
    qc.ry(theta2, qubits[1])
    qc.cx(qubits[1], qubits[2])

    return qc


def qft(qc: QuantumCircuit, qubits: List[int], inverse: bool = False) -> QuantumCircuit:
    """
    Apply Quantum Fourier Transform to specified qubits.

    Args:
        qc: Quantum circuit
        qubits: List of qubit indices
        inverse: If True, apply inverse QFT

    Returns:
        The quantum circuit with QFT gates applied
    """
    n = len(qubits)

    if not inverse:
        # Forward QFT
        for i in range(n):
            qc.h(qubits[i])
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                qc.rz(angle, qubits[j])

        # Swap qubits to get correct order
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
    else:
        # Inverse QFT
        # Swap qubits first
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])

        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                angle = -np.pi / (2 ** (j - i))
                qc.rz(angle, qubits[j])
            qc.h(qubits[i])

    return qc


def phase_estimation(
    qc: QuantumCircuit,
    counting_qubits: List[int],
    target_qubit: int,
    num_iterations: int,
) -> QuantumCircuit:
    """
    Apply quantum phase estimation algorithm structure.

    Args:
        qc: Quantum circuit
        counting_qubits: Qubits used for phase counting
        target_qubit: Target qubit for the unitary
        num_iterations: Number of controlled unitary applications

    Returns:
        The quantum circuit with phase estimation gates applied
    """
    n = len(counting_qubits)

    # Initialize counting qubits to superposition
    for qubit in counting_qubits:
        qc.h(qubit)

    # Apply controlled unitaries (simplified - using Z rotations as example)
    for i, qubit in enumerate(counting_qubits):
        repetitions = 2 ** (n - 1 - i)
        for _ in range(repetitions):
            qc.cz(qubit, target_qubit)

    # Apply inverse QFT
    qft(qc, counting_qubits, inverse=True)

    return qc


def amplitude_amplification(
    qc: QuantumCircuit, qubits: List[int], oracle_qubits: Optional[List[int]] = None
) -> QuantumCircuit:
    """
    Apply Grover's diffusion operator (amplitude amplification).

    Args:
        qc: Quantum circuit
        qubits: List of qubit indices
        oracle_qubits: Specific qubits for the oracle (if None, uses all qubits)

    Returns:
        The quantum circuit with diffusion operator applied
    """
    if oracle_qubits is None:
        oracle_qubits = qubits

    # Apply H gates
    for qubit in qubits:
        qc.h(qubit)

    # Apply X gates
    for qubit in qubits:
        qc.x(qubit)

    # Multi-controlled Z (simplified for 2 qubits)
    if len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    elif len(qubits) == 3:
        # Use Toffoli-like structure
        qc.ccx(qubits[0], qubits[1], qubits[2])

    # Apply X gates
    for qubit in qubits:
        qc.x(qubit)

    # Apply H gates
    for qubit in qubits:
        qc.h(qubit)

    return qc


def barrier(qc: QuantumCircuit, qubits: Optional[List[int]] = None) -> QuantumCircuit:
    """
    Add a visual barrier to the circuit (currently just returns the circuit).
    This is a placeholder for future visualization improvements.

    Args:
        qc: Quantum circuit
        qubits: Optional list of qubits (if None, applies to all)

    Returns:
        The quantum circuit (unchanged)
    """
    # In a full implementation, this would add a barrier instruction
    # For now, it's a no-op that maintains API compatibility
    return qc


def reset_qubit(qc: QuantumCircuit, qubit: int) -> QuantumCircuit:
    """
    Reset a qubit to |0⟩ state (measurement-based reset simulation).
    Note: This is a simplified implementation using measurement and conditional X.

    Args:
        qc: Quantum circuit
        qubit: Qubit index to reset

    Returns:
        The quantum circuit with reset operation
    """
    # In a real quantum computer, this would measure and conditionally flip
    # For simulation, we can't truly reset without measurements
    # This is a placeholder that would need simulator support
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
    from quirk.simulation.simulator import Simulator

    # Check for measurements
    for instruction in qc.instructions:
        if instruction.gate.name == "measure":
            raise ValueError("Cannot calculate unitary for circuit with measurements")

    # Build unitary by multiplying gate matrices
    dim = 2**qc.num_qubits
    unitary = np.eye(dim, dtype=complex)

    simulator = Simulator()
    for instruction in qc.instructions:
        gate_unitary = simulator._build_gate_unitary(instruction, qc.num_qubits)
        unitary = gate_unitary @ unitary

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
    if seed is not None:
        np.random.seed(seed)

    gates = ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
    two_qubit_gates = ["cx", "cz", "swap"]

    qc = QuantumCircuit(num_qubits, num_qubits if measure else 0)

    for _ in range(depth):
        # Add single-qubit gates
        for qubit in range(num_qubits):
            gate = np.random.choice(gates)
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
                qc.rx(np.random.uniform(0, 2 * np.pi), qubit)
            elif gate == "ry":
                qc.ry(np.random.uniform(0, 2 * np.pi), qubit)
            elif gate == "rz":
                qc.rz(np.random.uniform(0, 2 * np.pi), qubit)

        # Add two-qubit gate if we have multiple qubits
        if num_qubits > 1:
            gate = np.random.choice(two_qubit_gates)
            q1, q2 = np.random.choice(num_qubits, size=2, replace=False)

            if gate == "cx":
                qc.cx(q1, q2)
            elif gate == "cz":
                qc.cz(q1, q2)
            elif gate == "swap":
                qc.swap(q1, q2)

    if measure:
        qc.measure_all()

    return qc
