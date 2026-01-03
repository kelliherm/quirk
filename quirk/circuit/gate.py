"""
Quantum gates implementation with matrix representations.
"""

from typing import List, Optional

import numpy as np


class Gate:
    """Base class for quantum gates."""

    def __init__(
        self,
        name: str,
        num_qubits: int,
        matrix: np.ndarray,
        params: Optional[List[float]] = None,
    ):
        """
        Initialize a quantum gate.

        Args:
            name: Name of the gate
            num_qubits: Number of qubits the gate acts on
            matrix: Unitary matrix representation of the gate
            params: Optional parameters for parameterized gates
        """
        self.name = name
        self.num_qubits = num_qubits
        self.matrix = matrix
        self.params = params or []

    def __repr__(self) -> str:
        if self.params:
            params_str = ", ".join(f"{p:.4f}" for p in self.params)
            return f"{self.name}({params_str})"
        return self.name

    def to_matrix(self) -> np.ndarray:
        """Return the matrix representation of the gate."""
        return self.matrix


# Single-qubit gates
class XGate(Gate):
    """Pauli-X gate (NOT gate)."""

    def __init__(self):
        matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        super().__init__("X", 1, matrix)


class YGate(Gate):
    """Pauli-Y gate."""

    def __init__(self):
        matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        super().__init__("Y", 1, matrix)


class ZGate(Gate):
    """Pauli-Z gate."""

    def __init__(self):
        matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        super().__init__("Z", 1, matrix)


class HGate(Gate):
    """Hadamard gate."""

    def __init__(self):
        matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        super().__init__("H", 1, matrix)


class SGate(Gate):
    """S gate (phase gate)."""

    def __init__(self):
        matrix = np.array([[1, 0], [0, 1j]], dtype=complex)
        super().__init__("S", 1, matrix)


class SdgGate(Gate):
    """S dagger gate (inverse S gate)."""

    def __init__(self):
        matrix = np.array([[1, 0], [0, -1j]], dtype=complex)
        super().__init__("Sdg", 1, matrix)


class TGate(Gate):
    """T gate (π/8 gate)."""

    def __init__(self):
        matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        super().__init__("T", 1, matrix)


class TdgGate(Gate):
    """T dagger gate (inverse T gate)."""

    def __init__(self):
        matrix = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
        super().__init__("Tdg", 1, matrix)


class RXGate(Gate):
    """Rotation around X-axis."""

    def __init__(self, theta: float):
        matrix = np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )
        super().__init__("RX", 1, matrix, [theta])


class RYGate(Gate):
    """Rotation around Y-axis."""

    def __init__(self, theta: float):
        matrix = np.array(
            [
                [np.cos(theta / 2), -np.sin(theta / 2)],
                [np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )
        super().__init__("RY", 1, matrix, [theta])


class RZGate(Gate):
    """Rotation around Z-axis."""

    def __init__(self, theta: float):
        matrix = np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
        )
        super().__init__("RZ", 1, matrix, [theta])


class U3Gate(Gate):
    """Generic single-qubit rotation gate with 3 Euler angles."""

    def __init__(self, theta: float, phi: float, lambda_: float):
        matrix = np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lambda_)) * np.cos(theta / 2),
                ],
            ],
            dtype=complex,
        )
        super().__init__("U3", 1, matrix, [theta, phi, lambda_])


class IGate(Gate):
    """Identity gate."""

    def __init__(self):
        matrix = np.array([[1, 0], [0, 1]], dtype=complex)
        super().__init__("I", 1, matrix)


# Two-qubit gates
class CNOTGate(Gate):
    """Controlled-NOT gate (CX gate)."""

    def __init__(self):
        matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        super().__init__("CNOT", 2, matrix)


class CXGate(CNOTGate):
    """Alias for CNOT gate."""

    def __init__(self):
        super().__init__()
        self.name = "CX"


class CZGate(Gate):
    """Controlled-Z gate."""

    def __init__(self):
        matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        )
        super().__init__("CZ", 2, matrix)


class CYGate(Gate):
    """Controlled-Y gate."""

    def __init__(self):
        matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex
        )
        super().__init__("CY", 2, matrix)


class SWAPGate(Gate):
    """SWAP gate."""

    def __init__(self):
        matrix = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        super().__init__("SWAP", 2, matrix)


# Three-qubit gates
class ToffoliGate(Gate):
    """Toffoli gate (CCX gate, CCNOT gate)."""

    def __init__(self):
        matrix = np.eye(8, dtype=complex)
        matrix[6, 6] = 0
        matrix[7, 7] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        super().__init__("Toffoli", 3, matrix)


class CCXGate(ToffoliGate):
    """Alias for Toffoli gate."""

    def __init__(self):
        super().__init__()
        self.name = "CCX"


class FredkinGate(Gate):
    """Fredkin gate (CSWAP gate)."""

    def __init__(self):
        matrix = np.eye(8, dtype=complex)
        matrix[5, 5] = 0
        matrix[6, 6] = 0
        matrix[5, 6] = 1
        matrix[6, 5] = 1
        super().__init__("Fredkin", 3, matrix)


class CSWAPGate(FredkinGate):
    """Alias for Fredkin gate."""

    def __init__(self):
        super().__init__()
        self.name = "CSWAP"
