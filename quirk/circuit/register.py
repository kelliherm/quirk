"""
Quantum and classical register implementations.
"""

from typing import List


class Register:
    """Base class for registers."""

    def __init__(self, size: int = 1, name: str = ""):
        """
        Initialize a register.

        Args:
            size: Number of bits/qubits in the register
            name: Optional name for the register
        """
        if size < 1:
            raise ValueError("Register size must be at least 1")
        self.size = size
        self.name = name
        self._bits: List[int] = list(range(size))

    def __len__(self) -> int:
        """Return the size of the register."""
        return self.size

    def __getitem__(self, key: int) -> int:
        """Get a specific bit/qubit from the register."""
        if isinstance(key, int):
            if key < 0 or key >= self.size:
                raise IndexError(
                    f"Index {key} out of range for register of size {self.size}"
                )
            return self._bits[key]
        raise TypeError("Register indices must be integers")

    def __repr__(self) -> str:
        if self.name:
            return f"{self.__class__.__name__}({self.size}, '{self.name}')"
        return f"{self.__class__.__name__}({self.size})"


class QuantumRegister(Register):
    """Register for quantum bits (qubits)."""

    def __init__(self, size: int = 1, name: str = "q"):
        """
        Initialize a quantum register.

        Args:
            size: Number of qubits in the register
            name: Optional name for the register (default: 'q')
        """
        super().__init__(size, name)


class ClassicalRegister(Register):
    """Register for classical bits."""

    def __init__(self, size: int = 1, name: str = "c"):
        """
        Initialize a classical register.

        Args:
            size: Number of classical bits in the register
            name: Optional name for the register (default: 'c')
        """
        super().__init__(size, name)
