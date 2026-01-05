"""
Statevector implementation for representing and manipulating quantum states.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class Statevector:
    """Represents a quantum statevector."""

    def __init__(self, data: Union[np.ndarray, List[complex], "Statevector"]):
        """
        Initialize a statevector.

        Args:
            data: Initial state as numpy array, list, or another Statevector
        """
        if isinstance(data, Statevector):
            self._data = data._data.copy()
        elif isinstance(data, (list, tuple)):
            self._data = np.array(data, dtype=complex)
        elif isinstance(data, np.ndarray):
            self._data = data.astype(complex)
        else:
            raise TypeError("Statevector data must be array-like or Statevector")

        # Validate that the dimension is a power of 2
        dim = len(self._data)
        if dim == 0 or (dim & (dim - 1)) != 0:
            raise ValueError(f"Statevector dimension must be a power of 2, got {dim}")

        # Normalize the statevector
        norm = np.linalg.norm(self._data)
        if norm == 0:
            raise ValueError("Statevector cannot have zero norm")
        self._data = self._data / norm

        self.num_qubits = int(np.log2(dim))

    @classmethod
    def from_label(cls, label: str) -> "Statevector":
        """
        Create a statevector from a computational basis state label.

        Args:
            label: Binary string like '001', '101', etc.

        Returns:
            Statevector in the specified computational basis state
        """
        num_qubits = len(label)
        dim = 2**num_qubits

        # Convert binary label to integer
        if not all(c in "01" for c in label):
            raise ValueError(f"Label must be a binary string, got '{label}'")

        index = int(label, 2)
        data = np.zeros(dim, dtype=complex)
        data[index] = 1.0

        return cls(data)

    @classmethod
    def from_int(cls, i: int, num_qubits: int) -> "Statevector":
        """
        Create a statevector from an integer computational basis state.

        Args:
            i: Integer representing the basis state
            num_qubits: Number of qubits

        Returns:
            Statevector in the specified computational basis state
        """
        if i < 0 or i >= 2**num_qubits:
            raise ValueError(f"Integer {i} out of range for {num_qubits} qubits")

        dim = 2**num_qubits
        data = np.zeros(dim, dtype=complex)
        data[i] = 1.0

        return cls(data)

    def data(self) -> np.ndarray:
        """Return the raw statevector data."""
        return self._data.copy()

    def dim(self) -> int:
        """Return the dimension of the statevector."""
        return len(self._data)

    def is_valid(self, atol: float = 1e-8) -> bool:
        """
        Check if the statevector is properly normalized.

        Args:
            atol: Absolute tolerance for normalization check

        Returns:
            True if the statevector is normalized
        """
        norm_squared = np.sum(np.abs(self._data) ** 2)
        return np.abs(norm_squared - 1.0) < atol

    def probabilities(self) -> np.ndarray:
        """
        Calculate measurement probabilities for all computational basis states.

        Returns:
            Array of probabilities
        """
        return np.abs(self._data) ** 2

    def probabilities_dict(self, decimals: int = 10) -> Dict[str, float]:
        """
        Get measurement probabilities as a dictionary.

        Args:
            decimals: Number of decimal places to round probabilities

        Returns:
            Dictionary mapping basis states to probabilities
        """
        probs = self.probabilities()
        result = {}

        for i, prob in enumerate(probs):
            if prob > 1e-10:  # Only include non-negligible probabilities
                label = format(i, f"0{self.num_qubits}b")
                result[label] = round(float(prob), decimals)

        return result

    def sample_counts(
        self, shots: int = 1024, seed: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Sample measurement outcomes from the statevector.

        Args:
            shots: Number of measurement samples
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping basis states to counts
        """
        if seed is not None:
            np.random.seed(seed)

        probs = self.probabilities()
        indices = np.random.choice(len(probs), size=shots, p=probs)

        counts = {}
        for idx in indices:
            label = format(idx, f"0{self.num_qubits}b")
            counts[label] = counts.get(label, 0) + 1

        return dict(sorted(counts.items()))

    def measure(self, seed: Optional[int] = None) -> Tuple[str, int]:
        """
        Perform a single measurement and return the outcome.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (binary_label, integer_value)
        """
        if seed is not None:
            np.random.seed(seed)

        probs = self.probabilities()
        outcome = np.random.choice(len(probs), p=probs)
        label = format(outcome, f"0{self.num_qubits}b")

        return label, outcome

    def expectation_value(self, observable: np.ndarray) -> complex:
        """
        Calculate the expectation value of an observable.

        Args:
            observable: Hermitian matrix representing the observable

        Returns:
            Expectation value <ψ|O|ψ>
        """
        if observable.shape != (self.dim(), self.dim()):
            raise ValueError(
                f"Observable dimension {observable.shape} doesn't match statevector dimension {self.dim()}"
            )

        return complex(np.vdot(self._data, observable @ self._data))

    def evolve(self, operator: np.ndarray) -> "Statevector":
        """
        Evolve the statevector by applying a unitary operator.

        Args:
            operator: Unitary matrix to apply

        Returns:
            New evolved statevector
        """
        if operator.shape != (self.dim(), self.dim()):
            raise ValueError(
                f"Operator dimension {operator.shape} doesn't match statevector dimension {self.dim()}"
            )

        new_data = operator @ self._data
        return Statevector(new_data)

    def inner(self, other: "Statevector") -> complex:
        """
        Calculate the inner product with another statevector.

        Args:
            other: Another statevector

        Returns:
            Inner product <self|other>
        """
        if self.dim() != other.dim():
            raise ValueError(
                f"Statevector dimensions don't match: {self.dim()} vs {other.dim()}"
            )

        return complex(np.vdot(self._data, other._data))

    def purity(self) -> float:
        """
        Calculate the purity of the state (always 1 for pure states).

        Returns:
            Purity value
        """
        return float(np.abs(np.vdot(self._data, self._data)) ** 2)

    def __repr__(self) -> str:
        return f"Statevector({self._data}, dims=(2,)*{self.num_qubits})"

    def __str__(self) -> str:
        lines = [f"Statevector with {self.num_qubits} qubit(s):"]
        probs_dict = self.probabilities_dict(decimals=6)

        for label, prob in probs_dict.items():
            amplitude = self._data[int(label, 2)]
            real = amplitude.real
            imag = amplitude.imag

            # Format amplitude
            if abs(imag) < 1e-10:
                amp_str = f"{real:+.6f}"
            elif abs(real) < 1e-10:
                amp_str = f"{imag:+.6f}j"
            else:
                amp_str = f"{real:+.6f}{imag:+.6f}j"

            lines.append(f"  |{label}>: {amp_str} (prob: {prob:.6f})")

        return "\n".join(lines)

    def __getitem__(self, key: int) -> complex:
        """Get amplitude at a specific index."""
        return self._data[key]

    def __len__(self) -> int:
        """Return the dimension of the statevector."""
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        """Check equality with another statevector."""
        if not isinstance(other, Statevector):
            return False
        return np.allclose(self._data, other._data)
