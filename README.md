# Quirk

Quirk is an open-source software development kit for building, simulating, and executing quantum circuits. It provides an interface similar to Qiskit for creating quantum algorithms, applying quantum gates, simulating statevectors, and measuring qubits.

> [!NOTE]
> The current development of Quirk is focused on shifting the circuit representation to a DAG based data structure. Due to dependency differences between the `rustworkx` crate and the current project, development is being done in the `dag` branch.

## Installation

To install Quirk, begin by cloning the repository to the local directory.

```bash
git clone https://github.com/kelliherm/quirk.git
cd quirk
```

Then install the local dependencies with `pip` or your preferred Python package manager.

```bash
pip install -e .
```

## Getting Started

The following examples walks you through the process of creating and simulating a simple quantum circuit, known as the Bell state.

```py
from quirk import QuantumCircuit, Simulator

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Apply Hadamard gate
qc.h(0)

# Apply CNOT gate
qc.cx(0, 1)

# Measure both qubits
qc.measure(0, 0)
qc.measure(1, 1)

# Simulate the circuit
simulator = Simulator()
result = simulator.run(qc, shots=1000)

# View results
print(result)
print(result.get_counts())
```

## Gate Library

Quirk includes a large standard gate library, which consists of various one, two, and three-qubit gates.

### One-Qubit Gates

| Gate | Method                             | Description                   |
| ---- | ---------------------------------- | ----------------------------- |
| X    | `qc.x(qubit)`                      | Pauli-X (NOT) gate            |
| Y    | `qc.y(qubit)`                      | Pauli-Y gate                  |
| Z    | `qc.z(qubit)`                      | Pauli-Z gate                  |
| H    | `qc.h(qubit)`                      | Hadamard gate                 |
| S    | `qc.s(qubit)`                      | S gate (Phase gate)           |
| S†   | `qc.sdg(qubit)`                    | S dagger gate                 |
| T    | `qc.t(qubit)`                      | T gate (π/8 gate)             |
| T†   | `qc.tdg(qubit)`                    | T dagger gate                 |
| I    | `qc.i(qubit)`                      | Identity gate                 |
| RX   | `qc.rx(theta, qubit)`              | Rotation around X-axis        |
| RY   | `qc.ry(theta, qubit)`              | Rotation around Y-axis        |
| RZ   | `qc.rz(theta, qubit)`              | Rotation around Z-axis        |
| U3   | `qc.u3(theta, phi, lambda, qubit)` | Generic single-qubit rotation |

### Two-Qubit Gates

| Gate    | Method                    | Description         |
| ------- | ------------------------- | ------------------- |
| CNOT/CX | `qc.cx(control, target)`  | Controlled-NOT gate |
| CY      | `qc.cy(control, target)`  | Controlled-Y gate   |
| CZ      | `qc.cz(control, target)`  | Controlled-Z gate   |
| SWAP    | `qc.swap(qubit1, qubit2)` | SWAP gate           |

### Three-Qubit Gates

| Gate          | Method                             | Description                    |
| ------------- | ---------------------------------- | ------------------------------ |
| Toffoli/CCX   | `qc.ccx(ctrl1, ctrl2, target)`     | Toffoli gate (CCNOT)           |
| Fredkin/CSWAP | `qc.cswap(ctrl, target1, target2)` | Fredkin gate (Controlled-SWAP) |

## Project Structure

```
quirk/
├── quirk/
│   ├── __init__.py
│   ├── circuit/
│   │   ├── __init__.py
│   │   ├── gate.py           # Gate implementations
│   │   ├── instruction.py    # Instruction class
│   │   ├── quantumcircuit.py # Circuit builder
│   │   └── register.py       # Register classes
│   └── simulation/
│   │   ├── __init__.py
│   │   ├── simulator.py      # Circuit simulator
│   │   └── statevector.py    # Statevector class
│   └── utils/
│       ├── __init__.py
│       └── helpers.py        # Helper functions
├── README.md
└── pyproject.toml
```

## License

This project is open source and available under the MIT License. More details can be found in the [license](LICENSE) file.

## Acknowledgments

Quirk is inspired by IBM's Qiskit and aims to provide an educational and accessible quantum computing framework.
