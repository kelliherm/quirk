"""DAG-backed quantum circuit class."""

from collections import Counter
from copy import deepcopy
from numbers import Integral
from typing import List

import numpy as np
import rustworkx as rwx

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


class QuantumCircuit:
    """A circuit whose only operation storage is a wire-labelled multigraph.

    Qubit 0 is the most significant bit. Measurements are terminal on their
    wires. Graph access returns independent snapshots; edits go through append.
    """

    def __init__(self, qubits: int = 0):
        if isinstance(qubits, bool) or not isinstance(qubits, Integral):
            raise TypeError("qubits must be an integer")
        if qubits < 0:
            raise ValueError("qubits must be nonnegative")
        self._num_qubits = int(qubits)
        self._dag = rwx.PyDAG(multigraph=True, check_cycle=True)
        self._inputs = {}
        self._outputs = {}
        for q in range(self.num_qubits):
            self._inputs[q] = self._dag.add_node(dict(type="input", qubit=q, label=f"q_in[{q}]"))
            self._outputs[q] = self._dag.add_node(dict(type="output", qubit=q, label=f"q_out[{q}]"))
            self._dag.add_edge(self._inputs[q], self._outputs[q], dict(qubit=q))

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def dag(self):
        """Independent graph snapshot, including all input/output edges."""
        return deepcopy(self._dag)

    def get_dag(self):
        return self.dag

    @property
    def input_nodes(self):
        return dict(self._inputs)

    @property
    def output_nodes(self):
        return dict(self._outputs)

    def _validate_qubit_index(self, qubit):
        if isinstance(qubit, bool) or not isinstance(qubit, Integral):
            raise TypeError("qubit indices must be integers")
        if not 0 <= qubit < self.num_qubits:
            raise IndexError(f"Qubit {qubit} out of range for {self.num_qubits} qubits")

    def _append_node(self, data, qubits):
        tails = []
        for q in qubits:
            self._validate_qubit_index(q)
            tail = next(iter(self._dag.predecessor_indices(self._outputs[q])))
            if self._dag[tail]["type"] == "measurement":
                raise ValueError(f"Qubit {q} has already been measured")
            tails.append(tail)
        if len(set(qubits)) != len(qubits):
            raise ValueError("Operation qubits must be distinct")
        node = self._dag.add_node(data)
        for q, tail in zip(qubits, tails):
            self._dag.remove_edge(tail, self._outputs[q])
            self._dag.add_edge(tail, node, dict(qubit=int(q)))
            self._dag.add_edge(node, self._outputs[q], dict(qubit=int(q)))
        return self

    def append(self, gate: Gate, qubits):
        """Append a unitary gate on ordered, distinct qubit indices."""
        if not isinstance(gate, Gate):
            raise TypeError("gate must be a Gate")
        qubits = tuple(qubits)
        if len(qubits) != gate.num_qubits or not qubits:
            raise ValueError(f"Gate {gate.name} requires {gate.num_qubits} qubits")
        matrix = np.asarray(gate.to_matrix(), dtype=complex)
        dim = 2 ** len(qubits)
        if matrix.shape != (dim, dim) or not np.allclose(matrix.conj().T @ matrix, np.eye(dim)):
            raise ValueError("Gate matrix must be unitary with dimensions matching its qubits")
        return self._append_node(dict(type="gate", gate=deepcopy(gate), qubits=qubits, label=gate.name), qubits)

    def _add_gate(self, gate, qubits):
        return self.append(gate, qubits)

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

    def copy(self):
        """Return a fully independent circuit, including gate payloads."""
        return deepcopy(self)

    def remove_operation(self, node_index):
        """Remove an operation and reconnect each of its labelled wires."""
        if node_index not in self._dag.node_indices():
            raise IndexError("Unknown DAG node")
        if self._dag[node_index]["type"] not in ("gate", "measurement"):
            raise ValueError("Input and output nodes cannot be removed")
        incoming = {edge["qubit"]: src for src, _, edge in self._dag.in_edges(node_index)}
        outgoing = {edge["qubit"]: dst for _, dst, edge in self._dag.out_edges(node_index)}
        self._dag.remove_node(node_index)
        for q, src in incoming.items():
            self._dag.add_edge(src, outgoing[q], dict(qubit=q))
        return self

    def compose(self, other, qubits=None):
        """Append another circuit on an ordered wire mapping, atomically."""
        if not isinstance(other, QuantumCircuit):
            raise TypeError("other must be a QuantumCircuit")
        mapping = tuple(range(other.num_qubits)) if qubits is None else tuple(qubits)
        if len(mapping) != other.num_qubits or len(set(mapping)) != len(mapping):
            raise ValueError("Wire mapping must contain one distinct index per source qubit")
        for q in mapping:
            self._validate_qubit_index(q)
        candidate = self.copy()
        for _, node in other.get_gate_nodes():
            targets = tuple(mapping[q] for q in node["qubits"])
            if node["type"] == "measurement":
                candidate.measure(targets[0])
            else:
                candidate.append(node["gate"], targets)
        self._dag = candidate._dag
        return self

    def measure(self, qubit: int):
        """Append a terminal computational-basis readout of a qubit."""
        return self._append_node(dict(type="measurement", qubits=(qubit,), label=f"M[q{qubit}]"), (qubit,))

    def measure_all(self):
        """Measure all wires that do not already end in a measurement."""
        for q in range(self.num_qubits):
            tail = next(iter(self._dag.predecessor_indices(self._outputs[q])))
            if self._dag[tail]["type"] != "measurement":
                self.measure(q)
        return self

    def topological_sort(self):
        return self._ordered_indices()

    def _ordered_indices(self):
        # rustworkx topological_sort returns indices, including parallel nodes.
        return list(rwx.topological_sort(self._dag))

    def get_gate_nodes(self):
        """Operation snapshots in dependency order, including measurements."""
        return [(i, deepcopy(self._dag[i])) for i in self._ordered_indices()
                if self._dag[i]["type"] in ("gate", "measurement")]

    def dag_nodes(self):
        return deepcopy(list(self._dag.nodes()))

    def dag_edges(self):
        return deepcopy(list(self._dag.weighted_edge_list()))

    def dag_layers(self):
        levels = {}
        layers = []
        for i in self._ordered_indices():
            data = self._dag[i]
            level = max((levels[p] for p in self._dag.predecessor_indices(i)), default=0)
            if data["type"] in ("gate", "measurement"):
                while len(layers) <= level:
                    layers.append([])
                layers[level].append(i)
                level += 1
            levels[i] = level
        return layers

    def depth(self):
        return len(self.dag_layers())

    def size(self):
        return len(self.get_gate_nodes())

    def count_ops(self):
        return dict(Counter(data["gate"].name if data["type"] == "gate" else "measure"
                            for _, data in self.get_gate_nodes()))

    def draw(self, output="text"):
        if output != "text":
            raise NotImplementedError(f"Unsupported output format: {output}")
        lines = [f"Quantum Circuit with {self.num_qubits} qubits"]
        for layer, indices in enumerate(self.dag_layers()):
            for i in indices:
                data = self._dag[i]
                lines.append(f"{layer}: {data['label']} {data['qubits']}")
        lines.append(f"Operations: {self.size()}, Depth: {self.depth()}")
        return "\n".join(lines)

    def to_dot(self):
        """Graphviz source; does not require the Graphviz executable."""
        import graphviz
        dot = graphviz.Digraph(comment="Quantum Circuit DAG")
        dot.attr(rankdir="LR")
        for i in self._dag.node_indices():
            data = self._dag[i]
            dot.node(str(i), data["label"], shape="circle" if data["type"] in ("input", "output") else "box")
        for src, dst, data in self._dag.weighted_edge_list():
            dot.edge(str(src), str(dst), label=f"q[{data['qubit']}]")
        return dot.source

    def visualize_dag(self, filename="circuit_dag", view=False):
        """Render PNG; requires the Graphviz dot executable on PATH."""
        import graphviz
        return graphviz.Source(self.to_dot()).render(filename, format="png", cleanup=True, view=view)

    def __repr__(self):
        return f"<QuantumCircuit({self.num_qubits} qubits, {self.size()} operations)>"

    def __str__(self):
        return self.draw()
