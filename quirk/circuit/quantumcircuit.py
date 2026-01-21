"""
Quantum circuit implementation for building and managing quantum circuits.
"""

from typing import List, Optional, Union

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
from quirk.circuit.instruction import Instruction
from quirk.circuit.register import ClassicalRegister, QuantumRegister

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False


class QuantumCircuit:
    """Main class for building and managing quantum circuits."""

    def __init__(
        self,
        qubits: Optional[Union[int, QuantumRegister]] = None,
        clbits: Optional[Union[int, ClassicalRegister]] = None,
    ) -> None:
        """
        Initialize a quantum circuit.

        Args:
            qubits: Number of qubits or a QuantumRegister
            clbits: Number of classical bits or a ClassicalRegister
        """
        # Handle quantum register
        if isinstance(qubits, QuantumRegister):
            self.qubits = qubits.size
            self.qreg = qubits
        elif isinstance(qubits, int):
            self.qubits = qubits
            self.qreg = QuantumRegister(qubits)
        elif qubits is None:
            self.qubits = 0
            self.qreg = None
        else:
            raise TypeError("qubits must be an int or QuantumRegister")

        # Handle classical register
        if isinstance(clbits, ClassicalRegister):
            self.clbits = clbits.size
            self.creg = clbits
        elif isinstance(clbits, int):
            self.clbits = clbits
            self.creg = ClassicalRegister(clbits)
        elif clbits is None:
            self.clbits = 0
            self.creg = None
        else:
            raise TypeError("clbits must be an int or ClassicalRegister")

        # Store qubits and gates as a DAG
        self.dag: rwx.PyDAG = rwx.PyDAG()
        
        # Create input nodes for each qubit
        self.input_nodes = {}
        for i in range(self.qubits):
            node_idx = self.dag.add_node({"type": "input", "qubit": i, "label": f"q_in[{i}]"})
            self.input_nodes[i] = node_idx
        
        # Create output nodes for each qubit (final state)
        self.output_nodes = {}
        for i in range(self.qubits):
            node_idx = self.dag.add_node({"type": "output", "qubit": i, "label": f"q_out[{i}]"})
            self.output_nodes[i] = node_idx
        
        # Track the last gate node for each qubit wire
        # Initially, the "last" node for each qubit is the input node
        self.qubit_last_node = {i: self.input_nodes[i] for i in range(self.qubits)}
        
        # Track if DAG has been finalized to prevent duplicate edges
        self._dag_finalized = False
        
        # Store circuit instructions (for compatibility)
        self.instructions: List[Instruction] = []

    def _validate_qubit_index(self, qubit: int) -> None:
        """Validate that a qubit index is within bounds."""
        if qubit < 0 or qubit >= self.qubits:
            raise IndexError(
                f"Qubit index {qubit} out of range for circuit with {self.qubits} qubits"
            )

    def _validate_classical_bit_index(self, bit: int) -> None:
        """Validate that a classical bit index is within bounds."""
        if bit < 0 or bit >= self.clbits:
            raise IndexError(
                f"Classical bit index {bit} out of range for circuit with {self.clbits} classical bits"
            )

    def _add_gate(self, gate: Gate, qubits: List[int]) -> None:
        """
        Add a gate to the circuit DAG.

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

        # Create a node for this gate in the DAG
        gate_node_data = {
            "type": "gate",
            "gate": gate,
            "qubits": qubits,
            "label": f"{gate.name}"
        }
        gate_node_idx = self.dag.add_node(gate_node_data)
        
        # Add edges from the last node on each qubit wire to this gate node
        # Edges represent qubit states flowing from one gate to another
        for qubit in qubits:
            last_node = self.qubit_last_node[qubit]
            # Edge represents the qubit state connecting the previous operation to this gate
            self.dag.add_edge(last_node, gate_node_idx, {"qubit": qubit, "state": f"q[{qubit}]"})
        
        # Update the last node for each qubit to this gate
        for qubit in qubits:
            self.qubit_last_node[qubit] = gate_node_idx

        # Store instruction for backward compatibility
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

        # Create a special "measurement" gate node
        measure_gate = Gate("measure", 1, np.eye(2, dtype=complex))
        measure_node_data = {
            "type": "measurement",
            "gate": measure_gate,
            "qubits": [qubit],
            "classical_bits": [classical_bit],
            "label": f"M[q{qubit}->c{classical_bit}]"
        }
        measure_node_idx = self.dag.add_node(measure_node_data)
        
        # Connect from the last node on this qubit wire
        last_node = self.qubit_last_node[qubit]
        self.dag.add_edge(last_node, measure_node_idx, {"qubit": qubit, "state": f"q[{qubit}]"})
        
        # Update last node for this qubit
        self.qubit_last_node[qubit] = measure_node_idx
        
        # Store instruction for compatibility
        instruction = Instruction(measure_gate, [qubit], [classical_bit])
        self.instructions.append(instruction)
        return self

    def measure_all(self) -> "QuantumCircuit":
        """Measure all qubits to corresponding classical bits."""
        if self.clbits < self.qubits:
            raise ValueError(
                f"Not enough classical bits ({self.clbits}) to measure all qubits ({self.qubits})"
            )

        for i in range(self.qubits):
            self.measure(i, i)
        return self

    def depth(self) -> int:
        """Calculate the depth of the circuit (number of time steps)."""
        if not self.instructions:
            return 0

        # Track when each qubit is last used
        qubit_times = [0] * self.qubits
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
        lines.append(f"Quantum Circuit with {self.qubits} qubits")
        if self.clbits > 0:
            lines.append(f"Classical bits: {self.clbits}")
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
        lines.append(f"DAG: {len(self.dag.nodes())} nodes, {len(self.dag.edges())} edges")

        return "\n".join(lines)
    
    def finalize_dag(self) -> None:
        """
        Finalize the DAG by connecting all qubit wires to output nodes.
        This should be called before analyzing or visualizing the complete DAG.
        Only runs once to prevent duplicate edges.
        """
        # Only finalize once
        if self._dag_finalized:
            return
        
        for qubit in range(self.qubits):
            last_node = self.qubit_last_node[qubit]
            output_node = self.output_nodes[qubit]
            # Only add edge if not already connected to itself
            if last_node != output_node:
                self.dag.add_edge(last_node, output_node, {"qubit": qubit, "state": f"q[{qubit}]"})
        
        # Mark as finalized
        self._dag_finalized = True
    
    def get_dag(self) -> rwx.PyDAG:
        """
        Get the DAG representation of the circuit.
        Automatically finalizes the DAG before returning.
        
        Returns:
            The rustworkx DAG representing the circuit
        """
        self.finalize_dag()
        return self.dag
    
    def dag_nodes(self) -> list:
        """
        Get all nodes in the DAG.
        
        Returns:
            List of node data dictionaries
        """
        return list(self.dag.nodes())
    
    def dag_edges(self) -> list:
        """
        Get all edges in the DAG.
        
        Returns:
            List of edge data tuples
        """
        edges = []
        # In rustworkx, edges() returns edge data objects
        for edge_data in self.dag.edges():
            edges.append(edge_data)
        return edges
    
    def topological_sort(self) -> list:
        """
        Get a topological ordering of gates in the circuit.
        
        Returns:
            List of node indices in topological order
        """
        try:
            return rwx.topological_sort(self.dag)
        except rwx.DAGHasCycle:
            raise ValueError("Circuit DAG has a cycle - this should not happen!")
    
    def get_gate_nodes(self) -> list:
        """
        Get all gate nodes (excluding input/output nodes).
        
        Returns:
            List of (node_index, node_data) tuples for gate nodes
        """
        gate_nodes = []
        for idx in self.dag.node_indices():
            node_data = self.dag[idx]
            if node_data.get("type") == "gate" or node_data.get("type") == "measurement":
                gate_nodes.append((idx, node_data))
        return gate_nodes
    
    def dag_layers(self) -> list:
        """
        Get the circuit organized into layers (time steps).
        Each layer contains gates that can execute in parallel.
        
        Returns:
            List of layers, where each layer is a list of node indices
        """
        self.finalize_dag()
        
        # Use rustworkx's layers functionality
        layers = []
        node_to_layer = {}
        
        # Process nodes in topological order
        for node_idx in rwx.topological_sort(self.dag):
            node_data = self.dag[node_idx]
            
            # Skip input nodes
            if node_data.get("type") == "input":
                node_to_layer[node_idx] = 0
                continue
            
            # Find predecessors of this node
            predecessors = list(self.dag.predecessor_indices(node_idx))
            
            # Find the maximum layer of all predecessors
            if predecessors:
                max_pred_layer = max(node_to_layer.get(pred, 0) for pred in predecessors)
                layer = max_pred_layer + 1
            else:
                layer = 0
            
            node_to_layer[node_idx] = layer
            
            # Add to layers list
            while len(layers) <= layer:
                layers.append([])
            layers[layer].append(node_idx)
        
        # Filter out layers with only input/output nodes
        filtered_layers = []
        for layer in layers:
            gate_layer = [n for n in layer if self.dag[n].get("type") in ["gate", "measurement"]]
            if gate_layer:
                filtered_layers.append(gate_layer)
        
        return filtered_layers
    
    def visualize_dag(self, filename: str = "circuit_dag", view: bool = False) -> Optional[str]:
        """
        Visualize the DAG using Graphviz.
        
        Args:
            filename: Output filename (without extension)
            view: Whether to open the visualization automatically
            
        Returns:
            Path to the generated file, or None if Graphviz is not available
        """
        if not HAS_GRAPHVIZ:
            print("Warning: graphviz package not installed. Install with: pip install graphviz")
            return None
        
        self.finalize_dag()
        
        # Create a Graphviz digraph
        dot = graphviz.Digraph(comment='Quantum Circuit DAG')
        dot.attr(rankdir='LR')  # Left to right layout
        dot.attr('node', shape='box')
        
        # Build a mapping of node indices (needed since nodes() doesn't give us indices)
        node_map = {}
        for idx in range(len(self.dag.nodes())):
            node_map[idx] = self.dag[idx]
        
        # Add nodes with correct indices
        for idx, node_data in node_map.items():
            node_type = node_data.get('type', 'unknown')
            label = node_data.get('label', f'Node {idx}')
            
            # Style nodes based on type
            if node_type == 'input':
                dot.node(str(idx), label, shape='circle', style='filled', fillcolor='lightblue')
            elif node_type == 'output':
                dot.node(str(idx), label, shape='circle', style='filled', fillcolor='lightgreen')
            elif node_type == 'measurement':
                dot.node(str(idx), label, shape='box', style='filled', fillcolor='lightyellow')
            else:  # gate
                gate = node_data.get('gate')
                qubits = node_data.get('qubits', [])
                gate_label = f"{gate.name}\nq{qubits}" if gate else label
                dot.node(str(idx), gate_label, shape='box', style='filled', fillcolor='lightcoral')
        
        # Add edges with qubit labels
        edge_list = self.dag.edge_list()
        for src_idx, tgt_idx in edge_list:
            edge_data = self.dag.get_edge_data(src_idx, tgt_idx)
            qubit = edge_data.get('qubit', '?')
            edge_label = f"q[{qubit}]"
            dot.edge(str(src_idx), str(tgt_idx), label=edge_label)
        
        # Render
        output_path = dot.render(filename, format='png', cleanup=True, view=view)
        return output_path

    def __repr__(self) -> str:
        return f"<QuantumCircuit({self.qubits} qubits, {self.clbits} classical bits, {self.size()} gates)>"

    def __str__(self) -> str:
        return self.draw() or ""
