# API Reference

`QuantumCircuit(qubits=0)` accepts a nonnegative integer qubit count. Gate methods are chainable: x, y, z, h, s, sdg, t, tdg, i, rx, ry, rz, u3, cx/cnot, cy, cz, swap, ccx/toffoli, cswap/fredkin. Rotation angles precede qubits. Multi-qubit operands must be distinct; their order defines matrix bit order. `append(gate, qubits)` supports custom unitary Gate objects of any arity.

`measure(q)` appends a terminal readout; `measure_all()` measures remaining unmeasured wires. Repeated individual measurement or a gate on a measured wire raises ValueError before modifying the graph.

- `dag` / `get_dag()`: independent rustworkx PyDAG snapshot, always complete.
- `dag_nodes()`: independent node dictionaries with type input/output/gate/measurement.
- `dag_edges()`: (source, destination, {qubit: index}) triples.
- `topological_sort()`: node indices in dependency order.
- `get_gate_nodes()`: (index, operation dictionary) snapshots, including readout.
- `dag_layers()`: parallel operation layers, expressed as node indices.
- `depth()`, `size()`, `count_ops()`: analysis derived from the graph; include readout.
- `copy()`: independent circuit.
- `compose(other, qubits=None)`: append a circuit using an ordered source-to-target
  wire mapping; failure leaves the receiving circuit unchanged.
- `remove_operation(index)`: reconnect wires around a removed operation. Input
  and output nodes cannot be removed. Node indices may have gaps after removal.
- `draw()`: text showing layers; `to_dot()`: Graphviz source;
  `visualize_dag(filename, view=False)`: PNG, requiring the Graphviz executable.

Parallel wire edges between the same nodes are intentional. Use weighted edge iteration to retain every qubit label. Independent operations have no guaranteed relative order. Snapshots are not a mutation interface.

`Simulator(seed=None).run(circuit, shots=1024, initial_statevector=None)` returns SimulatorResult. Statevector evolves unitarily; terminal measurement samples do not collapse the returned statevector. Zero shots returns no counts. Without measurement nodes, counts are empty. A supplied Statevector must match circuit width. Sampling uses a local RNG and leaves NumPy global random state alone.

`get_statevector(circuit, initial_statevector=None)` returns the evolved state. Results expose get_counts(), get_statevector(), and get_probabilities(). Statevector supports basis construction, probabilities, sampling, evolution, expectation values, inner products, and amplitude access.

`quirk.utils` exposes create_bell_pair, create_ghz_state, initialize_state, pauli_string, controlled_gate, calculate_unitary, random_circuit, fidelity, and trace_distance. initialize_state appends X gates for a basis label; it is intended for initially zero wires and does not reset existing states. calculate_unitary traverses DAG operations and rejects measurements.
