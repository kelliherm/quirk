import itertools
import unittest
import numpy as np
import rustworkx as rwx
from quirk import QuantumCircuit, Simulator, Statevector, Gate, XGate, CXGate, CYGate, CZGate, SWAPGate, CCXGate, CSWAPGate
from quirk.utils import calculate_unitary, create_ghz_state, random_circuit


class CircuitTests(unittest.TestCase):
    def assert_wires(self, circuit):
        dag = circuit.get_dag()
        self.assertTrue(rwx.is_directed_acyclic_graph(dag))
        for q in range(circuit.num_qubits):
            edges = [(src, dst) for src, dst, payload in dag.weighted_edge_list() if payload['qubit'] == q]
            current = circuit.input_nodes[q]
            seen = set()
            while current != circuit.output_nodes[q]:
                self.assertNotIn(current, seen)
                seen.add(current)
                successors = [dst for src, dst in edges if src == current]
                self.assertEqual(len(successors), 1)
                current = successors[0]
            self.assertEqual(len(seen), len(edges))

    def test_empty_and_zero_width(self):
        for width in (0, 1, 4):
            q = QuantumCircuit(width)
            self.assertEqual(q.depth(), 0)
            self.assertEqual(q.size(), 0)
            self.assert_wires(q)
            self.assertEqual(Simulator().get_statevector(q)[0], 1)
        self.assertEqual(Statevector.from_label('').probabilities_dict(), {'': 1.0})

    def test_inspection_then_append(self):
        q = QuantumCircuit(3).h(0).cx(0, 1)
        old = q.get_dag()
        for _ in range(3):
            q.draw(); q.dag_layers(); q.get_dag(); q.to_dot()
        q.cx(0, 1).x(2)
        self.assertEqual(q.size(), 4)
        self.assertEqual(old.num_nodes(), 8)
        self.assert_wires(q)
        self.assertEqual(q.depth(), 3)
        self.assertEqual(q.count_ops(), {'H': 1, 'CX': 2, 'X': 1})

    def test_parallel_edges_and_dot(self):
        q = QuantumCircuit(2).cx(0, 1).cx(0, 1)
        operations = q.get_gate_nodes()
        edges = [(s, d, w) for s, d, w in q.dag_edges() if s == operations[0][0] and d == operations[1][0]]
        self.assertEqual({w['qubit'] for _, _, w in edges}, {0, 1})
        self.assertEqual(q.to_dot().count('label="q[0]"'), 3)
        self.assertEqual(q.to_dot().count('label="q[1]"'), 3)
        self.assert_wires(q)

    def test_snapshot_isolation(self):
        gate = XGate()
        q = QuantumCircuit(1).append(gate, [0])
        gate.matrix[:] = 0
        snapshot = q.dag
        snapshot[q.get_gate_nodes()[0][0]]['gate'].matrix[:] = 0
        snapshot.clear()
        operation = q.get_gate_nodes()[0][1]
        operation['gate'].matrix[:] = 0
        self.assertEqual(q.size(), 1)
        np.testing.assert_allclose(Simulator().get_statevector(q).data(), [0, 1])
        self.assertFalse(hasattr(q, 'instructions'))

    def test_invalid_operations_atomic(self):
        q = QuantumCircuit(2)
        for action, error in (
            (lambda: q.cx(0, 0), ValueError),
            (lambda: q.h(2), IndexError),
            (lambda: q.h(0.5), TypeError),
            (lambda: q.h(True), TypeError),
            (lambda: q.append(XGate(), []), ValueError),
            (lambda: q.append(Gate('bad', 1, np.zeros((2, 2))), [0]), ValueError),
        ):
            with self.assertRaises(error): action()
            self.assertEqual(q.size(), 0)
            self.assert_wires(q)
        for width, error in ((-1, ValueError), (1.2, TypeError), (True, TypeError)):
            with self.assertRaises(error): QuantumCircuit(width)

    def test_layers_follow_dependencies(self):
        q = QuantumCircuit(4).h(0).h(1).cx(0, 2).cx(1, 3).cx(2, 3)
        self.assertEqual([len(layer) for layer in q.dag_layers()], [2, 2, 1])
        graph = q.dag
        for layer in q.dag_layers():
            used = []
            for i in layer: used.extend(graph[i]['qubits'])
            self.assertEqual(len(used), len(set(used)))
        positions = {i: n for n, i in enumerate(q.topological_sort())}
        for src, dst, _ in q.dag_edges():
            self.assertLess(positions[src], positions[dst])

    def test_remove_shared_wires_and_index_gaps(self):
        q = QuantumCircuit(3).cx(0, 1).cx(0, 1).h(2)
        middle = q.get_gate_nodes()[1][0]
        q.remove_operation(middle)
        self.assertEqual(q.size(), 2)
        q.to_dot()
        self.assert_wires(q)
        q.cx(1, 2)
        self.assert_wires(q)
        with self.assertRaises(ValueError): q.remove_operation(q.input_nodes[0])
        with self.assertRaises(IndexError): q.remove_operation(1000)

    def test_compose_mapping_and_failure(self):
        sub = QuantumCircuit(2).x(0).cx(0, 1)
        q = QuantumCircuit(3).compose(sub, [2, 0])
        self.assertEqual(Simulator().get_statevector(q).probabilities_dict(), {'101': 1.0})
        q.measure(0)
        previous = q.size()
        with self.assertRaises(ValueError): q.compose(sub, [1, 0])
        self.assertEqual(q.size(), previous)
        self.assert_wires(q)
        clone = q.copy()
        clone.x(1)
        self.assertEqual(q.size(), previous)
        self.assertEqual(clone.size(), previous + 1)

    def test_measurement_readout_and_terminal_rule(self):
        q = QuantumCircuit(3).x(2).measure(2).measure(0)
        self.assertEqual(Simulator(2).run(q, shots=25).counts, {'01': 25})
        with self.assertRaises(ValueError): q.x(2)
        with self.assertRaises(ValueError): q.measure(0)
        q.h(1)  # Other unmeasured wires remain usable.
        q.measure_all().measure_all()
        self.assertEqual(q.count_ops()['measure'], 3)
        self.assert_wires(q)
        self.assertEqual(Simulator().run(q, shots=0).counts, {})
        with self.assertRaises(ValueError): calculate_unitary(q)

    def test_entanglement_sampling_and_rng_isolation(self):
        q = create_ghz_state(QuantumCircuit(3), [0, 1, 2])
        state = Simulator().get_statevector(q)
        np.testing.assert_allclose(state.data(), [2**-0.5, 0, 0, 0, 0, 0, 0, 2**-0.5])
        q.measure_all()
        np.random.seed(10)
        expected = np.random.random(3)
        np.random.seed(10)
        first = Simulator(42).run(q, shots=500)
        self.assertEqual(first.counts, Simulator(42).run(q, shots=500).counts)
        self.assertEqual(set(first.counts), {'000', '111'})
        self.assertEqual(sum(first.counts.values()), 500)
        np.testing.assert_equal(np.random.random(3), expected)
        self.assertEqual(first.statevector.probabilities_dict(), {'000': 0.5, '111': 0.5})

    def test_operand_order_for_all_multi_qubit_permutations(self):
        # Independent basis-bit reference catches transposed or sorted operands.
        for gate in (CXGate(), CYGate(), CZGate(), SWAPGate(), CCXGate(), CSWAPGate()):
            for targets in itertools.permutations(range(4), gate.num_qubits):
                circuit = QuantumCircuit(4).append(gate, targets)
                for basis in range(16):
                    bits = [int(b) for b in f'{basis:04b}']
                    local = sum(bits[q] << (len(targets)-1-i) for i, q in enumerate(targets))
                    result = int(np.argmax(abs(gate.to_matrix()[:, local])))
                    amplitude = gate.to_matrix()[result, local]
                    for i, q in enumerate(targets):
                        bits[q] = (result >> (len(targets)-1-i)) & 1
                    expected = int(''.join(map(str, bits)), 2)
                    state = Simulator().get_statevector(circuit, Statevector.from_int(basis, 4)).data()
                    np.testing.assert_allclose(state, amplitude * np.eye(16)[expected])

    def test_all_builtin_gates_and_custom_arity(self):
        for method in ('x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'i'):
            q = getattr(QuantumCircuit(1), method)(0)
            expected = q.get_gate_nodes()[0][1]['gate'].to_matrix()[:, 0]
            np.testing.assert_allclose(Simulator().get_statevector(q).data(), expected)
        for method in ('rx', 'ry', 'rz'):
            q = getattr(QuantumCircuit(1), method)(0.7, 0)
            np.testing.assert_allclose(Simulator().get_statevector(q).data(), q.get_gate_nodes()[0][1]['gate'].matrix[:, 0])
        q = QuantumCircuit(1).u3(0.3, 0.4, 0.5, 0)
        np.testing.assert_allclose(Simulator().get_statevector(q).data(), q.get_gate_nodes()[0][1]['gate'].matrix[:, 0])
        diagonal = np.diag(np.exp(1j * np.arange(16)))
        q = QuantumCircuit(4).append(Gate('phase', 4, diagonal), [3, 1, 0, 2])
        self.assert_wires(q)
        unitary = calculate_unitary(q)
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(16))
        initial = Statevector(np.ones(16))
        np.testing.assert_allclose(Simulator().get_statevector(q, initial).data(), unitary @ initial.data())

    def test_simulation_validation_and_helpers(self):
        for shots, error in ((-1, ValueError), (1.5, TypeError), (True, TypeError)):
            with self.assertRaises(error): Simulator().run(QuantumCircuit(1), shots=shots)
        with self.assertRaises(ValueError): Simulator().get_statevector(QuantumCircuit(1), Statevector.from_int(0, 2))
        for data in (np.ones((2, 2)), [np.nan, 0], [0, 0]):
            with self.assertRaises(ValueError): Statevector(data)
        q = random_circuit(3, 3, measure=True, seed=2)
        self.assert_wires(q)
        self.assertEqual(sum(Simulator(2).run(q, shots=30).counts.values()), 30)
        q = QuantumCircuit(2).h(1).cx(1, 0).ry(0.4, 0)
        np.testing.assert_allclose(calculate_unitary(q)[:, 0], Simulator().get_statevector(q).data())


if __name__ == '__main__':
    unittest.main()
