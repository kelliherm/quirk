"""Basic example program showing functionality of Quirk."""

from quirk import QuantumCircuit, Simulator


if __name__ == "__main__":
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    print(qc)

    print(f"Layers: {qc.dag_layers()}")
    print(f"Edges: {qc.dag_edges()}")

    qc.get_dag()
    qc.compose(QuantumCircuit(1).z(0), qubits=[2])

    print(f"State probabilities: {Simulator().get_statevector(qc).probabilities_dict()}")

    qc.measure_all()
    print(f"Readout: {Simulator().run(qc, shots=1024).get_counts()}")
