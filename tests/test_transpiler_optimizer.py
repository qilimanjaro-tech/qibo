import networkx as nx
import pytest

from qibo import gates
from qibo.models import Circuit
from qibo.transpiler.optimizer import Preprocessing, Rearrange


def star_connectivity(names=["q0", "q1", "q2", "q3", "q4"], middle_qubit_idx=2):
    chip = nx.Graph()
    chip.add_nodes_from(names)
    graph_list = [
        (names[i], names[middle_qubit_idx])
        for i in range(len(names))
        if i != middle_qubit_idx
    ]
    chip.add_edges_from(graph_list)
    return chip


def test_preprocessing_error():
    circ = Circuit(7)
    preprocesser = Preprocessing(connectivity=star_connectivity())
    with pytest.raises(ValueError):
        new_circuit = preprocesser(circuit=circ)


def test_preprocessing_same():
    circ = Circuit(5)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
    assert new_circuit.ngates == 1


def test_preprocessing_add():
    circ = Circuit(3)
    circ.add(gates.CNOT(0, 1))
    preprocesser = Preprocessing(connectivity=star_connectivity())
    new_circuit = preprocesser(circuit=circ)
    assert new_circuit.ngates == 1
    assert new_circuit.nqubits == 5


def test_fusion():
    circuit = Circuit(2)
    circuit.add(gates.X(0))
    circuit.add(gates.Z(0))
    circuit.add(gates.Y(0))
    circuit.add(gates.X(1))
    fusion = Rearrange(max_qubits=1)
    fused_circ = fusion(circuit)
    assert isinstance(fused_circ.queue[0], gates.Unitary)
