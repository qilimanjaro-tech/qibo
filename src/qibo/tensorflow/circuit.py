# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import circuit
from qibo.config import DTYPECPX
from qibo.tensorflow import gates, measurements
from typing import Optional


class TensorflowCircuit(circuit.BaseCircuit):
    """Implementation of circuit methods in Tensorflow."""

    def __init__(self, nqubits, dtype=DTYPECPX):
        """Initialize a Tensorflow circuit."""
        super(TensorflowCircuit, self).__init__(nqubits)
        self.dtype = dtype
        self.compiled_execute = None
        self._final_state = None

    def __add__(self, circuit: "TensorflowCircuit") -> "TensorflowCircuit":
        return TensorflowCircuit._circuit_addition(self, circuit)

    def _execute_func(self, state: tf.Tensor, nshots: Optional[int] = None
                      ) -> tf.Tensor:
        """Simulates the circuit gates.

        Can be compiled using `tf.function` or used as it is in Eager mode.
        """
        for gate in self.queue:
            state = gate(state)
        if self.measurement_gate is None or nshots is None:
            return None, tf.reshape(state, (2 ** self.nqubits,))
        return self.measurement_gate(state, nshots, samples_only=True), state

    def compile(self):
        """Compiles the circuit as a Tensorflow graph."""
        if self.compiled_execute is not None:
            raise RuntimeError("Circuit is already compiled.")
        self.compiled_execute = tf.function(self._execute_func)

    def execute(self, initial_state: Optional[tf.Tensor] = None,
                nshots: Optional[int] = None) -> tf.Tensor:
        """Executes the Tensorflow circuit.

        Args:
            initial_state: Initial state vector as a numpy array of shape
                (2 ** nqubits,).
                A Tensorflow tensor with shape nqubits * (2,) is also allowed
                as an initial state if it has the `dtype` of the circuit.
                If None the |000...0> state will be used as initial state.
            nshots: Number of shots to sample if the circuit contains
                measurement gates.
                If None the measurement gates will be ignored.

        Returns:
            If `nshots` is given and the circuit contains measurements this
                will return a `CircuitResult` object that contains information
                about the measured samples.
            If the circuit does not contain measurements or `nshots` is None
                this will return the final state vector as a Tensorflow tensor
                of shape (2 ** nqubits,).
        """
        if initial_state is None:
            state = self._default_initial_state()
        elif isinstance(initial_state, np.ndarray):
            state = tf.cast(initial_state.reshape(self.nqubits * (2,)),
                            dtype=self.dtype)
        elif isinstance(initial_state, tf.Tensor):
            if tuple(initial_state.shape) != self.nqubits * (2,):
                raise ValueError("Initial state should be a rank-n tensor if "
                                 "it is passed as a Tensorflow tensor but it "
                                 "has shape {}.".format(initial_state.shape))
            if initial_state.dtype != self.dtype:
                raise TypeError("Circuit is of type {} but initial state is "
                                "{}.".format(self.dtype, initial_state.dtype))
            state = initial_state
        else:
            raise TypeError("Initial state type {} is not recognized."
                            "".format(type(initial_state)))

        if self.compiled_execute is None:
            samples, self._final_state = self._execute_func(state, nshots)
        else:
            samples, self._final_state = self.compiled_execute(state, nshots)

        if self.measurement_gate is None or nshots is None:
            return self._final_state

        self.measurement_gate_result = measurements.GateResult(
            self.measurement_gate.qubits, state, decimal_samples=samples)
        return measurements.CircuitResult(
            self.measurement_sets, self.measurement_gate_result)

    @property
    def final_state(self) -> tf.Tensor:
        """"Returns the final state as a Tensorflow tensor.

        The tensor has shape (2 ** nqubits,).

        Raises:
            ValueError if the user attempts to access the final state before
                executing the circuit.
        """
        if self._final_state is None:
            raise ValueError("Cannot access final state before the circuit is "
                             "executed.")
        if self.measurement_gate_result is None:
            return self._final_state
        return tf.reshape(self._final_state, (2 ** self.nqubits,))

    def __call__(self, initial_state: Optional[tf.Tensor] = None,
                 nshots: Optional[int] = None) -> tf.Tensor:
        """Equivalent to `circuit.execute()`."""
        return self.execute(initial_state=initial_state, nshots=nshots)

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        initial_state = np.zeros(2 ** self.nqubits)
        initial_state[0] = 1
        initial_state = initial_state.reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(initial_state, dtype=self.dtype)
