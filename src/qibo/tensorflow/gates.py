# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import numpy as np
import tensorflow as tf
from qibo.base import gates as base_gates
from qibo.config import matrices
from typing import Optional, Sequence, Tuple


class TensorflowGate(base_gates.Gate):
    """The base Tensorflow gate.

    **Properties:**
        matrix: The matrix that represents the gate to be applied.
            This is (2, 2) for 1-qubit gates and (4, 4) for 2-qubit gates.
        qubits: List with the qubits that the gate is applied to.
    """

    dtype = matrices.dtype
    _chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self):
      super(TensorflowGate, self).__init__()
      self.einsum_string = None
      # For `controlled_by` gates
      # See the docstring of `_calculate_transpose_order` for more details
      self.transpose_order = None
      self.reverse_transpose_order = None

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        """Sets the number of qubit that this gate acts on.

        This is called automatically by the `Circuit.add` method if the gate
        is used on a `Circuit`. If the gate is called on a state then `nqubits`
        is set during the first `__call__`.
        When `nqubits` is set we also calculate the einsum string so that it
        is calculated only once per gate.
        """
        base_gates.Gate.nqubits.fset(self, n)
        if self.is_controlled_by:
            self.transpose_order, targets = self._calculate_transpose_order()
            ncontrol = len(self.control_qubits)
            self.einsum_string = self._create_einsum_str(targets, n - ncontrol)
            # Calculate the reverse order for transposing the state legs so that
            # control qubits are back to their original positions
            self.reverse_transpose_order = self.nqubits * [0]
            for i, r in enumerate(self.transpose_order):
                self.reverse_transpose_order[r] = i
        else:
            self.einsum_string = self._create_einsum_str(self.qubits, n)

    def __call__(self, state: tf.Tensor) -> tf.Tensor:
        """Implements the `Gate` on a given state."""
        if self._nqubits is None:
            self.nqubits = len(tuple(state.shape))

        if self.is_controlled_by:
            return self._controlled_by_call(state)

        return tf.einsum(self.einsum_string, state, self.matrix)

    def _calculate_transpose_order(self):
        """Helper method for `_controlled_by_call`.

        This is used only for `controlled_by` gates.

        The loop generates:
          A) an `order` that is used to transpose `state`
             so that control legs are moved in the front
          B) a `targets` list which is equivalent to the
             `target_qubits` tuple but each index is reduced
             by the amount of control qubits that preceed it.
        This method is called by the `nqubits` setter so that the loop runs
        once per gate (and not every time the gate is called).
        """
        loop_start = 0
        order = list(self.control_qubits)
        targets = list(self.target_qubits)
        for control in self.control_qubits:
            for i in range(loop_start, control):
                order.append(i)
            loop_start = control + 1

            for i, t in enumerate(self.target_qubits):
                if t > control:
                    targets[i] -= 1
        for i in range(loop_start, self.nqubits):
            order.append(i)

        return order, targets

    def _controlled_by_call(self, state: tf.Tensor) -> tf.Tensor:
        """Gate __call__ method for `controlled_by` gates."""
        ncontrol = len(self.control_qubits)
        nactive = self.nqubits - ncontrol

        # Apply `einsum` only to the part of the state where all controls
        # are active. This should be `state[-1]`
        state = tf.transpose(state, self.transpose_order)
        state = tf.reshape(state, (2 ** ncontrol,) + nactive * (2,))
        updates = tf.einsum(self.einsum_string, state[-1], self.matrix)

        # Concatenate the updated part of the state `updates` with the
        # part of of the state that remained unaffected `state[:-1]`.
        state = tf.concat([state[:-1], updates[tf.newaxis]], axis=0)
        state = tf.reshape(state, self.nqubits * (2,))
        return tf.transpose(state, self.reverse_transpose_order)

    @classmethod
    def _create_einsum_str(cls, qubits: Sequence[int], nqubits: int) -> str:
        """Creates index string for `tf.einsum`.

        Args:
            qubits: List with the qubit indices that the gate is applied to.

        Returns:
            String formated as {input state}{gate matrix}->{output state}.
        """
        if len(qubits) + nqubits > len(cls._chars):
            raise NotImplementedError("Not enough einsum characters.")

        input_state = list(cls._chars[: nqubits])
        output_state = input_state[:]
        gate_chars = list(cls._chars[nqubits : nqubits + len(qubits)])

        for i, q in enumerate(qubits):
            gate_chars.append(input_state[q])
            output_state[q] = gate_chars[i]

        input_str = "".join(input_state)
        gate_str = "".join(gate_chars)
        output_str = "".join(output_state)
        return "{},{}->{}".format(input_str, gate_str, output_str)


class H(TensorflowGate, base_gates.H):

    def __init__(self, *args):
        base_gates.H.__init__(self, *args)
        self.matrix = matrices.H


class X(TensorflowGate, base_gates.X):

    def __init__(self, *args):
        base_gates.X.__init__(self, *args)
        self.matrix = matrices.X

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if controls are one or two."""
        gate = super(X, self).controlled_by(*q)
        if len(q) == 1:
            return CNOT(q[0], self.target_qubits[0])
        elif len(q) == 2:
            return TOFFOLI(q[0], q[1], self.target_qubits[0])
        return gate


class Y(TensorflowGate, base_gates.Y):

    def __init__(self, *args):
        base_gates.Y.__init__(self, *args)
        self.matrix = matrices.Y


class Z(TensorflowGate, base_gates.Z):

    def __init__(self, *args):
        base_gates.Z.__init__(self, *args)
        self.matrix = matrices.Z


class M(TensorflowGate, base_gates.M):
    from qibo.tensorflow import measurements

    def __init__(self, *args, register_name: Optional[str] = None):
        base_gates.M.__init__(self, *args, register_name=register_name)

    @base_gates.Gate.nqubits.setter
    def nqubits(self, n: int):
        base_gates.Gate.nqubits.fset(self, n)

    def __call__(self, state: tf.Tensor, nshots: int,
                 samples_only: bool = False) -> tf.Tensor:
        if self._nqubits is None:
            self.nqubits = len(tuple(state.shape))

        # Trace out unmeasured qubits
        probs = tf.reduce_sum(tf.square(tf.abs(state)),
                              axis=self.unmeasured_qubits)
        logits = tf.math.log(tf.reshape(probs, (2 ** len(self.target_qubits),)))
        # Generate samples
        samples_dec = tf.random.categorical(logits[tf.newaxis], nshots,
                                            dtype=tf.int64)[0]
        if samples_only:
            return samples_dec

        return self.measurements.GateResult(
            self.qubits, state, decimal_samples=samples_dec)


class RX(TensorflowGate, base_gates.RX):

    def __init__(self, *args):
        base_gates.RX.__init__(self, *args)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)
        self.matrix = phase * (cos * matrices.I - 1j * sin * matrices.X)


class RY(TensorflowGate, base_gates.RY):

    def __init__(self, *args):
        base_gates.RY.__init__(self, *args)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta / 2.0)
        cos = tf.cast(tf.math.real(phase), dtype=self.dtype)
        sin = tf.cast(tf.math.imag(phase), dtype=self.dtype)
        self.matrix = phase * (cos * matrices.I - 1j * sin * matrices.Y)


class RZ(TensorflowGate, base_gates.RZ):

    def __init__(self, *args):
        base_gates.RZ.__init__(self, *args)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta)
        rz = tf.eye(2, dtype=self.dtype)
        self.matrix = tf.tensor_scatter_nd_update(rz, [[1, 1]], [phase])

    def controlled_by(self, *q):
        """Fall back to CRZ if control is one."""
        gate = super(RZ, self).controlled_by(*q)
        if len(q) == 1:
            return CRZ(q[0], self.target_qubits[0], self.theta)
        return gate


class CNOT(TensorflowGate, base_gates.CNOT):

    def __init__(self, *args):
        base_gates.CNOT.__init__(self, *args)
        self.matrix = matrices.CNOT


class CRZ(TensorflowGate, base_gates.CRZ):

    def __init__(self, *args):
        base_gates.CRZ.__init__(self, *args)

        theta = tf.cast(self.theta, dtype=self.dtype)
        phase = tf.exp(1j * np.pi * theta)
        crz = tf.eye(4, dtype=self.dtype)
        crz = tf.tensor_scatter_nd_update(crz, [[3, 3]], [phase])
        self.matrix = tf.reshape(crz, 4 * (2,))


class SWAP(TensorflowGate, base_gates.SWAP):

    def __init__(self, *args):
        base_gates.SWAP.__init__(self, *args)
        self.matrix = matrices.SWAP


class TOFFOLI(TensorflowGate, base_gates.TOFFOLI):

    def __init__(self, *args):
        base_gates.TOFFOLI.__init__(self, *args)
        self.matrix = matrices.TOFFOLI


class Unitary(TensorflowGate, base_gates.Unitary):

    def __init__(self, *args, name: Optional[str] = None):
        base_gates.Unitary.__init__(self, *args, name=name)
        rank = 2 * len(self.target_qubits)
        # This reshape will raise an error if the number of target qubits
        # given is incompatible to the shape of the given unitary.
        self.matrix = tf.convert_to_tensor(self.unitary, dtype=self.dtype)
        self.matrix = tf.reshape(self.matrix, rank * (2,))


class Flatten(TensorflowGate, base_gates.Flatten):

    def __init__(self, *args):
        base_gates.Flatten.__init__(self, *args)

    def __call__(self, state):
        if self.nqubits is None:
            self.nqubits = len(tuple(state.shape))

        if len(self.coefficients) != 2 ** self.nqubits:
                raise ValueError(
                    "Circuit was created with {} qubits but the "
                    "flatten layer state has {} coefficients."
                    "".format(self.nqubits, self.coefficients)
                )

        _state = np.array(self.coefficients).reshape(self.nqubits * (2,))
        return tf.convert_to_tensor(_state, dtype=state.dtype)
