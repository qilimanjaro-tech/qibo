import numpy as np
from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


class Grover(object):
    """Model that performs Grover's algorithm.

    # TODO: Add a few more details on Grover's algorithm and/or reference.

    Args:
        oracle (:class:`qibo.core.circuit.Circuit`): quantum circuit that flips
            the sign using a Grover ancilla initialized with -X-H-
            and expected to have the total size of the circuit.
        superposition_circuit (:class:`qibo.core.circuit.Circuit`): quantum circuit that
            takes an initial state to a superposition. Expected to use the first
            set of qubits to store the relevant superposition.
        initial_state_circuit (:class:`qibo.core.circuit.Circuit`): quantum circuit
            that initializes the state. If empty defaults to |000..00>
        superposition_qubits (int): number of qubits that store the relevant superposition.
            Leave empty if superposition does not use ancillas.
        superposition_size (int): how many states are in a superposition.
            Leave empty if its an equal superposition of quantum states.
        number_solutions (int): number of expected solutions. Needed for normal Grover.
            Leave empty for iterative version.
        check (function): function that returns True if the solution has been
            found. Required of iterative approach.
            First argument should be the bitstring to check.
        check_args (tuple): arguments needed for the check function.
            The found bitstring not included.
        iterative (Bool): force the use of the iterative Grover
    """

    def __init__(self, oracle, superposition_circuit=None, initial_state_circuit=None,
                 superposition_qubits=None, superposition_size=None, number_solutions=None,
                 check=None, check_args=(), iterative=False):

        self.oracle = oracle
        self.initial_state_circuit = initial_state_circuit

        if superposition_circuit:
            self.superposition = superposition_circuit
        else:
            if not superposition_qubits:
                raise_error(ValueError)
            self.superposition = Circuit(superposition_qubits)
            self.superposition.add([gates.H(i) for i in range(superposition_qubits)])

        if superposition_qubits:
            self.sup_qubits = superposition_qubits
        else:
            self.sup_qubits = self.superposition.nqubits

        if superposition_size:
            self.sup_size = superposition_size
        else:
            self.sup_size = int(2**self.superposition.nqubits)

        self.check = check
        self.check_args = check_args
        self.num_sol = number_solutions
        self.iterative = iterative

    def initialize(self):
        """Initialize the Grover algorithm with the superposition and Grover ancilla."""

        c = Circuit(self.oracle.nqubits)
        c.add(gates.X(self.oracle.nqubits - 1))
        c.add(gates.H(self.oracle.nqubits - 1))
        c.add(self.superposition.on_qubits(*range(self.superposition.nqubits)))
        return c

    def diffusion(self):
        """Construct the diffusion operator out of the superposition circuit."""
        nqubits = self.superposition.nqubits
        c = Circuit(nqubits + 1)
        c.add(self.superposition.invert().on_qubits(*range(nqubits)))
        if self.initial_state_circuit:
            c.add(self.initial_state_circuit.invert().on_qubits(*range(self.initial_state_circuit.nqubits)))
        c.add([gates.X(i) for i in range(nqubits)])
        c.add(gates.X(nqubits).controlled_by(*range(nqubits)))
        c.add([gates.X(i) for i in range(nqubits)])
        if self.initial_state_circuit:
            c.add(self.initial_state_circuit.on_qubits(*range(self.initial_state_circuit.nqubits)))
        c.add(self.superposition.on_qubits(*range(nqubits)))
        return c

    def step(self):
        """Combine oracle and diffusion for a Grover step."""
        c = Circuit(self.oracle.nqubits)
        c += self.oracle
        diffusion = self.diffusion()
        qubits = list(range(diffusion.nqubits - 1))
        qubits.append(self.oracle.nqubits - 1)
        c.add(diffusion.on_qubits(*qubits))
        return c

    def circuit(self, iterations):
        """Creates circuit that performs Grover's algorithm with a set amount of iterations.

        Args:
            iterations (int): number of times to repeat the Grover step.

        Returns:
            :class:`qibo.core.circuit.Circuit` that performs Grover's algorithm.
        """
        c = Circuit(self.oracle.nqubits)
        c += self.initialize()
        for _ in range(iterations):
            c += self.step()
        c.add(gates.M(*range(self.sup_qubits)))
        return c

    def iterative_grover(self, lamda_value=6/5):
        """Iterative approach of Grover for when the number of solutions is not known.

        Args:
            lamda_value (real): parameter that controls the evolution of the iterative method.
                                Must be between 1 and 4/3.

        Returns:
            measured (str): bitstring measured and checked as a valid solution.
            total_iterations (int): number of times the oracle has been called.
        """
        k = 1
        lamda = lamda_value
        total_iterations = 0
        while True:
            it = np.random.randint(k + 1)
            if it != 0:
                total_iterations += it
                circuit = self.circuit(it)
                result = circuit(nshots=1)
                measured = result.frequencies(binary=True).most_common(1)[0][0]
                if self.check(measured, *self.check_args):
                    return measured, total_iterations
            k = min(lamda * k, np.sqrt(self.sup_size))
            if total_iterations > (9/4) * np.sqrt(self.sup_size):
                raise_error(TimeoutError, "Cancelling iterative method as too "
                                          "many iterations have taken place.")

    def execute(self, nshots=100, freq=False):
        """Execute Grover's algorithm.

        If the number of solutions is given, calculates iterations,
        otherwise it uses an iterative approach.

        Args:
            nshots (int): number of shots in order to get the frequencies.
            freq (bool): print the full frequencies after the exact Grover algorithm.

        Returns:
            self.solution (str): bitstring (or list of bitstrings) measured as solution of the search.
            self.iterations (int): number of oracle calls done to reach a solution.
        """
        if self.num_sol and not self.iterative:
            it = int(np.pi * np.sqrt(self.sup_size / self.num_sol) / 4)
            circuit = self.circuit(it)
            result = circuit(nshots=nshots).frequencies(binary=True)
            if freq:
                log.info("Result of sampling Grover's algorihm")
                log.info(result)
                self.frequencies = result
            log.info(f"Most common states found using Grover's algorithm with {it} iterations:")
            most_common = result.most_common(self.num_sol)
            self.solution = []
            self.iterations = it
            for i in most_common:
                log.info(i[0])
                self.solution.append(i[0])
                if self.check:
                    if self.check(i[0], *self.check_args):
                        log.info('Solution checked and successful.')
                    else:
                        log.info('Not a solution of the problem. Something went wrong.')
        else:
            if not self.check:
                raise_error(ValueError, "Check function needed for iterative approach.")
            measured, total_iterations = self.iterative_grover()
            log.info('Solution found in an iterative process.')
            log.info(f'Solution: {measured}')
            log.info(f'Total Grover iterations taken: {total_iterations}')
            self.solution = measured
            self.iterations = total_iterations
        return self.solution, self.iterations