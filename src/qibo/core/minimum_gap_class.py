import numpy as np
# import matplotlib.pyplot as plt
import copy
from qibo.config import EIGVAL_CUTOFF


class DegeneratedGap():
    def __init__(self, energy_levels):
        """
        Trims the energy levels data to have only the states that correspond to the (degenerated) ground state and the
        first excited state at the end of the evolution.

        Args:
            energy_levels (List[List[float(Energy_Value)]]): List of energy levels, each element corresponds to a list
            with the values of the energy at each time step for a given energy level.
        """
        self._energy_levels, self._n_levels = self._get_degenerated_levels(energy_levels)
        self._n_steps = len(self._energy_levels[0])

    def _get_degenerated_levels(self, energy_levels):
        """Returns the list of energy levels until the first excited state (included) at the end of the evolution.

        Args:
            energy_levels (List[List[float(Energy_Value)]]): List of energy levels, each element corresponds to a list
            with the values of the energy at each time step for a given energy level.

        Raises:
            ValueError: Raises an error if all the energy levels are degenerated at the end of the evolution (if there
            isn't any excited state).

        Returns:
            energy_levels (List[List[float(Energy_Value)]]): List of energy levels, each element corresponds to a list
            with the values of the energy at each time step for a given energy level. It includes all degenerated
            ground states and the first excited state.
        """
        for i, level in enumerate(energy_levels):
            if level[-1] - energy_levels[0][-1] > EIGVAL_CUTOFF:
                return energy_levels[:i+1], i+1

        raise ValueError('ERROR: All energy levels passed are degenerated with the ground state.')

    def _check_larger_gaps(self, current, step):
        """Checks which gaps between two consecutive energy levels are larger than the gap between current[0] and
        current[1] levels. Only levels above current[1] are taken into consideration.

        Args:
            current (tuple): Indicates the two energy levels that are used to compute the gap until that time step.
            step (int): Time step of the adiabatic evolution

        Returns:
            larger_gaps (List[tuple]): List of tuples containing pairs of consecutive energy levels with a gap larger
            than the current gap.
        """
        larger_gaps = []
        for i in range(self._n_levels-1, current[1], -1):
            if (self._energy_levels[i][step] - self._energy_levels[i-1][step]) > (
                self._energy_levels[current[1]][step] - self._energy_levels[current[0]][step]):
                larger_gaps.append((i-1, i))
        return larger_gaps


class Method_1(DegeneratedGap):
    """This method starts by taking the gap between the first two states and keeps checking if there is a larger gap
    between consecutive levels above the the energy levels used in the previous step.
    It gives an approximate value for the minimum gap, the quality of the solution will depend on the shape of the
    energy levels evolution."""
    def __init__(self, energy_levels):
        super(Method_1, self).__init__(energy_levels)

    def compute_gap(self):
        """Computes the value of the gap for each time step such that at the end of the evolution we end in one of the
        ground states.

        Returns:
            gap (List[List[float], List[tuple(int, int)]): List containing two lists: the first one has the values of
            the gap at each time step. The second list contains a set of tuples (lower_level, upper_level) giving which
            pairs of energy levels were used to compute the gap at each time step.
            energy_levels (List[List[float(Energy_Value)]]): List of energy levels, each element corresponds to a list
            with the values of the energy at each time step for a given energy level. It includes all degenerated
            ground states and the first excited state.
        """
        current = (0, 1)  # Start by considering the gap between the first two levels
        gap = [[], []]

        for step in range(self._n_steps):
            larger_gaps = self._check_larger_gaps(current, step)
            # If there are larger gaps, we take the one closer to the first excited state at the end of the evolution
            # until the end of the evolution or until a larger gap is found.
            if larger_gaps != []:
                current = larger_gaps[0]

            gap[0].append(self._energy_levels[current[1]][step] - self._energy_levels[current[0]][step])
            gap[1].append((current[0], current[1]))
        return gap, self._energy_levels

class Method_2(DegeneratedGap):
    """Method that explores all possible paths using a discrete set of energy values during the adiabatic evolution.
    It works by checking the evolution of the minimum gap considering that we jump to higher energy states (inside the
    set of states that end up being one of the degenerated ground states) in different parts of the evolution such that
    we find the path that gives the maximum value for the minimum gap.
    By path we mean the set of energy levels used to compute the gap at each time step during all the evolution.
    It also includes the mode 'integrate' that instead of looking for the maximum value of the instantaneous minimum
    gap, it maximizes the sum of the square of the gap at each time step.
    This method gives the most optimal solution within the precision of the energy levels data."""
    def __init__(self, energy_levels):
        super(Method_2, self).__init__(energy_levels)

    def compute_gap(self, mode = 'minimum'):
        """Computes the value of the gap for each time step such that at the end of the evolution we end in one of the
        ground states.

        Args:
            mode (str): if 'minimum', the method maximizes the value of the minimum gap of the entire evolution.
            If 'integrate', it maximizes the sum of (gap)^2 with the values of the gap at each time step.

        Returns:
            gap (List[List[float], List[tuple(int, int)]): List containing two lists: the first one has the values of
            the gap at each time step. The second list contains a set of tuples (lower_level, upper_level) giving which
            pairs of energy levels were used to compute the gap at each time step.
            energy_levels (List[List[float(Energy_Value)]]): List of energy levels, each element corresponds to a list
            with the values of the energy at each time step for a given energy level. It includes all degenerated
            ground states and the first excited state.
        """
        # Each element of the list gaps will correspond to one possible path
        gaps = [[[self._energy_levels[1][0]-self._energy_levels[0][0]], [(0,1)]]]

        for step in range(1, self._n_steps):  # Check paths at each time step
            prev = copy.deepcopy(gaps)  # Copy the information of all paths considered

            for i, k in enumerate(prev):  # Check each proposed path
                current = k[1][-1]
                # If there are larger gaps above, create a new path that starts using that gap
                to_add = self._check_larger_gaps(current, step)
                for new in to_add:
                    gaps.append([ k[0]+[self._energy_levels[new[1]][step] - self._energy_levels[new[0]][step]],
                                k[1]+[new] ])

                gaps[i][0].append(self._energy_levels[current[1]][step]-self._energy_levels[current[0]][step])
                gaps[i][1].append(current)

            # When multiple paths reach the same energy level at any point during the evolution, we only keep the one
            # that gives the best solution and remove all the other ones.
            prev = copy.deepcopy(gaps)
            to_delete=[]
            for i, k in enumerate(prev):
                for j, l in enumerate(prev[i+1:]):
                    if k[1][-1] == l[1][-1]:
                        if mode == 'minimum':
                            # We keep the path with the maximum value of the minimum gap
                            if min(k[0]) < min(l[0]):
                                to_delete.append(i)
                            else:
                                to_delete.append(j+i+1)

                        elif mode == 'integrate':
                            # We keep the path with the largest sum of the square of the instantaneous gaps
                            if sum(g**2 for g in k[0]) < sum(g**2 for g in l[0]):
                                to_delete.append(i)
                            else:
                                to_delete.append(j+i+1)

            to_delete = list(set(to_delete))
            to_delete.sort(reverse = True)
            for index in to_delete:
                del gaps[index]

        # At the end of the evolution we will check which of the paths gives the most optimal result.
        optimal_way = gaps[0]
        if mode == 'minimum':
            for way in gaps:
                if min(way[0]) > min(optimal_way[0]):
                    optimal_way = way

        elif mode == 'integrate':
            for way in gaps:
                if sum(way[0]) > sum(optimal_way[0]):
                    optimal_way = way

        return optimal_way, self._energy_levels

class Method_3(DegeneratedGap):
    """Numerical method that will return the maximum value of the minimum gap. It returns the most optimal solution
    considering the specified gap precision, and the precision of the energy levels data."""
    def __init__(self, energy_levels):
        super(Method_3, self).__init__(energy_levels)

    def _check_larger_gaps(self, current, step):
        raise NotImplementedError()

    def _check_ground_state(self, gap, get_gap_list=False):
        """Checks if the evolution under a given value of the minimum gap ends in the ground state or in an excited
        state.

        Args:
            gap (float): Value of the proposed minimum gap.
            get_gap_list (bool, optional): If True, the method will keep track of the gap at each time step.
            Defaults to False.

        Returns:
            bool: returns True if the final state corresponds to the ground state, otherwise False
            gap_list (List[List[float], List[tuple(int, int)]): List containing two lists: the first one has the values
            of the gap at each time step. The second list contains a set of tuples (lower_level, upper_level) giving
            which pairs of energy levels were used to compute the gap at each time step.
        """
        level = 0
        gap_list = [[], []]

        for step in range(self._n_steps):
            level = self._check_gap(level, step, gap)

            if get_gap_list:
                gap_list[0].append(self._energy_levels[level+1][step]-self._energy_levels[level][step])
                gap_list[1].append((level, level+1))

        if level == self._n_levels-1:
            return False, gap_list
        else:
            return True, gap_list

    def _check_gap(self, level, step, gap):
        """Checks to which energy level can jump the system assuming that it can jump a given gap.

        Args:
            level (int): Initial energy level
            step (int): Time step of the evolution.
            gap (float): Maximum energy gap that the system can jump

        Returns:
            final_level (int): Maximum energy level that the system will reach
        """
        # If the last level (first excited state at the end of the evolution) is reached, we return that energy level
        if level == self._n_levels-1:
            return level

        # If the actual gap is larger than the proposed gap, the system will stay on the same energy level
        elif self._energy_levels[level+1][step] - self._energy_levels[level][step] >= gap:
            return level

        # If the actual gap is smaller than the proposed gap, we keep checking the above energy levels until the
        # it cannot go to a higher level or until the last level is reached.
        else:
            level = level + 1
            level = self._check_gap(level, step, gap)
            return level

    def compute_gap(self, precision, estimation = None):
        """Computes the value of the gap for each time step such that at the end of the evolution we end in one of the
        ground states.

        Args:
            precision (float): precision of the minimum gap.
            estimation (tuple): bounds for minimum gap value as (lower_bound, upper_bound), if the minimum gap is not
            in the specified range, an error will be raised. Defaults to None.

        Returns:
            gap (List[List[float], List[tuple(int, int)]): List containing two lists: the first one has the values of
            the gap at each time step. The second list contains a set of tuples (lower_level, upper_level) giving which
            pairs of energy levels were used to compute the gap at each time step.
            energy_levels (List[List[float(Energy_Value)]]): List of energy levels, each element corresponds to a list
            with the values of the energy at each time step for a given energy level. It includes all degenerated
            ground states and the first excited state.
        """
        # If no estimation of the gap is specified, the energy gap at the end of the evolution will be taken as a bound
        if estimation == None:
            max_gap = self._energy_levels[-1][-1] - self._energy_levels[0][-1]
            min_gap = 0
        # If the minimum gap is not in the estimated range, an error will be raised.
        else:
            max_gap = estimation[1]
            min_gap = estimation[0]
            if (self._check_ground_state(min_gap) == False) or (
                self._check_ground_state(max_gap) == True):
                print('ERROR: The minimum gap is not in the estimation range.')

        # We set the maximum number of iterations depending on the desired precision
        max_iter = int(np.log2((max_gap - min_gap) / precision)) + 1
        for i in range(max_iter):
            # At each step we compute the final state using a gap value between the lower and upper bound
            new_gap = (max_gap - min_gap) / 2. + min_gap

            kept_ground_state, gap_list = self._check_ground_state(new_gap)

            # If with the new gap we end in the ground state, we update the lower bound value with the new gap
            if kept_ground_state == True:
                min_gap = new_gap
            # If with the new gap we end in an excited state, we update the upper bound value with the new gap
            else:
                max_gap = new_gap

            # If we reach the desired precision before finishing the iterations, we return the solution
            if (max_gap - min_gap) < precision:
                _, gap_list = self._check_ground_state(min_gap, get_gap_list=True)
                return gap_list, self._energy_levels

        # Once we have the value of the minimum gap, we compute the gap for all time steps and return the solution
        _, gap_list = self._check_ground_state(min_gap, get_gap_list=True)
        return gap_list, self._energy_levels


    
