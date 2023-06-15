from polaris import TestGroup
from polaris.ocean.tests.manufactured_solution.space_convergence import (
    SpaceConvergence,
)
from polaris.ocean.tests.manufactured_solution.time_convergence import (
    TimeConvergence,
)


class ManufacturedSolution(TestGroup):
    """
    A test group for manufactured solution test cases
    """
    def __init__(self, component):
        """
        component : polaris.ocean.Ocean
            the ocean component that this test group belongs to
        """
        super().__init__(component=component, name='manufactured_solution')

        self.add_test_case(SpaceConvergence(test_group=self))
        self.add_test_case(TimeConvergence(test_group=self))
