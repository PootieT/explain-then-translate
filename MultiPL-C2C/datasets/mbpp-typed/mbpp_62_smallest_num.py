from typing import List, Dict, Tuple

def smallest_num(xs: List[int]) -> int:
    """
	Write a python function to find smallest number in a list.
	"""
    ### Canonical solution below ###
    pass

### Unit tests below ###
def check(candidate):
    assert candidate([10, 20, 1, 45, 99]) == 1
    assert candidate([1, 2, 3]) == 1
    assert candidate([45, 46, 50, 60]) == 45

def test_check():
    check(smallest_num)

