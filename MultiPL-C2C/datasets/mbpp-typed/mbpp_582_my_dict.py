from typing import List, Dict, Tuple

def my_dict(dict1: Set[int]) -> bool:
    """
	Write a function to check if a dictionary is empty
	"""
    ### Canonical solution below ###
    pass

### Unit tests below ###
def check(candidate):
    assert candidate({10}) == False
    assert candidate({11}) == False
    assert candidate({}) == True

def test_check():
    check(my_dict)

