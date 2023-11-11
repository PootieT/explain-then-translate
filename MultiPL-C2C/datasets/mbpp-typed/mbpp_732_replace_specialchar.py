from typing import List, Dict, Tuple

def replace_specialchar(text: str) -> str:
    """
	Write a function to replace all occurrences of spaces, commas, or dots with a colon.
	"""
    ### Canonical solution below ###
    pass

### Unit tests below ###
def check(candidate):
    assert candidate('Python language, Programming language.') == 'Python:language::Programming:language:'
    assert candidate('a b c,d e f') == 'a:b:c:d:e:f'
    assert candidate('ram reshma,ram rahim') == 'ram:reshma:ram:rahim'

def test_check():
    check(replace_specialchar)

