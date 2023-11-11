from typing import Dict, List
import graphviz
from collections.abc import MutableMapping
from collections import defaultdict
from typing import Tuple, Union
from pprint import pprint


ANSIcolor2number={'black': 0,
                 'red': 1,
                 'green': 2,
                 'yellow': 3,
                 'blue': 4,
                 'magenta': 5,
                 'cyan': 6,
                 'white': 7,
                 'default': 9}


def color_substr_with_span(string, substring_span, backgr_color: str = "red", text_color: str = "black"):
    assert backgr_color in ANSIcolor2number
    assert text_color in ANSIcolor2number
    beginning, end = substring_span
    first_string = string[0:beginning]
    substr_to_highlight = string[beginning:end]
    last_string = string[end:]
    backgr_code = ANSIcolor2number[backgr_color]
    text_color = ANSIcolor2number[text_color]
    # \33[  starts the coloring and \33[ ends
    formatted_substr = f"\033[4{backgr_code};3{text_color}m{substr_to_highlight}\033[m"
    return first_string + formatted_substr + last_string


class Node:
    def __init__(self, name: str = "", children: List = None, node_id: int = None, parent = None):
        self.name = name
        assert children is None or isinstance(children, list)
        self.children = children if children else []
        self.parent = parent
        self.node_id = node_id
    def add_name(self, name):
        self.name = name
    def add_child(self, child):
        self.children.append(child)
    def increment_name(self, char: str):
        self.name += char
    def add_parent(self, parent):
        self.parent = parent
    def add_node_id(self, node_id):
        self.node_id = node_id

    def __iter__(self):
        yield self
        for child in self.children:
            yield from iter(child)


def build_graphviz(node: Node):
    dot = graphviz.Digraph(comment="Program")
    dot.node(str(node.node_id), node.name)
    itr = iter(node)
    n = next(itr)
    for child in itr:
        dot.node(str(child.node_id), child.name)
        dot.edge(str(child.parent.node_id), str(child.node_id))
    return dot


class PositionDict(MutableMapping):
    """
    PositionDict is a dictionary that takes in an iterable of spans.
    It maps all spanned indices to the spans that encompass them.
    """

    def __init__(self, span_list: Union[List, Tuple]):
        self.store = defaultdict(list)
        for span in span_list:
            self.add_span(span)

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return repr(dict(self.store))

    def __str__(self):
        return str(dict(self.store))

    def _keytransform(self, key):
        return key

    def add_span(self, span: Tuple[int]):
        if len(span) == 0:
            return
        assert len(span) == 2 and span[1] >= span[0]
        # unpack span into range
        length = span[1] - span[0]
        for i in range(span[0], span[1] + 1):
            self.store[i].append((length, span))
        return

    def pprint(self):
        pprint(dict(self.store))