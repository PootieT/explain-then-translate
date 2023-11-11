import ast
import asttokens
from data.ast.SpanAnnotator import *
from data.ast.PythonProgram import PythonProgram
PYTHON_TOKENS = ["DEDENT", "INDENT", "NEWLINE"]

class PythonSpanAnnotator(SpanAnnotator, PythonProgram):

    def __init__(self, input_program, obfuscate = False, bpe_model = None):
        """ Initialize the span annotator.
        """
        PythonProgram.__init__(self, input_program, obfuscate, bpe_model)
        SpanAnnotator.__init__(self, input_program, obfuscate, bpe_model, call_super=False)

    def get_spans(self):
        """ Get spans for the input program. To be overridden by subclasses.
        """
        assert self.spans
        return self.spans

    def _set_spans(self, prune_spans=True):
        """ Set spans for the input program. To be overridden by subclasses.
        """
        root = asttokens.ASTTokens(self.input_program, parse=True).tree
        spans = set()
        for node in ast.walk(root):
            span = self._get_span(node)
            if span != ():
                spans.add(span)
        self.spans = list(spans)
        if prune_spans:
            self.prune_spans(self.input_program, self.spans)
        return

    @staticmethod
    def _get_span(node: ast.AST):
        if hasattr(node, "first_token"):
            beginning = node.first_token.startpos
            end = node.last_token.endpos
            return (beginning, end)
        else:
            return ()

if __name__ == "__main__":
    prog=\
    """def build_graphviz(node: Node):
    dot = graphviz.Digraph(comment="Program")
    dot.node(str(node.node_id), node.name)
    foo = "bar"
    bar = "foo"
    itr = iter(node)
    n = next(itr)
    for i, child in enumerate(itr):
        dot.node(str(child.node_id), child.name)
        dot.edge(str(child.parent.node_id), str(child.node_id))
    return dot"""
    orig_prog = prog
    prog = PythonSpanAnnotator.processor.tokenize_code(prog)
    print("input is :\n", " ".join(prog))
    annotated = PythonSpanAnnotator(prog).get_annotated_prog_str()
    tokenized = " ".join(PythonSpanAnnotator.processor.tokenize_code(annotated))
    print("tokenized code: \n", tokenized)
    detokenized = PythonSpanAnnotator.processor.detokenize_code(tokenized)
    print("detokenized tokenized code : \n", detokenized)
    print("reconstructed program: \n", PythonSpanAnnotator.processor.detokenize_code(
        " ".join([c for c in tokenized.split() if c not in SpanAnnotator.special_char2char.keys()]) ))

    # node, _, _ = SpanAnnotator.parse_parens_tree_string_split_terminals(tree_string=annotated)
    # dot = build_graphviz(node)
    # dot.render("demo_graphs/python_test_output_split_children.gv", view=True)

    node, _, _ = SpanAnnotator.parse_parens_tree_string(tree_string=annotated)
    dot = build_graphviz(node)
    dot.render("demo_graphs/python_test_output_non_split.gv", view=True)

    obfuscated, _ = PythonSpanAnnotator.processor.obfuscate_code(orig_prog)
    print("obfuscated is : \n", obfuscated)

    annotated = PythonSpanAnnotator(prog, obfuscate=True).get_annotated_prog_str()
    print("obfuscated spanned program: \n", annotated)

    node, _, _ = SpanAnnotator.parse_parens_tree_string(tree_string=annotated)
    dot = build_graphviz(node)
    dot.render("demo_graphs/python_obfuscated_test_output_non_split_children.gv", view=True)

    annotated = PythonSpanAnnotator(prog, obfuscate=True).get_annotated_prog_str(tokenized_style=True)
    print("obfuscated and tokenized style is program: \n", annotated)


    # prog="""def first_missing_positive ( nums ) : NEW_LINE INDENT if len ( nums ) <= 0 : NEW_LINE INDENT return 1 NEW_LINE DEDENT a = 0 NEW_LINE for i in range ( len ( nums ) ) : NEW_LINE INDENT a = nums [ i ] NEW_LINE while a > 0 and a < len ( nums ) and nums [ a - 1 ] != a : NEW_LINE INDENT temp = nums [ a - 1 ] NEW_LINE nums [ a - 1 ] = nums [ i ] NEW_LINE nums [ i ] = temp NEW_LINE a = nums [ i ] NEW_LINE DEDENT DEDENT for i in range ( len ( nums ) ) : NEW_LINE INDENT if nums [ i ] != i + 1 : NEW_LINE INDENT return i + 1 NEW_LINE DEDENT DEDENT return len ( nums ) + 1 NEW_LINE DEDENT"""
    # annotated = PythonSpanAnnotator(prog).get_annotated_prog_str()
    # print(annotated)
    # # #
    # # # node, _, _ = SpanAnnotator.parse_parens_tree_strinimport ast