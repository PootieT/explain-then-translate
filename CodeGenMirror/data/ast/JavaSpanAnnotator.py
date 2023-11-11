from data.ast.SpanAnnotator import *
import subprocess
from data.ast.JavaProgram import JavaProgram

# JAVA_HOME="/usr/local/opt/openjdk/bin/"
#JAVA_HOME=""
# PATH_TO_JAVA_SPAN_ANNOTATOR="/Users/Alex/Documents/jp_span_analyzer_maven/main.jar"
JAVA_HOME="/home/shypula/java_8/ibm-java-ppc64le-80/bin/"
PATH_TO_JAVA_SPAN_ANNOTATOR="/home/shypula/CodeGenMirror/data/ast/java_span.jar"

class JavaSpanAnnotator(SpanAnnotator, JavaProgram):

    def __init__(self, input_program, obfuscate = False, bpe_model = None):
        super().__init__(input_program=input_program, obfuscate=obfuscate, bpe_model=bpe_model)

    def get_spans(self):
        """ Get spans for the input program.
        """
        assert self.spans
        return self.spans

    def _set_spans(self, prune_spans=True):
        """ Set spans for the input program. Relies on javaparser jar file at path.
        """
        proc = subprocess.run([f"{JAVA_HOME}java",  "-jar",  PATH_TO_JAVA_SPAN_ANNOTATOR,
                        "--program_string", self.input_program],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError("Could not annotate program with javaparser with error {} and program string {}".
                               format(proc.stderr, self.input_program))
        else:
            span_string = proc.stdout.decode("utf-8")
            self.spans = self._parse_span_string(span_string)
            if prune_spans:
                self.spans = self.prune_spans(self.input_program, self.spans)
        return

    @staticmethod
    def _parse_span_string(span_string):
        spans = span_string.strip().split("\n")
        spans = [span.strip().split(" ") for span in spans]
        spans = [(int(span[0]), int(span[1])+1) for span in spans]
        return list(set(spans))


if __name__ == "__main__":
    prog=\
    """
    class BinarySearchStripped {
         int binarySearchStripped(int arr[], int l, int r, int x)
         {
             if (r >= l) {
                 String bar = "testing testing 123"; 
                 int mid = l + (r - l) / 2;
                 if (arr[mid] == x)
                     return mid;
                 if (arr[mid] > x)
                     return binarySearchStripped(arr, l, mid - 1, x);
                 return binarySearchStripped(arr, mid + 1, r, x);
             }
             return -1;
         }
    }"""
    # annotated = JavaSpanAnnotator(prog).get_annotated_prog_str()
    # print(annotated)
    #
    # res, dico = JavaSpanAnnotator.processor.obfuscate_code(prog)
    # print("obfuscated prog is :", res)

    # node, _, _ = SpanAnnotator.parse_parens_tree_string_split_terminals(tree_string=annotated)
    # dot = build_graphviz(node)
    # dot.render("demo_graphs/java_test_output_split_children.gv", view=True)

    # node, _, _ = SpanAnnotator.parse_parens_tree_string(tree_string=annotated)
    # dot = build_graphviz(node)
    # dot.render("demo_graphs/java_test_output_non_split.gv", view=True)

    # prog = "static void test_ci_neg ( int [ ] a , float [ ] b ) { for ( int i = a . length - 1 ; i >= 0 ; i -= 1 ) { a [ i ] = - 123 ; b [ i ] = - 103.f ; } }"
    prog = "public static String capitalize ( String s ) { if ( ( char ) s . charAt ( 0 ) >= ' a ' ) { return ( String ) ( ( char ) ( s . charAt ( 0 ) + ( ' A ' - ' a ' ) ) + s . substring ( 1 ) ) ; } else { return s ; } }"

    print("input program is: \n", prog)
    annotated = JavaSpanAnnotator(prog).get_annotated_prog_str(tokenized_style=True)
    print(annotated)

    node, _, _ = SpanAnnotator.parse_parens_tree_string(tree_string=annotated)
    dot = build_graphviz(node)
    dot.render("demo_graphs/ci_neg_without_parent_class.gv", view=True)

