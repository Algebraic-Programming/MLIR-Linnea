from tatsu.model import ModelBuilderSemantics

from tools.frontend.AST_translation import LinneaWalker
#from .parser import LinneaParser

#def parse_input(input):
#    """Parses the input.
#
#    Args:
#        input (string): Description of the input in the Linnea language.
#
#    Returns:
#        Equations: The input equations.
#    """
#    print(input)
#    parser = LinneaParser(semantics=ModelBuilderSemantics())
#    ast = parser.parse(input, rule_name = "model")
#    print(ast)
#
#    walker = LinneaWalker()
#    walker.walk(ast)
#    return walker.equations

def parse_input(input):
  print(input)
