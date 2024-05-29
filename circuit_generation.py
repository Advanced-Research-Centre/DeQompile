import ast
import random
import time
from tqdm import tqdm
from graphviz import Digraph

def random_qubit_expr(n, include_i=False):
    """Generate a random qubit index expression using arithmetic, modulus, or simple variables."""
    if random.choice([True, False]):
        if include_i and random.choice([True, False]):
            return ast.Name(id='i', ctx=ast.Load())
        else:
            return ast.Name(id='n-1', ctx=ast.Load())
    else:
        operators = [ast.Add(), ast.Sub(), ast.Mult()]
        operands = [ast.Name(id='n', ctx=ast.Load()), ast.BinOp(left=ast.Name(id='n', ctx=ast.Load()), right=ast.Constant(value=1), op=ast.Sub())]
        if include_i:
            operands.append(ast.Name(id='i', ctx=ast.Load()))

        expr = ast.BinOp(
            left=random.choice(operands),
            op=random.choice(operators),
            right=random.choice(operands)
        )

        mod_expr = ast.BinOp(
            left=expr,
            op=ast.Mod(),
            right=ast.Name(id='n', ctx=ast.Load())
        )

        return mod_expr

def random_phase_expr(include_i=False):
    """Generate a random phase expression of the form pi * 1 / (2^a + b + c)."""
    # Define a
    a = random.choice([ast.Name(id='i', ctx=ast.Load()), ast.Constant(value=0)]) if include_i else ast.Constant(value=0)
    
    # Define b
    b = random.choice([ast.Name(id='n', ctx=ast.Load()), ast.Name(id='i', ctx=ast.Load()), ast.Constant(value=0)]) if include_i else random.choice([ast.Name(id='n', ctx=ast.Load()), ast.Constant(value=0)])
    
    # Define c
    c = ast.Constant(value=random.randint(0, 4))
    
    # Create the expression 2^a + b + c
    expr_inner = ast.BinOp(
        left=ast.BinOp(
            left=ast.Constant(value=2),
            op=ast.Pow(),
            right=a
        ),
        op=ast.Add(),
        right=ast.BinOp(
            left=b,
            op=ast.Add(),
            right=c
        )
    )
    
    # Create the expression pi * 1 / (2^a + b + c)
    phase_expr = ast.BinOp(
        left=ast.Name(id='pi', ctx=ast.Load()),
        op=ast.Mult(),
        right=ast.BinOp(
            left=ast.Constant(value=1),
            op=ast.Div(),
            right=expr_inner
        )
    )
    
    return phase_expr

def generate_random_circuit_ast(num_nodes, operations):
    args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg='n', annotation=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )
    
    body = [
        ast.Assign(
            targets=[ast.Name(id="qc", ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id='QuantumCircuit', ctx=ast.Load()),
                args=[ast.Name(id='n', ctx=ast.Load())],
                keywords=[]
            )
        )
    ]
    
    for i in range(num_nodes):
        gate = random.choice(operations)

        if random.choice([True, False]):  # Randomly decide to use a loop or a single operation
            qubit_index_expr = random_qubit_expr('n', include_i=True)  # Index for loop context
            loop_body = []
            if gate == 'cx':
                target_qubit_index_expr = random_qubit_expr('n', include_i=True)
                gate_call = ast.Expr(value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                    args=[qubit_index_expr, target_qubit_index_expr],
                    keywords=[]
                ))
            else:
                if gate in ['rx', 'ry', 'rz']:
                    phase_expr = random_phase_expr(include_i=True)
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[phase_expr, qubit_index_expr],
                        keywords=[]
                    ))
                else:
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[qubit_index_expr],
                        keywords=[]
                    ))
            loop_body.append(gate_call)

            loop = ast.For(
                target=ast.Name(id='i', ctx=ast.Store()),
                iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Name(id='n', ctx=ast.Load())], keywords=[]),
                body=loop_body,
                orelse=[]
            )
            body.append(loop)
        else:
            qubit_index_expr = random_qubit_expr('n', include_i=False)  # Index for non-loop context
            if gate == 'cx':
                target_qubit_index_expr = random_qubit_expr('n', include_i=False)
                gate_call = ast.Expr(value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                    args=[qubit_index_expr, target_qubit_index_expr],
                    keywords=[]
                ))
            else:
                if gate in ['rx', 'ry', 'rz']:
                    phase_expr = random_phase_expr(include_i=False)
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[phase_expr, qubit_index_expr],
                        keywords=[]
                    ))
                else:
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[qubit_index_expr],
                        keywords=[]
                    ))
            body.append(gate_call)

    body.append(ast.Return(value=ast.Name(id="qc", ctx=ast.Load())))

    function_def = ast.FunctionDef(
        name="generate_random_circuit_ast",
        args=args,
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None
    )

    module = ast.Module(body=[function_def], type_ignores=[])
    ast.fix_missing_locations(module)
    return module

def print_qubit_indices(node, n_value):
    """
    Traverse the AST and print each qubit index expression, evaluating them if possible,
    or replacing 'n' with a specific value.
    """
    class IndexPrinter(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and node.func.attr in ['x', 'h', 'cx', 'rx', 'ry', 'rz']:
                for arg in node.args:
                    print(self.evaluate_expr(arg, n_value))
            self.generic_visit(node)

        def evaluate_expr(self, expr, n_value):
            if isinstance(expr, ast.Name) and expr.id == 'i':
                return 'i'  # Return 'i' directly
            try:
                compiled_expr = compile(ast.Expression(expr), filename="<ast>", mode="eval")
                return str(eval(compiled_expr, {"n": n_value, "i": "i", "pi": 3.141592653589793}))
            except Exception as e:
                return f"Error evaluating expression: {e}"

    visitor = IndexPrinter()
    visitor.visit(node)

def set_specific_n(module, n_value):
    print_qubit_indices(module, n_value)


if __name__ == "__main__":
    # Example usage
    operations = ['h', 'x', 'cx', 'rx', 'ry', 'rz']  # Available quantum gate types
    circuit_ast = generate_random_circuit_ast(5, operations)
    code = ast.unparse(circuit_ast)
    print(code)
    set_specific_n(circuit_ast, 2)