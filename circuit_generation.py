import ast
import random
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from graphviz import Digraph

def random_positive_gaussian_integers(mu=0, sigma=1):
    """Generate positive random integers from a Gaussian distribution, ensuring all numbers are within a given range [1, upper_bound].
    
    Args:
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        num_samples (int): Number of samples to generate.
        upper_bound (int): Maximum value of the random integer (inclusive).
    """
    # Generate number, take absolute value, round, and apply upper bound
    number = random.gauss(mu, sigma)
    positive_integer = int(abs(number))
    return positive_integer

def random_expr(depth, max_expr_operators, var_depth):
    """Generate a random qubit index expression using arithmetic, modulus, or simple variables.

    Args:
    depth (int): Number of loop variables.
    max_expr_operators (int): Number of binary operations to perform.
    var_depth (int): Number of additional variables.
    """
    # Generate variable names for loop indices and variables
    loop_vars = [f"i{ind}" for ind in range(depth)]
    vars = ['n']+[f"{ind}" for ind in range(var_depth+1)]

    # All possible variables include 'n' and loop indices
    choices = [ast.Name(id='n', ctx=ast.Load())] + \
              [ast.Name(id=var, ctx=ast.Load()) for var in loop_vars ]

    # Start with a random variable
    # expr = ast.Name(id=random.choice(vars),ctx=ast.Load())
    expr=random.choice(choices)
    # Add binary operations
    for _ in range(max_expr_operators+1):
        left = expr
        right = ast.Name(id=random.choice(vars),ctx=ast.Load())
        op = random.choice([ast.Add(), ast.Sub()])
        expr = ast.BinOp(left=left, op=op, right=right)
    return expr
    # Apply modulo operation to ensure the result is within valid index range

def random_qubit_expr(expr):
    mod_expr = ast.BinOp(left=expr, op=ast.Mod(), right=ast.Name(id='n', ctx=ast.Load()))

    return mod_expr

# Example of using the function
###  num_loops   num_operators  len_var
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

def random_phase_expr(depth):
    """Generate a random phase expression of the form pi * 1 / (2^a + b + c)."""
    loop_vars = [f"i{ind}" for ind in range(depth)]
 
    # Define a
    a = random_expr(depth,depth,0)

    # Define b
    b = random_expr(depth,depth,0)
    # Define c
    c = ast.Constant(value=random_positive_gaussian_integers())
    
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

def loop_index(depth):
    if depth == 1:
        return ast.Name(id='n', ctx=ast.Load())
    else:
        # Generate variable names for loop indices
        vars = [f"i{ind}" for ind in range(depth-1)]
        choices = [ast.Name(id='n', ctx=ast.Load())]+\
        [ast.Name(id=var, ctx=ast.Load()) for var in vars]

        # Start with a random variable
        expr = random.choice(choices)

        # Add binary operations
        for _ in range(depth - 1):
            left = expr
            right = random.choice(choices)
            op = random.choice([ast.Add(), ast.Sub()])
            expr = ast.BinOp(left=left, op=op, right=right)

        # Random value to add/subtract
        value = random.randint(0, depth-1)  # Using a random integer instead of string
        index = ast.BinOp(left=expr, op=random.choice([ast.Add(), ast.Sub()]), 
                          right=ast.Constant(value=value))

        return ast.Call(
            func=ast.Name(id='abs', ctx=ast.Load()),
            args=[index],
            keywords=[])

def set_specific_n(module, n_value):
    print_qubit_indices(module, n_value)

# Define example operations and number of nodes
rotation_gates = ['rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'crx', 'cry', 'crz', 'cp', 'cu1', 'cu3']
multi_qubit_gates = ['cx', 'cz', 'swap', 'ch', 'csx', 'cy', 'ccx', 'cswap', 'cu', 'cp']
three_qubit_gates = ['ccx', 'cswap']  # Toffoli (CCX) gate and Fredkin (CSWAP) gate


def generate_random_circuit_ast(num_nodes, operations, max_loop_depth):
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
        depth = 0
        gate = random.choice(operations)
        if random.choice([True, False]):  # Randomly decide to use a loop or a single operation
            loop_body = []
    
            loop_depth = random.randint(1, max_loop_depth)
            loop_vars = [f"i{ind}" for ind in range(loop_depth)]
            current_body = loop_body  # Initialize current_body to loop_body
            for j in range(loop_depth):
                depth += 1
                loop = ast.For(
                    target=ast.Name(id=loop_vars[depth-1], ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[loop_index(depth)], keywords=[]),  # Use loop_index here
                    body=[],
                    orelse=[]
                )
                current_body.append(loop)  # Append loop to the current body
                current_body = loop.body  # Update current_body to the new loop's body

            choices = [ast.Name(id='n', ctx=ast.Load())] + \
                [ast.Name(id=var, ctx=ast.Load()) for var in loop_vars]
            qubit_index = random.choice(choices)
            ### if multiple qubits
            if gate in multi_qubit_gates:
                target_expr = random_expr(depth, 2, 3)
                target_qubit_index = random_qubit_expr(target_expr)
                ### if rotation
                if gate in rotation_gates:
                    phase = random_phase_expr(depth)
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[phase, qubit_index, target_qubit_index],
                        keywords=[]
                    ))
                else:
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[qubit_index, target_qubit_index],
                        keywords=[]
                    ))
            else:
                if gate in rotation_gates:
                    phase = random_phase_expr(depth)
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[phase, qubit_index],
                        keywords=[]
                    ))
                else:
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[qubit_index],
                        keywords=[]
                    ))

            current_body.append(gate_call)
            body.extend(loop_body)
        else:
            expr = random_expr(depth, 1, 2)
            index = random_qubit_expr(expr)
            if gate in multi_qubit_gates:
                target_expr = random_expr(depth, 1, 2)
                target_qubit_index = random_qubit_expr(target_expr)
                ### if rotation
                if gate in rotation_gates:
                    phase = random_phase_expr(depth)
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[phase, index, target_qubit_index],
                        keywords=[]
                    ))
                else:
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[index, target_qubit_index],
                        keywords=[]
                    ))
            else:
                if gate in rotation_gates:
                    phase = random_phase_expr(depth)
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[phase, index],
                        keywords=[]
                    ))
                else:
                    gate_call = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                        args=[index],
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


if __name__ == "__main__":


    # data = []
    # for i in range (1000):\
    #     data.append(random_positive_gaussian_integers(mu=0, sigma=2))

    # plt.figure(figsize=(10, 6))
    # plt.hist(data, bins=range(0, max(data) + 2), edgecolor='black', alpha=0.75)
    # plt.title('Histogram of 1000 Positive Gaussian Integers')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.xticks(range(0, 11))  
    # plt.grid(True)
    # plt.show()
    # print(random_positive_gaussian_integers())

    # Example usage
    operations = ['h', 'x', 'cx', 'rx', 'ry', 'rz']  # Available quantum gate types
    # Usage
    module = generate_random_circuit_ast(3, operations, 2)
    print(ast.unparse(module))

   
