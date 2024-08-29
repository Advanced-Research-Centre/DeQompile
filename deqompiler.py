import ast
import random
from sympy import *

rotation_gates = ['rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'crx', 'cry', 'crz', 'cp', 'cu1', 'cu3']
multi_qubit_gates = ['cx', 'cz', 'swap', 'ch', 'csx', 'cy', 'ccx', 'cswap', 'cu', 'cp']
three_qubit_gates = ['ccx', 'cswap']
loop_prob = 0.99

def random_expr(curr_loop_depth, max_expr_operators, var_depth):
    """
    Generate a random expression using arithmetic, modulus, or simple variables.
    This expr is used both for qubit indexing and phase

    Args:
        depth (int): Number of loop variables.
        max_expr_operators (int): Number of binary operations to perform.
        var_depth (int): Number of additional variables.
    """
    # Generate variable names for loop indices and variables
    loop_vars = [f"i{ind}" for ind in range(curr_loop_depth)]
    vars = ['n']+[f"{ind}" for ind in range(var_depth+1)]

    # All possible variables include 'n' and loop indices
    choices = [ast.Name(id='n', ctx=ast.Load())] + [ast.Name(id=var, ctx=ast.Load()) for var in loop_vars ]

    # Start with a random variable
    # expr = ast.Name(id=random.choice(vars),ctx=ast.Load())
    expr = random.choice(choices)
    # Add binary operations
    for _ in range(max_expr_operators+1):
        left = expr
        right = ast.Name(id=random.choice(vars),ctx=ast.Load())
        op = random.choice([ast.Add(), ast.Sub()])
        expr = ast.BinOp(left=left, op=op, right=right)
    return expr    

def random_qubit_expr(curr_loop_depth, num_ops, num_vars):
    
    qubit_expr = random_expr(curr_loop_depth, num_ops, num_vars)
    mod_qubit_expr = ast.BinOp(left=qubit_expr, op=ast.Mod(), right=ast.Name(id='n', ctx=ast.Load()))   # Apply modulo operation to ensure the result is within valid index range
    sympl_qubit_expr = ast.parse(str(simplify(ast.unparse(mod_qubit_expr)))).body[0].value
    # print("Simplified expression:",ast.unparse(mod_qubit_expr),"--->",ast.unparse(sympl_qubit_expr))
    return sympl_qubit_expr

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

def random_phase_expr(depth):
    """
    Generate a random phase expression of the form pi * 1 / (2**a + b + c).
    """
 
    a = random_expr(depth,depth,0)
    b = random_expr(depth,depth,0)
    c = ast.Constant(value=random_positive_gaussian_integers())
    
    # Create the expression 2**a + b + c
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

    # Create the expression pi * 1 / (2**a + b + c)
    phase_expr = ast.BinOp(
        left=ast.Name(id='pi', ctx=ast.Load()),
        op=ast.Mult(),
        right=ast.BinOp(
            left=ast.Constant(value=1),
            op=ast.Div(),
            right=expr_inner
        )
    )

    sympl_phase_expr = ast.parse(str(simplify(ast.unparse(phase_expr)))).body[0].value
    return sympl_phase_expr

def loop_index(depth):
    if depth == 0:
        return ast.Name(id='n', ctx=ast.Load())
    else:
        # Generate variable names for loop indices
        vars = [f"i{ind}" for ind in range(depth)]
        choices = [ast.Name(id='n', ctx=ast.Load())]+\
        [ast.Name(id=var, ctx=ast.Load()) for var in vars]

        # Start with a random variable
        expr = random.choice(choices)

        # Add binary operations
        for _ in range(depth):
            left = expr
            right = random.choice(choices)
            op = random.choice([ast.Add(), ast.Sub()])
            expr = ast.BinOp(left=left, op=op, right=right)

        # Random value to add/subtract
        value = random.randint(0, depth)  # Using a random integer instead of string
        index = ast.BinOp(left=expr, op=random.choice([ast.Add(), ast.Sub()]), 
                          right=ast.Constant(value=value))

        return ast.Call(
            func=ast.Name(id='abs', ctx=ast.Load()),
            args=[index],
            keywords=[])
    
def random_qiskit_ast_gate(operations, curr_loop_depth, num_ops, num_vars):
    gate = random.choice(operations)
    index1 = random_qubit_expr(curr_loop_depth, num_ops, num_vars)

    if gate in multi_qubit_gates:
        index2 = random_qubit_expr(curr_loop_depth, num_ops, num_vars)
        if gate in rotation_gates:
            phase = random_phase_expr(curr_loop_depth)
            gate_call = ast.Expr(value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                args=[phase, index1, index2],
                keywords=[]
            ))
        else:
            gate_call = ast.Expr(value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                args=[index1, index2],
                keywords=[]
            ))
    else:
        if gate in rotation_gates:
            phase = random_phase_expr(curr_loop_depth)
            gate_call = ast.Expr(value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                args=[phase, index1],
                keywords=[]
            ))
        else:
            gate_call = ast.Expr(value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="qc", ctx=ast.Load()), attr=gate, ctx=ast.Load()),
                args=[index1],
                keywords=[]
            ))
    return gate_call

def random_qiskit_ast_body(body, operations, num_nodes, max_loop_depth, curr_loop_depth = 0):
    """
    Generate a random quantum algorithm body.
    operations: List of quantum operations to choose from.
    num_nodes: Number of lines in level 0. for loops (along with their body) are counted as 1 line.
    max_loop_depth: Maximum depth of nested loops.
    """

    for i in range(num_nodes):
        if random.random() < loop_prob:       # Randomly decide node is a loop or a gate operation

            loop_body = []    
            loop_depth = random.randint(1, max_loop_depth)
            loop_vars = [f"i{ind}" for ind in range(loop_depth)]
            current_body = loop_body  # Initialize current_body to loop_body
            for j in range(loop_depth):
                loop = ast.For(
                    target=ast.Name(id=loop_vars[j], ctx=ast.Store()),
                    iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[loop_index(j)], keywords=[]),  # Use loop_index here
                    body=[],
                    orelse=[]
                )
                current_body.append(loop)   # Append loop to the current body
                current_body = loop.body    # Update current_body to the new loop's body

            current_body.append(random_qiskit_ast_gate(operations, loop_depth, 2, 3))
            body.extend(loop_body)
            
        else:
            body.append(random_qiskit_ast_gate(operations, curr_loop_depth, 1, 2))

    return body

def random_qiskit_ast_generator(operations, num_nodes, max_loop_depth):

    import_def = [
        ast.ImportFrom(module='qiskit', names=[ast.alias(name='QuantumCircuit', asname=None)], level=0),
        ast.ImportFrom(module='math', names=[ast.alias(name='pi', asname=None)], level=0),
        ast.Import(names=[ast.alias(name='numpy', asname='np')]),
        ast.Import(names=[ast.alias(name='random', asname=None)])
    ]
    
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

    body = random_qiskit_ast_body(body, operations, num_nodes, max_loop_depth)
    
    body.append(ast.Return(value=ast.Name(id="qc", ctx=ast.Load())))

    function_def = ast.FunctionDef(
        name="quantum_algorithm",
        args=args,
        body=body,
        decorator_list=[],          # Decorators
        returns=None,               # Annotate function's return type
        type_comment=None
    )

    module = ast.Module(body=[import_def, function_def], type_ignores=[])
    ast.fix_missing_locations(module)
    return module

if __name__ == "__main__":

    operations = ['h', 'x', 'cx', 'rx', 'ry', 'rz']
    module = random_qiskit_ast_generator(operations=operations, num_nodes=2, max_loop_depth=2)    # max_loop_depth=0 sometimes gives error
    print(ast.unparse(module))
    # print()
    # pprint(ast.dump(module, indent=4))