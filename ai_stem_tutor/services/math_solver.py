from sympy import symbols, Eq, solve, diff, integrate, sympify, sin, cos, tan, expand_trig, simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers.solveset import linsolve
from typing import List, Tuple

class MathSolver:
    @staticmethod
    def solve_problem(problem: str) -> Tuple[List[str], str]:
        steps = []
        answer = None
        try:
            # Handle systems of equations (comma-separated)
            if problem.count('=') > 1:
                eqs = [parse_expr(eq.replace('=', '-(') + ")") for eq in problem.split(',')]
                vars = set()
                for eq in eqs:
                    vars.update(eq.free_symbols)
                vars = list(vars)
                steps.append(f"Parsed system of equations: {eqs}")
                sol = linsolve(eqs, vars)
                steps.append(f"Solving for {vars}: {sol}")
                answer = str(sol)
            # Try to parse as an equation to solve
            elif '=' in problem:
                left, right = problem.split('=')
                left_expr = parse_expr(left.strip())
                right_expr = parse_expr(right.strip())
                eq = Eq(left_expr, right_expr)
                steps.append(f"Parsed equation: {eq}")
                vars = eq.free_symbols
                if vars:
                    var = vars.pop()
                else:
                    var = symbols('x')
                # Trigonometric equation detection
                if any(str(f) in str(eq) for f in [sin(var), cos(var), tan(var)]):
                    steps.append("Detected trigonometric equation. Expanding and simplifying...")
                    eq = Eq(expand_trig(left_expr), expand_trig(right_expr))
                    steps.append(f"Expanded equation: {eq}")
                sol = solve(eq, var)
                steps.append(f"Solving for {var}: {sol}")
                answer = str(sol)
            # Try to parse as a derivative
            elif 'd/d' in problem or 'derivative' in problem.lower():
                var = symbols('x')
                expr_str = problem.replace('d/dx', '').replace('derivative', '').strip()
                expr = parse_expr(expr_str)
                steps.append(f"Parsed expression for derivative: {expr}")
                deriv = diff(expr, var)
                steps.append(f"Step 1: Differentiate with respect to {var}")
                steps.append(f"Result: {deriv}")
                answer = str(deriv)
            # Try to parse as an integral
            elif 'integrate' in problem.lower() or '∫' in problem:
                var = symbols('x')
                expr_str = problem.replace('integrate', '').replace('∫', '').replace('dx', '').strip()
                expr = parse_expr(expr_str)
                steps.append(f"Parsed expression for integration: {expr}")
                steps.append(f"Step 1: Integrate with respect to {var}")
                integ = integrate(expr, var)
                steps.append(f"Result: {integ} + C")
                answer = str(integ) + ' + C'
            else:
                expr = parse_expr(problem)
                steps.append(f"Parsed expression: {expr}")
                val = expr.evalf()
                steps.append(f"Evaluated value: {val}")
                answer = str(val)
        except Exception as e:
            steps.append(f"Error: {str(e)}")
            answer = "Could not solve the problem. Please check your input."
        return steps, answer
