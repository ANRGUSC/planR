import cvxpy as cp

# Define the CVXPY variables for the state (I) and the action (A)
I = cp.Variable()
A = cp.Variable()

# Define the constants
const1, const2 = 0.005, 0.01
community_risk = 0.8  # Example community risk value

# System dynamics in CVXPY
I_next = const1 * I * A + const2 * community_risk * A**2

# Define the Lyapunov function in CVXPY
a = cp.Variable(1, nonneg=True)
V = a * I**2

# Compute derivative of V (approximated as a discrete difference)
V_next = a * I_next**2
V_dot = V_next - V

# Optimization problem in CVXPY
constraints = [V_dot <= 0]

# Solve the problem using the SCS solver
prob = cp.Problem(cp.Minimize(0), constraints)
prob.solve(solver=cp.SCS)

if prob.status == cp.OPTIMAL:
    print(f"Found Lyapunov function: V(I) = {a.value[0]} * I^2")
else:
    print("No feasible solution found.")
