import pandas
import streamlit
import plotly

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    SolverFactory,
    NonNegativeReals,
    value,
)

model = ConcreteModel()
model.x = Var(domain=NonNegativeReals)
model.minimum = Constraint(expr=model.x >= 5)
model.cost = Objective(expr=model.x)

result = SolverFactory("highs").solve(model)

print("Packages imported successfully.")
print("Solver status:", result.solver.termination_condition)
print("Test solution:", value(model.x))