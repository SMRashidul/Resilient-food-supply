from pathlib import Path
import pandas as pd
import pyomo.environ as pyo


def run_optimization(demand_mult_modifier=1.0, custom_penalty=100):
  # Find the main project folder, then load the CSV files from data/
  project_folder = Path(__file__).resolve().parents[1]
  data_folder = project_folder / "data"

  centers = pd.read_csv(data_folder / "centers.csv")
  communities = pd.read_csv(data_folder / "communities.csv")
  shipping = pd.read_csv(data_folder / "shipping_costs.csv")
  scenarios = pd.read_csv(data_folder / "scenarios.csv")

  center_names = centers["center"].tolist()
  community_names = communities["community"].tolist()
  scenario_names = scenarios["scenario"].tolist()

  capacity = centers.set_index("center")["capacity_tonnes"].to_dict()
  fixed_cost = centers.set_index("center")["fixed_opening_cost"].to_dict()
  base_demand = (
      communities.set_index("community")["base_demand_tonnes"].to_dict()
  )

  shipping_cost = {
      (row.center, row.community): row.cost_per_tonne
      for row in shipping.itertuples(index=False)
  }

  probability = scenarios.set_index("scenario")["probability"].to_dict()
  demand_multiplier = {
      k: v * demand_mult_modifier
      for k, v in scenarios.set_index("scenario")["demand_multiplier"]
      .to_dict()
      .items()
  }

  disrupted_center = {}
  for row in scenarios.itertuples(index=False):
    value = row.disrupted_center
    if pd.isna(value) or str(value).strip().lower() in {"", "none", "nan"}:
      disrupted_center[row.scenario] = None
    else:
      disrupted_center[row.scenario] = str(value).strip()

  shortage_penalty = custom_penalty

  # Create the optimization model
  model = pyo.ConcreteModel()

  model.J = pyo.Set(initialize=center_names)
  model.I = pyo.Set(initialize=community_names)
  model.S = pyo.Set(initialize=scenario_names)

  model.open_center = pyo.Var(model.J, domain=pyo.Binary)
  model.ship = pyo.Var(model.J, model.I, model.S, domain=pyo.NonNegativeReals)
  model.unmet = pyo.Var(model.I, model.S, domain=pyo.NonNegativeReals)

  def demand_rule(model, community, scenario):
    scenario_demand = base_demand[community] * demand_multiplier[scenario]
    return (
        sum(
            model.ship[center, community, scenario] for center in model.J
        )
        + model.unmet[community, scenario]
        >= scenario_demand
    )

  model.meet_demand = pyo.Constraint(model.I, model.S, rule=demand_rule)

  def capacity_rule(model, center, scenario):
    is_disrupted = disrupted_center[scenario] == center
    available_capacity = 0 if is_disrupted else capacity[center]
    return (
        sum(
            model.ship[center, community, scenario] for community in model.I
        )
        <= available_capacity * model.open_center[center]
    )

  model.respect_capacity = pyo.Constraint(
      model.J, model.S, rule=capacity_rule
  )

  def total_cost_rule(model):
    opening_costs = sum(
        fixed_cost[center] * model.open_center[center] for center in model.J
    )
    expected_scenario_costs = sum(
        probability[scenario]
        * (
            sum(
                shipping_cost[(center, community)]
                * model.ship[center, community, scenario]
                for center in model.J
                for community in model.I
            )
            + shortage_penalty
            * sum(
                model.unmet[community, scenario] for community in model.I
            )
        )
        for scenario in model.S
    )
    return opening_costs + expected_scenario_costs

  model.total_cost = pyo.Objective(
      rule=total_cost_rule,
      sense=pyo.minimize,
  )

  solver = pyo.SolverFactory("highs")
  result = solver.solve(model)

  status = str(result.solver.termination_condition)
  if result.solver.termination_condition != pyo.TerminationCondition.optimal:
    return {"status": status, "success": False}

  opened_centers = {
      center: int(pyo.value(model.open_center[center]) > 0.5)
      for center in model.J
  }

  shipment_records = []
  for scenario in model.S:
    for center in model.J:
      for community in model.I:
        amt = pyo.value(model.ship[center, community, scenario])
        if amt > 0.001:
          shipment_records.append({
              "Scenario": scenario,
              "Center": center,
              "Community": community,
              "Shipment": round(amt, 2),
          })

  return {
      "success": True,
      "status": status,
      "total_cost": round(pyo.value(model.total_cost), 2),
      "opened_centers": opened_centers,
      "shipments": pd.DataFrame(shipment_records),
  }