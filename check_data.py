from pathlib import Path
import pandas as pd

data_folder = Path(__file__).parent / "data"

centers = pd.read_csv(data_folder / "centers.csv")
communities = pd.read_csv(data_folder / "communities.csv")
shipping = pd.read_csv(data_folder / "shipping_costs.csv")
scenarios = pd.read_csv(data_folder / "scenarios.csv")

print("Centers:", len(centers))
print("Communities:", len(communities))
print("Base demand:", communities["base_demand_tonnes"].sum(), "tonnes")
print("Scenario probability total:", scenarios["probability"].sum())

assert abs(scenarios["probability"].sum() - 1.0) < 1e-9
print("Data check passed.")