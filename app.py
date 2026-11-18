import pandas as pd
import plotly.express as px
import streamlit as st
from src.model import run_optimization

st.set_page_config(
    page_title="Resilient Food Distribution Under Disruptions", layout="wide"
)

st.title("Resilient Food Distribution Under Disruptions")
st.write(
    "Independent portfolio optimization dashboard modeling supply chain"
    " resilience under uncertainty."
)

st.sidebar.header("Simulation Settings")
demand_mod = st.sidebar.slider(
    "Demand Multiplier", min_value=0.8, max_value=1.5, value=1.0, step=0.05
)
penalty = st.sidebar.number_input(
    "Shortage Penalty ($/tonne)", min_value=10, max_value=500, value=100
)

if st.button("Run Optimization Model", type="primary"):
  with st.spinner("Solving stochastic model using Pyomo & HiGHS..."):
    res = run_optimization(
        demand_mult_modifier=demand_mod, custom_penalty=penalty
    )

  if res["success"]:
    st.success("Optimization solved successfully!")

    col1, col2 = st.columns(2)
    with col1:
      st.metric(label="Expected Total Cost", value=f"${res['total_cost']:,.2f}")
    with col2:
      opened = [k for k, v in res["opened_centers"].items() if v == 1]
      st.metric(
          label="Centers Selected to Open",
          value=", ".join(opened) if opened else "None",
      )

    st.subheader("Optimal Shipment Plan Across Scenarios")
    if not res["shipments"].empty:
      st.dataframe(res["shipments"])

      fig = px.bar(
          res["shipments"],
          x="Community",
          y="Shipment",
          color="Center",
          facet_col="Scenario",
          title="Optimal Shipments per Community & Scenario",
          barmode="stack",
      )
      st.plotly_chart(fig, use_container_width=True)
    else:
      st.info("No active shipments recorded.")
  else:
    st.error(f"Solver failed with condition: {res['status']}")
else:
  st.info("Click **Run Optimization Model** above to compute the schedule.")