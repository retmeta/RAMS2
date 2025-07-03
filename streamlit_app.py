import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(
    page_title="RET Infrastructure Reliability Analysis",
    layout="wide"
)

st.title("RAMS en SLA van Optical Repeater incl. SLA kosten")
st.markdown("Welkom in deze vergelijking tool. Ik heb hem heel snel gecreeërd en het is niet 100% getest. Het kan zijn dat er fouten ergens :-)")
st.markdown("Hier draait een Monte Carlo simulatie de MTBF en vergelijkt de SLA gegevens die door de gebruiker worden ingevuld.")

# --- Helper Functions ---
@st.cache_data
def simulate(
    components: dict,
    mttr_key: str,
    sla_active: bool,
    n_sims: int,
    sla_yearly_cost: float,
    callout: float,
    engineer_hourly: float,
    seasonal_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    hours_per_year = 8760 * seasonal_factor

    total_downtime = []
    total_cost = []

    for _ in range(n_sims):
        downtime = 0
        cost = sla_yearly_cost if sla_active else 0

        for name, params in components.items():
            mtbf = params["MTBF"]
            mttr = params[mttr_key]

            num_failures = np.random.poisson(hours_per_year / mtbf)

            if num_failures > 0:
                repair_times = np.random.exponential(mttr, num_failures)
                downtime += repair_times.sum()
                cost += num_failures * callout + repair_times.sum() * engineer_hourly

        total_downtime.append(downtime)
        total_cost.append(cost)

    return np.array(total_downtime), np.array(total_cost)

# --- Sidebar Inputs ---
st.sidebar.header("Simulatie Parameters")

if 'components' not in st.session_state:
    st.session_state.components = {
        "Repeater": {"MTBF": 100_000, "MTTR_no_SLA": 48, "MTTR_with_SLA": 12},
        "Voeding/UPS": {"MTBF": 50_000, "MTTR_no_SLA": 48, "MTTR_with_SLA": 8},
        "Glasvezel": {"MTBF": 100_000, "MTTR_no_SLA": 24, "MTTR_with_SLA": 24},
        "Omgevingsfactoren": {"MTBF": 50_000, "MTTR_no_SLA": 36, "MTTR_with_SLA": 10}
    }

components = st.session_state.components

for name, params in components.items():
    st.sidebar.subheader(name)
    params["MTBF"] = st.sidebar.number_input(f"{name} - MTBF (uren)", 1000, 1_000_000, params["MTBF"], step=1000)
    params["MTTR_no_SLA"] = st.sidebar.number_input(f"{name} - MTTR zonder SLA (uren)", 1, 200, params["MTTR_no_SLA"])
    params["MTTR_with_SLA"] = st.sidebar.number_input(f"{name} - MTTR met SLA (uren)", 1, 100, params["MTTR_with_SLA"])

sla_cost_total = st.sidebar.number_input("Totale SLA kosten (5 jaar) (€)", 100, 20_000, 8_665, step=50)
inflation_rate = st.sidebar.number_input("Inflatie per jaar (%)", 0.0, 10.0, 4.5, step=0.1)

n_simulations = st.sidebar.number_input("Aantal simulaties", 1_000, 50_000, 10_000, step=1_000)

# --- Run Simulation ---
if st.sidebar.button("Start Simulatie", type="primary"):
    with st.spinner("Monte Carlo simulatie wordt uitgevoerd..."):
        inflation_factor = (1 + inflation_rate / 100) ** 3
        sla_cost_per_year = (sla_cost_total / 5) * inflation_factor

        downtime_no_sla, costs_no_sla = simulate(
            components, "MTTR_no_SLA", False, n_simulations,
            sla_cost_per_year, callout=250, engineer_hourly=150
        )
        downtime_with_sla, costs_with_sla = simulate(
            components, "MTTR_with_SLA", True, n_simulations,
            sla_cost_per_year, callout=175, engineer_hourly=110
        )

        st.session_state["results"] = {
            "downtime_no_sla": downtime_no_sla,
            "costs_no_sla": costs_no_sla,
            "downtime_with_sla": downtime_with_sla,
            "costs_with_sla": costs_with_sla
        }

# --- Display Results ---
if "results" in st.session_state:
    res = st.session_state["results"]

    st.header("Gebruikte Simulatie Parameters")
    param_data = []
    for comp_name, comp in components.items():
        param_data.append([comp_name, "MTBF (uren)", f"{comp['MTBF']:,}"])
        param_data.append([comp_name, "MTTR zonder SLA (uren)", f"{comp['MTTR_no_SLA']}"])
        param_data.append([comp_name, "MTTR met SLA (uren)", f"{comp['MTTR_with_SLA']}"])
    param_df = pd.DataFrame(param_data, columns=["Component", "Parameter", "Waarde"])
    st.dataframe(param_df, use_container_width=True, hide_index=True)

    cost_params = [
        ["SLA kosten (5 jaar)", f"€{sla_cost_total:,.2f}"],
        ["SLA kosten per jaar (geïndexeerd)", f"€{sla_cost_per_year:,.2f}"],
        ["Inflatie per jaar (%)", f"{inflation_rate}%"],
        ["Inflatie factor (3 jaar)", f"{inflation_factor:.2f}x"]
    ]
    cost_df = pd.DataFrame(cost_params, columns=["Parameter", "Waarde"])
    st.dataframe(cost_df, use_container_width=True, hide_index=True)

    st.header("Wiskundige Formules")
    st.markdown(r"""
    **Aantal Storingen (Poisson):**
    $$
    N \sim \text{Poisson}\left(\frac{8760}{\text{MTBF}}\right)
    $$

    **Hersteltijd per Storing (Exponential):**
    $$
    T_{herstel} \sim \text{Exponential}(\text{MTTR})
    $$

    **Totale Downtime:**
    $$
    D_{totaal} = \sum_{i=1}^{N} T_{\text{herstel},i}
    $$

    **Totale Kosten:**
    $$
    C_{totaal} = \text{SLA-kosten} + \left(N \cdot \text{Callout-kosten}\right) + \left(D_{totaal} \cdot \text{Engineer-uurloon}\right)
    $$
    """)

    st.header("Belangrijkste Resultaten")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gem. Downtime (Zonder SLA)", f"{np.mean(res['downtime_no_sla']):.1f} uren")
        st.metric("Gem. Kosten (Zonder SLA)", f"€{np.mean(res['costs_no_sla']):,.0f}")
    with col2:
        st.metric("Gem. Downtime (Met SLA)", f"{np.mean(res['downtime_with_sla']):.1f} uren")
        st.metric("Gem. Kosten (Met SLA)", f"€{np.mean(res['costs_with_sla']):,.0f}")

    st.header("Meest Waarschijnlijke Downtime")
    col1, col2 = st.columns(2)
    with col1:
        hist, bin_edges = np.histogram(res['downtime_no_sla'], bins=50)
        mode_bin = np.argmax(hist)
        mode_downtime = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
        median_downtime = np.median(res['downtime_no_sla'])
        st.metric("Mediaan Downtime (Zonder SLA)", f"{median_downtime:.1f} uren")
        st.metric("Meest Voorkomende Downtime (Zonder SLA)", f"{mode_downtime:.1f} uren")
    with col2:
        hist, bin_edges = np.histogram(res['downtime_with_sla'], bins=50)
        mode_bin = np.argmax(hist)
        mode_downtime = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
        median_downtime = np.median(res['downtime_with_sla'])
        st.metric("Mediaan Downtime (Met SLA)", f"{median_downtime:.1f} uren")
        st.metric("Meest Voorkomende Downtime (Met SLA)", f"{mode_downtime:.1f} uren")

    st.header("Kansberekening Downtime")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Zonder SLA")
        percentiles = [25, 50, 75, 90, 95]
        data_no_sla = [[f"{p}%", f"{np.percentile(res['downtime_no_sla'], p):.1f} uren"] for p in percentiles]
        df_no_sla = pd.DataFrame(data_no_sla, columns=["Kans", "Downtime"])
        st.dataframe(df_no_sla, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Met SLA")
        data_with_sla = [[f"{p}%", f"{np.percentile(res['downtime_with_sla'], p):.1f} uren"] for p in percentiles]
        df_with_sla = pd.DataFrame(data_with_sla, columns=["Kans", "Downtime"])
        st.dataframe(df_with_sla, use_container_width=True, hide_index=True)

    st.header("Statistische Samenvatting")
    summary_df = pd.DataFrame({
        'Statistiek': ['Gemiddelde', 'Mediaan', 'Standaardafwijking', '95e Percentiel'],
        'Downtime Zonder SLA': [
            f"{np.mean(res['downtime_no_sla']):.1f}u",
            f"{np.median(res['downtime_no_sla']):.1f}u",
            f"{np.std(res['downtime_no_sla']):.1f}u",
            f"{np.percentile(res['downtime_no_sla'], 95):.1f}u",
        ],
        'Downtime Met SLA': [
            f"{np.mean(res['downtime_with_sla']):.1f}u",
            f"{np.median(res['downtime_with_sla']):.1f}u",
            f"{np.std(res['downtime_with_sla']):.1f}u",
            f"{np.percentile(res['downtime_with_sla'], 95):.1f}u",
        ],
        'Kosten Zonder SLA': [
            f"€{np.mean(res['costs_no_sla']):,.0f}",
            f"€{np.median(res['costs_no_sla']):,.0f}",
            f"€{np.std(res['costs_no_sla']):,.0f}",
            f"€{np.percentile(res['costs_no_sla'], 95):,.0f}",
        ],
        'Kosten Met SLA': [
            f"€{np.mean(res['costs_with_sla']):,.0f}",
            f"€{np.median(res['costs_with_sla']):,.0f}",
            f"€{np.std(res['costs_with_sla']):,.0f}",
            f"€{np.percentile(res['costs_with_sla'], 95):,.0f}",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.header("Downtime Verdeling")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=res['downtime_no_sla'], name="Zonder SLA", opacity=0.6, marker_color="red"))
    fig.add_trace(go.Histogram(x=res['downtime_with_sla'], name="Met SLA", opacity=0.6, marker_color="green"))
    fig.update_layout(
        barmode='overlay',
        xaxis_title="Downtime (uren)",
        yaxis_title="Frequentie",
        legend=dict(x=0.7, y=0.95)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("Kosten Vergelijking")
    df_costs = pd.DataFrame({
        "Scenario": ["Zonder SLA"] * len(res['costs_no_sla']) + ["Met SLA"] * len(res['costs_with_sla']),
        "Kosten": np.concatenate([res['costs_no_sla'], res['costs_with_sla']])
    })
    fig_box = px.box(df_costs, x="Scenario", y="Kosten", color="Scenario",
                     color_discrete_map={"Zonder SLA": "red", "Met SLA": "green"},
                     points="all")
    fig_box.update_layout(yaxis_title="Kosten (€)")
    st.plotly_chart(fig_box, use_container_width=True)

    st.header("Download Resultaten")
    csv = pd.DataFrame({
        'Downtime_Zonder_SLA': res['downtime_no_sla'],
        'Kosten_Zonder_SLA': res['costs_no_sla'],
        'Downtime_Met_SLA': res['downtime_with_sla'],
        'Kosten_Met_SLA': res['costs_with_sla']
    }).to_csv(index=False)

    txt_summary = summary_df.to_string(index=False)

    st.download_button("Download Simulatie Resultaten (CSV)", csv, "simulatie_resultaten.csv", "text/csv")
    st.download_button("Download Samenvatting (TXT)", txt_summary, "simulatie_samenvatting.txt", "text/plain")
