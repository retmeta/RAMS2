import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# Page configuration
st.set_page_config(
    page_title="RET Infrastructure Reliability Analysis",
    layout="wide"
)

st.title("RAMS en SLA van Optical Repeater incl. SLA kosten")
st.markdown("Welkom in deze vergelijking tool. Ik heb hem heel snel gecreeërd en het is niet 100% getest. Het kan zijn dat er fouten ergens :-)")
st.markdown("Hier draait een Monte Carlo simulatie de MTBF en vergelijkt de SLA gegevens die door de gebruiker worden ingevuld.")

# Mathematical Formulas Section
st.header("Wiskundige Formules")

with st.expander(
        "Klik om de wiskundige formules te bekijken die gebruikt worden in de berekeningen"
):
    st.markdown("""
    ### Betrouwbaarheidsanalyse Formules
    
    **1. Storingspercentage (λ):**
    ```
    λ = 1 / MTBF
    ```
    Waarbij:
    - λ = Storingspercentage (storingen per uur)
    - MTBF = Gemiddelde Tijd Tussen Storingen (uren)
    
    **2. Aantal Storingen per Jaar:**
    ```
    Storingen ~ Poisson(λ × Operationele_Uren)
    ```
    Waarbij:
    - Storingen volgen een Poisson distributie
    - Operationele_Uren = 8760 × Seizoensfactor
    - Seizoensfactor = Aantal operationele maanden / 12
    
    **3. Reparatietijd Distributie:**
    ```
    Reparatietijd ~ Exponentieel(MTTR)
    ```
    Waarbij:
    - Reparatietijden volgen een exponentiële distributie
    - MTTR = Gemiddelde Tijd Tot Reparatie (uren)
    
    **4. Totale Downtime per Component:**
    ```
    Totale_Downtime = Σ(Reparatietijd_i) voor i = 1 tot Aantal_Storingen
    ```
    
    **5. Kostenberekening:**
    ```
    Totale_Kosten = Basis_Kosten + (Storingen × Oproepkosten_Geïndexeerd) + (MTTR × Monteur_Tarief_Geïndexeerd)
    ```
    Waarbij:
    - Basis_Kosten = SLA jaarlijkse kosten (gecorrigeerd voor inflatie, indien SLA actief is)
    - Oproepkosten_Geïndexeerd = Vaste kosten per incident × Inflatie_Factor
    - Monteur_Tarief_Geïndexeerd = Kosten per uur monteur tijd × Inflatie_Factor
    
    **6. Systeemtotaal (Alle Componenten):**
    ```
    Systeem_Downtime = Σ(Component_Downtime_i) voor alle componenten
    Systeem_Kosten = Σ(Component_Kosten_i) voor alle componenten
    ```
    
    **7. Monte Carlo Statistieken:**
    ```
    Gemiddelde = (1/n) × Σ(x_i) voor i = 1 tot n
    Standaardafwijking = √[(1/n) × Σ(x_i - Gemiddelde)²]
    Percentiel_95 = Waarde op 95e percentiel van distributie
    ```
    
    **8. Inflatie Correctie:**
    ```
    SLA_Kosten_Gecorrigeerd = SLA_Kosten_Basis × (1 + Inflatie_Factor)^Jaren
    ```
    Waarbij:
    - Inflatie_Factor = Jaarlijks inflatie percentage (als decimaal)
    - Jaren = Aantal jaren vanaf basistijd
    """)

st.markdown("---")

# Sidebar for parameters
st.sidebar.header("Component Instellingen")

# Initialize session state for component data
if 'components' not in st.session_state:
    st.session_state.components = {
        "Repeater": {
            "MTBF": 100000,
            "MTTR_no_SLA": 72,
            "MTTR_with_SLA": 16
        },
        "Voeding/UPS": {
            "MTBF": 50000,
            "MTTR_no_SLA": 30,
            "MTTR_with_SLA": 8
        },
        "Glasvezel": {
            "MTBF": 100000,
            "MTTR_no_SLA": 24,
            "MTTR_with_SLA": 24
        },
        "Omgevingsfactoren": {
            "MTBF": 50000,
            "MTTR_no_SLA": 30,
            "MTTR_with_SLA": 10
        }
    }

# Component parameter controls
components = st.session_state.components.copy()

for component_name in components.keys():
    st.sidebar.subheader(f"{component_name}")

    components[component_name]["MTBF"] = st.sidebar.number_input(
        f"MTBF (uren)",
        min_value=1000,
        max_value=1000000,
        value=components[component_name]["MTBF"],
        step=1000,
        key=f"{component_name}_mtbf")

    components[component_name]["MTTR_no_SLA"] = st.sidebar.number_input(
        f"MTTR zonder SLA (uren)",
        min_value=1,
        max_value=200,
        value=components[component_name]["MTTR_no_SLA"],
        step=1,
        key=f"{component_name}_mttr_no_sla")

    components[component_name]["MTTR_with_SLA"] = st.sidebar.number_input(
        f"MTTR met SLA (uren)",
        min_value=1,
        max_value=100,
        value=components[component_name]["MTTR_with_SLA"],
        step=1,
        key=f"{component_name}_mttr_with_sla")

# Cost parameters
st.sidebar.header("Kosten Parameters")
sla_cost_total = st.sidebar.number_input("Totale SLA kosten (5 jaar) (€)",
                                         min_value=100,
                                         max_value=20000,
                                         value=8665,
                                         step=50)

# Inflation factor
inflation_rate = st.sidebar.number_input("Jaarlijkse inflatie percentage (%)",
                                         min_value=0.0,
                                         max_value=10.0,
                                         value=4.5,
                                         step=0.1,
                                         format="%.1f")

st.sidebar.subheader("Kosten Met SLA")
callout_fee_sla = st.sidebar.number_input(
    "Oproepkosten per incident (Met SLA) (€)",
    min_value=50,
    max_value=500,
    value=175,
    step=25)

engineer_cost_per_hour_sla = st.sidebar.number_input(
    "Monteur kosten per uur (Met SLA) (€)",
    min_value=50,
    max_value=200,
    value=110,
    step=10)

st.sidebar.subheader("Kosten Zonder SLA")
callout_fee_no_sla = st.sidebar.number_input(
    "Oproepkosten per incident (Zonder SLA) (€)",
    min_value=50,
    max_value=500,
    value=100,
    step=1,
    help="Eigen RET mensen, kosten zjin lager")

engineer_cost_per_hour_no_sla = st.sidebar.number_input(
    "Monteur kosten per uur (Zonder SLA) (€)",
    min_value=50,
    max_value=300,
    value=88,
    step=1,
    help="Eigen RET mensen, kosten zjin lager")

# Simulation parameters
st.sidebar.header("Simulatie Parameters")
n_simulations = st.sidebar.number_input("Aantal simulaties",
                                        min_value=1000,
                                        max_value=50000,
                                        value=10000,
                                        step=1000)

# Seasonal operation parameters
st.sidebar.header("Seizoensgebonden Operatie")
seasonal_operation = st.sidebar.checkbox(
    "Alleen operationeel tijdens strandseizoen",
    value=True,
    help=
    "Activeer deze optie als de locatie alleen tijdens het strandseizoen operationeel is"
)

if seasonal_operation:
    season_start_month = st.sidebar.selectbox(
        "Begin strandseizoen (maand)",
        options=[4, 5, 6, 7],
        index=1,  # Default to May
        format_func=lambda x: ["", "", "", "", "April", "Mei", "Juni", "Juli"][
            x])

    season_end_month = st.sidebar.selectbox(
        "Einde strandseizoen (maand)",
        options=[8, 9, 10, 11],
        index=1,  # Default to September
        format_func=lambda x: [
            "", "", "", "", "", "", "", "", "Augustus", "September", "Oktober",
            "November"
        ][x])

    # Calculate seasonal factor (fraction of year operational)
    season_months = season_end_month - season_start_month + 1
    seasonal_factor = season_months / 12

    st.sidebar.info(
        f"Operationeel: {season_months} maanden per jaar ({seasonal_factor:.1%} van het jaar)"
    )
else:
    seasonal_factor = 1.0
    season_months = 12

# Update session state
st.session_state.components = components

# Derived parameters with inflation correction
sla_cost_per_year_base = sla_cost_total / 5
# Apply compound inflation correction for year 3 (middle of 5-year period)
inflation_factor = (1 +
                    inflation_rate / 100)**3  # Compound inflation for year 3
sla_cost_per_year = sla_cost_per_year_base * inflation_factor

# Apply inflation to cost components (different rates for SLA vs non-SLA)
callout_fee_sla_indexed = callout_fee_sla * inflation_factor
engineer_cost_per_hour_sla_indexed = engineer_cost_per_hour_sla * inflation_factor
callout_fee_no_sla_indexed = callout_fee_no_sla * inflation_factor
engineer_cost_per_hour_no_sla_indexed = engineer_cost_per_hour_no_sla * inflation_factor
hours_per_year = 8760


# Simulation function
@st.cache_data
def simulate(components_dict,
             mttr_key,
             sla_active,
             n_sims,
             sla_yearly_cost,
             callout,
             engineer_hourly,
             seasonal_factor=1.0):
    """Run Monte Carlo simulation for infrastructure reliability analysis"""
    total_downtime_dist = []
    total_cost_dist = []

    # Set random seed for reproducibility
    np.random.seed(42)

    # Calculate operational hours based on seasonal factor
    operational_hours = hours_per_year * seasonal_factor

    for _ in range(n_sims):
        total_downtime = 0
        total_cost = sla_yearly_cost if sla_active else 0

        for component_name, vals in components_dict.items():
            MTBF = vals["MTBF"]
            MTTR = vals[mttr_key]

            # Number of failures follows Poisson distribution - adjusted for operational hours
            failures = np.random.poisson(operational_hours / MTBF)

            # Repair times follow exponential distribution
            if failures > 0:
                repair_times = np.random.exponential(MTTR, failures)
                total_downtime += sum(repair_times)
                total_cost += failures * callout + sum(
                    repair_times) * engineer_hourly

        total_downtime_dist.append(total_downtime)
        total_cost_dist.append(total_cost)

    return np.array(total_downtime_dist), np.array(total_cost_dist)


# Run simulation button
if st.sidebar.button("Start Simulatie", type="primary"):
    with st.spinner("Monte Carlo simulatie wordt uitgevoerd..."):
        # Run simulations
        downtime_no_SLA, costs_no_SLA = simulate(
            components, "MTTR_no_SLA", False, n_simulations, sla_cost_per_year,
            callout_fee_no_sla_indexed, engineer_cost_per_hour_no_sla_indexed,
            seasonal_factor)

        downtime_with_SLA, costs_with_SLA = simulate(
            components, "MTTR_with_SLA", True, n_simulations,
            sla_cost_per_year, callout_fee_sla_indexed,
            engineer_cost_per_hour_sla_indexed, seasonal_factor)

        # Store results in session state
        st.session_state.downtime_no_SLA = downtime_no_SLA
        st.session_state.costs_no_SLA = costs_no_SLA
        st.session_state.downtime_with_SLA = downtime_with_SLA
        st.session_state.costs_with_SLA = costs_with_SLA
        st.session_state.simulation_complete = True

# Display results if simulation is complete
if hasattr(st.session_state,
           'simulation_complete') and st.session_state.simulation_complete:

    # Input samenvatting
    st.header("Gebruikte Simulatie Parameters")

    # Show key inputs used in simulation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Component Parameters")
        input_data = []
        for comp_name, comp_data in components.items():
            input_data.append(
                [f"{comp_name} - MTBF", f"{comp_data['MTBF']:,} uren"])
            input_data.append([
                f"{comp_name} - MTTR (zonder SLA)",
                f"{comp_data['MTTR_no_SLA']} uren"
            ])
            input_data.append([
                f"{comp_name} - MTTR (met SLA)",
                f"{comp_data['MTTR_with_SLA']} uren"
            ])

        input_df = pd.DataFrame(input_data, columns=['Parameter', 'Waarde'])
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Kosten Parameters")
        cost_data = [
            ["Aantal simulaties", f"{n_simulations:,}"],
            ["SLA kosten (5 jaar)", f"€{sla_cost_total:,}"],
            [
                "SLA kosten per jaar (geïndexeerd)",
                f"€{sla_cost_per_year:,.2f}"
            ], ["Inflatie percentage (per jaar)", f"{inflation_rate:.1f}%"],
            ["Inflatie factor (3 jaar compound)", f"{inflation_factor:.2f}x"],
            ["Oproepkosten Met SLA (basis)", f"€{callout_fee_sla}"],
            [
                "Oproepkosten Met SLA (geïndexeerd)",
                f"€{callout_fee_sla_indexed:.2f}"
            ], ["Oproepkosten Zonder SLA (basis)", f"€{callout_fee_no_sla}"],
            [
                "Oproepkosten Zonder SLA (geïndexeerd)",
                f"€{callout_fee_no_sla_indexed:.2f}"
            ],
            ["Monteur Met SLA (basis)", f"€{engineer_cost_per_hour_sla}/uur"],
            [
                "Monteur Met SLA (geïndexeerd)",
                f"€{engineer_cost_per_hour_sla_indexed:.2f}/uur"
            ],
            [
                "Monteur Zonder SLA (basis)",
                f"€{engineer_cost_per_hour_no_sla}/uur"
            ],
            [
                "Monteur Zonder SLA (geïndexeerd)",
                f"€{engineer_cost_per_hour_no_sla_indexed:.2f}/uur"
            ],
            [
                "Seizoensgebonden operatie",
                "Ja" if seasonal_operation else "Nee"
            ], ["Operationele periode", f"{season_months} maanden/jaar"],
            ["Seizoensfactor", f"{seasonal_factor:.1%}"]
        ]

        cost_df = pd.DataFrame(cost_data, columns=['Parameter', 'Waarde'])
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

    # Key metrics
    st.header("Belangrijkste Resultaten")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Gem. Downtime (Zonder SLA)",
            f"{np.mean(st.session_state.downtime_no_SLA):.1f} uren",
            f"{np.mean(st.session_state.downtime_no_SLA) - np.mean(st.session_state.downtime_with_SLA):.1f} vs SLA"
        )

    with col2:
        st.metric("Gem. Downtime (Met SLA)",
                  f"{np.mean(st.session_state.downtime_with_SLA):.1f} uren")

    with col3:
        st.metric(
            "Gem. Kosten (Zonder SLA)",
            f"€{np.mean(st.session_state.costs_no_SLA):,.0f}",
            f"€{np.mean(st.session_state.costs_no_SLA) - np.mean(st.session_state.costs_with_SLA):,.0f} vs SLA"
        )

    with col4:
        st.metric("Gem. Kosten (Met SLA)",
                  f"€{np.mean(st.session_state.costs_with_SLA):,.0f}")

    # Most likely downtime (median and mode)
    st.header("Meest Waarschijnlijke Downtime")

    col1, col2, col3, col4 = st.columns(4)

    # Calculate median (50th percentile - most likely middle value)
    median_downtime_no_sla = np.median(st.session_state.downtime_no_SLA)
    median_downtime_with_sla = np.median(st.session_state.downtime_with_SLA)

    # Calculate mode (most frequent value range)
    # Use histogram to find the most common range
    hist_no_sla, bin_edges_no_sla = np.histogram(
        st.session_state.downtime_no_SLA, bins=50)
    mode_bin_no_sla = np.argmax(hist_no_sla)
    mode_downtime_no_sla = (bin_edges_no_sla[mode_bin_no_sla] +
                            bin_edges_no_sla[mode_bin_no_sla + 1]) / 2

    hist_with_sla, bin_edges_with_sla = np.histogram(
        st.session_state.downtime_with_SLA, bins=50)
    mode_bin_with_sla = np.argmax(hist_with_sla)
    mode_downtime_with_sla = (bin_edges_with_sla[mode_bin_with_sla] +
                              bin_edges_with_sla[mode_bin_with_sla + 1]) / 2

    with col1:
        st.metric(
            "Mediaan Downtime (Zonder SLA)",
            f"{median_downtime_no_sla:.1f} uren",
            f"{median_downtime_no_sla - median_downtime_with_sla:.1f} vs SLA")

    with col2:
        st.metric("Mediaan Downtime (Met SLA)",
                  f"{median_downtime_with_sla:.1f} uren")

    with col3:
        st.metric(
            "Meest Voorkomende (Zonder SLA)",
            f"{mode_downtime_no_sla:.1f} uren",
            f"{mode_downtime_no_sla - mode_downtime_with_sla:.1f} vs SLA")

    with col4:
        st.metric("Meest Voorkomende (Met SLA)",
                  f"{mode_downtime_with_sla:.1f} uren")

    # Probability ranges
    st.header("Kansberekening Downtime")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Zonder SLA")
        # Calculate percentiles for probability analysis
        p25_no_sla = np.percentile(st.session_state.downtime_no_SLA, 25)
        p75_no_sla = np.percentile(st.session_state.downtime_no_SLA, 75)
        p90_no_sla = np.percentile(st.session_state.downtime_no_SLA, 90)
        p95_no_sla = np.percentile(st.session_state.downtime_no_SLA, 95)

        prob_data_no_sla = [
            ["25% kans dat downtime ≤", f"{p25_no_sla:.1f} uren"],
            ["50% kans dat downtime ≤", f"{median_downtime_no_sla:.1f} uren"],
            ["75% kans dat downtime ≤", f"{p75_no_sla:.1f} uren"],
            ["90% kans dat downtime ≤", f"{p90_no_sla:.1f} uren"],
            ["95% kans dat downtime ≤", f"{p95_no_sla:.1f} uren"],
            [
                "Meest waarschijnlijke waarde",
                f"{mode_downtime_no_sla:.1f} uren"
            ]
        ]

        prob_df_no_sla = pd.DataFrame(prob_data_no_sla,
                                      columns=['Kans', 'Downtime'])
        st.dataframe(prob_df_no_sla, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Met SLA")
        # Calculate percentiles for probability analysis
        p25_with_sla = np.percentile(st.session_state.downtime_with_SLA, 25)
        p75_with_sla = np.percentile(st.session_state.downtime_with_SLA, 75)
        p90_with_sla = np.percentile(st.session_state.downtime_with_SLA, 90)
        p95_with_sla = np.percentile(st.session_state.downtime_with_SLA, 95)

        prob_data_with_sla = [[
            "25% kans dat downtime ≤", f"{p25_with_sla:.1f} uren"
        ], ["50% kans dat downtime ≤", f"{median_downtime_with_sla:.1f} uren"
            ], ["75% kans dat downtime ≤", f"{p75_with_sla:.1f} uren"
                ], ["90% kans dat downtime ≤", f"{p90_with_sla:.1f} uren"
                    ], ["95% kans dat downtime ≤", f"{p95_with_sla:.1f} uren"],
                              [
                                  "Meest waarschijnlijke waarde",
                                  f"{mode_downtime_with_sla:.1f} uren"
                              ]]

        prob_df_with_sla = pd.DataFrame(prob_data_with_sla,
                                        columns=['Kans', 'Downtime'])
        st.dataframe(prob_df_with_sla,
                     use_container_width=True,
                     hide_index=True)

    # Visualization section
    st.header("Visualisaties")

    # Downtime distribution histogram
    st.subheader("Downtime Verdeling")

    fig_downtime = go.Figure()

    fig_downtime.add_trace(
        go.Histogram(x=st.session_state.downtime_no_SLA,
                     name="Zonder SLA",
                     opacity=0.7,
                     nbinsx=50,
                     marker_color='red'))

    fig_downtime.add_trace(
        go.Histogram(x=st.session_state.downtime_with_SLA,
                     name="Met SLA",
                     opacity=0.7,
                     nbinsx=50,
                     marker_color='green'))

    fig_downtime.update_layout(title="Jaarlijkse Downtime Verdeling",
                               xaxis_title="Downtime (uren)",
                               yaxis_title="Frequentie",
                               barmode='overlay',
                               showlegend=True)

    st.plotly_chart(fig_downtime, use_container_width=True)

    # Cost comparison box plot
    st.subheader("Kosten Vergelijking")

    cost_data = pd.DataFrame({
        'Scenario': ['Zonder SLA'] * len(st.session_state.costs_no_SLA) +
        ['Met SLA'] * len(st.session_state.costs_with_SLA),
        'Cost':
        np.concatenate(
            [st.session_state.costs_no_SLA, st.session_state.costs_with_SLA])
    })

    fig_cost = px.box(cost_data,
                      x='Scenario',
                      y='Cost',
                      title="Jaarlijkse Kosten Verdeling per Scenario",
                      color='Scenario',
                      color_discrete_map={
                          'Zonder SLA': 'red',
                          'Met SLA': 'green'
                      })

    fig_cost.update_layout(yaxis_title="Kosten (€)")
    st.plotly_chart(fig_cost, use_container_width=True)



    # Break-even analysis
    st.header("Break-even Analyse")

    cost_diff = np.mean(st.session_state.costs_no_SLA) - np.mean(
        st.session_state.costs_with_SLA)

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **Financiële Analyse Samenvatting:**
        
        • Gemiddelde jaarlijkse kosten zonder SLA: €{np.mean(st.session_state.costs_no_SLA):,.2f}
        • Gemiddelde jaarlijkse kosten met SLA: €{np.mean(st.session_state.costs_with_SLA):,.2f}
        • Jaarlijks voordeel met SLA: €{cost_diff:,.2f}
        • Inflatie factor toegepast: {inflation_factor:.2f}x
        
        **Conclusie:** {'SLA is financieel voordelig' if cost_diff > 0 else 'SLA is financieel niet voordelig'}
        """)

    with col2:
        # Statistical summary
        st.subheader("Statistische Samenvatting")

        summary_data = {
            'Statistiek':
            ['Gemiddelde', 'Mediaan', 'Std. Afwijking', '95e Percentiel'],
            'Zonder SLA (Downtime)': [
                f"{np.mean(st.session_state.downtime_no_SLA):.1f}u",
                f"{np.median(st.session_state.downtime_no_SLA):.1f}u",
                f"{np.std(st.session_state.downtime_no_SLA):.1f}u",
                f"{np.percentile(st.session_state.downtime_no_SLA, 95):.1f}u"
            ],
            'Met SLA (Downtime)': [
                f"{np.mean(st.session_state.downtime_with_SLA):.1f}u",
                f"{np.median(st.session_state.downtime_with_SLA):.1f}u",
                f"{np.std(st.session_state.downtime_with_SLA):.1f}u",
                f"{np.percentile(st.session_state.downtime_with_SLA, 95):.1f}u"
            ],
            'Zonder SLA (Kosten)': [
                f"€{np.mean(st.session_state.costs_no_SLA):,.0f}",
                f"€{np.median(st.session_state.costs_no_SLA):,.0f}",
                f"€{np.std(st.session_state.costs_no_SLA):,.0f}",
                f"€{np.percentile(st.session_state.costs_no_SLA, 95):,.0f}"
            ],
            'Met SLA (Kosten)': [
                f"€{np.mean(st.session_state.costs_with_SLA):,.0f}",
                f"€{np.median(st.session_state.costs_with_SLA):,.0f}",
                f"€{np.std(st.session_state.costs_with_SLA):,.0f}",
                f"€{np.percentile(st.session_state.costs_with_SLA, 95):,.0f}"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    # Download section
    st.header("Download Resultaten")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Create CSV with simulation results
        results_df = pd.DataFrame({
            'Simulatie':
            range(1,
                  len(st.session_state.downtime_no_SLA) + 1),
            'Downtime_Zonder_SLA':
            st.session_state.downtime_no_SLA,
            'Kosten_Zonder_SLA':
            st.session_state.costs_no_SLA,
            'Downtime_Met_SLA':
            st.session_state.downtime_with_SLA,
            'Kosten_Met_SLA':
            st.session_state.costs_with_SLA
        })

        csv = results_df.to_csv(index=False)
        st.download_button(label="Download Simulatie Resultaten (CSV)",
                           data=csv,
                           file_name="simulatie_resultaten.csv",
                           mime="text/csv")

    with col2:
        # Create summary report
        summary_report = f"""
RAMS en SLA van Optical Repeater - Analyse Rapport
=================================================

Simulatie Parameters:
- Aantal simulaties: {n_simulations:,}
- SLA kosten (5 jaar): €{sla_cost_total:,}
- SLA kosten (per jaar): €{sla_cost_per_year:,.2f}
- Inflatie factor (3 jaar compound): {inflation_factor:.2f}x
- Inflatie percentage (per jaar): {inflation_rate:.1f}%

Kosten Met SLA:
- Oproepkosten (basis): €{callout_fee_sla}
- Oproepkosten (geïndexeerd): €{callout_fee_sla_indexed:.2f}
- Monteur kosten per uur (basis): €{engineer_cost_per_hour_sla}
- Monteur kosten per uur (geïndexeerd): €{engineer_cost_per_hour_sla_indexed:.2f}

Kosten Zonder SLA:
- Oproepkosten (basis): €{callout_fee_no_sla}
- Oproepkosten (geïndexeerd): €{callout_fee_no_sla_indexed:.2f}
- Monteur kosten per uur (basis): €{engineer_cost_per_hour_no_sla}
- Monteur kosten per uur (geïndexeerd): €{engineer_cost_per_hour_no_sla_indexed:.2f}

Seizoensgebonden Operatie:
- Operationeel: {"Ja" if seasonal_operation else "Nee"}
- Periode: {season_months} maanden per jaar
- Seizoensfactor: {seasonal_factor:.1%}

Component Configuratie:
"""
        for comp_name, comp_data in components.items():
            summary_report += f"""
{comp_name}:
  - MTBF: {comp_data['MTBF']:,} uren
  - MTTR (zonder SLA): {comp_data['MTTR_no_SLA']} uren
  - MTTR (met SLA): {comp_data['MTTR_with_SLA']} uren
"""

        summary_report += f"""

Resultaten Samenvatting:
=======================

Downtime Analyse:
- Gemiddelde downtime zonder SLA: {np.mean(st.session_state.downtime_no_SLA):.1f} uren
- Gemiddelde downtime met SLA: {np.mean(st.session_state.downtime_with_SLA):.1f} uren
- Downtime reductie met SLA: {np.mean(st.session_state.downtime_no_SLA) - np.mean(st.session_state.downtime_with_SLA):.1f} uren

Kosten Analyse:
- Gemiddelde jaarlijkse kosten zonder SLA: €{np.mean(st.session_state.costs_no_SLA):,.2f}
- Gemiddelde jaarlijkse kosten met SLA: €{np.mean(st.session_state.costs_with_SLA):,.2f}
- Jaarlijks voordeel met SLA: €{cost_diff:,.2f}

Conclusie:
{('SLA is financieel voordelig' if cost_diff > 0 else 'SLA is financieel niet voordelig')}
"""

        st.download_button(label="Download Samenvatting Rapport (TXT)",
                           data=summary_report,
                           file_name="analyse_rapport.txt",
                           mime="text/plain")

    with col3:
        # Download sensitivity analysis
        sensitivity_csv = sensitivity_df.to_csv(index=False)
        st.download_button(label="Download Gevoeligheidsanalyse (CSV)",
                           data=sensitivity_csv,
                           file_name="gevoeligheidsanalyse.csv",
                           mime="text/csv")

else:
    st.info(
        "Configureer parameters in de sidebar en klik op 'Start Simulatie' om de analyse te beginnen."
    )

    # Show component configuration table
    st.subheader("Huidige Component Configuratie")

    config_data = []
    for comp_name, comp_data in components.items():
        config_data.append({
            'Component': comp_name,
            'MTBF (uren)': f"{comp_data['MTBF']:,}",
            'MTTR zonder SLA (uren)': comp_data['MTTR_no_SLA'],
            'MTTR met SLA (uren)': comp_data['MTTR_with_SLA']
        })

    config_df = pd.DataFrame(config_data)
    st.dataframe(config_df, use_container_width=True)

    # Show methodology
    st.subheader("Methodologie")
    st.markdown("""
    Deze applicatie voert Monte Carlo simulatie uit om IT infrastructuur betrouwbaarheid en kosten te analyseren:
    
    **Simulatie Proces:**
    1. Voor elke component worden storingen gemodelleerd met een Poisson distributie gebaseerd op MTBF
    2. Reparatietijden volgen een exponentiële distributie gebaseerd op MTTR
    3. Totale downtime en kosten worden berekend voor elke simulatie run
    4. Statistische analyse wordt uitgevoerd op de resultaten
    
    **Kostenberekening:**
    - Vaste SLA kosten (jaarlijks, gecorrigeerd voor inflatie)
    - Variabele kosten: oproepkosten + monteur uren × uurtarief
    
    **Belangrijke Aannames:**
    - 8.760 uren per jaar
    - Onafhankelijke component storingen
    - Exponentiële reparatietijd distributie
    - Poisson storings distributie
    """)

    # Input summary section
    st.subheader("Samenvatting Gebruikte Inputs")

    # Create input summary table
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Component Parameters:**")
        input_summary_components = []
        for comp_name, comp_data in components.items():
            input_summary_components.append({
                'Component':
                comp_name,
                'MTBF':
                f"{comp_data['MTBF']:,} uren",
                'MTTR (zonder SLA)':
                f"{comp_data['MTTR_no_SLA']} uren",
                'MTTR (met SLA)':
                f"{comp_data['MTTR_with_SLA']} uren"
            })

        input_summary_df = pd.DataFrame(input_summary_components)
        st.dataframe(input_summary_df,
                     use_container_width=True,
                     hide_index=True)

    with col2:
        st.markdown("**Kosten & Simulatie Parameters:**")
        cost_summary = pd.DataFrame({
            'Parameter': [
                'Totale SLA kosten (5 jaar)', 'SLA kosten per jaar (basis)',
                'SLA kosten per jaar (met inflatie)',
                'Inflatie percentage (per jaar)',
                'Inflatie factor (3 jaar compound)',
                'Oproepkosten Met SLA (basis)',
                'Oproepkosten Met SLA (geïndexeerd)',
                'Oproepkosten Zonder SLA (basis)',
                'Oproepkosten Zonder SLA (geïndexeerd)',
                'Monteur Met SLA (basis)', 'Monteur Met SLA (geïndexeerd)',
                'Monteur Zonder SLA (basis)',
                'Monteur Zonder SLA (geïndexeerd)',
                'Seizoensgebonden operatie', 'Operationele periode',
                'Aantal simulaties'
            ],
            'Waarde': [
                f"€{sla_cost_total:,}", f"€{sla_cost_per_year_base:,.2f}",
                f"€{sla_cost_per_year:,.2f}", f"{inflation_rate:.1f}%",
                f"{inflation_factor:.2f}x", f"€{callout_fee_sla}",
                f"€{callout_fee_sla_indexed:.2f}", f"€{callout_fee_no_sla}",
                f"€{callout_fee_no_sla_indexed:.2f}",
                f"€{engineer_cost_per_hour_sla}/uur",
                f"€{engineer_cost_per_hour_sla_indexed:.2f}/uur",
                f"€{engineer_cost_per_hour_no_sla}/uur",
                f"€{engineer_cost_per_hour_no_sla_indexed:.2f}/uur",
                f"{'Ja' if seasonal_operation else 'Nee'}",
                f"{season_months} maanden/jaar", f"{n_simulations:,}"
            ]
        })
        st.dataframe(cost_summary, use_container_width=True, hide_index=True)
