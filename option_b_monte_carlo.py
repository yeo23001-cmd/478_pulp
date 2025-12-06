"""
Option B: Monte Carlo Simulation for Uncertainty Analysis
The Goal - Production Optimization

Demonstrates what Python can do that Excel Solver cannot:
- Run 1,000+ optimization scenarios with uncertainty
- Probability distributions of outcomes
- Risk analysis and confidence intervals
- Interactive uncertainty exploration
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from the_goal_optimization import create_goal_optimization_model

# Page configuration
st.set_page_config(
    page_title="Monte Carlo Simulation - The Goal",
    page_icon="🎲",
    layout="wide"
)

# Header
st.markdown("""
<div style="background-color: #1e3a8a; color: white; padding: 20px; margin-bottom: 20px; border-radius: 8px;">
    <h1 style="margin: 0;">🎲 The Goal: Monte Carlo Simulation</h1>
    <p style="margin: 5px 0 0 0; font-style: italic;">Account for Uncertainty in Your Optimization</p>
</div>
""", unsafe_allow_html=True)

# Excel vs Python callout
st.info("""
**💡 Why This Matters:** In the real world, demand, capacity, and prices are uncertain. Excel Solver gives you ONE answer 
assuming everything is certain. Python can run 1,000+ scenarios to show you the RANGE of possible outcomes and their probabilities!
""")

# Sidebar - Configuration
st.sidebar.header("🎲 Simulation Settings")

st.sidebar.markdown("""
**Monte Carlo Simulation:** Runs optimization hundreds of times with random variations in parameters to account for uncertainty.
""")

st.sidebar.markdown("---")

# Number of simulations
n_simulations = st.sidebar.slider(
    "Number of Simulations:",
    min_value=100,
    max_value=2000,
    value=500,
    step=100,
    help="More simulations = more accurate probability estimates (but slower)"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Base Case Parameters")

# Base parameters
base_heat_treatment = st.sidebar.slider("Heat Treatment Capacity:", 80, 240, 160, 10)
base_machining = st.sidebar.slider("Machining Capacity:", 100, 300, 200, 10)
base_assembly = st.sidebar.slider("Assembly Capacity:", 100, 300, 180, 10)
base_demand_a = st.sidebar.slider("Expected Demand - Product A:", 0, 100, 50, 5)
base_demand_b = st.sidebar.slider("Expected Demand - Product B:", 0, 150, 80, 5)
base_profit_a = st.sidebar.slider("Expected Profit - Product A ($):", 50, 150, 90, 5)
base_profit_b = st.sidebar.slider("Expected Profit - Product B ($):", 30, 100, 60, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Uncertainty Levels")

st.sidebar.markdown("**How much variation do you expect?**")

# Uncertainty parameters (as % of base)
demand_uncertainty = st.sidebar.slider(
    "Demand Uncertainty (%):",
    min_value=0,
    max_value=50,
    value=20,
    step=5,
    help="Demand can vary ±20% from expected"
)

capacity_uncertainty = st.sidebar.slider(
    "Capacity Uncertainty (%):",
    min_value=0,
    max_value=30,
    value=10,
    step=5,
    help="Capacity can vary ±10% due to breakdowns, maintenance, etc."
)

price_uncertainty = st.sidebar.slider(
    "Price Uncertainty (%):",
    min_value=0,
    max_value=40,
    value=15,
    step=5,
    help="Prices can vary ±15% due to market conditions"
)

st.sidebar.markdown("---")

# Run simulation button
if st.sidebar.button("🎲 Run Simulation", type="primary", use_container_width=True):
    st.session_state['run_simulation'] = True

if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.clear()
    st.rerun()

# Run baseline for comparison
baseline_results, _ = create_goal_optimization_model(
    heat_treatment_capacity=base_heat_treatment,
    machining_capacity=base_machining,
    assembly_capacity=base_assembly,
    demand_a=base_demand_a,
    demand_b=base_demand_b,
    profit_a=base_profit_a,
    profit_b=base_profit_b
)

# Main content
if 'run_simulation' not in st.session_state:
    # Instructions before simulation
    st.subheader("📋 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **1. Set Base Parameters** (left sidebar)
        - Your expected/planned values
        - Best guess for demand, capacity, prices
        
        **2. Define Uncertainty** (left sidebar)
        - How much could each parameter vary?
        - Demand uncertainty: ±20% is typical
        - Capacity uncertainty: ±10% for breakdowns
        - Price uncertainty: ±15% for market changes
        
        **3. Run Simulation**
        - Click "Run Simulation" button
        - Python runs optimization 500+ times
        - Each time with different random values
        - Shows you range of possible outcomes
        """)
    
    with col2:
        st.markdown("""
        **Questions This Answers:**
        
        ✅ What's the probability we achieve $5,000+ throughput?
        
        ✅ What's our worst-case scenario?
        
        ✅ What's our best-case scenario?
        
        ✅ How much risk are we taking?
        
        ✅ Should we build safety margins?
        
        ✅ What's the 95% confidence interval?
        """)
    
    # Show baseline
    st.markdown("---")
    st.subheader("📊 Baseline Scenario (No Uncertainty)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Expected Throughput",
            f"${baseline_results['total_throughput']:,.2f}",
            help="Assumes all parameters at expected values"
        )
    
    with col2:
        st.metric(
            "Product A",
            f"{baseline_results['product_a']:.1f} units"
        )
    
    with col3:
        st.metric(
            "Product B",
            f"{baseline_results['product_b']:.1f} units"
        )
    
    st.warning("⚠️ **Reality Check:** This baseline assumes perfect certainty. But demand fluctuates, machines break down, and prices change. Click 'Run Simulation' to see the range of possible outcomes!")

else:
    # Run Monte Carlo simulation
    st.subheader(f"🎲 Running {n_simulations:,} Simulations...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Storage for results
    simulation_results = []
    
    # Run simulations
    for i in range(n_simulations):
        # Update progress
        if i % 50 == 0:
            progress_bar.progress(i / n_simulations)
            status_text.text(f"Simulation {i}/{n_simulations}")
        
        # Generate random variations (normal distribution)
        # Demand
        demand_a = max(0, np.random.normal(base_demand_a, base_demand_a * demand_uncertainty / 100))
        demand_b = max(0, np.random.normal(base_demand_b, base_demand_b * demand_uncertainty / 100))
        
        # Capacity (can't go negative)
        heat_treatment = max(50, np.random.normal(base_heat_treatment, base_heat_treatment * capacity_uncertainty / 100))
        machining = max(50, np.random.normal(base_machining, base_machining * capacity_uncertainty / 100))
        assembly = max(50, np.random.normal(base_assembly, base_assembly * capacity_uncertainty / 100))
        
        # Prices
        profit_a = max(10, np.random.normal(base_profit_a, base_profit_a * price_uncertainty / 100))
        profit_b = max(10, np.random.normal(base_profit_b, base_profit_b * price_uncertainty / 100))
        
        # Solve optimization
        result, _ = create_goal_optimization_model(
            heat_treatment_capacity=heat_treatment,
            machining_capacity=machining,
            assembly_capacity=assembly,
            demand_a=demand_a,
            demand_b=demand_b,
            profit_a=profit_a,
            profit_b=profit_b
        )
        
        # Store results
        simulation_results.append({
            'throughput': result['total_throughput'],
            'product_a': result['product_a'],
            'product_b': result['product_b'],
            'bottleneck': result['bottleneck'],
            'ht_utilization': result['heat_treatment_utilization']
        })
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed {n_simulations:,} simulations!")
    
    # Convert to DataFrame
    df = pd.DataFrame(simulation_results)
    
    # Calculate statistics
    mean_throughput = df['throughput'].mean()
    std_throughput = df['throughput'].std()
    min_throughput = df['throughput'].min()
    max_throughput = df['throughput'].max()
    percentile_5 = df['throughput'].quantile(0.05)
    percentile_95 = df['throughput'].quantile(0.95)
    median_throughput = df['throughput'].median()
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Simulation Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean Throughput",
            f"${mean_throughput:,.2f}",
            delta=f"{mean_throughput - baseline_results['total_throughput']:,.2f}",
            help="Average across all simulations"
        )
    
    with col2:
        st.metric(
            "Worst Case (5th %ile)",
            f"${percentile_5:,.2f}",
            delta=f"{percentile_5 - baseline_results['total_throughput']:,.2f}",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Best Case (95th %ile)",
            f"${percentile_95:,.2f}",
            delta=f"{percentile_95 - baseline_results['total_throughput']:,.2f}"
        )
    
    with col4:
        st.metric(
            "Risk (Std Dev)",
            f"${std_throughput:,.2f}",
            help="Higher = more uncertainty"
        )
    
    # Distribution chart
    st.markdown("---")
    st.subheader("📈 Throughput Distribution")
    
    fig_dist = go.Figure()
    
    # Histogram
    fig_dist.add_trace(go.Histogram(
        x=df['throughput'],
        nbinsx=50,
        name='Simulated Outcomes',
        marker_color='#3b82f6',
        opacity=0.7
    ))
    
    # Add baseline line
    fig_dist.add_vline(
        x=baseline_results['total_throughput'],
        line_dash="dash",
        line_color="red",
        annotation_text="Baseline (no uncertainty)",
        annotation_position="top"
    )
    
    # Add mean line
    fig_dist.add_vline(
        x=mean_throughput,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean: ${mean_throughput:,.0f}",
        annotation_position="top right"
    )
    
    fig_dist.update_layout(
        title=f'Distribution of Throughput ({n_simulations:,} simulations)',
        xaxis_title='Throughput ($)',
        yaxis_title='Frequency',
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Probability analysis
    st.markdown("---")
    st.subheader("🎯 Probability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Probability of exceeding targets
        # Initialize target if not in session state
        if 'target_throughput' not in st.session_state:
            st.session_state.target_throughput = int(baseline_results['total_throughput'])
        
        target_input = st.number_input(
            "Target Throughput ($):",
            min_value=0,
            max_value=int(max_throughput),
            value=st.session_state.target_throughput,
            step=100,
            key='target_input_widget'
        )
        
        # Update session state when user changes it
        st.session_state.target_throughput = target_input
        
        prob_exceed = (df['throughput'] >= target_input).mean() * 100
        prob_below = (df['throughput'] < target_input).mean() * 100
        
        st.markdown(f"""
        **Probability Analysis for ${target_input:,}:**
        - ✅ Probability of **achieving or exceeding**: **{prob_exceed:.1f}%**
        - ⚠️ Probability of **falling short**: **{prob_below:.1f}%**
        """)
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_exceed,
            title={'text': f"Probability ≥ ${target_input:,}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#16a34a" if prob_exceed >= 50 else "#dc2626"},
                'steps': [
                    {'range': [0, 50], 'color': "#fee2e2"},
                    {'range': [50, 80], 'color': "#fef3c7"},
                    {'range': [80, 100], 'color': "#dcfce7"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Confidence intervals
        st.markdown("**Confidence Intervals:**")
        
        confidence_levels = [50, 80, 90, 95, 99]
        ci_data = []
        
        for conf in confidence_levels:
            lower = (100 - conf) / 2
            upper = conf + lower
            lower_bound = df['throughput'].quantile(lower / 100)
            upper_bound = df['throughput'].quantile(upper / 100)
            ci_data.append({
                'Confidence': f"{conf}%",
                'Lower Bound': f"${lower_bound:,.2f}",
                'Upper Bound': f"${upper_bound:,.2f}",
                'Range': f"${upper_bound - lower_bound:,.2f}"
            })
        
        ci_df = pd.DataFrame(ci_data)
        st.dataframe(ci_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        **Interpretation:**
        - 95% confidence: Throughput will be between ${percentile_5:,.2f} and ${percentile_95:,.2f}
        - Range: ${percentile_95 - percentile_5:,.2f}
        """)
    
    # Product mix analysis
    st.markdown("---")
    st.subheader("📦 Product Mix Variation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_prod_a = px.histogram(
            df,
            x='product_a',
            nbins=30,
            title='Product A Distribution',
            labels={'product_a': 'Units Produced'}
        )
        fig_prod_a.update_traces(marker_color='#10b981')
        st.plotly_chart(fig_prod_a, use_container_width=True)
    
    with col2:
        fig_prod_b = px.histogram(
            df,
            x='product_b',
            nbins=30,
            title='Product B Distribution',
            labels={'product_b': 'Units Produced'}
        )
        fig_prod_b.update_traces(marker_color='#f59e0b')
        st.plotly_chart(fig_prod_b, use_container_width=True)
    
    # Scatter plot - relationship
    st.markdown("---")
    st.subheader("🔍 Product Mix Relationship")
    
    fig_scatter = px.scatter(
        df,
        x='product_a',
        y='product_b',
        color='throughput',
        title='Product Mix vs. Throughput',
        labels={'product_a': 'Product A (units)', 'product_b': 'Product B (units)', 'throughput': 'Throughput ($)'},
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    # Calculate some interesting stats
    prob_above_baseline = (df['throughput'] > baseline_results['total_throughput']).mean() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"""
        **Risk Assessment:**
        - Mean throughput is ${mean_throughput - baseline_results['total_throughput']:+,.2f} vs. baseline
        - {prob_above_baseline:.1f}% chance of exceeding baseline
        - Worst case: ${percentile_5:,.2f} ({(percentile_5/baseline_results['total_throughput']-1)*100:+.1f}%)
        - Best case: ${percentile_95:,.2f} ({(percentile_95/baseline_results['total_throughput']-1)*100:+.1f}%)
        """)
    
    with col2:
        # Bottleneck frequency
        bottleneck_counts = df['bottleneck'].value_counts()
        most_common_bottleneck = bottleneck_counts.index[0]
        bottleneck_pct = (bottleneck_counts.iloc[0] / len(df)) * 100
        
        st.info(f"""
        **Bottleneck Analysis:**
        - Most common bottleneck: **{most_common_bottleneck}**
        - Frequency: **{bottleneck_pct:.1f}%** of simulations
        - This tells you where to focus improvement efforts!
        """)
    
    # Download results
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Simulation Data (CSV)",
        data=csv,
        file_name=f"monte_carlo_simulation_{n_simulations}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Why Python > Excel for Monte Carlo:</strong> Excel can do basic Monte Carlo with macros, 
    but it's slow (minutes for 1,000 runs), hard to visualize, and can't integrate with optimization at scale. 
    Python does 1,000+ optimizations in seconds with beautiful interactive visualizations!</p>
    <p><em>Based on "The Goal" by Eliyahu M. Goldratt</em></p>
</div>
""", unsafe_allow_html=True)
