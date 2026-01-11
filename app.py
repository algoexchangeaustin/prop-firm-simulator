"""
Prop Firm Simulation Chatbot
Upload backtest CSV reports and simulate against prop firm rules
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Import simulation engine
from simulation import (
    parse_trades_from_csv,
    run_monte_carlo_simulation,
    get_trade_statistics,
    AggregateResults
)

# Page config
st.set_page_config(
    page_title="Prop Firm Simulator",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for chat-like interface
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .assistant-message {
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .pass-rate-high {
        color: #28a745;
        font-size: 2rem;
        font-weight: bold;
    }
    .pass-rate-medium {
        color: #ffc107;
        font-size: 2rem;
        font-weight: bold;
    }
    .pass-rate-low {
        color: #dc3545;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_prop_firms():
    """Load prop firm rules from JSON file"""
    json_path = Path(__file__).parent / "prop_firms.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['firms']


def display_chat_message(role: str, content: str):
    """Display a chat-style message"""
    css_class = "assistant-message" if role == "assistant" else "user-message"
    icon = "🤖" if role == "assistant" else "👤"
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {role.title()}</strong>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def create_equity_chart(results: AggregateResults, starting_balance: float):
    """Create equity curve visualization"""
    fig = go.Figure()
    
    # Plot sample equity curves
    for i, curve in enumerate(results.equity_curves[:10]):
        fig.add_trace(go.Scatter(
            y=curve,
            mode='lines',
            name=f'Sim {i+1}',
            opacity=0.5,
            line=dict(width=1)
        ))
    
    # Add starting balance line
    max_len = max(len(c) for c in results.equity_curves[:10])
    fig.add_trace(go.Scatter(
        y=[starting_balance] * max_len,
        mode='lines',
        name='Starting Balance',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.update_layout(
        title="Sample Equity Curves (10 Simulations)",
        xaxis_title="Trading Days",
        yaxis_title="Account Equity ($)",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def create_pass_rate_gauge(pass_rate: float):
    """Create a gauge chart for pass rate"""
    color = "#28a745" if pass_rate >= 0.7 else "#ffc107" if pass_rate >= 0.5 else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pass_rate * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Pass Rate"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "#ffebee"},
                {'range': [50, 70], 'color': "#fff8e1"},
                {'range': [70, 100], 'color': "#e8f5e9"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': pass_rate * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_failure_chart(failure_reasons: dict):
    """Create pie chart of failure reasons"""
    if not failure_reasons:
        return None
    
    fig = px.pie(
        values=list(failure_reasons.values()),
        names=list(failure_reasons.keys()),
        title="Failure Reasons Breakdown"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def main():
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'trades' not in st.session_state:
        st.session_state.trades = None
    if 'selected_firm' not in st.session_state:
        st.session_state.selected_firm = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Load prop firms
    prop_firms = load_prop_firms()

    # Header
    st.title("📊 Prop Firm Simulation Chatbot")
    st.markdown("*Upload your backtest results and simulate your chances of passing prop firm evaluations*")
    
    st.divider()

    # Sidebar for info
    with st.sidebar:
        st.header("ℹ️ How It Works")
        st.markdown("""
        1. **Upload** your backtest CSV file
        2. **Select** a prop firm to simulate against
        3. **Run** 200 Monte Carlo simulations
        4. **Review** your pass probability
        
        ---
        
        **Supported CSV Formats:**
        - Must have a P&L column (Profit, PnL, Net Profit)
        - Date column optional but recommended
        - One row per trade
        
        ---
        
        **Disclaimer:**
        *This simulation is based on historical backtest data and uses random resampling. Past performance does not guarantee future results. This tool is for educational purposes only.*
        """)
        
        st.divider()
        st.header("📋 Supported Firms")
        for firm_id, firm in prop_firms.items():
            st.markdown(f"• {firm['display_name']}")

    # Main chat area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Step 1: File Upload
        st.subheader("Step 1: Upload Your Backtest CSV")
        
        uploaded_file = st.file_uploader(
            "Drop your backtest CSV here",
            type=['csv'],
            help="Your CSV should have at least a P&L column. Date column is optional."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(df)} rows found")
                
                # Show preview
                with st.expander("Preview uploaded data"):
                    st.dataframe(df.head(10))
                
                # Parse trades
                trades = parse_trades_from_csv(df)
                st.session_state.trades = trades
                
                # Show trade statistics
                stats = get_trade_statistics(trades)
                
                st.markdown("### 📈 Your Backtest Statistics")
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Total Trades", stats['total_trades'])
                with stat_cols[1]:
                    st.metric("Total P&L", f"${stats['total_pnl']:,.2f}")
                with stat_cols[2]:
                    st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
                with stat_cols[3]:
                    st.metric("Profit Factor", f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "∞")
                
                st.divider()
                
                # Step 2: Select Prop Firm
                st.subheader("Step 2: Select a Prop Firm")
                
                firm_options = {firm['display_name']: firm_id for firm_id, firm in prop_firms.items()}
                selected_display = st.selectbox(
                    "Choose a prop firm to simulate against:",
                    options=list(firm_options.keys())
                )
                
                if selected_display:
                    selected_firm_id = firm_options[selected_display]
                    selected_rules = prop_firms[selected_firm_id]
                    st.session_state.selected_firm = selected_rules
                    
                    # Display selected firm rules
                    with st.expander("📋 View Selected Firm Rules"):
                        rule_cols = st.columns(2)
                        with rule_cols[0]:
                            st.markdown(f"**Account Size:** ${selected_rules['account_size']:,}")
                            st.markdown(f"**Profit Target:** ${selected_rules['profit_target']:,}")
                            st.markdown(f"**Max Trailing Drawdown:** ${selected_rules['max_trailing_drawdown']:,}")
                        with rule_cols[1]:
                            st.markdown(f"**Trailing Type:** {selected_rules['trailing_drawdown_type'].replace('_', ' ').title()}")
                            st.markdown(f"**Min Trading Days:** {selected_rules['min_trading_days']}")
                            if selected_rules.get('daily_loss_limit'):
                                st.markdown(f"**Daily Loss Limit:** ${selected_rules['daily_loss_limit']:,}")
                            if selected_rules.get('consistency_rule'):
                                st.markdown(f"**Consistency Rule:** Max {selected_rules['consistency_max_day_percent']}% from single day")
                        st.info(selected_rules.get('notes', ''))
                    
                    st.divider()
                    
                    # Step 3: Run Simulation
                    st.subheader("Step 3: Run Monte Carlo Simulation")
                    
                    num_sims = st.slider(
                        "Number of simulations:",
                        min_value=50,
                        max_value=500,
                        value=200,
                        step=50,
                        help="More simulations = more accurate results, but takes longer"
                    )
                    
                    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
                        with st.spinner(f"Running {num_sims} simulations..."):
                            try:
                                results = run_monte_carlo_simulation(
                                    trades=st.session_state.trades,
                                    rules=selected_rules,
                                    num_simulations=num_sims
                                )
                                st.session_state.results = results
                            except ValueError as e:
                                st.error(f"❌ Error: {str(e)}")
                                st.session_state.results = None
                
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
                st.info("Please ensure your CSV has a column with P&L data (named 'Profit', 'PnL', 'P&L', or 'Net Profit')")

    # Results column
    with col2:
        if st.session_state.results is not None:
            results = st.session_state.results
            rules = st.session_state.selected_firm
            
            st.subheader("📊 Simulation Results")
            
            # Pass rate gauge
            st.plotly_chart(create_pass_rate_gauge(results.pass_rate), use_container_width=True)
            
            # Key metrics
            st.markdown("### Key Metrics")
            st.metric(
                "Simulations Passed",
                f"{results.pass_count} / {results.total_simulations}"
            )
            
            if results.pass_count > 0:
                st.metric(
                    "Avg Days to Target",
                    f"{results.avg_days_to_target:.1f} days"
                )
                st.metric(
                    "Median Days to Target",
                    f"{results.median_days_to_target:.0f} days"
                )
            
            st.metric(
                "Avg Max Drawdown",
                f"${results.avg_max_drawdown:,.0f}"
            )
            
            # Interpretation
            st.markdown("### 🎯 Interpretation")
            if results.pass_rate >= 0.7:
                st.success(f"""
                **Strong pass probability!** 
                
                Based on {results.total_simulations} simulations of your backtest data, 
                you have a **{results.pass_rate*100:.1f}%** chance of passing this evaluation.
                """)
            elif results.pass_rate >= 0.5:
                st.warning(f"""
                **Moderate pass probability.**
                
                With a **{results.pass_rate*100:.1f}%** pass rate, consider:
                - Reducing position sizes
                - Tightening stop losses
                - Choosing a firm with looser drawdown rules
                """)
            else:
                st.error(f"""
                **Low pass probability.**
                
                A **{results.pass_rate*100:.1f}%** pass rate suggests this strategy 
                may not be well-suited for this prop firm's rules. Consider significant 
                adjustments or a different prop firm.
                """)

    # Full width charts below
    if st.session_state.results is not None:
        results = st.session_state.results
        rules = st.session_state.selected_firm
        
        st.divider()
        st.subheader("📈 Detailed Analysis")
        
        chart_cols = st.columns(2)
        
        with chart_cols[0]:
            # Equity curves
            fig = create_equity_chart(results, rules['account_size'])
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_cols[1]:
            # Failure reasons
            if results.failure_reasons:
                fig = create_failure_chart(results.failure_reasons)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No failures to analyze - all simulations passed! 🎉")
        
        # Disclaimer
        st.divider()
        st.caption("""
        **⚠️ Important Disclaimer:** This simulation uses Monte Carlo methods to randomly resample your historical backtest trades. 
        Results are probabilistic estimates, not guarantees. Actual trading involves additional factors including:
        - Market conditions and regime changes
        - Execution quality and slippage
        - Psychological factors
        - Platform-specific rules and timing
        
        This tool is for educational and planning purposes only. Past performance does not guarantee future results.
        Always trade responsibly and only risk capital you can afford to lose.
        """)


if __name__ == "__main__":
    main()
