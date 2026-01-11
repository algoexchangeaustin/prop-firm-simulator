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
        1. **Select** your trading platform
        2. **Upload** your backtest CSV file
        3. **Select** a prop firm to simulate against
        4. **Run** 200 Monte Carlo simulations
        5. **Review** your pass probability
        """)
        
        st.divider()
        
        st.header("🖥️ Supported Platforms")
        st.markdown("""
        - **TradeStation** - Performance Reports
        - **NinjaTrader** - Trade Performance CSV
        - **Generic CSV** - Any CSV with P&L column
        """)
        
        st.divider()
        
        st.header("📋 Supported Prop Firms")
        for firm_id, firm in prop_firms.items():
            st.markdown(f"• {firm['display_name']}")
        
        st.divider()
        
        st.caption("""
        **Disclaimer:**
        *This simulation is based on historical backtest data and uses random resampling. Past performance does not guarantee future results. This tool is for educational purposes only.*
        """)

    # Main chat area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Step 1: Select Platform
        st.subheader("Step 1: Select Your Trading Platform")
        
        platform = st.selectbox(
            "What platform is your backtest from?",
            options=["Auto-Detect", "TradeStation", "NinjaTrader", "Generic CSV"],
            help="Select your platform for best results, or use Auto-Detect"
        )
        
        # Platform-specific guidance
        platform_help = {
            "TradeStation": "Export your **Performance Report** as CSV from TradeStation.",
            "NinjaTrader": "Export your **Trade Performance** results as CSV from NinjaTrader.",
            "Generic CSV": "Any CSV with a **Profit** or **PnL** column and optionally a **Date** column.",
            "Auto-Detect": "We'll try to automatically detect your file format."
        }
        st.caption(platform_help.get(platform, ""))
        
        st.divider()
        
        # Step 2: File Upload
        st.subheader("Step 2: Upload Your Backtest CSV")
        
        uploaded_file = st.file_uploader(
            "Drop your backtest CSV here",
            type=['csv'],
            help="Your CSV should have at least a P&L column. Date column is optional."
        )
        
        if uploaded_file is not None:
            try:
                # Read raw text first for format detection
                raw_text = uploaded_file.getvalue().decode('utf-8')
                uploaded_file.seek(0)  # Reset file pointer
                
                # Determine format based on selection or auto-detect
                if platform == "TradeStation" or (platform == "Auto-Detect" and 'TradeStation' in raw_text):
                    detected_platform = "TradeStation"
                    st.success(f"✅ TradeStation report detected")
                    df = None  # Not used for TradeStation
                elif platform == "NinjaTrader" or (platform == "Auto-Detect" and 'Trade number' in raw_text and 'Market pos.' in raw_text):
                    detected_platform = "NinjaTrader"
                    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                    st.success(f"✅ NinjaTrader export detected: {len(df)} trades found")
                    
                    # Show preview
                    with st.expander("Preview uploaded data"):
                        st.dataframe(df.head(10))
                else:
                    detected_platform = "Generic"
                    df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                    st.success(f"✅ File loaded: {len(df)} rows found")
                    
                    # Show preview
                    with st.expander("Preview uploaded data"):
                        st.dataframe(df.head(10))
                
                # Parse trades with format info
                trades = parse_trades_from_csv(df, raw_text, platform_hint=detected_platform)
                st.session_state.trades = trades
                
                st.info(f"📊 **{len(trades)} trades** parsed from {detected_platform} export")
                
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
                
                # Step 3: Select Prop Firm
                st.subheader("Step 3: Select a Prop Firm")
                
                firm_options = {firm['display_name']: firm_id for firm_id, firm in prop_firms.items()}
                selected_display = st.selectbox(
                    "Choose a prop firm to simulate against:",
                    options=list(firm_options.keys())
                )
                
                if selected_display:
                    selected_firm_id = firm_options[selected_display]
                    selected_rules = prop_firms[selected_firm_id]
                    st.session_state.selected_firm = selected_rules
                    st.session_state.selected_firm_key = selected_firm_id  # Track the key for comparison
                    
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
                    
                    # Step 4: Run Simulation
                    st.subheader("Step 4: Run Monte Carlo Simulation")
                    
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
                "Avg Max Drawdown Hit",
                f"${results.avg_max_drawdown:,.0f}"
            )
            
            # Show the rules being tested
            st.markdown("### 📋 Rules Being Tested")
            rules_col1, rules_col2 = st.columns(2)
            with rules_col1:
                st.markdown(f"**Profit Target:** ${rules['profit_target']:,}")
                st.markdown(f"**Max Drawdown:** ${rules['max_trailing_drawdown']:,}")
                dd_type = "Intraday" if rules.get('trailing_drawdown_type') == 'intraday' else "End-of-Day"
                st.markdown(f"**DD Type:** {dd_type}")
            with rules_col2:
                dll = rules.get('daily_loss_limit')
                st.markdown(f"**Daily Loss Limit:** {'$'+f'{dll:,}' if dll else 'None'}")
                if rules.get('consistency_rule'):
                    st.markdown(f"**Consistency:** {rules.get('consistency_max_day_percent')}% max/day")
                else:
                    st.markdown("**Consistency:** None")
                st.markdown(f"**Min Days:** {rules.get('min_trading_days', 0)}")
            
            # Failure Reasons Summary - DETAILED BREAKDOWN
            if results.failure_reasons:
                st.markdown("---")
                st.markdown("## ❌ WHY SIMULATIONS FAILED")
                st.markdown("*Understanding why your strategy didn't pass helps you improve*")
                
                # Calculate totals
                total_failures = results.total_simulations - results.pass_count
                sorted_reasons = sorted(results.failure_reasons.items(), key=lambda x: x[1], reverse=True)
                
                # Create a detailed breakdown table
                failure_data = []
                for reason, count in sorted_reasons:
                    pct_of_total = (count / results.total_simulations) * 100
                    pct_of_failures = (count / total_failures) * 100 if total_failures > 0 else 0
                    failure_data.append({
                        'Failure Type': reason,
                        'Count': count,
                        '% of All Sims': f"{pct_of_total:.1f}%",
                        '% of Failures': f"{pct_of_failures:.1f}%"
                    })
                
                # Display as table
                st.dataframe(
                    pd.DataFrame(failure_data),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Detailed explanation for each failure type
                st.markdown("### 📖 What Each Failure Means:")
                
                for reason, count in sorted_reasons:
                    pct = (count / results.total_simulations) * 100
                    
                    if "Max Drawdown Exceeded" in reason:
                        dd_type = "Intraday" if rules.get('trailing_drawdown_type') == 'intraday' else "End-of-Day"
                        st.error(f"""
                        **🔴 {reason}** — {count} simulations ({pct:.1f}%)
                        
                        Your account equity dropped below the maximum allowed drawdown limit.
                        - **Your limit:** ${rules['max_trailing_drawdown']:,} ({dd_type} trailing)
                        - **Avg max DD hit:** ${results.avg_max_drawdown:,.0f}
                        - **What this means:** Your winning streaks create a high water mark, then losing streaks pull you below the trailing threshold.
                        
                        **💡 Fix:** Try firms with larger drawdown allowances (Lucid has 4% = ${int(rules['account_size']*0.04):,}) or reduce position sizes.
                        """)
                        
                    elif "Consistency Rule Violated" in reason:
                        max_pct = rules.get('consistency_max_day_percent', 50)
                        st.warning(f"""
                        **🟡 {reason}** — {count} simulations ({pct:.1f}%)
                        
                        One of your trading days made too much profit relative to your total.
                        - **Rule:** No single day can exceed {max_pct}% of total profits
                        - **What this means:** You had "home run" days that were too large compared to your other days.
                        
                        **💡 Fix:** Try firms with NO consistency rule (Apex eval, Bulenox, Tradeify Growth) or trade more consistently sized positions.
                        """)
                        
                    elif "Daily Loss Limit" in reason:
                        dll = rules.get('daily_loss_limit', 0)
                        st.warning(f"""
                        **🟡 {reason}** — {count} simulations ({pct:.1f}%)
                        
                        You lost more than the maximum allowed amount in a single day.
                        - **Daily limit:** ${dll:,}
                        - **What this means:** A bad day exceeded the firm's daily risk tolerance.
                        
                        **💡 Fix:** Try firms with NO daily loss limit (Apex, Lucid Flex, Take Profit Trader, MFF) or add stricter daily stop losses.
                        """)
                        
                    elif "Profit Target Not Reached" in reason:
                        st.info(f"""
                        **🔵 {reason}** — {count} simulations ({pct:.1f}%)
                        
                        The simulation ended before reaching the required profit target.
                        - **Target needed:** ${rules['profit_target']:,}
                        - **What this means:** The random sequence of your trades didn't accumulate enough profit before the data ran out.
                        
                        **💡 Fix:** This often resolves with more trading days. Your strategy may need more time to reach targets, or consider smaller account sizes with lower targets.
                        """)
                    else:
                        st.info(f"**{reason}** — {count} simulations ({pct:.1f}%)")
                
                # Summary recommendation box
                st.markdown("---")
                st.markdown("### 🎯 TOP RECOMMENDATION")
                primary_failure = sorted_reasons[0][0] if sorted_reasons else None
                
                if primary_failure and "Drawdown" in primary_failure:
                    # Find best alternative firm
                    current_dd = rules['max_trailing_drawdown']
                    better_firms = [f for f in prop_firms.values() 
                                   if f['account_size'] == rules['account_size'] 
                                   and f['max_trailing_drawdown'] > current_dd]
                    
                    if better_firms:
                        best = max(better_firms, key=lambda x: x['max_trailing_drawdown'])
                        st.success(f"""
                        **Drawdown is your #1 issue.** 
                        
                        Try **{best['display_name']}** which has a ${best['max_trailing_drawdown']:,} drawdown 
                        (${best['max_trailing_drawdown'] - current_dd:,} more room than your current selection).
                        """)
                    else:
                        st.success("""
                        **Drawdown is your #1 issue.**
                        
                        Consider reducing position sizes by 20-30% or choosing a smaller account size with proportionally larger drawdown allowance.
                        """)
                        
                elif primary_failure and "Consistency" in primary_failure:
                    st.success("""
                    **Consistency rule is your #1 issue.**
                    
                    Try **Apex Trader Funding** or **Bulenox** — they have NO consistency rule during evaluation.
                    """)
                    
                elif primary_failure and "Daily Loss" in primary_failure:
                    st.success("""
                    **Daily loss limit is your #1 issue.**
                    
                    Try **My Funded Futures** or **Lucid Flex** — they have NO daily loss limit.
                    """)
            
            else:
                st.success("🎉 **All simulations passed!** No failures to analyze.")
            
            # Interpretation
            st.markdown("### 🎯 Overall Assessment")
            if results.pass_rate >= 0.7:
                st.success(f"""
                **Strong pass probability!** 
                
                Based on {results.total_simulations} simulations, 
                you have a **{results.pass_rate*100:.1f}%** chance of passing.
                """)
            elif results.pass_rate >= 0.5:
                st.warning(f"""
                **Moderate pass probability.**
                
                With a **{results.pass_rate*100:.1f}%** pass rate, review the 
                failure reasons above and consider adjustments.
                """)
            elif results.pass_rate >= 0.25:
                st.error(f"""
                **Low pass probability.**
                
                A **{results.pass_rate*100:.1f}%** pass rate suggests trying 
                a different prop firm or adjusting your strategy.
                """)
            else:
                st.error(f"""
                **Very low pass probability.**
                
                At **{results.pass_rate*100:.1f}%**, this strategy is not 
                well-suited for these rules. See recommendations above.
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
            # Failure reasons pie chart
            if results.failure_reasons:
                fig = create_failure_chart(results.failure_reasons)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No failures to analyze - all simulations passed! 🎉")
        
        # Quick Comparison Section
        st.divider()
        st.subheader("🔄 Quick Compare: Same Account Size")
        st.caption("See how your backtest performs across different firms with similar account sizes")
        
        # Find firms with same account size
        account_size = rules['account_size']
        similar_firms = {k: v for k, v in prop_firms.items() 
                        if v['account_size'] == account_size and k != st.session_state.get('selected_firm_key')}
        
        if similar_firms and st.session_state.trades:
            if st.button("🔍 Compare Against Other Firms", type="secondary"):
                comparison_results = []
                progress = st.progress(0)
                
                for i, (firm_key, firm_rules) in enumerate(similar_firms.items()):
                    try:
                        comp_results = run_monte_carlo_simulation(
                            trades=st.session_state.trades,
                            rules=firm_rules,
                            num_simulations=100  # Fewer sims for speed
                        )
                        
                        # Get primary failure reason
                        primary_fail = "N/A"
                        if comp_results.failure_reasons:
                            primary_fail = max(comp_results.failure_reasons.items(), key=lambda x: x[1])[0]
                        
                        comparison_results.append({
                            'Firm': firm_rules['display_name'],
                            'Pass Rate': f"{comp_results.pass_rate*100:.1f}%",
                            'Max DD': f"${firm_rules['max_trailing_drawdown']:,}",
                            'DD Type': 'Intraday' if firm_rules.get('trailing_drawdown_type') == 'intraday' else 'EOD',
                            'DLL': f"${firm_rules['daily_loss_limit']:,}" if firm_rules.get('daily_loss_limit') else 'None',
                            'Consistency': f"{firm_rules.get('consistency_max_day_percent', '-')}%" if firm_rules.get('consistency_rule') else 'None',
                            'Primary Failure': primary_fail[:30] + '...' if len(primary_fail) > 30 else primary_fail,
                            '_pass_rate': comp_results.pass_rate  # For sorting
                        })
                    except Exception as e:
                        pass
                    
                    progress.progress((i + 1) / len(similar_firms))
                
                progress.empty()
                
                if comparison_results:
                    # Sort by pass rate descending
                    comparison_results.sort(key=lambda x: x['_pass_rate'], reverse=True)
                    
                    # Remove sort key and display
                    for r in comparison_results:
                        del r['_pass_rate']
                    
                    # Show as dataframe with color coding
                    df_compare = pd.DataFrame(comparison_results)
                    
                    st.markdown("**Results ranked by pass rate (highest first):**")
                    st.dataframe(
                        df_compare,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Highlight best option
                    best = comparison_results[0]
                    st.success(f"🏆 **Best match:** {best['Firm']} with {best['Pass Rate']} pass rate")
        
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
