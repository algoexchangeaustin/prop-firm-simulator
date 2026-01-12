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
    run_rolling_window_simulation,
    run_actual_sequence_test,
    simulate_funded_account_with_payouts,
    get_trade_statistics,
    AggregateResults,
    PayoutSimulationResult
)

# Import tooltips and help text
try:
    from tooltips import TOOLTIPS, EVAL_VS_FUNDED_INFO, FIRM_WARNINGS, PAYOUT_COLUMNS
except ImportError:
    # Fallback if tooltips.py not found
    TOOLTIPS = {}
    EVAL_VS_FUNDED_INFO = ""
    FIRM_WARNINGS = {}
    PAYOUT_COLUMNS = {}

# Page config
st.set_page_config(
    page_title="Prop Firm Rule Comparison Tool",
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
        background-color: rgba(240, 242, 246, 0.1);
    }
    .user-message {
        background-color: rgba(227, 242, 253, 0.1);
    }
    /* Remove custom metric styling - let Streamlit handle it */
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


def create_equity_chart(results: AggregateResults, starting_balance: float, num_curves: int = 10):
    """Create equity curve visualization"""
    fig = go.Figure()
    
    # Limit to available curves
    curves_to_show = min(num_curves, len(results.equity_curves))
    
    # Plot sample equity curves
    for i, curve in enumerate(results.equity_curves[:curves_to_show]):
        fig.add_trace(go.Scatter(
            y=curve,
            mode='lines',
            name=f'Sim {i+1}',
            opacity=0.5,
            line=dict(width=1)
        ))
    
    # Add starting balance line
    if curves_to_show > 0:
        max_len = max(len(c) for c in results.equity_curves[:curves_to_show])
        fig.add_trace(go.Scatter(
            y=[starting_balance] * max_len,
            mode='lines',
            name='Starting Balance',
            line=dict(color='gray', dash='dash')
        ))
    
    fig.update_layout(
        title=f"Sample Equity Curves ({curves_to_show} Simulations)",
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
    st.title("📊 Prop Firm Rule Comparison Tool")
    st.markdown("*Upload your backtest results to see how they compare against various prop firm rule sets — for educational purposes only*")
    
    st.divider()

    # Sidebar for info
    with st.sidebar:
        st.header("ℹ️ How It Works")
        st.markdown("""
        1. **Select** your trading platform
        2. **Upload** your backtest CSV file
        3. **Select** a prop firm to compare against
        4. **Run** simulations using your historical data
        5. **Review** the hypothetical results
        """)
        
        st.divider()
        
        # COMPLIANCE: Add prominent disclaimer
        st.warning("""
        ⚠️ **Important Notice**
        
        This tool provides **hypothetical simulations** based on your uploaded backtest data. Results are for **educational and informational purposes only**.
        
        - Past performance does not predict future results
        - Simulations do not account for slippage, commissions, or market conditions
        - This is NOT investment advice
        - We are not registered investment advisors
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
        
        # NEW: Eval vs Funded Info Panel
        with st.expander("🔄 Eval vs Funded Rules", expanded=False):
            st.markdown("""
            **Rules often CHANGE when you pass evaluation:**
            
            | Firm | Key Change |
            |------|------------|
            | **Topstep** | MLL resets to $0! |
            | **Apex** | 30% consistency ADDED |
            | **TPT** | 50% consistency REMOVED |
            | **MFF** | 40% consistency ADDED |
            | **Tradeify Select** | 40% consistency REMOVED |
            | **Lucid Flex** | 50% consistency REMOVED |
            
            ⚠️ Always check funded rules before passing!
            """)
        
        st.divider()
        
        st.caption("""
        **Disclaimer:**
        *This tool provides hypothetical simulations based on historical backtest data using random resampling methods. Results are for educational and informational purposes only. Past performance does not predict future results. Actual trading involves risks not captured in these simulations, including slippage, commissions, market volatility, and execution differences. This tool does not provide investment advice and should not be relied upon for trading decisions. Users are solely responsible for their own trading decisions.*
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
                    
                    # Display selected firm rules - EVALUATION PHASE
                    with st.expander("📋 View Evaluation Rules", expanded=False):
                        rule_cols = st.columns(2)
                        with rule_cols[0]:
                            st.markdown(f"**Account Size:** ${selected_rules['account_size']:,}", help=TOOLTIPS.get('account_size', ''))
                            st.markdown(f"**Profit Target:** ${selected_rules['profit_target']:,}", help=TOOLTIPS.get('profit_target', ''))
                            st.markdown(f"**Max Trailing Drawdown:** ${selected_rules['max_trailing_drawdown']:,}", help=TOOLTIPS.get('trailing_drawdown', ''))
                        with rule_cols[1]:
                            dd_type = selected_rules['trailing_drawdown_type'].replace('_', ' ').title()
                            dd_help = TOOLTIPS.get('eod_trailing', '') if 'end' in dd_type.lower() else TOOLTIPS.get('intraday_trailing', '')
                            st.markdown(f"**Trailing Type:** {dd_type}", help=dd_help)
                            st.markdown(f"**Min Trading Days:** {selected_rules['min_trading_days']}", help=TOOLTIPS.get('min_trading_days', ''))
                            if selected_rules.get('daily_loss_limit'):
                                st.markdown(f"**Daily Loss Limit:** ${selected_rules['daily_loss_limit']:,}", help=TOOLTIPS.get('daily_loss_limit', ''))
                            if selected_rules.get('consistency_rule'):
                                st.markdown(f"**Consistency Rule:** Max {selected_rules['consistency_max_day_percent']}% from single day", help=TOOLTIPS.get('consistency_rule', ''))
                        if selected_rules.get('notes'):
                            st.info(selected_rules.get('notes', ''))
                    
                    # NEW: Display funded account rules with changes highlighted
                    payout_rules = selected_rules.get('payout_rules', {})
                    with st.expander("🏦 View Funded Account Rules (After Passing)", expanded=False):
                        st.markdown("### What Changes When You Pass Evaluation?")
                        
                        # Activation fee
                        activation_fee = selected_rules.get('activation_fee', 0)
                        if activation_fee > 0:
                            st.warning(f"💰 **Activation Fee:** ${activation_fee:,} (one-time)", help=TOOLTIPS.get('activation_fee', ''))
                        else:
                            st.success("✅ **No Activation Fee** - Free to activate!")
                        
                        # Rule changes comparison
                        change_cols = st.columns(2)
                        with change_cols[0]:
                            st.markdown("**📋 Evaluation Phase:**")
                            eval_consistency = selected_rules.get('consistency_max_day_percent') if selected_rules.get('consistency_rule') else None
                            payout_consistency = payout_rules.get('funded_consistency_percent') or payout_rules.get('consistency_percent')
                            
                            st.markdown(f"- Consistency: {f'{eval_consistency}%' if eval_consistency else 'None'}")
                            st.markdown(f"- Buffer Required: None")
                            st.markdown(f"- MLL: Trails from start")
                        
                        with change_cols[1]:
                            st.markdown("**🏦 Funded Phase:**")
                            st.markdown(f"- Consistency: {f'{payout_consistency}%' if payout_consistency else '**None**'}")
                            buffer = payout_rules.get('min_balance_buffer', 0)
                            st.markdown(f"- Buffer Required: {'**None**' if buffer == 0 else f'${buffer:,}'}")
                            if payout_rules.get('mll_resets_on_payout'):
                                st.markdown("- MLL: **⚠️ RESETS TO $0!**", help=TOOLTIPS.get('mll_reset', ''))
                            else:
                                st.markdown("- MLL: Continues trailing")
                        
                        # Firm-specific warning
                        firm_key = selected_firm_id.split('_')[0]
                        if firm_key in FIRM_WARNINGS:
                            st.info(FIRM_WARNINGS[firm_key])
                        elif 'topstep' in selected_firm_id:
                            st.info(FIRM_WARNINGS.get('topstep', ''))
                        elif 'apex' in selected_firm_id:
                            st.info(FIRM_WARNINGS.get('apex', ''))
                        elif 'tpt' in selected_firm_id:
                            st.info(FIRM_WARNINGS.get('tpt', ''))
                        elif 'lucid_flex' in selected_firm_id:
                            st.info(FIRM_WARNINGS.get('lucid_flex', ''))
                    
                    st.divider()
                    
                    # Step 4: Run Simulation
                    st.subheader("Step 4: Choose Simulation Mode")
                    
                    # Simulation mode selector with explanations
                    sim_mode = st.radio(
                        "How do you want to test your strategy?",
                        options=["🎲 Monte Carlo (Stress Test)", "📅 Rolling Window (Historical)", "✅ My Actual Sequence"],
                        help="Each mode tests your strategy differently"
                    )
                    
                    # Mode explanations
                    if "Monte Carlo" in sim_mode:
                        st.info("""
                        **🎲 Monte Carlo Simulation**
                        
                        Randomly shuffles your daily P&L to stress-test your historical data against different trade sequences. 
                        Shows: *"How did my historical trades perform when arranged in different random orders?"*
                        
                        ⚠️ **Note:** This is a hypothetical analysis tool. Results show how your past data 
                        performed under simulated conditions, not a prediction of future performance.
                        """)
                        
                        num_sims = st.slider(
                            "Number of random sequences to test:",
                            min_value=50,
                            max_value=500,
                            value=200,
                            step=50,
                            help="More simulations = larger sample of hypothetical scenarios"
                        )
                        
                    elif "Rolling Window" in sim_mode:
                        st.info("""
                        **📅 Rolling Window Simulation**
                        
                        Tests starting the challenge at different dates in your backtest, **preserving your actual trade sequence**. 
                        Answers: *"If I had started this challenge on different days, how often would I have passed?"*
                        
                        ✅ More realistic than Monte Carlo because it keeps your actual win/loss patterns intact.
                        """)
                        
                        num_sims = st.slider(
                            "Number of different start dates to test:",
                            min_value=20,
                            max_value=300,
                            value=100,
                            step=20,
                            help="Tests starting the challenge at evenly-spaced dates throughout your backtest"
                        )
                        
                    else:  # Actual Sequence
                        st.success("""
                        **✅ Your Actual Sequence**
                        
                        Tests your **exact track record** from Day 1 - no shuffling, no rolling. 
                        Answers: *"Would my actual backtest have passed this prop firm's rules?"*
                        
                        This is the most accurate for evaluating a specific historical period.
                        """)
                        num_sims = 1  # Only one run needed
                    
                    # Run button
                    button_text = "🚀 Run Simulation" if num_sims > 1 else "🚀 Test My Track Record"
                    
                    if st.button(button_text, type="primary", use_container_width=True):
                        with st.spinner(f"Running {'simulation' if num_sims > 1 else 'test'}..."):
                            try:
                                if "Monte Carlo" in sim_mode:
                                    results = run_monte_carlo_simulation(
                                        trades=st.session_state.trades,
                                        rules=selected_rules,
                                        num_simulations=num_sims
                                    )
                                    st.session_state.sim_mode = "monte_carlo"
                                elif "Rolling Window" in sim_mode:
                                    results = run_rolling_window_simulation(
                                        trades=st.session_state.trades,
                                        rules=selected_rules,
                                        num_windows=num_sims
                                    )
                                    st.session_state.sim_mode = "rolling_window"
                                else:  # Actual Sequence
                                    actual_result = run_actual_sequence_test(
                                        trades=st.session_state.trades,
                                        rules=selected_rules
                                    )
                                    # Wrap single result in AggregateResults format
                                    results = AggregateResults(
                                        total_simulations=1,
                                        pass_count=1 if actual_result.passed else 0,
                                        pass_rate=1.0 if actual_result.passed else 0.0,
                                        avg_days_to_target=actual_result.days_to_target,
                                        median_days_to_target=actual_result.days_to_target,
                                        avg_max_drawdown=actual_result.max_drawdown,
                                        failure_reasons={actual_result.failure_reason: 1} if actual_result.failure_reason else {},
                                        equity_curves=[actual_result.equity_curve],
                                        daily_pnls=[actual_result.daily_pnl]
                                    )
                                    st.session_state.sim_mode = "actual"
                                    st.session_state.actual_result = actual_result
                                
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
            sim_mode = st.session_state.get('sim_mode', 'monte_carlo')
            
            # Different headers based on mode
            if sim_mode == "actual":
                st.subheader("✅ Your Actual Track Record")
            elif sim_mode == "rolling_window":
                st.subheader("📅 Rolling Window Results")
            else:
                st.subheader("🎲 Monte Carlo Results")
            
            # For ACTUAL mode, show big PASS/FAIL
            if sim_mode == "actual":
                actual_result = st.session_state.get('actual_result')
                if actual_result and actual_result.passed:
                    st.success(f"""
                    # ✅ MET RULE CRITERIA
                    
                    Based on this hypothetical simulation, your uploaded backtest data **met the rule criteria** for this evaluation.
                    
                    - **Simulated Days to Target:** {actual_result.days_to_target}
                    - **Max Drawdown in Simulation:** ${actual_result.max_drawdown:,.2f}
                    - **Drawdown Limit:** ${rules['max_trailing_drawdown']:,}
                    - **Buffer in Simulation:** ${rules['max_trailing_drawdown'] - actual_result.max_drawdown:,.2f}
                    
                    *Note: This is a hypothetical result based on historical data. Actual trading results may differ materially.*
                    """)
                else:
                    st.error(f"""
                    # ❌ DID NOT MEET CRITERIA
                    
                    Based on this hypothetical simulation, your uploaded backtest data **did not meet** the rule criteria.
                    
                    **Simulated Failure Reason:** {actual_result.failure_reason if actual_result else 'Unknown'}
                    
                    *Note: This is a hypothetical result for educational purposes only.*
                    """)
                    if actual_result:
                        st.metric("Max Drawdown Hit", f"${actual_result.max_drawdown:,.2f}")
                        st.metric("Drawdown Limit", f"${rules['max_trailing_drawdown']:,}")
            
            else:
                # For Monte Carlo and Rolling Window, show pass rate gauge
                st.plotly_chart(create_pass_rate_gauge(results.pass_rate), use_container_width=True)
                
                # Mode-specific explanation
                if sim_mode == "rolling_window":
                    st.caption(f"Tested {results.total_simulations} different start dates with your actual trade sequence preserved.")
                else:
                    st.caption(f"Tested {results.total_simulations} randomly shuffled sequences of your daily P&L.")
                
                # Key metrics
                st.markdown("### Key Metrics")
                st.metric(
                    "Tests Passed" if sim_mode == "rolling_window" else "Simulations Passed",
                    f"{results.pass_count} / {results.total_simulations}",
                    help=TOOLTIPS.get('pass_rate', 'Percentage of simulations that passed the evaluation')
                )
                
                if results.pass_count > 0:
                    st.metric(
                        "Avg Days to Target",
                        f"{results.avg_days_to_target:.1f} days",
                        help=TOOLTIPS.get('days_to_target', 'Average number of trading days to reach profit target')
                    )
                    st.metric(
                        "Median Days to Target",
                        f"{results.median_days_to_target:.0f} days",
                        help="The middle value - 50% of simulations finished faster, 50% slower"
                    )
                
                st.metric(
                    "Avg Max Drawdown Hit",
                    f"${results.avg_max_drawdown:,.0f}"
                )
            
            # Show the rules being tested (all modes)
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
            if results.failure_reasons and sim_mode != "actual":
                st.markdown("---")
                if sim_mode == "rolling_window":
                    st.markdown("## ❌ WHY TESTS FAILED")
                    st.markdown("*These are the reasons your strategy failed when starting at different dates*")
                else:
                    st.markdown("## ❌ WHY SIMULATIONS FAILED")
                    st.markdown("*These are worst-case scenarios from randomly shuffled trade sequences*")
                
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
                        
                        In these hypothetical simulations, account equity dropped below the maximum allowed drawdown limit.
                        - **Rule limit:** ${rules['max_trailing_drawdown']:,} ({dd_type} trailing)
                        - **Avg max DD in simulations:** ${results.avg_max_drawdown:,.0f}
                        
                        *For reference: Different firms have different drawdown allowances and calculation methods (EOD vs Intraday).*
                        """)
                        
                    elif "Consistency Rule Violated" in reason:
                        max_pct = rules.get('consistency_max_day_percent', 50)
                        st.warning(f"""
                        **🟡 {reason}** — {count} simulations ({pct:.1f}%)
                        
                        In these hypothetical simulations, a single trading day exceeded the allowed percentage of total profits.
                        - **Rule:** No single day can exceed {max_pct}% of total profits
                        
                        *For reference: Some firms have no consistency rule during evaluation, while others enforce various percentages.*
                        """)
                        
                    elif "Daily Loss Limit" in reason:
                        dll = rules.get('daily_loss_limit', 0)
                        st.warning(f"""
                        **🟡 {reason}** — {count} simulations ({pct:.1f}%)
                        
                        In these hypothetical simulations, a single day loss exceeded the firm's daily limit.
                        - **Daily limit for this rule set:** ${dll:,}
                        
                        *For reference: Some firms have no daily loss limit, while others enforce strict daily caps.*
                        """)
                        
                    elif "Profit Target Not Reached" in reason:
                        st.info(f"""
                        **🔵 {reason}** — {count} simulations ({pct:.1f}%)
                        
                        In these hypothetical simulations, the data ended before reaching the required profit target.
                        - **Target for this rule set:** ${rules['profit_target']:,}
                        
                        *This may indicate the historical data period was too short, or the rule set requires a higher profit target than the data supported.*
                        """)
                    else:
                        st.info(f"**{reason}** — {count} simulations ({pct:.1f}%)")
                
                # Summary recommendation box
                st.markdown("---")
                st.markdown("### 📚 Educational Notes on Rule Differences")
                primary_failure = sorted_reasons[0][0] if sorted_reasons else None
                
                if primary_failure and "Drawdown" in primary_failure:
                    # Find firms with different drawdown rules
                    current_dd = rules['max_trailing_drawdown']
                    different_firms = [f for f in prop_firms.values() 
                                   if f['account_size'] == rules['account_size'] 
                                   and f['max_trailing_drawdown'] > current_dd]
                    
                    if different_firms:
                        best = max(different_firms, key=lambda x: x['max_trailing_drawdown'])
                        st.info(f"""
                        **Drawdown was the primary factor in simulated failures.** 
                        
                        For reference, different firms have varying drawdown allowances. For example, 
                        **{best['display_name']}** has a ${best['max_trailing_drawdown']:,} drawdown limit 
                        (${best['max_trailing_drawdown'] - current_dd:,} more than the currently selected rules).
                        
                        *This is general information about rule differences, not a recommendation.*
                        """)
                    else:
                        st.info("""
                        **Drawdown was the primary factor in simulated failures.**
                        
                        Different prop firms have varying drawdown allowances. Reviewing rule structures 
                        across firms may be educational for understanding how different rules work.
                        
                        *This is general information for educational purposes.*
                        """)
                        
                elif primary_failure and "Consistency" in primary_failure:
                    st.info("""
                    **Consistency rule was the primary factor in simulated failures.**
                    
                    For reference, some firms have no consistency rule during evaluation (such as Apex and Bulenox), 
                    while others require profits to be spread across multiple days.
                    
                    *This is general information about rule differences, not a recommendation.*
                    """)
                    
                elif primary_failure and "Daily Loss" in primary_failure:
                    st.info("""
                    **Daily loss limit was the primary factor in simulated failures.**
                    
                    For reference, some firms have no daily loss limit (such as MFF and Lucid Flex), 
                    while others enforce strict daily limits.
                    
                    *This is general information about rule differences, not a recommendation.*
                    """)
            
            else:
                st.success("🎉 **All simulations passed!** No failures to analyze.")
            
            # Interpretation
            st.markdown("### 🎯 Simulation Summary")
            if results.pass_rate >= 0.7:
                st.success(f"""
                **High pass rate in simulations** 
                
                In {results.total_simulations} hypothetical simulations using your historical data, 
                **{results.pass_rate*100:.1f}%** met the rule criteria.
                
                *Remember: Past performance in simulations does not predict future results.*
                """)
            elif results.pass_rate >= 0.5:
                st.warning(f"""
                **Moderate pass rate in simulations**
                
                With a **{results.pass_rate*100:.1f}%** pass rate in simulations, review the 
                analysis above to understand the rule criteria.
                
                *This is hypothetical analysis for educational purposes.*
                """)
            elif results.pass_rate >= 0.25:
                st.error(f"""
                **Lower pass rate in simulations**
                
                A **{results.pass_rate*100:.1f}%** pass rate in these hypothetical simulations 
                suggests reviewing how your historical data compares to different rule sets.
                
                *Results are for educational purposes only.*
                """)
            else:
                st.error(f"""
                **Low pass rate in simulations**
                
                At **{results.pass_rate*100:.1f}%**, your historical data showed limited compatibility 
                with these specific rules in simulations. See the educational analysis above.
                
                *This is not a prediction of future performance.*
                """)

    # Full width charts below
    if st.session_state.results is not None:
        results = st.session_state.results
        rules = st.session_state.selected_firm
        
        st.divider()
        st.subheader("📈 Detailed Analysis")
        
        chart_cols = st.columns(2)
        
        with chart_cols[0]:
            # Slider for number of curves (only show if more than 1 simulation)
            if results.total_simulations > 1:
                max_curves = min(50, len(results.equity_curves))
                num_curves = st.slider(
                    "Number of equity curves to display:",
                    min_value=1,
                    max_value=max_curves,
                    value=min(10, max_curves),
                    help="Show more curves to see the range of possible outcomes"
                )
            else:
                num_curves = 1
            
            # Equity curves
            fig = create_equity_chart(results, rules['account_size'], num_curves)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_cols[1]:
            # Failure reasons pie chart
            if results.failure_reasons:
                fig = create_failure_chart(results.failure_reasons)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No failures to analyze - all simulations passed! 🎉")
        
        # Payout Projection Section
        st.divider()
        st.subheader("💰 Hypothetical Payout Simulation")
        st.caption("Educational simulation showing how payouts work under different firm rules — based on your historical data")
        
        # Check if firm has payout rules
        if 'payout_rules' in rules and st.session_state.trades:
            payout_rules = rules['payout_rules']
            
            # Show payout rules summary
            with st.expander("📋 View Funded Account Payout Rules", expanded=False):
                pr_col1, pr_col2 = st.columns(2)
                with pr_col1:
                    st.markdown(f"**Min Profitable Days:** {payout_rules.get('min_profitable_days', 'N/A')}")
                    st.markdown(f"**Min Profit/Day:** ${payout_rules.get('min_profit_per_day', 0):,}")
                    st.markdown(f"**Days Between Payouts:** {payout_rules.get('days_between_payouts', 'N/A')}")
                    if payout_rules.get('consistency_percent'):
                        st.markdown(f"**Payout Consistency:** {payout_rules['consistency_percent']}% max/day")
                    else:
                        st.markdown("**Payout Consistency:** None")
                with pr_col2:
                    first_cap = payout_rules.get('first_payout_cap')
                    sub_cap = payout_rules.get('subsequent_payout_cap')
                    st.markdown(f"**First Payout Cap:** {'Unlimited' if first_cap is None else f'${first_cap:,}'}")
                    st.markdown(f"**Subsequent Cap:** {'Unlimited' if sub_cap is None else f'${sub_cap:,}'}")
                    if payout_rules.get('max_payout_percent'):
                        st.markdown(f"**Max % Per Payout:** {payout_rules['max_payout_percent']}%")
                    if payout_rules.get('mll_resets_on_payout'):
                        st.warning("⚠️ **MLL resets to $0 after first payout!**")
                    split_first = payout_rules.get('profit_split_first', 10000) or 0
                    split_after = payout_rules.get('profit_split_after', 90)
                    if split_first == 0:
                        st.markdown(f"**Profit Split:** 90/10 from start")
                    else:
                        st.markdown(f"**Profit Split:** 100% first ${split_first:,}, then {split_after}%")
                
                if payout_rules.get('notes'):
                    st.info(payout_rules['notes'])
            
            # Withdrawal strategy slider
            st.markdown("#### 🎚️ Withdrawal Strategy Simulation")
            withdrawal_pct = st.slider(
                "Simulated withdrawal % of available profits:",
                min_value=10,
                max_value=100,
                value=100,
                step=10,
                help="Adjust to see how different withdrawal strategies affect the hypothetical simulation. This is for educational comparison only."
            )
            
            if withdrawal_pct < 100:
                st.info(f"💡 **Simulating Conservative Approach:** Withdrawing only {withdrawal_pct}% of available profits in this hypothetical scenario.")
            else:
                st.caption("📈 **Simulating Maximum Withdrawal:** Taking maximum allowed each payout in this scenario.")
            
            # Run payout simulation button
            if st.button("📊 Run Payout Projection", type="secondary"):
                with st.spinner("Simulating funded account with payouts..."):
                    try:
                        payout_result = simulate_funded_account_with_payouts(
                            trades=st.session_state.trades,
                            rules=rules,
                            withdrawal_percent=withdrawal_pct
                        )
                        st.session_state.payout_result = payout_result
                        st.session_state.payout_withdrawal_pct = withdrawal_pct
                    except Exception as e:
                        st.error(f"Error running payout simulation: {str(e)}")
            
            # Display payout results
            if 'payout_result' in st.session_state and st.session_state.payout_result:
                pr = st.session_state.payout_result
                used_pct = st.session_state.get('payout_withdrawal_pct', 100)
                
                if not pr.passed_eval:
                    st.error(f"""
                    ### ❌ Did Not Meet Evaluation Criteria in Simulation
                    
                    In this hypothetical simulation, the uploaded backtest data did not meet the evaluation criteria for this rule set.
                    
                    **Simulated Failure Reason:** {pr.blow_reason}
                    
                    *This is a hypothetical result based on historical data for educational purposes only.*
                    """)
                else:
                    # Success metrics
                    st.success(f"### ✅ Evaluation Passed on Day {pr.days_to_pass_eval}")
                    
                    payout_cols = st.columns(4)
                    with payout_cols[0]:
                        st.metric("Days to First Payout", f"{pr.days_to_first_payout}" if pr.days_to_first_payout > 0 else "N/A",
                                  help="Trading days from start until first withdrawal was eligible")
                    with payout_cols[1]:
                        st.metric("Total Payouts", pr.total_payouts,
                                  help="Number of successful withdrawals taken")
                    with payout_cols[2]:
                        st.metric("Total Withdrawn", f"${pr.total_withdrawn:,.0f}",
                                  help=TOOLTIPS.get('total_withdrawn', 'Gross amount withdrawn before profit split'))
                    with payout_cols[3]:
                        st.metric("You Keep (after split)", f"${pr.total_kept_after_split:,.0f}",
                                  help=TOOLTIPS.get('you_keep', 'Net amount after profit split with prop firm'))
                    
                    # Additional info
                    info_cols = st.columns(3)
                    with info_cols[0]:
                        st.metric("Avg Payout Amount", f"${pr.avg_payout_amount:,.0f}" if pr.avg_payout_amount > 0 else "N/A",
                                  help=TOOLTIPS.get('avg_payout', 'Average withdrawal amount per payout'))
                    with info_cols[1]:
                        st.metric("Peak Balance", f"${pr.peak_balance:,.0f}",
                                  help=TOOLTIPS.get('peak_balance', 'Highest account balance reached'))
                    with info_cols[2]:
                        if pr.account_blown:
                            st.metric("Account Lifespan", f"{pr.blown_on_day} days", delta="Blown", delta_color="inverse",
                                      help=TOOLTIPS.get('account_lifespan', 'How many days account lasted before blow'))
                        else:
                            st.metric("Account Status", "Active ✅",
                                      help="Account survived entire backtest period")
                    
                    # Account Blown Explanation
                    if pr.account_blown:
                        st.markdown("---")
                        st.markdown("### ❌ Why The Account Was Blown")
                        st.error(f"""
                        **Day {pr.blown_on_day}:** {pr.blow_reason}
                        
                        **Final Balance:** ${pr.final_account_balance:,.0f}
                        """)
                        
                        # Provide context
                        if pr.total_payouts > 0:
                            st.markdown(f"""
                            #### 📊 Hypothetical Simulation Summary
                            
                            | Metric | Simulated Value |
                            |--------|-------|
                            | Days in Simulation | {pr.blown_on_day} |
                            | Simulated Payouts | {pr.total_payouts} |
                            | Simulated Withdrawals | ${pr.total_withdrawn:,.0f} |
                            | **Simulated Net (after split)** | **${pr.total_kept_after_split:,.0f}** |
                            | Final Balance | ${pr.final_account_balance:,.0f} |
                            
                            *This is a hypothetical simulation based on historical data. Actual results may differ materially due to slippage, commissions, market conditions, and other factors not accounted for in this simulation.*
                            """)
                            
                            # Suggestion for conservative strategy
                            if used_pct == 100:
                                st.info(f"""
                                💡 **Educational Note:** In this simulation, maximum withdrawals were taken. 
                                You can adjust the withdrawal percentage slider above to see how different 
                                withdrawal strategies affect the hypothetical simulation results.
                                """)
                    else:
                        st.success(f"""
                        ### ✅ Account Survived!
                        
                        Your account remained active through all {len(pr.equity_curve)} trading days in your backtest.
                        
                        **Final Balance:** ${pr.final_account_balance:,.0f}
                        """)
                    
                    # Payout history table with cushion info
                    if pr.payout_history:
                        st.markdown("#### 📜 Payout History")
                        history_data = []
                        for i, p in enumerate(pr.payout_history):
                            # Use cushion_to_trailing (real danger zone), fallback to old field
                            real_cushion = p.get('cushion_to_trailing', p.get('cushion_after', 0))
                            history_data.append({
                                'Payout #': i + 1,
                                'Day': p['day'],
                                'Withdrawn': f"${p['amount']:,.0f}",
                                'You Keep': f"${p['kept']:,.0f}",
                                'Balance After': f"${p['balance_after']:,.0f}",
                                'Blow Threshold': f"${p.get('trailing_threshold', 0):,.0f}" if p.get('trailing_threshold') else "-",
                                'Cushion': f"${real_cushion:,.0f}"
                            })
                        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
                        
                        # Detailed column explanations
                        with st.expander("ℹ️ Column Explanations"):
                            st.markdown(f"""
                            | Column | Meaning |
                            |--------|---------|
                            | **Payout #** | {PAYOUT_COLUMNS.get('payout_num', 'Sequential payout number')} |
                            | **Day** | {PAYOUT_COLUMNS.get('day', 'Trading day of withdrawal')} |
                            | **Withdrawn** | {PAYOUT_COLUMNS.get('withdrawn', 'Gross amount withdrawn')} |
                            | **You Keep** | {PAYOUT_COLUMNS.get('you_keep', 'After profit split')} |
                            | **Balance After** | {PAYOUT_COLUMNS.get('balance_after', 'Account balance after withdrawal')} |
                            | **Blow Threshold** | {PAYOUT_COLUMNS.get('blow_threshold', 'Account blows if balance drops here')} |
                            | **Cushion** | {PAYOUT_COLUMNS.get('cushion', 'Safety margin above blow threshold')} |
                            """)
                        
                        st.caption("⚠️ **Key:** When **Cushion** approaches $0, your account is about to blow!")
                    
                    # MLL Warning for Topstep
                    if payout_rules.get('mll_resets_on_payout') and pr.total_payouts > 0:
                        st.warning(f"""
                        ⚠️ **Topstep MLL Rule Note:** In this simulation, the Maximum Loss Limit 
                        reset to $0 after the first payout. Under these rules, the account cannot go below 
                        ${rules['account_size']:,} without being closed.
                        
                        *This is general information about how Topstep's MLL reset rule works, for educational purposes.*
                        """)
        else:
            st.info("Payout rules not available for this firm. Contact support to add them.")
        
        # Quick Comparison Section
        st.divider()
        st.subheader("🔄 Quick Compare: Same Account Size")
        
        # Show which mode will be used
        if sim_mode == "rolling_window":
            st.caption("Comparing using **Rolling Window** simulation (preserves your actual trade sequence)")
        elif sim_mode == "actual":
            st.caption("Comparing using **Actual Sequence** test (your exact track record)")
        else:
            st.caption("Comparing using **Monte Carlo** simulation (stress test with shuffled sequences)")
        
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
                        # Use the same simulation mode as selected
                        if sim_mode == "rolling_window":
                            comp_results = run_rolling_window_simulation(
                                trades=st.session_state.trades,
                                rules=firm_rules,
                                num_windows=100
                            )
                        elif sim_mode == "actual":
                            actual_result = run_actual_sequence_test(
                                trades=st.session_state.trades,
                                rules=firm_rules
                            )
                            # Wrap in AggregateResults format
                            comp_results = AggregateResults(
                                total_simulations=1,
                                pass_count=1 if actual_result.passed else 0,
                                pass_rate=1.0 if actual_result.passed else 0.0,
                                avg_days_to_target=actual_result.days_to_target,
                                median_days_to_target=actual_result.days_to_target,
                                avg_max_drawdown=actual_result.max_drawdown,
                                failure_reasons={actual_result.failure_reason: 1} if actual_result.failure_reason else {},
                                equity_curves=[actual_result.equity_curve],
                                daily_pnls=[actual_result.daily_pnl]
                            )
                        else:  # Monte Carlo
                            comp_results = run_monte_carlo_simulation(
                                trades=st.session_state.trades,
                                rules=firm_rules,
                                num_simulations=100
                            )
                        
                        # Get primary failure reason
                        primary_fail = "N/A"
                        if comp_results.failure_reasons:
                            primary_fail = max(comp_results.failure_reasons.items(), key=lambda x: x[1])[0]
                        
                        # Format pass rate based on mode
                        if sim_mode == "actual":
                            pass_display = "✅ PASS" if comp_results.pass_rate == 1.0 else "❌ FAIL"
                        else:
                            pass_display = f"{comp_results.pass_rate*100:.1f}%"
                        
                        comparison_results.append({
                            'Firm': firm_rules['display_name'],
                            'Pass Rate': pass_display,
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
                    
                    # Highlight highest rate
                    best = comparison_results[0]
                    st.info(f"📊 **Highest simulation pass rate:** {best['Firm']} at {best['Pass Rate']} in these hypothetical tests")
        
        # Disclaimer
        st.divider()
        st.error("""
        **⚠️ IMPORTANT DISCLAIMER — PLEASE READ**
        
        **HYPOTHETICAL PERFORMANCE RESULTS HAVE INHERENT LIMITATIONS.** Unlike actual trading records, simulated results do not represent actual trading. Since trades have not actually been executed, results may have under- or over-compensated for the impact of certain market factors, such as lack of liquidity, slippage, commissions, and market volatility.
        
        **NO REPRESENTATION IS BEING MADE** that any account will or is likely to achieve profits or losses similar to those shown. In fact, there are frequently sharp differences between hypothetical performance results and actual results subsequently achieved.
        
        **THIS TOOL IS FOR EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY.** It is designed to help users understand how different prop firm rules work by comparing historical backtest data against various rule sets. This is NOT investment advice, trading advice, or a recommendation to use any particular prop firm.
        
        **WE ARE NOT REGISTERED INVESTMENT ADVISORS.** We do not provide personalized or individualized trading advice. All information presented is general in nature and should not be construed as advice tailored to your specific situation.
        
        **PAST PERFORMANCE DOES NOT PREDICT FUTURE RESULTS.** Trading futures involves substantial risk of loss and is not suitable for all investors. You should carefully consider whether trading is suitable for you in light of your circumstances, knowledge, and financial resources.
        
        **USERS ARE SOLELY RESPONSIBLE** for their own trading decisions. Always conduct your own research and consult with qualified professionals before making any trading or investment decisions.
        """)
        
        st.caption("*By using this tool, you acknowledge that you have read and understood this disclaimer.*")


if __name__ == "__main__":
    main()
