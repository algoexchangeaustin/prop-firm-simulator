"""
Tooltips and Help Text for Prop Firm Simulator
Centralized definitions for all help text shown on hover
Educational information about prop firm rules and terminology
"""

# ============================================================================
# MAIN CONCEPTS - These appear as tooltips throughout the app
# ============================================================================

TOOLTIPS = {
    # Account/Evaluation Terms
    "account_size": "The starting balance of the simulated account in this rule set.",
    "profit_target": "The profit level required during evaluation under these rules. Typically 6-8% of account size.",
    "trailing_drawdown": "Maximum allowed loss from peak balance under these rules. This number 'trails' up as the account grows but does not move down.",
    "daily_loss_limit": "Maximum loss allowed in a single trading day under these rules.",
    "consistency_rule": "Rule limiting how much profit can come from a single day under certain firm rules.",
    "min_trading_days": "Minimum number of days required before evaluation completion under these rules.",
    
    # Drawdown Types
    "intraday_trailing": "Drawdown calculated in real-time during trading session. A temporary dip can trigger the limit even if recovered by close.",
    "eod_trailing": "End-of-Day trailing drawdown. Only calculated at session close.",
    "static_drawdown": "Fixed drawdown that does not trail. Once set, this level does not change.",
    
    # Payout Terms
    "buffer": "Minimum profit cushion required to remain in account before withdrawing under certain rules.",
    "payout_cap": "Maximum amount that can be withdrawn per payout period under certain rules.",
    "profit_split": "How profits are divided between trader and firm. Example: 90/10 means trader keeps 90%.",
    "min_profitable_days": "Number of profitable days required between payouts under certain rules.",
    "days_between_payouts": "Minimum waiting period between payout requests under certain rules.",
    "consistency_payout": "Rule limiting how much of total profit can come from a single day when requesting payout.",
    
    # Funded Account Changes
    "activation_fee": "One-time fee to activate funded account after passing evaluation. Varies by firm.",
    "mll_reset": "Some firms reset the Maximum Loss Limit after first payout. Check specific firm rules.",
    "eval_vs_funded_consistency": "Some firms have different consistency rules for evaluation vs funded phases.",
    "scaling_plan": "Funded accounts often have position size limits that change based on account balance.",
    
    # Simulation Terms
    "pass_rate": "Percentage of hypothetical simulations where the historical data met the rule criteria.",
    "monte_carlo": "Statistical method that shuffles trades randomly to test many hypothetical outcomes.",
    "rolling_window": "Tests the exact trade sequence starting at different dates in the historical data.",
    "cushion": "Distance between current balance and the rule threshold in simulations.",
    "blow_threshold": "The balance at which the account would be closed under these rules.",
    
    # Results Metrics
    "days_to_target": "Number of trading days to reach the profit target in simulations.",
    "max_drawdown_hit": "The largest peak-to-trough decline during the simulation.",
    "peak_balance": "Highest account balance reached during the simulation.",
    "account_lifespan": "How many days the account lasted in the simulation.",
    "total_withdrawn": "Gross amount withdrawn in the simulation (before profit split).",
    "you_keep": "Net amount after the profit split in the simulation.",
    "avg_payout": "Average withdrawal amount across all simulated payouts.",
    
    # Strategy Helpers
    "withdrawal_strategy": "Adjust withdrawal percentage to see how different approaches affect simulations.",
    "conservative_approach": "Simulation with lower withdrawal percentages.",
    "aggressive_approach": "Simulation with maximum withdrawal percentages.",
}

# ============================================================================
# EVAL vs FUNDED COMPARISON - Educational information about rule differences
# ============================================================================

EVAL_VS_FUNDED_INFO = """
### 🔄 Rule Differences: Evaluation vs Funded Phases

For educational purposes, here are common rule changes when moving from evaluation to funded phases:

| Rule | Evaluation Phase | Funded Phase |
|------|------------|----------------|
| **Consistency Rule** | Often none or relaxed | May be added or stricter |
| **Buffer/Reserve** | Usually none | Often required before withdrawals |
| **MLL (Max Loss Limit)** | Trails from starting balance | May reset after first payout |
| **Position Scaling** | Usually full size from day 1 | Often scaled based on balance |
| **Daily Loss Limit** | Varies by firm | May be removed or added |
| **Activation Fee** | N/A | Varies from $0 to $900+ |

*This is general information about how different firms structure their rules. Always verify current rules directly with each firm.*
"""

# ============================================================================
# FIRM-SPECIFIC INFORMATION - Educational notes about rule differences
# ============================================================================

FIRM_WARNINGS = {
    "topstep": "ℹ️ **Topstep Rule Note:** Under Topstep's rules, the MLL resets to $0 after the first payout. This is how their funded account structure works.",
    "apex": "ℹ️ **Apex Rule Note:** A 30% consistency rule applies in funded PA accounts (not in evaluation). A safety net is required for the first 3 payouts only.",
    "tpt": "ℹ️ **TPT Rule Note:** A 50% consistency rule applies in the TEST phase only. Funded PRO accounts have no consistency rule. PRO+ is available after demonstrating $10K profit.",
    "mff": "ℹ️ **MFF Rule Note:** Activation fee was removed (July 2025). A 40% consistency rule applies in the funded stage (not evaluation).",
    "tradeify_growth": "ℹ️ **Tradeify Growth Rule Note:** Can be completed in 1 day. 35% consistency applies only in funded (not eval). No buffer requirement.",
    "tradeify_select": "ℹ️ **Tradeify Select Rule Note:** 40% consistency applies in eval only — removed when funded. Options include Flex (no buffer) or Daily payout policy.",
    "bulenox": "ℹ️ **Bulenox Rule Note:** Activation fee ranges from $98-$898 by account size. 40% consistency rule applies. First 3 payouts are capped, then unlimited.",
    "lucid_pro": "ℹ️ **Lucid Pro Rule Note:** $0 activation fee. 40% consistency applies in both eval and funded. Buffer = MLL + $100.",
    "lucid_flex": "ℹ️ **Lucid Flex Rule Note:** 50% consistency in eval only. No consistency, no buffer, no daily loss limit in funded phase.",
}

# ============================================================================
# PAYOUT HISTORY COLUMN EXPLANATIONS - Educational descriptions
# ============================================================================

PAYOUT_COLUMNS = {
    "payout_num": "Sequential payout number in the simulation (1st, 2nd, 3rd, etc.)",
    "day": "Trading day in the simulation when the payout occurred",
    "withdrawn": "Gross amount withdrawn in the simulation",
    "you_keep": "Net amount after profit split in the simulation",
    "balance_after": "Account balance after the withdrawal in the simulation",
    "blow_threshold": "The threshold where the account would be closed under these rules",
    "cushion": "Distance between balance and threshold in the simulation",
    "max_available": "Maximum amount available for withdrawal under these rules",
}
