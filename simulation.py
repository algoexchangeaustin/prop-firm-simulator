"""
Monte Carlo Simulation Engine for Prop Firm Rule Validation
Handles trailing drawdowns, daily loss limits, consistency rules, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import re


@dataclass
class SimulationResult:
    """Results from a single simulation run"""
    passed: bool
    final_equity: float
    max_drawdown: float
    days_to_target: int
    failure_reason: str = None
    equity_curve: List[float] = None
    daily_pnl: List[float] = None


@dataclass
class AggregateResults:
    """Aggregate results from all simulations"""
    total_simulations: int
    pass_count: int
    pass_rate: float
    avg_days_to_target: float
    median_days_to_target: float
    avg_max_drawdown: float
    failure_reasons: Dict[str, int]
    equity_curves: List[List[float]]
    daily_pnls: List[List[float]]


def parse_currency(value):
    """Parse currency string like '$123.45' or '($123.45)' to float"""
    if pd.isna(value) or value == '' or value is None:
        return 0.0
    
    value_str = str(value).strip()
    
    # Check if negative (parentheses format)
    is_negative = '(' in value_str and ')' in value_str
    
    # Remove currency symbols, parentheses, commas
    cleaned = re.sub(r'[$,()]', '', value_str)
    
    try:
        result = float(cleaned)
        return -result if is_negative else result
    except ValueError:
        return 0.0


def detect_csv_format(df: pd.DataFrame, raw_text: str = None) -> str:
    """Detect the format of the uploaded CSV"""
    
    # Check for TradeStation format markers
    if raw_text:
        if 'TradeStation Performance Summary' in raw_text or 'TradeStation Trades List' in raw_text:
            return 'tradestation'
    
    # Check column names for common formats
    columns_lower = [str(c).lower() for c in df.columns]
    
    if any('tradestation' in c for c in columns_lower):
        return 'tradestation'
    
    # NinjaTrader format detection
    if 'instrument' in columns_lower and 'market pos.' in columns_lower:
        return 'ninjatrader'
    
    # Generic format with P&L column
    pnl_columns = ['pnl', 'profit', 'profit_loss', 'net_profit', 'p&l', 'net profit', 'profit/loss']
    if any(col in columns_lower for col in pnl_columns):
        return 'generic'
    
    return 'unknown'


def parse_tradestation_csv(raw_text: str) -> List[Dict]:
    """
    Parse TradeStation performance report CSV.
    Extracts individual trade P&L from the Trades List section.
    
    TradeStation format has:
    - Entry rows with trade number in column 0 (1, 2, 3...)
    - Exit rows with empty column 0
    - P&L is in column 7 of ENTRY rows
    """
    trades = []
    lines = raw_text.split('\n')
    
    # Find the "TradeStation Trades List" section
    trades_section_start = None
    for i, line in enumerate(lines):
        if 'TradeStation Trades List' in line:
            trades_section_start = i
            break
    
    if trades_section_start is None:
        raise ValueError("Could not find 'TradeStation Trades List' section in the file")
    
    # Find the header row (contains #,Type,Date/Time,...)
    header_row = None
    for i in range(trades_section_start, min(trades_section_start + 10, len(lines))):
        if '#,Type,Date/Time' in lines[i] or '#, Type, Date/Time' in lines[i]:
            header_row = i
            break
    
    if header_row is None:
        raise ValueError("Could not find trades header row")
    
    # Parse trades starting after header
    for i in range(header_row + 2, len(lines)):  # Skip header and empty line
        line = lines[i].strip()
        
        # Stop at next section or empty content
        if not line:
            continue
        if 'AEA ' in line or 'Parameters' in line:
            break
        
        # Split the line by comma
        parts = line.split(',')
        
        if len(parts) < 8:
            continue
        
        # Entry rows have a trade number in the first column (1, 2, 3, etc.)
        first_col = parts[0].strip()
        
        # Check if this is an entry row (has numeric trade number)
        if first_col.isdigit():
            try:
                # Column layout for entry rows:
                # 0: Trade #
                # 1: Type (Buy, Sell Short)
                # 2: Date/Time
                # 3: Signal
                # 4: Price
                # 5: Roll Over
                # 6: Shares/Contracts
                # 7: Profit/Loss (THIS IS WHAT WE WANT)
                
                date_str = parts[2].strip() if len(parts) > 2 else ''
                pnl_str = parts[7].strip() if len(parts) > 7 else '0'
                
                pnl = parse_currency(pnl_str)
                
                trade_date = None
                if date_str:
                    try:
                        trade_date = pd.to_datetime(date_str)
                    except:
                        trade_date = None
                
                trades.append({
                    'pnl': pnl,
                    'date': trade_date
                })
                
            except Exception as e:
                continue  # Skip problematic rows
    
    if not trades:
        raise ValueError("No trades found in the TradeStation report. Please ensure the file contains trade data.")
    
    return trades


def parse_ninjatrader_csv(df: pd.DataFrame) -> List[Dict]:
    """Parse NinjaTrader trade export format"""
    trades = []
    
    # Common NinjaTrader column names
    pnl_cols = ['Profit', 'Net P&L', 'P&L', 'Net Profit']
    date_cols = ['Exit time', 'Exit Time', 'Time', 'Date']
    
    pnl_col = None
    date_col = None
    
    for col in df.columns:
        if col in pnl_cols:
            pnl_col = col
        if col in date_cols:
            date_col = col
    
    if pnl_col is None:
        raise ValueError("Could not find P&L column in NinjaTrader export")
    
    for idx, row in df.iterrows():
        trade = {
            'pnl': parse_currency(row[pnl_col]),
            'date': None
        }
        
        if date_col and pd.notna(row[date_col]):
            try:
                trade['date'] = pd.to_datetime(row[date_col])
            except:
                pass
        
        if trade['pnl'] != 0:
            trades.append(trade)
    
    return trades


def parse_generic_csv(df: pd.DataFrame) -> List[Dict]:
    """
    Parse generic CSV with P&L column.
    Attempts to detect column names automatically.
    """
    # Common column name mappings
    date_columns = ['date', 'datetime', 'time', 'entry_time', 'entry_date', 'exit_time', 'exit_date',
                    'Date', 'DateTime', 'Time', 'Entry Time', 'Exit Time']
    pnl_columns = ['pnl', 'profit', 'profit_loss', 'net_profit', 'P&L', 'PnL', 'Profit', 
                   'Net Profit', 'profit/loss', 'Profit/Loss', 'Net P&L', 'NetProfit']
    
    # Find date column
    date_col = None
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    # Find PnL column
    pnl_col = None
    for col in pnl_columns:
        if col in df.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        # Try to find any column with 'profit' or 'pnl' in the name
        for col in df.columns:
            col_lower = str(col).lower()
            if 'profit' in col_lower or 'pnl' in col_lower or 'p&l' in col_lower:
                pnl_col = col
                break
    
    if pnl_col is None:
        raise ValueError(
            "Could not find a P&L column. Please ensure your CSV has a column named "
            "'PnL', 'Profit', 'Net Profit', or similar."
        )
    
    trades = []
    for idx, row in df.iterrows():
        pnl_value = row[pnl_col]
        pnl = parse_currency(pnl_value)
        
        trade = {
            'pnl': pnl,
            'date': None
        }
        
        if date_col and pd.notna(row[date_col]):
            try:
                trade['date'] = pd.to_datetime(row[date_col])
            except:
                trade['date'] = None
        
        if pnl != 0:
            trades.append(trade)
    
    return trades


def parse_trades_from_csv(df: pd.DataFrame, raw_text: str = None, platform_hint: str = None) -> List[Dict]:
    """
    Parse uploaded CSV into standardized trade format.
    
    Args:
        df: pandas DataFrame (may be None for TradeStation)
        raw_text: Raw CSV text (required for TradeStation)
        platform_hint: Optional hint for platform type ('TradeStation', 'NinjaTrader', 'Generic')
    """
    # Use platform hint if provided, otherwise auto-detect
    if platform_hint:
        csv_format = platform_hint.lower()
    else:
        csv_format = detect_csv_format(df, raw_text)
    
    if csv_format == 'tradestation' and raw_text:
        return parse_tradestation_csv(raw_text)
    elif csv_format == 'ninjatrader':
        return parse_ninjatrader_csv(df)
    else:
        return parse_generic_csv(df)


def group_trades_by_day(trades: List[Dict]) -> List[Dict]:
    """Group trades into daily P&L"""
    if not trades:
        return []
    
    # Check if we have valid dates
    has_dates = any(t.get('date') is not None for t in trades)
    
    if not has_dates:
        # If no dates, treat each trade as a separate "day"
        return [{'date': i, 'daily_pnl': t['pnl'], 'trades': [t]} for i, t in enumerate(trades)]
    
    # Group by date
    daily = {}
    for trade in trades:
        if trade.get('date') is not None:
            date_key = trade['date'].date()
        else:
            date_key = 'unknown'
        
        if date_key not in daily:
            daily[date_key] = {'date': date_key, 'daily_pnl': 0, 'trades': []}
        daily[date_key]['daily_pnl'] += trade['pnl']
        daily[date_key]['trades'].append(trade)
    
    # Sort by date
    sorted_daily = sorted(
        [v for v in daily.values() if v['date'] != 'unknown'],
        key=lambda x: x['date']
    )
    
    # Add unknown date trades at the end
    if 'unknown' in daily:
        sorted_daily.append(daily['unknown'])
    
    return sorted_daily


def run_single_simulation(
    daily_pnls: List[float],
    rules: Dict,
    starting_balance: float
) -> SimulationResult:
    """
    Run a single simulation with shuffled daily P&L sequence.
    
    Args:
        daily_pnls: List of daily P&L values
        rules: Prop firm rules dictionary
        starting_balance: Account starting balance
    
    Returns:
        SimulationResult with pass/fail and metrics
    """
    # Shuffle the daily P&Ls (block bootstrap - keeps daily integrity)
    shuffled_pnls = daily_pnls.copy()
    random.shuffle(shuffled_pnls)
    
    # Initialize tracking variables
    equity = starting_balance
    high_water_mark = starting_balance
    equity_curve = [equity]
    max_drawdown = 0
    trading_days = 0
    daily_profits = []
    
    # Rule thresholds
    profit_target = rules.get('profit_target', float('inf'))
    max_trailing_dd = rules.get('max_trailing_drawdown', float('inf'))
    daily_loss_limit = rules.get('daily_loss_limit')
    trailing_type = rules.get('trailing_drawdown_type', 'end_of_day')
    trailing_stops_at = rules.get('trailing_stops_at')
    
    failure_reason = None
    target_reached = False
    days_to_target = 0
    
    for day_idx, daily_pnl in enumerate(shuffled_pnls):
        trading_days += 1
        daily_profits.append(daily_pnl)
        
        # Check daily loss limit BEFORE applying P&L (soft breach in most firms)
        if daily_loss_limit and daily_pnl < -daily_loss_limit:
            failure_reason = f"Daily Loss Limit Breached (lost ${abs(daily_pnl):,.0f}, limit ${daily_loss_limit:,})"
            break
        
        # Apply daily P&L
        equity += daily_pnl
        equity_curve.append(equity)
        
        # Update high water mark based on trailing type
        if trailing_type == 'end_of_day':
            # EOD trailing - only updates at end of day
            if trailing_stops_at and high_water_mark >= trailing_stops_at:
                pass  # Don't update high water mark anymore
            else:
                high_water_mark = max(high_water_mark, equity)
        else:
            # Intraday trailing - updates in real-time (more aggressive)
            high_water_mark = max(high_water_mark, equity)
        
        # Calculate current drawdown from high water mark
        current_dd = high_water_mark - equity
        max_drawdown = max(max_drawdown, current_dd)
        
        # Check trailing drawdown violation
        if current_dd > max_trailing_dd:
            dd_type = "Intraday" if trailing_type == 'intraday' else "EOD"
            failure_reason = f"Max Drawdown Exceeded ({dd_type}: ${current_dd:,.0f} > ${max_trailing_dd:,} limit)"
            break
        
        # Check if profit target reached
        profit = equity - starting_balance
        if profit >= profit_target and not target_reached:
            target_reached = True
            days_to_target = trading_days
    
    # Check minimum trading days
    min_days = rules.get('min_trading_days', 0)
    if target_reached and trading_days < min_days:
        # Need to continue trading to meet minimum days
        # For simulation purposes, we'll just note this
        days_to_target = max(days_to_target, min_days)
    
    # Check consistency rule
    if rules.get('consistency_rule') and target_reached and failure_reason is None:
        max_day_pct = rules.get('consistency_max_day_percent', 50)
        total_profit = sum(p for p in daily_profits if p > 0)
        if total_profit > 0:
            best_day = max(daily_profits)
            best_day_pct = (best_day / total_profit) * 100
            if best_day_pct > max_day_pct:
                failure_reason = f"Consistency Rule Violated (best day {best_day_pct:.0f}% > {max_day_pct}% limit)"
                target_reached = False
    
    # Add failure reason for not reaching profit target
    if not target_reached and failure_reason is None:
        final_profit = equity - starting_balance
        pct_of_target = (final_profit / profit_target) * 100 if profit_target > 0 else 0
        failure_reason = f"Profit Target Not Reached (${final_profit:,.0f} = {pct_of_target:.0f}% of ${profit_target:,} target)"
    
    # Determine pass/fail
    passed = target_reached and failure_reason is None
    
    return SimulationResult(
        passed=passed,
        final_equity=equity,
        max_drawdown=max_drawdown,
        days_to_target=days_to_target if target_reached else 0,
        failure_reason=failure_reason,
        equity_curve=equity_curve,
        daily_pnl=daily_profits
    )


def run_monte_carlo_simulation(
    trades: List[Dict],
    rules: Dict,
    num_simulations: int = 200
) -> AggregateResults:
    """
    Run Monte Carlo simulation with specified number of iterations.
    
    Args:
        trades: List of trade dictionaries with 'pnl' and optionally 'date'
        rules: Prop firm rules dictionary
        num_simulations: Number of simulation runs
    
    Returns:
        AggregateResults with pass rate and statistics
    """
    # Group trades by day
    daily_data = group_trades_by_day(trades)
    daily_pnls = [d['daily_pnl'] for d in daily_data]
    
    if len(daily_pnls) < 5:
        raise ValueError(f"Need at least 5 trading days of data for meaningful simulation. Found {len(daily_pnls)} days.")
    
    starting_balance = rules.get('account_size', 50000)
    
    # Run simulations
    results = []
    for _ in range(num_simulations):
        result = run_single_simulation(daily_pnls, rules, starting_balance)
        results.append(result)
    
    # Aggregate results
    pass_count = sum(1 for r in results if r.passed)
    pass_rate = pass_count / num_simulations
    
    # Calculate statistics for passing simulations
    passing_results = [r for r in results if r.passed]
    if passing_results:
        avg_days = np.mean([r.days_to_target for r in passing_results])
        median_days = np.median([r.days_to_target for r in passing_results])
    else:
        avg_days = 0
        median_days = 0
    
    avg_max_dd = np.mean([r.max_drawdown for r in results])
    
    # Count failure reasons with cleaner categories
    failure_reasons = {}
    for r in results:
        if r.failure_reason:
            # Extract the main category from the failure reason
            reason = r.failure_reason
            if "Max Drawdown Exceeded" in reason:
                category = "Max Drawdown Exceeded"
            elif "Daily Loss Limit" in reason:
                category = "Daily Loss Limit Breached"
            elif "Consistency Rule" in reason:
                category = "Consistency Rule Violated"
            elif "Profit Target Not Reached" in reason:
                category = "Profit Target Not Reached"
            else:
                category = reason.split('(')[0].strip()
            
            failure_reasons[category] = failure_reasons.get(category, 0) + 1
    
    # Get sample equity curves (up to 50 for visualization with user selection)
    sample_curves = [r.equity_curve for r in results[:50]]
    sample_daily_pnls = [r.daily_pnl for r in results[:50]]
    
    return AggregateResults(
        total_simulations=num_simulations,
        pass_count=pass_count,
        pass_rate=pass_rate,
        avg_days_to_target=avg_days,
        median_days_to_target=median_days,
        avg_max_drawdown=avg_max_dd,
        failure_reasons=failure_reasons,
        equity_curves=sample_curves,
        daily_pnls=sample_daily_pnls
    )


def run_single_simulation_sequential(
    daily_pnls: List[float],
    rules: Dict,
    starting_balance: float,
    start_index: int = 0
) -> SimulationResult:
    """
    Run a single simulation with ACTUAL trade sequence (no shuffling).
    Starts from a specific index in the daily P&L list.
    
    Args:
        daily_pnls: List of daily P&L values (in original order)
        rules: Prop firm rules dictionary
        starting_balance: Account starting balance
        start_index: Which day to start from (for rolling window)
    
    Returns:
        SimulationResult with pass/fail and metrics
    """
    # Use trades starting from start_index
    sequence_pnls = daily_pnls[start_index:]
    
    # Initialize tracking variables
    equity = starting_balance
    high_water_mark = starting_balance
    equity_curve = [equity]
    max_drawdown = 0
    trading_days = 0
    daily_profits = []
    
    # Rule thresholds
    profit_target = rules.get('profit_target', float('inf'))
    max_trailing_dd = rules.get('max_trailing_drawdown', float('inf'))
    daily_loss_limit = rules.get('daily_loss_limit')
    trailing_type = rules.get('trailing_drawdown_type', 'end_of_day')
    trailing_stops_at = rules.get('trailing_stops_at')
    
    failure_reason = None
    target_reached = False
    days_to_target = 0
    
    for day_idx, daily_pnl in enumerate(sequence_pnls):
        trading_days += 1
        daily_profits.append(daily_pnl)
        
        # Check daily loss limit BEFORE applying P&L (soft breach in most firms)
        if daily_loss_limit and daily_pnl < -daily_loss_limit:
            failure_reason = f"Daily Loss Limit Breached (lost ${abs(daily_pnl):,.0f}, limit ${daily_loss_limit:,})"
            break
        
        # Apply daily P&L
        equity += daily_pnl
        equity_curve.append(equity)
        
        # Update high water mark based on trailing type
        if trailing_type == 'end_of_day':
            if trailing_stops_at and high_water_mark >= trailing_stops_at:
                pass
            else:
                high_water_mark = max(high_water_mark, equity)
        else:
            high_water_mark = max(high_water_mark, equity)
        
        # Calculate current drawdown from high water mark
        current_dd = high_water_mark - equity
        max_drawdown = max(max_drawdown, current_dd)
        
        # Check trailing drawdown violation
        if current_dd > max_trailing_dd:
            dd_type = "Intraday" if trailing_type == 'intraday' else "EOD"
            failure_reason = f"Max Drawdown Exceeded ({dd_type}: ${current_dd:,.0f} > ${max_trailing_dd:,} limit)"
            break
        
        # Check if profit target reached
        profit = equity - starting_balance
        if profit >= profit_target and not target_reached:
            target_reached = True
            days_to_target = trading_days
            break  # Stop once target is reached (challenge passed)
    
    # Check minimum trading days
    min_days = rules.get('min_trading_days', 0)
    if target_reached and trading_days < min_days:
        days_to_target = max(days_to_target, min_days)
    
    # Check consistency rule
    if rules.get('consistency_rule') and target_reached and failure_reason is None:
        max_day_pct = rules.get('consistency_max_day_percent', 50)
        total_profit = sum(p for p in daily_profits if p > 0)
        if total_profit > 0:
            best_day = max(daily_profits)
            best_day_pct = (best_day / total_profit) * 100
            if best_day_pct > max_day_pct:
                failure_reason = f"Consistency Rule Violated (best day {best_day_pct:.0f}% > {max_day_pct}% limit)"
                target_reached = False
    
    # Add failure reason for not reaching profit target
    if not target_reached and failure_reason is None:
        final_profit = equity - starting_balance
        pct_of_target = (final_profit / profit_target) * 100 if profit_target > 0 else 0
        failure_reason = f"Profit Target Not Reached (${final_profit:,.0f} = {pct_of_target:.0f}% of ${profit_target:,} target)"
    
    passed = target_reached and failure_reason is None
    
    return SimulationResult(
        passed=passed,
        final_equity=equity,
        max_drawdown=max_drawdown,
        days_to_target=days_to_target if target_reached else 0,
        failure_reason=failure_reason,
        equity_curve=equity_curve,
        daily_pnl=daily_profits
    )


def run_rolling_window_simulation(
    trades: List[Dict],
    rules: Dict,
    num_windows: int = 200
) -> AggregateResults:
    """
    Run rolling window simulation - tests starting the challenge at different dates.
    Preserves actual trade sequence (no shuffling).
    
    Args:
        trades: List of trade dictionaries
        rules: Prop firm rules dictionary
        num_windows: Number of different start dates to test
    
    Returns:
        AggregateResults with pass rate and statistics
    """
    # Group trades by day
    daily_data = group_trades_by_day(trades)
    daily_pnls = [d['daily_pnl'] for d in daily_data]
    
    if len(daily_pnls) < 10:
        raise ValueError(f"Need at least 10 trading days for rolling window simulation. Found {len(daily_pnls)} days.")
    
    starting_balance = rules.get('account_size', 50000)
    
    # Calculate step size to spread windows across the data
    # Leave at least 30 days at the end for a meaningful test
    min_days_needed = 30
    usable_days = max(1, len(daily_pnls) - min_days_needed)
    
    # Generate evenly spaced start indices
    if num_windows >= usable_days:
        start_indices = list(range(usable_days))
    else:
        step = usable_days / num_windows
        start_indices = [int(i * step) for i in range(num_windows)]
    
    # Run simulations
    results = []
    for start_idx in start_indices:
        result = run_single_simulation_sequential(daily_pnls, rules, starting_balance, start_idx)
        results.append(result)
    
    # Aggregate results
    pass_count = sum(1 for r in results if r.passed)
    pass_rate = pass_count / len(results)
    
    passing_results = [r for r in results if r.passed]
    if passing_results:
        avg_days = np.mean([r.days_to_target for r in passing_results])
        median_days = np.median([r.days_to_target for r in passing_results])
    else:
        avg_days = 0
        median_days = 0
    
    avg_max_dd = np.mean([r.max_drawdown for r in results])
    
    # Count failure reasons with cleaner categories
    failure_reasons = {}
    for r in results:
        if r.failure_reason:
            reason = r.failure_reason
            if "Max Drawdown Exceeded" in reason:
                category = "Max Drawdown Exceeded"
            elif "Daily Loss Limit" in reason:
                category = "Daily Loss Limit Breached"
            elif "Consistency Rule" in reason:
                category = "Consistency Rule Violated"
            elif "Profit Target Not Reached" in reason:
                category = "Profit Target Not Reached"
            else:
                category = reason.split('(')[0].strip()
            
            failure_reasons[category] = failure_reasons.get(category, 0) + 1
    
    # Get sample equity curves (up to 50 for visualization with user selection)
    sample_curves = [r.equity_curve for r in results[:50]]
    sample_daily_pnls = [r.daily_pnl for r in results[:50]]
    
    return AggregateResults(
        total_simulations=len(results),
        pass_count=pass_count,
        pass_rate=pass_rate,
        avg_days_to_target=avg_days,
        median_days_to_target=median_days,
        avg_max_drawdown=avg_max_dd,
        failure_reasons=failure_reasons,
        equity_curves=sample_curves,
        daily_pnls=sample_daily_pnls
    )


def run_actual_sequence_test(
    trades: List[Dict],
    rules: Dict
) -> SimulationResult:
    """
    Test your EXACT track record against prop firm rules.
    No shuffling, no rolling - just your actual sequence from day 1.
    
    Returns:
        SimulationResult for your actual track record
    """
    daily_data = group_trades_by_day(trades)
    daily_pnls = [d['daily_pnl'] for d in daily_data]
    starting_balance = rules.get('account_size', 50000)
    
    return run_single_simulation_sequential(daily_pnls, rules, starting_balance, start_index=0)


@dataclass
class PayoutSimulationResult:
    """Results from a funded account payout simulation"""
    passed_eval: bool
    days_to_pass_eval: int
    days_to_first_payout: int
    total_payouts: int
    total_withdrawn: float
    total_kept_after_split: float
    avg_payout_amount: float
    account_blown: bool
    blown_on_day: int
    final_account_balance: float
    payout_history: List[Dict]  # List of {day, amount, balance_after}


def simulate_funded_account_with_payouts(
    trades: List[Dict],
    rules: Dict,
    max_days: int = None
) -> PayoutSimulationResult:
    """
    Simulate passing eval THEN trading the funded account with payouts.
    Uses actual trade sequence (no shuffling).
    
    This answers: "If I passed the eval, how many payouts could I take 
    and how much money would I actually withdraw?"
    """
    daily_data = group_trades_by_day(trades)
    daily_pnls = [d['daily_pnl'] for d in daily_data]
    
    if max_days:
        daily_pnls = daily_pnls[:max_days]
    
    starting_balance = rules.get('account_size', 50000)
    profit_target = rules.get('profit_target', float('inf'))
    max_trailing_dd = rules.get('max_trailing_drawdown', float('inf'))
    daily_loss_limit = rules.get('daily_loss_limit')
    trailing_type = rules.get('trailing_drawdown_type', 'end_of_day')
    
    # Payout rules
    payout_rules = rules.get('payout_rules', {})
    min_profitable_days = payout_rules.get('min_profitable_days', 5)
    min_profit_per_day = payout_rules.get('min_profit_per_day', 100)
    days_between_payouts = payout_rules.get('days_between_payouts', 5)
    payout_consistency = payout_rules.get('consistency_percent')
    first_payout_cap = payout_rules.get('first_payout_cap', 2000)
    subsequent_payout_cap = payout_rules.get('subsequent_payout_cap', 4000)
    max_payout_percent = payout_rules.get('max_payout_percent')  # e.g., 50% for Topstep
    min_balance_buffer = payout_rules.get('min_balance_buffer', 100)
    mll_resets_on_payout = payout_rules.get('mll_resets_on_payout', False)
    buffer_resets_on_payout = payout_rules.get('buffer_resets_on_payout', False)
    profit_split_first = payout_rules.get('profit_split_first', 10000)
    profit_split_after = payout_rules.get('profit_split_after', 90)
    
    # Phase 1: Evaluation
    equity = starting_balance
    high_water_mark = starting_balance
    eval_passed = False
    days_to_pass = 0
    
    for day_idx, daily_pnl in enumerate(daily_pnls):
        # Check DLL
        if daily_loss_limit and daily_pnl < -daily_loss_limit:
            return PayoutSimulationResult(
                passed_eval=False, days_to_pass_eval=0, days_to_first_payout=0,
                total_payouts=0, total_withdrawn=0, total_kept_after_split=0,
                avg_payout_amount=0, account_blown=True, blown_on_day=day_idx+1,
                final_account_balance=equity + daily_pnl, payout_history=[]
            )
        
        equity += daily_pnl
        
        if trailing_type == 'end_of_day':
            high_water_mark = max(high_water_mark, equity)
        else:
            high_water_mark = max(high_water_mark, equity)
        
        current_dd = high_water_mark - equity
        if current_dd > max_trailing_dd:
            return PayoutSimulationResult(
                passed_eval=False, days_to_pass_eval=0, days_to_first_payout=0,
                total_payouts=0, total_withdrawn=0, total_kept_after_split=0,
                avg_payout_amount=0, account_blown=True, blown_on_day=day_idx+1,
                final_account_balance=equity, payout_history=[]
            )
        
        profit = equity - starting_balance
        if profit >= profit_target and not eval_passed:
            eval_passed = True
            days_to_pass = day_idx + 1
            break
    
    if not eval_passed:
        return PayoutSimulationResult(
            passed_eval=False, days_to_pass_eval=0, days_to_first_payout=0,
            total_payouts=0, total_withdrawn=0, total_kept_after_split=0,
            avg_payout_amount=0, account_blown=False, blown_on_day=0,
            final_account_balance=equity, payout_history=[]
        )
    
    # Phase 2: Funded Account with Payouts
    # Reset for funded phase - start from where eval ended
    funded_start_day = days_to_pass
    remaining_pnls = daily_pnls[funded_start_day:]
    
    # Funded account starts fresh
    funded_balance = starting_balance + (equity - starting_balance)  # Keep eval profits
    high_water_mark = funded_balance
    min_balance = starting_balance + min_balance_buffer  # Can't go below this
    
    # For MLL tracking (especially Topstep where MLL resets to $0 after payout)
    current_mll = max_trailing_dd  # Start with full drawdown allowance
    
    profitable_days_count = 0
    days_since_last_payout = 0
    total_payouts = 0
    total_withdrawn = 0.0
    total_kept = 0.0
    payout_history = []
    days_to_first_payout = 0
    account_blown = False
    blown_on_day = 0
    
    # Track profits for consistency rule
    profits_since_last_payout = []
    
    for day_idx, daily_pnl in enumerate(remaining_pnls):
        actual_day = funded_start_day + day_idx + 1
        days_since_last_payout += 1
        
        # Check DLL
        if daily_loss_limit and daily_pnl < -daily_loss_limit:
            account_blown = True
            blown_on_day = actual_day
            break
        
        funded_balance += daily_pnl
        profits_since_last_payout.append(daily_pnl)
        
        # Track profitable days
        if daily_pnl >= min_profit_per_day:
            profitable_days_count += 1
        
        # Update HWM
        high_water_mark = max(high_water_mark, funded_balance)
        
        # Check drawdown - use current MLL (may have been reset after payout)
        if mll_resets_on_payout and total_payouts > 0:
            # Topstep: After first payout, MLL is $0 (can't go below starting balance)
            if funded_balance <= starting_balance:
                account_blown = True
                blown_on_day = actual_day
                break
        else:
            current_dd = high_water_mark - funded_balance
            if current_dd > current_mll:
                account_blown = True
                blown_on_day = actual_day
                break
        
        # Check if below minimum balance
        if funded_balance < min_balance:
            account_blown = True
            blown_on_day = actual_day
            break
        
        # Check payout eligibility
        can_payout = (
            profitable_days_count >= min_profitable_days and
            days_since_last_payout >= days_between_payouts and
            funded_balance > min_balance
        )
        
        # Check consistency rule for payout
        if can_payout and payout_consistency:
            total_profit = sum(p for p in profits_since_last_payout if p > 0)
            if total_profit > 0:
                best_day = max(profits_since_last_payout)
                best_day_pct = (best_day / total_profit) * 100
                if best_day_pct > payout_consistency:
                    can_payout = False
        
        if can_payout:
            # Calculate available for withdrawal
            available = funded_balance - min_balance
            
            # Apply payout cap
            if total_payouts == 0:
                payout_cap = first_payout_cap
            else:
                payout_cap = subsequent_payout_cap
            
            # Apply percentage cap if exists (Topstep 50% rule)
            if max_payout_percent:
                pct_cap = available * (max_payout_percent / 100)
                payout_cap = min(payout_cap, pct_cap)
            
            payout_amount = min(available, payout_cap)
            
            if payout_amount >= 100:  # Minimum payout threshold
                # Calculate profit split
                if total_withdrawn < profit_split_first:
                    # 100% of first X profits
                    kept = payout_amount
                else:
                    # Split after threshold
                    kept = payout_amount * (profit_split_after / 100)
                
                # Execute payout
                funded_balance -= payout_amount
                total_payouts += 1
                total_withdrawn += payout_amount
                total_kept += kept
                
                payout_history.append({
                    'day': actual_day,
                    'amount': payout_amount,
                    'kept': kept,
                    'balance_after': funded_balance
                })
                
                if days_to_first_payout == 0:
                    days_to_first_payout = actual_day
                
                # Reset counters
                profitable_days_count = 0
                days_since_last_payout = 0
                profits_since_last_payout = []
                
                # Handle MLL reset (Topstep)
                if mll_resets_on_payout:
                    # MLL goes to $0 - can't lose below starting balance
                    high_water_mark = funded_balance
                
                # Handle buffer reset (MFF)
                if buffer_resets_on_payout:
                    high_water_mark = funded_balance
    
    avg_payout = total_withdrawn / total_payouts if total_payouts > 0 else 0
    
    return PayoutSimulationResult(
        passed_eval=True,
        days_to_pass_eval=days_to_pass,
        days_to_first_payout=days_to_first_payout,
        total_payouts=total_payouts,
        total_withdrawn=total_withdrawn,
        total_kept_after_split=total_kept,
        avg_payout_amount=avg_payout,
        account_blown=account_blown,
        blown_on_day=blown_on_day,
        final_account_balance=funded_balance,
        payout_history=payout_history
    )


def get_trade_statistics(trades: List[Dict]) -> Dict:
    """Calculate basic statistics from trade data"""
    pnls = [t['pnl'] for t in trades]
    
    if not pnls:
        return {}
    
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    
    return {
        'total_trades': len(pnls),
        'total_pnl': sum(pnls),
        'avg_trade': np.mean(pnls),
        'win_rate': len(winners) / len(pnls) * 100 if pnls else 0,
        'avg_winner': np.mean(winners) if winners else 0,
        'avg_loser': np.mean(losers) if losers else 0,
        'profit_factor': abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf'),
        'max_winner': max(winners) if winners else 0,
        'max_loser': min(losers) if losers else 0,
        'std_dev': np.std(pnls)
    }
