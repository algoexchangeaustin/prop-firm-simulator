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


def parse_trades_from_csv(df: pd.DataFrame) -> List[Dict]:
    """
    Parse uploaded CSV into standardized trade format.
    Attempts to detect column names automatically.
    """
    # Common column name mappings
    date_columns = ['date', 'datetime', 'time', 'entry_time', 'entry_date', 'Date', 'DateTime', 'Time']
    pnl_columns = ['pnl', 'profit', 'profit_loss', 'net_profit', 'P&L', 'PnL', 'Profit', 'Net Profit', 'profit/loss']
    
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
            if 'profit' in col.lower() or 'pnl' in col.lower() or 'p&l' in col.lower():
                pnl_col = col
                break
    
    if pnl_col is None:
        raise ValueError("Could not find a P&L column. Please ensure your CSV has a column named 'PnL', 'Profit', or 'Net Profit'")
    
    trades = []
    for idx, row in df.iterrows():
        trade = {
            'pnl': float(row[pnl_col]) if pd.notna(row[pnl_col]) else 0.0,
            'date': None
        }
        
        if date_col and pd.notna(row[date_col]):
            try:
                trade['date'] = pd.to_datetime(row[date_col])
            except:
                trade['date'] = None
        
        trades.append(trade)
    
    return trades


def group_trades_by_day(trades: List[Dict]) -> List[Dict]:
    """Group trades into daily P&L"""
    if not trades or trades[0]['date'] is None:
        # If no dates, treat each trade as a separate "day"
        return [{'date': i, 'daily_pnl': t['pnl'], 'trades': [t]} for i, t in enumerate(trades)]
    
    # Group by date
    daily = {}
    for trade in trades:
        date_key = trade['date'].date() if trade['date'] else 'unknown'
        if date_key not in daily:
            daily[date_key] = {'date': date_key, 'daily_pnl': 0, 'trades': []}
        daily[date_key]['daily_pnl'] += trade['pnl']
        daily[date_key]['trades'].append(trade)
    
    return sorted(daily.values(), key=lambda x: str(x['date']))


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
        
        # Check daily loss limit BEFORE applying P&L
        if daily_loss_limit and daily_pnl < -daily_loss_limit:
            failure_reason = f"Daily loss limit exceeded (${abs(daily_pnl):.0f} > ${daily_loss_limit})"
            break
        
        # Apply daily P&L
        equity += daily_pnl
        equity_curve.append(equity)
        
        # Update high water mark (end of day for EOD trailing)
        if trailing_type == 'end_of_day':
            # Check if trailing should stop
            if trailing_stops_at and high_water_mark >= trailing_stops_at:
                pass  # Don't update high water mark anymore
            else:
                high_water_mark = max(high_water_mark, equity)
        else:
            # Real-time trailing (more aggressive)
            high_water_mark = max(high_water_mark, equity)
        
        # Calculate current drawdown from high water mark
        current_dd = high_water_mark - equity
        max_drawdown = max(max_drawdown, current_dd)
        
        # Check trailing drawdown violation
        if current_dd > max_trailing_dd:
            failure_reason = f"Trailing drawdown exceeded (${current_dd:.0f} > ${max_trailing_dd})"
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
            for day_profit in daily_profits:
                if day_profit > 0:
                    day_pct = (day_profit / total_profit) * 100
                    if day_pct > max_day_pct:
                        failure_reason = f"Consistency rule violated ({day_pct:.1f}% > {max_day_pct}% max)"
                        target_reached = False
                        break
    
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
        raise ValueError("Need at least 5 trading days of data for meaningful simulation")
    
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
    
    # Count failure reasons
    failure_reasons = {}
    for r in results:
        if r.failure_reason:
            reason_type = r.failure_reason.split('(')[0].strip()
            failure_reasons[reason_type] = failure_reasons.get(reason_type, 0) + 1
    
    # Get sample equity curves (first 10 for visualization)
    sample_curves = [r.equity_curve for r in results[:10]]
    sample_daily_pnls = [r.daily_pnl for r in results[:10]]
    
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
