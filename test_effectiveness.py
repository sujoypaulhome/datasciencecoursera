#!/usr/bin/env python3
"""
Comprehensive effectiveness testing for the pairs trading strategy.
Generates synthetic data and analyzes strategy performance.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
import seaborn as sns

# Import the original strategy functions
from pairs_trading_backtest import main as run_backtest, load_ohlc
import pairs_trading_backtest as ptb

def generate_synthetic_pairs_data(n_symbols=10, n_days=500, start_date='2020-01-01'):
    """
    Generate synthetic OHLC data with known cointegrated pairs for testing.
    """
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    base_price = 100
    
    # Create data directory
    data_dir = Path("data/eod")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    
    # Generate base random walks
    returns = np.random.normal(0.001, 0.02, (n_days, n_symbols))
    
    # Create some cointegrated pairs by making certain symbols follow each other
    # Pair 1: SYM00 and SYM01 (strong cointegration)
    returns[:, 1] = 0.8 * returns[:, 0] + 0.2 * returns[:, 1]
    
    # Pair 2: SYM02 and SYM03 (moderate cointegration)  
    returns[:, 3] = 0.6 * returns[:, 2] + 0.4 * returns[:, 3]
    
    # Pair 3: SYM04 and SYM05 (weak cointegration)
    returns[:, 5] = 0.4 * returns[:, 4] + 0.6 * returns[:, 5]
    
    for i, symbol in enumerate(symbols):
        # Generate price series
        prices = base_price * np.exp(np.cumsum(returns[:, i]))
        
        # Add some volatility to create realistic OHLC
        volatility = 0.01
        highs = prices * (1 + np.abs(np.random.normal(0, volatility, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0, volatility, n_days)))
        
        # Ensure OHLC relationships are maintained
        opens = prices + np.random.normal(0, volatility * prices, n_days)
        closes = prices + np.random.normal(0, volatility * prices, n_days)
        
        # Fix any violations of OHLC constraints
        for j in range(n_days):
            low = min(opens[j], closes[j], lows[j])
            high = max(opens[j], closes[j], highs[j])
            lows[j] = low
            highs[j] = high
        
        # Create data structure
        data = []
        for j in range(n_days):
            data.append({
                "date": dates[j].strftime("%Y-%m-%d"),
                "open": round(opens[j], 2),
                "high": round(highs[j], 2),
                "low": round(lows[j], 2),
                "close": round(closes[j], 2),
                "volume": np.random.randint(100000, 1000000)
            })
        
        # Save to JSON file
        with open(data_dir / f"{symbol}.json", 'w') as f:
            json.dump({"data": data}, f, indent=2)
    
    print(f"Generated synthetic data for {n_symbols} symbols over {n_days} days")
    print(f"Known cointegrated pairs: SYM00/SYM01, SYM02/SYM03, SYM04/SYM05")

def calculate_performance_metrics(trades_data, initial_capital=100000):
    """
    Calculate comprehensive performance metrics for the strategy.
    """
    if not trades_data:
        return {
            "total_return": 0,
            "total_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "profit_factor": 0
        }
    
    pnls = [trade["pnl"] for trade in trades_data]
    total_pnl = sum(pnls)
    
    # Basic metrics
    total_trades = len(trades_data)
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Calculate cumulative returns for drawdown and Sharpe
    cumulative_pnl = np.cumsum([0] + pnls)
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdown = peak - cumulative_pnl
    max_drawdown = np.max(drawdown)
    
    # Total return percentage
    total_return = (total_pnl / initial_capital) * 100
    
    # Sharpe ratio (simplified - using PnL variance)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0
    
    # Profit factor
    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 1
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    return {
        "total_return": total_return,
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "win_rate": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "pnls": pnls
    }

def run_strategy_with_different_parameters():
    """
    Test the strategy with different parameter combinations.
    """
    # Suppress plotting for parameter testing
    original_plot_func = ptb.plot_trade_context
    ptb.plot_trade_context = lambda *args, **kwargs: None
    
    param_combinations = [
        {"PROFIT_THR": 0.05, "SIDE_MODE": "both", "COINT_PVAL": 0.05},
        {"PROFIT_THR": 0.10, "SIDE_MODE": "both", "COINT_PVAL": 0.05},
        {"PROFIT_THR": 0.15, "SIDE_MODE": "both", "COINT_PVAL": 0.05},
        {"PROFIT_THR": 0.10, "SIDE_MODE": "buy", "COINT_PVAL": 0.05},
        {"PROFIT_THR": 0.10, "SIDE_MODE": "sell", "COINT_PVAL": 0.05},
        {"PROFIT_THR": 0.10, "SIDE_MODE": "both", "COINT_PVAL": 0.01},
        {"PROFIT_THR": 0.10, "SIDE_MODE": "both", "COINT_PVAL": 0.10},
    ]
    
    results = []
    
    for params in param_combinations:
        print(f"\nTesting parameters: {params}")
        
        # Update global parameters
        ptb.PROFIT_THR = params["PROFIT_THR"]
        ptb.SIDE_MODE = params["SIDE_MODE"]
        ptb.COINT_PVAL = params["COINT_PVAL"]
        
        # Capture trades by modifying the main function
        trades_captured = []
        
        def capture_trades():
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            ohlc_all = load_ohlc()
            if not ohlc_all:
                return []
            
            all_opens = pd.DataFrame({s: df["open"] for s, df in ohlc_all.items()})
            all_closes = pd.DataFrame({s: df["close"] for s, df in ohlc_all.items()})
            dates = sorted(all_closes.index)
            
            trades_all = []
            
            for idx in range(ptb.MIN_DAYS, len(dates) - 1):
                hist = dates[idx - ptb.MIN_DAYS:idx]
                trade_day = dates[idx]
                future = dates[idx + 1:]
                
                syms = [s for s, df in ohlc_all.items()
                        if set(hist + [trade_day]).issubset(df.index)]
                if len(syms) < 2:
                    continue
                
                hist_ohlc = {s: ohlc_all[s].loc[hist] for s in syms}
                pairs = ptb.compute_top_correlated(hist_ohlc)
                closes_hist = all_closes.loc[hist, syms]
                cis = ptb.find_cointegrated(pairs, closes_hist)
                if not cis:
                    continue
                
                for s1, s2, beta in cis:
                    spread = closes_hist[s1] - beta * closes_hist[s2]
                    d0, d1, d2 = spread.iloc[-3], spread.iloc[-2], spread.iloc[-1]
                    if not (d0 < d1 and d1 > d2):
                        continue
                    
                    e = trade_day
                    o1e, o2e = all_opens.at[e, s1], all_opens.at[e, s2]
                    direction = 1 if d1 > 0 else -1
                    if ptb.SIDE_MODE == "buy" and direction < 0: continue
                    if ptb.SIDE_MODE == "sell" and direction > 0: continue
                    
                    threshold = abs(d1) * ptb.PROFIT_THR
                    
                    for ed in future:
                        sp_new = all_opens.at[ed, s1] - beta * all_opens.at[ed, s2]
                        move = direction * (sp_new - d1)
                        if move >= threshold:
                            if direction > 0:
                                pnl1 = all_opens.at[ed, s1] - o1e
                                pnl2 = o2e - all_opens.at[ed, s2]
                            else:
                                pnl1 = all_opens.at[ed, s2] - o2e
                                pnl2 = o1e - all_opens.at[ed, s1]
                            total = pnl1 + pnl2
                            trades_all.append({
                                "pair": f"{s1}/{s2}",
                                "entry": e,
                                "exit": ed,
                                "pnl": total,
                                "beta": beta
                            })
                            break
            
            return trades_all
        
        trades = capture_trades()
        metrics = calculate_performance_metrics(trades)
        
        result = {**params, **metrics}
        results.append(result)
        
        print(f"Total trades: {metrics['total_trades']}")
        print(f"Total return: {metrics['total_return']:.2f}%")
        print(f"Win rate: {metrics['win_rate']:.1f}%")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Restore original plot function
    ptb.plot_trade_context = original_plot_func
    
    return results

def plot_effectiveness_analysis(results):
    """
    Create visualizations for strategy effectiveness analysis.
    """
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Total Return by Profit Threshold
    profit_thr_results = df_results[df_results['SIDE_MODE'] == 'both']
    axes[0, 0].bar(profit_thr_results['PROFIT_THR'], profit_thr_results['total_return'])
    axes[0, 0].set_title('Total Return by Profit Threshold')
    axes[0, 0].set_xlabel('Profit Threshold')
    axes[0, 0].set_ylabel('Total Return (%)')
    
    # 2. Win Rate by Side Mode
    side_mode_results = df_results[df_results['PROFIT_THR'] == 0.10]
    axes[0, 1].bar(side_mode_results['SIDE_MODE'], side_mode_results['win_rate'])
    axes[0, 1].set_title('Win Rate by Side Mode')
    axes[0, 1].set_xlabel('Side Mode')
    axes[0, 1].set_ylabel('Win Rate (%)')
    
    # 3. Sharpe Ratio Comparison
    axes[0, 2].bar(range(len(results)), df_results['sharpe_ratio'])
    axes[0, 2].set_title('Sharpe Ratio by Parameter Set')
    axes[0, 2].set_xlabel('Parameter Set')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].set_xticks(range(len(results)))
    
    # 4. Trade Count vs Return
    axes[1, 0].scatter(df_results['total_trades'], df_results['total_return'])
    axes[1, 0].set_title('Trade Count vs Total Return')
    axes[1, 0].set_xlabel('Number of Trades')
    axes[1, 0].set_ylabel('Total Return (%)')
    
    # 5. Profit Factor by Cointegration P-value
    coint_results = df_results[df_results['SIDE_MODE'] == 'both'][df_results['PROFIT_THR'] == 0.10]
    axes[1, 1].bar(coint_results['COINT_PVAL'], coint_results['profit_factor'])
    axes[1, 1].set_title('Profit Factor by Cointegration P-value')
    axes[1, 1].set_xlabel('Cointegration P-value')
    axes[1, 1].set_ylabel('Profit Factor')
    
    # 6. Max Drawdown Analysis
    axes[1, 2].bar(range(len(results)), df_results['max_drawdown'])
    axes[1, 2].set_title('Max Drawdown by Parameter Set')
    axes[1, 2].set_xlabel('Parameter Set')
    axes[1, 2].set_ylabel('Max Drawdown')
    axes[1, 2].set_xticks(range(len(results)))
    
    plt.tight_layout()
    plt.savefig('strategy_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_report(results):
    """
    Generate a comprehensive performance report.
    """
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("PAIRS TRADING STRATEGY EFFECTIVENESS REPORT")
    print("="*80)
    
    # Best performing configurations
    best_return = df.loc[df['total_return'].idxmax()]
    best_sharpe = df.loc[df['sharpe_ratio'].idxmax()]
    best_win_rate = df.loc[df['win_rate'].idxmax()]
    
    print(f"\n📈 BEST TOTAL RETURN ({best_return['total_return']:.2f}%)")
    print(f"   Parameters: Profit Threshold={best_return['PROFIT_THR']}, "
          f"Side Mode={best_return['SIDE_MODE']}, Coint P-val={best_return['COINT_PVAL']}")
    print(f"   Trades: {best_return['total_trades']}, Win Rate: {best_return['win_rate']:.1f}%")
    
    print(f"\n📊 BEST SHARPE RATIO ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"   Parameters: Profit Threshold={best_sharpe['PROFIT_THR']}, "
          f"Side Mode={best_sharpe['SIDE_MODE']}, Coint P-val={best_sharpe['COINT_PVAL']}")
    print(f"   Total Return: {best_sharpe['total_return']:.2f}%, Trades: {best_sharpe['total_trades']}")
    
    print(f"\n🎯 BEST WIN RATE ({best_win_rate['win_rate']:.1f}%)")
    print(f"   Parameters: Profit Threshold={best_win_rate['PROFIT_THR']}, "
          f"Side Mode={best_win_rate['SIDE_MODE']}, Coint P-val={best_win_rate['COINT_PVAL']}")
    print(f"   Total Return: {best_win_rate['total_return']:.2f}%, Trades: {best_win_rate['total_trades']}")
    
    # Overall statistics
    print(f"\n📋 OVERALL STATISTICS")
    print(f"   Average Return: {df['total_return'].mean():.2f}% (±{df['total_return'].std():.2f}%)")
    print(f"   Average Sharpe: {df['sharpe_ratio'].mean():.2f} (±{df['sharpe_ratio'].std():.2f})")
    print(f"   Average Win Rate: {df['win_rate'].mean():.1f}% (±{df['win_rate'].std():.1f}%)")
    print(f"   Average Trades per Config: {df['total_trades'].mean():.1f}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    
    # Profit threshold analysis
    both_mode = df[df['SIDE_MODE'] == 'both']
    if len(both_mode) > 1:
        best_thr = both_mode.loc[both_mode['total_return'].idxmax(), 'PROFIT_THR']
        print(f"   • Optimal profit threshold appears to be {best_thr} for maximum returns")
    
    # Side mode analysis
    side_comparison = df.groupby('SIDE_MODE')['total_return'].mean()
    best_side = side_comparison.idxmax()
    print(f"   • Best side mode: {best_side} (avg return: {side_comparison[best_side]:.2f}%)")
    
    # Cointegration threshold analysis
    coint_comparison = df.groupby('COINT_PVAL')['total_return'].mean()
    best_coint = coint_comparison.idxmax()
    print(f"   • Best cointegration p-value: {best_coint} (avg return: {coint_comparison[best_coint]:.2f}%)")
    
    print(f"\n⚠️  RISK ASSESSMENT")
    high_drawdown = df[df['max_drawdown'] > df['max_drawdown'].mean() + df['max_drawdown'].std()]
    if len(high_drawdown) > 0:
        print(f"   • {len(high_drawdown)} configurations showed high drawdown risk")
    
    low_trades = df[df['total_trades'] < 5]
    if len(low_trades) > 0:
        print(f"   • {len(low_trades)} configurations generated very few trades (< 5)")
    
    print("="*80)

def main():
    """
    Main function to run comprehensive effectiveness testing.
    """
    print("Starting Pairs Trading Strategy Effectiveness Testing")
    print("="*60)
    
    # Generate synthetic test data
    print("\n1. Generating synthetic test data...")
    generate_synthetic_pairs_data(n_symbols=12, n_days=400)
    
    # Run strategy with different parameters
    print("\n2. Testing strategy with different parameter combinations...")
    results = run_strategy_with_different_parameters()
    
    # Create visualizations
    print("\n3. Creating effectiveness visualizations...")
    plot_effectiveness_analysis(results)
    
    # Generate comprehensive report
    print("\n4. Generating performance report...")
    create_performance_report(results)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('strategy_effectiveness_results.csv', index=False)
    print(f"\n5. Results saved to 'strategy_effectiveness_results.csv'")
    
    print(f"\n✅ Effectiveness testing completed!")
    print(f"   Generated {len(results)} parameter combinations")
    print(f"   Total trades across all tests: {results_df['total_trades'].sum()}")
    print(f"   Best configuration achieved {results_df['total_return'].max():.2f}% return")

if __name__ == "__main__":
    main()