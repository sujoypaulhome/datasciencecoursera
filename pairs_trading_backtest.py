#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pairs-trading walk-forward backtest with SIDE_MODE, realized & unrealized PnL,
and per-trade candlestick + spread context plots.
"""

import json, warnings
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

import matplotlib.pyplot as plt
import mplfinance as mpf

# ─── CONFIG ─────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/eod")
MIN_DAYS   = 90         # lookback window
TOP_N      = 50
COINT_PVAL = 0.05
PROFIT_THR = 0.10       # 10% profit threshold on spread
SIDE_MODE  = "both"     # "both" | "buy" | "sell"
LOOKBACK   = 30         # bars to show in context plot
# ──────────────────────────────────────────────────────────────────────────

def load_ohlc():
    ohlc = {}
    for fn in sorted(DATA_DIR.glob("*.json")):
        sym = fn.stem
        recs = json.loads(fn.read_text()).get("data", [])
        df = pd.DataFrame(recs)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = (df.dropna(subset=["date"])
                .drop_duplicates("date")
                .set_index("date")
                .sort_index())
        if not {"open","high","low","close"}.issubset(df.columns):
            continue
        sub = df[["open","high","low","close"]]
        if len(sub) >= MIN_DAYS + 1:  # need MIN_DAYS history + trade_day
            ohlc[sym] = sub
    return ohlc

def compute_top_correlated(ohlc_hist):
    closes = pd.DataFrame({s:df["close"] for s,df in ohlc_hist.items()})
    rets   = closes.pct_change().dropna(how="all")
    corr   = rets.corr().stack().reset_index()
    corr.columns = ["sym1","sym2","rho"]
    corr = corr[corr.sym1 < corr.sym2]
    top  = corr.nlargest(TOP_N,"rho")[["sym1","sym2"]].values.tolist()
    return top

def find_cointegrated(pairs, closes_hist):
    cis = []
    for s1,s2 in pairs:
        df = pd.concat([closes_hist[s1], closes_hist[s2]], axis=1).dropna()
        if len(df) < MIN_DAYS:
            continue
        _, pval, _ = coint(df[s1], df[s2])
        if pval < COINT_PVAL:
            beta = sm.OLS(df[s1], sm.add_constant(df[s2])).fit().params[1]
            cis.append((s1, s2, beta))
    return cis

def plot_trade_context(ohlc_all, trade, lookback=LOOKBACK):
    """
    3-panel context plot for a single trade (closed or open).
    trade: dict with keys "pair","entry","beta"
    """
    s1,s2 = trade["pair"].split("/")
    beta   = trade["beta"]
    entry  = trade["entry"]

    df1 = ohlc_all[s1].loc[:entry].copy().iloc[-lookback:]
    df2 = ohlc_all[s2].loc[:entry].copy().iloc[-lookback:]

    spread = df1["close"] - beta*df2["close"]
    ent_str = entry.strftime("%Y-%m-%d")

    fig, axes = plt.subplots(3,1,figsize=(10,8),
                             gridspec_kw={"height_ratios":[2,2,1]},
                             sharex=True)

    # s1 candles + entry marker
    mk1 = pd.Series(np.nan, index=df1.index)
    if entry in df1.index:
        mk1.at[entry] = df1.at[entry,"open"]
    ap1 = mpf.make_addplot(mk1, type="scatter", marker="v", markersize=100,
                            color="red", ax=axes[0])
    mpf.plot(df1.rename(columns=str.title), type="candle",
             ax=axes[0], addplot=[ap1], show_nontrading=False)
    axes[0].set_title(f"{s1} up to {ent_str}")

    # s2 candles + entry marker
    mk2 = pd.Series(np.nan, index=df2.index)
    if entry in df2.index:
        mk2.at[entry] = df2.at[entry,"open"]
    ap2 = mpf.make_addplot(mk2, type="scatter", marker="v", markersize=100,
                            color="red", ax=axes[1])
    mpf.plot(df2.rename(columns=str.title), type="candle",
             ax=axes[1], addplot=[ap2], show_nontrading=False)
    axes[1].set_title(f"{s2} up to {ent_str}")

    # spread + entry line
    axes[2].plot(spread.index, spread, color="black", label="spread")
    axes[2].axvline(entry, color="purple", linestyle="--", label="entry")
    axes[2].legend(loc="upper left")
    axes[2].set_title(f"Spread = {s1} – {beta:.2f}×{s2}")

    plt.tight_layout()
    plt.show()

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Load OHLC
    ohlc_all = load_ohlc()
    if not ohlc_all:
        print("No symbols with enough data.")
        return

    all_opens  = pd.DataFrame({s:df["open"]  for s,df in ohlc_all.items()})
    all_closes = pd.DataFrame({s:df["close"] for s,df in ohlc_all.items()})
    dates      = sorted(all_closes.index)

    trades_all     = []
    open_positions = []

    # Walk-forward loop
    for idx in range(MIN_DAYS, len(dates)-1):
        hist = dates[idx-MIN_DAYS:idx]
        trade_day = dates[idx]
        future    = dates[idx+1:]

        # restrict syms
        syms = [s for s,df in ohlc_all.items()
                if set(hist+[trade_day]).issubset(df.index)]
        if len(syms)<2:
            continue

        # pick top correlated & cointegrated
        hist_ohlc   = {s:ohlc_all[s].loc[hist] for s in syms}
        pairs       = compute_top_correlated(hist_ohlc)
        closes_hist = all_closes.loc[hist, syms]
        cis         = find_cointegrated(pairs, closes_hist)
        if not cis:
            continue

        for s1,s2,beta in cis:
            spread = closes_hist[s1] - beta*closes_hist[s2]
            d0,d1,d2 = spread.iloc[-3], spread.iloc[-2], spread.iloc[-1]
            if not (d0<d1 and d1>d2):
                continue

            # entry
            e = trade_day
            o1e, o2e = all_opens.at[e,s1], all_opens.at[e,s2]
            direction = 1 if d1>0 else -1
            if SIDE_MODE=="buy"  and direction<0: continue
            if SIDE_MODE=="sell" and direction>0: continue

            threshold = abs(d1)*PROFIT_THR
            exited = False

            # search exit
            for ed in future:
                sp_new = all_opens.at[ed,s1] - beta*all_opens.at[ed,s2]
                move   = direction*(sp_new - d1)
                if move >= threshold:
                    if direction>0:
                        pnl1 = all_opens.at[ed,s1] - o1e
                        pnl2 = o2e - all_opens.at[ed,s2]
                    else:
                        pnl1 = all_opens.at[ed,s2] - o2e
                        pnl2 = o1e - all_opens.at[ed,s1]
                    total = pnl1 + pnl2
                    trades_all.append({
                        "pair":  f"{s1}/{s2}",
                        "entry": e,
                        "exit":  ed,
                        "pnl":   total,
                        "beta":  beta
                    })
                    exited = True
                    break

            if not exited:
                open_positions.append({
                    "pair":      f"{s1}/{s2}",
                    "entry":     e,
                    "beta":      beta,
                    "o1_e":      o1e,
                    "o2_e":      o2e,
                    "direction": direction
                })

    # Print closed trades
    realized_total = 0.0
    print(f"\n=== Closed Trades ({len(trades_all)}) ===")
    for tr in trades_all:
        e_date = tr["entry"].date()
        x_date = tr["exit"].date()
        s1,s2  = tr["pair"].split("/")
        p1_e = all_opens.at[tr["entry"], s1]
        p2_e = all_opens.at[tr["entry"], s2]
        p1_x = all_opens.at[tr["exit"] , s1]
        p2_x = all_opens.at[tr["exit"] , s2]
        pnl  = tr["pnl"]
        realized_total += pnl

        print(f"{e_date} : {s1} : BUY  : {p1_e:.2f}")
        print(f"{e_date} : {s2} : SELL : {p2_e:.2f}")
        print(f"{x_date} : {s1} : SELL : {p1_x:.2f}")
        print(f"{x_date} : {s2} : BUY  : {p2_x:.2f}")
        print(f"{s1}/{s2}: Trade PnL = {pnl:.2f}\n")

    print(f"Total Realized PnL: {realized_total:.2f}\n")

    # Print open positions
    if open_positions:
        last_day = dates[-1].date()
        unreal_total = 0.0
        print(f"=== Open Positions as of {last_day} ({len(open_positions)}) ===")
        for op in open_positions:
            s1,s2 = op["pair"].split("/")
            e_date= op["entry"].date()
            o1e   = op["o1_e"]
            o2e   = op["o2_e"]
            beta  = op["beta"]
            dirn  = op["direction"]

            p1_now = all_opens.at[dates[-1], s1]
            p2_now = all_opens.at[dates[-1], s2]

            if dirn>0:
                pnl1 = p1_now - o1e
                pnl2 = o2e   - p2_now
            else:
                pnl1 = p2_now - o2e
                pnl2 = o1e   - p1_now
            total = pnl1 + pnl2
            unreal_total += total

            print(f"{e_date} : {s1} : BUY     : {o1e:.2f}")
            print(f"{last_day} : {s1} : current : {p1_now:.2f}  unreal PnL = {pnl1:.2f}")
            print(f"{e_date} : {s2} : SELL    : {o2e:.2f}")
            print(f"{last_day} : {s2} : current : {p2_now:.2f}  unreal PnL = {pnl2:.2f}")
            print(f"{s1}/{s2}: Unrealized PnL = {total:.2f}\n")

        print(f"Total Unrealized PnL: {unreal_total:.2f}\n")
    else:
        print("No Open Positions.\n")

    # Plot contexts
    for tr in trades_all:
        print(f"Plotting closed trade {tr['pair']} entry {tr['entry'].date()}")
        plot_trade_context(ohlc_all, tr)

    for op in open_positions:
        print(f"Plotting open position {op['pair']} entry {op['entry'].date()}")
        plot_trade_context(ohlc_all, op)

if __name__=="__main__":
    main()