# BTC ATM Short Straddle - NautilusTrader + Bybit

Files
- app.py              : Streamlit UI dashboard
- bot_runner.py       : Background TradingNode runner with JSON state bridge
- sim_engine.py       : Offline scenario simulation engine for UI
- straddle_strategy.py: the strategy (Level 1 + Level 2)
- run_live.py         : TradingNode + Bybit config and launcher
- test_straddle.py    : offline tests (synthetic options, simulated exchange)

Setup & Running the UI
    pip install -r requirements.txt
    cp .env.example .env               # configure your BYBIT_API_KEY and BYBIT_API_SECRET in .env
    streamlit run app.py               # launches the interactive dashboard at http://localhost:8501

Offline Tests
    python test_straddle.py            # offline, no keys needed -> ALL CHECKS PASSED
    python run_live.py                 # loads keys automatically from .env

Option instrument ids look like  BTC-28NOV26-95000-C-USDT-OPTION.BYBIT
(the -OPTION suffix is required by the adapter).

Flow (for the walkthrough)
1. run_live builds TradingNodeConfig: Bybit data + exec client, products OPTION+LINEAR,
   instrument provider load_all with filter base_coin=BTC.
2. On connect the provider loads BTC option contracts + BTCUSDT perp into the Cache.
3. on_start subscribes quotes on the perp (reference price = mid).
4. First perp quote -> _select_straddle: group cached options by expiry/strike,
   skip expiries < 24h away, take nearest expiry that has call AND put, strike nearest ref.
5. Subscribe quotes on both legs; once both have a live quote -> _maybe_enter sells both
   (limit at bid, marketable). Phase flips to ENTERING first so entry happens once.
6. on_order_accepted / on_order_filled / on_order_rejected track the lifecycle; fills
   accumulate per leg (partials supported); both legs full -> ACTIVE, entry premium =
   sum of per-leg average fill prices.
7. Level 2: every option quote while ACTIVE: buyback = ask_call + ask_put.
   SL if buyback >= entry*1.01, TP if buyback <= entry*0.98.
8. Exit: reduce-only BUY limits at the ask for only the still-open qty.
   Duplicate guards: phase check (one exit sequence) + one live exit order per leg.
   Watchdog timer cancels stale exit orders and re-prices at the new ask; gives up
   (FAILED) after max_exit_attempts.
9. One leg filled, other not: after entry_timeout the working leg is cancelled and the
   filled leg is bought back, so you are never left half-hedged.
