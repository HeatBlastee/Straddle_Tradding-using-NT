"""
Simulation engine for the Streamlit UI.
Runs offline backtests of ShortStraddleStrategy with synthetic BTC options
under various scenarios (Take-Profit, Stop-Loss, Inside Band, Unfilled Leg Abort)
and captures all C-level and Python NautilusTrader engine logs.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Tuple
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import BTC, USDT
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType, OmsType, OptionKind
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoOption
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from straddle_strategy import ShortStraddleConfig, ShortStraddleStrategy, State

VENUE = Venue("BYBIT")
T0 = pd.Timestamp("2026-11-07 00:00:00", tz="UTC")
EXPIRY = pd.Timestamp("2026-11-28 08:00:00", tz="UTC")
NEXT_EXPIRY = pd.Timestamp("2026-12-26 08:00:00", tz="UTC")
SOON_EXPIRY = pd.Timestamp("2026-11-07 08:00:00", tz="UTC")


def capture_c_logs(func: Callable) -> Tuple[Any, List[str]]:
    """Captures C-level stdout/stderr output produced by NautilusTrader Cython/Rust."""
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()
    out_fd = os.open(tf.name, os.O_WRONLY)
    err_fd = os.open(tf.name, os.O_WRONLY)

    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    os.dup2(out_fd, 1)
    os.dup2(err_fd, 2)

    result = None
    try:
        result = func()
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(out_fd)
        os.close(err_fd)
        os.close(old_stdout)
        os.close(old_stderr)

    with open(tf.name, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\r\n") for line in f.readlines()]
    try:
        os.remove(tf.name)
    except OSError:
        pass

    return result, lines


def make_option(strike: int, kind: OptionKind, expiry: pd.Timestamp) -> CryptoOption:
    k = "C" if kind == OptionKind.CALL else "P"
    sym = f"BTC-{expiry.strftime('%d%b%y').upper()}-{strike}-{k}-USDT-OPTION"
    return CryptoOption(
        instrument_id=InstrumentId(Symbol(sym), VENUE),
        raw_symbol=Symbol(sym),
        underlying=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        option_kind=kind,
        strike_price=Price.from_str(f"{strike}.0"),
        activation_ns=int(T0.value),
        expiration_ns=int(expiry.value),
        price_precision=1,
        size_precision=2,
        price_increment=Price.from_str("0.5"),
        size_increment=Quantity.from_str("0.01"),
        ts_event=0,
        ts_init=0,
        min_quantity=Quantity.from_str("0.01"),
        margin_init=Decimal("0.1"),
        margin_maint=Decimal("0.05"),
    )


class ForceNoFillPutStrategy(ShortStraddleStrategy):
    """Priced put sell far above the market so it never fills."""
    def _entry_price(self, leg, quote):
        base = quote.bid_price.as_double()
        return base + 500 if leg == "PUT" else base


class TracingStraddleStrategy(ShortStraddleStrategy):
    """Strategy that records step-by-step history for visualization."""
    def __init__(self, config: ShortStraddleConfig) -> None:
        super().__init__(config)
        self.history: List[Dict[str, Any]] = []

    def on_quote_tick(self, tick: QuoteTick) -> None:
        super().on_quote_tick(tick)
        self._record_step(tick.ts_event)

    def _record_step(self, ts_ns: int) -> None:
        bb = float(self.buyback_value()) if self.buyback_value() is not None else None
        entry = float(self.entry_premium) if self.entry_premium is not None else None
        pnl = None
        if entry is not None and bb is not None:
            pnl = (entry - bb) * float(self.qty_target)

        self.history.append({
            "timestamp_ns": ts_ns,
            "seconds": (ts_ns - int(T0.value)) / 1e9,
            "phase": self.phase.value,
            "ref_price": self.ref_price,
            "buyback_value": bb,
            "entry_premium": entry,
            "pnl": pnl,
            "short_call": float(self.short_qty.get("CALL", 0)),
            "short_put": float(self.short_qty.get("PUT", 0)),
        })


def _q(inst, t, bid, ask) -> QuoteTick:
    ts = int((T0 + pd.Timedelta(seconds=t)).value)
    return QuoteTick(
        instrument_id=inst.id,
        bid_price=inst.make_price(bid),
        ask_price=inst.make_price(ask),
        bid_size=inst.make_qty(100),
        ask_size=inst.make_qty(100),
        ts_event=ts,
        ts_init=ts,
    )


def run_scenario(
    scenario_type: str,
    quantity: float = 0.10,
    stop_loss_pct: float = 0.01,
    take_profit_pct: float = 0.02,
    ref_mid: float = 95100.0,
) -> Dict[str, Any]:
    def _exec():
        flat = lambda bid, ask, n=30: [(bid, ask)] * n

        strategy_cls = TracingStraddleStrategy
        if scenario_type == "take_profit":
            c_path = flat(1000.0, 1004.0, 5) + flat(890.0, 900.0, 25)
            p_path = flat(900.0, 904.0, 5) + flat(890.0, 900.0, 25)
        elif scenario_type == "stop_loss":
            c_path = flat(1000.0, 1010.0, 5) + flat(1020.0, 1030.0, 25)
            p_path = flat(900.0, 910.0, 5) + flat(910.0, 920.0, 25)
        elif scenario_type == "inside_band":
            c_path = flat(1000.0, 1004.0, 30)
            p_path = flat(900.0, 904.0, 30)
        elif scenario_type == "unfilled_abort":
            c_path = flat(1000.0, 1010.0, 30)
            p_path = flat(900.0, 910.0, 30)
            strategy_cls = ForceNoFillPutStrategy
        else:
            raise ValueError(f"Unknown scenario: {scenario_type}")

        perp = TestInstrumentProvider.btcusdt_perp_binance()
        chain = []
        for strike in (94_000, 95_000, 96_000):
            for kind in (OptionKind.CALL, OptionKind.PUT):
                chain.append(make_option(strike, kind, EXPIRY))
        chain.append(make_option(95_000, OptionKind.CALL, NEXT_EXPIRY))
        chain.append(make_option(95_000, OptionKind.PUT, NEXT_EXPIRY))
        chain.append(make_option(95_100, OptionKind.CALL, SOON_EXPIRY))
        chain.append(make_option(95_100, OptionKind.PUT, SOON_EXPIRY))

        engine = BacktestEngine(
            config=BacktestEngineConfig(logging=LoggingConfig(log_level="INFO")),
        )
        engine.add_venue(
            venue=VENUE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=None,
            starting_balances=[Money(100_000, USDT)],
        )
        engine.add_venue(
            venue=perp.id.venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=None,
            starting_balances=[Money(100_000, USDT)],
        )
        engine.add_instrument(perp)
        for inst in chain:
            engine.add_instrument(inst)

        call = make_option(95_000, OptionKind.CALL, EXPIRY)
        put = make_option(95_000, OptionKind.PUT, EXPIRY)

        data = [
            _q(perp, 0, ref_mid - 0.5, ref_mid + 0.5),
            _q(call, 0.5, *c_path[0]),
            _q(put, 0.5, *p_path[0]),
        ]
        for t in range(1, len(c_path)):
            data.append(_q(perp, t + 1, ref_mid - 0.5, ref_mid + 0.5))
            data.append(_q(call, t + 1, *c_path[t]))
            data.append(_q(put, t + 1, *p_path[t]))

        engine.add_data(data)

        config = ShortStraddleConfig(
            venue="BYBIT",
            reference_instrument_id=str(perp.id),
            quantity=str(quantity),
            stop_loss_pct=str(stop_loss_pct),
            take_profit_pct=str(take_profit_pct),
            entry_timeout_secs=5.0,
            exit_reprice_secs=3.0,
        )
        strat = strategy_cls(config)
        engine.add_strategy(strat)
        engine.run()

        # Collect orders
        orders_data = []
        for o in engine.cache.orders():
            role = strat._order_role.get(o.client_order_id, ("UNKNOWN", "UNKNOWN"))
            orders_data.append({
                "client_order_id": str(o.client_order_id),
                "instrument": str(o.instrument_id),
                "leg": role[0],
                "role": role[1],
                "side": o.side.name,
                "quantity": float(o.quantity),
                "price": float(o.price) if o.price else 0.0,
                "status": o.status.name,
                "filled_qty": float(o.filled_qty),
            })

        history_records = getattr(strat, "history", [])
        df_history = pd.DataFrame(history_records) if history_records else pd.DataFrame()

        entry_px = float(strat.entry_premium) if strat.entry_premium else None
        sl_target = round(entry_px * (1 + stop_loss_pct), 2) if entry_px else None
        tp_target = round(entry_px * (1 - take_profit_pct), 2) if entry_px else None

        final_pnl = None
        if entry_px is not None:
            if strat.exit_reason and "buyback=" in strat.exit_reason:
                try:
                    bb_str = strat.exit_reason.split("buyback=")[1].split()[0]
                    bb_val = float(bb_str)
                    final_pnl = round((entry_px - bb_val) * quantity, 4)
                except Exception:
                    pass

        res = {
            "scenario": scenario_type,
            "final_state": strat.phase.value,
            "exit_reason": strat.exit_reason,
            "abort_reason": strat._abort_reason,
            "entry_premium": entry_px,
            "stop_loss_target": sl_target,
            "take_profit_target": tp_target,
            "final_pnl": final_pnl,
            "ref_price": strat.ref_price,
            "legs": {k: str(v) for k, v in strat.legs.items()},
            "open_positions": len(engine.cache.positions_open()),
            "short_qty": {k: float(v) for k, v in strat.short_qty.items()},
            "orders": orders_data,
            "timeline": df_history,
        }
        engine.dispose()
        return res

    result, logs = capture_c_logs(_exec)
    result["logs"] = logs
    return result


def run_full_test_suite() -> Dict[str, Any]:
    """Runs all 21 checks from test_straddle.py and captures complete results and engine logs."""
    from test_straddle import run as run_test_case, make_option, ForceNoFillPutStrategy

    checks: List[Dict[str, Any]] = []

    def _exec():
        flat = lambda bid, ask, n=30: [(bid, ask)] * n

        # 1. ATM Selection
        engine, s, call, put = run_test_case(
            ShortStraddleStrategy, flat(1000.0, 1010.0), flat(900.0, 910.0), enable_risk=False
        )
        checks.append({"name": "ATM Strike Selection", "desc": "Strike 95000 is nearest to ref 95100", "pass": "95000" in str(s.legs["CALL"])})
        checks.append({"name": "Expiry Selection", "desc": "Nearest valid expiry used (28NOV26, >24h)", "pass": "28NOV26" in str(s.legs["CALL"])})
        checks.append({"name": "Strike & Expiry Match", "desc": "Call & Put share same strike & expiry", "pass": str(s.legs["CALL"]).replace("-C-", "-P-") == str(s.legs["PUT"])})
        checks.append({"name": "State Progression", "desc": "State transitions to ACTIVE upon fills", "pass": s.phase == State.ACTIVE})
        checks.append({"name": "Entry Premium Sum", "desc": f"Entry premium = bid_c + bid_p (${s.entry_premium})", "pass": s.entry_premium == Decimal("1900.0")})
        checks.append({"name": "Short Position Sizing", "desc": "Short 0.10 BTC on both legs", "pass": s.short_qty == {"CALL": Decimal("0.10"), "PUT": Decimal("0.10")}})
        checks.append({"name": "Single Sell Execution", "desc": "Exactly one sell order submitted per leg", "pass": len([o for o in engine.cache.orders() if o.instrument_id == call.id]) == 1})
        checks.append({"name": "Engine Positions Verified", "desc": "Both legs registered as short in engine cache", "pass": len(engine.cache.positions_open()) == 2})
        engine.dispose()

        # 2. Take-Profit
        c = flat(1000.0, 1004.0, 5) + flat(890.0, 900.0, 25)
        p = flat(900.0, 904.0, 5) + flat(890.0, 900.0, 25)
        engine, s, call, put = run_test_case(ShortStraddleStrategy, c, p)
        checks.append({"name": "Take-Profit Trigger", "desc": "Strategy closes positions and finishes DONE", "pass": s.phase == State.DONE})
        checks.append({"name": "TP Reason Recorded", "desc": f"Exit reason confirms TP: {s.exit_reason}", "pass": bool(s.exit_reason and "TAKE-PROFIT" in s.exit_reason)})
        checks.append({"name": "TP Flattening", "desc": "Zero open positions remaining after TP exit", "pass": len(engine.cache.positions_open()) == 0})
        checks.append({"name": "TP Order Count", "desc": "Exactly 2 orders per leg (1 entry sell + 1 exit buy)", "pass": len([o for o in engine.cache.orders() if o.instrument_id == call.id]) == 2})
        engine.dispose()

        # 3. Stop-Loss
        c = flat(1000.0, 1010.0, 5) + flat(1020.0, 1030.0, 25)
        p = flat(900.0, 910.0, 5) + flat(910.0, 920.0, 25)
        engine, s, call, put = run_test_case(ShortStraddleStrategy, c, p)
        checks.append({"name": "Stop-Loss Trigger", "desc": "Strategy closes positions and finishes DONE", "pass": s.phase == State.DONE})
        checks.append({"name": "SL Reason Recorded", "desc": f"Exit reason confirms SL: {s.exit_reason}", "pass": bool(s.exit_reason and "STOP-LOSS" in s.exit_reason)})
        checks.append({"name": "SL Flattening", "desc": "Zero open positions remaining after SL exit", "pass": len(engine.cache.positions_open()) == 0})
        buys = [o for o in engine.cache.orders() if o.side.name == "BUY"]
        sells = [o for o in engine.cache.orders() if o.side.name == "SELL"]
        checks.append({"name": "Duplicate Exit Guard", "desc": "Exactly 1 buy per leg despite continuous triggering quotes", "pass": len(buys) == 2 and len(sells) == 2})
        checks.append({"name": "Reduce-Only Limit Exits", "desc": "All exit orders are marked reduce_only=True", "pass": all(o.is_reduce_only for o in buys)})
        engine.dispose()

        # 4. Inside Band
        c = flat(1000.0, 1004.0, 30)
        p = flat(900.0, 904.0, 30)
        engine, s, call, put = run_test_case(ShortStraddleStrategy, c, p)
        checks.append({"name": "Band Guard (Active)", "desc": "Stays ACTIVE with 0 exit orders when quotes are safe", "pass": s.phase == State.ACTIVE and len([o for o in engine.cache.orders() if o.side.name == "BUY"]) == 0})
        engine.dispose()

        # 5. One Leg Timeout Abort
        engine, s, call, put = run_test_case(ForceNoFillPutStrategy, flat(1000.0, 1010.0), flat(900.0, 910.0))
        checks.append({"name": "Unfilled Order Cancellation", "desc": "Unfilled put sell cancelled on entry timeout", "pass": any(o.instrument_id == put.id and o.side.name == "SELL" and o.status.name == "CANCELED" for o in engine.cache.orders())})
        checks.append({"name": "Filled Leg Liquidation", "desc": "Filled call leg bought back immediately", "pass": any(o.instrument_id == call.id and o.side.name == "BUY" for o in engine.cache.orders())})
        checks.append({"name": "Unhedged Protection Complete", "desc": "Ends 100% flat and DONE", "pass": s.phase == State.DONE and len(engine.cache.positions_open()) == 0})
        engine.dispose()

        return True

    t_start = time.time()
    _, raw_logs = capture_c_logs(_exec)
    t_end = time.time()

    all_passed = all(c["pass"] for c in checks)

    return {
        "all_passed": all_passed,
        "total": len(checks),
        "passed": sum(1 for c in checks if c["pass"]),
        "failed": sum(1 for c in checks if not c["pass"]),
        "duration_sec": round(t_end - t_start, 2),
        "checks": checks,
        "logs": raw_logs,
    }
