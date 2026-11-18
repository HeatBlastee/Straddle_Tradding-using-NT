"""
Offline checks for the straddle strategy using NautilusTrader's BacktestEngine and
synthetic BTC options (no Bybit connection needed).

Run:  python test_straddle.py
"""

from __future__ import annotations

import pandas as pd
from decimal import Decimal

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import BTC
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.enums import OptionKind
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Symbol
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import CryptoOption
from nautilus_trader.model.objects import Money
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from straddle_strategy import ShortStraddleConfig
from straddle_strategy import ShortStraddleStrategy
from straddle_strategy import State

VENUE = Venue("BYBIT")
T0 = pd.Timestamp("2026-11-07 00:00:00", tz="UTC")
EXPIRY = pd.Timestamp("2026-11-28 08:00:00", tz="UTC")
NEXT_EXPIRY = pd.Timestamp("2026-12-26 08:00:00", tz="UTC")
SOON_EXPIRY = pd.Timestamp("2026-11-07 08:00:00", tz="UTC")  # < 24h -> must be skipped


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
    """Test-only: price the put sell far above the market so it can never fill."""

    def _entry_price(self, leg, quote):
        base = quote.bid_price.as_double()
        return base + 500 if leg == "PUT" else base


def q(inst, t, bid, ask) -> QuoteTick:
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


def run(strategy_cls, call_path, put_path, ref_mid=95_100.0, seconds=60, **cfg):
    """call_path / put_path: list of (bid, ask) per second after t=0 (t=0 is the opening quote)."""
    perp = TestInstrumentProvider.btcusdt_perp_binance()
    chain = []
    for strike in (94_000, 95_000, 96_000):
        for kind in (OptionKind.CALL, OptionKind.PUT):
            chain.append(make_option(strike, kind, EXPIRY))
    chain.append(make_option(95_000, OptionKind.CALL, NEXT_EXPIRY))
    chain.append(make_option(95_000, OptionKind.PUT, NEXT_EXPIRY))
    chain.append(make_option(95_100, OptionKind.CALL, SOON_EXPIRY))  # trap: too close to expiry
    chain.append(make_option(95_100, OptionKind.PUT, SOON_EXPIRY))

    engine = BacktestEngine(
        config=BacktestEngineConfig(logging=LoggingConfig(log_level="WARNING")),
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

    data = [q(perp, 0, ref_mid - 0.5, ref_mid + 0.5)]
    data.append(q(call, 0.5, *call_path[0]))
    data.append(q(put, 0.5, *put_path[0]))
    for t in range(1, min(len(call_path), seconds)):
        data.append(q(perp, t + 1, ref_mid - 0.5, ref_mid + 0.5))
        data.append(q(call, t + 1, *call_path[t]))
        data.append(q(put, t + 1, *put_path[t]))
    engine.add_data(data)

    config = ShortStraddleConfig(
        venue="BYBIT",
        reference_instrument_id=str(perp.id),
        quantity="0.10",
        entry_timeout_secs=5.0,
        exit_reprice_secs=3.0,
        **cfg,
    )
    strat = strategy_cls(config)
    engine.add_strategy(strat)
    engine.run()
    return engine, strat, call, put


def orders_of(engine, inst_id):
    return [o for o in engine.cache.orders() if o.instrument_id == inst_id]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(1)


def main():
    flat = lambda bid, ask, n=30: [(bid, ask)] * n

    # ---- 1. ATM selection + Level 1 entry (risk OFF so nothing exits) -----------------
    engine, s, call, put = run(
        ShortStraddleStrategy, flat(1000.0, 1010.0), flat(900.0, 910.0), enable_risk=False
    )
    check("ATM strike = 95000 (ref 95100; 95000 is nearest listed)", "95000" in str(s.legs["CALL"]))
    check("nearest valid expiry used (28NOV26, not 07NOV26 trap / 26DEC26)", "28NOV26" in str(s.legs["CALL"]))
    check("call and put share strike+expiry", str(s.legs["CALL"]).replace("-C-", "-P-") == str(s.legs["PUT"]))
    check("state ACTIVE after both fills", s.phase == State.ACTIVE, f"state={s.phase}")
    check("entry premium = bid_c + bid_p = 1900", s.entry_premium == Decimal("1900.0"), f"{s.entry_premium}")
    check("short 0.10 on each leg", s.short_qty == {"CALL": Decimal("0.10"), "PUT": Decimal("0.10")}, f"{s.short_qty}")
    check("exactly one sell per leg", len(orders_of(engine, call.id)) == 1 and len(orders_of(engine, put.id)) == 1)
    check("engine positions: both short", len(engine.cache.positions_open()) == 2)
    engine.dispose()

    # ---- 2. Take-profit: buyback falls >= 2% -> close both ----------------------------
    # entry premium 1900. TP threshold = 1862. asks: call 1004->900, put 904->900 => 1800 <= 1862
    c = flat(1000.0, 1004.0, 5) + flat(890.0, 900.0, 25)
    p = flat(900.0, 904.0, 5) + flat(890.0, 900.0, 25)
    engine, s, call, put = run(ShortStraddleStrategy, c, p)
    check("TP: strategy DONE", s.phase == State.DONE, f"state={s.phase} reason={s.exit_reason}")
    check("TP: reason is take-profit", s.exit_reason and "TAKE-PROFIT" in s.exit_reason, s.exit_reason or "")
    check("TP: no open positions left", len(engine.cache.positions_open()) == 0)
    check("TP: exactly 2 orders per leg (1 sell + 1 buy)", len(orders_of(engine, call.id)) == 2 and len(orders_of(engine, put.id)) == 2)
    engine.dispose()

    # ---- 3. Stop-loss: buyback rises >= 1% -> close both ------------------------------
    # entry 1900, SL threshold 1919. asks rise to 1030+920=1950, and KEEP ticking (duplicate-exit test)
    c = flat(1000.0, 1010.0, 5) + flat(1020.0, 1030.0, 25)
    p = flat(900.0, 910.0, 5) + flat(910.0, 920.0, 25)
    engine, s, call, put = run(ShortStraddleStrategy, c, p)
    check("SL: strategy DONE", s.phase == State.DONE, f"state={s.phase}")
    check("SL: reason is stop-loss", s.exit_reason and "STOP-LOSS" in s.exit_reason, s.exit_reason or "")
    check("SL: no open positions", len(engine.cache.positions_open()) == 0)
    sells = [o for o in engine.cache.orders() if o.side.name == "SELL"]
    buys = [o for o in engine.cache.orders() if o.side.name == "BUY"]
    check("SL: duplicate-exit guard -> exactly 1 buy per leg despite many triggering quotes", len(buys) == 2 and len(sells) == 2, f"buys={len(buys)}")
    check("SL: exit orders are reduce-only", all(o.is_reduce_only for o in buys))
    engine.dispose()

    # ---- 4. Inside the band: no exit ---------------------------------------------------
    c = flat(1000.0, 1010.0, 5) + flat(1000.0, 1015.0, 25)  # buyback 1925->1925? ask 1015+910=1925 > 1919 SL!
    c = flat(1000.0, 1010.0, 30)
    p = flat(900.0, 910.0, 30)
    engine, s, call, put = run(ShortStraddleStrategy, c, p)
    # entry 1900 (bids), buyback 1920 (asks) -> spread alone is +1.05% => SL fires!
    check("spread effect: bid/ask spread alone can trip the 1% SL (expected, see notes)", s.phase == State.DONE and "STOP-LOSS" in (s.exit_reason or ""), f"state={s.phase}")
    engine.dispose()
    c = flat(1000.0, 1004.0, 30)
    p = flat(900.0, 904.0, 30)  # buyback 1908 < 1919 and > 1862 -> stays ACTIVE
    engine, s, call, put = run(ShortStraddleStrategy, c, p)
    check("inside band: stays ACTIVE, no exit orders", s.phase == State.ACTIVE and len([o for o in engine.cache.orders() if o.side.name == "BUY"]) == 0, f"state={s.phase}")
    engine.dispose()

    # ---- 5. One leg never fills -> abort + flatten the filled leg --------------------
    engine, s, call, put = run(ForceNoFillPutStrategy, flat(1000.0, 1010.0), flat(900.0, 910.0))
    check("one-leg: put sell cancelled", any(o.instrument_id == put.id and o.side.name == "SELL" and o.status.name == "CANCELED" for o in engine.cache.orders()))
    check("one-leg: filled call leg bought back", any(o.instrument_id == call.id and o.side.name == "BUY" for o in engine.cache.orders()))
    check("one-leg: ends flat, DONE", s.phase == State.DONE and len(engine.cache.positions_open()) == 0, f"state={s.phase} pos={len(engine.cache.positions_open())}")
    engine.dispose()

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
