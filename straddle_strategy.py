"""
BTC ATM short straddle on Bybit with NautilusTrader.

Level 1: find ATM strike, sell 1 Call + 1 Put (same strike, same expiry), track orders/positions.
Level 2: monitor combined ask-side buyback value, auto stop-loss (+1%) and take-profit (-2%).

State machine (one linear path, easy to explain on the call):

    WAIT_DATA -> ENTERING -> ACTIVE -> EXITING -> DONE
                    |                              ^
                    +--(entry failed: flatten)-----+   (or FAILED if an exit keeps failing)
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from enum import Enum

from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import OptionKind
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderCanceled
from nautilus_trader.model.events import OrderDenied
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.events import OrderRejected
from nautilus_trader.model.identifiers import ClientOrderId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.trading.strategy import Strategy


class State(str, Enum):
    WAIT_DATA = "WAIT_DATA"  # waiting for reference price + chain + option quotes
    ENTERING = "ENTERING"  # entry sell orders working
    ACTIVE = "ACTIVE"  # both legs short, SL/TP being monitored
    EXITING = "EXITING"  # exit triggered, buying back
    DONE = "DONE"  # flat, finished
    FAILED = "FAILED"  # could not exit cleanly -> needs a human


class ShortStraddleConfig(StrategyConfig, frozen=True):
    # --- instrument selection ---
    venue: str = "BYBIT"
    underlying: str = "BTC"  # matched against instrument.underlying code
    reference_instrument_id: str = "BTCUSDT-LINEAR.BYBIT"  # BTC reference price (perp mid)
    min_hours_to_expiry: float = 24.0  # skip contracts about to expire

    # --- sizing ---
    quantity: str = "0.01"  # per leg, in BTC (Bybit BTC option min is 0.01)

    # --- Level 2 risk management (set enable_risk=False for pure Level 1) ---
    enable_risk: bool = True
    stop_loss_pct: str = "0.01"  # exit if buyback >= entry * (1 + 1%)
    take_profit_pct: str = "0.02"  # exit if buyback <= entry * (1 - 2%)

    # --- order handling ---
    entry_timeout_secs: float = 15.0  # unfilled entry orders are cancelled after this
    exit_reprice_secs: float = 5.0  # unfilled exit order is cancelled + re-priced after this
    max_exit_attempts: int = 10  # per leg, before we give up (FAILED)
    timer_interval_secs: float = 1.0


class ShortStraddleStrategy(Strategy):
    def __init__(self, config: ShortStraddleConfig) -> None:
        super().__init__(config)
        self.venue = Venue(config.venue)
        self.ref_id = InstrumentId.from_str(config.reference_instrument_id)
        self.qty_target = Decimal(config.quantity)
        self.sl_pct = Decimal(config.stop_loss_pct)
        self.tp_pct = Decimal(config.take_profit_pct)

        self.phase = State.WAIT_DATA
        self.ref_price: float | None = None

        # leg name ("CALL"/"PUT") -> instrument id
        self.legs: dict[str, InstrumentId] = {}
        self.latest_quote: dict[InstrumentId, QuoteTick] = {}
        self._subscribed: set[InstrumentId] = set()

        # order bookkeeping: client_order_id -> (leg, "ENTRY"|"EXIT")
        self._order_role: dict[ClientOrderId, tuple[str, str]] = {}
        self._entry_ts_ns: int | None = None
        self._exit_order_ts_ns: dict[str, int] = {}  # leg -> submit time of the live exit order
        self._exit_attempts: dict[str, int] = {"CALL": 0, "PUT": 0}

        # our own fill accounting (deterministic, independent of engine event ordering)
        self.short_qty: dict[str, Decimal] = {"CALL": Decimal(0), "PUT": Decimal(0)}
        self._entry_notional: dict[str, Decimal] = {"CALL": Decimal(0), "PUT": Decimal(0)}
        self._entry_filled: dict[str, Decimal] = {"CALL": Decimal(0), "PUT": Decimal(0)}
        self.entry_premium: Decimal | None = None  # combined, per 1 unit of each leg
        self.exit_reason: str | None = None

        self._abort_reason: str | None = None

    # ------------------------------------------------------------------ lifecycle
    def on_start(self) -> None:
        ref = self.cache.instrument(self.ref_id)
        if ref is None:
            self.log.error(f"Reference instrument {self.ref_id} not in cache; stopping")
            self.stop()
            return
        self.subscribe_quote_ticks(self.ref_id)
        self.clock.set_timer(
            name="straddle-watchdog",
            interval=timedelta(seconds=self.config.timer_interval_secs),
            callback=self._on_timer,
        )
        self.log.info(
            f"Started. ref={self.ref_id} qty={self.qty_target} "
            f"risk={'ON' if self.config.enable_risk else 'OFF'}",
        )

    def on_stop(self) -> None:
        if "straddle-watchdog" in self.clock.timer_names:
            self.clock.cancel_timer("straddle-watchdog")
        open_shorts = {k: str(v) for k, v in self.short_qty.items() if v > 0}
        if open_shorts:
            self.log.warning(f"Stopping with OPEN short positions: {open_shorts}")

    # ------------------------------------------------------------------ market data
    def on_quote_tick(self, tick: QuoteTick) -> None:
        if tick.ask_price.as_double() <= 0 or tick.bid_price.as_double() <= 0:
            return  # Bybit can publish one-sided/empty books for illiquid strikes
        self.latest_quote[tick.instrument_id] = tick

        if tick.instrument_id == self.ref_id:
            self.ref_price = (tick.bid_price.as_double() + tick.ask_price.as_double()) / 2.0
            if self.phase == State.WAIT_DATA and not self.legs:
                self._select_straddle()
            return

        if self.phase == State.WAIT_DATA:
            self._maybe_enter()
        elif self.phase == State.ACTIVE and self.config.enable_risk:
            self._check_risk()

    # ------------------------------------------------------------------ instrument selection
    def _select_straddle(self) -> None:
        """Pick nearest-ATM strike on the nearest valid expiry that has BOTH a call and a put."""
        now_ns = self.clock.timestamp_ns()
        min_ns = int(self.config.min_hours_to_expiry * 3600 * 1e9)

        chain: dict[int, dict[float, dict[str, object]]] = {}
        for inst in self.cache.instruments(venue=self.venue):
            kind = getattr(inst, "option_kind", None)
            if kind is None:
                continue  # not an option (e.g. the perp)
            if inst.underlying.code != self.config.underlying:
                continue
            if inst.expiration_ns - now_ns < min_ns:
                continue
            leg = "CALL" if kind == OptionKind.CALL else "PUT"
            strike = inst.strike_price.as_double()
            chain.setdefault(inst.expiration_ns, {}).setdefault(strike, {})[leg] = inst

        for expiry in sorted(chain):  # nearest valid expiry first
            complete = {k: v for k, v in chain[expiry].items() if "CALL" in v and "PUT" in v}
            if not complete:
                continue
            strike = min(complete, key=lambda k: (abs(k - self.ref_price), k))
            call, put = complete[strike]["CALL"], complete[strike]["PUT"]
            self.legs = {"CALL": call.id, "PUT": put.id}
            self.log.info(
                f"ATM selection: ref={self.ref_price:.2f} expiry={call.expiration_utc} "
                f"strike={strike:.0f} CALL={call.id} PUT={put.id}",
            )
            for inst_id in self.legs.values():
                if inst_id not in self._subscribed:
                    self.subscribe_quote_ticks(inst_id)
                    self._subscribed.add(inst_id)
            return

        self.log.error("No option expiry with both a call and a put found; stopping")
        self.stop()

    # ------------------------------------------------------------------ entry (Level 1)
    def _entry_price(self, leg: str, quote: QuoteTick) -> float:
        """Sell at the current bid (marketable limit). Overridable in tests."""
        return quote.bid_price.as_double()

    def _maybe_enter(self) -> None:
        if len(self.legs) != 2 or self.phase != State.WAIT_DATA:
            return
        if not all(i in self.latest_quote for i in self.legs.values()):
            return  # need a live two-sided quote on both legs before selling anything

        self.phase = State.ENTERING  # set first: guarantees we only enter once
        self._entry_ts_ns = self.clock.timestamp_ns()
        for leg, inst_id in self.legs.items():
            inst = self.cache.instrument(inst_id)
            order = self.order_factory.limit(
                instrument_id=inst_id,
                order_side=OrderSide.SELL,
                quantity=inst.make_qty(self.qty_target),
                price=inst.make_price(self._entry_price(leg, self.latest_quote[inst_id])),
                time_in_force=TimeInForce.GTC,
            )
            self._order_role[order.client_order_id] = (leg, "ENTRY")
            self.submit_order(order)
            self.log.info(f"ENTRY submit {leg}: {order}")

    # ------------------------------------------------------------------ order events
    def on_order_accepted(self, event) -> None:
        role = self._order_role.get(event.client_order_id)
        if role:
            self.log.info(f"ACCEPTED {role[0]} {role[1]} {event.client_order_id}")

    def on_order_rejected(self, event: OrderRejected) -> None:
        self._handle_order_failure(event.client_order_id, f"REJECTED: {event.reason}")

    def on_order_denied(self, event: OrderDenied) -> None:
        self._handle_order_failure(event.client_order_id, f"DENIED: {event.reason}")

    def on_order_canceled(self, event: OrderCanceled) -> None:
        role = self._order_role.get(event.client_order_id)
        if not role:
            return
        leg, kind = role
        self.log.info(f"CANCELED {leg} {kind}")
        if kind == "EXIT" and self.phase == State.EXITING:
            self._exit_order_ts_ns.pop(leg, None)
            self._send_exits()  # re-price with a fresh ask
        elif kind == "ENTRY" and self.phase == State.ENTERING:
            self._resolve_entry_if_settled()

    def on_order_filled(self, event: OrderFilled) -> None:
        role = self._order_role.get(event.client_order_id)
        if not role:
            return
        leg, kind = role
        qty, px = Decimal(str(event.last_qty)), Decimal(str(event.last_px))
        self.log.info(f"FILL {leg} {kind} {event.order_side.name} qty={qty} px={px}")

        if kind == "ENTRY":  # partial fills simply accumulate
            self.short_qty[leg] += qty
            self._entry_filled[leg] += qty
            self._entry_notional[leg] += qty * px
            self._resolve_entry_if_settled()
        else:
            self.short_qty[leg] -= qty
            if self.short_qty[leg] < 0:  # defensive; should never happen with reduce_only
                self.log.error(f"Negative short qty on {leg}: {self.short_qty[leg]}")
            if self._is_flat():
                self._finish()

    def _handle_order_failure(self, client_order_id: ClientOrderId, why: str) -> None:
        role = self._order_role.get(client_order_id)
        if not role:
            return
        leg, kind = role
        self.log.error(f"{leg} {kind} order {why}")
        if kind == "ENTRY" and self.phase == State.ENTERING:
            self._abort_entry(f"entry order {why}")
        elif kind == "EXIT" and self.phase == State.EXITING:
            self._exit_order_ts_ns.pop(leg, None)  # watchdog will retry (counts as an attempt)

    # ------------------------------------------------------------------ entry resolution
    def _entry_orders(self):
        return [
            self.cache.order(oid) for oid, (_, k) in self._order_role.items() if k == "ENTRY"
        ]

    def _resolve_entry_if_settled(self) -> None:
        if self.phase != State.ENTERING:
            return
        full = all(self._entry_filled[leg] >= self.qty_target for leg in self.legs)
        if full:
            self.entry_premium = sum(
                (self._entry_notional[leg] / self._entry_filled[leg] for leg in self.legs),
                Decimal(0),
            )
            self.phase = State.ACTIVE
            self.log.info(f"STRADDLE ACTIVE. combined entry premium = {self.entry_premium}")
            return
        if self._abort_reason is not None:
            # entry is being unwound: wait until all entry orders are closed, then flatten
            if all(o is None or o.is_closed for o in self._entry_orders()):
                self._begin_exit(f"entry failed: {self._abort_reason}")

    def _abort_entry(self, reason: str) -> None:
        """One leg missing/failed -> never stay half-hedged. Cancel working legs, flatten the rest."""
        if self._abort_reason is not None:
            return
        self._abort_reason = reason
        self.log.warning(f"ABORTING ENTRY: {reason}")
        for order in self._entry_orders():
            if order is not None and order.is_open:
                self.cancel_order(order)
        self._resolve_entry_if_settled()

    # ------------------------------------------------------------------ risk (Level 2)
    def buyback_value(self) -> Decimal | None:
        """Combined cost to buy back both legs = ask(call) + ask(put). None if a quote is missing."""
        try:
            asks = [self.latest_quote[i].ask_price for i in self.legs.values()]
        except KeyError:
            return None
        return sum((Decimal(str(a)) for a in asks), Decimal(0))

    def _check_risk(self) -> None:
        if self.phase != State.ACTIVE or self.entry_premium is None:
            return
        value = self.buyback_value()
        if value is None:
            return
        stop = self.entry_premium * (Decimal(1) + self.sl_pct)
        take = self.entry_premium * (Decimal(1) - self.tp_pct)
        if value >= stop:
            self._begin_exit(f"STOP-LOSS buyback={value} >= {stop} (entry={self.entry_premium})")
        elif value <= take:
            self._begin_exit(f"TAKE-PROFIT buyback={value} <= {take} (entry={self.entry_premium})")

    # ------------------------------------------------------------------ exit
    def _begin_exit(self, reason: str) -> None:
        if self.phase in (State.EXITING, State.DONE, State.FAILED):
            return  # duplicate-exit guard #1: only one exit sequence, ever
        self.phase = State.EXITING
        self.exit_reason = reason
        self.log.warning(f"EXIT TRIGGERED: {reason}")
        self._send_exits()
        if self._is_flat():
            self._finish()

    def _send_exits(self) -> None:
        for leg, inst_id in self.legs.items():
            remaining = self.short_qty[leg]
            if remaining <= 0:
                continue
            if self._live_exit_order(leg) is not None:
                continue  # duplicate-exit guard #2: one live exit order per leg
            quote = self.latest_quote.get(inst_id)
            if quote is None:
                continue  # watchdog retries once a quote arrives
            if self._exit_attempts[leg] >= self.config.max_exit_attempts:
                self.phase = State.FAILED
                self.log.error(f"Giving up exiting {leg} after {self._exit_attempts[leg]} tries")
                return
            self._exit_attempts[leg] += 1
            inst = self.cache.instrument(inst_id)
            order = self.order_factory.limit(
                instrument_id=inst_id,
                order_side=OrderSide.BUY,
                quantity=inst.make_qty(remaining),  # only what is still open (handles partials)
                price=inst.make_price(quote.ask_price.as_double()),  # buy back at the ask
                time_in_force=TimeInForce.GTC,
                reduce_only=True,
            )
            self._order_role[order.client_order_id] = (leg, "EXIT")
            self._exit_order_ts_ns[leg] = self.clock.timestamp_ns()
            self.submit_order(order)
            self.log.info(f"EXIT submit {leg} qty={remaining} attempt={self._exit_attempts[leg]}")

    def _live_exit_order(self, leg: str):
        for oid, (l, kind) in self._order_role.items():
            if l == leg and kind == "EXIT":
                order = self.cache.order(oid)
                if order is not None and not order.is_closed:
                    return order
        return None

    def _is_flat(self) -> bool:
        return all(v <= 0 for v in self.short_qty.values())

    def _finish(self) -> None:
        if self.phase in (State.DONE, State.FAILED):
            return
        self.phase = State.DONE
        self.log.info(f"ALL FLAT. reason={self.exit_reason}")
        self.stop()

    # ------------------------------------------------------------------ watchdog
    def _on_timer(self, event) -> None:
        now = self.clock.timestamp_ns()
        if self.phase == State.ENTERING and self._abort_reason is None:
            if now - (self._entry_ts_ns or now) > self.config.entry_timeout_secs * 1e9:
                self._abort_entry("entry timeout: not both legs filled")
        elif self.phase == State.EXITING:
            stale_ns = int(self.config.exit_reprice_secs * 1e9)
            for leg in self.legs:
                order = self._live_exit_order(leg)
                if order is not None:
                    if now - self._exit_order_ts_ns.get(leg, now) > stale_ns and order.is_open:
                        self.cancel_order(order)  # on_order_canceled re-sends at the new ask
                elif self.short_qty[leg] > 0:
                    self._send_exits()  # no live order but still short -> retry
            if self._is_flat():
                self._finish()
        elif self.phase == State.ENTERING and self._abort_reason is not None:
            self._resolve_entry_if_settled()

    # unused but kept so every event is visible in the logs when debugging
    def on_event(self, event: Event) -> None:
        pass
