"""
Background Bot Runner with Live State Bridge for Streamlit UI.
Runs Bybit Paper/Live short straddle strategy while continuously updating straddle_state.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

# Load local .env if available
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from nautilus_trader.adapters.bybit import (
    BybitDataClientConfig,
    BybitEnvironment,
    BybitExecClientConfig,
    BybitLiveDataClientFactory,
    BybitLiveExecClientFactory,
    BybitProductType,
)
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import (
    InstrumentProviderConfig,
    LiveExecEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.events import OrderCanceled, OrderDenied, OrderFilled, OrderRejected

from straddle_strategy import ShortStraddleConfig, ShortStraddleStrategy, State

STATE_FILE = os.path.join(os.path.dirname(__file__), "straddle_state.json")
STOP_FLAG = os.path.join(os.path.dirname(__file__), "bot_stop.flag")
PROCESS_FILE = os.path.join(os.path.dirname(__file__), "bot_process.json")


class LiveUIStraddleStrategy(ShortStraddleStrategy):
    """Subclass of ShortStraddleStrategy that publishes state to JSON for Streamlit."""

    def __init__(self, config: ShortStraddleConfig, mode: str, bybit_env: str) -> None:
        super().__init__(config)
        self.mode = mode
        self.bybit_env = bybit_env
        self.log_messages: list[dict[str, str]] = []
        self._last_state_write: float = 0.0
        self._save_state(force=True)

    def _append_log(self, level: str, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.log_messages.append({"time": ts, "level": level, "message": msg})
        if len(self.log_messages) > 150:
            self.log_messages = self.log_messages[-150:]

    def on_start(self) -> None:
        self._append_log("INFO", f"Strategy starting on {self.venue}. Target Qty: {self.qty_target}")
        super().on_start()
        self._save_state(force=True)

    def on_stop(self) -> None:
        self._append_log("INFO", f"Strategy stopped. Reason: {self.exit_reason or 'Normal stop'}")
        super().on_stop()
        self._save_state(force=True)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        super().on_quote_tick(tick)
        now = time.time()
        # Rate limit state file writes to at most 4 Hz or on phase change
        if now - self._last_state_write >= 0.25:
            self._save_state()

    def _select_straddle(self) -> None:
        prev_legs = dict(self.legs)
        super()._select_straddle()
        if self.legs != prev_legs and len(self.legs) == 2:
            self._append_log("SUCCESS", f"ATM Straddle selected: CALL={self.legs['CALL']}, PUT={self.legs['PUT']}")
            self._save_state(force=True)

    def _maybe_enter(self) -> None:
        prev_phase = self.phase
        super()._maybe_enter()
        if self.phase != prev_phase:
            self._append_log("INFO", f"Entering straddle orders at bid...")
            self._save_state(force=True)

    def on_order_accepted(self, event) -> None:
        role = self._order_role.get(event.client_order_id)
        if role:
            self._append_log("INFO", f"Order accepted: {role[0]} {role[1]} ({event.client_order_id})")
        super().on_order_accepted(event)
        self._save_state(force=True)

    def on_order_filled(self, event: OrderFilled) -> None:
        role = self._order_role.get(event.client_order_id)
        if role:
            self._append_log(
                "SUCCESS",
                f"FILL {role[0]} {role[1]} {event.order_side.name} qty={event.last_qty} px={event.last_px}",
            )
        super().on_order_filled(event)
        self._save_state(force=True)

    def on_order_canceled(self, event: OrderCanceled) -> None:
        role = self._order_role.get(event.client_order_id)
        if role:
            self._append_log("WARN", f"Order canceled: {role[0]} {role[1]}")
        super().on_order_canceled(event)
        self._save_state(force=True)

    def on_order_rejected(self, event: OrderRejected) -> None:
        self._append_log("ERROR", f"Order rejected: {event.reason}")
        super().on_order_rejected(event)
        self._save_state(force=True)

    def _begin_exit(self, reason: str) -> None:
        self._append_log("WARN", f"Exit triggered: {reason}")
        super()._begin_exit(reason)
        self._save_state(force=True)

    def _finish(self) -> None:
        self._append_log("SUCCESS", f"All positions flat. Strategy Finished.")
        super()._finish()
        self._save_state(force=True)

    def _on_timer(self, event) -> None:
        if os.path.exists(STOP_FLAG):
            self._append_log("WARN", "Stop flag detected. Initiating graceful shutdown...")
            try:
                os.remove(STOP_FLAG)
            except OSError:
                pass
            if not self._is_flat():
                self._begin_exit("User requested stop from UI")
            else:
                self.stop()
            self._save_state(force=True)
            return

        super()._on_timer(event)
        self._save_state()

    def _save_state(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_state_write < 0.2):
            return
        self._last_state_write = now

        entry_px = float(self.entry_premium) if self.entry_premium is not None else None
        bb = self.buyback_value()
        bb_px = float(bb) if bb is not None else None

        pnl = None
        pnl_pct = None
        if entry_px is not None and bb_px is not None:
            pnl = round((entry_px - bb_px) * float(self.qty_target), 4)
            pnl_pct = round(((entry_px - bb_px) / entry_px) * 100, 2)

        sl_target = round(entry_px * (1 + float(self.sl_pct)), 2) if entry_px else None
        tp_target = round(entry_px * (1 - float(self.tp_pct)), 2) if entry_px else None

        # Build orders list
        orders_list = []
        if self.cache:
            for oid, (leg, role) in self._order_role.items():
                o = self.cache.order(oid)
                if o is not None:
                    orders_list.append({
                        "order_id": str(oid),
                        "instrument": str(o.instrument_id),
                        "leg": leg,
                        "role": role,
                        "side": o.side.name,
                        "price": float(o.price) if o.price else 0.0,
                        "quantity": float(o.quantity),
                        "filled_qty": float(o.filled_qty),
                        "status": o.status.name,
                    })

        # Quotes mapping
        quotes_dict = {}
        for inst_id, q in self.latest_quote.items():
            quotes_dict[str(inst_id)] = {
                "bid": q.bid_price.as_double(),
                "ask": q.ask_price.as_double(),
                "bid_size": q.bid_size.as_double(),
                "ask_size": q.ask_size.as_double(),
                "spread": round(q.ask_price.as_double() - q.bid_price.as_double(), 2),
                "mid": round((q.ask_price.as_double() + q.bid_price.as_double()) / 2.0, 2),
            }

        state_payload = {
            "is_running": True,
            "mode": self.mode,
            "bybit_env": self.bybit_env,
            "phase": self.phase.value,
            "ref_price": round(self.ref_price, 2) if self.ref_price is not None else None,
            "target_qty": float(self.qty_target),
            "short_qty": {k: float(v) for k, v in self.short_qty.items()},
            "legs": {k: str(v) for k, v in self.legs.items()},
            "entry_premium": entry_px,
            "buyback_value": bb_px,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "sl_pct": float(self.sl_pct),
            "tp_pct": float(self.tp_pct),
            "sl_target": sl_target,
            "tp_target": tp_target,
            "exit_reason": self.exit_reason,
            "abort_reason": self._abort_reason,
            "orders": orders_list,
            "quotes": quotes_dict,
            "logs": self.log_messages,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Write atomically
        tmp_file = f"{STATE_FILE}.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state_payload, f, indent=2)
            os.replace(tmp_file, STATE_FILE)
        except Exception:
            pass


def build_live_config(mode: str, bybit_env: str) -> TradingNodeConfig:
    paper = mode == "PAPER"
    default_env = "MAINNET" if paper else "TESTNET"
    env_str = bybit_env if bybit_env else default_env
    env = getattr(BybitEnvironment, env_str.upper())
    products = (BybitProductType.OPTION, BybitProductType.LINEAR)
    provider = InstrumentProviderConfig(load_all=True, filters={"base_coin": "BTC"})
    key = os.environ.get("BYBIT_API_KEY")
    secret = os.environ.get("BYBIT_API_SECRET")

    if paper:
        exec_clients = {
            "BYBIT": SandboxExecutionClientConfig(
                venue="BYBIT",
                starting_balances=["100000 USDT"],
                account_type="MARGIN",
                oms_type="NETTING",
            ),
        }
        exec_engine = LiveExecEngineConfig(reconciliation=False)
    else:
        exec_clients = {
            "BYBIT": BybitExecClientConfig(
                api_key=key,
                api_secret=secret,
                product_types=products,
                environment=env,
                instrument_provider=provider,
            ),
        }
        exec_engine = LiveExecEngineConfig(reconciliation=True)

    return TradingNodeConfig(
        trader_id="STRADDLE-UI",
        logging=LoggingConfig(log_level="INFO"),
        exec_engine=exec_engine,
        data_clients={
            "BYBIT": BybitDataClientConfig(
                api_key=None if paper else key,
                api_secret=None if paper else secret,
                product_types=products,
                environment=env,
                instrument_provider=provider,
            ),
        },
        exec_clients=exec_clients,
        timeout_connection=60.0,
        timeout_reconciliation=20.0,
        timeout_portfolio=20.0,
        timeout_disconnection=10.0,
        timeout_post_stop=10.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Short Straddle Live/Paper Runner")
    parser.add_argument("--mode", default="PAPER", choices=["PAPER", "LIVE"])
    parser.add_argument("--env", default="MAINNET", choices=["MAINNET", "TESTNET"])
    parser.add_argument("--quantity", default="0.01", type=str)
    parser.add_argument("--sl", default="0.01", type=str)
    parser.add_argument("--tp", default="0.02", type=str)
    parser.add_argument("--entry-timeout", default=15.0, type=float)
    parser.add_argument("--exit-reprice", default=5.0, type=float)
    args = parser.parse_args()

    # Record process PID for Streamlit
    with open(PROCESS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pid": os.getpid(),
                "mode": args.mode,
                "env": args.env,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
        )

    config = build_live_config(mode=args.mode, bybit_env=args.env)
    node = TradingNode(config=config)

    strategy = LiveUIStraddleStrategy(
        ShortStraddleConfig(
            reference_instrument_id="BTCUSDT-LINEAR.BYBIT",
            quantity=args.quantity,
            stop_loss_pct=args.sl,
            take_profit_pct=args.tp,
            entry_timeout_secs=args.entry_timeout,
            exit_reprice_secs=args.exit_reprice,
            enable_risk=True,
        ),
        mode=args.mode,
        bybit_env=args.env,
    )

    node.trader.add_strategy(strategy)
    node.add_data_client_factory("BYBIT", BybitLiveDataClientFactory)
    if args.mode == "PAPER":
        node.add_exec_client_factory("BYBIT", SandboxLiveExecClientFactory)
    else:
        node.add_exec_client_factory("BYBIT", BybitLiveExecClientFactory)

    node.build()

    try:
        node.run()
    finally:
        # Mark as not running
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    curr = json.load(f)
                curr["is_running"] = False
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(curr, f, indent=2)
            except Exception:
                pass
        if os.path.exists(PROCESS_FILE):
            try:
                os.remove(PROCESS_FILE)
            except OSError:
                pass
        node.dispose()


if __name__ == "__main__":
    main()
