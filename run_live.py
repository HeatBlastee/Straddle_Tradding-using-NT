"""
Runner: BTC ATM short straddle on Bybit via NautilusTrader.

MODE=PAPER (default)  -> LIVE Bybit market data, orders filled by a LOCAL simulator.
                         No order ever reaches Bybit. No API key needed (public data only).
MODE=LIVE             -> real Bybit execution (set BYBIT_ENV + keys). Real orders!

    python run_live.py                      # safe paper trading
    $env:MODE="LIVE"; $env:BYBIT_ENV="TESTNET"; python run_live.py
"""

import os

# Load local .env if available
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

from nautilus_trader.adapters.bybit import BybitDataClientConfig # Configuration for receiving Bybit market data
from nautilus_trader.adapters.bybit import BybitEnvironment # Selects MAINNET or TESTNET
from nautilus_trader.adapters.bybit import BybitExecClientConfig # Configuration for actual Bybit order execution
from nautilus_trader.adapters.bybit import BybitLiveDataClientFactory # Creates the live Bybit data client
from nautilus_trader.adapters.bybit import BybitLiveExecClientFactory # Creates the live Bybit execution client
from nautilus_trader.adapters.bybit import BybitProductType
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode

from straddle_strategy import ShortStraddleConfig
from straddle_strategy import ShortStraddleStrategy

MODE = os.environ.get("MODE", "PAPER").upper()


def build_config() -> TradingNodeConfig:
    paper = MODE == "PAPER"
    # PAPER reads real market data, so default to MAINNET (public data, no key, no orders sent).
    default_env = "MAINNET" if paper else "TESTNET"
    env = getattr(BybitEnvironment, os.environ.get("BYBIT_ENV", default_env).upper())
    products = (BybitProductType.OPTION, BybitProductType.LINEAR)
    provider = InstrumentProviderConfig(load_all=True, filters={"base_coin": "BTC"})
    key, secret = os.environ.get("BYBIT_API_KEY"), os.environ.get("BYBIT_API_SECRET")

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
                api_key=key, api_secret=secret, product_types=products,
                environment=env, instrument_provider=provider,
            ),
        }
        exec_engine = LiveExecEngineConfig(reconciliation=True)

    return TradingNodeConfig(
        trader_id="STRADDLE-001",
        logging=LoggingConfig(log_level="INFO"),
        exec_engine=exec_engine,
        data_clients={
            "BYBIT": BybitDataClientConfig(
                api_key=None if paper else key,
                api_secret=None if paper else secret,
                product_types=products, environment=env,
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
    config = build_config()
    print(f"*** MODE={MODE} ***" + ("  (simulated fills, no real orders)" if MODE == "PAPER" else "  (REAL ORDERS)"))
    node = TradingNode(config=config)
    node.trader.add_strategy(
        ShortStraddleStrategy(
            ShortStraddleConfig(
                reference_instrument_id="BTCUSDT-LINEAR.BYBIT",
                quantity="0.01",
                enable_risk=True,  # False = Level 1 only
            ),
        ),
    )
    node.add_data_client_factory("BYBIT", BybitLiveDataClientFactory)
    if MODE == "PAPER":
        node.add_exec_client_factory("BYBIT", SandboxLiveExecClientFactory)
    else:
        node.add_exec_client_factory("BYBIT", BybitLiveExecClientFactory)
    node.build()
    try:
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    main()