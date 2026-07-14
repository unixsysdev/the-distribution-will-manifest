"""Load settings from .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

WALLET_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "")
GRPC_ENDPOINT = os.getenv("GRPC_ENDPOINT", "grpc.erpc.global:443")
GRPC_TOKEN = os.getenv("GRPC_TOKEN", "")
SOLANA_RPC_ENDPOINT = os.getenv(
    "SOLANA_RPC_ENDPOINT",
    f"https://edge.erpc.global?api-key={GRPC_TOKEN}" if GRPC_TOKEN else "",
)
USE_GRPC = os.getenv("USE_GRPC", "false").lower() in ("1", "true", "yes")
NUM_SLOTS = int(os.getenv("NUM_SLOTS", "16"))
BET_SIZE_SOL = float(os.getenv("BET_SIZE_SOL", "0.1"))


def rpc_http_url() -> str:
  """HTTP RPC for blockhash / block time / tx send."""
  return SOLANA_RPC_ENDPOINT or "https://api.mainnet-beta.solana.com"


def rpc_ws_url() -> str:
    """Websocket URL for logsSubscribe."""
    explicit = os.getenv("SOLANA_RPC_WS")
    if explicit:
        return explicit
    http = SOLANA_RPC_ENDPOINT
    if http.startswith("https://"):
        return "wss://" + http[len("https://") :]
    if http.startswith("http://"):
        return "ws://" + http[len("http://") :]
    return "wss://api.mainnet-beta.solana.com"


def wallet_configured() -> bool:
    key = WALLET_PRIVATE_KEY.strip()
    return bool(key) and "your_actual" not in key
