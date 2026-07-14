"""pump.fun TradeEvent parser — decode the Anchor event from a `Program data:` line.

Layout locked against live capture (disc bddb7fd34ee661ee). Core fields sit at fixed
offsets right after the 8-byte discriminator; newer trailing fields (fee/creator/
"buy"|"sell" string/track_volume) are ignored. Pure + unit-testable, no I/O.

Validated: decoded timestamp lands in 2026 and virtual_sol_reserves >= ~30 SOL on real
payloads, which only holds if the offsets are correct.
"""
from __future__ import annotations
import base58
import base64
from dataclasses import dataclass

TRADE_EVENT_DISC = bytes.fromhex("bddb7fd34ee661ee")
_MIN_LEN = 129  # disc(8)+mint(32)+sol(8)+tok(8)+isbuy(1)+user(32)+ts(8)+vsol(8)+vtok(8)+rsol(8)+rtok(8)


@dataclass(slots=True)
class TradeEvent:
    mint: str
    sol_amount: int          # lamports
    token_amount: int        # raw (6 decimals)
    is_buy: bool
    user: str
    timestamp: int           # unix seconds
    virtual_sol_reserves: int
    virtual_token_reserves: int
    real_sol_reserves: int
    real_token_reserves: int
    slot: int | None = None  # set by gRPC listener for same-block aim; None on WS path
    # gRPC-exclusive per-tx context (Task 44 audit half). Set by the gRPC
    # listener with extracted fee/cu/jito_tip/route. None on WS path or when
    # extraction fails. The model does NOT consume this; the harness aggregates
    # K-window stats and stamps them onto entry_decision events for offline
    # correlation analysis between buyer sophistication and per-fire P&L.
    grpc_extras: dict | None = None

    @property
    def mid(self) -> float:
        """Bonding-curve mark = vsol/vtok (matches offline `mid`)."""
        return self.virtual_sol_reserves / self.virtual_token_reserves if self.virtual_token_reserves else 0.0

    @property
    def sol(self) -> float:
        return self.sol_amount / 1e9

    @property
    def is_classic_curve(self) -> bool:
        """Classic pre-graduation pump.fun curve: initial virtual SOL = 30, so vsol = 30 + rsol.
        The model was trained on this population (82% of training rows, the bonding-curve phase).
        Live `bddb` stream mixes in post-graduation/variant pools that violate this — gate them out."""
        return abs(self.virtual_sol_reserves - 30_000_000_000 - self.real_sol_reserves) < 50_000_000

    @property
    def fill_k(self) -> float:
        """Curve fill fraction (1-fill_k = runway). Only meaningful on the classic curve."""
        return max(0.0, min(1.0, (self.virtual_sol_reserves / 1e9 - 30.0) / 85.0))


def _u64(b: bytes, off: int) -> int:
    return int.from_bytes(b[off:off + 8], "little")


def parse_trade_event(data: bytes) -> TradeEvent | None:
    """Decode a TradeEvent payload (already base64-decoded). Returns None if not a TradeEvent."""
    if len(data) < _MIN_LEN or data[:8] != TRADE_EVENT_DISC:
        return None
    return TradeEvent(
        mint=base58.b58encode(data[8:40]).decode(),
        sol_amount=_u64(data, 40),
        token_amount=_u64(data, 48),
        is_buy=data[56] == 1,
        user=base58.b58encode(data[57:89]).decode(),
        timestamp=_u64(data, 89),
        virtual_sol_reserves=_u64(data, 97),
        virtual_token_reserves=_u64(data, 105),
        real_sol_reserves=_u64(data, 113),
        real_token_reserves=_u64(data, 121),
    )


def parse_program_data_line(log_line: str) -> TradeEvent | None:
    """Decode a raw `Program data: <base64>` log line into a TradeEvent (or None)."""
    if "Program data:" not in log_line:
        return None
    b64 = log_line.split("Program data:")[1].strip()
    try:
        return parse_trade_event(base64.b64decode(b64))
    except Exception:
        return None
