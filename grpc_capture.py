"""Standalone gRPC capture process for pump.fun TradeEvents.

Subscribes to the pump.fun program via Geyser gRPC. For every transaction it
inspects log_messages for `Program data:` lines, decodes the base64 payload,
and writes one JSONL row per parsed TradeEvent. Files are rotated hourly and
gzipped post-rotation. Raw base64 is included so any future parser bug can be
re-applied to the historical archive.

Verified-working endpoint (Jun 7): grpc-fra1-1.erpc.global:80 (plain text gRPC, no TLS).
Auth via x-token metadata header from .env GRPC_TOKEN.

NO model dependency. NO scoring. NO wallet. Pure data recording. Designed to
survive bot restarts, calibration changes, model swaps, and policy revisions.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import gzip
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent if "tools" in __file__ else _P(__file__).resolve().parent) + "/grpc_stubs")
import grpc
import base58
import geyser_pb2
import geyser_pb2_grpc
import config
from pumpfun_parse import parse_program_data_line, parse_trade_event, TRADE_EVENT_DISC


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="grpc-fra1-1.erpc.global:80",
                    help="gRPC endpoint host:port (default: grpc-fra1-1.erpc.global:80)")
    ap.add_argument("--insecure", action="store_true", default=True,
                    help="use plain text gRPC channel (matches grpc-fra1-1 port 80)")
    ap.add_argument("--data-dir", default="grpc_capture",
                    help="output directory for hour-rotated jsonl.gz files")
    ap.add_argument("--rotate-secs", type=int, default=3600,
                    help="rotate output file every N seconds (default 3600 = 1 hour)")
    ap.add_argument("--include-raw-b64", action="store_true", default=True,
                    help="store raw base64 payload for future re-parsing")
    ap.add_argument("--print-stats-secs", type=int, default=30,
                    help="emit progress stats every N seconds")
    ap.add_argument("--commitment", default="processed",
                    choices=["processed", "confirmed", "finalized"],
                    help="gRPC commitment level (default: processed)")
    ap.add_argument("--include-meta-extras", action="store_true", default=True,
                    help="save per-tx fee, compute_units, account_keys, "
                         "n_inner_ix — gRPC-exclusive fields the historical "
                         "snapshot never had. Marginal disk cost, big feature "
                         "headroom for future retrains.")
    return ap.parse_args()


# Canonical Jito mainnet tip accounts (8 fixed pubkeys). Verified 2026-06-09
# against docs.jito.wtf/lowlatencytxnsend (`getTipAccounts` response).
# Two of these were previously WRONG in this file — txs tipping to those two
# accounts were being silently mis-classified as "no jito tip". Fixed:
#   DfXygSm…NDe9B  ->  DfXygSm…NDXjh        (single-char typo)
#   ADuUkR4…XTNb   ->  ADuUkR4…WDcEt        (completely different address)
JITO_TIP_ACCOUNTS = frozenset({
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",      # was ..NDe9B (typo)
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",       # was ADuUkR4..XTNb (wrong)
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
})

# Known DEX, aggregator, and router program IDs (base58). When any of these
# appears in the tx's account_keys we tag the record with `route` so an
# extractor can split direct-pumpfun trades from routed/aggregator trades.
# Expanded 2026-06-09 from web research (jito-docs, raydium-docs, official
# project pages). Categories:
#   aggregator: txs that fan-out to multiple venues per swap
#   amm/clob:   the underlying execution venue (could be Jupiter target or
#               directly called by a bot)
#   pumpfun_ecosystem: tools specific to pump.fun (bots, sniper routers)
KNOWN_ROUTERS = {
    # ---------- aggregators ----------
    "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4":  "jupiter_v6",
    "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB":  "jupiter_v4",
    "routeUGWgWzqBWFcrCfv8tritsqukccJPu3q5GPP3xS":  "raydium_router",
    # ---------- Raydium family ----------
    "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "raydium_amm_v4",     # legacy CP AMM
    "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "raydium_clmm",       # concentrated liquidity
    "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C": "raydium_cpmm",       # newer CP AMM
    # ---------- Orca family ----------
    "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc":  "orca_whirlpool",     # concentrated liquidity
    "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP": "orca_swap_v2",       # constant product
    # ---------- order books ----------
    "PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY":  "phoenix",            # CLOB
    "srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX":  "openbook_v1",        # ex-Serum
    "opnb2LAfJYbRMAHHvqjCwQxanZn7ReEHp1k81EohpZb":  "openbook_v2",
    # ---------- other AMMs ----------
    "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo":  "meteora_dlmm",       # dynamic LMM
    "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB": "meteora_pools",      # constant product
    "2wT8Yq49kHgDzXuPxZSaeLaH1qbmGXtEyPy64bL7aD3c": "lifinity_v2",        # oracle-priced AMM
    "SSwpkEEcbUqx4vtoEByFjSkhKdCT862DNVb52nZg1UZ":  "saber",              # stable curves
    # ---------- pump.fun ecosystem routers (other than the trade programs we
    # already capture as PUMP_FUN_PROG / PUMPSWAP_PROG) ----------
    "M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K":  "moonshot",
    # NOTE on user-facing bots (Photon / BananaGun / Trojan / BullX / GMGN /
    # Axiom / Bonkbot): these are mostly WEB / TELEGRAM apps that build txs
    # client-side and submit them through ordinary RPC, often going through
    # Jupiter or directly to the bonding curve. They don't have a single
    # on-chain program ID — their signature pattern is a small set of
    # rotating signer wallets. If we want to classify by which bot built the
    # tx, that's a wallet-pattern lookup (separate detector), not a program-id
    # match. The `route` field stays focused on on-chain program IDs.
}

# Pump.fun program family (bonding curve + AMM + fees). Subscribed to all of
# them so post-graduation activity (PumpSwap) is captured alongside pre-grad.
PUMP_FUN_PROG  = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"   # bonding curve
PUMPSWAP_PROG  = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"   # PumpSwap AMM (post-grad)
PUMP_FEES_PROG = "pfeeXjVdkLAjAsfFqdtshb3aJxJrAcj62YotL5XPFCq"   # pump_fees admin
COMPUTE_BUDGET = "ComputeBudget111111111111111111111111111111"

# Anchor event discriminators we expect to see in `Program data:` log lines.
# Pulled 2026-06-09 from pump-fun/pump-public-docs idl/{pump,pump_amm}.json.
# Used to dispatch non-TradeEvent events into separate records (instead of
# silently dropping them like the original recorder did).
PUMPFUN_EVENT_NAMES: dict[bytes, str] = {
    bytes.fromhex("bddb7fd34ee661ee"): "TradeEvent",        # bonding curve buy/sell
    bytes.fromhex("1b72a94ddeeb6376"): "CreateEvent",        # token birth
    bytes.fromhex("5f72619cd42e9808"): "CompleteEvent",      # bonding curve filled
    bytes.fromhex("bde95db95c94ea94"): "CompletePumpAmmMigrationEvent",
    bytes.fromhex("ed347b25f5fb48d2"): "SetCreatorEvent",
    bytes.fromhex("9ba768dcd56cf303"): "MigrateBondingCurveCreatorEvent",
    bytes.fromhex("e2d6f62107f293e5"): "ClaimCashbackEvent",
    bytes.fromhex("4facf631cd5bcee8"): "ClaimTokenIncentivesEvent",
    bytes.fromhex("929fbdac925838f4"): "CloseUserVolumeAccumulatorEvent",
    bytes.fromhex("86240d48e86582d8"): "InitUserVolumeAccumulatorEvent",
    bytes.fromhex("c57aa77c74515bff"): "SyncUserVolumeAccumulatorEvent",
    bytes.fromhex("7a027f010ebf0caf"): "CollectCreatorFeeEvent",
    bytes.fromhex("a537817004b3ca28"): "DistributeCreatorFeesEvent",
    bytes.fromhex("6161d7905d92167c"): "ExtendAccountEvent",
    bytes.fromhex("dfc39ff63e308f83"): "SetParamsEvent",
    bytes.fromhex("a8d884efebb63134"): "MinimumDistributableFeeEvent",
    bytes.fromhex("2bbcfa12dd4bbb5f"): "ReservedFeeRecipientsEvent",
    bytes.fromhex("b6c3892a23cecff7"): "UpdateGlobalAuthorityEvent",
    bytes.fromhex("757be4b6a1a8dcd6"): "UpdateMayhemVirtualParamsEvent",
    bytes.fromhex("8ecb06207f69bfa2"): "SetMetaplexCreatorEvent",
    bytes.fromhex("93fa6c78f71d43de"): "AdminUpdateTokenIncentivesEvent",
    bytes.fromhex("4045c0681d1e196b"): "AdminSetCreatorEvent",
    bytes.fromhex("f53b46224bb96d5c"): "AdminSetIdlAuthorityEvent",
    # PumpSwap AMM (pAMMBay) events — same disc namespace, different program
    bytes.fromhex("67f4521f2cf57777"): "PumpSwap.BuyEvent",
    bytes.fromhex("3e2f370aa503dc2a"): "PumpSwap.SellEvent",
    bytes.fromhex("b1310cd2a076a774"): "PumpSwap.CreatePoolEvent",
    bytes.fromhex("78f83d531f8e6b90"): "PumpSwap.DepositEvent",
    bytes.fromhex("1609851aa02c47c0"): "PumpSwap.WithdrawEvent",
    bytes.fromhex("a039592ab58b2b42"): "PumpSwap.CollectCoinCreatorFeeEvent",
    bytes.fromhex("6b34598137e25116"): "PumpSwap.CreateConfigEvent",
    bytes.fromhex("6bfdc14ce4ca1b68"): "PumpSwap.DisableEvent",
    bytes.fromhex("aadd52c793a5f72e"): "PumpSwap.MigratePoolCoinCreatorEvent",
    bytes.fromhex("f2e7eb664163bdd3"): "PumpSwap.SetBondingCurveCoinCreatorEvent",
    bytes.fromhex("966bc77b7ccf66e4"): "PumpSwap.SetMetaplexCoinCreatorEvent",
    bytes.fromhex("e198ab57f63f42ea"): "PumpSwap.UpdateAdminEvent",
    bytes.fromhex("5a1741233ef4bcd0"): "PumpSwap.UpdateFeeConfigEvent",
    bytes.fromhex("2ddc5d181961ac68"): "PumpSwap.AdminSetCoinCreatorEvent",
    bytes.fromhex("e8f5c2eeeada3a59"): "PumpSwap.CollectCoinCreatorFeeEvent2",
}


def _commitment_to_enum(name: str) -> int:
    m = {"processed": geyser_pb2.CommitmentLevel.PROCESSED,
         "confirmed": geyser_pb2.CommitmentLevel.CONFIRMED,
         "finalized": geyser_pb2.CommitmentLevel.FINALIZED}
    return m.get((name or "processed").lower(), geyser_pb2.CommitmentLevel.PROCESSED)


class GzipRotatingWriter:
    """Append to current/<basename>.jsonl, rotate at every rotate_secs interval,
    then gzip the rolled-off file in the background. Crash-safe: if the process dies
    mid-rotation, the current file is just a plain .jsonl that the next run will pick
    up via fresh rotation.
    """
    def __init__(self, data_dir: Path, rotate_secs: int):
        self.data_dir = Path(data_dir); self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_secs = rotate_secs
        self.current_path: Path | None = None
        self.current_fh = None
        self.next_rotate_t = 0.0
        # Startup orphan recovery: any *.jsonl left ungzipped by a previous
        # crash / restart gets compressed now (rather than sitting around at
        # ~5-10x the gzipped size until someone notices manually).
        try:
            orphans = sorted(self.data_dir.glob("capture_*.jsonl"))
            if orphans:
                print(f"[capture] startup: found {len(orphans)} ungzipped orphan(s); "
                      f"compressing", flush=True)
                for o in orphans:
                    if o.stat().st_size == 0:
                        try: o.unlink()
                        except Exception: pass
                        print(f"[capture] removed empty orphan {o.name}", flush=True)
                        continue
                    gz = o.with_suffix(".jsonl.gz")
                    if gz.exists():
                        partial = gz.with_name(gz.name + f".partial_{int(time.time())}")
                        try:
                            gz.rename(partial)
                            print(f"[capture] quarantined pre-existing {gz.name} -> "
                                  f"{partial.name}; recompressing {o.name}", flush=True)
                        except Exception as e:
                            print(f"[capture] could not quarantine {gz.name}: {e}",
                                  flush=True)
                            continue
                    try:
                        tmp = gz.with_name(gz.name + f".tmp.{os.getpid()}")
                        try: tmp.unlink()
                        except FileNotFoundError: pass
                        with open(o, "rb") as f_in, gzip.open(tmp, "wb", compresslevel=6) as f_out:
                            shutil.copyfileobj(f_in, f_out, length=1 << 20)
                        tmp.replace(gz)
                        o.unlink()
                        print(f"[capture] gzipped orphan {gz.name} "
                              f"({gz.stat().st_size:,} bytes)", flush=True)
                    except Exception as e:
                        try: tmp.unlink()
                        except Exception: pass
                        print(f"[capture] gzip orphan {o} failed: {e}", flush=True)
        except Exception as e:
            print(f"[capture] startup orphan scan err: {e}", flush=True)

    def _open_new(self) -> None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.current_path = self.data_dir / f"capture_{ts}.jsonl"
        self.current_fh = open(self.current_path, "a", buffering=1)
        self.next_rotate_t = time.time() + self.rotate_secs
        print(f"[capture] opened {self.current_path}", flush=True)

    def _rotate(self) -> None:
        if self.current_fh is None: return
        try: self.current_fh.flush(); self.current_fh.close()
        except Exception: pass
        rolled = self.current_path
        self.current_fh = None; self.current_path = None
        # gzip the rolled-off file
        if rolled is not None and rolled.exists() and rolled.stat().st_size > 0:
            gz = rolled.with_suffix(".jsonl.gz")
            tmp = gz.with_name(gz.name + f".tmp.{os.getpid()}")
            try:
                try: tmp.unlink()
                except FileNotFoundError: pass
                with open(rolled, "rb") as f_in, gzip.open(tmp, "wb", compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out, length=1 << 20)
                tmp.replace(gz)
                rolled.unlink()
                print(f"[capture] rotated -> {gz} ({gz.stat().st_size:,} bytes)", flush=True)
            except Exception as e:
                try: tmp.unlink()
                except Exception: pass
                print(f"[capture] gzip error on {rolled}: {e}", flush=True)
        elif rolled is not None and rolled.exists():
            rolled.unlink()
        self._open_new()

    def write(self, rec: dict) -> None:
        if self.current_fh is None or time.time() >= self.next_rotate_t:
            if self.current_fh is None: self._open_new()
            else: self._rotate()
        try:
            self.current_fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception as e:
            print(f"[capture] write error: {e}", flush=True)

    def close(self) -> None:
        if self.current_fh is not None:
            try: self.current_fh.flush(); self.current_fh.close()
            except Exception: pass
        # Gzip the file we just closed so a clean shutdown leaves nothing
        # ungzipped. (Startup orphan scan would catch it on next run, but
        # the right time to compress is now while we still have CPU time.)
        if self.current_path is not None and self.current_path.exists():
            if self.current_path.stat().st_size > 0:
                gz = self.current_path.with_suffix(".jsonl.gz")
                if not gz.exists():
                    tmp = gz.with_name(gz.name + f".tmp.{os.getpid()}")
                    try:
                        try: tmp.unlink()
                        except FileNotFoundError: pass
                        with open(self.current_path, "rb") as f_in, \
                             gzip.open(tmp, "wb", compresslevel=6) as f_out:
                            shutil.copyfileobj(f_in, f_out, length=1 << 20)
                        tmp.replace(gz)
                        self.current_path.unlink()
                        print(f"[capture] shutdown: gzipped {gz.name}", flush=True)
                    except Exception as e:
                        try: tmp.unlink()
                        except Exception: pass
                        print(f"[capture] shutdown gzip {self.current_path} failed: {e}",
                              flush=True)
            else:
                try: self.current_path.unlink()
                except Exception: pass


async def subscribe(args, writer: GzipRotatingWriter, stop: asyncio.Event,
                    stats: dict) -> None:
    endpoint = args.endpoint
    token = config.GRPC_TOKEN
    metadata = (("x-token", token),) if token else None
    while not stop.is_set():
        try:
            if args.insecure:
                ch = grpc.aio.insecure_channel(endpoint)
            else:
                ch = grpc.aio.secure_channel(endpoint, grpc.ssl_channel_credentials())
            async with ch as channel:
                stub = geyser_pb2_grpc.GeyserStub(channel)
                req = geyser_pb2.SubscribeRequest()
                # 2026-06-09: widened from bonding-curve-only to the whole
                # pump.fun program family (bonding curve + PumpSwap AMM +
                # pump_fees). PumpSwap captures post-graduation trading that
                # was previously invisible. `failed` filter removed so we
                # ALSO see reverted txs (spoofs, sandwich misfires) — record
                # carries `failed` flag so consumers can filter trivially.
                req.transactions["pumpfun"].account_include.append(PUMP_FUN_PROG)
                req.transactions["pumpfun"].account_include.append(PUMPSWAP_PROG)
                req.transactions["pumpfun"].account_include.append(PUMP_FEES_PROG)
                # Slot -> block_time sidecar. The raw grpc_firehose already
                # archives block_meta; writing a compact decoded row here keeps
                # downstream model builders from needing to decode raw proto
                # frames just to recover UTC time.
                req.blocks_meta["pumpfun_bm"].SetInParent()
                # (Intentionally NOT setting .failed — capture both success
                # and failure. Each record carries a `failed` field.)
                req.commitment = _commitment_to_enum(args.commitment)
                async def req_iter():
                    yield req
                    while not stop.is_set():
                        await asyncio.sleep(1)
                print(f"[capture] subscribing to {endpoint} "
                      f"(insecure={args.insecure}, commitment={args.commitment}, "
                      f"programs=[pumpfun,pumpswap,pump_fees], failed=both, +blocks_meta)",
                      flush=True)
                async for resp in stub.Subscribe(req_iter(), metadata=metadata):
                    if stop.is_set(): break
                    stats["msgs"] += 1
                    if resp.HasField("block_meta"):
                        stats["block_meta"] = stats.get("block_meta", 0) + 1
                        bm = resp.block_meta
                        block_time = None
                        try:
                            block_time = (int(bm.block_time.timestamp)
                                          if bm.block_time.timestamp else None)
                        except Exception:
                            pass
                        writer.write({
                            "t": time.time(),
                            "slot": int(bm.slot) if bm.slot else None,
                            "event": "BlockMeta",
                            "block_time": block_time,
                        })
                        continue
                    if not resp.HasField("transaction"):
                        continue
                    stats["txs"] += 1
                    tx = resp.transaction
                    slot = int(tx.slot) if hasattr(tx, "slot") else None
                    sig_raw = tx.transaction.signature
                    sig = base58.b58encode(sig_raw).decode()
                    meta = tx.transaction.meta
                    failed_flag = bool(meta.err.err) if meta.err.err else False
                    # ----- pre-compute tx-level meta extras (used for every
                    # event record below). Wrapped in try; capture must never
                    # raise — any field that fails to extract just stays None.
                    fee_lam = 0; cu = 0; n_inner_ix = 0
                    cu_limit = 0; priority_fee_micro = 0
                    n_keys = 0; ak_b58 = []
                    route = None
                    jito_tip_idx = -1; jito_tip_lam = 0
                    pre_token_balances = []; post_token_balances = []
                    loaded_writable = []; loaded_readonly = []
                    inner_ix_summary = []
                    try:
                        fee_lam = int(meta.fee) if meta.fee else 0
                        cu = (int(meta.compute_units_consumed)
                              if meta.compute_units_consumed else 0)
                        if meta.inner_instructions:
                            n_inner_ix = sum(len(g.instructions) for g in meta.inner_instructions)
                        # account keys (V0 + Legacy alike, base58 strings)
                        message = (tx.transaction.transaction.message
                                   if hasattr(tx.transaction, "transaction") else None)
                        if message is not None:
                            ak_bytes = list(message.account_keys)
                            ak_b58 = [base58.b58encode(bytes(k)).decode() for k in ak_bytes]
                            n_keys = len(ak_b58)
                            # GAP G: parse ComputeBudget outer ixs to get
                            # the REQUESTED cu_limit (matches the shred-intent
                            # path which already captures this).
                            for ix in message.instructions:
                                if ix.program_id_index >= len(ak_b58): continue
                                prog = ak_b58[ix.program_id_index]
                                if prog != COMPUTE_BUDGET: continue
                                d = bytes(ix.data)
                                if len(d) >= 5 and d[0] == 0x02:
                                    cu_limit = int.from_bytes(d[1:5], "little")
                                elif len(d) >= 9 and d[0] == 0x03:
                                    priority_fee_micro = int.from_bytes(d[1:9], "little")
                            # route detection across the visible keys
                            for k in ak_b58:
                                if k in KNOWN_ROUTERS:
                                    route = KNOWN_ROUTERS[k]; break
                            # jito tip detection + lamport delta
                            for i, k in enumerate(ak_b58):
                                if k in JITO_TIP_ACCOUNTS:
                                    jito_tip_idx = i
                                    try:
                                        pre  = int(meta.pre_balances[i])
                                        post = int(meta.post_balances[i])
                                        jito_tip_lam = post - pre
                                    except Exception:
                                        pass
                                    break
                        # GAP F: address-lookup-table loaded addresses for V0 txs.
                        # If the tx used ALT, these are the additional accounts.
                        loaded_writable = [
                            base58.b58encode(bytes(a)).decode()
                            for a in meta.loaded_writable_addresses
                        ] if meta.loaded_writable_addresses else []
                        loaded_readonly = [
                            base58.b58encode(bytes(a)).decode()
                            for a in meta.loaded_readonly_addresses
                        ] if meta.loaded_readonly_addresses else []
                        # GAP D: pre/post token balances - compact summary.
                        # Each entry: {acc_idx, mint, owner, ui_amt_pre, ui_amt_post, dec}.
                        # Joining pre[i] vs post[i] by (acc_idx, mint) gives the
                        # exact tokens-changed for that token-account.
                        def _tb_row(tb):
                            try:
                                ua = tb.ui_token_amount
                                return {"i": int(tb.account_index),
                                        "mint": tb.mint,
                                        "owner": tb.owner,
                                        "amt_str": ua.amount if ua else None,
                                        "dec": int(ua.decimals) if ua else None}
                            except Exception:
                                return None
                        pre_token_balances  = [r for r in (_tb_row(tb) for tb in meta.pre_token_balances) if r]
                        post_token_balances = [r for r in (_tb_row(tb) for tb in meta.post_token_balances) if r]
                        # GAP E: inner-instruction summary. For each inner ix,
                        # record (group_idx, program, disc_first_8B). Keeps it
                        # cheap but lets us reconstruct the CPI tree for sandwich
                        # / MEV detection without dumping full instruction data.
                        if meta.inner_instructions:
                            full_keys = ak_b58 + loaded_writable + loaded_readonly
                            for grp in meta.inner_instructions:
                                gi = int(grp.index)
                                for ix in grp.instructions:
                                    prog = full_keys[ix.program_id_index] if ix.program_id_index < len(full_keys) else "?"
                                    d = bytes(ix.data)[:8]
                                    inner_ix_summary.append({"g": gi, "p": prog, "d": d.hex()})
                    except Exception:
                        stats["meta_extract_err"] = stats.get("meta_extract_err", 0) + 1

                    # ----- common per-tx meta dict (attached to every event below)
                    tx_meta = {
                        "failed":     failed_flag,
                        "fee_lam":    fee_lam,
                        "cu":         cu,
                        "cu_limit":   cu_limit,
                        "priority_fee_micro": priority_fee_micro,
                        "n_inner_ix": n_inner_ix,
                        "n_keys":     n_keys,
                        "jito_tip_idx": jito_tip_idx if jito_tip_idx >= 0 else None,
                        "jito_tip_lam": jito_tip_lam if jito_tip_idx >= 0 else None,
                        "route":      route,
                        "loaded_w":   loaded_writable,
                        "loaded_r":   loaded_readonly,
                        "pre_tb":     pre_token_balances,
                        "post_tb":    post_token_balances,
                        "inner_ix":   inner_ix_summary,
                    }

                    # GAP A + B: dispatch every Program-data event by discriminator.
                    # TradeEvent gets the existing decoded record (backwards-compat).
                    # Other events get a minimal record {type=event_name, raw=b64}.
                    # We also emit a record per tx-with-no-event so failed txs and
                    # non-event activity (create, fees, etc) are still visible.
                    any_event = False
                    for ln in meta.log_messages:
                        if "Program data:" not in ln:
                            continue
                        b64 = ln.split("Program data:", 1)[1].strip()
                        try:
                            data = base64.b64decode(b64)
                        except Exception:
                            continue
                        if len(data) < 8: continue
                        disc = bytes(data[:8])
                        ev_name = PUMPFUN_EVENT_NAMES.get(disc)
                        if ev_name == "TradeEvent":
                            ev = parse_trade_event(data)
                            if ev is None:
                                stats["parse_fail"] += 1
                                continue
                            stats["events"] += 1
                            any_event = True
                            rec = {"t": time.time(), "slot": slot, "sig": sig,
                                   "event": "TradeEvent",
                                   "ev_ts": ev.timestamp,
                                   "mint": ev.mint, "user": ev.user, "is_buy": ev.is_buy,
                                   "sol": ev.sol_amount, "tok": ev.token_amount,
                                   "vsol": ev.virtual_sol_reserves,
                                   "vtok": ev.virtual_token_reserves,
                                   "rsol": ev.real_sol_reserves,
                                   "rtok": ev.real_token_reserves,
                                   **tx_meta}
                            if args.include_raw_b64: rec["raw"] = b64
                            writer.write(rec)
                        elif ev_name is not None:
                            # Non-trade pump.fun / PumpSwap event — save raw so
                            # we can decode later if we add a parser. Keeps the
                            # full b64 (these are small, ~100-300B each).
                            stats["other_events"] = stats.get("other_events", 0) + 1
                            any_event = True
                            writer.write({
                                "t": time.time(), "slot": slot, "sig": sig,
                                "event": ev_name,
                                "raw": b64,
                                **tx_meta,
                            })
                        else:
                            # Unknown event disc — could be a new ABI; save it
                            # so we can identify it later.
                            stats["unknown_events"] = stats.get("unknown_events", 0) + 1
                            any_event = True
                            writer.write({
                                "t": time.time(), "slot": slot, "sig": sig,
                                "event": "Unknown",
                                "disc": disc.hex(),
                                "raw": b64,
                                **tx_meta,
                            })

                    if not any_event:
                        # Tx touched a pump.fun program but emitted no event we
                        # parsed (likely admin / non-event ix, or a failed tx
                        # that reverted before logging). Save a stub so the
                        # tx is still visible in the timeline.
                        stats["no_event_tx"] = stats.get("no_event_tx", 0) + 1
                        writer.write({
                            "t": time.time(), "slot": slot, "sig": sig,
                            "event": "NoEvent",
                            **tx_meta,
                        })
        except asyncio.CancelledError:
            raise
        except grpc.aio.AioRpcError as e:
            stats["grpc_errors"] += 1
            print(f"[capture] gRPC error {e.code()}: {str(e.details())[:200]}; reconnect 5s",
                  flush=True)
            await asyncio.sleep(5)
        except Exception as e:
            stats["other_errors"] += 1
            print(f"[capture] error: {type(e).__name__}: {e}; reconnect 5s", flush=True)
            await asyncio.sleep(5)


async def stats_printer(stats: dict, interval: int, stop: asyncio.Event) -> None:
    t0 = time.time()
    last = dict(stats); last_t = t0
    while not stop.is_set():
        try: await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError: pass
        now = time.time(); dt = now - last_t or 1.0
        msgs_ps = (stats["msgs"] - last["msgs"]) / dt
        ev_ps   = (stats["events"] - last["events"]) / dt
        print(f"[capture] uptime={now - t0:.0f}s  msgs={stats['msgs']}  tx={stats['txs']}  "
              f"events={stats['events']}  block_meta={stats.get('block_meta', 0)}  "
              f"parse_fail={stats['parse_fail']}  "
              f"errors={stats['grpc_errors']}+{stats['other_errors']}  "
              f"({msgs_ps:.0f} msgs/s  {ev_ps:.0f} ev/s)",
              flush=True)
        last = dict(stats); last_t = now


async def amain(args):
    writer = GzipRotatingWriter(Path(args.data_dir), args.rotate_secs)
    writer._open_new()
    stats = {"msgs": 0, "txs": 0, "events": 0, "block_meta": 0, "parse_fail": 0,
             "grpc_errors": 0, "other_errors": 0}
    stop = asyncio.Event()
    def shutdown(*_): stop.set()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try: loop.add_signal_handler(s, shutdown)
        except NotImplementedError: pass
    sub = asyncio.create_task(subscribe(args, writer, stop, stats))
    pr  = asyncio.create_task(stats_printer(stats, args.print_stats_secs, stop))
    await stop.wait()
    for t in (sub, pr): t.cancel()
    try: await asyncio.gather(sub, pr, return_exceptions=True)
    except Exception: pass
    writer._rotate()
    writer.close()
    print(f"[capture] shutdown — final stats {stats}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(amain(args))
