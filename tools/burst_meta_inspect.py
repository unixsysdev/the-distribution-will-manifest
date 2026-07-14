"""Quick inspector — what does burst actually return in tx.meta?"""
import asyncio, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "grpc_stubs"))
import grpc, geyser_pb2, geyser_pb2_grpc, config


async def main():
    ep = "grpc-fra1-burst.erpc.global:80"
    n = 0
    print(f"inspecting {ep} ...")
    async with grpc.aio.insecure_channel(ep) as ch:
        stub = geyser_pb2_grpc.GeyserStub(ch)
        req = geyser_pb2.SubscribeRequest()
        req.transactions["f"].account_include.append(config.PUMP_FUN_PROGRAM)
        req.transactions["f"].failed = False
        req.commitment = geyser_pb2.CommitmentLevel.PROCESSED
        cancelled = asyncio.Event()
        async def itr():
            yield req
            await cancelled.wait()
        t0 = time.time()
        async for resp in stub.Subscribe(itr(),
                                         metadata=(("x-token", config.GRPC_TOKEN),)):
            if time.time() - t0 > 8:
                cancelled.set(); break
            if not resp.HasField("transaction"):
                continue
            n += 1
            tx = resp.transaction
            meta = tx.transaction.meta
            if n <= 3:
                print(f"=== resp #{n} slot={tx.slot} ===")
                print(f"  filters={list(resp.filters)}")
                print(f"  signature_len={len(tx.transaction.signature)}")
                print(f"  has_meta={tx.transaction.HasField('meta')}")
                if tx.transaction.HasField("meta"):
                    print(f"  meta.log_messages_len={len(meta.log_messages)}")
                    print(f"  meta.fee={meta.fee}")
                    print(f"  meta.pre_balances_len={len(meta.pre_balances)}")
                    print(f"  meta.post_balances_len={len(meta.post_balances)}")
                    print(f"  meta.inner_instructions_len={len(meta.inner_instructions)}")
                    print(f"  meta.pre_token_balances_len={len(meta.pre_token_balances)}")
                    print(f"  meta.compute_units_consumed={meta.compute_units_consumed}")
                    print(f"  meta.return_data_none={meta.return_data_none}")
                    print(f"  meta.log_messages_none={meta.log_messages_none}")
                print(f"  has_message={tx.transaction.transaction.HasField('message')}")
                tx_msg = tx.transaction.transaction.message
                print(f"  message.account_keys_len={len(tx_msg.account_keys)}")
        print(f"total_tx={n}  dur={time.time()-t0:.1f}s")

if __name__ == "__main__":
    asyncio.run(main())
