"""CPU-only behavioral evidence for prefetch observation, with no torch/Ray."""
import asyncio
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
for parent in HERE.parents:
    if (parent / "fsdp_v3" / "split_serial.py").is_file():
        sys.path.insert(0, str(parent / "fsdp_v3"))
        break
from split_serial import CpuReplayPrefetch
from observed_cpu_prefetch import ObservedCpuReplayPrefetch


class Writer:
    def __init__(self):
        self.events = []
    def event(self, kind, **fields):
        self.events.append((kind, fields["details"]))


def record(index, size=3, wm=False):
    meta = {"sample_ids": [f"{index}:{i}" for i in range(size)]}
    return {"raw": (index, {"capture_meta": meta} if wm else meta), "sample_time": .01}


def observed(fetch, writer=None, **kwargs):
    return ObservedCpuReplayPrefetch(fetch, writer=writer or Writer(), model="policy",
                                     rank=0, world_size=4, local_batch_size=3, **kwargs)


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_and_observed_fetch_identity_and_order_match(self):
        async def exercise(cls):
            calls, consumed = [], []
            values = [record(i) for i in range(6)]
            async def fetch():
                index = len(calls)
                calls.append(index)
                await asyncio.sleep(0)
                return values[index]
            q = cls(fetch)
            q.start()
            for i in range(5):
                value = await q.take()
                self.assertIs(value, values[i])
                consumed.append(value["raw"][0])
                if isinstance(q, ObservedCpuReplayPrefetch):
                    q.release_in_use()
                await asyncio.sleep(0)
            await q.close()
            return calls, consumed
        self.assertEqual(await exercise(CpuReplayPrefetch), await exercise(observed))

    async def test_ready_pending_and_in_use_are_distinct_not_resident(self):
        gate = asyncio.Event()
        calls = []
        async def fetch():
            calls.append(len(calls))
            await gate.wait()
            return record(calls[-1])
        writer = Writer()
        q = observed(fetch, writer)
        q.start()
        await asyncio.sleep(0)
        self.assertEqual(q.snapshot()["pending_fetches"], 1)
        waiter = asyncio.create_task(q.take())
        await asyncio.sleep(0)
        self.assertEqual(q.snapshot()["empty_waiters"], 1)
        gate.set()
        value = await waiter
        self.assertEqual(value["raw"][0], 0)
        await asyncio.sleep(0)
        snap = q.snapshot()
        self.assertEqual((snap["ready_batches"], snap["in_use_batches"]), (1, 1))
        self.assertEqual((snap["ready_sample_count"], snap["in_use_sample_count"]), (3, 3))
        self.assertEqual(len(calls), 2)  # One in use plus exactly one raw slot.
        q.release_in_use()
        q.release_in_use()  # Off-schedule phases are no-ops.
        self.assertEqual(q.snapshot()["batches_released"], 1)
        self.assertTrue(all(kind == "cpu_replay_prefetch_state" for kind, _ in writer.events))
        self.assertFalse(any("consume" == kind for kind, _ in writer.events))
        await q.close()
        self.assertEqual(q.snapshot()["ready_batches"], 0)

    async def test_fetch_error_wakes_consumer_without_retry(self):
        calls = []
        async def fetch():
            calls.append(1)
            raise ValueError("failed raw read")
        q = observed(fetch)
        q.start()
        with self.assertRaisesRegex(RuntimeError, "not redrawn"):
            await q.take()
        self.assertEqual(calls, [1])
        self.assertTrue(q.snapshot()["failed"])
        await q.close()

    async def test_close_cancels_fetch_and_no_redraw(self):
        gate = asyncio.Event()
        calls = []
        async def fetch():
            calls.append(1)
            await gate.wait()
        q = observed(fetch)
        q.start()
        await asyncio.sleep(0)
        await q.close()
        self.assertEqual(calls, [1])
        self.assertFalse(q.snapshot()["pending_fetches"])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await q.take()

    async def test_cancelled_consumer_preserves_same_reserved_batch(self):
        gate = asyncio.Event()
        value = record(7)
        calls = []
        async def fetch():
            calls.append(1)
            await gate.wait()
            return value
        q = observed(fetch)
        q.start()
        task = asyncio.create_task(q.take())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(q.snapshot()["empty_waiters"], 0)
        gate.set()
        self.assertIs(await q.take(), value)
        self.assertEqual(calls, [1])
        await q.close()

    async def test_rejects_accidental_cuda_value_like_original(self):
        class NotCpu:
            is_cuda = True
        async def fetch():
            return {"raw": [NotCpu()]}
        q = observed(fetch)
        q.start()
        with self.assertRaisesRegex(RuntimeError, "not redrawn"):
            await q.take()
        self.assertEqual(q.snapshot()["ready_batches"], 0)
        await q.close()

    async def test_observer_failure_does_not_redraw_or_corrupt_value(self):
        class BrokenWriter:
            def event(self, *args, **kwargs):
                raise OSError("telemetry full")
        value = record(2)
        async def fetch():
            return value
        q = observed(fetch, BrokenWriter())
        q.start()
        self.assertIs(await q.take(), value)
        self.assertGreater(q.snapshot()["observer_errors"], 0)
        await q.close()

    async def test_wm_metadata_sample_count(self):
        value = record(9, 256, wm=True)
        async def fetch():
            return value
        q = observed(fetch)
        q.start()
        await q.take()
        self.assertEqual(q.snapshot()["in_use_sample_count"], 256)
        await q.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
