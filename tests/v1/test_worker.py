"""Worker, model runner, block tables, and the slot-mapping oracle. R7, R8."""

from __future__ import annotations

import numpy as np
import pytest

from pvllm.engine.arg_utils import EngineArgs
from pvllm.sampling_params import SamplingParams
from pvllm.sim.clock import VirtualClock
from pvllm.v1.core.sched.scheduler import Scheduler
from pvllm.v1.executor.abstract import Executor
from pvllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from pvllm.v1.request import Request
from pvllm.v1.worker.gpu.block_table import PAD_SLOT_ID, BlockTables


class Engine:
    """The M1f stack, wired the way the engine core will wire it in M1g."""

    def __init__(self, **overrides):
        args = {
            "model": "dense-0.6b",
            "max_model_len": 1024,
            "block_size": 16,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 8,
            "enable_prefix_caching": False,
            "device_card": "workstation-24gb",
        }
        args.update(overrides)
        self.config = EngineArgs(**args).create_engine_config()
        self.clock = VirtualClock()
        self.executor = Executor.get_class(self.config)(self.config, self.clock)

        kv_bytes = self.executor.determine_available_memory()[0]
        specs = self.executor.get_kv_cache_specs()[0]
        spec = next(iter(specs.values()))
        num_blocks = kv_bytes // (spec.page_size_bytes * len(specs))
        self.kv_cache_config = KVCacheConfig(
            num_blocks=int(num_blocks),
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=list(specs), kv_cache_spec=spec)
            ],
        )
        self.executor.initialize_from_config([self.kv_cache_config])
        self.executor.compile_or_warm_up_model()
        self.scheduler = Scheduler(self.config, self.kv_cache_config, log_stats=False)

    @property
    def runner(self):
        return self.executor.driver_worker.model_runner

    def add(
        self, request_id: str, prompt_len: int = 20, max_tokens: int = 4
    ) -> Request:
        request = Request(
            request_id,
            list(range(prompt_len)),
            SamplingParams(max_tokens=max_tokens),
            arrival_time=self.clock.time(),
        )
        self.scheduler.add_request(request)
        return request

    def step(self):
        output = self.scheduler.schedule()
        runner_output = self.executor.execute_model(output)
        self.scheduler.update_from_output(output, runner_output)
        return output, runner_output

    def drain(self, limit: int = 200) -> int:
        steps = 0
        while self.scheduler.has_requests() and steps < limit:
            self.step()
            steps += 1
        return steps


# --- the boundary works end to end -----------------------------------------


def test_a_request_runs_from_prefill_to_completion():
    engine = Engine()
    engine.add("r0", prompt_len=20, max_tokens=4)

    first, output = engine.step()
    assert first.num_scheduled_tokens == {"r0": 20}
    assert output.sampled_token_ids[0], "prefill should sample one token"

    steps = engine.drain()
    assert engine.scheduler.running == []
    assert engine.scheduler.get_kv_cache_usage() == 0.0
    assert steps < 10


def test_prefill_batches_and_decode_narrows():
    """Continuous batching, observed through the boundary."""
    engine = Engine()
    for i in range(3):
        engine.add(f"r{i}", prompt_len=20 + i * 10, max_tokens=4)

    prefill, _ = engine.step()
    assert prefill.total_num_scheduled_tokens == 20 + 30 + 40

    decode, _ = engine.step()
    assert decode.num_scheduled_tokens == {"r0": 1, "r1": 1, "r2": 1}


def test_the_clock_only_advances_through_the_device():
    engine = Engine()
    engine.add("r0")
    before = engine.clock.elapsed
    engine.step()
    assert engine.clock.elapsed > before


def test_startup_spends_modeled_time():
    """R10.4: a product that polls readiness exercises real behaviour only if
    startup takes plausible time."""
    engine = Engine()
    startup = engine.executor.driver_worker.startup
    assert startup.profile_run_seconds > 0
    assert startup.total_seconds > 0
    assert "modeled" in startup.summary(1.0)


# --- persistent batch state (R7.3) -----------------------------------------


def test_worker_state_is_patched_not_rebuilt():
    engine = Engine()
    engine.add("r0", prompt_len=20, max_tokens=8)
    engine.step()

    runner = engine.runner
    req_idx = runner.req_states.req_id_to_index["r0"]
    assert runner.req_states.prompt_len[req_idx] == 20
    assert runner.req_states.total_len[req_idx] == 20

    engine.step()  # one decode
    assert runner.req_states.total_len[req_idx] == 21
    assert runner.req_states.num_output_tokens[req_idx] == 1


def test_finished_requests_release_their_slot():
    """R5.8: the worker drops state when the scheduler says the request is done."""
    engine = Engine()
    engine.add("r0", max_tokens=2)
    engine.drain()
    assert engine.runner.num_cached_requests == 0
    assert (
        len(engine.runner.req_states.free_indices)
        == engine.config.scheduler_config.max_num_seqs
    )


def test_slots_are_recycled():
    engine = Engine()
    engine.add("a", max_tokens=2)
    engine.drain()
    engine.add("b", max_tokens=2)
    engine.step()
    assert engine.runner.num_cached_requests == 1


def test_a_cached_request_the_worker_never_saw_is_an_error():
    """Silently accepting it would mean generating from a token history missing its
    middle."""
    engine = Engine()
    engine.add("r0")
    output = engine.scheduler.schedule()
    engine.executor.execute_model(output)

    output.scheduled_cached_reqs.req_ids.append("ghost")
    output.scheduled_cached_reqs.new_token_ids.append([1])
    output.scheduled_cached_reqs.new_block_ids.append(None)
    output.scheduled_cached_reqs.num_computed_tokens.append(0)
    output.scheduled_cached_reqs.num_output_tokens.append(0)
    output.num_scheduled_tokens["ghost"] = 1
    output.total_num_scheduled_tokens += 1

    with pytest.raises(KeyError, match="holds no state for it"):
        engine.executor.execute_model(output)


# --- attention metadata (R8.2) ---------------------------------------------


def test_attention_metadata_is_real():
    engine = Engine()
    engine.add("a", prompt_len=20, max_tokens=4)
    engine.add("b", prompt_len=30, max_tokens=4)
    engine.step()

    metadata = engine.runner.last_attn_metadata
    assert metadata is not None
    assert metadata.num_reqs == 2
    assert metadata.num_actual_tokens == 50
    # query_start_loc is cumulative and ends at the token count.
    assert metadata.query_start_loc[0] == 0
    assert metadata.query_start_loc[-1] == 50
    assert len(metadata.slot_mapping) == 50
    assert metadata.num_prefills == 2 and metadata.num_decodes == 0


def test_metadata_splits_prefill_from_decode():
    """The split drives the cost model's compute term and the graph decision."""
    engine = Engine()
    engine.add("a", prompt_len=20, max_tokens=8)
    engine.step()
    engine.add("b", prompt_len=30, max_tokens=8)
    engine.step()

    metadata = engine.runner.last_attn_metadata
    assert metadata is not None
    assert metadata.num_decodes == 1  # "a" is decoding
    assert metadata.num_prefills == 1  # "b" is prefilling
    assert metadata.is_mixed_batch


def test_seq_lens_grow_with_context():
    engine = Engine()
    engine.add("r0", prompt_len=20, max_tokens=8)
    engine.step()
    assert int(engine.runner.last_attn_metadata.seq_lens[0]) == 20
    engine.step()
    assert int(engine.runner.last_attn_metadata.seq_lens[0]) == 21


# --- block tables (R7.4) ---------------------------------------------------


def test_block_tables_are_sized_for_max_model_len():
    """R7.4: the metadata cost of a large max_model_len is visible up front."""
    tables = BlockTables(
        [16], max_num_reqs=4, max_model_len=1024, max_num_batched_tokens=512
    )
    assert tables.max_blocks_per_req == [64]
    assert tables.block_tables[0].shape == (4, 64)


def test_block_tables_grow_incrementally():
    tables = BlockTables(
        [16], max_num_reqs=4, max_model_len=1024, max_num_batched_tokens=512
    )
    tables.set_block_ids(0, ([5, 6],))
    assert tables.get_block_ids(0).tolist() == [5, 6]
    tables.append_block_ids(0, ([7],))
    assert tables.get_block_ids(0).tolist() == [5, 6, 7]


def test_a_request_cannot_exceed_its_block_budget():
    tables = BlockTables(
        [16], max_num_reqs=2, max_model_len=32, max_num_batched_tokens=32
    )
    with pytest.raises(ValueError, match="max_model_len allows only"):
        tables.set_block_ids(0, (list(range(10)),))


# --- the slot-mapping oracle (R8.3) ----------------------------------------


def make_tables() -> BlockTables:
    tables = BlockTables(
        [4], max_num_reqs=4, max_model_len=64, max_num_batched_tokens=64
    )
    tables.set_block_ids(0, ([10, 11],))  # request 0 owns blocks 10 and 11
    tables.set_block_ids(1, ([20, 21],))
    return tables


def test_slot_mapping_is_block_times_size_plus_offset():
    tables = make_tables()
    slots = tables.compute_slot_mapping(
        req_indices=np.array([0]),
        positions=np.arange(6),
        query_start_loc=np.array([0, 6]),
    )
    # Positions 0-3 land in block 10, positions 4-5 in block 11.
    assert slots.tolist() == [40, 41, 42, 43, 44, 45]


def test_slot_mapping_covers_a_multi_request_batch():
    tables = make_tables()
    slots = tables.compute_slot_mapping(
        req_indices=np.array([0, 1]),
        positions=np.array([0, 1, 0, 1]),
        query_start_loc=np.array([0, 2, 4]),
    )
    assert slots.tolist() == [40, 41, 80, 81]


def test_writing_past_the_block_table_raises():
    """The scheduler scheduled a token the KV manager never allocated for."""
    tables = make_tables()
    with pytest.raises(AssertionError, match="runs past the block table"):
        tables.compute_slot_mapping(
            req_indices=np.array([0]),
            positions=np.array([100]),  # needs block index 25; only 2 are owned
            query_start_loc=np.array([0, 1]),
        )


def test_two_requests_sharing_a_slot_raises():
    """R21.1: no slot written twice in one step."""
    tables = make_tables()
    tables.block_tables[0][1, 0] = 10  # request 1 now points at request 0's block
    with pytest.raises(AssertionError, match="written twice in one step"):
        tables.compute_slot_mapping(
            req_indices=np.array([0, 1]),
            positions=np.array([0, 0]),
            query_start_loc=np.array([0, 1, 2]),
        )


def test_a_block_held_by_two_live_requests_raises():
    """The real corruption mode, and the reason ownership is checked across every
    live row rather than only the scheduled ones: the pool can hand one block to two
    requests that are never scheduled together, and the collision would then surface
    much later."""
    tables = make_tables()
    tables.block_tables[0][1, 0] = 10  # request 1 also claims request 0's block
    with pytest.raises(AssertionError, match="held by two live requests"):
        tables.validate_block_ownership()


def test_ownership_validation_passes_on_a_healthy_table():
    make_tables().validate_block_ownership()


def test_padded_positions_map_to_the_sentinel():
    tables = make_tables()
    slots = tables.compute_slot_mapping(
        req_indices=np.array([0, 1]),
        positions=np.array([0]),
        query_start_loc=np.array([0, 1, 1]),  # request 1 scheduled nothing
    )
    assert slots.tolist() == [40]
    assert PAD_SLOT_ID == -1


def test_the_oracle_runs_on_every_real_step():
    """Not just in the unit tests: a full drain exercises it on every step, which is
    what makes it a check on the KV manager rather than on itself."""
    engine = Engine()
    for i in range(4):
        engine.add(f"r{i}", prompt_len=20 + i * 7, max_tokens=5)
    engine.drain()
    assert engine.scheduler.get_kv_cache_usage() == 0.0


# --- graph capture (R8.4) --------------------------------------------------


def test_graph_capture_records_shapes():
    engine = Engine()
    assert engine.runner.captured_sizes
    assert (
        max(engine.runner.captured_sizes) <= engine.config.scheduler_config.max_num_seqs
    )


def test_enforce_eager_captures_nothing():
    engine = Engine(enforce_eager=True)
    assert engine.runner.captured_sizes == frozenset()
    assert engine.executor.driver_worker.startup.graph_capture_seconds == 0.0


# --- executor --------------------------------------------------------------


def test_worker_class_comes_from_the_platform_not_a_hardcoded_import():
    """B2: short-circuiting this would remove the seam the design rests on."""
    engine = Engine()
    assert (
        engine.config.parallel_config.worker_cls == "pvllm.v1.worker.sim_worker.Worker"
    )
    assert type(engine.executor.driver_worker).__name__ == "Worker"


def test_collective_rpc_reaches_the_worker():
    engine = Engine()
    assert engine.executor.collective_rpc("check_health") == [None]


def test_an_unimplemented_executor_backend_names_itself():
    config = EngineArgs(model="tiny-test").create_engine_config()
    config.parallel_config.distributed_executor_backend = "ray"
    with pytest.raises(NotImplementedError, match="multiprocess engine core"):
        Executor.get_class(config)


def test_a_worker_cannot_create_its_own_clock():
    """R19.1: two clocks would mean time advancing in two places."""
    from pvllm.v1.worker.sim_worker import Worker

    config = EngineArgs(model="tiny-test").create_engine_config()
    with pytest.raises(ValueError, match="requires the engine core's clock"):
        Worker(config, clock=None)
