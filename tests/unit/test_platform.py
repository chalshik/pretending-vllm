"""The platform seam. B1, B2.

The config object is duck-typed here rather than imported: `pvllm.config` lands in
M1, and the seam's contract with it is small enough to state directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from pvllm.platforms import PlatformEnum, resolve_current_platform_cls_qualname
from pvllm.platforms.sim import SimPlatform


@dataclass
class _ParallelConfig:
    worker_cls: str = "auto"


@dataclass
class _VllmConfig:
    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)
    use_v2_model_runner: bool = True


def test_current_platform_resolves_to_the_simulator():
    """B2: selection goes through the real out-of-tree plugin mechanism."""
    assert resolve_current_platform_cls_qualname() == "pvllm.platforms.sim.SimPlatform"


def test_simulator_presents_as_an_out_of_tree_backend():
    """From vLLM's point of view a simulated device *is* an OOT backend, so the seam
    is the one a hardware vendor would actually use."""
    platform = SimPlatform()
    assert platform.is_out_of_tree()
    assert platform._enum is PlatformEnum.OOT
    assert not platform.is_cuda()


def test_check_and_update_config_supplies_the_worker_class():
    """The hinge the whole simulation boundary turns on.

    `CudaPlatform` fills in `vllm.v1.worker.gpu_worker.Worker` here; this fills in a
    simulated one, and nothing above the boundary is any the wiser.
    """
    config = _VllmConfig()
    SimPlatform.check_and_update_config(config)
    assert config.parallel_config.worker_cls == "pvllm.v1.worker.sim_worker.Worker"


def test_explicitly_configured_worker_class_is_left_alone():
    config = _VllmConfig(parallel_config=_ParallelConfig(worker_cls="my.custom.Worker"))
    SimPlatform.check_and_update_config(config)
    assert config.parallel_config.worker_cls == "my.custom.Worker"


def test_requesting_the_legacy_v1_runner_fails_loudly():
    """F1/D6: upstream defaults to V2 and only V2 is mirrored.

    Silently running V2 while the config says V1 is exactly the kind of quiet
    divergence a test double must never have.
    """
    config = _VllmConfig(use_v2_model_runner=False)
    with pytest.raises(NotImplementedError, match="legacy V1 model runner"):
        SimPlatform.check_and_update_config(config)


def test_attention_backend_is_the_simulated_one():
    assert SimPlatform.get_attn_backend_cls(
        None, 128, "bfloat16", None, 16, use_mla=False, has_sink=False
    ).endswith("SimAttentionBackend")


def test_unsupported_attention_backends_name_themselves():
    with pytest.raises(NotImplementedError, match="FLASH_ATTN"):
        SimPlatform.get_attn_backend_cls(
            "FLASH_ATTN", 128, "bfloat16", None, 16, use_mla=False, has_sink=False
        )
    with pytest.raises(NotImplementedError, match="MLA"):
        SimPlatform.get_attn_backend_cls(
            None, 128, "bfloat16", None, 16, use_mla=True, has_sink=False
        )


def test_unimplemented_features_raise_rather_than_no_op():
    """The unsupported-path discipline: fail loudly, naming the upstream feature."""
    with pytest.raises(NotImplementedError, match="LoRA"):
        SimPlatform.get_punica_wrapper()


def test_device_facts_come_from_the_card_not_from_hardware():
    from pvllm.sim.hardware_db import load_device_card, set_active_device_card

    set_active_device_card(load_device_card("tiny-2gb"))
    assert SimPlatform.get_device_name() == "tiny-2gb"
    assert SimPlatform.get_device_total_memory() == 2 * 1024**3
