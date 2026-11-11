# SPDX-License-Identifier: Apache-2.0
"""Use one low-latency communication allocation for DSpark and its target."""


def dspark_buffer_args(args, vllm_config):
    spec = vllm_config.speculative_config
    if spec is None or not spec.use_dspark():
        return args

    config = vllm_config.model_config.hf_text_config
    # Both models execute sequentially on the same EP group. SAILSHMEM permits
    # one RDMA team, so their allocation keys must agree even though the draft
    # has fewer experts. Dispatch still receives each model's actual sizes.
    num_experts = max(config.n_routed_experts, config.dspark_n_routed_experts)
    return dict(
        args,
        num_global_experts=max(args["num_global_experts"], num_experts),
        num_local_experts=max(
            args["num_local_experts"], num_experts // args["num_ep_ranks"]
        ),
    )


def dspark_uses_shared_buffer(vllm_config):
    spec = vllm_config.speculative_config
    if spec is None or not spec.use_dspark():
        return False
    config = vllm_config.model_config.hf_text_config
    return config.n_routed_experts != config.dspark_n_routed_experts
