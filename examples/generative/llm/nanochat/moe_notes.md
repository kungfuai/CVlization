# MoE Notes (2025)

This file captures a quick reference on modern Mixture-of-Experts (MoE) patterns
and recent large-scale architectures (DeepSeek V3, Kimi K2, GLM 4.6/4.7).

## Common Design Patterns

- **Fine-grained experts**: Many small experts (256-384) vs a few large experts.
- **Top-k routing**: Typical top-k=8 routed experts + 1 shared expert per token.
- **Shared expert**: Always-on expert for common knowledge and stability.
- **Sigmoid routing**: Use sigmoid affinities, then normalize selected experts.
- **Aux-loss-free load balancing**: Bias term influences routing selection but
  does not affect gating weights.

## DeepSeek V3 / V3.2 Highlights

- 256 routed experts + 1 shared expert, top-k=8
- Sigmoid affinities, gating normalization on selected experts
- Bias-based load balancing (no aux loss)
- Node-limited routing to reduce cross-node communication
- Multi-token prediction (MTP) during training

## Kimi K2 Highlights

- 384 experts, top-k=8, 1 shared expert
- Extreme sparsity (about 3.2% active params)
- Single dense layer across the stack
- MuonClip optimizer for stability at scale
- INT4 QAT and long context in the Thinking variant

## GLM 4.6 / 4.7 Highlights

- Large MoE with strong quantization and chip support
- Interleaved thinking modes (4.7)

## Missing / TODO

- **Expert parallelism strategies**: How to shard experts across devices (EP vs TP vs DP combinations)
- **Expert capacity handling**: Fixed capacity with drop/pad during training vs dynamic routing during inference
- **Memory/communication tradeoffs**: All-to-all vs hierarchical routing, node-limited routing to reduce cross-node traffic
- **Scaling considerations**: At what model size does MoE become worthwhile? Communication overhead vs compute savings

## References

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Auxiliary-Loss-Free Load Balancing: https://arxiv.org/abs/2408.15664
- Kimi K2 GitHub: https://github.com/MoonshotAI/Kimi-K2
- GLM 4.7 Release: https://docs.z.ai
