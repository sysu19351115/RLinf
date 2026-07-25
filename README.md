# RLinf 复现

本项目提供多个端到端训练与数据采集示例，覆盖仿真环境与真实机器人场景。每个示例的详细步骤已拆分到独立文档中，请点击对应链接查看。

## 示例列表

| 示例 | 场景 | 文档 |
|------|------|------|
| 示例1 | LIBERO + PI0 + PPO（仿真） | [docs/examples/libero_pi0_ppo.md](docs/examples/libero_pi0_ppo.md) |
| 示例2 | Gym Aloha + PI0 + PPO（仿真） | [docs/examples/gym_aloha_pi0_ppo.md](docs/examples/gym_aloha_pi0_ppo.md) |
| 示例3 | ReBot + PI0.5 + PPO 异步真机训练 | [docs/examples/rebot_pi05_ppo_async.md](docs/examples/rebot_pi05_ppo_async.md) |
| 示例4 | SO101 + PI0.5 + PPO 异步真机训练 | [docs/source-zh/rst_source/examples/embodied/so101.rst](docs/source-zh/rst_source/examples/embodied/so101.rst) |
| 示例5 | SO101 人在环（HIL）数据采集 | [docs/examples/so101_hil_data_collection.md](docs/examples/so101_hil_data_collection.md) |
| 示例6 | Dobot CR5AF + PI0.5 + PPO 真机训练 | [docs/examples/dobot_pi05_ppo.md](docs/examples/dobot_pi05_ppo.md) |
| 示例7 | Dobot CR5AF 人在环（HIL）数据采集 | [docs/examples/dobot_hil_data_collection.md](docs/examples/dobot_hil_data_collection.md) |

## 通用资源

- Reward Model 规划与原理：[docs/reward_model_integration.md](docs/reward_model_integration.md)
  - 可执行的 reward model 步骤请直接参考真机示例文档
- 网络方案：
  - WireGuard 搭建：[docs/wireguard_build.md](docs/wireguard_build.md)
  - SSH 双向隧道：[docs/ssh_reverse_tunnel_build.md](docs/ssh_reverse_tunnel_build.md)
- 参数约束说明：[docs/PARAM_CONSTRAINTS.md](docs/PARAM_CONSTRAINTS.md)
- 自定义环境（PI0）：[docs/ADD_CUSTOM_ENV_WITH_PI0.md](docs/ADD_CUSTOM_ENV_WITH_PI0.md)

## 快速开始

选择对应示例文档，按步骤完成环境安装、模型准备和训练启动。
