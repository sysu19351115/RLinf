# WireGuard 星型拓扑 + RLinf 真机强化学习部署方案

> 基于 RLinf 分离式异步架构（Disaggregated Async PPO），通过 WireGuard Overlay VPN 打通云端 GPU 与本地真机，实现零端口映射、零 iptables DNAT 的分布式训练。

---

## 阶段一：云服务器选型

### 1.1 选型原则

| 维度 | 要求 | 原因 |
|------|------|------|
| **地域** | 华南（深圳/广州） | 本地在深圳，就近选择降低延迟 |
| **带宽** | ≥ 100 Mbps（建议 200 Mbps） | RLinf 权重同步（~500MB/轮）和 trajectory 传输需要 |
| **CPU/内存** | 2核 4GB 足够 | Hub 只负责数据包转发，不跑训练 |
| **系统** | Ubuntu 22.04 LTS | 内核较新，WireGuard 支持完善 |
| **流量** | 按量计费或包月大流量 | 训练期间流量消耗大（预估 50-200 GB/天） |

### 1.2 推荐配置

| 厂商 | 实例规格 | 带宽 | 预估月费 | 备注 |
|------|---------|------|---------|------|
| **阿里云** | ECS 共享型 n4 / 2核4G | 100 Mbps | ~200-300 元 | 深圳节点，稳定 |
| **腾讯云** | 轻量应用服务器 4M 套餐升级带宽 | 100 Mbps | ~150-250 元 | 广州节点，性价比高 |
| **UCloud** | 快杰云主机 2核4G | 100 Mbps | ~180 元 | 深圳节点，流量包灵活 |
| **AWS** | t3.medium | 按量 | ~$50-80 | 仅推荐已有账号 |

**建议**：选择 **阿里云 ECS（深圳）** 或 **腾讯云轻量（广州）**，购买时选择 **按固定带宽计费 100Mbps**，系统镜像选 **Ubuntu 22.04**。

### 1.3 购买后记录信息

购买完成后，记录以下信息（后续配置用到）：

```bash
# 示例值，请替换为你的实际信息
HUB_PUBLIC_IP=47.107.137.4        # 公网服务器公网IP
HUB_VPN_IP=10.200.200.1           # WireGuard虚拟IP（Hub）
CLOUD_VPN_IP=10.200.200.2         # 云端GPU虚拟IP
LOCAL_VPN_IP=10.200.200.3         # 本地真机虚拟IP
WG_PORT=51820                     # WireGuard监听端口
```

---

## 阶段二：网络架构与 IP 规划

### 2.1 虚拟子网规划

```
WireGuard 子网: 10.200.200.0/24
├─ 10.200.200.1/32  → 公网服务器 (Hub)
├─ 10.200.200.2/32  → 云端 GPU 节点
└─ 10.200.200.3/32  → 本地真机节点
```

### 2.2 端口规划（Ray + RLinf）

| 端口 | 用途 | 绑定地址 | 所在节点 |
|------|------|---------|---------|
| `51820/udp` | WireGuard 隧道 | 公网服务器公网IP | 公网服务器 |
| `6389` | Ray GCS | `10.200.200.2` | 云端 GPU |
| `6391` | Ray Object Manager | `10.200.200.2` / `10.200.200.3` | 两端 |
| `6392` | Ray Node Manager | `10.200.200.2` / `10.200.200.3` | 两端 |
| `20000-20099` | 云端 Worker 端口 | `10.200.200.2` | 云端 GPU |
| `20100-20136` | 本地 Worker 端口 | `10.200.200.3` | 本地真机 |

---

## 阶段三：公网服务器部署（Hub）

### 3.1 系统初始化

```bash
# SSH 登录公网服务器
ssh root@123.45.67.89

# 更新系统
apt-get update && apt-get upgrade -y

# 安装必要工具
apt-get install -y wireguard wireguard-tools iptables-persistent net-tools

# 启用 IP 转发（永久生效）
echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
sysctl -p
```

### 3.2 生成 WireGuard 密钥对

```bash
cd /etc/wireguard

# Hub 密钥
wg genkey | tee hub-private.key | wg pubkey > hub-public.key

# 云端 GPU 密钥（在 Hub 上预生成，后续分发）
wg genkey | tee cloud-private.key | wg pubkey > cloud-public.key

# 本地真机密钥（在 Hub 上预生成，后续分发）
wg genkey | tee local-private.key | wg pubkey > local-public.key

# 查看公钥（后续配置需要）
cat hub-public.key
cat cloud-public.key
cat local-public.key
```

### 3.3 配置 WireGuard 接口

```bash
cat > /etc/wireguard/wg0.conf << 'EOF'
[Interface]
PrivateKey = <粘贴 hub-private.key 内容>
Address = 10.200.200.1/24
ListenPort = 51820

# 开启IP转发，允许两个Peer互相路由
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
# 云端GPU
PublicKey = <粘贴 cloud-public.key 内容>
AllowedIPs = 10.200.200.2/32

[Peer]
# 本地真机
PublicKey = <粘贴 local-public.key 内容>
AllowedIPs = 10.200.200.3/32
EOF
```

### 3.4 启动并设置开机自启

```bash
# 启动接口
wg-quick up wg0

# 设置开机自启
systemctl enable wg-quick@wg0

# 验证状态
wg show
ip addr show wg0
```

**预期输出**：
```
interface: wg0
  public key: <hub公钥>
  private key: (hidden)
  listening port: 51820

peer: <cloud公钥>
  allowed ips: 10.200.200.2/32

peer: <local公钥>
  allowed ips: 10.200.200.3/32
```

### 3.5 防火墙配置

```bash
# 开放 WireGuard UDP 端口
ufw allow 51820/udp
ufw allow OpenSSH
ufw enable

# 确认规则
ufw status verbose
```

---

## 阶段四：云端 GPU 节点部署

### 4.1 安装 WireGuard

```bash
# 登录云端GPU节点
ssh root@172.26.0.82  # 或你的实际登录方式

# 安装
apt-get update
apt-get install -y wireguard wireguard-tools

# 如果容器环境无权限安装，使用 conda/pip 安装用户态实现
# pip install wireguard-tools  # 仅用于密钥生成，wg-quick 仍需 root
```

### 4.2 配置 WireGuard 客户端

```bash
mkdir -p /etc/wireguard

# 将之前在 Hub 上生成的 cloud-private.key 内容写入
cat > /etc/wireguard/wg0.conf << 'EOF'
[Interface]
PrivateKey = <粘贴 cloud-private.key 内容>
Address = 10.200.200.2/24

[Peer]
PublicKey = <粘贴 hub-public.key 内容>
AllowedIPs = 10.200.200.0/24
Endpoint = 123.45.67.89:51820
PersistentKeepalive = 25
EOF
```

### 4.3 启动并验证

```bash
wg-quick down wg0 && wg-quick up wg0
systemctl enable wg-quick@wg0

# 验证
wg show
ping -c 4 10.200.200.1   # 应通 Hub
ping -c 4 10.200.200.3   # 此时本地未连接，应不通
ip addr show wg0
```

---

## 阶段五：本地真机节点部署

### 5.1 macOS 环境（MacBook Air M4）

```bash
# 安装 WireGuard 工具
brew install wireguard-tools

# 创建配置目录
mkdir -p /usr/local/etc/wireguard

# 写入配置
cat > /usr/local/etc/wireguard/wg0.conf << 'EOF'
[Interface]
PrivateKey = <粘贴 local-private.key 内容>
Address = 10.200.200.3/24

[Peer]
PublicKey = <粘贴 hub-public.key 内容>
AllowedIPs = 10.200.200.0/24
Endpoint = 123.45.67.89:51820
PersistentKeepalive = 25
EOF

# 启动（macOS 需使用 wireguard-go 或官方 GUI）
sudo wg-quick up wg0

# 验证
sudo wg show
ping -c 4 10.200.200.1
ping -c 4 10.200.200.2
```

**注意**：macOS 上 `wg-quick` 需要安装 `wireguard-tools` 和 `wireguard-go`。如果遇到 TUN 设备问题，可使用官方 WireGuard App（App Store 下载）导入配置文件。

### 5.2 Linux 环境（如本地是 Linux 工作站）

```bash
sudo apt-get install -y wireguard wireguard-tools

sudo tee /etc/wireguard/wg0.conf << 'EOF'
[Interface]
PrivateKey = <粘贴 local-private.key 内容>
Address = 10.200.200.3/24

[Peer]
PublicKey = <粘贴 hub-public.key 内容>
AllowedIPs = 10.200.200.0/24
Endpoint = 123.45.67.89:51820
PersistentKeepalive = 25
EOF

sudo wg-quick up wg0
sudo systemctl enable wg-quick@wg0
```

### 5.3 验证本地 ↔ 云端双向连通

```bash
# 在本地执行
ping -c 4 10.200.200.2
nc -zv 10.200.200.2 6389
nc -zv 10.200.200.2 6392
nc -zv 10.200.200.2 6391

# 在云端执行
ping -c 4 10.200.200.3
nc -zv 10.200.200.3 6392
nc -zv 10.200.200.3 6391
```

**全部应显示 `succeeded`**。如果失败，检查 Hub 的 `ip_forward` 和防火墙规则。

---

## 阶段六：Ray 集群启动（零 iptables、零 SSH 隧道）

### 6.1 清理旧环境（如之前用过 SSH 隧道）

```bash
# === 在云端执行 ===
ray stop --force
pkill -f "ssh -N -T" || true
pkill socat || true
iptables -t nat -F OUTPUT 2>/dev/null || true

# === 在本地执行 ===
ray stop
pkill -f "ssh.*163[89]" || true
sudo iptables -t nat -F OUTPUT 2>/dev/null || true
```

### 6.2 启动 Ray Head（云端）

```bash
# === 在云端执行 ===
source .venv/bin/activate  # 或你的 conda 环境

export RLINF_NODE_RANK=0

ray start --head   --port=6389   --node-ip-address=10.200.200.2   --object-manager-port=6391   --node-manager-port=6392   --min-worker-port=20000   --max-worker-port=20099   --include-dashboard=false   --disable-usage-stats

# 验证
ray status
# 预期: 1 node Active
```

### 6.3 启动 Ray Worker（本地）

```bash
# === 在本地执行 ===
source .venv/bin/activate

export RLINF_NODE_RANK=1

# 恢复默认心跳参数（不再需要放宽）
ray start   --address='10.200.200.2:6389'   --node-manager-port=6392   --object-manager-port=6391   --min-worker-port=20100   --max-worker-port=20136   --node-ip-address=10.200.200.3   --disable-usage-stats

# 验证
ray status
# 预期: 2 nodes Active
```

### 6.4 验证节点存活

```bash
# === 在云端执行 ===
python -c "
import ray
ray.init(address='auto')
for n in ray.nodes():
    print(f"alive={str(n['Alive']):5s}  ip={n['NodeManagerAddress']:18s}")
ray.shutdown()
"
```

**预期输出**：
```
alive=True   ip=10.200.200.2
alive=True   ip=10.200.200.3
```

---

## 阶段七：RLinf 配置与训练启动

### 7.1 修改 RLinf YAML 配置

```yaml
# 保存为 configs/cloud_edge_async.yaml
algorithm:
  name: async_ppo
  loss_type: decoupled_actor_critic

cluster:
  num_nodes: 2
  node_groups:
    - name: cloud_gpu
      node_ranks: [0]
      env_configs:
        python_interpreter_path: /path/to/cloud/.venv/bin/python
    - name: edge_robot
      node_ranks: [1]
      env_configs:
        python_interpreter_path: /path/to/local/.venv/bin/python

placement:
  env: edge_robot
  rollout: edge_robot
  actor: cloud_gpu
  reference: cloud_gpu

rollout:
  recompute_logprobs: false
  weight_syncer:
    strategy: patch  # 增量同步，减少跨公网流量

actor:
  model:
    add_value_head: true
```

### 7.2 启动训练

```bash
# === 仅在云端执行 ===
cd /path/to/RLinf
source .venv/bin/activate

python examples/embodiment/train_async.py   --config-name cloud_edge_async
```

---

## 阶段八：测试验证

### 8.1 网络层测试

```bash
# === 在本地执行 ===
# 测试到云端所有 Ray 端口的连通性
for port in 6389 6391 6392 20050; do
  echo -n "Testing 10.200.200.2:$port ... "
  timeout 2 nc -zv 10.200.200.2 $port && echo "OK" || echo "FAIL"
done

# === 在云端执行 ===
# 测试到本地所有 Ray 端口的连通性
for port in 6391 6392 20100 20120 20136; do
  echo -n "Testing 10.200.200.3:$port ... "
  timeout 2 nc -zv 10.200.200.3 $port && echo "OK" || echo "FAIL"
done
```

### 8.2 Ray 跨节点任务测试

```bash
# === 在云端执行 ===
python -c "
import ray
import time

ray.init(address='auto')

@ray.remote
def test_remote(x):
    import socket
    return f'Processed on {socket.gethostname()}, input={x}'

# 强制调度到本地节点
ref = test_remote.options(resources={'node:10.200.200.3': 0.1}).remote(42)
result = ray.get(ref)
print(result)

ray.shutdown()
"
```

**预期输出**：包含本地主机名，证明跨节点任务调度成功。

### 8.3 RLinf 训练冒烟测试

```bash
# === 在云端执行 ===
# 先以极小配置跑一轮，验证端到端通路
python examples/embodiment/train_async.py   --config-name cloud_edge_async   rollout.n_envs=1   train_batch_size=2   max_epochs=1
```

观察日志：
- 本地应出现 `EnvWorker` 和 `RolloutWorker` 启动日志
- 云端应出现 `TrainerWorker` 和 `ReferenceWorker` 日志
- 权重同步（`patch`）应成功，无 `ConnectionRefused` 错误

### 8.4 带宽与延迟测试

```bash
# === 在本地执行 ===
# 测试到云端的带宽（安装 iperf3）
iperf3 -c 10.200.200.2 -t 30

# 测试延迟
ping -c 100 10.200.200.2
```

**参考指标**：
- 延迟：深圳 ↔ 深圳公网服务器，预期 5-20ms
- 带宽：应接近你购买的公网服务器带宽（如 100 Mbps）

---

## 阶段九：监控与运维

### 9.1 WireGuard 连接监控

```bash
# 在任意节点查看握手时间
wg show

# 预期: latest handshake 应在 2 分钟内（PersistentKeepalive=25 保证）
```

### 9.2 自动重连保障

WireGuard 本身是无状态的，网络闪断后会自动重连。如果希望更强健，在本地和云端添加 systemd 守护：

```bash
# 已在阶段三/四/五中设置
systemctl enable wg-quick@wg0
```

### 9.3 日志监控脚本

```bash
# === 在云端创建监控脚本 ===
cat > /opt/monitor_ray.sh << 'EOF'
#!/bin/bash
while true; do
  if ! ray status | grep -q "2 node"; then
    echo "$(date): Ray cluster degraded!" >> /var/log/ray_monitor.log
  fi
  sleep 30
done
EOF
chmod +x /opt/monitor_ray.sh
nohup /opt/monitor_ray.sh &
```

---

## 阶段十：安全加固

### 10.1 最小化防火墙

```bash
# === 公网服务器 ===
# 仅允许 WG 端口和 SSH
ufw default deny incoming
ufw default allow outgoing
ufw allow 51820/udp
ufw allow 22/tcp
ufw enable

# === 云端 GPU ===
# 仅允许来自 WG 子网的 Ray 端口
ufw allow from 10.200.200.0/24 to any port 6389,6391,6392,20000:20099 proto tcp
ufw allow 22/tcp
ufw enable

# === 本地真机 ===
# 仅允许来自 WG 子网的 Ray 端口
sudo ufw allow from 10.200.200.0/24 to any port 6391,6392,20100:20136 proto tcp
sudo ufw enable
```

### 10.2 密钥安全

```bash
# 公网服务器上限制密钥文件权限
chmod 600 /etc/wireguard/*-private.key
chmod 600 /etc/wireguard/*.conf

# 定期轮换（建议每月）
wg genkey | tee new-local-private.key | wg pubkey > new-local-public.key
# 更新 wg0.conf 后 reload: wg-quick down wg0 && wg-quick up wg0
```

### 10.3 禁用 Hub 上不必要的服务

```bash
# 公网服务器仅作为 VPN Hub，不跑其他服务
systemctl disable --now apache2 nginx  # 如有
```

---

## 附录：一键检查清单

在提交训练任务前，逐项确认：

- [ ] 公网服务器 `wg show` 显示 2 个 Peer
- [ ] 云端 `ping 10.200.200.3` 成功
- [ ] 本地 `ping 10.200.200.2` 成功
- [ ] 云端 `ray status` 显示 2 nodes
- [ ] 本地 `ray status` 显示 2 nodes
- [ ] 跨节点 Actor 测试成功
- [ ] RLinf 冒烟测试（1 epoch）成功
- [ ] `iperf3` 带宽 ≥ 50 Mbps
- [ ] 防火墙仅开放必要端口

---

## 预期效果

完成上述部署后，RLinf 真机训练将以最原生的方式运行：

- 云端 GPU 和本地真机通过 WireGuard 虚拟子网直连
- Ray 无需任何欺骗性配置（无 iptables、无 socat、无 SSH 隧道）
- Async PPO 的权重同步和 trajectory 传输完全透明
- 运维复杂度从"近 40 个进程 + 100 个端口映射"降至"3 个 WireGuard 接口 + 0 个端口映射"
- 任意随机端口（如 53177）天然支持，无需预先映射
