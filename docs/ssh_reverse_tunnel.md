# SSH 双向隧道 + Ray 双向通信修复方案

## 背景

云端 GPU 服务器与本地真机节点通过 Feilian VPN 连接。本地可访问云端，但云端无法主动访问本地（单向 NAT）。

Ray 节点存活检测涉及两个方向的通信：

1. **本地 raylet → 云端 GCS：注册 + 心跳**（gRPC，端口 6389）
2. **云端 GCS → 本地 raylet：反向健康检查**（连 `NodeManagerAddress:NodeManagerPort`）

此外，`ray start` 启动时本地 raylet 还需要连接 head 节点的 raylet（6392）和 object manager（6391）做 bootstrap 握手。

**最关键的是**：跨节点任务调度时，driver 需要直接与被调度节点的 worker 进程通信。云端 driver 必须能连上本地 worker 的端口（`20000-20036`），反之亦然。

### 端口说明

| 端口 | 用途 | 端 |
|------|------|-----|
| `6389` | GCS（Ray 集群控制服务） | 云端 |
| `6391` | object manager | 两端 |
| `6392` | node manager / raylet gRPC | 两端 |
| `20000-20099` | 云端 worker 端口范围（本地→云端也需 SSH -L + iptables 转发） | 两端 |
| `20100-20136` | 本地 worker 端口范围（通过 SSH -R 中继到云端） | 本地 |
| `16389` | SSH `-L`：本地 → 云端 GCS:6389 | 本地监听 |
| `16393` | SSH `-L`：本地 → 云端 raylet:6392 | 本地监听 |
| `16394` | SSH `-L`：本地 → 云端 objstore:6391 | 本地监听 |
| `16392` | SSH `-R`：云端 → 本地 raylet:6392 | 云端监听 |
| `16391` | SSH `-R`：云端 → 本地 objstore:6391 | 云端监听 |

### IP 地址说明

| 地址 | 含义 | 谁可以访问 |
|------|------|-----------|
| `192.168.115.216` | 本地真机节点 Feilian VPN IP | 云端**不能**直连 |
| `10.190.242.162` | 云端主机内网 IP（Ray 绑定在此） | 本地**不能**直连 |
| `172.26.0.82` | 云端机器学习平台分配的可访问 IP | 本地可通过 SSH / 端口映射访问 |

### 原理

```
本地 (192.168.115.216)                              云端
┌────────────────────────────────────┐          ┌─────────────────────────────────┐
│                                    │          │                                 │
│ ① 本地 raylet 启动                 │          │ ② 云端 Ray Head                 │
│   --address=127.0.0.1:16389       │          │   监听 10.190.242.162            │
│                                    │          │                                 │
│   SSH -L 正向:                     │          │                                 │
│   16389 ──────────────────→ ───────┼──→ ──────┼──→ 10.190.242.162:6389 (GCS)   │
│   16393 ──────────────────→ ───────┼──→ ──────┼──→ 10.190.242.162:6392 (raylet) │
│   16394 ──────────────────→ ───────┼──→ ──────┼──→ 10.190.242.162:6391 (objstr) │
│                                    │          │                                 │
│ 本地 iptables DNAT:                │          │                                │
│   10.190.242.162:6392              │          │                                │
│     → 127.0.0.1:16393              │          │                                │
│   10.190.242.162:6391              │          │                                │
│     → 127.0.0.1:16394              │          │                                │
│   10.190.242.162:20000-20099       │          │                                │
│     → 127.0.0.1:20000-20099        │          │                                │
│                                    │          │                                │
│   SSH -L: 本地 → 云端 worker 端口   │          │                                │
│   workers 20000-20099 ─────────────┼──→ ──────┼──→ 10.190.242.162:20000-20099 │
│                                    │          │                                 │
│   SSH -R 反向:                      │          │                                 │
│   raylet :6392 ────────────────────┼──→ ──────┼──→ 127.0.0.1:16392             │
│   objstr :6391 ────────────────────┼──→ ──────┼──→ 127.0.0.1:16391             │
│   workers 20100-20136 ─────────────┼──→ ──────┼──→ 127.0.0.1:20100-20136       │
│                                    │          │           │                     │
│                                    │          │     socat relay (自启动脚本)     │
│                                    │          │   监听 10.190.242.162           │
│                                    │          │           │                     │
│                                    │          │ 云端 iptables DNAT:             │
│                                    │          │ 192.168.115.216:6392             │
│                                    │          │   → 10.190.242.162:16392        │
│                                    │          │ 192.168.115.216:6391             │
│                                    │          │   → 10.190.242.162:16391        │
│                                    │          │ 192.168.115.216:20100-20136      │
│                                    │          │   → 10.190.242.162:20100-20136  │
└────────────────────────────────────┘          └─────────────────────────────────┘
```

---

## 执行步骤

**注意**：后续命令中的 `<feilian_iface>` 需替换为飞连 VPN 网卡名（通过 `ip addr | grep 192.168.115.216` 查看，通常为 `tun0` 或 `utun0`）。

---

### 步骤 0：确认端口无冲突

```bash
# === 在云端执行 ===
ss -tlnp | grep -E "6389|6391|6392|20000"
# 预期无输出（所有端口空闲）

# === 在本地执行 ===
ss -tlnp | grep -E "6391|6392|20000"
# 预期无输出。如有进程占用（如 VS Code），需换端口或关掉该进程
```

---

### 步骤 1：检查并停止旧的 Ray 会话

```bash
# === 在云端执行 ===
ray stop --force
sleep 3
pkill -f "ssh -N -T" || true
pkill socat || true
iptables -t nat -F OUTPUT 2>/dev/null || true

# 确认无残留
ps aux | grep -E 'ray|gcs_server|raylet' | grep -v grep
```

```bash
# === 在本地执行 ===
ray stop
pkill -f "ssh.*163[89]" || true
sudo iptables -t nat -F OUTPUT 2>/dev/null || true

# 确认无残留
ps aux | grep -E 'ray|gcs_server|raylet' | grep -v grep
```

---

### 步骤 2：启动云端 Ray Head

```bash
# === 在云端执行 ===
source .venv/bin/activate

export RLINF_NODE_RANK=0

ray start --head \
  --port=6389 \
  --node-ip-address=10.190.242.162 \
  --object-manager-port=6391 \
  --node-manager-port=6392 \
  --min-worker-port=20000 \
  --max-worker-port=20099 \
  --include-dashboard=false \
  --disable-usage-stats

# 验证
ray status
# 预期: 1 node Active

# 确认无残留 named actor
ray list actors --filter "state=ALIVE" 2>/dev/null || python -c "
import ray; ray.init(address='auto')
actors = ray._private.state.actors()
for aid, info in actors.items():
    if info.get('State') == 'ALIVE':
        print(info.get('Name', ''))
ray.shutdown()
"
# 预期: 没有任何 ALIVE 的 named actor
```

---

### 步骤 3：本地建立 SSH 双向隧道 + 本地 iptables DNAT

> **必须在启动本地 Ray Worker 之前完成！**

```bash
# === 在本地执行 ===

# 3.1 杀掉可能残留的旧隧道
pkill -f "ssh.*163[89]" 2>/dev/null || true

# 3.2 安装 iptables（如未安装）
sudo apt-get install -y iptables

# 3.3 清理本地旧的 DNAT 规则
sudo iptables -t nat -F OUTPUT 2>/dev/null || true

# 3.4 启用 route_localnet（DNAT 到 127.0.0.1 必需，Ubuntu 默认关闭）
sudo sysctl -w net.ipv4.conf.all.route_localnet=1

# 3.5 建立 SSH 隧道
#     正向 -L：GCS(16389) + raylet(16393) + objstore(16394) + 云端 worker(20000-20099) ← 本地 → 云端
#     反向 -R：raylet(16392) + objstore(16391) + 本地 worker(20100-20136) ← 云端 → 本地
#
#     使用 eval 拼接动态端口参数
ssh -N -T \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -L '*:16389:10.190.242.162:6389' \
  -L '*:16393:10.190.242.162:6392' \
  -L '*:16394:10.190.242.162:6391' \
  $(for p in $(seq 20000 20099); do echo "-L *:$p:10.190.242.162:$p"; done) \
  -R 16392:localhost:6392 \
  -R 16391:localhost:6391 \
  $(for p in $(seq 20100 20136); do echo "-R $p:localhost:$p"; done) \
  172.26.0.82 &

# 3.6 确认隧道进程存活
ps aux | grep "ssh.*163[89]" | grep -v grep

# 3.7 验证正向隧道
timeout 5 nc -zv 127.0.0.1 16389   # GCS
timeout 5 nc -zv 127.0.0.1 16393   # head raylet
timeout 5 nc -zv 127.0.0.1 16394   # head object manager
# 预期: 三条 Connection ... succeeded!

# 3.8 在云端验证反向隧道端口已监听
ssh 172.26.0.82 "ss -tlnp | grep -E '16391|16392|20100' | head -10"
# 预期: 应包含 127.0.0.1:16391, 127.0.0.1:16392, 127.0.0.1:20100

# 3.9 本地 iptables DNAT：劫持发往云端内网 IP 的流量到 SSH 正向隧道
sudo iptables -t nat -A OUTPUT \
  -d 10.190.242.162 -p tcp --dport 6392 \
  -j DNAT --to-destination 127.0.0.1:16393

sudo iptables -t nat -A OUTPUT \
  -d 10.190.242.162 -p tcp --dport 6391 \
  -j DNAT --to-destination 127.0.0.1:16394

# 云端 worker 端口范围（本地 raylet 需调用云端 NodeManager/actor）
sudo iptables -t nat -A OUTPUT \
  -d 10.190.242.162 -p tcp --dport 20000:20099 \
  -j DNAT --to-destination 127.0.0.1

# 3.10 确认本地 DNAT 规则已生效
sudo iptables -t nat -L OUTPUT -n -v | grep "10.190.242.162"
# 预期输出: 三行 DNAT 规则
```

---

### 步骤 4：云端配置 socat 中继

SSH 反向隧道端口监听在 `127.0.0.1`，而 iptables DNAT 出站后无法路由到 `127.0.0.1`（需要 `route_localnet=1`，但容器文件系统只读不可设置）。用 socat 把隧道端口对外暴露到云端的 `10.190.242.162` 上。

```bash
# === 在云端执行 ===

# 4.1 安装 socat（如未安装）
apt-get install -y socat

# 4.2 启动 socat 中继（后台运行）
#     管理员端口（2 个） + 本地 worker 端口范围（37 个，20100-20136）
socat TCP-LISTEN:16392,reuseaddr,fork,bind=10.190.242.162 TCP:127.0.0.1:16392 &
socat TCP-LISTEN:16391,reuseaddr,fork,bind=10.190.242.162 TCP:127.0.0.1:16391 &

# 本地 Worker 端口范围（20100-20136）
for port in $(seq 20100 20136); do
  socat TCP-LISTEN:$port,reuseaddr,fork,bind=10.190.242.162 TCP:127.0.0.1:$port &
done

# 4.3 确认 socat 已监听（管理员端口）
ss -tlnp | grep -E "16391|16392" | grep "10.190.242.162"

# 4.4 确认 socat 已监听（worker 端口，抽查前 3 个）
ss -tlnp | grep -E "20100|20101|20102" | grep "10.190.242.162"
# 预期: 3 行 LISTEN，源地址为 10.190.242.162
```

---

### 步骤 5：云端配置 iptables DNAT

劫持云端进程发往本地 IP 的出站连接，重定向到 socat 中继。

```bash
# === 在云端执行 ===

# 5.1 安装 iptables（如未安装）
apt-get install -y iptables

# 5.2 管理员端口
iptables -t nat -A OUTPUT \
  -d 192.168.115.216 -p tcp --dport 6392 \
  -j DNAT --to-destination 10.190.242.162:16392

iptables -t nat -A OUTPUT \
  -d 192.168.115.216 -p tcp --dport 6391 \
  -j DNAT --to-destination 10.190.242.162:16391

# 5.3 Worker 端口范围（本地 worker 用 20100-20136，不与云端 20000-20036 冲突）
iptables -t nat -A OUTPUT \
  -d 192.168.115.216 -p tcp --dport 20100:20136 \
  -j DNAT --to-destination 10.190.242.162

# 5.4 确认规则已生效
iptables -t nat -L OUTPUT -n -v | grep -E "192.168.115.216"
# 预期: 三行 DNAT 规则
```

---

### 步骤 6：启动本地 Ray Worker

```bash
# === 在本地执行 ===
source .venv/bin/activate

export RLINF_NODE_RANK=1

# 放宽 Ray 心跳容忍（本地 raylet 通过 SSH 隧道向云端 GCS 发心跳，
# SSH 长连接在 Feilian VPN 上间歇性断连，默认 5 次丢失即标记 dead 太敏感）
export RAY_health_check_initial_delay_ms=30000
export RAY_health_check_period_ms=10000
export RAY_health_check_timeout_ms=60000
export RAY_num_heartbeats_timeout=300

# worker 端口范围：本地用 20100-20136，云端用 20000-20036，互不冲突
ray start \
  --address='127.0.0.1:16389' \
  --node-manager-port=6392 \
  --object-manager-port=6391 \
  --min-worker-port=20100 \
  --max-worker-port=20136 \
  --node-ip-address=192.168.115.216 \
  --disable-usage-stats

# 验证：应显示 2 node Active
ray status
```

---

### 步骤 7：验证 Ray 节点存活

```bash
# === 在云端执行 ===
.venv/bin/python -c "
import ray
ray.init(address='auto')
for n in ray.nodes():
    print(f\"alive={str(n['Alive']):5s}  ip={n['NodeManagerAddress']:18s}\")
ray.shutdown()
"
```

预期输出：

```
alive=True   ip=10.190.242.162
alive=True   ip=192.168.115.216
```

---


## 常见问题

### Q1: SSH 隧道断了怎么办？

使用了 `ServerAliveInterval=60`（每 60 秒心跳），`ServerAliveCountMax=3`（连续 3 次无响应才判定断开）。Feilian VPN 短暂中断后恢复时，SSH 会自行重连。

如需更强健的自动重连，用 `autossh`：

```bash
# === 在本地执行 ===
apt-get install -y autossh

pkill -f "ssh.*163[89]" 2>/dev/null || true

autossh -M 0 -N -T \
  -o ServerAliveInterval=60 \
  -o ExitOnForwardFailure=yes \
  -L '*:16389:10.190.242.162:6389' \
  -L '*:16393:10.190.242.162:6392' \
  -L '*:16394:10.190.242.162:6391' \
  $(for p in $(seq 20000 20099); do echo "-L *:$p:10.190.242.162:$p"; done) \
  -R 16392:localhost:6392 \
  -R 16391:localhost:6391 \
  $(for p in $(seq 20100 20136); do echo "-R $p:localhost:$p"; done) \
  172.26.0.82 &
```

### Q2: 云端报 "Address already in use"

```bash
# === 在云端执行 ===
ss -tlnp | grep -E "16391|16392|20100"

# 清理残留
pkill socat
```

### Q3: 步骤 6 `ray start` 超时或失败

按顺序排查：

```bash
# === 在本地执行 ===
# 1. 确认端口无冲突
ss -tlnp | grep -E "6391|6392|20000"

# 2. 正向隧道是否全通
nc -zv 127.0.0.1 16389   # GCS
nc -zv 127.0.0.1 16393   # head raylet
nc -zv 127.0.0.1 16394   # head objstr
nc -zv 127.0.0.1 20050   # 云端 worker（抽查）

# 3. 本地 iptables 规则是否存在
sudo iptables -t nat -L OUTPUT -n -v | grep "10.190.242.162"

# 4. route_localnet 是否已启用
sysctl net.ipv4.conf.all.route_localnet

# 5. 本地 raylet 日志
cat /tmp/ray/session_latest/logs/raylet.err | tail -20
```

### Q4: 步骤 7 本地节点 alive=False

按顺序排查：

```bash
# === 在本地执行 ===
nc -zv 127.0.0.1 16389 && nc -zv 127.0.0.1 16393 && nc -zv 127.0.0.1 16394
tail -20 /tmp/ray/session_latest/logs/raylet.err

# === 在云端执行 ===
iptables -t nat -L OUTPUT -n -v | grep "192.168.115.216"
ss -tlnp | grep -E "16391|16392" | grep "10.190.242.162"
nc -zv 10.190.242.162 16392
```

### Q5: Worker 报 "FD Shutdown" / 跨节点任务超时

确认 worker 端口隧道是否全部建立：

```bash
# === 在云端执行 ===
# 抽查几个本地 worker 端口
nc -zv 10.190.242.162 20100
nc -zv 10.190.242.162 20110
nc -zv 10.190.242.162 20136

# 确认所有 socat 实例都在运行
ps aux | grep socat | grep -v grep | wc -l
# 预期: 39（2 管理员端口 + 37 worker 端口）
```

### Q6: 本地 IP 变了怎么办？

```bash
# === 在本地执行，获取新 IP ===
NEW_LOCAL_IP=$(ip addr show <feilian_iface> | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
echo "新本地 IP: $NEW_LOCAL_IP"

# === 在云端执行，更新 iptables ===
# 删旧
iptables -t nat -D OUTPUT -d 192.168.115.216 -p tcp --dport 6392 -j DNAT --to-destination 10.190.242.162:16392
iptables -t nat -D OUTPUT -d 192.168.115.216 -p tcp --dport 6391 -j DNAT --to-destination 10.190.242.162:16391
iptables -t nat -D OUTPUT -d 192.168.115.216 -p tcp --dport 20100:20136 -j DNAT --to-destination 10.190.242.162
# 加新
iptables -t nat -A OUTPUT -d $NEW_LOCAL_IP -p tcp --dport 6392 -j DNAT --to-destination 10.190.242.162:16392
iptables -t nat -A OUTPUT -d $NEW_LOCAL_IP -p tcp --dport 6391 -j DNAT --to-destination 10.190.242.162:16391
iptables -t nat -A OUTPUT -d $NEW_LOCAL_IP -p tcp --dport 20100:20136 -j DNAT --to-destination 10.190.242.162
```

### Q7: 步骤 8 卡在 `Waiting for 2 nodes` 后无输出

RLinf 的 `Cluster` 初始化时会通过 `NodeProbe` 向每个节点派发 `_RemoteNodeProbe` actor。如果 actor 未成功调度（named actor 冲突 / 资源不足），RLinf 会永久等待。

按顺序排查：

```bash
# === 在云端执行 ===
python -c "
import ray; ray.init(address='auto')
for aid, info in ray._private.state.actors().items():
    print(f\"{info.get('Name','')} state={info.get('State','')} node={info.get('Address',{}).get('NodeID','')[:16]}\")
ray.shutdown()
" 2>/dev/null

# === 在本地执行 ===
grep -r "RemoteNodeProbe\|NodeProbe" /tmp/ray/session_latest/logs/worker-*.err 2>/dev/null | head -5
ps aux | grep ray:: | grep -v grep | wc -l
```

---

## 清理（不再需要隧道时）

```bash
# === 云端 ===
iptables -t nat -D OUTPUT -d 192.168.115.216 -p tcp --dport 6392 -j DNAT --to-destination 10.190.242.162:16392
iptables -t nat -D OUTPUT -d 192.168.115.216 -p tcp --dport 6391 -j DNAT --to-destination 10.190.242.162:16391
iptables -t nat -D OUTPUT -d 192.168.115.216 -p tcp --dport 20100:20136 -j DNAT --to-destination 10.190.242.162
pkill socat || true
ray stop

# === 本地 ===
sudo iptables -t nat -D OUTPUT -d 10.190.242.162 -p tcp --dport 6392 -j DNAT --to-destination 127.0.0.1:16393
sudo iptables -t nat -D OUTPUT -d 10.190.242.162 -p tcp --dport 6391 -j DNAT --to-destination 127.0.0.1:16394
sudo iptables -t nat -D OUTPUT -d 10.190.242.162 -p tcp --dport 20000:20099 -j DNAT --to-destination 127.0.0.1
kill %1 2>/dev/null || pkill -f "ssh.*163[89]"
ray stop
```
