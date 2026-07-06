# 跨节点测试：云端 driver → 本地 node
#
# 在云端执行：
#   source .venv/bin/activate
#   python tests/unit_tests/diag_cloud_to_local.py

import ray, socket, time

print("=== Connecting ===")
ray.init(address="auto")
start_time = time.time()

# Check both nodes
print(f"\nNodes ({len(ray.nodes())}):")
for n in ray.nodes():
    print(f"  {n['NodeManagerAddress']:18s} alive={n['Alive']} "
          f"CPU={n['Resources'].get('CPU','?')} "
          f"node_id={n['NodeID'][:16]}")

# Find cloud / local node IDs by fixed IPs
#   cloud = 192.168.3.223, local = 192.168.3.224
cloud_id = None
cloud_ip = None
local_id = None
local_ip = None
for n in ray.nodes():
    addr = n["NodeManagerAddress"]
    if "192.168.3.223" in addr:
        cloud_id = n["NodeID"]
        cloud_ip = addr
    elif "192.168.3.224" in addr:
        local_id = n["NodeID"]
        local_ip = addr

if not cloud_id:
    print("\nERROR: cloud node not found (no 192.168.3.223 address)")
    ray.shutdown()
    exit(1)
if not local_id:
    print("\nERROR: local node not found (no 192.168.3.224 address)")
    ray.shutdown()
    exit(1)

print(f"  Cloud node: {cloud_ip}")
print(f"  Local node: {local_ip}")

@ray.remote
def where():
    import socket, os
    return f"host={socket.gethostname()} pid={os.getpid()}"

# ---- Test 1: task on LOCAL node, long timeout, verbose errors ----
print(f"\n{'='*60}")
print(f"Test 1: cloud driver -> LOCAL node ({local_id[:16]}...)")
print(f"{'='*60}")

sched = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
    node_id=local_id, soft=False)

t0 = time.time()
ref = where.options(scheduling_strategy=sched).remote()

# Wait up to 60 seconds, checking progress
for i in range(60):
    ready, not_ready = ray.wait([ref], timeout=1)
    if ready:
        elapsed = time.time() - t0
        result = ray.get(ref)
        print(f"\n  SUCCESS after {elapsed:.1f}s: {result}")
        break
    if i == 0:
        print(f"  Waiting for local worker...", end="", flush=True)
    elif i % 5 == 0:
        print(f".{i}s", end="", flush=True)
else:
    elapsed = time.time() - t0
    print(f"\n  TIMEOUT after {elapsed:.0f}s")
    print(f"  Task ref: {ref}")

# ---- Test 2: cloud baseline ----
print(f"\n{'='*60}")
print(f"Test 2: cloud driver -> CLOUD node ({cloud_id[:16]}...)")
print(f"{'='*60}")

cloud_sched = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
    node_id=cloud_id, soft=False)

t0 = time.time()
for i in range(3):
    try:
        ref = where.options(scheduling_strategy=cloud_sched).remote()
        print(f"  {ray.get(ref, timeout=5)}")
    except Exception as e:
        print(f"  ERROR: {e}")
print(f"  Done in {time.time()-t0:.1f}s")

# ---- Test 3: check local node still alive ----
print(f"\n{'='*60}")
print("Test 3: node status after tests")
print(f"{'='*60}")

for n in ray.nodes():
    print(f"  {n['NodeManagerAddress']:18s} alive={n['Alive']}")

total = time.time() - start_time
print(f"\nTotal time: {total:.0f}s")
ray.shutdown()
