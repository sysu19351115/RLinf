import torch
import time

# 1. 基础张量运算
print("=== Test 1: Basic Tensor Operations ===")
a = torch.randn(10000, 10000, device='cuda')
b = torch.randn(10000, 10000, device='cuda')
c = torch.matmul(a, b)
print(f"Matrix multiply result shape: {c.shape}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 2. BF16 混合精度测试（Blackwell 原生支持）
print("\n=== Test 2: BF16 Mixed Precision ===")
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    d = torch.randn(4096, 4096, device='cuda')
    e = torch.matmul(d, d)
print(f"BF16 matmul result dtype: {e.dtype}")

# 3. FP8 测试（Blackwell 新特性，可选）
print("\n=== Test 3: FP8 Support (Blackwell-specific) ===")
try:
    # PyTorch 2.7+ 可能支持 FP8
    f = torch.randn(4096, 4096, device='cuda', dtype=torch.float8_e4m3fn)
    g = torch.matmul(f, f)
    print(f"FP8 matmul: SUCCESS")
except Exception as ex:
    print(f"FP8 matmul: {ex}")

# 4. 神经网络前向/反向传播
print("\n=== Test 4: Neural Network Forward/Backward ===")
model = torch.nn.Sequential(
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 4096),
).cuda()
x = torch.randn(128, 4096, device='cuda')
y = model(x)
loss = y.sum()
loss.backward()
print(f"Backward pass: SUCCESS")
print(f"Gradients computed: {model[0].weight.grad is not None}")

# 5. 性能基准
print("\n=== Test 5: Performance Benchmark ===")
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    a = torch.randn(8192, 8192, device='cuda')
    b = torch.randn(8192, 8192, device='cuda')
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
elapsed = time.time() - start
print(f"10x (8192x8192) matmul: {elapsed:.2f}s, avg {elapsed/10*1000:.1f}ms/op")

print("\n=== ALL TESTS PASSED ===")