"""Carga sostenida en la 4070 durante N segundos. Uso: python gpu_load.py 120"""
import sys, time, torch
segundos = int(sys.argv[1]) if len(sys.argv) > 1 else 60
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
t0 = time.time(); n = 0
while time.time() - t0 < segundos:
    c = a @ b; n += 1
    if n % 50 == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
tflops = n * 2 * 8192**3 / (time.time() - t0) / 1e12
print(f"{n} multiplicaciones en {segundos} s ({tflops:.1f} TFLOPS fp16); max VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
