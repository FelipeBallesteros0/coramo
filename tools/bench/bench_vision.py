"""FPS de YOLO11n (personas) en CUDA sobre un cuadro de 640x480. venv: ~/venvs/vision"""
import time, cv2
from ultralytics import YOLO
from ultralytics.utils import ASSETS
m = YOLO("yolo11n.pt")
img = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (640, 480))
for _ in range(10): m.predict(img, device=0, imgsz=640, classes=[0], verbose=False)  # calentamiento
N = 200; t0 = time.perf_counter()
for _ in range(N): r = m.predict(img, device=0, imgsz=640, classes=[0], verbose=False)
dt = (time.perf_counter() - t0) / N
print(f"YOLO11n 640x480 CUDA: {1/dt:.0f} FPS | {dt*1000:.1f} ms/cuadro | personas detectadas: {len(r[0].boxes)}")
