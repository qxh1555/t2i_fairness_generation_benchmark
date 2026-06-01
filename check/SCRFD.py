import torch
import onnxruntime as ort

# 关键：在 InsightFace 创建 ONNX session 之前预加载 CUDA/cuDNN 动态库
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(cuda=True, cudnn=True, directory="")

print("torch cuda:", torch.cuda.is_available(), torch.version.cuda)
print("onnxruntime providers:", ort.get_available_providers())


from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))