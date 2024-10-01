import torch
import torchvision

model = torchvision.models.mobilenet_v2(pretrained=True)
# 모바일 장비는 일반적으로 ARM 아키텍처를 탑재함
# ARM 아키텍처에서 모델이 작동되게 하려면 qnnpack을 파라미터로 넣어줘야 함
# x86 아키텍처가 탑재된 컴퓨터라면 x86을 파라미터로 사용함
# x86 대신 이전에 사용하던 'fbgemm' 또한 사용가능하지만 권장하지 않는다고 함
model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
torch.backends.quantized.engine = "qnnpack"
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
