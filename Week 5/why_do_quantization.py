import os.path

import torch
import torchvision

"""
양자화 왜 해요?

양자화된 모델은 원본 모델과 거의 같은 정확도를 내면서
사이즈가 줄어들고 추론 속도가 빨라짐

서버 모델과 모바일 모델 모두 적용 가능하지만, 특히 모바일 환경에서 매우 중요함
양자화를 적용하지 않은 모델의 크기가 iOS나 Android 앱에서 허용하는 크기 한도를
초과할 수 있고, 그로 인해 모델의 배포나 업데이트가 오래 걸릴 수 있다.
또한, 추론 속도가 너무 느려서 사용자가 쾌적하게 사용하기 어려움!
"""

# 양자화된 모델 불러오기
model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)

# 양자화 되지 않은 모델 불러오기
model = torchvision.models.mobilenet_v2(pretrained=True)

# 모델 사이즈 비교용 함수
def print_model_size(model):
    torch.save(model.state_dict(), "model.pt")
    print(f"{os.path.getsize('model.pt') / 1e6}MB") # 모델 사이즈 가져오기
    os.remove("model.pt") # 필요없으니 삭제

print("Normal model size:")
print_model_size(model) # 14.24MB
print("Quantized model size:")
print_model_size(model_quantized) # 3.62MB
