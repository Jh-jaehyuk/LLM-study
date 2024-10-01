"""
학습 후 동적 양자화

동적 양자화를 적용하면, 모델의 모든 가중치는 f32에서 uint8으로 전환됨
하지만, 활성화에 대한 계산을 진행하기 직전까지는 활성 함수는 uint8으로 전환되지 않음
나중에 얘기할 정적 양자화보다 정확도가 높으나,
현재 torch에서는 nn.Linear와 nn.LSTM만 qconfig_spec 옵션으로
지원한다는 문제가 있음

qconfig_spec이란 model 내에서 양자화 적용 대상인 내부 모듈을 지정하는 것
nn.Linear와 nn.LSTM만 지원한다는 의미는
nn.Conv2d와 같은 모듈을 양자화해야될 때는 동적 양자화를 사용할 수 없음을 의미함
"""
import torch
import torchvision

model = torchvision.models.mobilenet_v2(pretrained=True)

# f32 -> uint8 으로 동적 양자화
# 동적 양자화는 코드가 간단해서
# 모델을 양자화 해야하는 상황에서 가장 쉽게 선택할 수 있는 선택지
# 단, 위에서 언급한 단점을 고려해야함
model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec=[torch.nn.Linear], dtype=torch.qint8
)
