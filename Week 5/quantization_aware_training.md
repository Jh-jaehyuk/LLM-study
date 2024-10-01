# 양자화를 고려한 학습(Quantization Aware Training)
> 양자화를 고려한 학습은 모델 학습 과정에서 모든 가중치와
> 활성 함수에 가짜 양자화를 삽입함
> 학습 후 양자화하는 방법보다 높은 추론 정확도를 가진다
> 주로 CNN 모델에 사용된다고 함

QAT를 가능하게 하려면 모델 정의부분의 `__init__` 메서드에서 
*QuantStub*과 *DeQuantStub*을 정의해야함

```python
import torch.quantization

# tensor를 f32에서 양자화된 자료형으로 전환
self.quant = torch.quantization.QuantStub()
# tensor를 양자화된 자료형에서 f32로 전환
self.dequant = torch.quantization.DeQuantStub()
```
  
모델 정의 부분의 `forward` 메서드의 시작과 끝 부분에서   
`x = self.quant(x)` 와 `x = self.dequant(x)` 를 호출해야함  
  
```python
import torch
import torchvision

model = torchvision.models.mobilenet_v2(pretrained=True)

model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
model_qat = torch.quantization.prepare_qat(model, inplace=False)
# QAT가 이 사이에 정의되어야 함
model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)
```

