# register_buffer

Q: register_buffer에 등록된 텐서는 클래스 내부에서 self로 호출 가능한가요?
A: 맞습니다. 아래 예시를 보면 조금 더 이해가 될 것 같아요.  

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("a", torch.tensor(3))
    
testModule = Net()
print(testModule.a) # 출력값: tensor(3)
```
  
# view

Q: view 함수가 어떻게 동작하는거죠?  
A: view 함수는 아래와 같이 동작합니다.  

```python
import torch
import torch.nn as nn

A = torch.randn([1, 23])
B = A.view(2, 2, -1, 3)

print(A)
print(B)
```
   
A 텐서의 내부적인 요소 개수는 유지하면서 형태만 다르게 보여줍니다.  
2 * 3 * 4 = 2 * 2 * 2 * 3 좌우 계산 값이 동일해야 함을 의미함.  
계산 값이 다르면 오류가 발생합니다.  

추가적으로 reshape와 view의 차이가 무엇이냐 하면,  
이전에 view는 contiguous한 텐서에서만 사용가능하다고 했었습니다.  
반면 reshape는 바꾸고자 하는 텐서가 contiguous하다면,  
tensor.contiguous()를 사용하여 메모리 연속적으로 만들어주고  
view()를 적용하여 그 값을 반환해줍니다.