# Week 3

> 💡**요약** : **Transformer Decoder**

## Recap
* **Self-Attention**  
입력 문장(자신)에 대해 Query, Key를 보내 Attention 연산을 하므로 Self-Attention이라고 함  
Self-Attention을 통해서 같은 문장 내 모든 단어 사이의 의미적, 문법적 관계를 포착  
* **Multi-Head Self-Attention(MHA)**  
동일한 문장에 대해 여러 명의 사람(Head)가 관계를 분석하는 것과 유사함  
한 명의 사람이 분석하는 것보다 더 일반화되고 좋은 연관 관계를 분석 가능  
* **Add & Layer Normalization**  
  * Add: Layer가 깊어지더라도 학습을 용이하게 만듦(이전 정보를 잘 잊지 않는다는 의미)
  * Layer Normalization: CNN에 비해 상대적으로 무거운 모델인 Transformer 구조에서   
Batch Normalization 방식보다 메모리 효율적으로 학습 가능
* **Feed Forward Netword(FFN)**  
Feed Forward에서는 단어 그 자체의 의미를 파악하는데 중점을 둠
  
---
## Transformer
![Transformer structure](https://github.com/user-attachments/assets/2389f7c5-806a-4555-9e5d-f68a4cca3df2)  

### Encoder vs. Decoder
> 인공지능 모델에는 Encoder & Decoder 개념이 많이 나옴

1. **Encoder**  
* 데이터의 내재된 특징들을 추출하는 기능을 수행  
* 특징을 막 추출하는 것이 아니라 입력 데이터를 잘 표현하기 위한 양질의 특징을 추출  
* 추출된 특징들은 가중치 행렬에 저장됨  
→ **컴퓨터가 데이터를 잘 이해할 수 있도록 함**  

2. **Decoder**  
* Encoder에서 얻은 양질의 특징을 하고싶은 작업에 잘 적용할 수 있도록 바꿔주는 기능을 수행  
→ **추출된 정보를 컴퓨터가 아닌 사람이 잘 이해하고 사용할 수 있도록 함**

### Decoder
* **Masked Self-Attention**
```text
현재 시점보다 뒤에 있는 단어를 참조하지 못하도록 가림(Masking)
```

![Masked Self-Attention1](https://github.com/user-attachments/assets/b7fa94f3-57a0-4fae-93ee-fc11275fd892)  
![Masked Self-Attention2](https://github.com/user-attachments/assets/a14a8159-01ff-46de-b6e2-aa826227192a)
  
* **Encoder-Decoder Attention**
```text
Self-Attention이 자기 자신과 비교했다면,
여기는 Encoder를 통해 추출한 정보와 Decoder의 입력을 서로 비교

말 그대로 Encoder 정보와 Decoder 입력 사이 유사도를 구하기 위함
```
![Encoder-Decoder Attention](https://github.com/user-attachments/assets/d7c62e44-8ae0-415c-93ce-e73cdd32a0b3)

* **Output(예측 결과)**
```
여러 개의 단어 중 어떤 단어를 output으로 선택할 지 분류(Classification)하는 것과 같음
```
![Output](https://github.com/user-attachments/assets/db0f5082-058f-48ae-9bcb-5f1a043db68c)  
  
### Train vs. Predict
![image](https://github.com/user-attachments/assets/37b264fa-2046-45ad-996e-71b2b07485b5)  
![image](https://github.com/user-attachments/assets/0a2378aa-d655-43d8-9fd3-8c36ff98e7ee)  
![image](https://github.com/user-attachments/assets/ffb01ae3-511a-406f-8a3f-ae0a0a8f01d9)  
![image](https://github.com/user-attachments/assets/a6808b16-08be-49c7-9f9e-2907158d1fb3)  
  
---
### 결 론
* **Masked Self-Attention**  
다음 단어를 예측하는데, 뒤의 단어를 미리 알게되는 것은 반칙이므로   
뒤의 내용을 못 보게 가린(Masking) 후 Self-Attention을 하는 것  

* **Encoder-Decoder Attention**  
Encoder로부터 Key, Value를 받고 Decoder로부터 Query를 받아 Attention 연산함  
번역된 다음 단어 예측을 위함  

* **Workflow in Traning Process**  
입력으로 원본 시퀀스와 함께 해당 시퀀스에 대한 정답 레이블(번역된 문장)을 제공  
Decoder가 이전 단계에서 예측한 단어를 사용하는 대신,  
실제 정답 시퀀스에서 이전 단어를 사용하여 다음 단어를 예측하는 방식  

* **Workflow in Testing Process**  
테스트 단계에서는 모델이 새로운 입력(학습 데이터에 없는 문장)에 대해 예측을 수행  
Decoder는 이전 단계에서 예측한 단어를 사용해 다음 단어를 생성함
