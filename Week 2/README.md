# Week 2

> 💡**요약** : **Transformer Encoder**

## Recap
* **RNN의 단점**
1. Long-Term Dependencies
2. Recurrent 연산이 너무 무거움
* **RNN의 단점을 극복한 Transformer**  
  → *Transformer = RNN + Attention + Positional Encoding - Recurrent*
* **Attention**  
  → 현재 시점의 단어와 입력에 들어온 모든 단어 사이의 연관도를 내적을 이용하여 판단
* **Positional Encoding**  
  → 입력의 선후 관계(순서)를 알려주는 벡터
  
---
## Transformer
![Transformer structure](https://github.com/user-attachments/assets/2389f7c5-806a-4555-9e5d-f68a4cca3df2)  

### Encoder
* **Self-Attention**  
![Self attention-1](https://github.com/user-attachments/assets/eecb2b2d-1b3a-4a18-8aac-ff20b5fce23e)  
![Self attention-2](https://github.com/user-attachments/assets/6c18de49-985e-4e36-8f65-fabea9ff8e0c)  

  **Q(Query)**: 입력에서 관련된 부분을 찾으려고 하는 정보 벡터  
  **K(Key)**: 관계의 연관도를 결정하기 위해 Q와 비교하는데 사용되는 벡터  
  **V(Value)**: 특정 K에 해당하는 정보를 얼마나 가져올지에 대한 표현
  ![Self attention-3](https://github.com/user-attachments/assets/9896f527-ecaf-4c78-ab42-52cd3e466269)
    
  
* **Multi-Head Self-Attention(MHA)**  
🔎이 전 과정은 Head가 하나일 때의 Self-Attention    
MHA는 _Head를 여러 개_ 두어 각 Head마다 Attention을 수행함!  
⭐️Head가 하나인 경우보다 **더 좋은 성능**을 갖을 수 있다.  
![Multi Head Self attention-1](https://github.com/user-attachments/assets/be6cf61d-515a-46a4-a56a-d417dfeac1ac)
  
  
* **Add & Layer Normalization(LN)**
  > CNN에서 Feature map에 대한 Normalization을 위해 
  > Batch Normalization을 적용했듯이, Transformer에서는
  > Layer Normalization을 수행함

  **🤔Normalization 왜 씀?**
  1. **학습 안정성 향상**  
  2. **더 빠른 학습 속도**  
  3. **과적합 방지**  
  
  → Normalization은 딥러닝 모델의 성능을 향상시키고 학습을 보다 효율적이고 안정적으로 만들기 위해 필수적인 과정

  ![Add and LN-1](https://github.com/user-attachments/assets/b882a014-331f-4fab-9ac4-54287b713740)  
  ![Add and LN-2](https://github.com/user-attachments/assets/5c88cb49-3642-41d1-a1ed-7d3aae72e24c)
  ![Add and LN-3](https://github.com/user-attachments/assets/3d5f2cf9-4395-43f9-a4db-d637c92ce7d6)

---
### 결 론
![Result-1](https://github.com/user-attachments/assets/8d1a12ea-55f2-48c8-ab71-b2a73a03d73e)

* **Self-Attention**  
입력 문장(자신)에 대해 Query, Key로 보내 Attention 연산을 하므로 Self-Attention이라 부름!  
Self-Attention을 통해서 같은 문장 내 모든 단어 사이의 의미적, 문법적 관계를 포착
  

* **Multi-Head Self-Attention(MHA)**  
동일한 문장에 대해 여러 명의 독자(head)가 관계를 분석  
혼자 했을 때 보단 좀 더 일반화되고 좋은 관계를 분석 가능!
  

* **Add & Layer Normalization**  
_Add_ : layer가 깊어지더라도 학습을 용이하게 만듦 (이전의 정보를 잊지 않는다.)  
_Layer Normalization_ : CNN에 비해 상대적으로 무거운 모델인 Transformer 구조에서 BN 방식보다 메모리 효율적으로 학습 가능