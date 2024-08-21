import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.backends.mps.is_available():
    torch.set_default_device('mps')

# 하이퍼파라미터 설정
embedding_dim = 128
num_heads = 8 # Attention head 개수
attention_dim = embedding_dim // num_heads
seq_len = 10 # 시퀀스 길이
vocab_size = 1000

# 임베딩 레이어 정의
embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Positional Encoding 클래스
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

        # Positional Encoding을 계산할 행렬 초기화
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # unsqueeze(1)을 하는 이유
        # 1번째 차원에 크기가 1인 차원을 추가하여
        # div_term과 곱할 때, 차원 불일치 없이 연산을 하기 위함
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        # Positional Encoding 계산
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # unsqueeze(0)을 하는 이유
        # 0번째 차원에 크기가 1인 차원을 추가하여
        # (0번째 차원에 차원을 추가하는 것은 배치 크기 차원을 추가하는 것과 동일)
        # 나중에 pe와 입력 텐서 'x'를 더할 때, 각 배치에 동일한 PE를 적용하기 위함
        pe = pe.unsqueeze(0)

        # register_buffer란?
        # register_buffer로 등록된 텐서는 모델의 파라미터로 간주되지만,
        # 학습 중에 업데이트되지 않음.
        # 값이 학습되지않고 고정된 텐서로 남아 있어야 할 때 유용함.
        # 장 점
        # model.state_dict()에 저장되어 모델을 저장하거나 불러올 때 같이 저장 및 로드됨
        # 모델을 GPU나 CPU로 이동할 때 자동으로 함께 이동함. 모델과 항상 동일한 디바이스에 있음이 보장됨
        self.register_buffer('pe', pe)


    def forward(self, x):
        # 입력 x에 Positional Encoding을 더해줌
        # x.size(1)을 사용하는 이유
        # PE는 최대 max_len까지 준비되어 있지만, 실제 입력 시퀀스의 길이는 다를 수 있기 때문
        # 여기서 x.size(1)은 시퀀스의 길이를 의미함
        # 예를 들어 x가 [batch_size, seq_len, embedding_dim] 크기를 가진다면,
        # x.size(1)은 seq_len을 반환하는 것
        # 결과적으로 self.pe[:, :x.size(1), :]는
        # PE의 [batch_size, seq_len, embedding_dim] 크기를 가진 텐서를 반환하게됨
        x = x + self.pe[:, :x.size(1), :]
        return x

# Multi-head Self-Attention 구현
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = embedding_dim // num_heads

        # Query, Key, Value를 위한 가중치 행렬
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)

        # Multi-head attention output을 위한 가중치 행렬
        self.W_o = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size = x.size(0)

        """
        # Q, K, V 계산 및 분할
        
        입력 텐서를 어텐션 헤드 수에 따라 분할하는 과정
        1. self.W_q(x): 입력 텐서 x에 대해 선형 변환을 적용한 후, Q 텐서를 생성함
           이 때, Q 텐서의 크기는 (batch_size, seq_len, embedding_dim)
        2. view를 사용하여 텐서를 (batch_size, num_heads, seq_len, attention_dim) 으로 변환함
        
        이렇게 하면 뭐가 좋음?
        1. 독립적인 어텐션 계산: 각 헤드는 입력 시퀀스를 독립적으로 처리하고, 
           서로 다른 패턴을 학습할 수 있음. 이렇게 하면 모델이 다양한 어텐션 패턴을 학습하여
           풍부한 표현을 얻을 수 있음.
        2. 병렬 처리: 여러 헤드에서 병렬로 어텐션을 계산할 수 있어, 계산 효율성이 높아짐
        3. 고차원 표현 학습: 여러 헤드의 어텐션 출력을 결합함으로써, 
           다양한 관점에서 입력 데이터를 인식하고 이를 반영할 수 있음. 이는 모델이 보다 풍부하고
           복잡한 표현을 학습할 수 있게 함.
        """
        # view의 세번째 파라미터로 -1을 사용하는 이유
        # 동적 크기 조정(dynamic size adjustment)을 위해서
        # 세번째 차원(2번째 인덱스)은 시퀀스의 길이를 나타내는데, 입력 시퀀스 길이에 따라 달라질 수 있음
        # 매번 달라지는 입력값을 고정해주기 보단 -1을 주어, PyTorch가 해당 차원의 크기를 계산하도록 함
        # 사실 우리는 위에서 seq_len이라는 변수를 저장하여 사용하고 있으므로
        # -1 대신 seq_len을 사용해도 무방함
        Q = self.W_q(x).view(batch_size, self.num_heads, -1, self.attention_dim)
        K = self.W_k(x).view(batch_size, self.num_heads, -1, self.attention_dim)
        V = self.W_v(x).view(batch_size, self.num_heads, -1, self.attention_dim)

        # 각 헤드에서의 Attention score 계산
        """
        torch.einsum
        Einstein Summation Convention을 사용하여 텐서 연산을 수행하는 도구
        
        아래와 같은 코드에서 
        b: 배치 크기
        h: 어텐션 헤드 수
        q: 쿼리 시퀀스 길이
        k: 키 시퀀스 길이
        d: 어텐션 차원
        을 의미함
        
        bhqd, bhkd -> bhqk의 의미
        b, h는 두 텐서에서 동일한 위치에 있으므로, 이 차원들을 그대로 유지됨
        d는 두 텐서에서 마지막 차원이며, 이 차원을 따라 내적이 계산됨
        계산된 텐서는 q와 k 차원을 가지게 됨
        
        각 배치(b)와 각 헤드(h)에 대해
        Q의 각 벡터(q차원)와 K의 각 벡터(k차원)간의 내적을 계산함
        => Q * K^T와 동일함
        
        만약 einsum을 사용하는 과정이 이해되지 않는다면 아래와 같이 수정하여 사용가능함
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        """
        attention_scores = torch.einsum("bhqd, bhkd -> bhqk", Q, K) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 어텐션 가중치를 V에 곱하기
        """
        나머지는 위와 동일하며 추가된 것은
        k: 키 시퀀스의 길이(일반적으로 v와 동일)
        v: 값(value) 시퀀스의 길이
        d: 값 벡터의 차원
        
        b, h는 두 텐서에서 동일한 위치에 있으므로, 이 차원들은 그대로 유지됨
        k, v는 서로 대응되기 때문에 이 차원을 따라 합산이 이루어짐
        계산된 텐서는 q와 d 차원을 가짐
        
        각 배치(b)와 각 헤드(h)에 대해 attention_weights의 각 행(q차원)과 
        V의 각 열(d차원)간의 가중치 합을 계산함
        => attention_weights * V와 동일함
        
        위와 마찬가지로 만약에 einsum을 사용하는 과정이 이해되지 않는다면 아래와 같이 수정하여 사용가능함
        attention_output = torch.matmul(attention_weights, V)
        """
        attention_output = torch.einsum("bhqk, bhvd -> bhqd", attention_weights, V)

        # 헤드들을 연결하여 최종 출력 생성
        # contiguous()를 사용하는 이유
        # 텐서를 다룰 때, 특히 transpose나 permute와 같은 연산을 수행한 후에는
        # (transpose는 두 개 차원을 서로 바꾸는 것)
        # (permute는 기존 텐서의 차원 전체를 내가 원하는 순서로 바꾸는 것)
        # 텐서의 메모리 레이아웃이 바뀔 수 있음.
        # 이 경우 텐서의 메모리상에서 데이터가 연속적이지 않을 수 있으며,
        # 이는 view()와 같은 함수에서 기대하는 메모리 레이아웃과 다를 수 있음.
        # 연속적이지 않은 텐서에 대해 view()를 호출하면 오류가 발생할 수 있기 때문.
        # contiguous()를 호출함으로써 텐서의 데이터를 새로운 메모리 블록에 복사하여
        # 연속적인 메모리 레이아웃을 갖도록 만듦.
        # view()를 호출하면 왜 오류가 발생함?
        # view()는 메모리에서 연속된 데이터를 사용하여 텐서의 크기를 재구성하기 때문.
        # 결과적으로 contiguous()를 사용하는 것은 이후에 view() 연산이 안정적으로 작동하게 하기 위함.
        attention_output = attention_output.contiguous().view(batch_size, -1, self.num_heads * self.attention_dim)

        # Linear Transformation을 거쳐 출력
        output = self.W_o(attention_output)

        return output

# Add & Layer Normalization 구현
class AddNorm(nn.Module):
    def __init__(self, embedding_dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

# Multi-head self-attention 및 add & layer normalization 정의
self_attention = MultiHeadSelfAttention(embedding_dim, num_heads)
add_norm = AddNorm(embedding_dim)

# 임의의 입력 생성 (batch_size = 2, seq_len = 10)
random_input = torch.randint(0, vocab_size, (2, seq_len))

# Embedding vector 생성
embedded = embedding(random_input)

# Positional Encoding 정의
positional_encoding = PositionalEncoding(embedding_dim, max_len=seq_len)

# Positional Encoding 추가
encoded_input = positional_encoding(embedded)

# Multi-head self-attention 적용
attention_output = self_attention(encoded_input)

# Add & layer normalization 적용
final_output = add_norm(encoded_input, attention_output)

print(final_output)
