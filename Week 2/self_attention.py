import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_embedding import sentenceEmbedding

embedding_dim, embedded_inputs, _ = sentenceEmbedding()
"""
attention_dim은 embedding_dim과 동일하게 설정하여 사용할 수 있음
하지만 Transformer 같이 Multi-head를 사용하는 경우엔,
embedding_dim을 multi-head의 개수로 나누어 사용하는 것이 일반적임
이번 예시에서는 multi-head 구조가 아니기 때문에 아래와 같이
attention_dim = embedding_dim으로 사용함
"""
attention_dim = embedding_dim

# Query, Key, Value 가중치 행렬 초기화
W_q = torch.randn(embedding_dim, attention_dim)
W_k = torch.randn(embedding_dim, attention_dim)
W_v = torch.randn(embedding_dim, attention_dim)

# Q, K, V 계산
"""
embedded_inputs shape = (2, max_seq_len, embedding_dim) = (2, 4, 8)
W_q, W_k, W_v shape = (embedding_dim, attention_dim) = (8, 8)
두 개의 곱 shape = (2, 4, 8) * (8, 8)
Batched matrix multiple에 따라서
embedding_inputs의 batch_size인 2를 제외한
(4, 8) * (8, 8)이 계산되어 (4, 8)이 되고 앞에 batch_size를 붙여서 (2, 4, 8)이 됨
"""
Q = torch.matmul(embedded_inputs, W_q)
# print("embedded_inputs shape:", embedded_inputs.shape)
# print("W_q shape:", W_q.shape)
# print("Q shape:", Q.shape)
K = torch.matmul(embedded_inputs, W_k)
V = torch.matmul(embedded_inputs, W_v)

# Attention Score 계산 (Q와 K의 내적, dot product)
"""
K.transpose(-2, -1)을 하는 이유는 위와 같이 Batched matrix multiple이 가능하도록 하기 위함
Q shape = (2, 4, 8)
K shape = (2, 4, 8)
두 개의 곱이 가능하기 위해선 각 텐서의 batch_size를 제외한 부분이 (4, 8)과 (8, 4)이어야 함
K의 -1과 -2 인덱스가 바뀌면 (4, 8) -> (8, 4)가 될 것
따라서, K.transpose(-2, -1)을 해줌
"""
attention_scores = torch.matmul(Q, K.transpose(-2, -1))

# Attention Score에 대해 Softmax 적용
"""
Softmax에서 dim=-1을 해주는 이유는 입력 데이터의 차원 구조와 상관없이 일관되게 동작하도록 하기 위함
다시 말하자면, 입력 데이터의 차원과 관계없이 무조건 마지막 차원에 대해서만 Softmax를 적용하도록 한다
"""
attention_weights = F.softmax(attention_scores, dim=-1)

# Attention 가중치를 V에 곱해 최종 출력 계산
output = torch.matmul(attention_weights, V)

print("output:", output)

