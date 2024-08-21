import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_embedding import sentenceEmbedding

embedding_dim, embedded_input, decoder_embedded_input = sentenceEmbedding()
attention_dim = embedding_dim

# Query, Key, Value 가중치 행렬 초기화
W_q = torch.randn(embedding_dim, attention_dim)
W_k = torch.randn(embedding_dim, attention_dim)
W_v = torch.randn(embedding_dim, attention_dim)

"""
self-attention과 달리 일반적인 attention은 서로 다른 시퀀스의 관계성을 분석하는데 사용됨
여기서는 Encoder-Decoder 사이의 관계성을 분석한다고 가정
그렇기 때문에 self-attention과 다르게 Query와 Key, Value가 서로 다른 곳에서 생성됨

Attention: Q는 Decoder에서, K V는 Encoder에서 도출
Self-Attention: Q, K, V는 모두 동일한 Vector(Embedding Vector)에서 도출
"""
# 디코더 입력을 Query로 변환
Q = torch.matmul(decoder_embedded_input, W_q)

# 인코더 출력을 Key, Value로 변환
K = torch.matmul(embedded_input, W_k)
V = torch.matmul(embedded_input, W_v)

# Attention Score 계산 (Q와 K의 내적, dot product)
attention_scores = torch.matmul(Q, K.transpose(-2, -1))

# Attention Score에 대해 Softmax 적용
attention_weights = F.softmax(attention_scores, dim=-1)

# Attention 가중치를 V에 곱해 최종 출력 계산
output = torch.matmul(attention_weights, V)

print(output)