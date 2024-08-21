import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.backends.mps.is_available():
    torch.set_default_device('mps')

device = torch.device('mps')


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

    # Encoder와 Decoder에서 둘 다 사용가능하도록 파라미터를 수정함
    # Decoder에서는 Encoder-Decoder Attention 또한 필요함
    # 기존 Encoder에서는 Q, K, V가 모두 embedding vector로부터 생성되기 때문에
    # query, key, value를 구분하지 않고 x 하나로 퉁쳐서 입력할 수 있었음
    # 하지만, self-attention이 아니라 일반적인 attention(이번에는 Encoder-Decoder Attention)의 경우엔
    # Q는 Decoder의 입력으로부터,
    # K, V는 Encoder의 출력으로부터 생성되기 때문에
    # query와 key, value를 구분하여 넘겨줄 필요가 있었음
    # 사실 클래스 이름이 SelfAttention이라서 분리시키는 것이 더 좋았겠지만
    # 클래스가 너무 많아져서 그냥 재활용하기로 한 것..
    def forward(self, query, key, value, mask=None):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 여기서 Q.size(0), K.size(0), V.size(0)와 같은 부분은
        # 이전 스터디에서 batch_size 변수와 같음
        Q = Q.view(Q.size(0), -1, num_heads, attention_dim).transpose(1, 2)
        K = K.view(K.size(0), -1, num_heads, attention_dim).transpose(1, 2)
        V = V.view(V.size(0), -1, num_heads, attention_dim).transpose(1, 2)

        attention_scores = torch.einsum("bhqd, bhkd -> bhqk", Q, K) / (self.attention_dim ** 0.5)

        # Decoder에서 Masked Self-Attention할 때 다시 설명할 것
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.einsum("bhqk, bhvd -> bhqd", attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(query.size(0), -1, self.num_heads * self.attention_dim)

        return attention_output


# Add & Layer Normalization 구현
class AddNorm(nn.Module):
    def __init__(self, embedding_dim):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)


# FFN은 두개의 선형변환 레이어와 한개의 활성 함수 레이어로 구성됨
# 결과적으로 FFN을 통과하면 비선형성이 추가되는 것
class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 논문 내용에서는
        # return self.fc2(self.relu(self.fc1(x)))로 표현함
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        return output


class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MaskedMultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = embedding_dim // num_heads

        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
        self.W_o = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # torch.triu는 삼각행렬을 만듦
        # 정확히는 상삼각행렬(대각성분 기준으로 윗부분이 남은 행렬)
        # 하삼각행렬을 만들 때는 torch.tril을 사용함
        # 그리고 대각행렬을 살릴지 말지 선택하는 옵션이 diagonal
        # diagonal=1을 하면 대각행렬의 값이 그대로 남고
        # diagonal=-1을 하면 대각행렬까지 포함해서 0으로 만든다
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool().transpose(0, 1)
        # 위의 식을
        # torch.tril(torch.ones((seq_len, seq_len), diagonal=1).bool()
        # 이렇게 작성해도 동일함

        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.attention_dim).transpose(1, 2)
        # 위처럼 transpose를 하게 되면
        # Encoder에서 설명한 것처럼
        # shape가 (batch_size, self.num_head, seq_len, self.attention_dim)이 됨

        attention_scores = torch.einsum("bhqd, bhkd -> bhqk", Q, K) / self.attention_dim ** 0.5

        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        # 위에서 계산된 self attention 결과에 마스킹처리를 해서 미래의 값을 참조하지 못하도록 함
        # mask 행렬의 값이 0인 칸을 -inf로 만들어서 softmax 계산한 값이 작아지도록 만듦
        # 결과적으로 선택되지 않도록 하는 것

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.einsum("bhqk, bhvd -> bhqd", attention_weights, V)

        attention_output = attention_output.contiguous().view(batch_size, -1, self.attention_dim * self.num_heads)
        attention_output = self.W_o(attention_output)
        return attention_output


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForwardNetwork(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head Self-Attention의 출력과 Residual 연결
        # 그 후 Add & Norm
        self_attention_output = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(self_attention_output))

        # FFN과 Residual 연결
        # 그 후 Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, num_layers, vocab_size, max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # Embedding Vector 생성
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Embedding Vector에 Positional Encoding을 더해줌
        self.pe = PositionalEncoding(embedding_dim, max_len)
        # EncoderBlock을 여러 개(=num_layers만큼) 쌓는 것(자료 그림 참조)
        # nn.ModuleList는 iterable로 동작하도록 할 수 있음
        # 또한, 인덱스로 접근하는 것도 가능하게 함
        self.layers = nn.ModuleList(
            [EncoderBlock(embedding_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        super(DecoderBlock, self).__init__()
        # Masked Self-Attention Layer 정의
        self.masked_self_attention = MaskedMultiHeadSelfAttention(embedding_dim, num_heads)
        # Decoder는 Encoder와 연결되기 때문에
        # Encoder-Decoder Attention이 계산되어야 함
        # 아래의 mask는 위에서 torch.triu로 작성했던 것을 torch.tril로 바꿔본 것
        self.mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.encoder_self_attention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForwardNetwork(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        masked_self_attention_output = self.masked_self_attention(x)
        x = self.norm1(x + self.dropout(masked_self_attention_output))

        # query는 Decoder의 embedding vector로부터,
        # key, value는 Encoder의 출력값으로부터 생성됨
        encoder_attention_output = self.encoder_self_attention(x, encoder_output, encoder_output, mask=self.mask)
        x = self.norm2(x + self.dropout(encoder_attention_output))

        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, num_layers, vocab_size, max_len=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, max_len)
        self.layers = nn.ModuleList(
            [DecoderBlock(embedding_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output)

        output = self.fc_out(x)
        return output


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, embedding_dim, num_heads, hidden_dim, num_layers,
                 max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(embedding_dim, num_heads, hidden_dim, num_layers, input_vocab_size, max_len,
                                          dropout)
        self.decoder = TransformerDecoder(embedding_dim, num_heads, hidden_dim, num_layers, target_vocab_size, max_len,
                                          dropout)

    def forward(self, x, target):
        encoder_output = self.encoder(x)
        output = self.decoder(target, encoder_output)

        return output


# 하이퍼파라미터 설정
embedding_dim = 128
num_heads = 8  # Attention head 개수
attention_dim = embedding_dim // num_heads
seq_len = 10  # 시퀀스 길이
vocab_size = 1000 # 입력 vocabulary 크기
hidden_dim = 512
target_vocab_size = 1000 # 타겟 vocabulary 크기
batch_size = 32
num_layers = 6 # Encoder & Decoder Block을 얼마나 쌓을지

# Transformer 모델 정의
transformer = Transformer(vocab_size, target_vocab_size, embedding_dim, num_heads, hidden_dim, num_layers)

# 임의의 입력 시퀀스 생성
input_sequence = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
# 임의의 타겟 시퀀스 생성
target_sequence = torch.randint(0, target_vocab_size, (batch_size, seq_len)).to(device)

output = transformer(input_sequence, target_sequence)

print(output.shape) #