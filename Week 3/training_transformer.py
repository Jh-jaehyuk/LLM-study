# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Iterable, List
from data import fr_to_en
import utils
import pandas as pd
import json
import math

# Vocabulary 만들기
# 훈련 데이터 불러오기
# Fr -> En 번역을 위한 데이터셋(Multi-30k) 활용
fr_train = utils.open_text_set("data/training/train.fr")
en_train = utils.open_text_set("data/training/train.en")

# Vocab 만들기 / utils.py 참조
try:
    vocab_transform, token_transform = utils.make_vocab(fr_train, en_train)
except:
    # 오류 발생 시 spacy 설치 필요
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    spacy.cli.download("fr_core_news_sm")
    vocab_transform, token_transform = utils.make_vocab(fr_train, en_train)

# param
SRC_LANGUAGE = "fr"
TGT_LANGUAGE = "en"

# Transformer 모델 정의
class selfAttention(nn.Module):
    def __init__(self, embed_size, heads) -> None:
        """
        config.json 참고

        embed_size(=512) : embedding 차원
        heads(=8) : Attention 개수
        """
        super().__init__()
        self.embed_size = embed_size  # 512
        self.heads = heads  # 8
        self.head_dim = embed_size // heads  # 개별 attention의 embed_size(=64)

        # Query, Key, Value
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)  # 64 => 64
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)  # 64 => 64
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)  # 64 => 64

        # 8개 attention => 1개의 attention으로 생성
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)  # 8 * 64 => 512

    def forward(self, value, key, query, mask):
        """
        query, key, value size : (N, seq_len, embed_size)

        - N_batch = 문장 개수(=batch_size)
        - seq_len : 훈련 문장 내 최대 token 개수
        - embed_size : embedding 차원

        """

        N_batch = query.shape[0]  # 총 문장 개수
        value_len = value.shape[1]  # token 개수
        key_len = key.shape[1]  # token 개수
        query_len = query.shape[1]  # token 개수

        # n : batch_size(=128)
        # h : heads(=8)
        # value,key,query_len, : token_len
        # d_k : embed_size/h(=64)

        value = value.reshape(
            N_batch, self.heads, value_len, self.head_dim
        )  # (n, h, value_len, d_k)
        key = key.reshape(
            N_batch, self.heads, key_len, self.head_dim
        )  # (n x h x key_len x d_k)
        query = query.reshape(
            N_batch, self.heads, query_len, self.head_dim
        )  # (n x h x query_len x d_k)

        # Q,K,V 구하기
        V = self.value(value)
        K = self.key(key)
        Q = self.query(query)

        # score = Q dot K^T
        score = torch.matmul(Q, K.transpose(-2, -1))
        # query shape : (n, h, query_len, d_k)
        # transposed key shape : (n, h, d_k, key_len)
        # score shape : (n, h, query_len, key_len)

        if mask is not None:
            score = score.masked_fill(mask == 0, float("-1e20"))
            """
            mask = 0 인 경우 -inf(= -1e20) 대입
            softmax 계산시 -inf인 부분은 0이 됨.
            """

        # attention 정의

        # d_k로 나눈 뒤 => softmax
        d_k = self.embed_size ** (1 / 2)
        softmax_score = torch.softmax(score / d_k, dim=3)
        # softmax_score shape : (n, h, query_len, key_len)

        # softmax * Value => attention 통합을 위한 reshape
        out = torch.matmul(softmax_score, V).reshape(
            N_batch, query_len, self.heads * self.head_dim
        )
        # softmax_score shape : (n, h, query_len, key_len)
        # value shape : (n, h, value_len, d_k)
        # (key_len = value_len 이므로)
        # out shape : (n, h, query_len, d_k)
        # reshape out : (n, query_len, h, d_k)

        # concat all heads
        out = self.fc_out(out)
        # concat out : (n, query_len, embed_size)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        """
        config.json 참고

        embed_size(=512) : embedding 차원
        heads(=8) : Attention 개수
        dropout(=0.1): Node 학습 비율
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        """
        super().__init__()
        # Attention 정의
        self.attention = selfAttention(embed_size, heads)

        ### Norm & Feed Forward
        self.norm1 = nn.LayerNorm(embed_size)  # 512
        self.norm2 = nn.LayerNorm(embed_size)  # 512

        self.feed_forawrd = nn.Sequential(
            # 512 => 1024
            nn.Linear(embed_size, forward_expansion * embed_size),
            # ReLU 연산
            nn.ReLU(),
            # 1024 => 512
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        # self Attention
        attention = self.attention(value, key, query, mask)
        # Add & Normalization
        x = self.dropout(self.norm1(attention + query))
        # Feed_Forward
        forward = self.feed_forawrd(x)
        # Add & Normalization
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ) -> None:
        """
        config.json 참고

        src_vocab_size(=11509) : input vocab 개수
        embed_size(=512) : embedding 차원
        num_layers(=3) : Encoder Block 개수
        heads(=8) : Attention 개수
        device : cpu;
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        dropout(=0.1): Node 학습 비율
        max_length : batch 문장 내 최대 token 개수(src_token_len)
        """
        super().__init__()
        self.embed_size = embed_size
        self.device = device

        # input + positional_embeding
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)  # (11509, 512) 2

        # positional embedding
        pos_embed = torch.zeros(max_length, embed_size)  # (src_token_len, 512) 2
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)  # (1, src_token_len, 512) 3

        # Encoder Layer 구현
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        # dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _, seq_len = x.size()  # (n, src_token_len) 2
        # n : batch_size(=128)
        # src_token_len : batch 내 문장 중 최대 토큰 개수

        pos_embed = self.pos_embed[:, :seq_len, :]
        # (1, src_token_len, embed_size) 3

        out = self.dropout(self.word_embedding(x) + pos_embed)
        # (n, src_token_len, embed_size) 3

        for layer in self.layers:
            # Q,K,V,mask
            out = layer(out, out, out, mask)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion) -> None:
        """
        config.json 참고

        embed_size(=512) : embedding 차원
        heads(=8) : Attention 개수
        dropout(=0.1): Node 학습 비율
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = selfAttention(embed_size, heads=heads)
        self.encoder_block = EncoderBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_trg_mask, target_mask):
        """
        x : target input with_embedding (n, trg_token_len, embed_size) 3
        value, key : encoder_attention (n, src_token_len, embed_size) 3
        """

        # masked_attention
        attention = self.attention(x, x, x, target_mask)
        # (n, trg_token_len, embed_size) 3

        # add & Norm
        query = self.dropout(self.norm(attention + x))

        # encoder_decoder attention + feed_forward
        out = self.encoder_block(value, key, query, src_trg_mask)
        # (n, trg_token_len, embed_size) 3

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device,
    ) -> None:
        """
        config.json 참고

        trg_vocab_size(=10873) : input vocab 개수
        embed_size(=512) : embedding 차원
        num_layers(=3) : Encoder Block 개수
        heads(=8) : Attention 개수
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        dropout(=0.1): Node 학습 비율
        max_length : batch 문장 내 최대 token 개수
        device : cpu
        """
        super().__init__()
        self.device = device

        # 시작부분 구현(input + positional_embeding)
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)  # (10837,512) 2

        # positional embedding
        pos_embed = torch.zeros(max_length, embed_size)  # (trg_token_len, embed_size) 2
        pos_embed.requires_grad = False
        position = torch.arange(0, max_length).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size)
        )
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        self.pos_embed = pos_embed.unsqueeze(0).to(device)
        # (1, trg_token_len, embed_size) 3

        # Decoder Layer 구현
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_trg_mask, trg_mask):
        # n : batch_size(=128)
        # trg_token_len : batch 내 문장 중 최대 토큰 개수

        _, seq_len = x.size()
        # (n, trg_token_len)

        pos_embed = self.pos_embed[:, :seq_len, :]
        # (1, trg_token_len, embed_size) 3

        out = self.dropout(self.word_embedding(x) + pos_embed).to(self.device)
        # (n, trg_token_len, embed_size) 3

        for layer in self.layers:
            # Decoder Input, Encoder(K), Encoder(V) , src_trg_mask, trg_mask
            out = layer(out, enc_out, enc_out, src_trg_mask, trg_mask)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size,
        num_layers,
        forward_expansion,
        heads,
        dropout,
        device,
        max_length,
    ) -> None:
        """
        src_vocab_size(=11509) : source vocab 개수
        trg_vocab_size(=10873) : target vocab 개수
        src_pad_idx(=1) : source vocab의 <pad> idx
        trg_pad_idx(=1) : source vocab의 <pad> idx
        embed_size(=512) : embedding 차원
        num_layers(=3) : Encoder Block 개수
        forward_expansion(=2) : FFNN의 차원을 얼마나 늘릴 것인지 결정,
                                forward_expension * embed_size(2*512 = 1024)
        heads(=8) : Attention 개수
        dropout(=0.1): Node 학습 비율
        device : cpu
        max_length(=140) : batch 문장 내 최대 token 개수

        """
        super().__init__()
        self.Encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device,
        )
        self.Decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device,
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        # Probability Generlator
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)  # (512,10873) 2

    def encode(self, src):
        """
        Test 용도로 활용 encoder 기능
        """
        src_mask = self.make_pad_mask(src, src)
        return self.Encoder(src, src_mask)

    def decode(self, src, trg, enc_src):
        """
        Test 용도로 활용 decoder 기능
        """
        # decode
        src_trg_mask = self.make_pad_mask(trg, src)
        trg_mask = self.make_trg_mask(trg)
        out = self.Decoder(trg, enc_src, src_trg_mask, trg_mask)
        # Linear Layer
        out = self.fc_out(out)  # (n, decoder_query_len, trg_vocab_size) 3

        # Softmax
        out = F.log_softmax(out, dim=-1)
        return out

    def make_pad_mask(self, query, key):
        """
        Multi-head attention pad 함수
        """
        len_query, len_key = query.size(1), key.size(1)

        key = key.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size x 1 x 1 x src_token_len) 4

        key = key.repeat(1, 1, len_query, 1)
        # (batch_size x 1 x len_query x src_token_len) 4

        query = query.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # (batch_size x 1 x src_token_len x 1) 4

        query = query.repeat(1, 1, 1, len_key)
        # (batch_size x 1 x src_token_len x src_token_len) 4

        mask = key & query
        return mask

    def make_trg_mask(self, trg):
        """
        Masked Multi-head attention pad 함수
        """
        # trg = triangle
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src)
        # (n,1,src_token_len,src_token_len) 4

        trg_mask = self.make_trg_mask(trg)
        # (n,1,trg_token_len,trg_token_len) 4

        src_trg_mask = self.make_pad_mask(trg, src)
        # (n,1,trg_token_len,src_token_len) 4

        enc_src = self.Encoder(src, src_mask)
        # (n, src_token_len, embed_size) 3

        out = self.Decoder(trg, enc_src, src_trg_mask, trg_mask)
        # (n, trg_token_len, embed_size) 3

        # Linear Layer
        out = self.fc_out(out)  # embed_size => trg_vocab_size
        # (n, trg_token_len, trg_vocab_size) 3

        # Softmax
        out = F.log_softmax(out, dim=-1)
        return out

# 모델 불러오기
with open('config/transformer.json', 'r') as file:
    param = json.load(file)
    print("Model Parameter")
    print(param)
model = Transformer(**param)
device = param['device'] # gpu가 가능하면 gpu 이용

print("-" * 50)
print(f"현재 Device: {device}")

# Xavier Initializer
for param in model.parameters():
    if param.dim() > 1:
        nn.init.xavier_uniform_(param)

# loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# Training & Validation 설정
def collate_fn(batch_iter: Iterable):
    """
        Data_loader에서 불러온 데이터를 가공하는 함수
        토크나이징 => encoding => 시작 끝을 의미하는 spectial token(<bos>,<eos>) 추가 순으로 진행
        """
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = utils.sequential_transforms(
            token_transform[ln],  # 토크나이징
            vocab_transform[ln],  # encoding
            utils.tensor_transform,  # BOS/EOS를 추가하고 텐서를 생성
        )
        # sequential_transform, tensor_transform은 utils.py 참고

    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch_iter:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))

    # Pad 붙이기
    PAD_IDX = 1
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch.T, tgt_batch.T

# token을 단어로 바꾸기 위한 dict 생성, vocab의 key와 value 위치 변경
# 아래 helper 함수에서 활용됨.
decoder_en = {v:k for k,v in vocab_transform['en'].get_stoi().items()}
decoder_fr = {v:k for k,v in vocab_transform['fr'].get_stoi().items()}


def helper_what_sen(src, trg, logits, i, c=100, sen_num=0):
    '''
    문장이 제대로 학습되고 있는지를 확인하는 함수

    src = encoding 된 source_sentence
    trg = encoding 된 target_sentence
    logits = 모델 예측값
    i = 현재 batch 순서
    c = 결과를 보여주는 단계, ex) c = 100이면 100,200,300... 번째 batch에서 결과를 보여줌
    sen_num = batch 내 문장 중 몇 번째 문장을 추적할 것인지 설정
    '''
    if i % c == 0 and i != 0:
        src_sen = ' '.join([decoder_fr[i] for i in src.tolist()[sen_num] if decoder_fr[i][0] != '<'])
        trg_sen = ' '.join([decoder_en[i] for i in trg.tolist()[sen_num] if decoder_en[i][0] != '<'])
        prediction = logits.max(dim=-1, keepdim=False)[1][sen_num]
        prd_sen = ' '.join([decoder_en[i] for i in prediction.tolist() if decoder_en[i] != '<'])
        '''
        /*/* 모델의 예측 문장(prd_sen)을 구하는 방법 /*/* 

        n = batch size, trg_token_len = batch 내 문장의 최대 토큰 개수

        모델 output(=logits)은 (n, trg_token_len, trg_vocab_len)의 3차원 텐서임.

        1. 해당 텐서를 trg_vocab_len 차원의 기준으로 max를 하면 (n,trg_token_len)을 반환
        2. tensor.max()의 수행 결과는 [최댓값,idx]를 반환함.
        3. [1]을 넣어 idx를 선택, 그 결과는 (n, trg_token_len) 차원의 idx 반환
        4. 원하는 문장 순서(sen_num)을 선택한 뒤 정수를 다시 단어로 decoding 수행


        '''

        print('')
        print(f'{i}번째 batch에 있는 {sen_num}번째 문장 예측 결과 확인')
        print('src : ', src_sen)
        print('prd : ', prd_sen)
        print('trg : ', trg_sen)
        print('')

    return None


def train_epoch(model, optimizer):
    '''
    훈련 과정 설정
    '''
    # 모델 훈련 모드 변경
    model.train()
    losses = 0

    # training 데이터 불러오기(29,000개 문장)
    dataset = fr_to_en(set_type='training')

    # Data_loader
    batch_size = 128
    train_dataloader = DataLoader(dataset, batch_size, collate_fn=collate_fn)

    # 128개 문장을 1개의 batch로 총 227번 반복(128*227 => 29,000)

    for i, (src, tgt) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]  # 마지막 token은 <eos> 또는 <pad>이므로 불필요한 데이터 전처리

        logits = model(src, tgt_input)  # (n, trg_token_len, trg_vocab_len) 3

        # gradient를 0으로 초기화
        # 기본적으로 gradient는 더해지기 때문에,
        # 0으로 초기화하지 않으면 중복 계산이 됨
        optimizer.zero_grad()

        helper_what_sen(src, tgt_input, logits, i, 200, 0)  # 학습상태 확인

        tgt_output = tgt[:, 1:]  # 첫번째 token은 <bos>이므로 불필요한 데이터 전처리
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        # print(logits.reshape(-1, logits.shape[-1]))
        # print(tgt_output.reshape(-1))
        '''
        */*/*/*/ loss_fn(model_output,target_output) 설명 */*/*/*/*/*/

        model_output은 (n, trg_token_len, trg_vocab_len)의 3차원 텐서임 
        target_output은 (n,trg_token_len)의 2차원 텐서임
        이를 각각 2차원, 1차원으로 줄이면 
        model_output = (n*trg_token_len, trg_vocab_len)
        target_output = (n*trg_token_len) 

        이때 target_output은 개별 단어의 idx이므로 
        model_output의 idx로 활용하면 해당 단어의 확률을 찾을 수 있음.

        이러한 방법으로 정답인 단어와, 정답이 아닌 단어를 구분하여 Loss 계산 수행
        '''
        # 역전파 진행
        loss.backward()
        # 역전파 과정에서 계산된 gradient를 바탕으로 매개변수들을 조정
        optimizer.step()

        losses += loss.item()

    return losses / len(train_dataloader)

@torch.no_grad()
def evaluate(model):
    # 모델 평가모드
    model.eval()
    losses = 0

    # Load_Dataset
    dataset = fr_to_en(set_type='validation')

    # validation 데이터 불러오기
    batch_size = 128
    val_dataloader = DataLoader(dataset, batch_size, collate_fn=collate_fn)

    for i, (src, tgt) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]

        logits = model(src, tgt_input)

        helper_what_sen(src, tgt_input, logits, i, 5, 0)  # 학습상태 확인

        tgt_output = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))

        losses += loss.item()

    return losses / len(val_dataloader)


# 학습 및 평가 진행
from timeit import default_timer as timer
from tqdm import tqdm
from copy import deepcopy

NUM_EPOCHS = 15

prev_val_loss = 1e9
best_model_dict = model.state_dict()

for epoch in range(1, NUM_EPOCHS + 1):
    print('-' * 30)
    print(f'{epoch}번째 epoch 실행')
    start_time = timer()
    train_loss = train_epoch(model.to(device), optimizer)
    end_time = timer()
    val_loss = evaluate(model)

    print('----*' * 20)
    print((
              f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    print('----*' * 20)
    print('')

    if val_loss < prev_val_loss:
        prev_val_loss = val_loss
        best_model_dict = deepcopy(model.state_dict())

torch.save(best_model_dict, "model.pth")
