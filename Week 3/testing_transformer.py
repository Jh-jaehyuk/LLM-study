# Import Modules
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Iterable, List
from model import Transformer
from data import fr_to_en
import utils
import torch.nn as nn
import pandas as pd
import json
import torch

# Tokenizing에 활용할 Vocabulary 생성
# 훈련 데이터 불러오기
# Fr -> En 번역을 위한 데이터셋(Multi-30k) 활용
fr_data = utils.open_text_set("data/training/train.fr")
eng_data = utils.open_text_set("data/training/train.en")

# Vocab 만들기 / 관련 함수는 utils.py 참조
try :
  vocab_transform, token_transform = utils.make_vocab(fr_data, eng_data)
except :
  # 오류 발생 시 spacy 설치 필요
  # Download spacy tokenizer(en, fr)
  import spacy.cli
  spacy.cli.download("en_core_web_sm")
  spacy.cli.download("fr_core_news_sm")
  vocab_transform, token_transform = utils.make_vocab(fr_data, eng_data)

# parameters
SRC_LANGUAGE = "fr"
TGT_LANGUAGE = "en"

# training_transformer에서 학습한 모델 불러오기
# 현재 device="mps"(Mac) 이므로
# cpu로 바꾸거나 gpu로 변경 필요함
with open('config/transformer.json', 'r') as file:
    param = json.load(file)
    print('Model_Parameters')
    print('-'*50)
    print(param)

# multi-30k 데이터를 15번 epoch한 모델 불러오기
import os
model = Transformer(**param)
model_path = os.path.join(os.getcwd(), "models", "model.pth")
model.load_state_dict(torch.load(model_path))

device = model.device
print("device:", model.device)
model.to(device)

print('-'*50)
print(f'현재 devicde 설정값은 : "{model.device}" 입니다. 변경을 희망하실 경우 config/transformer.json을 수정해주세요.')
print('-'*50)

# loss_fn
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)

# token을 단어로 바꾸기 위한 dict 생성, vocab의 key와 value 위치 변경
# 아래 helper 함수에서 활용됨.
decoder_en = {v: k for k, v in vocab_transform['en'].get_stoi().items()}
decoder_fr = {v: k for k, v in vocab_transform['fr'].get_stoi().items()}

# Test model
def tokenizing_src(input_data: str):
    # input_data_tokenizing
    token_data = token_transform['fr'](input_data)
    vocab_src = vocab_transform['fr'](token_data)
    tokenized_src = [2] + vocab_src + [3]
    return tokenized_src


def select_random_item():
    num = torch.randint(1, 29000, (1,)).item()

    return fr_data[num], eng_data[num]

@torch.no_grad()
def test(model):
    '''
    * validation은 문제와 정답이 모두 주어진다면 test는 문제만 제공하는 상황임.

    * test 함수를 통해 Transformer의 실제 문제 예측 과정을 이해할 수 있음.

    * Transformer는 문제와 정답이 있다면 답을 구하는 과정을 병렬적으로 수행할 수 있음.

    * 하지만 테스트에서는 정답이 주어지지 않으므로 한 번의 하나의 토큰을 생산함.

    * < bos > token을 시작으로 다음 토큰을 예상하며 < eos > 토큰이 생성될때까지 반복적으로 예측을 수행하게 되는 알고리즘이 필요함.

    * 아래의 test 함수를 다뤄보면서 Transformer의 데이터 처리 과정을 이해할 수 있음.

    '''
    # 모델을 평가 모드로 설정
    # 평가 중 사용되지 않는 기능들을 비활성화(Dropout 등) -> 예측의 일관성 유지
    model.eval()

    # 임의의 훈련 데이터 선별
    fr_item, en_item = select_random_item()

    print('입력 :', fr_item)

    # Input Data 토크나이징
    # 입력 데이터를 토큰화하여 정수 인덱스의 리스트로 변환
    tokenized_input = tokenizing_src(fr_item)
    # 최대 길이를 토큰화된 입력 길이의 1.2배로 설정 -> 예측 문장의 최대 길이
    max_length = int(len(tokenized_input) * 1.2)

    # src Tensor에 Token 저장
    # 입력 데이터를 텐서로 변환하고 배치 차원(unsqueeze(0))을 추가한 후 device로 전송
    src = torch.LongTensor(tokenized_input).unsqueeze(0).to(device)

    # trg Tensor 생성(1, max_length)
    # 모든 값이 0으로 초기화된 출력용 텐서 생성
    trg = torch.zeros(1, max_length).type_as(src.data).to(device)

    # src encoding
    # 입력 데이터를 encoder에 통과시켜 인코딩된 출력을 생성
    enc_src = model.encode(src).to(device)

    next_trg = 2  # 문장 시작 <bos> idx

    # 문장 예측 시작
    for i in range(0, max_length):
        trg[0][i] = next_trg  # 생성된 토큰을 출력 텐서에 저장

        # Decoder를 통해 다음 토큰에 대한 확률 계산
        logits = model.decode(src, trg, enc_src)

        # 가장 높은 확률을 가진 토큰의 인덱스를 선택
        prd = logits.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prd.data[i]  # 현재 위치에 해당하는 토큰 인덱스를 가져옴
        next_trg = next_word.item() # 다음 토큰으로 설정
        if next_trg == 3:
            # <eos> 나오면 종료
            trg[0][i] = next_trg
            break

    # <pad> 토큰을 제거하고 최종 번역된 문장 추출
    if 3 in trg[0]:
        eos_idx = int(torch.where(trg[0] == 3)[0][0])
        trg = trg[0][:eos_idx].unsqueeze(0)
    else:
        pass

    # 번역된 문장 생성
    # 정수 인덱스를 실제 단어로 변환
    translation = [decoder_en[i] for i in trg.squeeze(0).tolist()]
    print('모델예측 :', ' '.join(translation[1:]))

    print('정답 :', en_item)
    print('')
    print('주의! 29,000개의 제한된 데이터로 학습을 수행했으므로 완벽한 예측이 불가능함.')


test(model)