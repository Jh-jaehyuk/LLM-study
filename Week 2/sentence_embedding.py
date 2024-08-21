import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

if torch.backends.mps.is_available():
    torch.set_default_device('mps')

def sentenceEmbedding():
    # 임의의 단어 사전 (토큰화된 단어에 대한 인덱스)
    vocab = {'I': 0, 'am': 1, 'a': 2, 'student': 3, 'learning': 4, 'about': 5, 'attention': 6}
    vocab_size = len(vocab)

    # 임베딩 차원 설정
    embedding_dim = 8

    # 임베딩 레이어 정의
    embedding = nn.Embedding(vocab_size, embedding_dim)

    # 임의의 문장을 토큰 인덱스로 변환
    sentence1 = ['I', 'am', 'a', 'student']  # 길이 4
    sentence2 = ['learning', 'about', 'attention']  # 길이 3

    # 인덱스로 변환
    input_ids_1 = torch.tensor([vocab[word] for word in sentence1])
    input_ids_2 = torch.tensor([vocab[word] for word in sentence2])
    # print("input_ids_1:", input_ids_1)
    # print("input_ids_2:", input_ids_2)

    # 패딩을 사용하여 동일한 길이로 맞추기
    # 길이가 안맞을 경우, 길이가 더 짧은 벡터의 뒷부분에 padding_value를 배치함
    # 앞부분에 배치하고 싶다면 batch_first=False로 설정
    padded_inputs = pad_sequence([input_ids_1, input_ids_2], batch_first=True, padding_value=0)  # shape: (2, max_seq_len)
    # print("padded_inputs:", padded_inputs)

    """
    만약 위처럼 벡터의 길이가 다르지 않고 동일하다면,
    아래와 같이 torch.stack을 사용하는 것도 가능함
    input_ids = torch.stack([input_ids_1, input_ids_2])
    """

    # 임베딩 통과하여 임베딩 벡터 생성 (batch_size, max_seq_len, embedding_dim)
    # 여기서 batch_size는 pad_sequence로 쌓은 벡터의 길이 len(sequences)
    embedded_inputs = embedding(padded_inputs)
    # print("embedded_inputs:", embedded_inputs)

    # 디코더 입력으로 사용할 다른 문장을 임베딩
    decoder_sentence = ['I', 'am', 'learning']
    decoder_input_ids = torch.tensor([vocab[word] for word in decoder_sentence])

    # 디코더 임베딩 벡터 생성 및 패딩
    decoder_embedded_input = embedding(decoder_input_ids).unsqueeze(0)

    return embedding_dim, embedded_inputs, decoder_embedded_input
