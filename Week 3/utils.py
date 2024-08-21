import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import torch


def open_text_set(dir: str) -> list:
    """
    train, validation 파일 오픈 용도

    dir : 파일경로
    """

    with open(dir, "r") as f:
        f = f.readlines()
        f = [v.rstrip() for v in f]

    return f


def make_vocab(
    src_lang: Iterable,
    trg_lang: Iterable,
) -> list:
    """
    Word embedding을 위한 vocab 생성(sub_word embedding X)
    src_lang : input 언어
    trg_lang : output 언어

    """

    token_transform = {} # 각 언어별 tokenizer를 저장할 dictionary
    vocab_transform = {} # 각 언어별 단어 집합(vocabulary)를 저장할 dictionary

    # 입력 언어(SRC_LANGUAGE)와 출력 언어(TRG_LANGUAGE) 설정
    SRC_LANGUAGE, TRG_LANGUAGE = ["fr", "en"]
    # spacy 모듈을 사용하여 각 언어별 tokenizer를 load
    token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="fr_core_news_sm")
    token_transform[TRG_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")

    # token 생성을 위한 iterator 생성
    # 주어진 언어에 대해 텍스트 데이터를 토큰으로 변환하는 generator 함수 정의
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:

        for data_sample in data_iter:
            # 데이터 샘플을 해당 언어의 tokenizer로 토큰화하여 반환
            yield token_transform[language](data_sample)  # memory 초과 방지용으로 yield 사용

    # special tokens
    # UNK: 단어 집합에 포함되지 않은 단어
    # PAD: 문장 길이를 맞추기 위해 사용
    # BOS: 문장의 시작을 나타냄
    # EOS: 문장의 끝을 나타냄
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    # src_lang과 trg_lang에 대해 각 언어별 단어 집합(vocabulary) 생성
    for train_iter, ln in [(src_lang, SRC_LANGUAGE), (trg_lang, TRG_LANGUAGE)]:

        # 언어별 vocab 생성
        vocab_transform[ln] = build_vocab_from_iterator(
            yield_tokens(train_iter, ln), # 주어진 언어의 텍스트 데이터를 토큰화하여 vocabulary 생성
            min_freq=1, # vocabulary에 포함되기 위한 최소 빈도 수
            specials=special_symbols, # 특수 토큰 추가하기
            special_first=True, # 특수 토큰을 vocabulary의 앞부분에 배치함
        )

    # OOV(Out Of Vocabulary)가 존재하면 <unk>을 반환하도록 default idx 설정
    # default idx 를 설정해야 OOV 문제에서 에러가 발생하지 않음
    for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    # vocab, tokenizer 저장
    # 최종적으로 각 언어의 단어 집합과 tokenizer를 리스트로 반환
    return [vocab_transform, token_transform]


def sequential_transforms(*transforms):
    """
    *transform에 포함된 함수를 연속적으로 수행하게 하는 메서드.
    """

    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids: List[int]):
    """
    토크나이징 한 문장에 시작과 끝을 의미하는 Special Token(<bos>, <eos>) 추가
    bos_idx = 2, eos_idx = 3
    """
    return torch.cat((torch.tensor([2]), torch.tensor(token_ids), torch.tensor([3])))
