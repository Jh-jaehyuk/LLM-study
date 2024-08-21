from torch.utils.data import Dataset
import utils


class fr_to_en(Dataset):
    """
    pytorch dataloader 사용을 위한 class
    """

    def __init__(self, set_type):
        super().__init__()
        if set_type == "training":
            self.src_lang = utils.open_text_set("data/training/train.fr")
            self.trg_lang = utils.open_text_set("data/training/train.en")

            print('► Dataset is "training"')

        elif set_type == "validation":
            self.src_lang = utils.open_text_set("data/validation/val.fr")
            self.trg_lang = utils.open_text_set("data/validation/val.en")

            print('► Dataset is "validation"')

        else:
            raise ValueError('set_type must be "training" or "validation"')

    # Custom Dataset을 사용하기 위해 필요한 메서드
    # __len__, __getitem__
    # Custom Dataset을 dataset이라는 변수에 저장했다고 하면
    # __len__은 len(dataset)을 했을 때 호출되는 메서드
    # __getitem__은 dataset[i]을 했을 때 호출되는 메서드
    # 즉, __len__은 데이터셋의 크기
    # __getitem__은 데이터셋의 i번째 데이터에 접근할 수 있도록 하는 메서드라고 할 수 있음
    def __len__(self):
        return len(self.src_lang)

    def __getitem__(self, idx):
        return self.src_lang[idx], self.trg_lang[idx]
