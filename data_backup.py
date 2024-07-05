import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from arguments import DataArguments


class Inbatch_Dataset(Dataset):
    def __init__(
            self,
            args: DataArguments,
            file_name:str,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query

        passages = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        passages.extend(negs)

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        name = self.dataset[item]["dataset_name"]
        return [name, query, passages]


@dataclass
class Inbatch_Collator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        name = [f[0] for f in features]
        query = [f[1] for f in features]
        passage = [f[2] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"batch": {"data_name": name, "txt1": q_collated, "txt2": d_collated}}


class Pair_Dataset(Dataset):
    def __init__(
            self,
            args: DataArguments,
            file_name:str,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, str]:
        txt1 = self.dataset[item]['txt1']
        txt2 = self.dataset[item]['txt2']
        label = self.dataset[item]['label']
        name = self.dataset[item]["dataset_name"]
        return [name, txt1, txt2, label]

@dataclass
class Pair_Collator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        name = [f[0] for f in features]
        txt1 = [f[1] for f in features]
        txt2 = [f[2] for f in features]
        label = [f[3] for f in features]

        neg_pos_idxs = [[0.0] * len(features) for _ in range(len(features))]
        for i in range(len(features)):
            for j in range(len(features)):
                if label[i] == 1 and label[j] == 0:
                    neg_pos_idxs[i][j] = 1.0
                elif label[i] == 0 and label[j] == 1:
                    neg_pos_idxs[i][j] = 1.0

        txt1_collated = self.tokenizer(
            txt1,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        txt2_collated = self.tokenizer(
            txt1,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"data_name": name, "txt1": txt1_collated, "txt2": txt2_collated, "idx": neg_pos_idxs}


def comb_data_loader(loaders, idx_list=None):
    if idx_list is None:
        idx_list = list(range(len(loaders)))
    loaders_iter = [iter(item) for item in loaders]
    idx_for_idx = 0
    while True:
        loader_idx = idx_list[idx_for_idx]
        try:
            yield next(loaders_iter[loader_idx])
        except StopIteration:
            loaders_iter[loader_idx] = iter(loaders[loader_idx])
            yield next(loaders_iter[loader_idx])
        idx_for_idx += 1
        if idx_for_idx % len(idx_list) == 0:
            random.shuffle(idx_list)
            idx_for_idx = 0


class VecDataSet(Dataset):
    """ pair 对数据集 """

    def __init__(self, data_loaders):
        self.lens = sum([len(i) for i in data_loaders])
        self.data = comb_data_loader(data_loaders)

    def __len__(self):
        return self.lens

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        return next(self.data)
