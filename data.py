import math
import os.path
import random
import json
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
from typing import List, Union

import datasets
from torch.utils.data import Dataset, RandomSampler
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
from arguments import DataArguments

@dataclass
class Inbatch:
    query:str
    passage:str
# (slots=True)
@dataclass
class Pair:
    txt1:str
    txt2:str
    label:list

#留给其他类型数据


@dataclass
class TaskBatchIndex:
    name:str
    batch_index:list

@dataclass
class DatawithInfo:
    dataset: list
    name: str

def load_data(file_path) -> list:
    data = []
    with open(file_path,'r', encoding='utf-8') as file:
        for line in tqdm(file,desc=f"loading {file_path}"):
            sample = json.loads(line)
            data.append(sample)
    return data
    
def load_all_datasets(root_dir):
    all_datasets = []
    for filename in os.listdir(root_dir):
        if filename.endswith("json"):
            file_path = os.path.join(root_dir,filename)
            sub_dataset = load_data(file_path)
            all_datasets.append(DatawithInfo(sub_dataset, filename))
            print(f"{filename} nas been loaded succesfully!")
    return all_datasets
                    


class UniDataset(Dataset):
    """
    dataset的组织形式
    都需要有type
    Inbatch：query，pos，neg
    Pair：txt1，txt2，label
    neg为列表格式，其余元素全部为字符串
    """
    def __init__(
            self,
        datasets:list,
        batch_size: int = 30,
        max_samples: Union[int, None] = None,
    ):     
        self.batch_size = batch_size
        self.datasets = datasets
        self.max_samples = max_samples
        self.name_dataset_map = {dataset.name:dataset.dataset for dataset in datasets}
        self.create_or_refresh_data()

    def __len__(self):
        return len(self.task_batch_index_list)

    def create_or_refresh_data(self):
        self.task_batch_index_list: list[TaskBatchIndex] = []
        for dataset in self.datasets:
            print("self.max_samples",self.max_samples)
            print("len(dataset.dataset)",len(dataset.dataset))
            max_samples = self.max_samples or len(dataset.dataset)
            batch_size = self.batch_size 
            num_samples = (max_samples // batch_size) * batch_size
            buffer = []
            for i in RandomSampler(dataset.dataset, num_samples=num_samples):
                buffer.append(i)
                if len(buffer) == batch_size:
                    self.task_batch_index_list.append(TaskBatchIndex(name=dataset.name, batch_index=buffer))
                    buffer = []
        self.random_index_list = list(RandomSampler(self.task_batch_index_list))

    def get_Inbatch(self, records):
        Inbatch_records = []
        for record in records:
            if 'query' not in record:
                print(record)
                exit()
            query = record['query']
            passage = [record['pos']]
            passage.extend(record['neg'])
            Inbatch_records.append(Inbatch(query, passage))
        assert len(Inbatch_records) == self.batch_size, 'error, current batch size not match !!!'
        return Inbatch_records


    def get_pair(self, records):
        pair_records = []
        for record in records:
            txt1 = record['txt1']
            txt2 = record['txt2']
            label = record['label']
            pair_records.append(Pair(txt1, txt2, label))
        assert len(pair_records) == self.batch_size, 'error, current batch size not match !!!'
        return pair_records

    def __getitem__(self, index: int):
        index = self.random_index_list[index]
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index

        hf_dataset = self.name_dataset_map[task_name]
        records = [hf_dataset[i] for i in batch_index]

        if hf_dataset[0]['type'] == 'inbatch':
            pair_records = self.get_Inbatch(records)
        elif hf_dataset[0]['type'] == 'pair':
            pair_records = self.get_pair(records) 
        else:
            raise NotImplementedError('only support pair contrast and pair scored')

        if not pair_records:
            print(f'records is empty', records)
            return self.__getitem__(index + 1)
        return pair_records


class UniCollator(DataCollatorWithPadding):
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

    def __call__(self, records):
        records = records[0]
        if isinstance(records[0], Inbatch):
            query = [record.query for record in records]
            passage = [record.passage for record in records]
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
            return {"type": "inbatch", "query": q_collated, "passage": d_collated}

        if isinstance(records[0],Pair):

            txt1 = [record.txt1 for record in records]
            txt2 = [record.txt2 for record in records]
            label = [record.label for record in records]

            neg_pos_idxs = [[0.0] * len(records) for _ in range(len(records))]
            for i in range(len(records)):
                for j in range(len(records)):
                    if label[i] == 1 and label[j] == 0:
                        neg_pos_idxs[i][j] = 1.0
                    elif label[i] == 0 and label[j] == 1:
                        neg_pos_idxs[i][j] = 1.0

            t1_collated = self.tokenizer(
                txt1,
                padding=True,
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
            )
            t2_collated = self.tokenizer(
                txt1,
                padding=True,
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
            )
            return {"type": "pair", "txt1": t1_collated, "txt2": t2_collated, "idx":neg_pos_idxs } 
        else:
            raise NotImplementedError("only support pair and inbatch")
