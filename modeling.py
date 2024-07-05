import logging
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)

def cosent_loss(neg_pos_idxs, pred_sims, cosent_ratio, zero_data):
    pred_sims = pred_sims * cosent_ratio
    pred_sims = pred_sims[:, None] - pred_sims[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    pred_sims = pred_sims - (1 - neg_pos_idxs) * 1e12
    pred_sims = pred_sims.view(-1)
    pred_sims = torch.cat((zero_data, pred_sims), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return torch.logsumexp(pred_sims, dim=0)

def get_mean_params(model):
    """
    :param model:
    :return:Dict[para_name, para_weight]
    """
    result = {}
    for param_name, param in model.named_parameters():
        result[param_name] = param.data.clone()
    return result

@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 ewc_ratio: float = 1.0,
                 cosent_ratio: float = 2.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.original_weight = get_mean_params(self.model)
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config
        self.cosent_ratio = cosent_ratio

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
    
    def cosent_loss(self, neg_pos_idxs, pred_sims, cosent_ratio, zero_data):
        pred_sims = pred_sims * cosent_ratio
        pred_sims = pred_sims[:, None] - pred_sims[None, :]  # 这里是算出所有位置 两两之间余弦的差值
        pred_sims = pred_sims - (1 - neg_pos_idxs) * 1e12
        pred_sims = pred_sims.view(-1)
        pred_sims = torch.cat((zero_data, pred_sims), dim=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
        return torch.logsumexp(pred_sims, dim=0)
    
    def cumpute_inbatch_loss(self, type, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.cross_entropy(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.cross_entropy(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )
    
    def cumpute_pair_loss(self, type, txt1: Dict[str, Tensor] = None, txt2: Dict[str, Tensor] = None, idx: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        vecs1 = self.encode(txt1)
        vecs2 = self.encode(txt2)
        neg_pos_idxs = torch.tensor(idx).float() .to(vecs1.device)
        pred_sims = F.cosine_similarity(vecs1, vecs2)
        # print(name, pred_sims.shape)
        loss = cosent_loss(
            neg_pos_idxs=neg_pos_idxs,
            pred_sims=pred_sims,
            cosent_ratio=self.cosent_ratio,
            zero_data=torch.tensor([0.0]).to(vecs1.device)
        )
        return EncoderOutput(
            loss=loss,
            scores=None,
            q_reps=vecs1,
            p_reps=vecs2,
        )
    
    def forward(self, **kwargs):
        if kwargs['type'] == 'inbatch':
            return self.cumpute_inbatch_loss(**kwargs)
        elif kwargs['type'] == 'pair':
            return self.cumpute_pair_loss(**kwargs)


    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
