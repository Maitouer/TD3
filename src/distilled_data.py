import json
import logging
import os
from abc import ABCMeta
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LearnerTrainConfig:
    train_step: int
    batch_size: int


@dataclass
class DistilledDataConfig:
    pretrained_data_path: str
    seq_num: int
    seq_len: int
    seq_dim1: int
    seq_dim2: int
    seq_dim3: int
    farzi_dim: int
    fix_order: bool


class DistilledFeature(metaclass=ABCMeta):
    def __init__(self):
        self.data: torch.Tensor

    def initialize_data(self, initialized_values: torch.Tensor, size_strict=True):
        if size_strict:
            assert (
                self.data.shape == initialized_values.shape
            ), f"{self.data.shape} should be matched to {initialized_values.shape}"
        else:
            raise NotImplementedError

        with torch.no_grad():
            self.data.copy_(initialized_values)

    def __getitem__(self, index):
        pass

    def cuda(self):
        if not self.data.is_cuda:
            grad = self.data.grad
            self.data = self.data.detach().cuda().requires_grad_(self.data.requires_grad)
            self.data.grad = grad


class DistilledInputEmbedding(DistilledFeature):
    def __init__(
        self,
        seq_num: int,
        seq_len: int,
        seq_dim: int,
    ):
        self.data = nn.Parameter(torch.randn(seq_num, seq_len, seq_dim))
        nn.init.kaiming_uniform_(self.data)

    def __getitem__(self, index):
        return self.data[index]


class DistilledDecoder(DistilledFeature):
    def __init__(
        self,
        seq_dim: int,
        item_num: int,
    ):
        self.data = nn.Parameter(torch.randn(seq_dim, item_num))
        nn.init.kaiming_uniform_(self.data)

    def __getitem__(self, index):
        return self.data


class DistilledG(DistilledFeature):
    def __init__(
        self,
        seq_dim1: int,
        seq_dim2: int,
        seq_dim3: int,
    ):
        self.data = nn.Parameter(torch.randn(seq_dim1, seq_dim2, seq_dim3))
        nn.init.kaiming_normal_(self.data)

    def __getitem__(self, index):
        return self.data


class DistilledU(DistilledFeature):
    def __init__(
        self,
        seq_num: int,
        seq_dim: int,
    ):
        self.data = nn.Parameter(torch.randn(seq_num, seq_dim))
        self.data.data.normal_(mean=0.0, std=0.02)

    def __getitem__(self, index):
        return self.data[index]


class DistilledV(DistilledFeature):
    def __init__(
        self,
        item_emb: None,
    ):
        self.data = torch.tensor(item_emb)
        self.data.requires_grad_(False)

    def __getitem__(self, index):
        return self.data


class DistilledT(DistilledFeature):
    def __init__(
        self,
        seq_len,
        seq_dim: int,
    ):
        self.data = nn.Parameter(torch.randn(seq_len, seq_dim))
        self.data.data.normal_(mean=0.0, std=0.02)

    def __getitem__(self, index):
        return self.data


class DistilledData:
    def __init__(
        self,
        config: DistilledDataConfig,
        train_config: LearnerTrainConfig,
        item_num: int,
        item_emb: None,
    ):
        if not isinstance(config, DistilledDataConfig):
            config = DistilledDataConfig(**config)
        self.config = config

        if not isinstance(train_config, LearnerTrainConfig):
            train_config = LearnerTrainConfig(**train_config)
        self.train_config = train_config

        if self.config.fix_order:
            assert config.seq_num % train_config.batch_size == 0

        self.item_num = item_num
        self.item_emb = item_emb

        self.G = DistilledG(seq_dim1=config.seq_dim1, seq_dim2=config.seq_dim2, seq_dim3=config.seq_dim3)
        self.U = DistilledU(seq_num=config.seq_num, seq_dim=config.seq_dim1)
        self.T = DistilledT(seq_len=config.seq_len, seq_dim=config.seq_dim2)
        self.V = DistilledV(item_emb=item_emb)
        self.data: dict[str, DistilledFeature] = {
            "G": self.G,
            "U": self.U,
            "V": self.V,
            "T": self.T,
        }

        # # For farzi
        # self.emb = DistilledInputEmbedding(seq_num=config.seq_num, seq_len=config.seq_len, seq_dim=config.farzi_dim)
        # self.decoder = DistilledDecoder(seq_dim=config.farzi_dim, item_num=item_num)
        # self.data: dict[str, DistilledFeature] = {
        #     "emb": self.emb,
        #     "decoder": self.decoder,
        # }

    def get_batch(self, step):
        indices = self.get_batch_indices(step)
        return {
            "inputs_embeds": torch.einsum(
                "ijk,ai,bj,ck->abc", self.G.data, self.U[indices], self.T.data, self.V.data
            ).softmax(dim=-1),
        }
        # return {
        #     "inputs_embeds": (self.emb[indices] @ self.decoder.data).softmax(dim=-1),
        # }

    def get_batch_indices(self, step):
        batch_size = self.train_config.batch_size
        data_size = self.config.seq_num
        if self.config.fix_order:
            cycle = step % int(data_size / batch_size)
            return torch.arange(batch_size * cycle, batch_size * (cycle + 1))
        else:
            return torch.randperm(data_size)[:batch_size]

    def data_dict(self, detach: bool = False):
        data_dict = {name: feature.data for name, feature in self.data.items()}
        if detach:
            data_dict = {name: data.detach().cpu() for name, data in data_dict.items()}
        return data_dict

    def save_pretrained(self, save_path: str | os.PathLike):
        os.makedirs(save_path, exist_ok=True)

        # save config as json file
        config = {
            "config": asdict(self.config),
            "train_config": asdict(self.train_config),
            "item_num": self.item_num,
            "item_emb": None,
        }
        json.dump(config, open(os.path.join(save_path, "config.json"), "w"), indent=4)

        # save distilled data
        torch.save(self.data_dict(detach=True), os.path.join(save_path, "data_dict"))

        logger.info(f"Save distilled data in `{save_path}`")

    def load_data_dict(self, data_dict: dict[str, torch.Tensor]):
        assert (
            self.data.keys() == data_dict.keys()
        ), f"given keys: {self.data.keys()}, expected keys: {data_dict.keys()}"
        for name in self.data.keys():
            self.data[name].initialize_data(data_dict[name])

    @classmethod
    def from_pretrained(cls, save_path: str | os.PathLike, item_emb):
        assert os.path.exists(save_path)

        # load config
        config = json.load(open(os.path.join(save_path, "config.json")))
        config["item_emb"] = item_emb
        distilled_data = cls(**config)

        # load data
        pretrained_data = torch.load(os.path.join(save_path, "data_dict"))
        distilled_data.load_data_dict(pretrained_data)
        logger.info(f"Load distilled data from `{save_path}`")

        return distilled_data

    def cuda(self):
        for feature in self.data.values():
            feature.cuda()
