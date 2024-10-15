import logging
import os
import warnings
from dataclasses import dataclass

import pandas as pd
from datasets import disable_progress_bar, load_dataset, load_from_disk
from recbole.data import data_preparation

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

disable_progress_bar()


TASK_ATTRS = {
    "magazine": {
        "load_args": ("McAuley-Lab/Amazon-Reviews-2023", "0core_rating_only_Magazine_Subscriptions"),
    },
    "epinions": {
        "load_args": (),
    },
    "ml-100k": {
        "load_args": (),
    },
    "ml-1m": {
        "load_args": (),
    },
}


@dataclass
class DataConfig:
    dataset_name: str
    dataset_path: str
    raw_dataset_path: str
    train_batch_size: int
    eval_batch_size: int
    test_batch_size: int
    recbole_model: str
    recbole_config: str


class DataModule:
    def __init__(self, config: DataConfig):
        self.config = config
        # preprocessed_dataset
        if not os.path.exists(self.config.dataset_path):
            self.dataset_attr = TASK_ATTRS[self.config.dataset_name]
            self.dataset = self.get_dataset()
            self.processed_dataset = self.process_dataset()
        # generate dataloader
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.get_dataloader()
        logger.info(f"Datasets: {self.dataset}")

    def get_dataset(self):
        """load raw datasets from source"""
        if os.path.exists(self.config.raw_dataset_path):
            dataset = load_from_disk(self.config.raw_dataset_path)
        else:
            assert self.config.dataset_name in TASK_ATTRS
            os.makedirs(self.config.raw_dataset_path, exist_ok=True)
            dataset = load_dataset(*self.dataset_attr["load_args"])
            dataset.save_to_disk(self.config.raw_dataset_path)
        return dataset

    def process_dataset(self):
        dataset_df = pd.DataFrame(self.dataset["full"])
        dataset_df.columns = ["uid", "iid", "rating", "timestamp"]

        ### Filter users and items with less than 3 interactions ###
        filtered_review_df = dataset_df.groupby("iid").filter(lambda x: len(x) >= 3)
        filtered_review_df = (
            filtered_review_df.groupby("uid")
            .filter(lambda x: len(x) >= 5)
            .groupby("uid")
            .apply(
                lambda x: x.sort_values(by=["timestamp"], ascending=[True]),
                include_groups=True,
            )
            .reset_index(drop=True)
        )
        filtered_review_df.columns = [
            "user_id:token",
            "item_id:token",
            "rating:float",
            "timestamp:float",
        ]

        logger.info(f"Save preprocessed datasets to `{self.config.dataset_path}`")
        filtered_review_df.to_csv(self.config.dataset_path, sep="\t", index=False)
        return filtered_review_df

    def get_dataloader(self):
        from omegaconf import OmegaConf
        from recbole.config import Config
        from recbole.data.dataset import SequentialDataset

        config = Config(
            model=self.config.recbole_model,
            dataset=self.config.dataset_name,
            config_dict=OmegaConf.to_container(self.config.recbole_config, resolve=True),
        )
        self.dataset = SequentialDataset(config)
        self.train_loader, self.valid_loader, self.test_loader = data_preparation(config, self.dataset)

    def train_loader(self):
        return self.train_loader

    def valid_loader(self):
        return self.valid_loader

    def test_loader(self):
        return self.test_loader
