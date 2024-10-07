import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.utils
from recbole.config import Config
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data.interaction import Interaction
from recbole.evaluator import Collector
from recbole.evaluator import Evaluator as RecboleEvaluator
from torch.cuda import amp
from tqdm import tqdm, trange

from distilled_data import DistilledData
from model import LearnerModel
from utils import average, build_optimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    skip_train: bool
    save_ckpt_dir: str

    epochs: int
    batch_size: int

    seq_lr: float
    seq_optim: str
    seq_scheduler: str
    seq_warmup_ratio: float
    seq_weight_decay: float

    net_lr: float
    net_optim: str
    net_weight_decay: float

    inner_steps: int
    window: int
    min_window: int

    max_grad_norm: float
    val_interval: int
    log_interval: int
    n_eval_model: int
    fp16: bool
    bf16: bool


@dataclass
class EvaluateConfig:
    dataset: str
    n_eval_model: int
    fp16: bool
    bf16: bool

    def __post_init__(self):
        assert not (self.fp16 and self.bf16)


class Evaluator:
    def __init__(
        self,
        config: EvaluateConfig,
        train_config: TrainConfig,
        recbole_config: Config,
        model: LearnerModel,
    ):
        self.config = config
        self.train_config = train_config
        self.recbole_config = recbole_config
        self.model = model
        self.eval_collector = Collector(self.recbole_config)
        self.recbole_evaluator = RecboleEvaluator(self.recbole_config)

    def evaluate(
        self,
        distilled_data: DistilledData,
        eval_loader: Optional[FullSortEvalDataLoader],
        n_eval_model: Optional[int] = None,
        verbose: bool = False,
    ) -> dict[str, tuple[float]]:
        self.model.cuda()
        distilled_data.cuda()
        if n_eval_model is None:
            n_eval_model = self.config.n_eval_model

        all_results = []
        for i in trange(
            n_eval_model,
            dynamic_ncols=True,
            ascii=" >=",
            desc="Evaluate ",
            leave=False,
        ):
            self.model.model.apply(self.model.model.init_weights)
            self.train_model(self.model, distilled_data)
            results = self.evaluate_model(self.model, eval_loader)
            if verbose:
                logger.info(
                    "[{:>{}}/{}]: {}".format(i, len(str(self.config.n_eval_model)), self.config.n_eval_model, results)
                )
            all_results.append(results)

        average_results = average(all_results, std=True)
        avg = {k: v[0] for k, v in average_results.items()}
        if verbose:
            logger.info(f"Average results: {avg}")

        return average_results

    def train_model(
        self,
        model: LearnerModel,
        distilled_data: DistilledData,
    ):
        model.train()
        optimizer_net = build_optimizer(
            params=model.parameters(),
            learner=self.train_config.net_optim,
            learning_rate=self.train_config.net_lr,
            weight_decay=self.train_config.net_weight_decay,
        )
        for step in trange(
            self.train_config.inner_steps,
            dynamic_ncols=True,
            ascii=" >=",
            desc="Training ",
            leave=False,
        ):
            optimizer_net.zero_grad()
            batch_syn = distilled_data.get_batch(step)
            inputs_embeds = batch_syn.pop("inputs_embeds")
            _, loss = model(inputs_embeds)
            loss.backward()
            optimizer_net.step()

    def evaluate_model(
        self,
        model: LearnerModel,
        data_loader: Optional[FullSortEvalDataLoader],
    ):
        self.device = model.device
        self.test_batch_size = data_loader._batch_size
        self.tot_item_num = data_loader._dataset.item_num
        self.item_id_field = model.model.ITEM_ID

        model.cuda()
        model.eval()

        if isinstance(data_loader, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval()
        else:
            eval_func = self._neg_sample_batch_eval

        total_loss, num_samples = 0, 0
        iter_data = tqdm(
            data_loader,
            total=len(data_loader),
            dynamic_ncols=True,
            ascii=" >=",
            desc="Evaluate ",
            leave=False,
        )
        with torch.no_grad():
            with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                for batch_idx, batched_data in enumerate(iter_data):
                    num_samples += len(batched_data)
                    interaction, scores, positive_u, positive_i = eval_func(model, batched_data)
                    self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
                self.eval_collector.model_collect(model)
                struct = self.eval_collector.get_data_struct()
                result = self.recbole_evaluator.evaluate(struct)
                result["loss"] = total_loss / num_samples

        return result

    def evaluate_fast(
        self,
        distilled_data: DistilledData,
        eval_loader: Optional[FullSortEvalDataLoader],
        n_eval_model: Optional[int] = None,
    ):
        self.device = self.model.device
        self.test_batch_size = eval_loader._batch_size
        self.tot_item_num = eval_loader._dataset.item_num
        self.item_id_field = self.model.model.ITEM_ID

        self.model.cuda()
        self.model.eval()
        distilled_data.cuda()

        if n_eval_model is None:
            n_eval_model = self.config.n_eval_model
        reset_model_interval = max(len(eval_loader) // n_eval_model, 1)

        if isinstance(eval_loader, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval()
        else:
            eval_func = self._neg_sample_batch_eval

        total_loss, num_samples = 0, 0
        iter_data = tqdm(
            eval_loader,
            total=len(eval_loader),
            dynamic_ncols=True,
            ascii=" >=",
            desc="Evaluate ",
            leave=False,
        )
        for batch_idx, batched_data in enumerate(iter_data):
            if batch_idx % reset_model_interval == 0:
                self.model.model.apply(self.model.model.init_weights)
                self.train_model(self.model, distilled_data)
            with torch.no_grad():
                with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    num_samples += len(batched_data)
                    interaction, scores, positive_u, positive_i = eval_func(self.model, batched_data)
                    self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.recbole_evaluator.evaluate(struct)
        result["loss"] = total_loss / num_samples

        return result

    def _full_sort_batch_eval(self, model, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        scores = model.model.full_sort_predict(interaction.to(self.device))
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def _neg_sample_batch_eval(self, model, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        batch_size = interaction.length
        if batch_size <= self.test_batch_size:
            origin_scores = model.model.predict(interaction.to(self.device))
        else:
            origin_scores = self._spilt_predict(model, interaction, batch_size)

        col_idx = interaction[self.item_id_field]
        batch_user_num = positive_u[-1] + 1
        scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
        scores[row_idx, col_idx] = origin_scores
        return interaction, scores, positive_u, positive_i

    def _spilt_predict(self, model, interaction, batch_size):
        spilt_interaction = dict()
        for key, tensor in interaction.interaction.items():
            spilt_interaction[key] = tensor.split(self.test_batch_size, dim=0)
        num_block = (batch_size + self.test_batch_size - 1) // self.test_batch_size
        result_list = []
        for i in range(num_block):
            current_interaction = dict()
            for key, spilt_tensor in spilt_interaction.items():
                current_interaction[key] = spilt_tensor[i]
            result = model.model.predict(Interaction(current_interaction).to(self.device))
            if len(result.shape) == 0:
                result = result.unsqueeze(0)
            result_list.append(result)
        return torch.cat(result_list, dim=0)

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16
