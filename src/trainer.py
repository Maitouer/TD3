import logging
import os
import random
from typing import List

import higher
import torch
import torch.nn.functional as F
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.utils import dict2str, get_gpu_usage
from torch.cuda import amp
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from transformers import get_scheduler

import wandb
from distilled_data import DistilledData
from evaluator import Evaluator, TrainConfig
from model import LearnerModel
from utils import (
    FeatureDataLoader,
    build_optimizer,
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config

    def fit(
        self,
        distilled_data: DistilledData,
        model: LearnerModel,
        train_loader: FeatureDataLoader,
        valid_loader: FullSortEvalDataLoader,
        evaluators: List[Evaluator],
    ):
        model.cuda()
        distilled_data.cuda()
        device = model.device

        wandb.define_metric("train_step")
        wandb.define_metric("valid_step")
        wandb.define_metric("train/loss", step_metric="train_step")

        outer_steps = len(train_loader)
        logger.info(f"outer steps: `{outer_steps}`")

        if self.config.log_interval == -1:
            self.config.log_interval = max(1, outer_steps // 10)

        scaler = amp.GradScaler(enabled=self.use_amp)
        optimizer_seq, scheduler_seq = self.configure_optimizer(
            distilled_data=distilled_data,
            max_training_steps=self.config.epochs * outer_steps,
        )

        # evaluate before training
        val_by = "mrr@10"
        best_val_metric = 0
        for idx, evaluator in enumerate(evaluators):
            results = evaluator.evaluate_fast(
                distilled_data,
                valid_loader,
                n_eval_model=self.config.n_eval_model,
            )
            original_keys = list(results.keys())
            for key in original_keys:
                results[f"{idx}-{key}"] = results.pop(key)
                wandb.define_metric(f"{idx}-{key}", step_metric="valid_step")
            best_val_metric += results[f"{idx}-{val_by}"]
            wandb.log({**results, "valid_step": 0})
            logger.info(f"VALIDATE [Epoch=0/{self.config.epochs}]: {dict2str(results)}")
        # save best checkpoint based on loss
        best_val_metric = best_val_metric / len(evaluators)
        best_ckpt_path = os.path.join(self.config.save_ckpt_dir, "best-ckpt")
        distilled_data.save_pretrained(best_ckpt_path)
        wandb.save(best_ckpt_path)

        # train synthetic data
        for epoch in range(self.config.epochs):
            model.init_model_params()
            model.train()

            log_train_loss = 0
            iter_data = tqdm(
                train_loader,
                total=len(train_loader),
                dynamic_ncols=True,
                ascii=" >=",
                desc="Training ",
                leave=False,
            )
            for outer_step, batch in enumerate(iter_data):
                inters_real, features_real = batch[0], batch[1]
                inters_real = inters_real.to(device)
                features_real = features_real.to(device)

                optimizer_net = build_optimizer(
                    params=model.parameters(),
                    learner=self.config.net_optim,
                    learning_rate=self.config.net_lr,
                    weight_decay=self.config.net_weight_decay,
                )
                curriculum = random.randint(self.config.min_window, self.config.inner_steps - self.config.window)
                for idx in range(curriculum):
                    optimizer_net.zero_grad()
                    inters_syn = distilled_data.get_batch(idx).pop("inputs_embeds")
                    _, loss_syn = model(inters_syn)
                    loss_syn.backward()
                    optimizer_net.step()
                with higher.innerloop_ctx(model, optimizer_net, copy_initial_weights=True) as (fnet, diffopt):
                    for idx in range(self.config.window):
                        inters_syn = distilled_data.get_batch(idx).pop("inputs_embeds")
                        _, loss_syn = fnet(inters_syn)
                        diffopt.step(loss_syn)
                    outputs_real, loss_real = fnet(inters_real.to(device))
                # feature alignment loss
                loss_latent = F.mse_loss(features_real.to(model.device), outputs_real[-1])

                loss_all = loss_real + loss_latent
                log_train_loss += loss_all.item()

                # compute gradient
                try:
                    optimizer_seq.zero_grad()
                    scaler.scale(loss_all).backward()
                except Exception:
                    logger.info("Backward Error!")

                # gradient clipping
                if self.config.max_grad_norm is not None:
                    scaler.unscale_(optimizer_seq)
                    torch.nn.utils.clip_grad_norm_(
                        distilled_data.data_dict().values(),
                        max_norm=self.config.max_grad_norm,
                    )

                # update distilled data
                scaler.step(optimizer_seq)
                scaler.update()
                scheduler_seq.step()

                iter_data.set_postfix({"lr": scheduler_seq.get_last_lr()[0]})

                if (outer_step + 1) % self.config.log_interval == 0:
                    log_train_loss /= self.config.log_interval
                    wandb.log({"train/loss": log_train_loss, "train_step": outer_steps * epoch + outer_step})
                    wandb.log({"lr": scheduler_seq.get_last_lr()[0], "train_step": outer_steps * epoch + outer_step})
                    logger.info(
                        f"TRAINING [Epoch={((outer_step + 1) / outer_steps + epoch):>3.1f}]: train = {log_train_loss:06.4f}, real = {loss_real:06.4f}, latent = {loss_latent:06.4f}, memory = {get_gpu_usage(device)}"
                    )
                    log_train_loss = 0

            if (epoch + 1) % self.config.val_interval == 0:
                val_metric = 0
                for idx, evaluator in enumerate(evaluators):
                    results = evaluator.evaluate_fast(
                        distilled_data,
                        valid_loader,
                        n_eval_model=self.config.n_eval_model,
                    )
                    original_keys = list(results.keys())
                    for key in original_keys:
                        results[f"{idx}-{key}"] = results.pop(key)
                    val_metric += results[f"{idx}-{val_by}"]
                    wandb.log({**results, "valid_step": outer_steps * (epoch + 1)})
                    logger.info(f"VALIDATE [Epoch={epoch + 1:03}/{self.config.epochs}]: {dict2str(results)}")
                val_metric = val_metric / len(evaluators)
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    distilled_data.save_pretrained(best_ckpt_path)
                    wandb.save(best_ckpt_path)
                    logger.info(f"Save best checkpoint at `{best_ckpt_path}`")

        logger.info("Finish training !!!")

        # save last checkpoint
        last_ckpt_path = os.path.join(self.config.save_ckpt_dir, "last-ckpt")
        distilled_data.save_pretrained(last_ckpt_path)
        wandb.save(last_ckpt_path)
        logger.info(f"Save last checkpoint at `{last_ckpt_path}`")

        # load best checkpoint
        best_checkpoint = torch.load(os.path.join(best_ckpt_path, "data_dict"))
        distilled_data.load_data_dict(best_checkpoint)

    def configure_optimizer(
        self,
        distilled_data: DistilledData,
        max_training_steps: int,
    ) -> tuple[Optimizer, _LRScheduler]:
        optimizer_class = {"sgd": SGD, "momentum": SGD, "adam": Adam, "adamw": AdamW}
        assert self.config.seq_optim in optimizer_class

        data_dict = distilled_data.data_dict()
        assert data_dict.keys() >= {"G"}, f"{data_dict.keys()}"
        grouped_params = [
            {
                "params": data_dict["G"],
                "weight_decay": self.config.seq_weight_decay,
                "lr": self.config.seq_lr,
            },
            {
                "params": data_dict["U"],
                "weight_decay": self.config.seq_weight_decay,
                "lr": self.config.seq_lr,
            },
            {
                "params": data_dict["T"],
                "weight_decay": self.config.seq_weight_decay,
                "lr": self.config.seq_lr,
            },
            # {
            #     "params": data_dict["emb"],
            #     "weight_decay": self.config.seq_weight_decay,
            #     "lr": self.config.seq_lr,
            # },
            # {
            #     "params": data_dict["decoder"],
            #     "weight_decay": self.config.seq_weight_decay,
            #     "lr": self.config.seq_lr,
            # },
        ]
        optimizer = optimizer_class[self.config.seq_optim](grouped_params, lr=0.01)  # `lr=1.0` is not used (dummy)
        logger.info(f"Optimizer: {optimizer}")

        # learning rate scheduler
        scheduler = get_scheduler(
            name=self.config.seq_scheduler,
            optimizer=optimizer if optimizer is not None else optimizer,
            num_warmup_steps=int(max_training_steps * self.config.seq_warmup_ratio),
            num_training_steps=max_training_steps,
        )

        return optimizer, scheduler

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16
