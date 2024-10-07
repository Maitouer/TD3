from time import time

import torch
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data.interaction import cat_interactions
from recbole.trainer import Trainer
from recbole.utils import (
    EvaluatorType,
    dict2str,
    early_stopping,
    get_gpu_usage,
    set_color,
)
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm


class PretrainTrainer(Trainer):
    def __init__(self, config, model):
        super(PretrainTrainer, self).__init__(config, model)
        self.epochs = self.config["epochs"]
        # for pretrain
        self.pretrain_epochs = self.config["pretrain_epochs"]
        self.saved_model_file = self.config["pretrained_path"]
        self.tensorboard = None
        self.wandblogger = None

    def save_pretrained_model(self, epoch, save=None, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id
            saved_model_file (str): file name for saved pretrained model

        """
        state = {
            "config": self.config,
            "epoch": epoch,
            "cur_step": self.cur_step,
            "best_valid_score": self.best_valid_score,
            "state_dict": self.model.state_dict(),
            "other_parameter": self.model.other_parameter(),
            "optimizer": self.optimizer.state_dict(),
        }
        if save:
            path = self.saved_model_file.split("epochs_")[0] + f"epochs_{epoch}.pth"
            torch.save(state, f"{path}", pickle_protocol=4)
        else:
            path = self.saved_model_file
            torch.save(state, self.saved_model_file, pickle_protocol=4)
        if verbose:
            self.logger.info(set_color("Saving current", "blue") + f": {path}")

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                ascii=" >=",
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )
        total_features = []
        total_interactions = []
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            features, losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))

            total_features.append(features[-1].detach().cpu())
            total_interactions.append(interaction.cpu())

        return (cat_interactions(total_interactions), torch.cat(total_features, dim=0)), total_loss

    @torch.no_grad()
    def generate_feature(self, train_data, loss_func=None, show_progress=True):
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                ascii=" >=",
                desc=set_color("Generate feature", "pink"),
            )
            if show_progress
            else train_data
        )
        total_features = []
        total_interactions = []
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            features, losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()

            total_features.append(features[-1].detach().cpu())
            total_interactions.append(interaction.cpu())

        return (cat_interactions(total_interactions), torch.cat(total_features, dim=0)), total_loss

    def pretrain(self, train_data, valid_data=None, saved=True, verbose=True, show_progress=False):
        valid_step = 0
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            (total_interactions, total_features), train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)

            # for eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self.save_pretrained_model(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)

                if update_flag:
                    best_features = total_features
                    best_interactions = total_interactions
                    if saved:
                        self.save_pretrained_model(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

            if epoch_idx >= 5 and epoch_idx <= 30 and epoch_idx % 5 == 0:
                self.save_pretrained_model(epoch=epoch_idx, save=True, verbose=verbose)

        return (best_interactions, best_features), self.best_valid_score, self.best_valid_result

    def fit(self, train_data, valid_data=None, saved=False, verbose=True, show_progress=False):
        valid_step = 0
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            (total_interactions, total_features), train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)

            # for eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self.save_pretrained_model(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)

                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)

                if update_flag:
                    if saved:
                        self.save_pretrained_model(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data._dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                ascii=" >=",
                desc=set_color("Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow"))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)

        return result
