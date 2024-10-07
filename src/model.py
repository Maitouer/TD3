import logging
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from torch.nn.init import xavier_normal_, xavier_uniform_

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    learner_model: dict
    use_pretrained_model: bool
    use_pretrained_embed: bool
    freeze_pretrained_embed: bool


class LearnerModel(nn.Module):
    def __init__(self, model, config, dataset):
        super().__init__()
        self.config = config
        self.model = eval(model)(self.config, dataset)
        self.checkpoint = torch.load(self.config.pretrained_path)
        self.initial_state_dict = self.checkpoint["state_dict"]

        self.init_weights()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def init_weights(self):
        self.model.load_state_dict(self.initial_state_dict)

        if not (self.config.use_pretrained_model and self.config.use_pretrained_embed):
            assert hasattr(self.model, "init_weights")
            assert hasattr(self.model, "init_embedding")
            self.model.apply(self.model.init_weights)
            self.model.apply(self.model.init_embedding)
        elif not self.config.use_pretrained_model:
            assert hasattr(self.model, "init_weights")
            self.model.apply(self.model.init_weights)
        elif not self.config.use_pretrained_embed:
            assert hasattr(self.model, "init_embedding")
            self.model.apply(self.model.init_embedding)

        if self.config.freeze_pretrained_embed:
            self.model.item_embedding.from_pretrained(self.initial_state_dict["item_embedding.weight"], freeze=True)
            self.model.position_embedding.from_pretrained(
                self.initial_state_dict["position_embedding.weight"], freeze=True
            )

    def init_model_params(self):
        # inner_choices = [32, 64, 128]
        epoch_choices = list(range(5, 31, 5))
        # inner = random.choice(inner_choices)
        epoch = random.choice(epoch_choices)

        path = re.sub(r"epochs_.*pth", f"epochs_{epoch}.pth", self.config.pretrained_path)
        # path = re.sub(r"inners_.*maxlen", f"inners_{inner}.maxlen", path)
        logger.info(f"Load checkpoint `{epoch}`")
        checkpoint = torch.load(path)
        initial_state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(initial_state_dict)
        self.model.item_embedding.from_pretrained(self.initial_state_dict["item_embedding.weight"], freeze=True)
        self.model.position_embedding.from_pretrained(self.initial_state_dict["position_embedding.weight"], freeze=True)

    @property
    def device(self):
        return self.model.device


class SASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRec, self).__init__(config, dataset)

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attn_dropout_prob = config.attn_dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps

        self.initializer_range = config.initializer_range
        self.loss_type = config.loss_type

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self.init_weights)
        self.apply(self.init_embedding)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_embedding(self, module):
        """Initialize the embedding"""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)

    def forward(self, interaction):
        if isinstance(interaction, torch.Tensor) and interaction.dim() == 3:
            device = interaction.device
            seq_num = interaction.size(0)
            seq_len = interaction.size(1)  # total seq len, last item as target pos item

            item_seq = torch.arange(1, seq_len, device=device).unsqueeze(0).repeat(seq_num, 1)
            item_seq_len = torch.full((seq_num,), seq_len - 1, device=device)

            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)

            item_emb = interaction @ self.item_embedding.weight
            input_emb = item_emb[:, :-1, :] + position_embedding
            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)

            extended_attention_mask = self.get_attention_mask(item_seq)

            trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
            features = [self.gather_indexes(layer_output, item_seq_len - 1) for layer_output in trm_output]
            output = features[-1]

            # KL-Div Loss
            pos_prob = interaction[:, -1, :]
            logits = torch.matmul(output, self.item_embedding.weight.transpose(0, 1))
            loss = F.kl_div(F.log_softmax(logits + 1e-9, dim=-1), pos_prob + 1e-9, reduction="batchmean")

            """ Augmentation """
            pos_item = torch.randint(
                low=5, high=seq_len - 1, size=(seq_num,), device=device
            )  # indices for target pos item
            mask = torch.arange(seq_len - 1, device=device) >= pos_item.unsqueeze(dim=1)
            aug_item_seq = item_seq
            aug_item_seq[mask] = 0
            aug_item_seq_len = pos_item

            extended_attention_mask = self.get_attention_mask(aug_item_seq)

            trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
            features = [self.gather_indexes(layer_output, aug_item_seq_len - 1) for layer_output in trm_output]
            output = features[-1]

            # KL-Div Loss
            pos_prob = interaction[torch.arange(seq_num), pos_item]
            logits = torch.matmul(output, self.item_embedding.weight.transpose(0, 1))
            loss += F.kl_div(F.log_softmax(logits + 1e-9, dim=-1), pos_prob + 1e-9, reduction="batchmean")

            loss /= 2

        if not (isinstance(interaction, torch.Tensor) and interaction.dim() == 3):
            features, loss = self.calculate_loss(interaction)

        return features, loss

    def _forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        features = [self.gather_indexes(layer_output, item_seq_len - 1) for layer_output in trm_output]
        output = features[-1]

        return features, output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        features, output = self._forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return features, loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        _, seq_output = self._forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        _, seq_output = self._forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class BERT4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(BERT4Rec, self).__init__(config, dataset)

        # load parameters info
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config["inner_size"]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.mask_ratio = config["mask_ratio"]

        self.MASK_ITEM_SEQ = config["MASK_ITEM_SEQ"]
        self.POS_ITEMS = config["POS_ITEMS"]
        self.NEG_ITEMS = config["NEG_ITEMS"]
        self.MASK_INDEX = config["MASK_INDEX"]

        self.loss_type = config["loss_type"]
        self.initializer_range = config["initializer_range"]

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_ffn = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_gelu = nn.GELU()
        self.output_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ["BPR", "CE"]
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self.init_weights)
        self.apply(self.init_embedding)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_embedding(self, module):
        """Initialize the embedding"""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)

    def reconstruct_test_data(self, item_seq, item_seq_len):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        item_seq = item_seq[:, 1:]
        return item_seq

    def mask_interaction(self, interaction):
        batch_size, num_items = interaction.shape
        mask_item_length = max(1, int(self.mask_ratio * num_items))
        index_matrix = torch.zeros((batch_size, mask_item_length), dtype=torch.long)
        mask_matrix = torch.full((batch_size, num_items), False, dtype=torch.bool)
        for i in range(batch_size):
            possible_indices = np.arange(1, num_items)
            num_choices = 1 if mask_item_length <= 1 else np.random.randint(1, mask_item_length)
            selected_indices = np.random.choice(possible_indices, num_choices, replace=False)
            mask_matrix[i, selected_indices] = True
            index_matrix[i, -len(selected_indices) :] = torch.tensor(np.sort(selected_indices))
        masked_interaction = interaction.clone()
        masked_interaction[mask_matrix] = self.n_items
        return masked_interaction, index_matrix

    def forward(self, interaction):
        """Synthetic data input"""
        if isinstance(interaction, torch.Tensor) and interaction.dim() == 3:
            position_ids = torch.arange(interaction.size(1), dtype=torch.long, device=interaction.device)
            position_ids = position_ids.unsqueeze(0).expand_as(interaction[:, :, 0])
            masked_item_seq, masked_index = self.mask_interaction(position_ids)

            position_embedding = self.position_embedding(position_ids)
            item_emb = interaction @ self.item_embedding.weight[: self.n_items, :]

            # Replace masked emb and save pos emb
            pos_items = torch.zeros(
                (masked_index.size(0), masked_index.size(1), interaction.size(-1)),
                dtype=item_emb.dtype,
                device=item_emb.device,
            )
            masked_emb = self.item_embedding.weight[self.n_items].view(1, -1)
            for raw in range(item_emb.size(0)):
                for idx, col in enumerate(masked_index[raw]):
                    pos_items[raw, idx] = interaction[raw, col]
                    if col != 0:
                        item_emb[raw, col] = masked_emb

            input_emb = item_emb + position_embedding
            input_emb = self.LayerNorm(input_emb)
            # input_emb = self.dropout(input_emb)

            extended_attention_mask = self.get_attention_mask(masked_item_seq, bidirectional=True)
            trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
            ffn_output = self.output_ffn(trm_output[-1])
            ffn_output = self.output_gelu(ffn_output)
            output = self.output_ln(ffn_output)

            pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
            # [B mask_len] -> [B mask_len max_len] multi hot
            pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1).to(
                output.device
            )  # [B mask_len max_len]
            # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
            # only calculate loss for masked position
            seq_output = torch.bmm(pred_index_map, output)  # [B mask_len H]

            """ KL-Div Loss """
            test_item_emb = self.item_embedding.weight[: self.n_items]  # [item_num H]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) + self.output_bias  # [B mask_len item_num]
            targets = (masked_index > 0).float().view(-1).to(logits.device)  # [B*mask_len]

            log_probs = F.log_softmax(logits, dim=-1)
            kl_div_loss = F.kl_div(log_probs, pos_items, reduction="none").sum(dim=-1).view(-1)
            loss = torch.sum(kl_div_loss * targets) / torch.sum(targets)

        """Real data input"""
        if not (isinstance(interaction, torch.Tensor) and interaction.dim() == 3):
            loss = self.calculate_loss(interaction)

        return None, loss

    def _forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        return output  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction):
        masked_item_seq = interaction[self.MASK_ITEM_SEQ]  # [B, MAX_ITEM_LIST_LENGTH], using "1683" as mask
        pos_items = interaction[self.POS_ITEMS]  # [B, mask_ratio * MAX_ITEM_LIST_LENGTH]
        neg_items = interaction[self.NEG_ITEMS]  # [B, mask_ratio * MAX_ITEM_LIST_LENGTH]
        masked_index = interaction[self.MASK_INDEX]  # [B, mask_ratio * MAX_ITEM_LIST_LENGTH]

        seq_output = self._forward(masked_item_seq)  # [B, MAX_ITEM_LIST_LENGTH, hidden_size]
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, seq_output)  # [B mask_len H]

        if self.loss_type == "BPR":
            pos_items_emb = self.item_embedding(pos_items)  # [B mask_len H]
            neg_items_emb = self.item_embedding(neg_items)  # [B mask_len H]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1) + self.output_bias[pos_items]  # [B mask_len]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1) + self.output_bias[neg_items]  # [B mask_len]
            targets = (masked_index > 0).float()
            loss = -torch.sum(torch.log(1e-14 + torch.sigmoid(pos_score - neg_score)) * targets) / torch.sum(targets)
            return loss

        elif self.loss_type == "CE":
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            test_item_emb = self.item_embedding.weight[: self.n_items]  # [item_num H]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) + self.output_bias  # [B mask_len item_num]
            targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

            loss = torch.sum(
                loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets
            ) / torch.sum(targets)
            return loss
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self._forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_item_emb = self.item_embedding(test_item)
        scores = (torch.mul(seq_output, test_item_emb)).sum(dim=1) + self.output_bias[test_item]  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_seq = self.reconstruct_test_data(item_seq, item_seq_len)
        seq_output = self._forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B H]
        test_items_emb = self.item_embedding.weight[: self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1)) + self.output_bias  # [B, item_num]
        return scores


class GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, interaction):
        if isinstance(interaction, torch.Tensor) and interaction.dim() == 3:
            device = interaction.device
            seq_num = interaction.size(0)
            seq_len = interaction.size(1)  # total seq len, last item as target pos item

            item_seq_len = torch.full((seq_num,), seq_len - 1, device=device)

            item_emb = interaction @ self.item_embedding.weight
            input_emb = item_emb[:, :-1, :]
            input_emb_dropout = self.emb_dropout(input_emb)
            gru_output, _ = self.gru_layers(input_emb_dropout)
            gru_output = self.dense(gru_output)
            seq_output = self.gather_indexes(gru_output, item_seq_len - 1)

            pos_prob = interaction[:, -1, :]
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            loss = F.kl_div(F.log_softmax(logits + 1e-9, dim=-1), pos_prob + 1e-9, reduction="batchmean")

        else:
            loss = self.calculate_loss(interaction)

        return None, loss

    def _forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self._forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self._forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self._forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
