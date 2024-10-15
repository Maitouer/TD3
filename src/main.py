import glob
import json
import logging
import os
import warnings
from dataclasses import dataclass
from functools import wraps

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from recbole.config import Config as RecBoleConfig
from recbole.utils import init_seed
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb
from data import DataConfig, DataModule
from distilled_data import DistilledData, DistilledDataConfig, LearnerTrainConfig
from evaluator import EvaluateConfig, Evaluator
from model import LearnerModel, ModelConfig, SASRec
from pretrainer import PretrainTrainer
from trainer import TrainConfig, Trainer
from utils import FeatureDataLoader

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    seed: int
    gpu_id: int
    experiment_name: str
    method: str
    run_name: str
    data_dir_root: str
    save_dir_root: str
    save_method_dir: str
    save_dir: str


@dataclass
class Config:
    base: BaseConfig
    data: DataConfig
    model: ModelConfig
    distilled_data: DistilledDataConfig
    learner_train: LearnerTrainConfig
    train: TrainConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def wandb_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: Config, *args, **kwargs):
        # Initialize wandb run
        wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        wandb.init(project=config.base.experiment_name, name=config.base.run_name)
        wandb.run.log_code("./", include_fn=lambda path: path.endswith((".py", ".ipynb", ".sh")))
        # Add hydra config
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
        for file in hydra_config_files:
            wandb.save(file)
        with logging_redirect_tqdm():
            out = func(config, *args, **kwargs)
        # Add main.log
        wandb.save(os.path.join(output_dir, "main.log"))
        wandb.finish()
        return out

    return wrapper


@hydra.main(config_path="../config", config_name="default", version_base=None)
@wandb_start_run_with_hydra
def main(config: Config):
    # Load configs
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Set seed and cuda device
    init_seed(config.base.seed, reproducibility=True)
    torch.cuda.set_device(f"cuda:{config.base.gpu_id}")

    # DataModule
    logger.info(f"Loading datasets: (`{config.data.dataset_name}`)")
    data_module = DataModule(config.data)

    # Model Pretrain
    for model_name, model_config in {**config.model.learner_model}.items():
        model_config = OmegaConf.to_container(model_config, resolve=True)
        recbole_config = RecBoleConfig(
            model=model_name,
            dataset=config.data.dataset_name,
            config_dict={
                **config.base,
                **config.data,
                **model_config,
            },
        )
        model = SASRec(recbole_config, data_module.dataset).cuda()
        ddppretrainer = PretrainTrainer(recbole_config, model)
        if not os.path.exists(model_config["pretrained_path"]):
            logger.info(f"Pretraining learner model: (`{model_name}`)")
            if not os.path.exists(model_config["checkpoint_dir"]):
                os.makedirs(model_config["checkpoint_dir"], exist_ok=True)
            (interactions, features), _, _ = ddppretrainer.pretrain(
                train_data=data_module.train_loader,
                valid_data=data_module.valid_loader,
                saved=True,
            )
        else:
            logger.info("Load pretrained learner model")
            ddppretrainer.resume_checkpoint(ddppretrainer.saved_model_file)
            (interactions, features), _ = ddppretrainer.generate_feature(train_data=data_module.train_loader)
        result = ddppretrainer.evaluate(data_module.test_loader, show_progress=True)
        print(result)
        # empty cuda cache
        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        data_module.train_loader = FeatureDataLoader(
            recbole_config,
            interactions,
            features,
            config.train.batch_size,
            data_module.train_loader.sampler,
        )

    # Build learner model and evaluator
    learners = []
    evaluators = []
    for model_name, model_config in {**config.model.learner_model}.items():
        model_config = OmegaConf.to_container(model_config, resolve=True)
        recbole_config = RecBoleConfig(
            model=model_name,
            dataset=config.data.dataset_name,
            config_dict={
                **config.base,
                **config.data,
                **config.model,
                **model_config,
            },
        )
        model = LearnerModel(model_name, recbole_config, data_module.dataset)
        evaluator = Evaluator(config.evaluate, config.train, recbole_config, model=model)
        learners.append(model)
        evaluators.append(evaluator)

    # for cross-net testing
    # model_save_path = (
    #     "./checkpoint/SASRec-music.heads_1.layers_1.hiddens_32.inners_64.maxlen_50.batch_2048.epochs_100.pth"
    # )
    # checkpoint = torch.load(model_save_path)
    # initial_state_dict = checkpoint["state_dict"]
    # item_emb = initial_state_dict["item_embedding.weight"].cpu().numpy().tolist()
    # with torch.no_grad():
    #     learners[0].model.item_embedding.weight[: learners[0].model.n_items, :] = initial_state_dict[
    #         "item_embedding.weight"
    #     ].to(learners[0].model.item_embedding.weight.device)

    # Distilled data
    if config.distilled_data.pretrained_data_path is not None:
        distilled_data = DistilledData.from_pretrained(
            save_path=config.distilled_data.pretrained_data_path,
            item_emb=learners[0].model.item_embedding.weight.data.cpu().numpy().tolist(),
        )
    else:
        distilled_data = DistilledData(
            config=config.distilled_data,
            train_config=config.learner_train,
            item_num=learners[0].model.n_items,
            item_emb=learners[0].model.item_embedding.weight.data.cpu().numpy().tolist(),
        )

    # Train distilled data
    if not config.train.skip_train:
        trainer = Trainer(config.train)
        trainer.fit(
            distilled_data=distilled_data,
            model=learners[0],
            train_loader=data_module.train_loader,
            valid_loader=data_module.valid_loader,
            evaluators=evaluators,
        )

    # Evaluate
    total_results = {}
    for idx, evaluator in enumerate(evaluators):
        results = evaluator.evaluate(
            distilled_data,
            eval_loader=data_module.test_loader,
            verbose=True,
        )
        original_keys = list(results.keys())
        for key in original_keys:
            results[f"{idx}-{key}"] = results.pop(key)
        total_results.update(results)
        wandb.log({f"avg.{k}": v[0] for k, v in results.items()})
        wandb.log({f"std.{k}": v[1] for k, v in results.items()})

    results = {k: f"{v[0]}±{v[1]}" for k, v in total_results.items()}
    logger.info(f"Final Results: {results}")
    save_path = os.path.join(config.base.save_dir, "results.json")
    json.dump(results, open(save_path, "w"))
    wandb.save(save_path)

    return


if __name__ == "__main__":
    main()
