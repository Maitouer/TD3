import glob
import logging
import os
from dataclasses import dataclass
from functools import wraps

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from recbole.config import Config as RecBoleConfig
from recbole.data import data_preparation
from recbole.data.dataset import SequentialDataset

from recbole.quick_start import run_recbole
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb
from data import DataConfig
from distilled_data import DistilledDataConfig, LearnerTrainConfig
from evaluator import EvaluateConfig
from model import ModelConfig, SASRec
from pretrainer import PretrainTrainer
from trainer import TrainConfig
from utils import average

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    seed: int = 42
    gpu_id: int = 0
    experiment_name: str = ""
    method: str = ""
    run_name: str = ""
    data_dir_root: str = "./data"
    save_dir_root: str = "./save"
    save_method_dir: str = ""
    save_dir: str = ""


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
        # wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        # wandb.init(project=config.base.experiment_name, name="Full-Data", settings=dict(init_timeout=120))
        # wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
        # Add hydra config
        # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        # hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
        # for file in hydra_config_files:
        #     wandb.save(file)
        with logging_redirect_tqdm():
            out = func(config, *args, **kwargs)
        # Add main.log
        # wandb.save(os.path.join(output_dir, "main.log"))
        # wandb.finish()
        return out

    return wrapper


@hydra.main(config_path="../config", config_name="default", version_base=None)
@wandb_start_run_with_hydra
def main(config: Config):
    # # Prepare test dataloader
    # original_recbole_config = RecBoleConfig(
    #     model=config.data.recbole_model,
    #     dataset=config.data.dataset_name,
    #     config_dict=OmegaConf.to_container(config.data.recbole_config, resolve=True),
    # )
    # origin_dataset = SequentialDataset(original_recbole_config)
    # _, _, test_loader = data_preparation(original_recbole_config, origin_dataset)

    # for frac in [0.005, 0.03, 0.1, 0.3, 0.5, 0.8]:
    #     ################
    #     # wandb logging
    #     wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    #     wandb.init(
    #         project=config.base.experiment_name, name=f"Full-Data-Longest-{frac}", settings=dict(init_timeout=120)
    #     )
    #     wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
    #     output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    #     hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
    #     for file in hydra_config_files:
    #         wandb.save(file)
    #     ################
    #     # Model training
    #     total_results = {}
    #     for idx, (model_name, model_config) in enumerate({**config.model.learner_model}.items()):
    #         model_config = OmegaConf.to_container(model_config, resolve=True)
    #         # model_config["eval_args"]["split"] = {"LS": "valid_only"}
    #         dataset = f"{config.data.dataset_name}_{frac}"
    #         recbole_config = RecBoleConfig(
    #             model=model_name,
    #             dataset=dataset,
    #             config_dict={
    #                 **config.base,
    #                 **config.data,
    #                 **model_config,
    #             },
    #         )
    #         sample_dataset = SequentialDataset(recbole_config)
    #         train_loader, valid_loader, _ = data_preparation(recbole_config, sample_dataset)
    #         model = SASRec(recbole_config, origin_dataset).cuda()
    #         ddppretrainer = PretrainTrainer(recbole_config, model)
    #         ddppretrainer.saved_model_file = os.path.join(
    #             model_config["checkpoint_dir"], f"{model_name}-{config.data.dataset_name}-{frac}.pth"
    #         )
    #         ddppretrainer.fit(
    #             train_data=train_loader,
    #             valid_data=valid_loader,
    #             saved=True,
    #             show_progress=True,
    #         )
    #         results = []
    #         for _ in range(1):
    #             result = ddppretrainer.evaluate(test_loader, show_progress=True)
    #             results.append(result)
    #         results = average(results, std=True)
    #         original_keys = list(results.keys())
    #         for key in original_keys:
    #             results[f"{idx}-{key}"] = results.pop(key)
    #         total_results.update(results)
    #         wandb.log({f"avg.{k}": v[0] for k, v in results.items()})
    #         wandb.log({f"std.{k}": v[1] for k, v in results.items()})

    #     results = {k: f"{v[0]}±{v[1]}" for k, v in total_results.items()}
    #     logger.info(f"Baseline Results: {results}")

    #     wandb.save(os.path.join(output_dir, "main.log"))
    #     wandb.finish()

    for frac in [0.005, 0.03, 0.1, 0.3, 0.5, 0.8]:
        wandb.config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        wandb.init(
            project=config.base.experiment_name, name=f"Full-Data-Random-{frac}", settings=dict(init_timeout=120)
        )
        wandb.run.log_code("./", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
        for file in hydra_config_files:
            wandb.save(file)
        total_results = {}
        for idx, (learner_name, learner_config) in enumerate({**config.model.learner_model}.items()):
            learner_config = OmegaConf.to_container(learner_config, resolve=True)
            results = []
            for _ in range(1):
                dataset = f"{config.data.dataset_name}_{frac}"
                result = run_recbole(
                    model=learner_name,
                    dataset=dataset,
                    config_dict={
                        **learner_config,
                        **config.base,
                        **config.data,
                        **config.model,
                    },
                )
                results.append(result["test_result"])
            results = average(results, std=True)
            original_keys = list(results.keys())
            for key in original_keys:
                results[f"{idx}-{key}"] = results.pop(key)
            total_results.update(results)
            wandb.log({f"avg.{k}": v[0] for k, v in results.items()})
            wandb.log({f"std.{k}": v[1] for k, v in results.items()})

        wandb.save(os.path.join(output_dir, "main.log"))
        wandb.finish()

    results = {k: f"{v[0]}±{v[1]}" for k, v in total_results.items()}
    logger.info(f"Baseline Results: {results}")


if __name__ == "__main__":
    main()
