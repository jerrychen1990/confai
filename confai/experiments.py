# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     experiments
   Description :
   Author :       chenhao
   date：          2021/4/6
-------------------------------------------------
   Change Activity:
                   2021/4/6:
-------------------------------------------------
"""
import logging
import os
import random
from abc import ABCMeta
from typing import Type

from snippets import jload_lines, jdump_lines

from confai.evaluate import *
from confai.models import get_model_cls
from confai.models.nn_core import NNModel, NNModelConfig
from confai.models.schema import *
from confai.output import get_output_on_task
from confai.utils import print_info, jdumps, get_current_time_str, jdump, random_split_list, read_config

logger = logging.getLogger(__name__)


class CommonConfig(BaseModel):
    project_name: str = Field(
        descriptiion="The name of the project, used as a directory name to save the experiment results"
    )
    owner: str = Field(
        descriptiion="The name of the owner of experiment"
    )
    desc: str = Field(
        description="The description of the experiment",
    )
    model_cls: str = Field(
        descriptiion="The model class name, e.g. ClsTokenClsModel",
    )
    model_name: str = Field(
        descriptiion="The name of the model, used as a directory name to save the experiment results",
    )
    ckpt_path: str = Field(
        default=None,
        description="The path of the checkpoint file, if not provided, the model will be trained from scratch",
    )
    experiment_dir: str = Field(
        default=os.environ["EXPERIMENT_HOME"],
        descriptiion="the root directory to save experiment results"
    )

    is_train: bool = Field(
        default=True,
        descriptiion="whether to train the model"
    )
    is_test: bool = Field(
        default=True,
        descriptiion="whether to test the model"
    )
    is_save: bool = Field(
        default=True,
        descriptiion="whether to save the model"
    )
    save_args: dict = Field(
        default=None,
        descriptiion="the args to save the model, if is_save is True, this Field is need"
    )
    is_overwrite_experiment: bool = Field(
        default=False,
        descriptiion=("whether to overwrite the experiment results"
                      "is_overwrite_experiment=True：experiment目录为{experiment_dir}/{project_name}/{"
                      "model_name",
                      "is_overwrite_experiment=False：experiment目录为"
                      "{experiment_dir}/{project_name}/{model_name}-{current_time}")
    )
    output_phase_list: List[str] = Field(
        default=("train", "eval", "test"),
        descriptiion="which datasets will generate output"
    )

    eval_phase_list: List[str] = Field(
        default=("train", "eval", "test"),
        descriptiion="which datasets will be evaluated"
    )
    seed: int = Field(
        default=random.randint(0, 1e9),
        descriptiion="random seed, for reproducibility"
    )


class DataConfig(BaseModel):
    dataset_name: str = Field(
        default=None,
        descriptiion="The name of the dataset",
    )
    eval_rate: float = Field(
        default=0.05,
        descriptiion="The rate of the eval dataset"
    )
    train_data_path: PathOrPaths = Field(
        default=None,
        descriptiion="The path or paths of train_data"

    )
    eval_data_path: PathOrPaths = Field(
        default=None,
        descriptiion="The path or paths of train_data"

    )
    test_data_path: PathOrPaths = Field(
        default=None,
        descriptiion="The path or paths of train_data"

    )


class ExperimentConfig(BaseModel):
    common_config: CommonConfig = Field(descriptiion="The common config of the experiment")
    data_config: DataConfig = Field(descriptiion="the data config")
    task_config: dict = Field(default_factory=lambda: {}, descriptiion="the special task config")
    train_config: dict = Field(default_factory=lambda: {}, descriptiion="the train config")
    test_config: dict = Field(default_factory=lambda: {}, descriptiion="the test config")
    nn_model_config: dict = Field(default_factory=lambda: {}, descriptiion="the nn_model config")
    tokenizer_config: dict = Field(default_factory=lambda: {}, descriptiion="the tokenizer config")

    callback_config: dict = Field(default_factory=lambda: {}, descriptiion="the callback config")


def star_print(info):
    return print_info(info, target_logger=logger, fix_length=128)


def get_model_config(experiment_config: ExperimentConfig) -> NNModelConfig:
    config_keys = ['task_config', 'nn_model_config', "tokenizer_config"]
    model_config = {k: v for k, v in experiment_config.__dict__.items() if k in config_keys}
    model_config.update(model_name=experiment_config.common_config.model_name)
    model_config.update(model_cls=experiment_config.common_config.model_cls)
    return NNModelConfig(**model_config)


# 基础实验类
class Experiment(metaclass=ABCMeta):
    # 通过config初始化实验
    def __init__(self, config: ExperimentConfig):
        self.config = config
        star_print("experiment config")
        logger.info(jdumps(config))

        # init model cls
        self.common_config = config.common_config
        self.data_config = config.data_config

        self.model_cls: Type[NNModel] = get_model_cls(self.common_config.model_cls)
        self.model: Optional[NNModel] = None
        self.task: Task = self.model_cls.task
        self.train_data_path, self.eval_data_path, self.test_data_path = None, None, None

        # update paths
        exp_name = f"{self.common_config.model_name}-{self.common_config.owner}"
        if not self.common_config.is_overwrite_experiment:
            exp_name = f"{exp_name}-{get_current_time_str(fmt='%Y-%m-%d-%H-%M-%S')}"
        self.experiment_path = os.path.join(self.common_config.experiment_dir, self.common_config.project_name,
                                            exp_name)
        self.model_path = os.path.join(self.experiment_path, "model")
        self.eval_path = os.path.join(self.experiment_path, "eval")
        self.output_path = os.path.join(self.experiment_path, "output")
        self.log_path = os.path.join(self.experiment_path, "log")

    # 初始化数据集
    def initialize_dataset(self):
        star_print("data initialize phase start")
        if self.data_config.dataset_name:
            dataset_path = os.path.join(os.environ["DATA_HOME"], self.task.name.lower(), self.data_config.dataset_name)
            if not os.path.exists(dataset_path):
                raise ValueError(f"dataset {dataset_path} not exists")

            logger.info(f"reading data from dataset path:{dataset_path}")
            train_data_path = os.path.join(dataset_path, "train.jsonl")
            if not train_data_path:
                raise ValueError(f"train data not exists in dataset {train_data_path}")

            label_path = os.path.join(dataset_path, "labels.txt")
            if os.path.exists(label_path):
                self.config.task_config["label_path"] = label_path

            eval_data_path = os.path.join(dataset_path, "eval.jsonl")
            if not os.path.exists(eval_data_path):
                logger.info(f"no eval data in dataset, will split {self.data_config.eval_rate} from train data")
                train_data = jload_lines(train_data_path)
                train_data, eval_data = random_split_list(train_data, 1. - self.data_config.eval_rate)
                cache_dir = os.environ.get("CONFAI_CACHE", os.path.join(os.environ["HOME"], ".waf_cache"))
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)

                self.train_data_path = os.path.join(cache_dir, f"{self.data_config.dataset_name}-"
                                                               f"{self.common_config.seed}-"
                                                               f"{self.data_config.eval_rate}-train")
                self.eval_data_path = os.path.join(cache_dir, f"{self.data_config.dataset_name}-"
                                                              f"{self.common_config.seed}-"
                                                              f"{self.data_config.eval_rate}-eval")

                logger.info(f"{len(train_data)} to {self.train_data_path}")
                logger.info(f"{len(eval_data)} to {self.eval_data_path}")

                jdump_lines(train_data, self.train_data_path)
                jdump_lines(eval_data, self.eval_data_path)
            else:
                self.train_data_path, self.eval_data_path = train_data_path, eval_data_path

            self.test_data_path = [os.path.join(dataset_path, t) for t in os.listdir(dataset_path) if "test" in t]
        else:
            self.train_data_path, self.eval_data_path, self.test_data_path = self.data_config.train_data_path, \
                                                                             self.data_config.eval_data_path, \
                                                                             self.data_config.test_data_path

        star_print("data initialize phase end")

    # 初始化要续联的模型
    def initialize_model(self):
        star_print("model initialize phase start")
        # 加载ckpt中的模型
        if self.common_config.ckpt_path:
            star_print("initialize nn_model from checkpoint")
            self.model = self.model_cls.load(path=self.common_config.ckpt_path, load_nn_model=True)
        # 根据配置，新建一个模型
        else:
            star_print("initialize model from model config")
            model_config = get_model_config(self.config)
            self.model = self.model_cls(config=model_config)
            self.model.build_model(**self.config.nn_model_config)
        star_print("model initialize phase end")

    def _update_callback_kwargs(self):
        for k, v in self.config.callback_config.items():
            if k == "eval_save":
                v.update(model=self.model, eval_data=self.eval_data_path, test_config=self.config.test_config,
                         experiment_path=self.experiment_path)

    # 训练模型
    def train_model(self):
        star_print("training phase start")
        assert self.model is not None, f"nn_model is not initialized, please call initialize_model first!"

        # set output_dir for huggingface trainer
        if "output_dir" not in self.config.train_config:
            self.config.train_config["output_dir"] = os.path.join(self.log_path, f"hf_output/")
        logger.info(f"training with {self.config.train_config=}")
        self._update_callback_kwargs()
        self.model.train(train_data=self.train_data_path,
                         eval_data=self.eval_data_path,
                         callback_kwargs=self.config.callback_config,
                         train_kwargs=self.config.train_config)
        star_print("training phase end")

    # 保存模型
    def save_model(self):
        star_print("saving phase start")
        self.model.save(path=self.model_path, **self.common_config.save_args)
        star_print("saving phase end")

    def test_on_data_path(self, data_path: str, output_path: str = None, eval_path: str = None):

        preds = self.model.predict(data=data_path, **self.config.test_config)
        examples: List[Example] = list(self.model.read_examples(data_path))
        output_rs, eval_rs = None, None
        if output_path:
            output_rs = get_output_on_task(examples, preds, self.task)
            logger.info(f"output pred :{len(output_rs)} result to {output_path}")
            jdump(output_rs, output_path)
        if eval_path:
            try:
                if examples[0].get_ground_truth() is None:
                    logger.warning(f"can't eval on data without ground truth!")
                else:
                    eval_rs = eval_on_task(examples, preds, self.task)
                    logger.info(jdumps(eval_rs))
                    logger.info("writing eval result to :{}".format(eval_path))
                    jdump(eval_rs, eval_path)
            except ValueError as e:
                logger.warning(e)
        return output_rs, eval_rs

    # 测试模型
    def test_model(self):
        star_print("testing phase start")
        for tag, data_path in zip(['train', 'eval', 'test'],
                                  [self.train_data_path, self.eval_data_path, self.test_data_path]):
            # 不需要eval也不需要output的情况
            if tag not in self.common_config.eval_phase_list and tag not in self.common_config.output_phase_list:
                continue
            if not data_path or not os.path.exists(data_path):
                logger.info(f"data_path {data_path} not exists, will not predict")
                continue
            # todo 处理多个test文件的情况， 暂时不处理
            if isinstance(data_path, list):
                data_path = data_path[0]
            logger.info("predict result on {} data:".format(tag))
            output_path = os.path.join(self.output_path, f"{tag}.json") \
                if tag in self.common_config.output_phase_list else None
            eval_path = os.path.join(self.eval_path, f"{tag}.json") \
                if tag in self.common_config.eval_phase_list else None
            self.test_on_data_path(data_path, output_path, eval_path)
        star_print("testing phase end")

    # 记录实验运行时的config
    def save_config(self):
        config_file_path = os.path.join(self.experiment_path, "config.json")
        star_print(f"saving config file to {config_file_path}")
        jdump(self.config, config_file_path)

    # 运行实验，实验的入口
    def run(self):
        star_print("experiment start")
        self.save_config()
        self.initialize_dataset()
        self.initialize_model()
        if self.common_config.is_train:
            self.train_model()
        if self.common_config.is_save:
            self.save_model()
        if self.common_config.is_test:
            self.test_model()
        star_print("experiment end")


def build_experiment(config_path: str) -> Experiment:
    config = read_config(config_path)
    exp_config = ExperimentConfig(**config)
    # logger.info(exp_config)
    exp = Experiment(exp_config)
    return exp
