#!/usr/bin/env python 
# encoding: utf-8 
"""
@author: chenhao
@file: callbacks.py
@time: 2022/11/11 11:37
"""
import logging
import os
from typing import List

from jsonpath_ng import parse
from snippets import jdump, jdump_lines
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, EarlyStoppingCallback

from confai.evaluate import eval_on_task
from confai.models.nn_core import NNModel
from confai.models.schema import Example

logger = logging.getLogger(__name__)


class EvalSaveCallback(TrainerCallback):
    def __init__(self, model: NNModel, eval_data, test_config: dict, experiment_path: str,
                 metric_jpath=None, greater_is_better=True, tgt_metric=None, load_best_model_at_end=True, patience=1,
                 eval_steps: int = None, eval_epochs: int = None):
        self.model = model
        self.eval_data = eval_data
        self.test_config = test_config
        self.eval_steps = eval_steps
        self.eval_epochs = eval_epochs
        self.experiment_path = experiment_path
        self.metric_jpath = metric_jpath
        self.greater_is_better = greater_is_better
        self.best_metric = None
        self.tgt_metric = tgt_metric
        self.best_model_path = os.path.join(self.experiment_path, "callbacks", "best_model")
        self.statistic_path = os.path.join(self.experiment_path, "callbacks", "eval_save.jsonl")
        self.statistic = list()
        self.load_best_model_at_end = load_best_model_at_end
        self.arrive_tgt = False
        self.origin_patience = patience
        self.patience = patience

        assert eval_steps or eval_epochs, "eval_steps or eval_epochs must be set"

    def do_eval(self, tag):
        preds = self.model.predict(data=self.eval_data, **self.test_config)
        examples: List[Example] = list(self.model.read_examples(self.eval_data))
        eval_rs = eval_on_task(examples, preds, self.model.task)
        eval_path = os.path.join(self.experiment_path, "callbacks", f"{tag}.json")
        logger.info(f"save eval result to {eval_path}")
        jdump(eval_rs, eval_path)
        return eval_rs

    def _is_better(self, m1, m2, allow_equal=False):
        if m1 is None:
            return False
        if m2 is None:
            return True
        if self.greater_is_better and (m1 > m2 or m1 == m2 and allow_equal):
            return True
        if not self.greater_is_better and (m1 < m2 or m1 == m2 and allow_equal):
            return True
        return False

    def do_save(self, eval_rs, tag, statistic):
        jsonpath_expression = parse(self.metric_jpath)
        match = jsonpath_expression.find(eval_rs)
        if not match:
            logger.warning(f"metric_jpath:{self.metric_jpath} not found in eval_rs:{eval_rs}")
        metric = match[0].value
        best_metric_str = f"{self.best_metric:2.3f}" if self.best_metric is not None else "None"

        statistic.update(metric=f"{metric:2.3f}", best_metric=best_metric_str)
        logger.info(f"get metric:{metric:2.3f} of {tag}, best_metric:{best_metric_str}")

        if self._is_better(metric, self.best_metric):
            self.patience = self.origin_patience
            self.best_metric = metric
            logger.info(f"save best model to {self.best_model_path}, with metric:{metric:2.3f}")
            self.model.save(path=self.best_model_path)

            if self.tgt_metric is not None and self._is_better(self.best_metric, self.tgt_metric, allow_equal=True):
                logger.info(f"{self.best_metric:2.3f} is not bad than {self.tgt_metric:2.3f}, stop training")
                self.arrive_tgt = True
        else:
            self.patience -= 1
            logger.info(f" metric:{metric:2.3f} is not better than best_metric:{best_metric_str},"
                        f" patience:{self.patience}")
            if self.patience <= 0:
                logger.info(f"patience is over, stop training")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        tag = f"step-{state.global_step}"
        if not self.eval_epochs and not self.eval_steps:
            self.eval_steps = args.logging_steps

        statistic = dict(epoch=state.epoch, step=state.global_step)

        if self.eval_steps and state.global_step % self.eval_steps == 0:
            logger.info(f"eval model on the end of {tag}")
            eval_rs = self.do_eval(tag)
            if self.metric_jpath:
                self.do_save(eval_rs, tag, statistic)
                self.statistic.append(statistic)

                control.should_training_stop = self.arrive_tgt or self.patience == 0

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        tag = f"epoch-{state.epoch}"
        statistic = dict(epoch=state.epoch, step=state.global_step)

        if self.eval_epochs and state.epoch % self.eval_epochs == 0:
            logger.info(f"eval model on the end of {tag}")
            eval_rs = self.do_eval(tag)
            if self.metric_jpath:
                self.do_save(eval_rs, tag, statistic)
                self.statistic.append(statistic)

                control.should_training_stop = self.arrive_tgt or self.patience == 0

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        logger.info("training is end")
        statistic = dict(epoch=state.epoch, step=state.global_step)

        if not self.arrive_tgt and self.patience > 0:
            tag = f"train_end"
            logger.info(f"eval model on the end of training")
            eval_rs = self.do_eval(tag)
            if self.metric_jpath:
                self.do_save(eval_rs, tag, statistic)
                self.statistic.append(statistic)
        logger.info("save statistics")
        jdump_lines(self.statistic, self.statistic_path)

        if self.load_best_model_at_end:
            logger.info(f"load best model from {self.best_model_path}")
            nn_model_path = os.path.join(self.best_model_path, "nn_model")
            assert os.path.exists(nn_model_path), f"{nn_model_path} not exists"
            self.model.load_nn_model(path=nn_model_path)


_callback_map = {
    "early_stop": EarlyStoppingCallback,
    "eval_save": EvalSaveCallback

}


def _update_callback(train_args: TrainingArguments, callback: TrainerCallback, kwargs):
    if issubclass(callback, EarlyStoppingCallback):
        logger.info("set load_best_model_at_end to True for EarlyStoppingCallback")
        train_args.load_best_model_at_end = True

        metric_for_best_model = kwargs.get("metric_for_best_model", "loss")
        logger.info(f"set metric_for_best_model to {metric_for_best_model} for EarlyStoppingCallback")
        train_args.metric_for_best_model = metric_for_best_model

        evaluation_strategy = kwargs.get("evaluation_strategy", "steps")
        logger.info(f"set evaluation_strategy to {evaluation_strategy} for EarlyStoppingCallback")
        train_args.eval_steps = train_args.logging_steps
        train_args.evaluation_strategy = evaluation_strategy
    if issubclass(callback, EvalSaveCallback):
        if not kwargs.get("eval_steps") and not kwargs.get("eval_epochs"):
            kwargs["eval_steps"] = train_args.logging_steps


def get_callbacks(train_args: TrainingArguments, callback_config: dict):
    callbacks = []
    for callback_name, kwargs in callback_config.items():
        if callback_name not in _callback_map:
            logger.warning(f"callback:{callback_name} not found!")
        else:
            callback_cls = _callback_map[callback_name]
            _update_callback(train_args, callback_cls, kwargs)
            logger.info(f"adding callback :{callback_cls.__name__} with kwargs:{kwargs}")
            callback = callback_cls(**kwargs)
            callbacks.append(callback)

    return callbacks
