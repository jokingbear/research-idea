import json

import torch
import torch.nn as nn
import torch.optim as opts

from ..hub import get_hub_entries
from .losses import weighted_bce
from .metrics import fb_fn, acc_fn
from .callbacks import LrFinder, SuperConvergence, CSVLogger, Tensorboard
from .trainers.standard_trainer import StandardTrainer
from pathlib import Path


class ConfigRunner:

    def __init__(self, config_file, verbose=1):
        if isinstance(config_file, dict):
            config = config_file
        else:
            with open(config_file) as handle:
                config = json.load(handle)

        repo_config = config["repo"]
        model_config = config["model"]
        loss_config = config["loss"]
        metrics_configs = config.get("metrics", [])
        opt_config = config.get("opt", {"name": "SGD",
                                        "lr": 1e-1, "momentum": 9e-1,
                                        "weight_decay": 1e-6, "nesterov": True})
        callbacks_configs = config.get("callbacks", [])
        trainer_config = config.get("trainer", {"name": "standard"})

        print("creating train, valid loader") if verbose else None
        self.train, self.valid = self._get_repo(repo_config)
        if verbose:
            print("train: ", len(self.train))
            print("valid: ", len(self.valid)) if self.valid is not None else None

        print("creating model") if verbose else None
        self.model = self._get_model(model_config)
        print(self.model) if verbose else None

        self.loss = self._get_loss(loss_config)
        self.metrics = self._get_metrics(metrics_configs)
        self.optimizer = self._get_optimizer(opt_config)
        self.trainer = self._get_trainer(trainer_config)
        self.callbacks = self._get_callbacks(callbacks_configs)

    def _get_repo(self, repo_config):
        repo_path, repo_module = self.get_module_name(repo_config["path"])
        repo_entries = get_hub_entries(repo_path, repo_module)

        entries = repo_entries.list()
        kwargs = self.get_kwargs(repo_config, ["path", "method"])

        if "method" in repo_config:
            method = repo_config["method"]
        elif "train_valid" in entries:
            method = "train_valid"
        elif "train" in entries:
            method = "train"
        else:
            raise NotImplementedError("repo need to have train_valid or train method for empty name config")

        loaders = repo_entries.load(method, **kwargs)

        if isinstance(loaders, (tuple, list)):
            train, valid = loaders
        else:
            train = loaders
            valid = None

        return train, valid

    def _get_model(self, model_config):
        model_path, model_module = self.get_module_name(model_config["path"])
        model_entries = get_hub_entries(model_path, model_module)

        kwargs = self.get_kwargs(model_config, ["path", "name", "parallel", "checkpoint"])
        name = model_config["name"]
        model = model_entries.load(name, **kwargs)

        if "checkpoint" in model_config:
            w = torch.load(model_config["checkpoint"], map_location="cpu")
            print(model.load(w, strict=False))

        if model_config.get("parallel", False):
            model = nn.DataParallel(model).cuda()
        elif model_config.get("gpu", False):
            model = model.cuda()

        return model

    def _get_loss(self, loss_config):
        name = loss_config["name"]
        kwargs = self.get_kwargs(loss_config, ["name", "path"])

        if "path" in loss_config:
            loss_path, loss_module = self.get_module_name(loss_config["path"])
            entries = get_hub_entries(loss_path, loss_module)
            loss = entries.load(name, **kwargs)
        elif name.lower() == "bce":
            loss = nn.BCELoss(**kwargs)
        elif name.lower() == "bce_logit":
            loss = nn.BCEWithLogitsLoss(**kwargs)
        elif name.lower() in {"cb_bce", "wbce"}:
            loss = weighted_bce(**kwargs)
        else:
            raise NotImplementedError("currently only support bce, bce_logit, cb_bce and wbce")

        return loss

    def _get_metrics(self, metrics_configs):
        if isinstance(metrics_configs, list):
            metrics = []

            for m_cfg in metrics_configs:
                name = m_cfg["name"].lower()
                kwargs = self.get_kwargs(m_cfg)

                if name == "f1_score":
                    metrics.append(fb_fn(beta=1, **kwargs))
                elif name == "f_score":
                    metrics.append(fb_fn(**kwargs))
                elif name in {"acc", "accuracy"}:
                    metrics.append(acc_fn(**kwargs))
        else:
            path, name = self.get_module_name(metrics_configs["path"])
            entries = get_hub_entries(path, name)

            name = metrics_configs["name"]
            kwargs = self.get_kwargs(metrics_configs, excludes=["name", "path"])

            metrics = entries.load(name, **kwargs)

        return metrics

    def _get_optimizer(self, opt_config):
        name = opt_config["name"]
        kwargs = self.get_kwargs(opt_config)

        if name.lower() == "sgd":
            opt = opts.SGD
        elif name.lower() == "adam":
            opt = opts.Adam
        else:
            raise NotImplementedError("only support adam and sgd")

        opt = opt(self.model.parameters(), **kwargs)

        return opt

    def _get_trainer(self, trainer_config):
        path = trainer_config.get("path", None)
        name = trainer_config.get("name", "standard").lower()
        kwargs = self.get_kwargs(trainer_config, excludes=["name", "path"])

        if path is not None:
            trainer_path, trainer_module = self.get_module_name(path)
            entries = get_hub_entries(trainer_path, trainer_module)
            trainer = entries.load(name, self.model, self.optimizer, self.loss, metrics=self.metrics, **kwargs)
        elif name == "standard":
            trainer = StandardTrainer(self.model, self.optimizer, self.loss, metrics=self.metrics, **kwargs)
        else:
            raise not NotImplementedError("only support standard trainer for empty trainer config")

        return trainer

    def _get_callbacks(self, callbacks_configs):
        cbs = []

        for cb_config in callbacks_configs:
            name = cb_config["name"].lower()

            kwargs = self.get_kwargs(cb_config)
            if name == "lrfinder":
                cbs.append(LrFinder(**kwargs))
            elif name == "superconvergence":
                cbs.append(SuperConvergence(**kwargs))
            elif name == "csvlogger":
                cbs.append(CSVLogger(**kwargs))
            elif name == "tensorboard":
                cbs.append(Tensorboard(**kwargs))

        return cbs

    def run(self):
        """
        run the training process based on config
        """
        self.trainer.fit(self.train, self.valid, callbacks=self.callbacks)

    def __call__(self, *args, **kwargs):
        self.run()

    @staticmethod
    def get_kwargs(configs, excludes=("name",)):
        excludes = set(excludes)

        return {k: configs[k] for k in configs if k not in excludes}

    @staticmethod
    def get_module_name(path):
        path = Path(path)
        name = path.name

        if ".npy" in name:
            name = name[:-4]

        parent_path = str(path.parent)

        return parent_path, name
