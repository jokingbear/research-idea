import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as opts

from .callbacks import __mapping__ as callback_map
from .losses import __mapping__ as loss_maps
from .metrics import __mapping__ as metric_maps
from .trainers.standard_trainer import StandardTrainer
from ..hub import get_hub_entries
from .optimizers import __mapping__ as optimizer_map


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
        opt_config = config.get("optimizer", {"name": "SGD",
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
        print("loss: ", self.loss) if verbose else None

        self.metrics = self._get_metrics(metrics_configs)
        print("metrics: ", self.metrics) if verbose else None

        self.optimizer = self._get_optimizer(opt_config)
        print("optimizer: ", self.optimizer) if verbose else None

        self.trainer = self._get_trainer(trainer_config)
        self.callbacks = self._get_callbacks(callbacks_configs)

    def _get_repo(self, repo_config):
        repo_path, repo_module = self.get_module_name(repo_config["path"])
        repo_entries = get_hub_entries(repo_path, repo_module)

        entries = repo_entries.list()
        kwargs = self.get_kwargs(repo_config, ["name", "method", "path"])

        if "name" in repo_config:
            method = repo_config["name"]
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
        name = loss_config["name"].lower()
        kwargs = self.get_kwargs(loss_config, ["name", "path"])

        if "path" in loss_config:
            loss_path, loss_module = self.get_module_name(loss_config["path"])
            entries = get_hub_entries(loss_path, loss_module)
            loss = entries.load(name, **kwargs)
        elif name in loss_maps:
            loss = loss_maps[name](**kwargs)
        else:
            raise NotImplementedError(f"currently only support {loss_maps.keys()}")

        return loss

    def _get_metrics(self, metrics_configs):
        if isinstance(metrics_configs, list):
            metrics = []

            for m_cfg in metrics_configs:
                name = m_cfg["name"].lower()
                kwargs = self.get_kwargs(m_cfg)

                if name in metric_maps:
                    metrics.append(metric_maps[name](**kwargs))
        else:
            path, name = self.get_module_name(metrics_configs["path"])
            entries = get_hub_entries(path, name)

            name = metrics_configs["name"]
            kwargs = self.get_kwargs(metrics_configs, excludes=["name", "path"])

            metrics = entries.load(name, **kwargs)

        return metrics

    def _get_optimizer(self, opt_config):
        name = opt_config["name"].lower()
        kwargs = self.get_kwargs(opt_config)

        if name in optimizer_map:
            opt = optimizer_map[name]
        else:
            raise NotImplementedError(f"only support {optimizer_map.keys()}")

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

            if name in callback_map:
                cbs.append(callback_map[name](**kwargs))

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
        name = path.name.replace(".py", "")
        parent_path = str(path.parent)

        return parent_path, name
