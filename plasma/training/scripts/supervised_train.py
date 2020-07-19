import argparse
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as opts

from ..trainers import Trainer
from ..losses import weighted_bce
from ..metrics import fb_fn
from ..callbacks import *
from .utils import *


def check_dictionary(d, *keys):
    assert isinstance(d, dict), f"{d} need to be a dictionary"
    for k in keys:
        assert k in d, f"config doesn't contain {k}"


def get_entries(module_path):
    full_path = Path(module_path)

    return get_hub_entries(full_path.parent, full_path.name)


def get_train_valid(repo_config):
    check_dictionary(repo_config, "path")
    repo = get_entries(repo_config["path"])

    repo_kwargs = repo_config.get("kwargs", {})

    if "method" in repo:
        train_valid = repo.load(repo["method"], **repo_kwargs)
        if isinstance(train_valid, tuple):
            train, valid = train_valid
        else:
            train = train_valid
            valid = None
    else:
        methods = set(repo.list())

        if "train" in methods:
            train = repo.load("train", **repo_kwargs)
            valid = None
        elif "train_valid" in methods:
            train, valid = repo.load("train_valid", **repo_kwargs)
        else:
            raise Exception("need to add method to config file, or repo file must have train or train_valid method")

    return train, valid


def get_model(model_config):
    check_dictionary(model_config, "path")
    model_entries = get_entries(model_config)

    methods = set(model_entries.list())
    model_kwargs = model_config.get("kwargs", {})

    if "class" in model_config:
        model_obj = model_entries.load(model_config["method"], **model_kwargs)
    elif "Classifier" in methods:
        model_obj = model_entries.load("Classifier", **model_kwargs)
    else:
        raise Exception("class must be in model config, or model module must contain class Classifier")

    if "checkpoint" in model_config:
        print(model_obj.load_state_dict(torch.load(model_config["checkpoint"], map_location="cpu")))

    if torch.cuda.device_count() > 1:
        model_obj = nn.DataParallel(model_obj)

    if torch.cuda.device_count() > 0:
        model_obj = model_obj.cuda()

    return model_obj


def get_optimizer(optimizer_config, model_obj):
    check_dictionary(optimizer_config, "class")

    opt_class = optimizer_config["class"]
    opt_kwargs = optimizer_config["kwargs"]

    if opt_class == "SGD":
        opt_obj = opts.SGD(model_obj.parameters(), **opt_kwargs)
    elif opt_class == "Adam":
        opt_obj = opts.Adam(model_obj.parameters(), **opt_kwargs)
    else:
        raise Exception("optimizer only support SGD and Adam")

    if "checkpoint" in optimizer_config:
        print(opt_obj.load_state_dict(torch.load(optimizer_config["checkpoint"], map_location="cpu")))

    return opt_obj


def get_loss(loss_config):
    check_dictionary(loss_config, "value")

    value = loss_config["value"]
    loss_kwargs = loss_config.get("kwargs", {})
    if value == "bce":
        loss_obj = nn.BCELoss(**loss_kwargs)
    elif value == "wbce":
        check_dictionary(loss_kwargs, "weights")

        weights = loss_kwargs["weights"]
        if ".npy" in weights:
            weights = np.load(weights)
            print("weights: ", weights.shape)
        elif ".csv" in weights:
            weights = pd.read_csv(weights, index_col=0)
            print(weights)

        weights = torch.tensor(weights, dtype=torch.float, device="cuda:0")
        loss_obj = weighted_bce(weights, loss_kwargs.get("smooth", None))
    elif value == "category":
        loss_obj = nn.CrossEntropyLoss()
    else:
        raise Exception("only support for bce, wbce, and category")

    return loss_obj


def get_metrics(metrics_configs):
    metric_objs = []

    for metric_conf in metrics_configs:
        value = metric_conf["value"]
        kwargs = metric_conf.get("kwargs", {})

        if "f" in value:
            metric_objs.append(fb_fn(**kwargs))

    return metric_objs


def get_callbacks(callbacks_configs):
    callback_objs = []

    for cb in callbacks_configs:
        value = cb["value"]
        kwargs = cb.get("kwargs", {})

        if value == "lr finder":
            callback_objs.append(LrFinder(**kwargs))
        elif value == "super convergence":
            callback_objs.append(SuperConvergence(**kwargs))
        elif value == "csv logger":
            callback_objs.append(CSVLogger(**kwargs))
        elif value == "tensorboard":
            callback_objs.append(Tensorboard(**kwargs))
        else:
            raise Exception("only support for lr finder, super convergence, csv logger and tensorboard")

    return callback_objs


parser = argparse.ArgumentParser(description='Plasma supervised training script')
parser.add_argument("--train-configs", default=None, help="path to additional train config json")

args = parser.parse_args()
config_file = args.train_configs


with open(config_file, "r") as handle:
    config = json.load(handle)

check_dictionary(config, "repo", "model")

train_ds, valid_ds = get_train_valid(config["repo"])
print("train: ", len(train_ds))
print("valid: ", len(valid_ds))

model = get_model(config["model"])
print(model)

opt = get_optimizer(config.get("optimizer", {"class": "SGD",
                                             "kwargs": {"lr": .1,
                                                        "momentum": .9,
                                                        "nesterov": True,
                                                        "weight_decay": 1e-6}}), model)

loss = get_loss(config["loss"])
metrics = get_metrics(config.get("metrics", []))

trainer = Trainer(model, opt, loss, metrics, x_device="cuda:0", y_device="cuda:0", y_type=torch.float)
cbs = get_callbacks(config.get("callbacks", []))

train_batch = config["train_batch"]
train_loader = train_ds.get_torch_loader(batch_size=config["train_batch"])

if valid_ds is not None:
    valid_loader = valid_ds.get_torch_loader(batch_size=config.get("valid_batch", train_batch), drop_last=True)
else:
    valid_loader = None

trainer.fit(train_loader, valid_loader, callbacks=cbs)
