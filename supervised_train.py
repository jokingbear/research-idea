import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as opts

from plasma.hub import get_hub_entries
from plasma.training import trainers, losses, metrics, callbacks


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

    repo_kwargs = {k: repo_config[k] for k in repo_config if k not in {"path", "method"}}

    if "method" in repo_config:
        train_valid = repo.load(repo["method"], **repo_kwargs)
        if isinstance(train_valid, (tuple, list)):
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
    check_dictionary(model_config, "path", "method")
    model_entries = get_entries(model_config["path"])

    model_kwargs = {k: model_config[k] for k in model_config if k not in {"path", "method", "checkpoint"}}
    model_obj = model_entries.load(model_config["method"], **model_kwargs)

    if "checkpoint" in model_config:
        print(model_obj.load_state_dict(torch.load(model_config["checkpoint"], map_location="cpu")))

    if torch.cuda.device_count() > 1:
        model_obj = nn.DataParallel(model_obj)

    if torch.cuda.device_count() > 0:
        model_obj = model_obj.cuda()

    return model_obj


def get_optimizer(optimizer_config, model_obj):
    check_dictionary(optimizer_config, "name")

    opt_class = optimizer_config["name"]
    opt_kwargs = {k: optimizer_config[k] for k in optimizer_config if k not in {"name", "checkpoint"}}

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
    check_dictionary(loss_config, "name")

    value = loss_config["name"]
    loss_kwargs = {k: loss_config[k] for k in loss_config if k not in {"name"}}
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

        device = "cpu" if torch.cuda.device_count() == 0 else "cuda:0"
        weights = torch.tensor(weights, dtype=torch.float, device=device)
        loss_obj = losses.weighted_bce(weights, loss_kwargs.get("smooth", None))
    elif value == "category":
        loss_obj = nn.CrossEntropyLoss()
    else:
        raise Exception("only support for bce, wbce, and category")

    return loss_obj


def get_metrics(metrics_configs):
    metric_objs = []

    for metric_conf in metrics_configs:
        value = metric_conf["name"]
        kwargs = {k: metric_conf[k] for k in metric_conf if k not in {"name"}}

        if "f" in value:
            metric_objs.append(metrics.fb_fn(**kwargs))

    return metric_objs


def get_callbacks(callbacks_configs):
    callback_objs = []

    for cb in callbacks_configs:
        value = cb["name"]
        kwargs = {k: cb[k] for k in cb if k not in {"name"}}

        if value in {"lr finder", "lr_finder"}:
            callback_objs.append(callbacks.LrFinder(**kwargs))
        elif value in {"super convergence", "super_convergence"}:
            callback_objs.append(callbacks.SuperConvergence(**kwargs))
        elif value in {"csv logger", "csv_logger"}:
            callback_objs.append(callbacks.CSVLogger(**kwargs))
        elif value == "tensorboard":
            callback_objs.append(callbacks.Tensorboard(**kwargs))
        else:
            raise Exception("only support for lr finder, super convergence, csv logger and tensorboard")

    return callback_objs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plasma supervised training script')
    parser.add_argument("--train-configs", help="path to train config json")

    args = parser.parse_args()
    config_file = args.train_configs

    with open(config_file, "r") as handle:
        config = json.load(handle)

    check_dictionary(config, "repo", "model", "loss")

    train_ds, valid_ds = get_train_valid(config["repo"])
    print("train: ", len(train_ds))
    print("valid: ", len(valid_ds))

    model = get_model(config["model"])
    print(model)

    opt = get_optimizer(config.get("optimizer", {"name": "SGD",
                                                 "lr": .1,
                                                 "momentum": .9,
                                                 "nesterov": True,
                                                 "weight_decay": 1e-6}), model)
    print(opt)

    loss = get_loss(config["loss"])
    metric_fns = get_metrics(config.get("metrics", []))

    device = "cpu" if torch.cuda.device_count() == 0 else "cuda:0"
    x_type = config.get("x_type", None)
    y_type = config.get("y_type", None)

    trainer = trainers.Trainer(model, opt, loss, metric_fns,
                               x_device=device, x_type=x_type, y_device=device, y_type=y_type)
    cbs = get_callbacks(config.get("callbacks", []))

    train_batch = config.get("train_batch", 128)
    train_loader = train_ds.get_torch_loader(batch_size=train_batch)

    if valid_ds is not None:
        valid_loader = valid_ds.get_torch_loader(batch_size=config.get("valid_batch", train_batch), drop_last=True)
    else:
        valid_loader = None

    trainer.fit(train_loader, valid_loader, callbacks=cbs)
