import json
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from .callbacks import __mapping__ as callback_map
from .losses import __mapping__ as loss_maps
from .metrics import __mapping__ as metric_maps
from .trainers import __mapping__ as trainer_maps
from ..hub import get_entries
from .optimizers import __mapping__ as optimizer_map

from distutils.log import warn
from ..functional import run_pipe


class ConfigRunner:

    def __init__(self, config, save_config_path=None, rank=0, ddp=False, verbose=True):
        self.config = config
        self.save_config_path = save_config_path
        self.rank = rank
        self.ddp = ddp
        self.verbose = verbose & rank == 0

        _format_print_dict(config)

        repo_config = config["repo"]
        model_config = config["model"]
        loss_config = config["loss"]
        metrics_configs = config.get("metrics", [])
        opt_config = config.get("optimizer", {"name": "SGD",
                                              "lr": 1e-1, "momentum": 9e-1,
                                              "weight_decay": 1e-6, "nesterov": True})
        callbacks_configs = config.get("callbacks", [])
        trainer_config = config.get("trainer", {"name": "standard"})

        if ddp:
            repo_config['rank'] = rank
            repo_config['num_replicas'] = torch.cuda.device_count()

            model_config['rank'] = rank

            trainer_config['rank'] = rank

        run_pipe({
            'load_repo': (self._set_repo, repo_config),
            'load_model': (self._set_model, model_config),
            'load_loss': (self._set_loss, loss_config),
            'load_metrics': (self._set_metrics, metrics_configs),
            'load_optimizer': (self._set_optimizer, opt_config),
            'load_trainer': (self._set_trainer, trainer_config),
            'load_callbacks': (self._set_callbacks, callbacks_configs),
        })  

    def _set_repo(self, **repo_config):
        repo_entries = get_entries(repo_config["path"])

        entries = repo_entries.list()
        kwargs = self.get_kwargs(repo_config, ["name", "method", "path"])

        if "name" in repo_config:
            method = repo_config["name"]
        elif "train_valid" in entries:
            method = "train_valid"
        elif "train" in entries:
            method = "train"
        else:
            raise NotImplementedError(
                "repo need to have train_valid or train method for empty name config")

        loaders = repo_entries.load(method, **kwargs)

        if isinstance(loaders, (tuple, list)):
            self.train, self.valid = loaders
        else:
            self.train = loaders
            self.valid = None
        
        if self.verbose:
            print("train iterations: ", len(self.train))
            print("valid: ", len(self.valid)) if self.valid is not None else None

    def _set_model(self, **model_config):
        model_entries = get_entries(model_config["path"])

        kwargs = self.get_kwargs(model_config, ["path", "name", "parallel", "checkpoint", "gpu", 'batchnorm'])
        name = model_config["name"]
        model = model_entries.load(name, **kwargs)

        if "checkpoint" in model_config:
            w = torch.load(model_config["checkpoint"], map_location="cpu")
            print(model.load_state_dict(w, strict=False)) if self.rank == 0 else None

        if self.ddp:
            model = model.to(self.rank)

            if model_config.get('batchnorm', False):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])
        elif model_config.get("parallel", False):
            model = nn.DataParallel(model).cuda(self.rank)
        elif model_config.get("gpu", False):
            model = model.cuda(self.rank)

        self.model = model

        if self.verbose:
            num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('model parameters:', num)

            print("printing model to model.txt")
            with open("model.txt", "w") as handle:
                handle.write(str(self.model))

    def _set_loss(self, **loss_config):
        name = loss_config["name"]
        kwargs = self.get_kwargs(loss_config, ["name", "path"])

        if "path" in loss_config:
            entries = get_entries(loss_config["path"])
            loss = entries.load(name, **kwargs)
        elif name in loss_maps:
            loss = loss_maps[name](**kwargs)
        else:
            raise NotImplementedError(
                f"currently only support {loss_maps.keys()}")
        
        self.loss = loss
        print("loss: ", self.loss) if self.verbose else None

    def _set_metrics(self, *metrics_configs):
        metrics = []

        for m_cfg in metrics_configs:
            name = m_cfg["name"].lower()
            kwargs = self.get_kwargs(m_cfg)

            if name in metric_maps:
                metric = metric_maps[name](**kwargs)
            else:
                entries = get_entries(m_cfg["path"])

                kwargs = self.get_kwargs(m_cfg, excludes=["name", "path"])
                metric = entries.load(name, **kwargs)
            
            metrics.append(metric)

        self.metrics = metrics
        print("metrics: ", self.metrics) if self.verbose else None

    def _set_optimizer(self, **opt_config):
        name = opt_config["name"].lower()
        kwargs = self.get_kwargs(opt_config, excludes=['name', 'checkpoint'])

        if name in optimizer_map:
            opt = optimizer_map[name]
        else:
            raise NotImplementedError(f"only support {optimizer_map.keys()}")

        if 'checkpoint' in opt_config:
            opt.load_state_dict(torch.load(
                opt_config['checkpoint'], map_location='cpu'))

        opt = opt([p for p in self.model.parameters()
                  if p.requires_grad], **kwargs)
        
        self.optimizer = opt
        print("optimizer: ", self.optimizer) if self.verbose else None

    def _set_trainer(self, **trainer_config):
        path = trainer_config.get("path", None)
        name = trainer_config.get("name", "standard").lower()
        kwargs = self.get_kwargs(trainer_config, excludes=["name", "path"])

        if path is not None:
            entries = get_entries(path)
            trainer = entries.load(
                name, self.model, self.optimizer, self.loss, metrics=self.metrics, **kwargs)
        elif name in trainer_maps:
            trainer = trainer_maps[name](
                self.model, self.optimizer, self.loss, metrics=self.metrics, **kwargs)
        else:
            raise NotImplementedError("only support standard trainer for empty trainer config")

        self.trainer = trainer
        print("trainer ", self.trainer) if self.verbose else None

    def _set_callbacks(self, *callbacks_configs):
        cbs = []

        for cb_config in callbacks_configs:
            name = cb_config["name"].lower()

            kwargs = self.get_kwargs(cb_config)

            if name in callback_map:
                cbs.append(callback_map[name](**kwargs))

        self.callbacks = cbs
        print("callbacks: ", self.callbacks) if self.verbose else None

    def run(self):
        """
        run the training process based on config
        """
        self.trainer.fit(self.train, self.valid, callbacks=self.callbacks)

        if self.save_config_path is not None:
            full_file = self.save_config_path
            with open(full_file, "w") as handle:
                json.dump(self.config, handle)

    @staticmethod
    def get_kwargs(configs, excludes=("name",)):
        excludes = set(excludes)

        return {k: configs[k] for k in configs if k not in excludes}


class DDPRunner:

    def __init__(self, config, backend, devices, addr='localhost', port='25389', deepspeed=False):
        self.config = config
        self.backend = backend
        self.devices = devices

        self._default_addr = addr
        self._default_port = port
        self.deepspeed = deepspeed
    
    def _setup(self, rank):
        os.environ['MASTER_ADDR'] = self._default_addr
        os.environ['MASTER_PORT'] = self._default_port

        dist.init_process_group(self.backend, rank=rank, world_size=self.devices)

    def _run(self, rank):
        self._setup(rank)

        runner = ConfigRunner(self.config, ddp=True, rank=rank, verbose=rank == 0)
        runner.run()

        dist.destroy_process_group()

    def run(self):
        mp.spawn(self._run, nprocs=self.devices, join=True)
    
    def set_addr_port(self, addr=None, port=None):
        if addr is not None:
            self._default_addr = addr

        if port is not None:
            self._default_port = port


def create(config, save_config_path=None, ddp=False, backend='nccl', verbose=True, addr='localhost', port='25389'):
    """
    create runner based on predefined configuration

    Args:
        config: config dict or path to config dict
        save_config_path: where to save config after training
        ddp: whether to use ddp or not
        backend: ddp backend
        verbose: print creation step
        addr: address for process to communicate, default=localhost
        port: communication port, default=25389
    
    Returns:
        runner
    """
    if not isinstance(config, dict):
        with open(config) as handle:
            config = json.load(handle)

    if ddp:
        devices = torch.cuda.device_count()

        if devices < 2: 
            warn(f'found {devices} device, default to 1 process')
        else:
            return DDPRunner(config, backend, devices, addr, port)

    return ConfigRunner(config, save_config_path=save_config_path, verbose=verbose)


def _format_print_dict(dic, tab=''):
    for key in dic:
        val = dic[key]

        if isinstance(val, dict):
            print(f'{tab}{key}:')
            _format_print_dict(val, tab + '\t')
        elif isinstance(val, list):
            print(f'{tab}{key}:')
            for item in val:
                _format_print_dict(item, tab + '\t')
        else:
            print(f'{tab}{key}: {val}')
