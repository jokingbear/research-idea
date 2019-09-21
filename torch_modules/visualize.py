import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from graphviz import Digraph
from uuid import uuid4 as uid


class GraphModule(ABC):

    @abstractmethod
    def graph_forward(self, graph: Digraph, input_nodes, input_tensors):
        pass


def render_module(module: nn.Module, graph: Digraph, input_tensors, input_nodes):
    if isinstance(module, nn.Sequential):
        for module in module._modules.values():
            input_nodes, input_tensors = render_module(module, graph, input_tensors, input_nodes)

        return input_nodes, input_tensors

    if isinstance(module, GraphModule):
        return module.graph_forward(graph, input_nodes, input_tensors)

    node_id = str(uid())
    node_name = str(module)
    node_value = module(*input_tensors)
    node_shape = node_value.shape
    node_label = f"{{{node_name}|{node_shape}}}"

    graph.node(node_id, node_label, shape="record")
    [graph.edge(input_node, node_id) for input_node in input_nodes]

    return [node_id], [node_value]


def get_input_node(graph, *size):
    a = torch.zeros(*size)

    graph.node("input")

    return ["input"], [a]


def graph_module(module, input_shape=()):
    g = Digraph(format="png")

    with torch.no_grad():
        a = torch.zeros(*input_shape)
        g.node("input", f"{{input|{a.shape}}}", shape="record")

        render_module(module, g, [a], ["input"])

    return g

