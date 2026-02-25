"""Lower the prepared model to QAttn backend."""

import logging
from collections import deque
from typing import Callable

import torch
import torch.fx as fx
import torch.nn as nn
from torch.ao.quantization.utils import _parent_name

from qattn.fx.transforms import remove_identity_layers
from qattn import nn as nnq
from qattn.nn.functional import attention as static_quant_attention
from qattn.quantize import dequantize

logger = logging.getLogger(__name__)

__all__ = [
    "lower",
    "lower_dynamic_quant",
    "lower_static_quant",
]


def _replace_attention_with_static(
    model: fx.GraphModule,
    node: fx.Node,
    scale_queue: deque,
):
    """Helper function to replace SDPA with static quantized attention.

    Replace the SDPA node with static quantized attention node inplace.

    Args:
        model (fx.GraphModule): FX-captured model graph.
        node (fx.Node): SDPA node in the graph to be replaced.
        scale_queue (deque): Helper queue to propagate quantization scale factors.
    """
    post_quant_node = list(node.users.keys())[0]
    out_scale_node = post_quant_node.args[1]
    post_dequant_node = post_quant_node.next
    qkv_dequant = node.args
    qkv_quant = [x.args[0] for x in node.args]
    qkv_inputs = [x.args[0] for x in qkv_quant]
    try:
        qkv_scale = scale_queue.pop()
    except IndexError:
        logger.warn(f"{node.name} is missing scale quant, picking K")
        qkv_scale = qkv_quant[1].args[1]
    with model.graph.inserting_after(out_scale_node):
        new_node = model.graph.call_function(
            static_quant_attention, args=tuple([*qkv_inputs, qkv_scale, out_scale_node])
        )
        post_dequant_node.replace_all_uses_with(new_node)
        post_quant_node.args = tuple()
        node.args = tuple()
        model.graph.erase_node(node)
    for n in [post_dequant_node, post_quant_node] + list(qkv_dequant) + list(qkv_quant):
        n.args = tuple()
        model.graph.erase_node(n)


def _replace_linear_layer_with_triton_dynamic(
    model: fx.GraphModule,
    node: fx.Node,
    modules: dict[str, nn.Module],
):
    """Helper function to replace the linear layer with triton dynamic quantized linear layer.

    Replace a reference linear layer with a dynamically quantized linear layer kernel.
    The model has to be lowered before to a reference quantized format.

    Args:
        model (fx.GraphModule): Reference quantized FX-captured model.
        node (fx.Node): Linear layer reference node.
        modules (dict[str, nn.Module]): Modules dict in the model.
    """
    dequant_node = node.args[0]
    if dequant_node.target != "dequantize":
        return
    quant_node = dequant_node.args[0]
    assert quant_node.target == torch.quantize_per_tensor_dynamic
    mod = modules[node.target]
    qmod = nnq.DynamicQuantizedLinear.from_reference(mod)
    parent_name, module_name = _parent_name(node.target)
    setattr(modules[parent_name], module_name, qmod)
    node.replace_input_with(dequant_node, quant_node.args[0])
    quant_node.args = tuple()
    dequant_node.args = tuple()
    model.graph.erase_node(quant_node)
    model.graph.erase_node(dequant_node)


def _replace_linear_layer_with_triton_static(
    model: fx.GraphModule,
    node: fx.Node,
    modules: dict[str, nn.Module],
    scale_queue: deque,
    quant_sdpa: bool = True,
):
    """Helper function to replace the reference linear layer with triton static linear layer.

    Replace a reference linear layer with a statically quantized linear layer. The model has
    to be lowered before to a reference quantized format.

    Args:
        model (fx.GraphModule): _description_
        node (fx.Node): _description_
        modules (dict[str, nn.Module]): _description_
        scale_queue (deque): _description_
        quant_sdpa (bool, optional): _description_. Defaults to True.
    """
    pre_dequant_node = node.args[0]
    if pre_dequant_node.target != "dequantize":
        return
    pre_quant_node = pre_dequant_node.args[0]
    is_per_channel = pre_quant_node.target == torch.quantize_per_channel
    assert pre_quant_node.target == torch.quantize_per_tensor or is_per_channel
    mod = modules[node.target]
    out_scale_node = node.next
    zero_point_node = out_scale_node.next
    out_scale = getattr(model, out_scale_node.target)
    qmod = nnq.QuantizedLinear.from_reference(mod, out_scale)
    parent_name, module_name = _parent_name(node.target)
    should_dequant = "qkv" not in module_name
    if "qkv" in module_name and quant_sdpa:
        scale_queue.append(out_scale_node)
    setattr(modules[parent_name], module_name, qmod)
    # remove quant and dequant nodes
    node.replace_input_with(pre_dequant_node, pre_quant_node.args[0])
    # set input and input scale
    node.args = (node.args[0], pre_quant_node.args[1])
    pre_quant_node.args = tuple()
    pre_dequant_node.args = tuple()
    post_quant_node = zero_point_node.next
    post_dequant_node = post_quant_node.next
    post_quant_node.args = tuple()
    if should_dequant or not quant_sdpa:
        with model.graph.inserting_after(post_dequant_node):
            new_dequant_node = model.graph.call_function(dequantize, args=(node, out_scale_node))
            post_dequant_node.replace_all_uses_with(new_dequant_node)
            model.graph.erase_node(post_dequant_node)
    else:
        post_dequant_node.args = tuple()
        post_dequant_node.replace_all_uses_with(node)
        model.graph.erase_node(post_dequant_node)
    model.graph.erase_node(pre_quant_node)
    model.graph.erase_node(pre_dequant_node)
    model.graph.erase_node(post_quant_node)


def _replace_static_shared_input(
    model: fx.GraphModule,
    node: fx.Node,
    modules: dict[str, nn.Module],
    cls_or_fn: Callable,
    is_module: bool,
):
    """Replace static module or function to quantzied equivalent.

    The module or function should be not observed

    Args:
        model (fx.GraphModule): reference quantzied model.
        node (fx.Node): node to be replaced
        modules (dict[str, nn.Module]): modules dictionary.
        cls_or_fn (Callable): quantized class or function.
        is_module (bool): is module or function.
    """
    dequant_node = node.args[0]
    has_scale = node.next.name.startswith(node.name) and "scale" in node.next.name
    if not has_scale:
        return
    has_postqdq = list(node.next.users)[0].target == torch.quantize_per_tensor
    new_args = dequant_node.args
    if is_module:
        mod = cls_or_fn()
        parent_name, module_name = _parent_name(node.target)
        modules[node.target] = mod
        setattr(modules[parent_name], module_name, mod)
        node.args = new_args
    else:
        with model.graph.inserting_after(node):
            new_node = model.graph.call_function(cls_or_fn, new_args, node.kwargs)
            node.replace_all_uses_with(new_node)
        model.graph.erase_node(node)
        node = new_node
    dequant_node.args = tuple()
    if has_postqdq:
        quant_node = list(node.next.users)[0]
        dq_node = quant_node.next
        dq_node.args = tuple()
        dq_node.replace_all_uses_with(node)
        model.graph.erase_node(dq_node)
        model.graph.erase_node(quant_node)
    if has_scale:
        model.graph.erase_node(node.next.next)
        model.graph.erase_node(node.next)
    model.graph.erase_node(dequant_node)


def lower(
    model: fx.GraphModule,
    is_dynamic: bool = False,
    quant_sdpa: bool = True,
) -> fx.GraphModule:
    """Lower the reference quantized model to GPU-accelerated quantized kernels.

    Given the reference quantized reference model, lower it to a static or dynamic quantized model based on the
    `is_dynamic` flag. By default, it lowers it to a static quantized model.

    Args:
        model (fx.GraphModule): Reference FX-captured reference quantized model.
        is_dynamic (bool, optional): If the quantization is dynamic or static. Defaults to False.
        quant_sdpa (bool, optional): If we should quantize SDPA function. Defaults to True.

    Returns:
        fx.GraphModule: Lowered quantzied model.
    """
    model = remove_identity_layers(model)
    modules = dict(model.named_modules())
    scale_queue = deque()
    if is_dynamic:
        lower_node_fn = _lower_node_dynamic
    else:
        lower_node_fn = _lower_node_static
    for node in model.graph.nodes:
        lower_node_fn(model, node, modules, quant_sdpa, scale_queue=scale_queue)
    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def _lower_node_static(
    model: fx.GraphModule,
    node: fx.Node,
    modules,
    quant_sdpa: bool,
    scale_queue: deque,
    **kwargs,
):
    if node.op == "call_module":
        module = modules.get(node.target)
        if isinstance(module, nn.Linear):
            _replace_linear_layer_with_triton_static(
                model,
                node,
                modules,
                scale_queue,
                quant_sdpa=quant_sdpa,
            )
    elif node.op == "call_function":
        if node.target == torch.nn.functional.scaled_dot_product_attention and quant_sdpa:
            _replace_attention_with_static(
                model,
                node,
                scale_queue,
            )


def _lower_node_dynamic(
    model: fx.GraphModule,
    node: fx.Node,
    modules,
    quant_sdpa: bool,
    **kwargs,
):
    if node.op == "call_module" and isinstance(
        modules.get(node.target),
        nn.Linear,
    ):
        _replace_linear_layer_with_triton_dynamic(
            model,
            node,
            modules,
        )
        return


def lower_dynamic_quant(model: fx.GraphModule, quant_sdpa: bool) -> fx.GraphModule:
    """Lower the reference model to dynamic quantized model.

    Args:
        model (fx.GraphModule): Reference FX-captured reference quantized model.
        quant_sdpa (bool): If we should quantize SDPA function

    Returns:
        fx.GraphModule: Lowered dynamically quantzied model.
    """
    return lower(model, True, quant_sdpa)


def lower_static_quant(model: fx.GraphModule, quant_sdpa: bool) -> fx.GraphModule:
    """Lower the reference model to static quantized model.

    Args:
        model (fx.GraphModule): Reference FX-captured reference quantized model.
        quant_sdpa (bool): If we should quantize SDPA function

    Returns:
        fx.GraphModule: Lowered static quantzied model.
    """
    return lower(model, False, quant_sdpa)
