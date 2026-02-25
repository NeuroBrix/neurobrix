from collections import deque
from typing import Sequence, Optional
import torch
import torch._dynamo as torchdynamo
import torch.fx as fx


aten = torch.ops.aten

_DEQUANT_OPS = {
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
}


@torchdynamo.register_backend(name="qattn")
def qattn_backend(gm: fx.GraphModule, sample_inputs: Sequence[torch.Tensor], **kwargs):
    """QAttn backend to lower supported ops to custom Triton INT8 kernels.

    Args:
        gm (fx.GraphModule): Converted model.
        sample_inputs (Sequence[torch.Tensor]): Sample input tensors
        compiler_kwargs (dict): Compile keyword arguments. Supported options
            "inductor" (bool): use inductor to optimize other ops with torch.compile.
                Defaults to false.

    Returns the model with lowered operations.

    """
    compiler_kwargs = kwargs.get("options", {})
    insert_dequant = compiler_kwargs.get("insert_dequant", False)
    use_inductor = compiler_kwargs.get("inductor", False)
    qdq_queue = deque()
    nodes = list(gm.graph.nodes)
    for node in nodes:
        if node.target == aten.linear.default:
            _lower_linear(
                gm,
                linear_node=node,
                qdq_queue=qdq_queue,
            )
        elif node.target == aten.scaled_dot_product_attention.default:
            _lower_sdpa(
                gm,
                node,
                sample_inputs=sample_inputs,
                qdq_queue=qdq_queue,
                insert_dequant=insert_dequant,
            )
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    if use_inductor:
        inductor = torchdynamo.lookup_backend("inductor")

        try:
            return inductor(gm, sample_inputs)
        except Exception:
            pass

    return gm.forward


def _lower_linear(
    gm: fx.GraphModule,
    linear_node: fx.Node,
    qdq_queue: deque,
    **kwargs,
):
    """Lower linear layer with qdq into quantized linear layer.

    Static:
    From:

              quant
        + - - - | - - - +
        |    dequant    |
        |       |       |
        |    linear <---- dequant <- weight
        + - - - | - - - +
             dequant
    To:
              quant
        + - - - | - - - +
        |       |       |
        |    linear <----  int8_weight
        + - - - | - - - +
             dequant

    Dynamic
    From:

          choose_qparams
                |
              quant
        + - - - | - - - +
        |    dequant    |
        |       |       |
        |    linear <---- dequant <- weight
        + - - - | - - - +
             dequant
    To:
          choose_qparams
        + - - - | - - - +
        |       |       |
        |    linear <----  int8_weight
        + - - - | - - - +
               out

    Args:
        linear_node (fx.Node): ATen Node to be replaced with static or dynamic triton kernel.
        qdq_queue (deque): Queue to keep quant dequant ops.
    """
    if len(linear_node.args) == 3:
        # has bias
        input_node, weight_node_dq, bias_node = linear_node.args
    else:
        input_node, weight_node_dq = linear_node.args
        bias_node = None
    input_per_channel = input_node.target == torch.ops.quantized_decomposed.dequantize_per_channel.default
    if not (input_node.target in _DEQUANT_OPS or input_per_channel):
        return
    if input_per_channel:
        (quant_node, inp_scale, _, dim, qmin, qmax, dtype) = input_node.args
    else:
        (quant_node, inp_scale, _, _, _, dtype) = input_node.args
    should_quant_input = True
    if qdq_queue:
        inp_scale_qdq = qdq_queue[-1]
        should_pop_qdq = inp_scale == inp_scale_qdq
        should_pop_qdq = should_pop_qdq or (
            (isinstance(inp_scale_qdq, fx.Node) and inp_scale_qdq.op == "get_attr")
            and torch.equal(
                getattr(gm, inp_scale.target),
                getattr(gm, inp_scale_qdq.target),
            )
        )
        if should_pop_qdq:
            should_quant_input = False
            _ = qdq_queue.pop()
    choose_qparams_node = _find_choose_qparams_node(quant_node)
    is_static = choose_qparams_node is None
    if should_quant_input and is_static:
        new_input_node = quant_node
    else:
        new_input_node = quant_node.args[0]
        quant_node.replace_all_uses_with(new_input_node)
        gm.graph.erase_node(quant_node)

    if len(weight_node_dq.args) == 6:
        # scalar scale
        (weight_node, weight_scale, _, _, _, _) = weight_node_dq.args
        weight_per_channel = False
    elif len(weight_node_dq.args) == 7:
        # per_channel
        (weight_node, weight_scale, _, axis, _, _, _) = weight_node_dq.args
        weight_per_channel = True
    if input_per_channel:
        out_quant_node = list(linear_node.users)[0]
    else:
        out_quant_node = linear_node.next if is_static else None
    if is_static:
        out_dq_node = out_quant_node.next
        out_scale = out_quant_node.args[1]
    if input_per_channel or weight_per_channel:
        linear_op = (
            torch.ops.qattn.static_linear_per_channel if is_static else torch.ops.qattn.dynamic_linear_per_channel
        )
        kwargs = {
            "input_per_channel": input_per_channel,
            "weight_per_channel": weight_per_channel,
        }
    else:
        linear_op = torch.ops.qattn.static_linear if is_static else torch.ops.qattn.dynamic_linear
        kwargs = {}
    if input_per_channel:
        ref_node = out_scale
    else:
        ref_node = linear_node
    if is_static:
        new_linear_args = (
            new_input_node,
            weight_node,
            inp_scale,
            weight_scale,
            out_scale,
            bias_node,
        )
    else:
        new_linear_args = (
            new_input_node,
            weight_node,
            inp_scale,
            weight_scale,
            bias_node,
        )
    with gm.graph.inserting_after(ref_node):
        new_node = gm.graph.call_function(
            linear_op,
            new_linear_args,
            kwargs=kwargs,
        )
        linear_node.replace_all_uses_with(new_node)
    if is_static:
        out_dq_node.args = (new_node, *out_dq_node.args[1:])
        qdq_queue.append(out_dq_node)
        gm.graph.erase_node(out_quant_node)
    else:
        qdq_queue.append(choose_qparams_node)
    for n in [
        linear_node,
        weight_node_dq,
        input_node,
    ]:
        gm.graph.erase_node(n)


def _lower_sdpa(
    gm: fx.GraphModule,
    sdpa_node: fx.Node,
    sample_inputs: Sequence[torch.Tensor],
    qdq_queue: deque,
    insert_dequant: bool = False,
    **kwargs,
):
    """Lower `torch.nn.functional.scaled_dot_product_attention` to its quantized version.

    This function takes a `scaled_dot_product_attention` node and converts it to the static or
    dynamic quantized node. It assumes that for the static quantization, it will share
    the quantization scale factor with the previous linear layer.

    For static quantization from:
              quant
        + - - - | - - - +
        |       |       |
        |    linear <----  int8_weight
        + - - - | - - - +
             dequant
                |
               sdpa
                |
              quant
                |
             dequant
    to:
              quant
        + - - - | - - - +
        |       |       |
        |    linear <----  int8_weight
        + - - - | - - - +
                |
            static_sdpa

    Args:
        gm (fx.GraphModule): GraphModule to be lowered
        sdpa_node (fx.Node): current scaled_dot_product_attention node.
        sample_inputs (Sequence[torch.Tensor]): Sample inputs passed from compile. Not used.
        qdq_queue (deque): Queue with quantization or choose_qparams node.
        insert_dequant (bool, optional): If we would like to pass the dequantization node after
            static attention. Defaults to False.
    """

    def _is_per_channel():
        if sdpa_node.next.op != "get_attr":
            return False
        scale_node = sdpa_node.next
        is_per_channel = False
        for n in scale_node.users.keys():
            is_per_channel = n.target in {
                torch.ops.quantized_decomposed.quantize_per_channel.default,
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
            }
            if is_per_channel:
                return True
        return is_per_channel

    try:
        qdq_node = qdq_queue.pop()
    except IndexError:
        # Previous node was not quantized so we are not observing SDPA
        return

    _quant_ops = {
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
    }
    _dynamic_ops = {
        torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor,
    }
    is_dynamic = qdq_node.target in _dynamic_ops
    if is_dynamic:
        return
    is_quantized = qdq_node.target in _quant_ops
    if not (_is_per_channel() or is_quantized):
        return
    q, k, v = sdpa_node.args[:3]
    if q.target == aten.to.dtype:
        q, k, v = [x.args[0] for x in [q, k, v]]
    post_quant_node = sdpa_node.next
    if post_quant_node.target not in _quant_ops:
        return
    sdpa_op = torch.ops.qattn.static_attention
    post_dqd_node = post_quant_node.next
    out_scale = post_quant_node.args[1]
    new_args = (q, k, v, qdq_node.args[1], out_scale)
    qdq_queue.append(out_scale)
    erase_nodes = [post_dqd_node, post_quant_node, sdpa_node]
    with gm.graph.inserting_after(sdpa_node):
        new_sdpa_node = gm.graph.call_function(sdpa_op, new_args)
        post_dqd_node.replace_all_uses_with(new_sdpa_node)

    if not insert_dequant and not is_dynamic:
        qdq_node.replace_all_uses_with(qdq_node.args[0])
        erase_nodes.append(qdq_node)
    for n in erase_nodes:
        n.args = tuple()
        gm.graph.erase_node(n)


def _find_choose_qparams_node(node: fx.Node) -> Optional[fx.Node]:
    """Assume that passed node isv the quantize node.

    for dynamic:
        quant_node.args[1] -> getitem_node.args[0] -> choose_qparam_node
    for static:
        quant_node.args[1] -> getattr_node.args

    Args:
        node (fx.Node): Quantization Node.

    Returns:
        Optional[fx.Node]: Node with choose_qparams op if present.
    """
    if isinstance(node.args[1], fx.Node) and len(node.args[1].args):
        return node.args[1].args[0]
    return None
