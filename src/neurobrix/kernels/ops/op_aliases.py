# Op Aliases - Maps topology op names to ATen Registry Keys
# NeuroBrix - Zero Hardcode compliant

# Map lowercase or weird names to valid ATen kernel registry keys
OP_ALIASES = {
    "instancenormalization": "InstanceNormalization",
    "layernormalization": "LayerNormalization",
    "batchnormalization": "BatchNormalization",
    "groupnormalization": "GroupNormalization",
    
    "reducemean": "ReduceMean",
    "reducesum": "ReduceSum",
    "reducemax": "ReduceMax",
    "reducemin": "ReduceMin",
    "reduceprod": "ReduceProd",
    
    "conv": "Conv",
    "convtranspose": "ConvTranspose",
    "gemm": "Gemm",
    "matmul": "MatMul",
    
    "constantofshape": "ConstantOfShape",
    "range": "Range",
    "shape": "Shape",
    "slice": "Slice",
    "gather": "Gather",
    "unsqueeze": "Unsqueeze",
    "squeeze": "Squeeze",
    "concat": "Concat",
    "flatten": "Flatten",
    "reshape": "Reshape",
    "transpose": "Transpose",
    "cast": "Cast",
    "split": "Split",
    
    "min": "Min",
    "max": "Max",
    "sum": "Sum",
    "mean": "Mean",
    "pow": "Pow",
    "sqrt": "Sqrt",
    "abs": "Abs",
    "exp": "Exp",
    "log": "Log",
    "sin": "Sin",
    "cos": "Cos",
    "tanh": "Tanh",
    
    "equal": "Equal",
    "greater": "Greater",
    "less": "Less",
    "where": "Where",
    "not": "Not",
    "and": "And",
    "or": "Or",
    
    "softmax": "Softmax",
    "gelu": "Gelu",
    "relu": "Relu",
    "sigmoid": "Sigmoid",
    "silu": "Silu",  # ATen silu is x * sigmoid(x)
                   
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
    "div": "Div",
}

def resolve_op_name(op: str) -> str:
    """Resolve topology op name to ATen Registry Key."""
    # 1. Direct alias
    if op in OP_ALIASES:
        return OP_ALIASES[op]
    
    # 2. Lowercase alias
    op_lower = op.lower()
    if op_lower in OP_ALIASES:
        return OP_ALIASES[op_lower]
        
    # 3. PascalCase heuristic (snake_case to PascalCase)
    # e.g. layer_norm -> LayerNorm
    pascal = "".join(word.capitalize() for word in op_lower.split("_"))
    return pascal
# Additional aliases (added for coverage)
OP_ALIASES.update({
    "constant": "Constant",
    "neg": "Neg",
    "tile": "Tile",
    "expand": "Expand",
    "silu": "Silu",
    "einsum": "Einsum",
    "resize": "Resize",
})
