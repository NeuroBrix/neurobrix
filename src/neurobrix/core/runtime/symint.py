"""
SymInt - Symbolic Integer for Runtime Shape Resolution.

Copied from trace/symbolic/shapes.py for runtime independence.
Only includes SymInt and SymExprOp (runtime needs only .resolve()).

ZERO IMPORT from trace/ module.
"""

from enum import Enum, auto
from typing import Dict, Set, Union, Any, Optional
import re
import ast


class SymExprOp(Enum):
    """Types of nodes in the expression tree."""
    SYMBOL = auto()
    CONST = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    FLOORDIV = auto()
    MOD = auto()
    NEG = auto()


class SymInt:
    """
    Symbolic Integer for shape tracking.

    Represents concrete values, symbol references, or expression trees.
    At runtime, call .resolve({"s0": 4, "s1": 5}) to evaluate.
    """
    __slots__ = ('_op', '_left', '_right', '_symbol_id', '_const_val', '_trace_val')

    def __init__(self, value: Optional[Union[int, 'SymInt']] = None):
        if isinstance(value, SymInt):
            self._op = value._op
            self._left = value._left
            self._right = value._right
            self._symbol_id = value._symbol_id
            self._const_val = value._const_val
            self._trace_val = value._trace_val
        elif isinstance(value, int):
            self._op = SymExprOp.CONST
            self._left = None
            self._right = None
            self._symbol_id = None
            self._const_val = value
            self._trace_val = value
        elif value is None:
            self._op = SymExprOp.CONST
            self._left = None
            self._right = None
            self._symbol_id = None
            self._const_val = 0
            self._trace_val = 0
        else:
            raise TypeError(f"SymInt cannot be created from {type(value)}")

    @classmethod
    def symbol(cls, symbol_id: str, trace_value: int) -> 'SymInt':
        sym = cls.__new__(cls)
        sym._op = SymExprOp.SYMBOL
        sym._left = None
        sym._right = None
        sym._symbol_id = symbol_id
        sym._const_val = None
        sym._trace_val = trace_value
        return sym

    @classmethod
    def _binary(cls, op: SymExprOp, left: 'SymInt', right: 'SymInt', trace_val: int) -> 'SymInt':
        node = cls.__new__(cls)
        node._op = op
        node._left = left
        node._right = right
        node._symbol_id = None
        node._const_val = None
        node._trace_val = trace_val
        return node

    @classmethod
    def _unary(cls, op: SymExprOp, operand: 'SymInt', trace_val: int) -> 'SymInt':
        node = cls.__new__(cls)
        node._op = op
        node._left = operand
        node._right = None
        node._symbol_id = None
        node._const_val = None
        node._trace_val = trace_val
        return node

    def _coerce(self, other: Union['SymInt', int]) -> 'SymInt':
        if isinstance(other, SymInt):
            return other
        if isinstance(other, int):
            return SymInt(other)
        raise TypeError(f"Cannot coerce {type(other)} to SymInt")

    @property
    def trace_value(self) -> int:
        return self._trace_val

    def is_concrete(self) -> bool:
        return self._op == SymExprOp.CONST

    def is_symbol(self) -> bool:
        return self._op == SymExprOp.SYMBOL

    def is_expression(self) -> bool:
        return self._op not in (SymExprOp.CONST, SymExprOp.SYMBOL)

    def get_symbols(self) -> Set[str]:
        if self._op == SymExprOp.CONST:
            return set()
        if self._op == SymExprOp.SYMBOL:
            assert self._symbol_id is not None
            return {self._symbol_id}
        if self._op == SymExprOp.NEG:
            assert self._left is not None
            return self._left.get_symbols()
        assert self._left is not None and self._right is not None
        return self._left.get_symbols() | self._right.get_symbols()

    def __mul__(self, other: Union['SymInt', int]) -> 'SymInt':
        other = self._coerce(other)
        if other.is_concrete() and other._const_val == 1:
            return self
        if self.is_concrete() and self._const_val == 1:
            return other
        if self.is_concrete() and other.is_concrete():
            return SymInt(self._const_val * other._const_val)
        if (self.is_concrete() and self._const_val == 0) or \
           (other.is_concrete() and other._const_val == 0):
            return SymInt(0)
        return SymInt._binary(SymExprOp.MUL, self, other, self._trace_val * other._trace_val)

    def __rmul__(self, other: Union['SymInt', int]) -> 'SymInt':
        return self.__mul__(other)

    def __add__(self, other: Union['SymInt', int]) -> 'SymInt':
        other = self._coerce(other)
        if other.is_concrete() and other._const_val == 0:
            return self
        if self.is_concrete() and self._const_val == 0:
            return other
        if self.is_concrete() and other.is_concrete():
            return SymInt(self._const_val + other._const_val)
        return SymInt._binary(SymExprOp.ADD, self, other, self._trace_val + other._trace_val)

    def __radd__(self, other: Union['SymInt', int]) -> 'SymInt':
        return self.__add__(other)

    def __sub__(self, other: Union['SymInt', int]) -> 'SymInt':
        other = self._coerce(other)
        if other.is_concrete() and other._const_val == 0:
            return self
        if self.is_concrete() and other.is_concrete():
            return SymInt(self._const_val - other._const_val)
        return SymInt._binary(SymExprOp.SUB, self, other, self._trace_val - other._trace_val)

    def __rsub__(self, other: Union['SymInt', int]) -> 'SymInt':
        other = self._coerce(other)
        return other.__sub__(self)

    def __floordiv__(self, other: Union['SymInt', int]) -> 'SymInt':
        other = self._coerce(other)
        if other.is_concrete() and other._const_val == 1:
            return self
        if self._structurally_equal(other):
            return SymInt(1)
        if self._op == SymExprOp.MUL:
            assert self._left is not None and self._right is not None
            if self._right._structurally_equal(other):
                return self._left
            if self._left._structurally_equal(other):
                return self._right
        if self.is_concrete() and other.is_concrete():
            if other._const_val == 0:
                raise ZeroDivisionError("SymInt floor division by zero")
            return SymInt(self._const_val // other._const_val)
        if other.is_concrete() and other._const_val == 0:
            raise ZeroDivisionError("SymInt floor division by zero")
        return SymInt._binary(
            SymExprOp.FLOORDIV, self, other,
            self._trace_val // other._trace_val if other._trace_val != 0 else 0
        )

    def __rfloordiv__(self, other: Union['SymInt', int]) -> 'SymInt':
        other = self._coerce(other)
        return other.__floordiv__(self)

    def __mod__(self, other: Union['SymInt', int]) -> 'SymInt':
        other = self._coerce(other)
        if self.is_concrete() and other.is_concrete():
            if other._const_val == 0:
                raise ZeroDivisionError("SymInt modulo by zero")
            return SymInt(self._const_val % other._const_val)
        if other.is_concrete() and other._const_val == 0:
            raise ZeroDivisionError("SymInt modulo by zero")
        return SymInt._binary(
            SymExprOp.MOD, self, other,
            self._trace_val % other._trace_val if other._trace_val != 0 else 0
        )

    def __rmod__(self, other: Union['SymInt', int]) -> 'SymInt':
        other = self._coerce(other)
        return other.__mod__(self)

    def __neg__(self) -> 'SymInt':
        if self.is_concrete():
            return SymInt(-self._const_val)
        return SymInt._unary(SymExprOp.NEG, self, -self._trace_val)

    def __pos__(self) -> 'SymInt':
        return self

    def _structurally_equal(self, other: 'SymInt') -> bool:
        if not isinstance(other, SymInt):
            return False
        if self._op != other._op:
            return False
        if self._op == SymExprOp.CONST:
            return self._const_val == other._const_val
        if self._op == SymExprOp.SYMBOL:
            return self._symbol_id == other._symbol_id
        if self._op == SymExprOp.NEG:
            assert self._left is not None and other._left is not None
            return self._left._structurally_equal(other._left)
        assert self._left is not None and self._right is not None
        assert other._left is not None and other._right is not None
        return (self._left._structurally_equal(other._left) and
                self._right._structurally_equal(other._right))

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return self.is_concrete() and self._const_val == other
        if isinstance(other, SymInt):
            return self._structurally_equal(other)
        return False

    def __hash__(self) -> int:
        if self.is_concrete():
            return hash(('const', self._const_val))
        if self.is_symbol():
            return hash(('symbol', self._symbol_id))
        return hash((self._op, hash(self._left), hash(self._right) if self._right else None))

    def __int__(self) -> int:
        if self.is_concrete():
            return self._const_val
        raise TypeError(f"Cannot convert symbolic expression to int: {self}")

    def free_symbols(self) -> Set[str]:
        return self.get_symbols()

    def substitute(self, old_symbol: str, new_expr: 'SymInt') -> 'SymInt':
        if self._op == SymExprOp.CONST:
            return self
        if self._op == SymExprOp.SYMBOL:
            if self._symbol_id == old_symbol:
                return new_expr
            return self
        if self._op == SymExprOp.NEG:
            assert self._left is not None
            new_operand = self._left.substitute(old_symbol, new_expr)
            if new_operand is self._left:
                return self
            return -new_operand
        assert self._left is not None and self._right is not None
        new_left = self._left.substitute(old_symbol, new_expr)
        new_right = self._right.substitute(old_symbol, new_expr)
        if new_left is self._left and new_right is self._right:
            return self
        op_map = {
            SymExprOp.ADD: lambda l, r: l + r,
            SymExprOp.SUB: lambda l, r: l - r,
            SymExprOp.MUL: lambda l, r: l * r,
            SymExprOp.FLOORDIV: lambda l, r: l // r,
            SymExprOp.MOD: lambda l, r: l % r,
        }
        return op_map[self._op](new_left, new_right)

    def resolve(self, values: Dict[str, int]) -> int:
        if self._op == SymExprOp.CONST:
            assert self._const_val is not None
            return self._const_val
        if self._op == SymExprOp.SYMBOL:
            assert self._symbol_id is not None
            if self._symbol_id not in values:
                raise KeyError(f"Symbol '{self._symbol_id}' not found in values")
            return values[self._symbol_id]
        if self._op == SymExprOp.NEG:
            assert self._left is not None
            return -self._left.resolve(values)
        assert self._left is not None and self._right is not None
        left_val = self._left.resolve(values)
        right_val = self._right.resolve(values)
        if self._op == SymExprOp.ADD:
            return left_val + right_val
        if self._op == SymExprOp.SUB:
            return left_val - right_val
        if self._op == SymExprOp.MUL:
            return left_val * right_val
        if self._op == SymExprOp.FLOORDIV:
            return left_val // right_val
        if self._op == SymExprOp.MOD:
            return left_val % right_val
        raise ValueError(f"Unknown SymInt operation: {self._op}")

    def to_json(self) -> Union[int, Dict[str, Any]]:
        if self._op == SymExprOp.CONST:
            assert self._const_val is not None
            return self._const_val
        if self._op == SymExprOp.SYMBOL:
            assert self._symbol_id is not None
            return {"type": "symbol", "id": self._symbol_id, "trace": self._trace_val}
        if self._op == SymExprOp.NEG:
            assert self._left is not None
            return {"type": "neg", "operand": self._left.to_json(), "trace": self._trace_val}
        assert self._left is not None and self._right is not None
        return {
            "type": self._op.name.lower(),
            "left": self._left.to_json(),
            "right": self._right.to_json(),
            "trace": self._trace_val,
        }

    @classmethod
    def from_json(cls, data: Union[int, str, Dict[str, Any]]) -> 'SymInt':
        if isinstance(data, int):
            return cls(data)
        if isinstance(data, str):
            if re.match(r'^s\d+$', data):
                return cls.symbol(data, trace_value=1)
            return cls._from_legacy_string(data)
        if isinstance(data, dict):
            type_str = data.get("type", "")
            trace = data.get("trace", 1)
            if type_str == "symbol":
                return cls.symbol(data["id"], trace)
            if type_str == "neg":
                operand = cls.from_json(data["operand"])
                return cls._unary(SymExprOp.NEG, operand, trace)
            op_map = {
                "add": SymExprOp.ADD, "sub": SymExprOp.SUB,
                "mul": SymExprOp.MUL, "floordiv": SymExprOp.FLOORDIV,
                "mod": SymExprOp.MOD,
            }
            if type_str in op_map:
                left = cls.from_json(data["left"])
                right = cls.from_json(data["right"])
                return cls._binary(op_map[type_str], left, right, trace)
            raise ValueError(f"Unknown SymInt type: {type_str}")
        raise TypeError(f"Cannot deserialize SymInt from {type(data)}")

    @classmethod
    def _from_legacy_string(cls, expr: str) -> 'SymInt':
        expr = expr.strip()
        if re.match(r'^s\d+$', expr):
            return cls.symbol(expr, trace_value=1)
        try:
            return cls(int(expr))
        except ValueError:
            pass
        tree = ast.parse(expr, mode='eval')
        return cls._from_ast(tree.body)

    @classmethod
    def _from_ast(cls, node: ast.AST) -> 'SymInt':
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return cls(node.value)
        if isinstance(node, ast.Num):
            # ast.Num.n is a complex type, need to handle as Any
            n_val: Any = node.n  # type: ignore[attr-defined]
            return cls(int(n_val))
        if isinstance(node, ast.Name):
            return cls.symbol(node.id, trace_value=1)
        if isinstance(node, ast.BinOp):
            left = cls._from_ast(node.left)
            right = cls._from_ast(node.right)
            op_map = {
                ast.Add: lambda l, r: l + r, ast.Sub: lambda l, r: l - r,
                ast.Mult: lambda l, r: l * r, ast.FloorDiv: lambda l, r: l // r,
                ast.Mod: lambda l, r: l % r,
            }
            op_type = type(node.op)
            if op_type in op_map:
                return op_map[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        if isinstance(node, ast.UnaryOp):
            operand = cls._from_ast(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    def to_expr_string(self) -> str:
        if self._op == SymExprOp.CONST:
            assert self._const_val is not None
            return str(self._const_val)
        if self._op == SymExprOp.SYMBOL:
            assert self._symbol_id is not None
            return self._symbol_id
        if self._op == SymExprOp.NEG:
            assert self._left is not None
            return f"-({self._left.to_expr_string()})"
        assert self._left is not None and self._right is not None
        op_str = {
            SymExprOp.ADD: "+", SymExprOp.SUB: "-", SymExprOp.MUL: "*",
            SymExprOp.FLOORDIV: "//", SymExprOp.MOD: "%",
        }.get(self._op, "?")
        return f"({self._left.to_expr_string()} {op_str} {self._right.to_expr_string()})"

    def __repr__(self) -> str:
        if self._op == SymExprOp.CONST:
            return f"SymInt({self._const_val})"
        if self._op == SymExprOp.SYMBOL:
            return f"SymInt({self._symbol_id}={self._trace_val})"
        return f"SymInt({self.to_expr_string()}={self._trace_val})"

    def __str__(self) -> str:
        return self.to_expr_string()
