"""
title: LiteralTuple lowering tests
"""

from __future__ import annotations

import re

from typing import cast

import astx
import pytest

from irx.builders.base import Builder
from irx.builders.llvmliteir import LLVMLiteIR, LLVMLiteIRVisitor
from llvmlite import ir

EXPECTED_TUPLE_LENGTH = 3


def _struct_int_values(const: ir.Constant) -> list[int]:
    """
    title: Extract integer values from struct constant
    parameters:
      const:
        type: ir.Constant
    returns:
      type: list[int]
    """
    return [int(v) for v in re.findall(r"i\d+\s+(-?\d+)", str(const))]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_empty(builder_class: type[Builder]) -> None:
    """
    title: Empty LiteralTuple lowering
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(astx.LiteralTuple(elements=()))
    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == 0


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_homogeneous_ints(
    builder_class: type[Builder],
) -> None:
    """
    title: Homogeneous LiteralTuple lowering
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    visitor.visit(
        astx.LiteralTuple(
            elements=(
                astx.LiteralInt32(1),
                astx.LiteralInt32(2),
                astx.LiteralInt32(3),
            )
        )
    )

    const = visitor.result_stack.pop()

    assert isinstance(const, ir.Constant)
    assert isinstance(const.type, ir.LiteralStructType)
    assert len(const.type.elements) == EXPECTED_TUPLE_LENGTH
    assert all(isinstance(t, ir.IntType) for t in const.type.elements)

    vals = _struct_int_values(const)
    assert vals == [1, 2, 3]


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_literal_tuple_heterogeneous_unsupported(
    builder_class: type[Builder],
) -> None:
    """
    title: Heterogeneous LiteralTuple rejection
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    with pytest.raises(TypeError, match="homogeneous"):
        visitor.visit(
            astx.LiteralTuple(
                elements=(
                    astx.LiteralInt32(1),
                    astx.LiteralFloat32(2.0),
                )
            )
        )


def _make_int_tuple(*vals: int) -> astx.LiteralTuple:
    return astx.LiteralTuple(
        elements=tuple(astx.LiteralInt32(v) for v in vals)
    )


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_tuple_index_first_element(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr returns the first tuple element for index 0.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_tuple(10, 20), index=astx.LiteralInt32(0)
    )
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    EXPECTED_FIRST = 10
    assert isinstance(result, ir.Constant)
    assert result.constant == EXPECTED_FIRST


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_tuple_index_second_element(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr returns the second tuple element for index 1.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_tuple(10, 20), index=astx.LiteralInt32(1)
    )
    visitor.visit(expr)
    result = visitor.result_stack.pop()

    EXPECTED_SECOND = 20
    assert isinstance(result, ir.Constant)
    assert result.constant == EXPECTED_SECOND


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_tuple_index_out_of_bounds(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr raises IndexError for an out-of-bounds index.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_tuple(10, 20), index=astx.LiteralInt32(5)
    )
    with pytest.raises(IndexError, match="out of range"):
        visitor.visit(expr)


@pytest.mark.parametrize("builder_class", [LLVMLiteIR])
def test_tuple_index_variable_rejected(builder_class: type[Builder]) -> None:
    """
    title: SubscriptExpr raises TypeError when a non-constant index is used.
    parameters:
      builder_class:
        type: type[Builder]
    """
    builder = builder_class()
    visitor = cast(LLVMLiteIRVisitor, builder.translator)
    visitor.result_stack.clear()

    expr = astx.SubscriptExpr(
        value=_make_int_tuple(10, 20), index=astx.Identifier("i")
    )
    with pytest.raises(TypeError, match="constant literal"):
        visitor.visit(expr)
