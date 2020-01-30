# Copyright (c) 2019, XMOS Ltd, All rights reserved

import numpy
import pytest

from pytest_cases import pytest_fixture_plus, pytest_parametrize_plus, fixture_ref
from tflite2xcore.xcore_model import XCOREModel, TensorType
from tflite2xcore.operator_codes import OperatorCode, BuiltinOpCodes
from tflite2xcore.transformation_passes import ReplaceDeepinAnyoutFullyConnectedOutputPass

from .model_builders import (
    build_fc, build_intermediate_fc, build_mlp, build_logistic
)

from .test_ReplaceDeepinAnyoutFullyConnectedIntermediatePass import (
    MATCHING_INPUT_SIZE,
    NON_MATCHING_INPUT_SIZE,
    outputs, input_size,
    fc_model, logistic
)
MATCHING_HIDDEN_NODES = [numpy.prod(t) for t in MATCHING_INPUT_SIZE]

from .test_ReplaceDeepinDeepoutConv2DPass import (
    NON_MATCHING_TENSORS
)


@pytest.fixture()
def trf_pass():
    return ReplaceDeepinAnyoutFullyConnectedOutputPass()


@pytest.fixture(params=MATCHING_HIDDEN_NODES)
def hidden_nodes(request):
    return request.param


@pytest.fixture()
def mlp(outputs, hidden_nodes, input_size):
    return build_mlp(outputs=outputs, hidden_nodes=hidden_nodes, input_size=input_size)


def test_matching_params(trf_pass, fc_model):
    assert trf_pass.match(fc_model.subgraphs[0].operators[-1])


@pytest.mark.parametrize('input_size', NON_MATCHING_INPUT_SIZE)
def test_non_matching_fc_input_size(trf_pass, outputs, input_size):
    model = build_fc(outputs=outputs, input_size=input_size)
    assert not trf_pass.match(model.subgraphs[0].operators[-1])


@pytest.mark.parametrize(*NON_MATCHING_TENSORS)
def test_non_matching_types(trf_pass, fc_model, tensor_name, new_type):
    subgraph = fc_model.subgraphs[0]
    subgraph.get_tensor(tensor_name).type = new_type
    assert not trf_pass.match(fc_model.subgraphs[0].operators[-1])


def test_logistic(logistic, trf_pass):
    assert not trf_pass.match(logistic.subgraphs[0].operators[0])
    assert not trf_pass.match(logistic.subgraphs[0].operators[-1])


def test_mlp_input_non_match(trf_pass, mlp):
    assert not trf_pass.match(mlp.subgraphs[0].operators[0])


def test_mlp_output_match(trf_pass, mlp):
    assert trf_pass.match(mlp.subgraphs[0].operators[-1])


if __name__ == "__main__":
    pytest.main()
