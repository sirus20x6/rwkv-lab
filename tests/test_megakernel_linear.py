import pytest
import torch
import torch.nn.functional as F

from rwkv_lab.megakernel_linear import (
    ffn_squared_relu_value,
    packed_rkv_projection,
    prepare_row_one_weight,
    qualify_b1t1_kernels,
    row_one_linear,
)

pytestmark = pytest.mark.gpu


def test_prepared_row_weight_layouts_and_cpu_fallback():
    torch.manual_seed(1)
    x = torch.randn(2, 1, 7)
    weight = torch.randn(11, 7)
    bias = torch.randn(11)
    expected = F.linear(x, weight, bias)
    for layout in ("out_in", "in_out"):
        prepared = prepare_row_one_weight(weight, layout=layout)
        actual = row_one_linear(x, prepared, bias, use_candidate=True)
        torch.testing.assert_close(actual, expected)
        assert prepared.source_signature


def test_packed_rkv_and_exact_ffn_cpu_fallback():
    torch.manual_seed(2)
    x = torch.randn(3, 1, 5)
    weights = torch.randn(3, 9, 5)
    biases = torch.randn(3, 9)
    expected_rkv = torch.stack(
        [F.linear(x, weights[index], biases[index]) for index in range(3)], dim=2)
    torch.testing.assert_close(
        packed_rkv_projection(x, weights, biases, use_candidate=True), expected_rkv)
    distinct_x = torch.randn(3, 1, 3, 5)
    expected_distinct = torch.stack([
        F.linear(distinct_x[:, :, index], weights[index], biases[index])
        for index in range(3)
    ], dim=2)
    torch.testing.assert_close(
        packed_rkv_projection(
            distinct_x, weights, biases, use_candidate=True), expected_distinct)

    key_weight = torch.randn(13, 5)
    value_weight = torch.randn(5, 13)
    expected_ffn = F.linear(torch.relu(F.linear(x, key_weight)).square(), value_weight)
    torch.testing.assert_close(
        ffn_squared_relu_value(
            x, key_weight, value_weight, use_candidate=True), expected_ffn)


def test_validation_and_fail_closed_cpu_qualification():
    x = torch.randn(1, 1, 4)
    with pytest.raises(ValueError, match="\[batch,1,features\]"):
        row_one_linear(torch.randn(1, 2, 4), torch.randn(3, 4))
    with pytest.raises(ValueError, match="\[3,out,in\]"):
        packed_rkv_projection(x, torch.randn(2, 3, 4))
    report = qualify_b1t1_kernels(
        x, torch.randn(6, 4), torch.randn(3, 6, 4),
        torch.randn(8, 4), torch.randn(4, 8), repeats=1, warmup=0,
    )
    assert report["schema"] == "rwkv-lab.b1t1-linear-qualification.v1"
    assert report["adopted"] is False
    assert report["candidates"] == {}
    assert report["geometry"]["input"] == 4


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_candidates_and_qualification():
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.float16
    x = torch.randn(1, 1, 64, device=device, dtype=dtype)
    # Fan-in scaling approximates initialized/checkpoint projection ranges and
    # avoids amplifying ordinary fp16 GEMV rounding through the square.
    row_weight = torch.randn(96, 64, device=device, dtype=dtype) / 8
    rkv_weights = torch.randn(3, 64, 64, device=device, dtype=dtype) / 8
    rkv_x = torch.randn(1, 1, 3, 64, device=device, dtype=dtype)
    key_weight = torch.randn(128, 64, device=device, dtype=dtype) / 8
    value_weight = torch.randn(64, 128, device=device, dtype=dtype) / 12
    with torch.no_grad():
        torch.testing.assert_close(
            row_one_linear(x, row_weight, use_candidate=True),
            row_one_linear(x, row_weight), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            packed_rkv_projection(rkv_x, rkv_weights, use_candidate=True),
            packed_rkv_projection(rkv_x, rkv_weights), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            ffn_squared_relu_value(
                x, key_weight, value_weight, use_candidate=True),
            ffn_squared_relu_value(x, key_weight, value_weight),
            atol=5e-2, rtol=2e-2)
        report = qualify_b1t1_kernels(
            x, row_weight, rkv_weights, key_weight, value_weight,
            rkv_x=rkv_x,
            warmup=1, repeats=3, min_speedup=0.0, atol=5e-2, rtol=2e-2)
    assert set(report["candidates"]) == {
        "row_one", "packed_rkv", "ffn_sqrelu_value"}
    assert all(item["parity"] for item in report["candidates"].values())
    assert report["geometry"]["rkv_input_mode"] == "distinct"
    assert report["runtime"]["compute_capability"]
