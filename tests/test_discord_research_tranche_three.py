import time
import torch
from rwkv_lab.cca_attention import CCAAttention
from rwkv_lab.data_filter_audit import filter_regime_report
from rwkv_lab.energy_refinement import CompatibilityEnergy, contrastive_energy_loss, refine_latent
from rwkv_lab.execution_plan import qualify_execution_plan
from rwkv_lab.runtime_backends import qualify_runtime_backend
from rwkv_lab.tool_length_generalization import ToolStep, length_generalization_report, validate_tool_tape

def test_energy_refinement_decreases_energy_and_trains():
    torch.manual_seed(4); m=CompatibilityEnergy(6); c=torch.randn(3,6); p=torch.randn(3,6); n=torch.randn(3,6)
    loss=contrastive_energy_loss(m,c,p,n);loss.backward();assert m.net[0].weight.grad is not None
    start=torch.randn(3,6); refined=refine_latent(m,c,start,steps=5)
    assert m(c,refined).mean() <= m(c,start).mean()+1e-5

def test_cca_operates_entirely_at_latent_width():
    m=CCAAttention(16,4);x=torch.randn(2,7,16);y=m(x)
    assert y.shape==x.shape and m.receipt(7)["cache_elements_per_token"]==8

def test_tool_length_tapes_and_extrapolation_curve():
    tape=[ToolStep("x","add",{"a":1},"1")]
    assert validate_tool_tape(tape,{"add"}) and not validate_tool_tape(tape,set())
    r=length_generalization_report([{"length":4,"train_max":4,"success":True},{"length":8,"train_max":4,"success":True},{"length":16,"train_max":4,"success":False}])
    assert r["extrapolation_success"]==.5

def test_filter_audit_only_compares_matched_regimes():
    runs=[{"parameters":1,"tokens":10,"compute":20,"filter":"none","score":.5},
          {"parameters":1,"tokens":10,"compute":20,"filter":"quality","score":.4},
          {"parameters":2,"tokens":10,"compute":30,"filter":"quality","score":.9}]
    r=filter_regime_report(runs);assert len(r["matched_comparisons"])==1 and r["matched_comparisons"][0]["delta_vs_raw"]<0

def test_backend_and_full_plan_qualification_are_fail_closed():
    def slow(x): time.sleep(.001); return x.square()
    fast=lambda x:x.square(); probes=[(torch.randn(8),)]
    b=qualify_runtime_backend("albatross",slow,fast,probes,repeats=2,minimum_speedup=1)
    assert b["adopted"] and b["backend"]=="albatross"
    plan=qualify_execution_plan(slow,fast,probes,operators_before=8,operators_after=3,repeats=2,minimum_speedup=1)
    assert plan["adopted"] and plan["launch_reduction"]==5
    rejected=qualify_execution_plan(slow,fast,probes,operators_before=3,operators_after=3,repeats=1,minimum_speedup=0)
    assert not rejected["adopted"]
