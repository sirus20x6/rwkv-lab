# Qwen3.6 caption distillation migration

The next Qwen3.6 caption-distillation run is authored by the compact
[`qwen-caption-lora-r256.recipe-instance.v1.json`](examples/qwen-caption-lora-r256.recipe-instance.v1.json)
document and expands through `hf_multimodal_sft@2`. The pre-existing
`hf_multimodal_sft@1` profile remains byte-for-byte semantically unchanged. The
migration does not register a
Qwen-caption trainer, handler, Python import, argv builder, or alternate dashboard
run type.

The instance preserves the corrected-v3 training decision: the exact local BF16
checkpoint and frozen dataset, 311 explicitly attested language/output LoRA targets,
rank 256 with alpha 512, effective batch 16, fused AdamW, 22-step warmup and cosine
decay through step 745, scalar and qualitative evaluation every 250 steps, and exact
checkpoint resume. The generic loader receipt binds `grouped_mm` along with every
other load option. The balanced qualitative set is selected from the content-hashed
`validation-fixed-100.jsonl` manifest instead of relying on incidental validation-file
order.

Each gallery revision carries the same held-out image and three aligned captions:
the stored teacher target, the untouched pre-adapter step-zero baseline, and the
caption at that checkpoint. Step zero and final evaluation are checkpoint anchored.
The full 674-image test split remains private until finalization.

## Parity and failure policy

[`qwen-caption-corrected-v3.parity.v1.json`](examples/qwen-caption-corrected-v3.parity.v1.json)
is the frozen migration contract. Tests compare the old and composed learning-rate
trajectory exactly and compare bounded fused-AdamW parameter updates within the
declared tolerance. They also verify the installed production model, dataset,
balanced held-out IDs, and target manifest when those assets are present.

The historical corrected-v3 run is not accepted as a successful final audit. Its
terminal scalar evaluation and ten-item gallery succeeded, but all 674 adapter-test
generations failed after a CUDA launch timeout poisoned the context. The failed
append-only ledger is preserved at:

```text
/thearray/git/moe-mla/runs/qwen36-caption-distill-lora-r256-corrected-v3/adapter-test.failed-cuda-timeout-20260806.jsonl
```

The generic route fails closed on any required test-generation error. Failed records
remain retryable and never count as complete; a later valid record for the same
identity wins. A final checkpoint records `finalization_pending`, so evaluation can
resume from that checkpoint without another optimizer update. Only a complete
zero-error final evidence bundle may produce `worker.completed`.

The old bespoke implementation and its artifacts remain historical evidence until a
real declarative run passes. CPU parity is not real-run qualification and does not
authorize retiring that path.

## Universal step-zero dependency

~~This route must consume the universal `rwkv-lab.eval-examples.v1` pre-mutation gate
before production launch.~~ It does. The `hf_multimodal_sft` recipe documents declare
a required `eval_examples` artifact in `spec.artifacts` and publish it from the train
node, which is what arms `invocation_requires_step_zero_eval_gate`; the engine
publishes typed `EvalExample`s at the attempt baseline, bound to the same-attempt
baseline checkpoint and to the frozen `eval_manifest_digest`, with the milestone kind
carried in `series_id`.

~~The migration does not create a Qwen-only substitute: scalar step-zero evaluation,
typed examples, checkpoint identity, and the controller acknowledgement must all be
durable before optimizer mutation.~~ They are, and the ordering is proved against the
controller rather than the adapter: `test_universal_step_zero_gate_orders_controller_mutation`
in `trainvm/tests/trainvm_tests.cpp` drives the real service and uses the journal as
the mutation sentinel — a refused step past the attempt baseline must leave the event
count unchanged. It shows the step refused with no evidence, still refused with a
durable baseline checkpoint and scalar but no examples, and admitted only once the
typed examples are durable. A resumed attempt is gated at its own baseline and cannot
be satisfied by the previous attempt's evidence.

**Still open, and why this is not yet a claim of production completion.** A successful
real run has not happened; CPU parity is still not real-run qualification. Two flags
named `required` govern this and nothing cross-checks them: the operation port's
`required` forces the node to publish the output, while only the experiment document's
`spec.artifacts[<name>].required` arms the controller gate. Setting one without the
other looks correct and gates nothing (recorded on `card-6331b63f`). The gate is also
not yet universal in practice — arming is per-document, and only the `rwkv-lm`,
`hf-multimodal-sft` and `lm-training` recipe documents declare an `eval_examples`
artifact at all today.
