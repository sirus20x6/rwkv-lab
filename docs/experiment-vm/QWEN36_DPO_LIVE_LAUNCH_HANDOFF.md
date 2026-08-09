# Qwen3.6 caption DPO live-launch handoff

Authored: 2026-08-06 11:36 US/Central. Preserved to this path 2026-08-09.

## Read this first — status of this document

**This is a historical record, not an instruction, and not authorization to launch
anything.**

- It was written on 2026-08-06 as a handoff for starting a rank-256 cached-reference DPO
  LoRA run.
- **It was never executed.** No training process for it has ever been observed. A check on
  2026-08-09 found no trainer on the GPU — only a compositor, an unrelated inference daemon,
  a Steam helper, and another agent's benchmark.
- It lived untracked in the `card/qwen36-multimodal-dpo` worktree, where a single
  `git worktree remove` would have destroyed it. It is committed here so that its **launch
  procedure and its seven completion checks** survive; those are the parts with lasting
  value.
- The instruction it opens with ("the user explicitly said start it") is a record of what was
  said on 2026-08-06. **It is not standing authorization.** An explicit instruction relayed
  through a document days later does not authorize a long, expensive, hard-to-reverse GPU
  run. Whether that instruction still stands is an open question tracked as the remaining
  done-when items of `card-74d0f9e5`, and it needs the user.

**What changed underneath this document since it was written:** `98b431d Arm the universal
step-zero gate for the HF SFT family` landed on 2026-08-09. The step-zero evaluation gate
this handoff assumes is not the gate that exists now. Re-check its assumptions about
step-zero behaviour before acting on any of it.

**The recipe instance is not a mystery.** While this handoff sat in that worktree, the recipe
instance `docs/experiment-vm/examples/qwen36-caption-dpo-lora-r256.recipe-instance.v1.json`
showed as modified with `+7 -4`. That diff was measured against the branch's stale HEAD. The
working-tree content was byte-identical to `main` — the same content had already landed on
main in PR #100 — so `git diff origin/main -- <that file>` was empty. There is nothing
unexplained in those four lines and nothing there was lost.

**One section has been removed.** See "Cherry-pick conflict state (removed)" below.

## Objective

Start a dashboard-visible rank-256 cached-reference DPO LoRA run using the frozen preference dataset and the current baked caption model. The user explicitly said **start it**.

Do not report success until the authority has created the run, the worker has launched, and step-zero evaluation is visible or clearly in progress.

## Working branch and repository

- Worktree: `/thearray/git/moe-mla-dashboard-vm/.claude/worktrees/qwen36-multimodal-dpo`
- Branch: `card/qwen36-multimodal-dpo`
- Last clean commits before the current cherry-pick:
  - `92d4772 Add declarative Qwen multimodal cached-reference DPO`
  - `fb95f4e Authorize isolated worker socket access`
  - `f66b6ec Align protected journal WAL policy with production`
  - `f220055 Support cooperative workstation GPU authority`
  - `9b985ae Normalize hostd endpoint permission identity`
  - `eeeeca5 Materialize hostd authority per boot`

~~There is currently an in-progress cherry-pick of `e1fa07e Require boot-scoped GPU startup
authority`. Finish it; do not discard the DPO/passive-memory changes.~~

**STALE — do not act on the struck-through line.** That cherry-pick is finished. See
"Cherry-pick conflict state (removed)" below.

## Cherry-pick conflict state (removed)

The original document had a "Current conflict state" section here: a per-file walkthrough of
an in-progress cherry-pick of `e1fa07e Require boot-scoped GPU startup authority`, listing
seven conflicted paths (`dashboard/README.md`, `trainvm/include/trainvm/linux_nvidia_inventory.hpp`,
`trainvm/src/host_resources.cpp`, `trainvm/src/hostd_daemon_runtime.cpp`,
`trainvm/tests/host_resources_tests.cpp`, `trainvm/tests/linux_nvidia_inventory_tests.cpp`, and
a modify/delete group under `deploy/`), plus a follow-on instruction to port `e5dc54e Qualify
privileged hostd authority boundary` for read-only journal access.

**That section is deleted rather than preserved, because the work it describes is finished and
leaving it in the document invites someone to redo resolved conflict work.**

Verified in the `card/qwen36-multimodal-dpo` worktree on 2026-08-09 before deleting it:

- No `CHERRY_PICK_HEAD`, `MERGE_HEAD`, `REVERT_HEAD`, `rebase-merge`, or `rebase-apply` exists
  in that worktree's git directory.
- `git diff --name-only --diff-filter=U` reports no unmerged paths.
- A recursive search finds no conflict markers anywhere in the tree.
- The branch HEAD had moved on to `af67529 Skip live boot observation where procfs cannot be
  pinned` (2026-08-07), and its pull request is merged.

If that history is ever needed, it is recoverable from this file's parent commit and from the
`card-74d0f9e5` card body.

## Why this compatibility work is required

The live service unit at `/etc/systemd/system/trainvm-controller.service` expects:

- `trainvm serve ... --worker-socket-gid 1001`
- boot-materialized hostd configuration
- boot-scoped GPU authorization
- a published `/run/trainvm-hostd/client.json`
- cooperative display-GPU authority

The DPO branch originally lacked these production integration commits. Incremental promotion exposed these failures in order, all before model training:

1. controller CLI rejected `--worker-socket-gid` — fixed by `fb95f4e`.
2. controller rejected live journal connection PRAGMAs — fixed by `f66b6ec`.
3. controller could not decode old `cooperative_compute` journal records — fixed by `f220055`.
4. controller compared full `st_mode` against permission-only endpoint identities — fixed by `9b985ae`.
5. old hostd lacked passive-memory status fields expected by the DPO controller.
6. matching DPO hostd lacked boot materialization/GPU-authorization CLI — `eeeeca5` plus the in-progress `e1fa07e` address this.

No trainer has started and these control-plane attempts did not acquire the GPU.

## Live service state at handoff

At the time of writing:

- `/usr/local/bin/trainvm` is the DPO controller build.
- `/usr/local/sbin/trainvm-hostd` is the DPO hostd build **before** the in-progress GPU-authorization cherry-pick, so it cannot currently satisfy the service unit.
- `trainvm-controller.service` appears active but is only the wrapper waiting for a fresh hostd client document.
- `trainvm-hostd.service` is inactive/skipped because its installed binary rejected `--check-gpu-authorization`.
- No `/run/trainvm-controller/trainvm.sock` exists yet.
- No training process is running.

Use these checks rather than trusting `systemctl is-active` alone:

```bash
systemctl status trainvm-controller.service trainvm-hostd.service --no-pager -l
ls -l /run/trainvm-controller/trainvm.sock /run/trainvm-hostd/client.json
sudo -n journalctl -u trainvm-controller.service -u trainvm-hostd.service --since '10 minutes ago' --no-pager -n 200
```

## Build, validation, and safe installation

Build directory:

`/thearray/git/moe-mla-dashboard-vm/.claude/worktrees/qwen36-multimodal-dpo/trainvm/build-dpo`

After resolving code:

```bash
cmake --build trainvm/build-dpo --target trainvm trainvm-hostd -j 8
cmake --build trainvm/build-dpo --target host_resources_tests resource_request_builder_tests linux_nvidia_inventory_tests -j 8
ctest --test-dir trainvm/build-dpo --output-on-failure -R '^(host_resources_tests|resource_request_builder_tests|linux_nvidia_inventory_tests)$'
```

Sudoers only permits installation from fixed paths in the parity worktree. Preserve the existing bytes at those paths, temporarily stage the new binary there, run the exact allowed install command, and restore the staged source bytes afterward.

Controller allowed source/destination:

```text
/thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm
/usr/local/bin/trainvm
```

Hostd allowed source/destination:

```text
/thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm-hostd
/usr/local/sbin/trainvm-hostd
```

Exact allowed installs:

```bash
sudo -n /usr/bin/install -o root -g root -m 0755 /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm /usr/local/bin/trainvm
sudo -n /usr/bin/install -o root -g root -m 0755 /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm-hostd /usr/local/sbin/trainvm-hostd
```

Then start hostd with the exact sudoers-authorized form:

```bash
sudo -n /usr/bin/systemctl start trainvm-hostd.service
```

If needed, restart controller first with:

```bash
sudo -n /usr/bin/systemctl restart trainvm-controller.service
```

Do not add flags such as `--no-block`; sudoers rejects altered command lines.

## Already installed live registries

These were merged with the five existing adapter/host profiles rather than replacing them:

- `/etc/trainvm/adapters.json`
  - SHA-256: `b00064cd1df82a713858012aba3654aaafd5ff094f296ce22995e76f28acfab9`
- `/etc/trainvm/host-launches.json`
  - SHA-256: `606b81d1e2913ce66ea9e18b0fecb3b5ee6efc2151ca0e6c2e2c88d7fac8646d`
- `/etc/trainvm/training-components.json`
  - SHA-256: `040ac1b72ecb862d838b0bb791670804d0fd67836ee7808dff0dc0c816421fb5`

Backup of the pre-promotion live files:

`/thearray/git/moe-mla/runs/trainvm-dpo-live-backup.FyhAd1`

Worker deployment:

`/thearray/git/moe-mla/runs/trainvm-worker-deployment-qwen36-dpo-20260806`

- worker artifact: `rwkv-lab-hf-multimodal-dpo.pyz`
- worker SHA-256: `25cec73cf5b3e3b7a2a28747eddab6b779bf8efde36c7a54538cd89606a25043`
- runtime closure digest: `sha256:c8d998eefc7e6c2419d95d20aedc1725d7e27dbfbfadde7525788f105de5afc4`

Isolated Python runtime:

`/thearray/git/moe-mla-dashboard-vm/.venvs/qwen36-caption-dpo/bin/python-worker`

## Training inputs

Model:

`/thearray/git/ob/text-generation-webui/models/Qwen3.6-35B-A3B-heretic-caption-step000745-merged-bf16`

Frozen DPO dataset:

`/thearray/git/datasets/captiontest/qwen36-caption-dpo-r256-v1`

- train pairs: 11,906
- validation: 680
- test: 674
- target manifest SHA-256: `43e46dc63bb94aa34c2c3afde7aa249dda0e1dcb2539b530152864928d46ef04`
- frozen dataset manifest SHA-256: `0ccc4e9a2d83f8fae2631107b42087c97b79170e6751a9b542e3169571bef8e6`
- exact-token audit: 11,906 pairs / 23,812 responses passed
- fixed gallery: 10 validation examples

Recipe registry:

`/thearray/git/moe-mla-dashboard-vm/.claude/worktrees/qwen36-multimodal-dpo/docs/experiment-vm/examples/hf-multimodal-sft.recipe-profiles.v1.json`

Recipe instance:

`/thearray/git/moe-mla-dashboard-vm/.claude/worktrees/qwen36-multimodal-dpo/docs/experiment-vm/examples/qwen36-caption-dpo-lora-r256.recipe-instance.v1.json`

Run identity: `qwen36-caption-dpo-lora-r256-v1`

Key settings:

- LoRA rank 256, alpha 512
- LR `5e-6`
- microbatch 1
- gradient accumulation 16
- 745 optimizer steps
- DPO beta 0.1
- evaluation at step 0, every 100 steps, and final
- checkpoint every 250 steps
- 10 qualitative gallery examples
- 90 GiB GPU admission floor
- cooperative display-GPU authority (`exclusive: false` in the recipe profiles)

## AuthorRun submission

> **Do not run this sequence on the strength of this document.** It is recorded for its shape,
> not as a green light. Two things must be settled first, both of which need the user:
> whether the 2026-08-06 instruction still stands, and whether the step-zero assumptions below
> survived `98b431d` arming the universal step-zero gate for the HF SFT family on 2026-08-09.
> Note also that every absolute path in this section points into the
> `.claude/worktrees/qwen36-multimodal-dpo` worktree, which may since have been removed; the
> recipe instance itself is on `main` at
> `docs/experiment-vm/examples/qwen36-caption-dpo-lora-r256.recipe-instance.v1.json`.

Once both services are genuinely healthy and `/run/trainvm-controller/trainvm.sock` exists, submit through the gRPC `AuthorRun` API. Python stubs already exist at:

`src/trainvm/v1/trainvm_pb2.py` and `src/trainvm/v1/trainvm_pb2_grpc.py`

Use:

- Python: `/thearray/git/moe-mla-dashboard-vm/.venvs/qwen36-caption-dpo/bin/python-worker`
- `PYTHONPATH=/thearray/git/moe-mla-dashboard-vm/.claude/worktrees/qwen36-multimodal-dpo/src`
- gRPC target: `unix:/run/trainvm-controller/trainvm.sock`

Author document shape:

```json
{
  "api_version": "trainvm.author-run/v1",
  "source": {
    "recipe": {
      "registry_path": "/thearray/git/moe-mla-dashboard-vm/.claude/worktrees/qwen36-multimodal-dpo/docs/experiment-vm/examples/hf-multimodal-sft.recipe-profiles.v1.json",
      "instance": { "...": "contents of qwen36-caption-dpo-lora-r256.recipe-instance.v1.json" }
    }
  },
  "author": "codex",
  "reason": "Launch validated rank-256 cached-reference DPO caption preference run"
}
```

Call sequence:

1. `AuthorRunRequest(request_document=..., source_format="json", dry_run=true)`
2. Stream to terminal update; require success and capture `plan_hash`.
3. Repeat with `dry_run=false` and `expected_plan_hash=<preview plan hash>`.
4. Capture run ID and dashboard URL.

The dry run measures the actual model/data content, so the final plan hash will differ from the old unmeasured expansion digest `sha256:a22c2cd64a5394bf6be08bec5af0327de2c95694087506ad66bde25c06df4da2`.

## Completion checks

After launch:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
sudo -n journalctl -u trainvm-controller.service -u trainvm-hostd.service --since '5 minutes ago' --no-pager -n 240
```

Confirm all of the following:

1. AuthorRun launch terminal update contains a run identity and no diagnostics.
2. The run is present in the live journal and normal dashboard run list.
3. The worker is launched from the sealed DPO artifact/runtime closure.
4. Step-zero alignment/evaluation runs before optimizer step 1.
5. Teacher target, step-zero baseline, and current caption examples are associated with the same fixed validation images.
6. GPU memory/utilization appears in the dashboard.
7. Training reaches at least its first heartbeat/metric without a launch or CUDA error.

Do not hide a failure behind the systemd wrapper's `active` state.
