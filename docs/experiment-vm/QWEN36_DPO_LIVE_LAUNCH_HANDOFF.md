# Qwen3.6 caption DPO live-launch handoff

Updated: 2026-08-06 11:36 US/Central

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

There is currently an in-progress cherry-pick of `e1fa07e Require boot-scoped GPU startup authority`. Finish it; do not discard the DPO/passive-memory changes.

## Current conflict state

Run:

```bash
cd /thearray/git/moe-mla-dashboard-vm/.claude/worktrees/qwen36-multimodal-dpo
git status --short
rg -n '<<<<<<<|=======|>>>>>>>' dashboard trainvm
```

Expected unresolved files at handoff:

- `dashboard/README.md` — already resolved by keeping the GPU-passive/operator-authorization text.
- `trainvm/include/trainvm/linux_nvidia_inventory.hpp` — already resolved by keeping the incoming context-owner/display-sharing label constants.
- `trainvm/src/host_resources.cpp` — already resolved; retain the braced implementation that permits `occupied` only for `cooperative_compute`, and requires the exact `display-sharing=operator-authorized-cooperative` label on display GPUs.
- `trainvm/src/hostd_daemon_runtime.cpp` — already partly resolved. The early pre-journal NVML inventory construction was removed. Retain the incoming order: acquire ledger singleton, bind socket, then construct the authorized NVML inventory collector and capture inventory. This preserves both the GPU-authorization boundary and the newer passive-memory status implementation.
- `trainvm/tests/host_resources_tests.cpp` — remove conflict markers and keep the entire incoming workstation/cooperative display test block.
- `trainvm/tests/linux_nvidia_inventory_tests.cpp` — already resolved by running **both** `free_memory_is_separate_from_inventory_identity()` and `display_sharing_requires_an_exact_authorized_uuid()`.
- `deploy/install-hostd-sudoers.sh`, `deploy/trainvm-hostd.service`, and `tests/test_trainvm_hostd_deployment.py` are modify/delete conflicts. These paths were intentionally deleted on the DPO branch; resolve by keeping them deleted.

The hostd runtime must eventually open the controller journal read-only. The current DPO branch predates `e5dc54e Qualify privileged hostd authority boundary`, so after completing `e1fa07e`, port/cherry-pick `e5dc54e` or minimally bring in `JournalAccessMode::read_only` and change the hostd `Journal` construction to:

```cpp
Journal(..., false, JournalAccessMode::read_only)
```

The full commit may conflict because the DPO branch has newer passive-memory code. Prefer preserving DPO features and production journal compatibility.

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
