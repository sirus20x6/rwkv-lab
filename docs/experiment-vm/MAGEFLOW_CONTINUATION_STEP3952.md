# MageFlow terminal continuation from optimizer step 3952

This handoff resumes the last coherent routed photo/animation terminal-expert
run without mutating its legacy checkpoint. The copy-on-write snapshot is:

`/thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-inputs-step3952/checkpoint-00003952`

The experiment preserves the complete original 12-block MageFlow path, the
three-block selected terminal expert, TREAD factored looping, VAE-REPA,
immiscible assignment, the final four trainable backbone blocks, and the
backbone learning rate at one half of the expert learning rate. It resumes at
step 3952 and targets step 12228. `eval_on_resume` is enabled and the training
node publishes the generated/original gallery directly to TrainVM.

## Ordered launch gate

1. Run full current-head acceptance, including a fresh native build, and commit
   the accepted source. Never install or qualify the stale pre-reboot binaries.
2. Install the accepted binaries, boot-safe controller/hostd units, corrected
   template, qualification registries, and training-component registry. Grant
   explicit boot-scoped GPU authority only after the machine is free, then
   require an admitted startup audit before the controller may mutate the
   journal.
3. Run the five-step `26fe458` MageFlow production qualification, including its
   checkpoint-first releasing pause and post-resume optimizer progress proof.
4. Materialize a fresh worker deployment containing
   `rwkv-lab.mageflow-terminal-expert`;
   the sparse `26fe458` qualification deployment contains only the full-backbone
   MageFlow adapter and cannot launch this continuation.
5. Reflink the existing frozen encoder cache to
   `/mnt/hypercard/mageflow-cache/trainvm-terminal-continuation-step3952/entries`
   and make only that snapshot readable by worker GID 1001. The source cached
   tensors remain owner-only (`0600`) and unmodified.
6. Content-lock the experiment using the checked-in root set. The roots are
   deliberately narrow: the two manifests, selected training image tree, the
   three image roots referenced by the mixed held-out eval manifest, cache
   entries, schedule, staged checkpoint, and base model. Unused pool metadata
   is not hashed.
7. Provision the exact new run directory as group-writable for worker GID
   1001. The worker has no authority to create paths beneath the broader
   `runs/` parent.
8. Submit only the locked document through the dashboard. Verify that the
   startup evaluation contains both photo and animation generated/original
   pairs before accepting optimizer progress.

The source documents are:

- `experiments/trainvm_mageflow_terminal_continuation_step3952.json`
- `experiments/trainvm_mageflow_terminal_continuation_step3952.input-roots.json`

After a host reboot, verify the explicitly reviewed journal identity before
installing the hostd template. This check reads only filesystem metadata and
does not start a service, load NVML, or touch the GPU:

```bash
/thearray/git/moe-mla-parity-integration/scripts/check_hostd_template_journal_identity.sh \
  /thearray/git/moe-mla-parity-integration/deploy/trainvm-hostd.json
sudo /usr/bin/install -o root -g root -m 0600 \
  /thearray/git/moe-mla-parity-integration/deploy/trainvm-hostd.json \
  /etc/trainvm/hostd.template.json
```

The three inodes survived the August 3 reboot, while their mount device number
changed from 52 to 54. The reviewed template now binds the current exact tuple.
Do not make boot materialization infer or rewrite journal identity: a mismatch
requires this explicit review and reinstall rather than weakening the fence.

Only after current-head acceptance has rebuilt the three native executables,
install the accepted stack and the sparse qualification registries. The GPU
UUID must be the exact display-active device that the operator intends to make
available for cooperative compute; do not copy the placeholder literally:

```bash
test -x /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm
test -x /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm-hostd
test -x /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm-gpu-fault-observer
sudo /usr/bin/install -o root -g root -m 0755 \
  /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm \
  /usr/local/bin/trainvm
sudo /usr/bin/install -o root -g root -m 0755 \
  /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm-hostd \
  /usr/local/sbin/trainvm-hostd
sudo /usr/bin/install -o root -g root -m 0755 \
  /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm-gpu-fault-observer \
  /usr/local/sbin/trainvm-gpu-fault-observer
sudo /usr/bin/install -o root -g root -m 0644 \
  /thearray/git/moe-mla-parity-integration/deploy/trainvm-controller.service \
  /etc/systemd/system/trainvm-controller.service
sudo /usr/bin/install -o root -g root -m 0644 \
  /thearray/git/moe-mla-parity-integration/deploy/trainvm-hostd.service \
  /etc/systemd/system/trainvm-hostd.service
sudo /usr/bin/install -o root -g root -m 0644 \
  /thearray/git/moe-mla-parity-integration/deploy/trainvm-gpu-fault-observer.service \
  /etc/systemd/system/trainvm-gpu-fault-observer.service
sudo /usr/bin/install -d -o root -g webteam -m 0750 /etc/trainvm
sudo /usr/bin/install -o root -g root -m 0600 \
  /thearray/git/moe-mla-parity-integration/deploy/trainvm-hostd.json \
  /etc/trainvm/hostd.template.json
sudo /usr/bin/install -o root -g root -m 0644 \
  /thearray/git/moe-mla/runs/trainvm-worker-deployment-26fe458/adapters.json \
  /etc/trainvm/adapters.json
sudo /usr/bin/install -o root -g root -m 0644 \
  /thearray/git/moe-mla/runs/trainvm-worker-deployment-26fe458/host-launches.json \
  /etc/trainvm/host-launches.json
sudo /usr/bin/install -o root -g root -m 0644 \
  /thearray/git/moe-mla-parity-integration/docs/experiment-vm/examples/training-components.v1.json \
  /etc/trainvm/training-components.json
sudo /usr/bin/systemctl daemon-reload
sudo /usr/bin/systemctl enable --now trainvm-gpu-fault-observer.service
sudo /usr/bin/systemctl enable --now trainvm-controller.service
GPU_UUID=GPU-replace-with-reviewed-display-device-uuid
[[ "$GPU_UUID" =~ ^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]
sudo /usr/local/sbin/trainvm-hostd --authorize-gpu-start \
  /etc/trainvm/hostd.template.json \
  /etc/trainvm/hostd-gpu-authorization.json \
  cooperative_allowlist "$GPU_UUID"
sudo /usr/bin/systemctl enable --now trainvm-hostd.service
```

Starting the controller before hostd is intentional: it publishes its stable
service/cgroup identity and waits for hostd to atomically replace the client
authority document. Hostd then admits that exact controller and releases the
wait. The boot-scoped authorization command is the first operation in this
sequence permitted to probe the GPU; therefore run it only in the free-machine
window.

After acceptance has produced a clean commit, seal the exact terminal worker
into the path already authorized by the sudoers policy:

```bash
test -z "$(git -C /thearray/git/moe-mla-parity-integration \
  status --porcelain=v1 --untracked-files=all)"
test ! -e /thearray/git/moe-mla/runs/trainvm-worker-deployment-terminal-step3952-v1
/thearray/git/moe-mla-parity-integration/scripts/materialize_trainvm_worker_deployment.py \
  --trainvm /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm \
  --adapter rwkv-lab.mageflow-terminal-expert \
  --adapter-python rwkv-lab.mageflow-terminal-expert=/thearray/git/moe-mla/.venv-mage-flow/bin/python-trainvm \
  --source-root /thearray/git/moe-mla-parity-integration/src \
  --output-directory /thearray/git/moe-mla/runs/trainvm-worker-deployment-terminal-step3952-v1 \
  --working-directory /thearray/git/moe-mla-parity-integration \
  --runtime-digest-cache /thearray/git/moe-mla/runs/trainvm-runtime-digests.json \
  --trusted-root /thearray/git/moe-mla-parity-integration \
  --trusted-root /thearray/git/moe-mla/.venv-mage-flow \
  --trusted-root /thearray/git/moe-mla/runs/trainvm-worker-deployment-terminal-step3952-v1 \
  --trusted-root /usr
jq -e '
  .source_dirty == false and
  ([.runtime_groups[].adapters[]] == ["rwkv-lab.mageflow-terminal-expert"])
' /thearray/git/moe-mla/runs/trainvm-worker-deployment-terminal-step3952-v1/deployment-receipt.json
```

The deployment destination must not already exist. Stop hostd before changing
the root-installed registry pair, require `PartOf=` to stop the controller too,
then install and restart through the exact sudoers-authorized commands:

```bash
sudo /usr/bin/systemctl stop trainvm-hostd.service
! /usr/bin/systemctl is-active --quiet trainvm-controller.service
sudo /usr/bin/install -o root -g root -m 0644 \
  /thearray/git/moe-mla/runs/trainvm-worker-deployment-terminal-step3952-v1/adapters.json \
  /etc/trainvm/adapters.json
sudo /usr/bin/install -o root -g root -m 0644 \
  /thearray/git/moe-mla/runs/trainvm-worker-deployment-terminal-step3952-v1/host-launches.json \
  /etc/trainvm/host-launches.json
sudo /usr/bin/systemctl start trainvm-hostd.service
sudo /usr/bin/systemctl is-active trainvm-hostd.service
sudo /usr/bin/systemctl is-active trainvm-controller.service
```

Run the accepted dashboard binary against the supervised controller socket.
This command deliberately omits `-gpu-telemetry`; TrainVM/hostd remains the
only GPU authority and the dashboard stays CPU-only:

```bash
/thearray/git/moe-mla-parity-integration/dashboard/trainboard \
  -repo /thearray/git/moe-mla-parity-integration \
  -runs /thearray/git/moe-mla/runs \
  -db /thearray/git/moe-mla-parity-integration/dashboard/trainboard.db \
  -addr 127.0.0.1:9124 \
  -trainvm-bin /thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm \
  -trainvm-schema /thearray/git/moe-mla-parity-integration/docs/experiment-vm/experiment-v1.schema.json \
  -trainvm-socket /run/trainvm-controller/trainvm.sock \
  -image-roots /thearray
```

In a second terminal, require the complete admitted-and-idle path before
qualification or continuation submission:

```bash
/thearray/git/moe-mla-parity-integration/scripts/check_trainvm_launch_readiness.sh \
  http://127.0.0.1:9124
```

An active controller unit without its Unix socket, a dashboard that cannot
issue commands, a failed/stale hostd audit, poisoned lease renewal, degraded
inventory, or a leftover fence/process all fail this gate.

Create the new-only cache snapshot before content locking:

```bash
test ! -e \
  /mnt/hypercard/mageflow-cache/trainvm-terminal-continuation-step3952/entries
install -d -m 0750 \
  /mnt/hypercard/mageflow-cache/trainvm-terminal-continuation-step3952
cp -a --reflink=always \
  /mnt/hypercard/mageflow-cache/mageflow-steps-00000501-00005500/entries \
  /mnt/hypercard/mageflow-cache/trainvm-terminal-continuation-step3952/entries
find /mnt/hypercard/mageflow-cache/trainvm-terminal-continuation-step3952 \
  -type d -exec chmod 0750 {} +
find /mnt/hypercard/mageflow-cache/trainvm-terminal-continuation-step3952 \
  -type f -exec chmod 0640 {} +
```

The destination must not already exist. `--reflink=always` fails rather than
silently duplicating the cache bytes when the filesystem cannot provide a
copy-on-write snapshot.

The content-lock command, after the machine is released, is:

```bash
test ! -e \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952.locked.json
/thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm \
  lock-input-content \
  /thearray/git/moe-mla-card-mageflow-continuation/experiments/trainvm_mageflow_terminal_continuation_step3952.json \
  /thearray/git/moe-mla-card-mageflow-continuation/experiments/trainvm_mageflow_terminal_continuation_step3952.input-roots.json \
  > /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952.locked.json
/thearray/git/moe-mla-parity-integration/trainvm/build-acceptance/trainvm \
  validate \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952.locked.json
/thearray/git/moe-mla-parity-integration/scripts/check_mageflow_continuation_contract.sh \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952.locked.json
```

Do not run the lock concurrently with cache permission changes. The worker
recomputes every declared content identity at startup, so the locked document
must describe the final readable cache tree. The intent-specific contract gate
then rechecks the two mandatory expert routes, final-backbone-third fraction,
half-rate backbone, REPA/TREAD/immiscible settings, eval/checkpoint cadence,
exact ten content roots, and the staged step-3952 state inventory; generic
schema validation alone does not establish those experiment-specific choices.

Provision the new output directory only after the locked document validates:

```bash
test ! -e \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952
install -d -m 0770 -g 1001 \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952
test "$(stat -c %g \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952)" = 1001
test "$(stat -c %a \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952)" = 770
```

Never reuse a partial run directory under the same experiment identity; move
an interrupted pre-launch directory aside and relock/resubmit under a new
identity instead.

Submit through the generic fenced dashboard client. It recompiles the locked
source, captures the current journal plus adapter/component lock identities,
writes the exact idempotent request before transmission, and prints the native
run ID:

```bash
/thearray/git/moe-mla-parity-integration/scripts/submit_locked_trainvm_experiment.sh \
  http://127.0.0.1:9124 \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952.locked.json \
  mageflow-terminal-step3952 \
  "continue accepted MageFlow terminal experts from frozen step 3952" \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952-submission
```

If the final POST has an ambiguous transport failure, do not regenerate a new
request. Resend the preserved
`trainvm-mageflow-terminal-continuation-step3952-submission/submission-request.json`
byte-for-byte; its idempotency key and all authority fences are already fixed:

```bash
/thearray/git/moe-mla-parity-integration/scripts/retry_locked_trainvm_submission.sh \
  http://127.0.0.1:9124 \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952-submission
```

The retry refuses to regenerate any field, requires the current journal to
match the preserved fence, and refuses to run when a nonempty response already
exists. It is safe if the original request succeeded but its HTTP response was
lost: native idempotency must return the same run identity.

## Startup-evaluation acceptance

The held-out manifest contains exactly 256 rows: 128 photo (64 Midjourney and
64 Reddit) and 128 animation (64 Midjourney and 64 Civitai). All 256 referenced
source images exist and their image IDs are unique. Scalar evaluation covers
the manifest under the same evaluation step; the gallery deterministically
publishes eight side-by-side items, four per route.
Before treating the resumed optimizer as healthy, require all of the following
from the dashboard's first step-3952 evaluation revision:

- `eval/gallery_samples` is 8;
- four items report the `photo` route and four report `animation`;
- every item has both an immutable generated object and its held-out target;
- the gallery condition digest is bound to this eval manifest;
- the selected route changes the generated result rather than bypassing both
  experts; and
- the run remains at optimizer step 3952 until the initial evaluation has
  completed and been published.

Only then accept a later optimizer step, finite loss, expert/base LR ratio, and
new checkpoint as proof that continuation training—not merely evaluation—has
started.

The GPU-passive dashboard verifier applies the machine-readable portion of
that gate and fetches all sixteen immutable image objects through the same
path-safe URLs used by the browser:

```bash
/thearray/git/moe-mla-parity-integration/scripts/check_mageflow_startup_gallery.sh \
  http://127.0.0.1:9124 RUN_ID 3952
```

It fails if the step-3952 revision is absent, contains anything other than four
photo and four animation items, lacks generated/target pairs, loses manifest or
checkpoint binding, duplicates an item identity, or cannot serve an image. The
checkpoint binding is an exact digest join: the gallery's declared checkpoint
manifest must match a step-3952 checkpoint whose immutable manifest the
dashboard has independently loaded and verified.

## Optimizer-progress acceptance

After accepting the startup gallery, require two routed updates and the first
new immutable checkpoint before calling the resumed training healthy:

```bash
/thearray/git/moe-mla-parity-integration/scripts/check_mageflow_optimizer_progress.sh \
  http://127.0.0.1:9124 \
  RUN_ID \
  /thearray/git/moe-mla/runs/trainvm-mageflow-terminal-continuation-step3952 \
  3952 \
  4000
```

This verifier tolerates a partially appended final `train.jsonl` line, but not
a missing or malformed completed optimizer record. It requires post-resume
updates from both the animation and photo routes, finite loss, positive
throughput, a positive expert learning rate, the shared-backbone learning rate
at exactly one half (within numeric tolerance), and the VAE-REPA projection at
the expert rate. The dashboard projection must have caught up to the local
append-only log and must not report failure or cancellation.

Finally, the dashboard must expose a cryptographically verified checkpoint at
or beyond step 4000 with model, optimizer, optimizer-group, expert-routing,
parameter-routing, cursor, scheduler, and RNG state. This is intentionally a
later gate than process startup: a live PID, heartbeats, or a completed startup
evaluation alone are not evidence that a recoverable optimizer continuation is
working.
