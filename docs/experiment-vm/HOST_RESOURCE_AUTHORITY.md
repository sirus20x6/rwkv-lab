# Host resource authority

Status: implementation contract for the P0 safe-execution gate

Scope: single-host resource allocation, process launch, crash recovery, and startup orphan audit

This contract is model-family-neutral. It applies equally to flow/diffusion, language, vision,
multimodal, distillation, post-training, evaluation, and future registered workers. Model code never
decides whether a host resource is free.

## Decision

Add one compiled C++ host daemon, referred to here as **hostd**, as the authority for physical host
resources and launched processes. Every TrainVM service, regardless of journal path, requests
physical resources and process operations from the same hostd instance.

Journal-local leases remain authoritative for logical run and workspace ownership. They are not a
physical GPU fence. A process may launch only when both authorities agree:

1. the run owns a live, boot-scoped journal lease; and
2. hostd has issued a live resource-bundle grant whose exact receipt is recorded by the journal.

The following invariants are mandatory:

- at most one conflicting live host grant exists for a physical resource;
- a multi-resource request is granted completely or not at all;
- physical resource generations are monotonic across journals and hostd restarts;
- no process starts running before its grant, launch intent, and process identity are durable;
- an expired lease does not make a resource reusable while a process, descendant, cgroup, or driver
  context may still use it;
- PIDs are observations, never identities by themselves;
- startup auditing completes before hostd enables grant or launch operations;
- unknown, unreadable, inconsistent, or partially observed state fails closed;
- legacy process-name scans and utilization samples are telemetry only.

No P0 feature may fall back to a journal-local lock, an abstract Unix socket, `CUDA_VISIBLE_DEVICES`,
or dashboard process detection when hostd is unavailable.

## Why the existing boundaries are insufficient

`AuthorityLock` protects one journal namespace and inode. Its co-located filesystem locks are useful
for one journal, but two services using different journals do not conflict. Its abstract Unix socket
is also scoped to a Linux network namespace and cannot be a host-wide fence.

The current `resource_leases` table is keyed by a plan-provided `concurrency_key`. Its fencing token
is monotonic only inside that journal, and it contains no physical accelerator UUID, partition, or
topology identity. Two journals can therefore both believe that they own the same GPU.

`HostLaunchRegistry` is an immutable allowlist for executable profiles. `HostLaunchResolver` seals
the selected bytes and working directory, but deliberately does not fork, execute, create cgroups,
or own a process. These types remain the code-authority boundary; hostd is a separate resource and
process authority.

The original dashboard samples `nvidia-smi` and recognizes a bounded list of Python command lines.
That misses renamed scripts, native workers, grandchildren, MPS servers, foreign compute processes,
and processes outside the dashboard's visible PID namespace. Its `gpuFree()` result must never
authorize a launch.

## System boundary

```text
        journal A / TrainVM A          journal B / TrainVM B
             logical lease                 logical lease
                  |                              |
                  +---------- typed RPC ----------+
                                 |
                                 v
                 +-------------------------------+
                 | trainvm-hostd                 |
                 | inventory and topology        |
                 | bundle grants and generations |
                 | launch/cgroup/pidfd authority |
                 | startup orphan audit          |
                 +----------+--------------------+
                            |
                  durable host ledger
                            |
                 cgroup v2 / device BPF / drivers
                            |
                     sealed worker process

        Go dashboard -------- read-only status/telemetry --------^
```

Hostd is a systemd-owned singleton for the physical host. The default deployment paths are:

- filesystem Unix socket: `/run/trainvm/hostd.sock`;
- durable ledger: `/var/lib/trainvm/host-resource.db`;
- worker cgroups: a root-owned subtree beneath `trainvm-workers.slice`.

Paths may be configured only by a root- or dedicated-authority-owned startup configuration. They are
not experiment fields.

Hostd uses a filesystem `AF_UNIX` `SOCK_SEQPACKET` endpoint with peer credentials and preferably
systemd socket activation. It does not use the abstract Unix namespace. Services in another mount
namespace must receive the same socket through an explicit bind mount or an inherited descriptor.
Failure to reach that shared endpoint is a hard failure, not permission to start another hostd.

## Authority and threat model

The P0 deployment has two supported enforcement grades:

### Strict host enforcement

A root or dedicated privileged hostd owns the ledger, socket, cgroup subtree, and cgroup-device BPF
programs. TrainVM services and workers run with less privilege. Peer authorization includes kernel
credentials and a root-owned service-cgroup identity. Workers cannot modify their cgroup membership
or device allowlist.

Strict mode is required when TrainVM must defend against an untrusted local process running under a
different service identity.

### Cooperative same-UID enforcement

All services may run under one user account for development. Hostd still prevents accidental
double allocation, stale journal decisions, and crash/restart races. It does not defend against a
hostile same-UID process: such a process can generally signal or inspect peers, connect to a
user-owned socket, and open device nodes outside hostd's launcher.

Hostd reports its enforcement grade. A plan or site policy requesting strict isolation is rejected
unless privileged cgroup-device enforcement and a protected client identity are active. The system
must not describe cooperative mode as a security boundary.

## Typed C++ data model

Persisted types use explicit API versions and canonical serialization. C++26 reflection may generate
decoders and descriptors, but compiler reflection metadata is never persisted.

The following declarations show the required shape. Exact field bounds and serialization tags belong
in the implementation schema.

```cpp
enum class HostResourceKind {
  accelerator,
  accelerator_partition,
  host_mutex,
};

enum class AcceleratorVendor { nvidia, amd, intel, other };

enum class ResourceAccessMode {
  exclusive_compute,
  exclusive_device,
  partition_exclusive,
};

enum class TopologyPolicy {
  any,
  same_numa_node,
  same_pcie_root,
  same_fabric_clique,
  exact_resources,
};

struct HostResourceId {
  HostResourceKind kind;
  std::optional<AcceleratorVendor> vendor;
  std::string stable_id;               // GPU UUID, MIG UUID, or canonical mutex ID
  std::optional<std::string> parent_id; // parent GPU for a partition
};

struct ResourceSelector {
  std::optional<AcceleratorVendor> vendor;
  std::optional<std::uint64_t> minimum_memory_bytes;
  std::map<std::string, std::string> exact_labels;
  std::vector<HostResourceId> exact_resources;
};

struct ResourceBundleRequest {
  std::string api_version;
  std::string request_id;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token;
  std::uint32_t count;
  ResourceAccessMode access_mode;
  TopologyPolicy topology;
  ResourceSelector selector;
  std::string canonical_request_digest;
};

struct ResourceFence {
  HostResourceId resource;
  std::uint64_t generation;
  std::string inventory_digest;
};

struct ResourceBundleGrant {
  std::string api_version;
  std::string allocation_id;
  std::string request_id;
  std::string request_digest;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::vector<ResourceFence> fences; // canonical stable-ID order
  std::int64_t granted_boottime_ns;
  std::int64_t granted_wall_time_ns;
  std::string previous_receipt_digest;
  std::string receipt_digest;
};
```

The complete bundle digest, not one selected token, is the process resource identity. Worker hello,
control acknowledgement, spawn, exit, and release receipts carry that digest. Any external effect
that requires accelerator authority validates the exact active bundle.

### Inventory and topology

Logical indices such as CUDA device `0` are never stable identities. NVIDIA resources use GPU UUIDs
and MIG device UUIDs. AMD and other vendors use a vendor UUID when available and retain PCI BDF and
DRM render-node metadata as observed evidence. PCI BDF alone is not assumed stable across hardware
replacement.

```cpp
struct ObservedHostResource {
  HostResourceId id;
  std::optional<std::string> pci_bdf;
  std::optional<std::uint32_t> device_major;
  std::optional<std::uint32_t> device_minor;
  std::optional<std::int32_t> numa_node;
  std::uint64_t total_memory_bytes;
  std::map<std::string, std::string> labels;
};

struct HostInventoryReceipt {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::vector<ObservedHostResource> resources;
  std::string topology_digest;
  std::string receipt_digest;
};
```

Inventory is sorted by stable ID. A full-device grant conflicts with every child partition. A
partition grant conflicts with its parent and the same partition, while nonoverlapping sibling
partitions may coexist. NVIDIA MPS is unsupported at P0 unless hostd owns and identifies the MPS
server itself.

`exclusive_compute` rejects unmanaged compute contexts but may coexist with explicitly classified
host graphics use. `exclusive_device` requires no foreign compute or graphics context.
`partition_exclusive` applies to an enumerated hardware partition. Driver memory used by allowed host
graphics remains part of capacity checks.

Topology is an observed, content-addressed property. Selection is deterministic. A disappearing UUID,
changed parent relationship, or incompatible topology digest marks the allocation degraded; hostd
never silently substitutes a different device.

### Process and terminal receipts

```cpp
struct DurableProcessIdentity {
  std::string host_id;
  std::string boot_id;
  std::int32_t host_pid;
  std::uint64_t proc_starttime_ticks;
  std::uint64_t cgroup_inode;
  std::string cgroup_relative_path;
  std::uint64_t pid_namespace_inode;
  std::uint64_t mount_namespace_inode;
  std::uint64_t user_namespace_inode;
  std::uint64_t network_namespace_inode;
  std::uint32_t uid;
  std::uint32_t gid;
  std::string executable_digest;
};

struct SpawnIntentReceipt {
  std::string api_version;
  std::string allocation_id;
  std::string grant_digest;
  std::string resolved_launch_digest;
  std::string launch_nonce;
  std::string cgroup_identity_digest;
  std::string receipt_digest;
};

struct SpawnReceipt {
  std::string api_version;
  std::string allocation_id;
  std::string grant_digest;
  std::string resolved_launch_digest;
  std::string launch_nonce;
  DurableProcessIdentity process;
  std::int64_t spawned_boottime_ns;
  std::int64_t spawned_wall_time_ns;
  std::string receipt_digest;
};

struct ExitReceipt {
  std::string api_version;
  std::string allocation_id;
  std::string spawn_digest;
  std::optional<std::int32_t> exit_code;
  std::optional<std::int32_t> terminating_signal;
  bool cgroup_empty;
  bool accelerator_contexts_absent;
  std::string evidence_digest;
  std::string receipt_digest;
};

enum class ReleaseReason {
  normal_exit,
  launch_aborted,
  terminated_orphan,
  crashed_process,
  reboot_recovery,
};

struct ResourceReleaseReceipt {
  std::string api_version;
  std::string allocation_id;
  std::string grant_digest;
  std::optional<std::string> exit_digest;
  ReleaseReason reason;
  std::int64_t released_boottime_ns;
  std::int64_t released_wall_time_ns;
  std::string audit_epoch;
  std::string receipt_digest;
};
```

An OS pidfd is live authority evidence but is not durable. After hostd restart, a pidfd may be opened
only after boot ID, PID start time, cgroup identity, executable identity, and the durable spawn receipt
all match. A numeric PID is never signalled after any mismatch.

## Host ledger

Hostd owns one strict SQLite ledger on a local filesystem. The file and its directory are
root/dedicated-authority-owned, non-linkable, non-group/world-writable, and resolved without following
symlinks. The ledger has its own stable identity, exact schema attestation, hash chain, boot-scoped
clock evidence, and synchronous durability policy.

The append-only authority records are:

- inventory receipts;
- resource request and grant receipts;
- spawn intents and spawn receipts;
- exit and release receipts;
- orphan observations and decisions;
- startup audit receipts;
- client registration and enforcement-grade receipts.

Mutable tables are rebuildable projections only:

- current resource inventory;
- next generation by stable resource ID;
- current grant by resource ID and allocation ID;
- current process by allocation ID;
- current per-resource audit classification.

Every physical resource generation is allocated by hostd in the same transaction as its grant
receipt. Multi-resource selection, conflict checks, generation increments, and grant publication are
one transaction. Resource locks are acquired in canonical stable-ID order as a second in-process
defence, but lock lifetime is not treated as durable state.

The ledger is the authority for physical resources. A TrainVM journal stores the exact canonical
host receipt bytes and digest associated with its run. Hostd does not infer journal state by opening
arbitrary journal paths, and a journal ID alone is not client authentication.

## Client identity and API

A hostd client session is bound to the connected socket and kernel peer credentials. In strict mode,
hostd also verifies the caller's root-owned service cgroup and its configured role. A copied journal
ID or bearer string is insufficient.

The minimum typed API is:

- `GetInventory` and `GetAuditStatus` for read-only clients;
- `RegisterJournalAuthority` for an authorized TrainVM service session;
- `RequestBundle` and `InspectAllocation`;
- `PrepareLaunch`, accepting the resource grant plus the sealed `ResolvedLaunch` descriptors;
- `CommitLaunch` or `AbortLaunch` for the stopped launcher boundary;
- `ObserveProcess`, `RequestTermination`, and `ReleaseBundle`;
- `ReconcileJournalReceipt` for idempotent restart repair.

The Go dashboard receives inventory, occupancy, and audit status through read-only methods. It cannot
request a grant or launch a worker.

## Cross-ledger saga

Hostd and a TrainVM journal are separate durability domains. The implementation uses an idempotent,
fail-safe saga rather than claiming atomic two-phase commit.

| Stage | Durable fact | External effect allowed | Recovery rule |
|---|---|---|---|
| request | journal records logical lease and resource request digest | none | repeat the same request ID |
| grant | hostd commits bundle grant | resource reserved, no process | journal attaches exact receipt or hostd records an evidence-backed abort |
| journal grant copy | journal records canonical host grant | resolve/seal only | repeat by receipt digest |
| spawn intent | hostd commits cgroup and launch intent | child may be created stopped | audit intent before retry |
| spawn | hostd commits process identity | release child to exec/worker handshake | exact identity may be adopted; mismatch blocks |
| journal spawn copy | journal records spawn receipt | normal worker protocol | repeat by spawn digest |
| exit | hostd records pidfd/wait and cgroup/context evidence | none | keep resource blocked until terminal evidence is complete |
| host release | hostd commits release receipt | resource may be reallocated | journal copies receipt idempotently |

Crashes at any boundary leave either no external effect or a durable reservation that blocks reuse.
There is no timeout path that guesses a resource is free.

## Linux launch and enforcement

### Cgroup boundary

Hostd creates one root-owned cgroup-v2 subtree per allocation. Workers cannot write the parent
`cgroup.procs`, their device policy, or sibling cgroups. CPU set, CPU weight, I/O weight, memory, and
other declared generic host policies are applied here and recorded as effective evidence.

All worker descendants inherit the cgroup. `cgroup.events` and recursive membership provide the
TrainVM-owned process boundary. An empty top-level PID or exited launcher is not sufficient if a
grandchild remains.

`cgroup.freeze` may stabilize inspection and termination, but it is not proof of containment: Linux
permits process movement and describes races with fork and migration. Root-owned, nondelegated
membership is the containment control. `cgroup.kill`, pidfd signalling, and a confirmed empty cgroup
form the termination path.

### Stopped launch boundary

Hostd validates the active grant and the sealed `ResolvedLaunch` bundle, then creates a small sealed
launcher stub with `clone3(CLONE_INTO_CGROUP | CLONE_PIDFD)`. The stub waits on a private eventfd or
pipe before executing worker bytes.

Before releasing the stub, hostd:

1. opens and verifies the cgroup identity;
2. records the spawn intent;
3. creates the child directly inside that cgroup and receives its pidfd;
4. reads host PID, `/proc` start time, credential, namespace, cgroup, and executable evidence;
5. commits the spawn receipt;
6. returns the canonical receipt for journal persistence;
7. releases the stub only after the protocol's required durable acknowledgement.

Execution uses the already sealed descriptors, not a re-resolved pathname. Hostd rechecks memfd
seals and the resolved launch digest before `execveat`/equivalent execution.

Pidfds drive liveness, `waitid(P_PIDFD)`, and signalling. Hostd may act as a child subreaper for
diagnostics, but cgroup membership—not parentage—is the process-tree boundary.

### Device enforcement

Environment variables are not security controls. Hostd supplies stable UUIDs through
`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, or an adapter-equivalent only after enforcing the
grant.

Strict mode attaches `BPF_PROG_TYPE_CGROUP_DEVICE` policy to the allocation cgroup. It permits the
assigned accelerator or partition device nodes and the minimum required shared driver control nodes,
and denies unassigned accelerator device minors. Vendor mappings from UUID/partition to device nodes
are part of the inventory receipt.

Device-node filtering prevents new unauthorized opens. Driver inventory remains necessary to detect
contexts established before policy attachment, shared control-node behavior, device hotplug, and
foreign processes outside the TrainVM cgroup tree.

## Resource discovery

Authority probes are C++ interfaces with fake implementations for tests. Production NVIDIA probing
dynamically loads NVML to enumerate UUIDs, PCI identity, MIG hierarchy, memory, and running compute
and graphics processes. AMD and other vendors use stable vendor APIs where available plus sysfs/DRM
evidence. A missing optional vendor backend blocks that vendor's resources; it does not make them
free.

Hostd runs in the host PID and cgroup namespaces. It cross-checks:

- all cgroups beneath its protected subtree;
- every process in those cgroups, using pidfd and `/proc` identity;
- vendor-reported accelerator compute and graphics contexts;
- durable grants, spawn receipts, exits, and releases;
- currently registered TrainVM journal authorities.

Command line, executable basename, log freshness, utilization, memory use, and open-device-FD scans
may enrich an observation. None can prove ownership or absence.

## Startup audit state machine

Hostd exposes no mutating RPC until it has committed a startup audit receipt.

```text
BOOTING
   |
   v
VERIFY_LEDGER ---- corruption/foreign host ----> FATAL
   |
   v
CAPTURE_INVENTORY -- required probe unknown ----> AUDIT_BLOCKED
   |
   v
ENUMERATE_CGROUPS_AND_PROCESSES
   |
   v
MATCH_DURABLE_RECEIPTS
   |
   v
COMMIT_AUDIT
   |
   +---- unresolved resources ----> READY_WITH_BLOCKS
   |
   +---- all resources known -----> READY
```

`READY_WITH_BLOCKS` means the audit is complete and only explicitly unblocked resources may be
granted. `AUDIT_BLOCKED` means the audit could not establish a complete inventory and no affected
resource may be granted. `FATAL` requires operator repair or ledger recovery.

Each observed resource receives one classification:

| Classification | Required evidence | Scheduling consequence |
|---|---|---|
| free | no live grant, no owned process, no conflicting driver context | eligible |
| owned_live | matching grant, spawn identity, cgroup, and context | reserved; eligible for exact adoption only |
| reserved_no_process | live grant or intent without a process | blocked until journal reconciliation or durable abort |
| exited_pending_release | process exited but terminal evidence/release is incomplete | blocked |
| foreign_occupied | driver context outside an exact TrainVM identity | blocked; never killed by default |
| orphan_exact | process/cgroup matches a receipt but its authority is absent | policy decides adopt, leave, or terminate |
| orphan_unknown | process/cgroup/context cannot be matched exactly | blocked; never adopted |
| degraded | granted resource disappeared or topology changed | blocked and surfaced as failure |
| probe_unknown | required kernel/driver evidence is unavailable | blocked |

Journal unavailability does not stop hostd from finishing the physical inventory, but every allocation
owned by that journal remains blocked. Other disjoint, completely audited resources may become
eligible.

### Orphan policy

- `leave_and_block`: retain the grant and report the orphan. This is the default.
- `adopt_if_identity_matches`: require exact grant, launch digest, nonce, boot ID, PID start time,
  executable, cgroup, namespaces, and journal recovery state. Any mismatch blocks.
- `terminate_and_recover`: freeze where supported, signal through pidfd and/or `cgroup.kill`, wait for
  `populated=0`, verify driver contexts are gone, then record exit and release receipts. A timeout
  remains blocked.

An orphan policy never authorizes signalling a reused numeric PID or killing a foreign context.

## Crash and reboot semantics

### TrainVM service crash

The host grant and worker remain supervised by hostd. A restarted service reconnects with its
authorized journal identity and reconciles exact receipts. Until then, the grant stays reserved.

### Worker or launcher crash

Pidfd readiness produces an exit observation. Hostd waits for the allocation cgroup to become empty
and for vendor contexts to disappear before committing a release. A surviving grandchild keeps the
grant live or orphaned.

### Hostd crash

Systemd stops admission while hostd is absent and restarts the singleton. Existing workers may
continue in the separate protected worker slice. The replacement hostd begins in `BOOTING` and audits
all workers before accepting any grant or launch. In-memory locks are never used as recovery truth.

### Host reboot

The Linux boot ID changes, invalidating every persisted PID and boottime comparison. Hostd retains
the persistent resource generations. After the new driver inventory is complete, cgroups and
processes are absent, and no conflicting context exists, hostd writes explicit `reboot_recovery`
release receipts. It never adopts a process from a prior boot or resets a generation counter.

A machine/host identity mismatch means the ledger belongs to another host and is fatal; it is not a
reboot.

## Namespace rules

### Network namespaces

Linux network namespaces isolate the abstract Unix socket namespace. The existing per-journal
abstract lock therefore cannot be a host fence. Hostd uses a filesystem socket whose inode is shared
when the filesystem view is shared. Peer network namespace identity is retained in process receipts
for audit but is not the allocator namespace.

### Mount namespaces

A pathname socket is useful only when every authorized service reaches the same inode. Containerized
or sandboxed services receive the socket through a controlled bind mount or inherited descriptor.
Hostd records and validates the endpoint inode where applicable. A service with a private `/run` or
hidden hostd socket fails closed.

No service may start a second broker merely because its mount namespace cannot see the host broker.
The systemd host unit and protected ledger directory are the singleton boundary.

### PID and cgroup namespaces

Hostd runs in the initial/host PID and cgroup namespaces. Durable receipts store host PIDs and
namespace inode identities. A worker-visible PID is never used for adoption or signalling.
The strict Linux session authority pins and continuously reattests the configured cgroup namespace
alongside mount, PID, and both time namespace identities before service-cgroup membership can become
an authorization input.

### User namespaces and same UID

`SO_PEERCRED` proves the kernel peer process, not that every process with the same UID is trustworthy.
Strict authorization also requires a protected service/cgroup identity. User-namespace mappings are
recorded and checked; an unexpected mapping blocks adoption.

## Dashboard and legacy telemetry

The Go dashboard obtains authoritative fields from hostd:

- inventory and stable resource IDs;
- audit state and per-resource classification;
- current allocation/run/journal identity;
- enforcement grade;
- blocked/orphan reason and receipt digests;
- utilization and memory telemetry as observations.

Legacy `nvidia-smi` shell-outs, process basename matching, PID existence checks, log age, and trainer
allowlists may continue to render diagnostic panels during migration. They cannot:

- mark a resource free;
- acquire or release a grant;
- adopt or terminate a process;
- advance a queue;
- override an unknown hostd result.

Queue start controls remain disabled unless hostd reports an audited eligible bundle and TrainVM owns
the matching logical lease.

## P0 implementation sequence

### P0.1 — typed inventory and fake kernel

Implement the receipt/data types, canonical codecs, `IHostKernel`, stable accelerator inventory, MIG
hierarchy, topology digests, and deterministic bundle selection. Add read-only hostd status output.

Gate:

- device-index reorder preserves UUID identity;
- full GPU and partition conflicts are correct;
- topology selection is deterministic;
- a missing/failed vendor probe is unknown, never free;
- legacy dashboard detection is proven non-authoritative.

### P0.2 — singleton hostd and durable grant ledger

Implement the protected socket, peer roles, exact ledger schema/hash chain, resource generations, and
atomic multi-resource grants. Do not launch processes yet.

Gate:

- two services using different journals race for one UUID and exactly one receives it;
- reversed multi-resource request order cannot deadlock;
- bundle grant is all-or-none under write/fsync failure;
- generations remain monotonic after restart and reboot simulation;
- network-namespace-separated abstract locks do not bypass the single shared hostd.

### P0.3 — journal/hostd receipt saga

Persist exact host grant/release receipts in the TrainVM journal and implement idempotent
reconciliation. Keep process launch disabled.

Gate:

- inject a crash before and after every host/journal receipt boundary;
- retries with the same request ID return the same outcome;
- receipt divergence, missing journal, and corrupt ledger all block reuse;
- logical lease expiry with an unreleased host grant remains blocked.

Current implementation note: sealed release requests and grant, busy, and
release results now have shared strict public codecs used by journal replay and
available to the transport. A v2 mutation envelope now defines the two-message
exchange: an open claim receives a server challenge, then one sealed command
echoes the exact challenge response and carries exactly one attributed bundle,
release, process-prepare, process-commit, or process-exit request.
Cross-message validators bind the open, challenge,
command digest, and typed reply; reconciliation may explicitly return missing
but can never disguise a new outcome as replayed. A separate mutation server
now dispatches that exchange through the accepted `SOCK_SEQPACKET` peer into
the coordinator. It reobserves the Linux process instance, consumes the
journal challenge once, obtains service access from a host-side authority
rather than request data, uses host-sampled ledger time, and always disconnects
the scoped coordinator session. Duplicate grant, exact reconciliation,
release, stale-fence, cross-scope, malformed-command, and abandoned-challenge
paths are covered at the cooperative test grade. Process prepare is the sole
command allowed to carry `SCM_RIGHTS`; descriptor roles and counts are exact,
the daemon reattests sealed bytes and working-directory identity, and other
packets continue to reject and close delegated descriptors. The TrainVM journal and service now
implement the central process crash window as a typed saga: the exact normalized prepare receipt is
hash-chained before exec commit, controller recovery accepts only ordered prepare/commit evidence
for the active bound launch, and injected lost replies replay without changing durable identity.
The same saga now records ordered terminal exit evidence. Service release reconciliation discovers
every durable process attempt after restart, finalizes committed processes first, then persists and
executes the physical release saga, and only afterward permits the builtin logical-lease release.
The concrete bounded process client now sends those prepare/commit operations over the authenticated
mutation transport with exact descriptor delegation and typed response checks. Journal-backed
mutation-claim provisioning now derives resource scope from immutable request/release records and
process scope from durable launch bindings. It uses cryptographically random controller identities,
durable per-concurrency generations, exact replay within a service process, restart generation
advancement, and fail-closed supersession. TrainVM now accepts an optional strict hostd client
document that pins the socket inode and owner credentials, bounds request time, obtains the broker
epoch from read-only status, and rejects host/boot mismatch before constructing both typed clients.
The remaining P0.3 gate is the unified privileged daemon entry point and the
privileged end-to-end process-crash matrix at strict
socket-pidfd/host-namespace grade. Deterministic transport checkpoints already prove that interruption at
each pre-dispatch boundary leaves no ledger outcome, while interruption after
the durable dispatch recovers the exact replay and remains releasable.
The additive host-ledger v4 process-authority extension is also implemented:
it maintains a separate immutable hash chain, preserving all v1 resource-chain
bytes, and atomically projects exact launch intents and stopped-child spawn
receipts. Intents require the exact active allocation attribution and persist
the resolved-launch, executable, and cgroup identities; spawn receipts require
the same still-active grant and bind boot ID, PID starttime, cgroup inode, and
executable digest. Pre-commit fault points roll back both chain and projection,
while post-commit lost replies resolve only to the exact canonical replay.

The client side now shares bounded deadline/correlation transport machinery while keeping resource
and process semantics in separate typed clients. Accelerator-backed experiment resources lower
deterministically to an exact sealed bundle request, and the service performs the durable grant saga
before worker-launch authorization. A durable busy outcome is surfaced as its own reconciliation
state and does not cause repeated mutation. Retry policy after busy and a typed CPU process-slot
resource are still required before general production admission is enabled.

### P0.4 — guarded launcher and strict cgroup enforcement

Integrate sealed launch descriptors, allocation cgroups, stopped `clone3`, pidfds, effective CPU/I/O
policy receipts, and device-BPF allowlists.

Current implementation note: the native stopped-child boundary and its v4
ledger join are exposed through the authenticated daemon mutation surface. The
cgroup authority pins and reattests cgroup-v2, uses deterministic
allocation directory names, reopens only empty retry directories, and retains
the directory after a durable intent. The launcher validates sealed
descriptors, clones directly into that cgroup with a pidfd, blocks on a private
pre-exec pipe, double-attests proc starttime and unified membership, and kills
and reaps by pidfd on every failure. The combined process authority commits the
intent before clone and the spawn identity before returning the closed gate.
The daemon-owned supervisor retains the stopped launch across separate request
connections, makes prepare/commit/finalize exact-replay safe within one daemon
lifetime, and releases the private pre-exec gate only after the caller has the
durable spawn receipt. The additive v5 terminal receipt is implemented as well: pidfd wait status,
the exact spawn identity, twice-empty cgroup evidence, and a complete trusted
accelerator-context audit must all agree before commit. The v6 recovery-terminal
receipt is a separate contract for a restarted daemon that is no longer the
worker's parent: it records exact pidfd-terminal, PID-absent, or
identity-superseded evidence and never invents a wait code/status. It is
mutually exclusive with v5 evidence and requires the same empty cgroup and
accelerator-context audit before release. The startup auditor now freezes a
one-shot recovery set and retains exact pidfds; recovery can reopen only the
durable cgroup inode, transfer a pidfd once, signal only through that handle,
and commit v6 evidence after terminal observation. Resource release fails
closed for every spawned allocation without either receipt, and the empty
cgroup is removed only afterward. CPU/I/O policy evidence, device BPF,
privileged end-to-end qualification, and daemon bootstrap/policy wiring remain
in this gate. A daemon crash is not claimed as ordinary in-memory supervisor
replay; startup policy must consume the durable recovery records.

Gate:

- no child executes before its spawn receipt is durable;
- worker descendants remain in the protected cgroup;
- an allocation cannot open an unassigned accelerator node in strict mode;
- a failed launch produces a terminal receipt or remains visibly blocked;
- PID reuse never causes adoption or signalling of the new process.

### P0.5 — startup orphan audit and policy actions

Implement the audit state machine, driver-context matching, exact adoption, guarded termination, and
reboot recovery.

Gate:

- hostd/service/worker crash at every saga stage converges without double grant;
- a surviving grandchild or accelerator context prevents release;
- foreign, renamed, native, and MPS processes are not missed by script-based assumptions;
- probe failure and permission denial fail closed;
- reboot invalidates PID evidence while preserving generation monotonicity;
- hostd never enables process authority before committing its audit receipt.

### P0.6 — dashboard authority migration

Replace dashboard queue admission with TrainVM/hostd status and command APIs. Retain legacy samplers
only for telemetry.

Gate:

- the dashboard cannot start a run while hostd is unavailable, auditing, or blocking the requested
  resource;
- UI resource identity is UUID/partition based rather than device-index based;
- orphan and enforcement-grade state is visible without trainer-specific handlers.

## Adversarial test contract

Core logic is tested without a real GPU through explicit interfaces:

```cpp
class IHostKernel;      // boot/host identity, inventory, cgroups, pidfds, contexts
class IHostLedger;      // transactional receipt append and projection reads
class IJournalClient;   // exact receipt reconciliation, no direct DB dependency
class ILaunchBackend;   // stopped child, exec release, wait and terminate
```

The fake kernel models boot IDs, PID reuse, process exit races, cgroup membership, descendants,
namespace inodes, device access, GPU contexts, topology changes, and probe errors. Test-only fault
injection exists at every ledger and kernel boundary; production code has no hidden bypass.

Required adversarial cases include:

1. two journals and services request the same physical UUID concurrently;
2. two- and four-device bundles race in opposite orders;
3. transaction, disk-full, fsync, and connection failures occur at every saga boundary;
4. hostd crashes after grant, after child creation, before spawn commit, and before release commit;
5. TrainVM crashes while its worker and grant remain live;
6. the launcher exits while a child or grandchild retains a driver context;
7. a PID exits and is reused with a different start time, executable, or cgroup;
8. an exact orphan is adopted and every one-field mismatch is refused;
9. a foreign renamed Python process, native process, and MPS server occupies a resource;
10. NVML/ROCm is unavailable, times out, returns partial data, or denies process details;
11. a full GPU conflicts with all MIG children while eligible sibling partitions coexist;
12. device indices reorder, a UUID disappears, or topology changes across restart;
13. two abstract locks succeed in separate network namespaces, but only shared hostd can grant;
14. a private mount namespace cannot see hostd and fails closed;
15. a same-UID peer outside an authorized service cgroup is refused in strict mode;
16. a worker attempts cgroup migration, fork/daemon escape, or an unassigned device open;
17. cgroup marker paths, symlinks, inode identities, or ledger receipt bytes are forged;
18. a process exits between enumeration, `/proc` inspection, and `pidfd_open`;
19. host ledger and journal disagree on allocation, generation, bundle, or receipt digest;
20. termination freezes or kills only the exact cgroup and a timeout keeps it blocked;
21. boot ID changes with stale PID/cgroup receipts and persistent resource generations;
22. machine identity changes and the foreign-host ledger is rejected.

Real-host integration tests use disposable cgroups and fake device nodes first. GPU-backed tests are a
separate privileged suite and do not replace deterministic fake coverage.

## Explicit non-goals for P0

- multi-host or cluster scheduling;
- fractional sharing without hardware partitions;
- trusting utilization thresholds as availability;
- adopting an unknown external process by command line or executable name;
- killing foreign accelerator users by default;
- defending against a hostile same-UID process in cooperative mode;
- making NVML, ROCm SMI, CUDA, PyTorch, or Python part of the durable schema;
- encoding model-family names or trainer-specific launch rules in hostd.

## Linux references

- [`network_namespaces(7)`](https://man7.org/linux/man-pages/man7/network_namespaces.7.html) documents
  isolation of the abstract Unix socket namespace.
- [`unix(7)`](https://man7.org/linux/man-pages/man7/unix.7.html) documents pathname sockets,
  filesystem permissions, peer credentials, and descriptor passing.
- [Linux cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html) documents `cgroup.events`,
  `cgroup.freeze`, `cgroup.kill`, and BPF-based v2 device control.
- [`clone(2)`](https://man7.org/linux/man-pages/man2/clone.2.html) documents `clone3`,
  `CLONE_INTO_CGROUP`, and `CLONE_PIDFD`.
- [`pidfd_open(2)`](https://man7.org/linux/man-pages/man2/pidfd_open.2.html) and
  [`pidfd_send_signal(2)`](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html) document
  race-resistant process observation and signalling.
