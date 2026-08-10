#!/usr/bin/env bash
# Full non-GPU acceptance: every suite CI runs, plus the native suites no
# hosted runner can build. Writes one machine-readable receipt so a claim of
# "acceptance passed" can be checked instead of believed.
#
#   scripts/acceptance.sh [--gpu] [--evidence DIR]
#
# --gpu also runs the GPU-marked suite. It is off by default because the GPU
# suites share one device with any live training run on this host.
#
# Two outputs, and the difference between them is the point:
#
#   evidence/acceptance.json   the receipt. A snapshot of THIS run's output,
#                              git-ignored, because a receipt records the commit
#                              it ran against but is served at whatever commit
#                              the reader has checked out. One was committed by
#                              accident and spent six days describing a dirty
#                              worktree.
#   docs/experiment-vm/acceptance-attestations.v1.jsonl
#                              the attestation. One appended line per run, and
#                              it IS tracked. A line is a historical fact --
#                              "on this date, at this commit, on this host,
#                              these suites did this" -- which stays true no
#                              matter what is checked out later. That is why it
#                              can live in git when the receipt cannot.
#
# The attestation exists because the four suites the hosted CI job excludes run
# only here, and nothing schedules this script. Without a tracked record there
# is no way to tell "acceptance passed last night" from "acceptance has never
# been run", and those two states looked identical for the life of the
# repository. Failures are appended too: an attestation that only appears on
# success would make an absent line ambiguous all over again.
set -uo pipefail

evidence_dir="evidence"
run_gpu=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) run_gpu=1; shift ;;
    --evidence) evidence_dir="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mkdir -p "$evidence_dir"
receipt="$evidence_dir/acceptance.json"
build_dir="${TRAINVM_BUILD_DIR:-trainvm/build-acceptance}"
attestation_log="docs/experiment-vm/acceptance-attestations.v1.jsonl"
exclusions="docs/experiment-vm/native-ci-exclusions.v1.json"
ctest_junit="$evidence_dir/native-ctest.xml"

declare -a names=() statuses=() details=()

record() {
  names+=("$1"); statuses+=("$2"); details+=("$3")
  printf '%-28s %s\n' "$1" "$2"
}

run_suite() {
  local name="$1"; shift
  local log="$evidence_dir/$name.log"
  if "$@" >"$log" 2>&1; then
    record "$name" passed "$log"
  else
    record "$name" FAILED "$log"
  fi
}

# Every suite that is not the GPU phase runs behind the accelerator mask. The
# mask has to be applied before the process starts: pytest imports test modules
# before it filters on markers, so a `-m "not gpu"` run still evaluates every
# module-level `torch.cuda.is_available()` and initializes the driver that is
# also driving this host's display. tests/conftest.py repeats the mask for
# direct `pytest` invocations, which this launcher cannot reach.
run_non_gpu_suite() {
  local name="$1"; shift
  run_suite "$name" python3 "$repo_root/scripts/non_gpu_environment.py" "$@"
}

echo "== TrainVM acceptance =="

# Native: the authority implementation. Requires GCC 16 with -freflection.
if command -v cmake >/dev/null && command -v g++ >/dev/null &&
   [[ "$(g++ -dumpversion | cut -d. -f1)" -ge 16 ]]; then
  run_non_gpu_suite native-configure cmake -S trainvm -B "$build_dir" -DCMAKE_BUILD_TYPE=Debug
  run_non_gpu_suite native-build cmake --build "$build_dir" -j
  # --output-junit so the attestation can name the individual suites the hosted
  # job excludes. A single native-ctest pass/fail cannot say which of the four
  # ran, and "which suites" is most of what the attestation is for.
  run_non_gpu_suite native-ctest ctest --test-dir "$build_dir" --output-on-failure \
    --output-junit "$repo_root/$ctest_junit"
  if [[ -x "$build_dir/trainvm" ]]; then
    run_non_gpu_suite compatibility-catalog "$build_dir/trainvm" validate-catalog \
      "$repo_root/docs/experiment-vm/compatibility-workflows.v1.json" "$repo_root"
  else
    record compatibility-catalog skipped "trainvm binary was not built"
  fi
else
  record native-ctest skipped "requires cmake and GCC 16+ for C++26 reflection"
  record compatibility-catalog skipped "requires the native build"
fi

# Portable halves, all runnable without the native toolchain.
run_non_gpu_suite schema-golden python3 scripts/validate_experiment_documents.py
run_non_gpu_suite authoring-matrix python3 scripts/validate_no_code_authoring_matrix.py
# The coverage gate is a `--collect-only` run, so it imports every test module
# and is exactly as capable of initializing the driver as the suite itself.
run_non_gpu_suite coverage-gate python3 scripts/ci_coverage_gate.py -m "not gpu"
run_non_gpu_suite python-cpu python3 -m pytest -q -n "${PYTEST_WORKERS:-auto}" \
  --dist worksteal -m "not gpu" tests \
  --junitxml="$evidence_dir/python-cpu.xml"

# What that run did NOT do. A suite can lose coverage without losing a test:
# every skip here is conditional on a host fact, so an asset that moves or an
# optional package that stops importing drains coverage while the line above
# still reports a pass. The count on a real host is a property of the machine
# and is reported rather than pinned; a reason nobody declared fails.
run_non_gpu_suite skip-reasons python3 scripts/ci_skip_reason_gate.py \
  "$evidence_dir/python-cpu.xml" --environment real-host-acceptance \
  --receipt "$evidence_dir/skip-receipt.json"

if command -v go >/dev/null; then
  run_non_gpu_suite go bash -c \
    'cd "$1/dashboard" && go build ./... && go vet ./... && go test ./...' \
    _ "$repo_root"
else
  record go skipped "go toolchain not installed"
fi

if [[ "$run_gpu" == "1" ]]; then
  # The one place accelerator access is granted, and it is granted explicitly:
  # the suite is opt-in twice over, by --gpu and by the marker expression.
  run_suite gpu env TRAINVM_TEST_ACCELERATOR_ACCESS=1 \
    python3 -m pytest -q -m gpu tests \
    --junitxml="$evidence_dir/gpu.xml"
else
  record gpu skipped "not requested; pass --gpu (shares the device with live training)"
fi

# One receipt, including the exact commit and whether the tree was dirty, so a
# result can never be attributed to source it did not run against.
commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
dirty=false
# The attestation log is excluded from the dirtiness test on purpose. This
# script appends to it, so counting it would make every run after the first
# report a dirty worktree caused by nothing but the previous run's own record --
# a false alarm that would then be baked into the attestation as a reason to
# distrust it. Everything else still counts.
if ! git diff --quiet -- . ":(exclude)$attestation_log" 2>/dev/null ||
   ! git diff --cached --quiet -- . ":(exclude)$attestation_log" 2>/dev/null; then
  dirty=true
fi

failed=0
{
  printf '{\n  "api_version": "trainvm.acceptance/v1",\n'
  printf '  "commit": "%s",\n  "dirty_worktree": %s,\n' "$commit" "$dirty"
  printf '  "generated_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "suites": [\n'
  for index in "${!names[@]}"; do
    [[ "${statuses[$index]}" == "FAILED" ]] && failed=1
    printf '    {"name": "%s", "status": "%s", "detail": "%s"}' \
      "${names[$index]}" "${statuses[$index]}" "${details[$index]}"
    [[ $index -lt $((${#names[@]} - 1)) ]] && printf ','
    printf '\n'
  done
  printf '  ]\n}\n'
} >"$receipt"

# The attestation: one appended line recording that this run happened. Derived
# from the receipt that was just written, so the two cannot disagree.
if python3 scripts/append_acceptance_attestation.py \
     --receipt "$receipt" \
     --exclusions "$exclusions" \
     --ctest-junit "$ctest_junit" \
     --log "$attestation_log" \
     --gpu-requested "$run_gpu"; then
  :
else
  # Loud, because an unattested run is exactly the state this log exists to end.
  # Not fatal: refusing to report an otherwise-complete acceptance because its
  # bookkeeping failed would trade a known gap for a confusing one.
  echo "WARNING: attestation NOT written to $attestation_log" >&2
  echo "         this run leaves no durable record that it happened" >&2
fi

echo
echo "receipt:     $receipt      (git-ignored; this run's output)"
echo "attestation: $attestation_log  (tracked; commit it)"
if [[ "$failed" == "1" ]]; then
  echo "ACCEPTANCE FAILED"
  exit 1
fi
echo "acceptance passed"
