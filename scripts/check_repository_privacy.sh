#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

failed=0

report_matches() {
  local label=$1
  local pattern=$2
  local matches
  matches=$(git grep -nI -E "$pattern" -- \
    ':!scripts/check_repository_privacy.sh' \
    ':!src/rwkv_lab/assets/rwkv_vocab_v20230424.txt' 2>/dev/null || true)
  if [[ -n "$matches" ]]; then
    printf 'privacy check failed: %s\n' "$label" >&2
    # Report only repository path and line number. Never repeat leaked values
    # into CI logs, review comments, or terminal scrollback.
    printf '%s\n' "$matches" | cut -d: -f1,2 | sort -u >&2
    failed=1
  fi
}

report_matches \
  'personal machine paths are tracked' \
  '/thearray(/|$)|/home/sirus(/|$)'
report_matches \
  'private dataset provenance is tracked' \
  '(^|[^[:alnum:]_])(porn|nsfw|hentai|ao3|civitai|gelbooru|adult[ _-]?dataset)([^[:alnum:]_]|$)'

tracked_private_roots=$(git ls-files | grep -E \
  '^(data|dataset|datasets|corpora|private|local|models|runs|artifacts|checkpoints|weights|downloads|cache|caches)/' || true)
if [[ -n "$tracked_private_roots" ]]; then
  printf 'privacy check failed: files are tracked beneath a private/generated root\n' >&2
  printf '%s\n' "$tracked_private_roots" >&2
  failed=1
fi

tracked_sensitive_names=$(git ls-files | grep -Ei \
  '(^|/)(porn|nsfw|hentai|ao3|civitai|gelbooru)([^/]*)(/|$)' || true)
if [[ -n "$tracked_sensitive_names" ]]; then
  printf 'privacy check failed: tracked filenames disclose private provenance\n' >&2
  printf '%s\n' "$tracked_sensitive_names" >&2
  failed=1
fi

if ((failed)); then
  cat >&2 <<'EOF'
Move local data below an ignored root and pass it to the application through a
configuration value or environment variable. Do not suppress this check with a
new exception; review any necessary vocabulary/test fixture exception explicitly.
EOF
  exit 1
fi

printf 'repository privacy check passed\n'
