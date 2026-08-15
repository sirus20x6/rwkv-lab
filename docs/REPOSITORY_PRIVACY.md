# Repository privacy boundary

This repository contains application code, tests with synthetic fixtures, and
portable example configuration. It must not contain local datasets, downloaded
models, generated manifests, run artifacts, personal filesystem paths, or
private dataset provenance.

Store local inputs below an ignored root such as `datasets/`, `corpora/`, or
`private/`, or outside the checkout. Pass their locations through declarative
configuration or environment variables. Examples use neutral `/workspace/...`
paths and must never be copied from a developer machine.

Enable the versioned pre-commit hook after cloning:

```bash
git config core.hooksPath .githooks
```

The same privacy check runs in CI. If it fails, move the data out of the
repository or replace the machine-specific default with an explicit input.
Do not add a broad exception to make a private experiment public.

Language-model vocabulary assets may naturally contain sensitive words. The
guard excludes the reviewed vocabulary asset from content matching, but not
from filename, path, or large-file checks.
