#pragma once

namespace trainvm {

// The exit-status vocabulary of the `trainvm` binary.
//
// Every subcommand returns one of these, and each value has exactly one
// meaning across every subcommand. Before this existed each command spelled
// its status as a bare integer literal and two of them disagreed: `validate`
// answered 2 for a document that would not compile while `qualify-evidence`
// answered 1 for evidence that would not decode. Since 1 is also what the
// top-level catch in `main` returns for an uncaught exception, a wrapper
// keying on `qualify-evidence`'s status could not tell a broken document from
// a crashed binary. The point of naming the codes is that the disagreement
// becomes visible at the call site rather than living in two comments.
//
// The distinction the vocabulary exists to protect is between a *document*
// that cannot be read (kExitMalformedInput) and a *verdict* the tool reached
// by reading it (kExitNegativeVerdict). A rejection is a normal, reportable
// outcome; a broken document is not, and the two must not share a code.
//
// Keep this list in sync with the "Exit status" section of trainvm/README.md
// and with the summary printed by `usage()` in trainvm/src/main.cpp.

// The command ran and its answer was affirmative: the document compiled, the
// evidence qualified, the catalog matched its pins.
inline constexpr int kExitSuccess = 0;

// An exception escaped to `main`. This is a defect in trainvm or an
// environment failure it does not model — never a statement about the input.
// Reserved for the top-level catch so that "trainvm broke" stays
// distinguishable from every outcome trainvm decided on purpose.
inline constexpr int kExitUncaughtException = 1;

// The input document could not be read as what it claims to be: unparseable
// JSON, a schema the reflected decoder rejects, or a document that parses but
// does not compile. Covers both "not parseable" and "semantically invalid" —
// the two are not distinguished, deliberately, because no caller has yet
// needed to act differently on them and inventing a split now would guess.
inline constexpr int kExitMalformedInput = 2;

// The document was well formed and the answer is no: evidence that does not
// qualify, a preflight that does not pass, a journal whose hash chain does not
// verify, a run that failed. Reportable, not exceptional.
inline constexpr int kExitNegativeVerdict = 3;

// The document was well formed and the thing it named does not exist.
inline constexpr int kExitNotFound = 4;

// The argument vector itself was wrong: unknown subcommand, wrong arity,
// unknown or repeated flag. Value 64 is `EX_USAGE` from sysexits.h, which is
// what the shell conventions around this binary already assumed.
inline constexpr int kExitUsage = 64;

}  // namespace trainvm
