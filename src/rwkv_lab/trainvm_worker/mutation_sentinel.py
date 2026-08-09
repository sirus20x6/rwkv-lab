"""Fail-closed ordering sentinel for the mandatory pre-mutation gate boundary.

A stateful training path must reach its controller boundary *before* the
optimizer mutates parameters. A boundary call placed after the mutation still
raises, but only once the mutation it was meant to guard has already happened,
which is exactly the failure the attempt-baseline eval gate exists to prevent.
Native post-step rejection is defense in depth; it is not the boundary.

Writing the two calls in the right order inside one helper makes the ordering a
convention. This sentinel makes it enforceable. ``cross()`` calls the boundary
for one exact next step and arms a single-use token; the mutation consumes it.
An optimizer step that finds nothing armed raises, so the boundary cannot be
moved after the mutation, deleted, or bypassed by a second optimizer instance
or a fused update path that never routes through the helper.

The observation point is
``torch.optim.optimizer.register_optimizer_step_pre_hook``, a process-global
pre-hook PyTorch runs inside ``Optimizer.step`` for every optimizer instance —
including subclasses and instances constructed after installation. Wrapping one
optimizer object would miss both, and an alternate update path is precisely
what a bypass looks like.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager


class MutationSentinelError(RuntimeError):
    """An optimizer mutation was not preceded by its declared boundary."""


class OptimizerMutationSentinel:
    """Bind every optimizer mutation to one preceding boundary crossing."""

    __slots__ = ("_armed", "_handle", "_journal", "_mutations")

    def __init__(self) -> None:
        self._armed: int | None = None
        self._handle = None
        self._mutations = 0
        self._journal: list[tuple[str, int]] = []

    @property
    def observed_mutations(self) -> int:
        return self._mutations

    @property
    def journal(self) -> tuple[tuple[str, int], ...]:
        """Ordered ``("boundary" | "mutation", step)`` record of what happened.

        Tests assert against this rather than against the mere existence of a
        boundary call, because presence and order are different properties.
        """

        return tuple(self._journal)

    def cross(
        self, next_step: int, boundary: Callable[[int], None] | None = None
    ) -> None:
        """Call the controller boundary for ``next_step``, then arm one token.

        ``boundary`` is the controller-facing call — for TrainVM workers,
        ``WorkerControlRuntime.pre_optimizer_step``. It runs before the token
        is armed, so a boundary that refuses (an unsatisfied attempt-baseline
        eval gate, a cancellation) leaves the sentinel disarmed and the
        mutation that would have followed fails closed.
        """

        if (
            not isinstance(next_step, int)
            or isinstance(next_step, bool)
            or next_step < 1
        ):
            raise MutationSentinelError(
                "the pre-mutation boundary names the positive step about to run"
            )
        if self._armed is not None:
            raise MutationSentinelError(
                "the pre-mutation boundary was crossed twice without a mutation"
            )
        if boundary is not None:
            boundary(next_step)
        self._journal.append(("boundary", next_step))
        self._armed = next_step

    def _observe(self, optimizer: object, args: object, kwargs: object) -> None:
        del optimizer, args, kwargs
        if self._armed is None:
            raise MutationSentinelError(
                "optimizer mutation reached the parameters without first "
                "crossing the mandatory pre-optimizer-step boundary"
            )
        self._mutations += 1
        self._journal.append(("mutation", self._armed))
        self._armed = None

    @contextmanager
    def installed(self) -> Iterator[OptimizerMutationSentinel]:
        from torch.optim.optimizer import register_optimizer_step_pre_hook

        if self._handle is not None:
            raise MutationSentinelError("the mutation sentinel is already installed")
        self._handle = register_optimizer_step_pre_hook(self._observe)
        try:
            yield self
        finally:
            handle, self._handle = self._handle, None
            handle.remove()


__all__ = [
    "MutationSentinelError",
    "OptimizerMutationSentinel",
]
