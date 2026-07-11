"""Paged recurrent-state serving primitives for continuous RWKV batching.

Inspired by AUXStar/RWKV-Server's dynamic batching, persistent prompt-state
forking, and asynchronous state copies:
https://github.com/AUXStar/RWKV-Server and
https://discord.com/channels/992359628979568762/992362493055881276/1513945855760007343.

This module contains scheduler-neutral storage primitives, not a network server.
States stay on CPU between decode quanta and are copied explicitly to a chosen
device. No request can observe or mutate another request's state.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import fields, is_dataclass
import threading
from typing import Any

import torch


def _map_state(value: Any, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_state(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_state(item, fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_state(item, fn) for item in value)
    if is_dataclass(value):
        return type(value)(**{field.name: _map_state(getattr(value, field.name), fn)
                              for field in fields(value)})
    return copy.deepcopy(value)


class PagedRecurrentStatePool:
    def __init__(self, *, max_entries: int = 4096, pin_memory: bool = True):
        if max_entries < 1:
            raise ValueError("state-pool capacity must be positive")
        self.max_entries, self.pin_memory = int(max_entries), bool(pin_memory)
        self._states: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()

    def __len__(self):
        return len(self._states)

    def put(self, key: str, state: Any) -> None:
        def page(tensor: torch.Tensor):
            cpu = tensor.detach().to("cpu").contiguous().clone()
            return cpu.pin_memory() if self.pin_memory and torch.cuda.is_available() else cpu
        with self._lock:
            self._states[str(key)] = _map_state(state, page)
            self._states.move_to_end(str(key))
            while len(self._states) > self.max_entries:
                self._states.popitem(last=False)

    def get(self, key: str, *, device: str | torch.device = "cpu", non_blocking: bool = True):
        with self._lock:
            if key not in self._states:
                raise KeyError(key)
            value = self._states[key]; self._states.move_to_end(key)
            return _map_state(value, lambda tensor: tensor.to(device, non_blocking=non_blocking).clone())

    def fork(self, parent: str, child: str) -> None:
        with self._lock:
            if parent not in self._states:
                raise KeyError(parent)
            self.put(child, self._states[parent])

    def release(self, key: str) -> bool:
        with self._lock:
            return self._states.pop(key, None) is not None

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._states)


def stack_recurrent_states(states: list[Any]) -> Any:
    """Stack identically-shaped request states for one continuous-batch quantum."""
    if not states:
        raise ValueError("cannot stack an empty recurrent batch")
    first = states[0]
    if torch.is_tensor(first):
        return torch.cat(states, dim=0)
    if isinstance(first, dict):
        return {key: stack_recurrent_states([state[key] for state in states]) for key in first}
    if isinstance(first, list):
        return [stack_recurrent_states([state[i] for state in states]) for i in range(len(first))]
    if isinstance(first, tuple):
        return tuple(stack_recurrent_states([state[i] for state in states]) for i in range(len(first)))
    if is_dataclass(first):
        return type(first)(**{field.name: stack_recurrent_states(
            [getattr(state, field.name) for state in states]) for field in fields(first)})
    if isinstance(first, (str, int, float, bool, type(None))) and all(item == first for item in states):
        return first
    raise TypeError(f"unsupported recurrent state leaf {type(first).__name__}")


def split_recurrent_state(state: Any, count: int) -> list[Any]:
    if count < 1:
        raise ValueError("split count must be positive")
    if torch.is_tensor(state):
        if state.shape[0] != count:
            raise ValueError("batched state leading dimension does not match request count")
        return [state[i:i + 1] for i in range(count)]
    if isinstance(state, dict):
        pieces = {key: split_recurrent_state(value, count) for key, value in state.items()}
        return [{key: pieces[key][i] for key in pieces} for i in range(count)]
    if isinstance(state, (list, tuple)):
        pieces = [split_recurrent_state(value, count) for value in state]
        rows = [[part[i] for part in pieces] for i in range(count)]
        return [tuple(row) if isinstance(state, tuple) else row for row in rows]
    if is_dataclass(state):
        pieces = {field.name: split_recurrent_state(getattr(state, field.name), count)
                  for field in fields(state)}
        return [type(state)(**{name: values[i] for name, values in pieces.items()})
                for i in range(count)]
    if isinstance(state, (str, int, float, bool, type(None))):
        return [state for _ in range(count)]
    raise TypeError(f"unsupported recurrent state leaf {type(state).__name__}")
