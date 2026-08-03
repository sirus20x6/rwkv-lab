"""Closed, typed tensor-runtime implementations for TrainVM components.

Each algorithm category owns one module.  The stable compatibility facade is
``rwkv_lab.training_components``; trainers should not couple categories by
importing implementation internals from one another.
"""
