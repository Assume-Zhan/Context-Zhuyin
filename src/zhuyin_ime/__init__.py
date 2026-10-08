"""Zhuyin input method: framework independent engine core, conversion server,
and front ends (IBus).

The heavy parts (lexicon, n-gram decoder, LM reranker) live in the
conversion server; front ends only forward key events and draw the state the
server returns.
"""
