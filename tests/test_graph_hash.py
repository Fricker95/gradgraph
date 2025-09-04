#!/usr/bin/env python3
# 
# test_graph_hash.py
# 
# Created by Nicolas Fricker on 08/22/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import pytest
import numpy as np

from gradgraph.graph.hash import (
    hash,
    hash64,
    hash32
)

### MARK: hash

def test_hash_forward_and_reverse_same_hash():
    a = np.array([1, 2, 3, 4])
    assert hash(a) == hash(a[::-1])


def test_hash_list_and_array_consistency():
    data_list = [10, 20, 30]
    data_array = np.array(data_list)
    assert hash(data_list) == hash(data_array)


def test_hash_different_sequences_different_hashes():
    a = [1, 2, 3]
    b = [1, 2, 4]
    assert hash(a) != hash(b)


def test_hash_tuple_input():
    tup = (5, 6, 7)
    arr = np.array(tup)
    assert hash(tup) == hash(arr)


def test_hash_invalid_ndim_input():
    arr_2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="1-dimensional"):
        hash(arr_2d)

### MARK: hash64

def test_hash64_returns_int_and_range():
    v = hash64("hello")
    assert isinstance(v, int)
    assert 0 <= v <= (2**64 - 1)

def test_hash64_deterministic_same_input_same_output():
    s = "consistent string"
    v1 = hash64(s)
    v2 = hash64(s)
    assert v1 == v2

def test_hash64_different_strings_typically_different_hashes():
    v1 = hash64("foo")
    v2 = hash64("bar")
    assert v1 != v2

@pytest.mark.parametrize("s", ["", "a", "🙂", "multi-byte 🌍🚀", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢"])
def test_hash64_unicode_and_edge_cases(s):
    v = hash64(s)
    assert isinstance(v, int)
    assert 0 <= v <= (2**64 - 1)

@pytest.mark.parametrize("bad", [None, 123, 1.5, b"bytes", ["list"], {"dict": 1}])
def test_hash64_type_error_for_non_string(bad):
    with pytest.raises(TypeError):
        hash64(bad)

def test_hash64_long_string():
    s = "x" * 10_000
    v = hash64(s)
    assert isinstance(v, int)
    assert 0 <= v <= (2**64 - 1)

def test_hash64_small_corpus_uniqueness():
    n = 1_000_000
    vals = {hash64(f"item-{i}") for i in range(n)}
    assert len(vals) == n

### MARK: hash32

def test_hash32_returns_int_and_range():
    v = hash32("hello")
    assert isinstance(v, int)
    assert 0 <= v <= (2**32 - 1)

def test_hash32_deterministic_same_input_same_output():
    s = "consistent string"
    v1 = hash32(s)
    v2 = hash32(s)
    assert v1 == v2

def test_hash32_different_strings_typically_different_hashes():
    v1 = hash32("foo")
    v2 = hash32("bar")
    assert v1 != v2

@pytest.mark.parametrize("s", ["", "a", "🙂", "multi-byte 🌍🚀", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢"])
def test_hash32_unicode_and_edge_cases(s):
    v = hash32(s)
    assert isinstance(v, int)
    assert 0 <= v <= (2**32 - 1)

@pytest.mark.parametrize("bad", [None, 123, 1.5, b"bytes", ["list"], {"dict": 1}])
def test_hash32_type_error_for_non_string(bad):
    with pytest.raises(TypeError):
        hash32(bad)

def test_hash32_long_string():
    s = "x" * 10_000
    v = hash32(s)
    assert isinstance(v, int)
    assert 0 <= v <= (2**32 - 1)

def test_hash32_small_corpus_uniqueness():
    n = 1_000_000
    vals = {hash32(f"item-{i}") for i in range(n)}
    # 999_862 < 1_000_000
    assert len(vals) < n
