#!/usr/bin/env python3
# 
# hash.py
# 
# Created by Nicolas Fricker on 08/22/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import hashlib
import numpy as np

def hash64(s: str) -> int:
    """
    Return a 64-bit integer hash of a string using BLAKE2b.

    This function computes a 64-bit hash of the input string by applying
    the BLAKE2b cryptographic hash function with an 8-byte digest.
    The result is returned as a non-negative integer in the range
    [0, 2**64 - 1].

    Parameters
    ----------
    s : str
        Input string to be hashed. Must be a valid Python string.

    Returns
    -------
    int
        A 64-bit integer hash of the input string.

    Raises
    ------
    TypeError
        If the input `s` is not a string.

    Notes
    -----
    * The hash is deterministic: the same string always yields the same value.
    * The function is not intended for cryptographic security purposes when
      truncated to 64 bits, but provides a compact integer identifier.

    Examples
    --------
    >>> hash64("hello")
    12085107955937391741 
    >>> hash64("world")
    6355075502240600847 
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected input of type str, got {type(s).__name__}")
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")

def hash32(s: str) -> int:
    """
    Return a 32-bit integer hash of a string using BLAKE2b.

    This function computes a 32-bit hash of the input string by applying
    the BLAKE2b cryptographic hash function with an 4-byte digest.
    The result is returned as a non-negative integer in the range
    [0, 2**32 - 1].

    Parameters
    ----------
    s : str
        Input string to hash.

    Returns
    -------
    int
        Unsigned 32-bit integer in the range [0, 2**32 - 1].

    Raises
    ------
    TypeError
        If ``s`` is not a string.

    Notes
    -----
    * Uses ``hashlib.blake2b`` with ``digest_size=4`` (32 bits).
    * Collisions are possible due to the reduced 32-bit space.
    * Output is platform-independent since it uses big-endian conversion.

    Examples
    --------
    >>> hash32("hello")
    3932620535 
    >>> hash32("world")
    2160676788
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected str, got {type(s).__name__}")
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=4)
    return int.from_bytes(h.digest(), "big")

def hash(arr: np.ndarray | list[int] | list[float] | tuple[int] | tuple[float]) -> str:
    """
    Compute a direction-invariant SHA-256 hash for a 1D numeric sequence.

    This function generates a SHA-256 hash for the given numeric sequence,
    treating the forward and reversed order as equivalent. The smaller of
    the two byte representations (forward or reversed) is chosen to ensure
    that a sequence and its reverse map to the same hash value.

    Parameters
    ----------
    arr : numpy.ndarray or list or tuple of int or float
        One-dimensional numeric sequence. If a list is provided, it is
        converted internally to a NumPy array.

    Returns
    -------
    str
        SHA-256 hexadecimal hash string representing the input sequence
        in a direction-invariant manner.

    Raises
    ------
    ValueError
        If the input array is not one-dimensional.

    Notes
    -----
    * Both forward and reversed sequences produce the same hash.
    * Non-numeric inputs may produce undefined results, as the function
      assumes numeric data.

    Examples
    --------
    >>> import numpy as np
    >>> hash(np.array([1, 2, 3]))
    e2e2033a...c54ef
    >>> hash([3, 2, 1])  # reversed sequence yields the same hash
    e2e2033a...c54ef'
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"Input must be 1-dimensional, got {arr.ndim}D")
    fwd = arr.tobytes()
    rev = arr[::-1].tobytes()
    return hashlib.sha256(min(fwd, rev)).hexdigest()

