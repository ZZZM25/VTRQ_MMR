
from __future__ import annotations

import hashlib
import struct


# ============================================================
# Fixed Binary Encoding
# ============================================================

_TRAJECTORY_LEAF_STRUCT = struct.Struct(">III")

# MMR Leaf:
#
# trajectory_id : 32 bytes
# start         : uint32
# end           : uint32
#
_MMR_LEAF_STRUCT = struct.Struct(">32sII")

# MMR Internal:
#
# min_start     : uint32
# max_end       : uint32
# left_hash     : 32 bytes
# right_hash    : 32 bytes
#
_MMR_INTERNAL_STRUCT = struct.Struct(
    ">II32s32s"
)


# ============================================================
# SHA-256
# ============================================================

def sha256(
    data: bytes,
) -> bytes:

    return hashlib.sha256(
        data
    ).digest()


# ============================================================
# Legacy Function Retained for Now
# ============================================================

def hash_trajectory_leaf(
    eid: int,
    start: int,
    end: int,
) -> bytes:

    return sha256(
        _TRAJECTORY_LEAF_STRUCT.pack(
            eid,
            start,
            end,
        )
    )


def hash_merkle_internal(
    left_hash: bytes,
    right_hash: bytes,
) -> bytes:

    h = hashlib.sha256()

    h.update(
        left_hash
    )

    h.update(
        right_hash
    )

    return h.digest()


# ============================================================
# MMR Leaf Hash
# ============================================================

def hash_mmr_leaf(
    trajectory_id: bytes,
    start: int,
    end: int,
) -> bytes:
    """
    MMR leaf:

        H(
            trajectory_id
            || start
            || end
        )
    """

    if len(trajectory_id) != 32:

        raise ValueError(
            "trajectory_id must be 32 bytes"
        )

    return sha256(
        _MMR_LEAF_STRUCT.pack(
            trajectory_id,
            start,
            end,
        )
    )


# ============================================================
# MMR Internal Hash
# ============================================================

def hash_mmr_internal(
    min_start: int,
    max_end: int,
    left_hash: bytes,
    right_hash: bytes,
) -> bytes:
    """
    MMR internal node:

        H(
            min_start
            || max_end
            || left_hash
            || right_hash
        )
    """

    return sha256(
        _MMR_INTERNAL_STRUCT.pack(
            min_start,
            max_end,
            left_hash,
            right_hash,
        )
    )


# ============================================================
# Time Overlap
# ============================================================

def time_overlap(
    start: int | float,
    end: int | float,
    query_start: int | float,
    query_end: int | float,
) -> bool:
    """
    Closed interval overlap:

        [start, end]
        and
        [query_start, query_end]
    """

    return not (
        end < query_start
        or
        start > query_end
    )


# ============================================================
# PyCharm Green Triangle Test
# ============================================================

def main():

    print(
        "===== SHA256 Test ====="
    )

    value = sha256(
        b"hello"
    )

    print(
        value.hex()
    )

    print(
        "Hash length:",
        len(value),
        "bytes",
    )

    # ========================================================
    # MMR Leaf
    # ========================================================

    print()

    print(
        "===== MMR Leaf Test ====="
    )

    trajectory_id = sha256(
        b"trajectory-test"
    )

    leaf_hash = (
        hash_mmr_leaf(
            trajectory_id=
            trajectory_id,

            start=
            100,

            end=
            200,
        )
    )

    print(
        leaf_hash.hex()
    )

    print(
        "Hash length:",
        len(leaf_hash),
        "bytes",
    )

    # ========================================================
    # MMR Internal
    # ========================================================

    print()

    print(
        "===== MMR Internal Test ====="
    )

    right_hash = (
        hash_mmr_leaf(
            trajectory_id=
            sha256(b"trajectory-test-2"),

            start=
            300,

            end=
            400,
        )
    )

    parent_hash = (
        hash_mmr_internal(
            min_start=
            100,

            max_end=
            400,

            left_hash=
            leaf_hash,

            right_hash=
            right_hash,
        )
    )

    print(
        parent_hash.hex()
    )

    print(
        "Hash length:",
        len(parent_hash),
        "bytes",
    )

    # ========================================================
    # Time Overlap
    # ========================================================

    print()

    print(
        "===== Time Overlap Test ====="
    )

    print(
        "[100,200] and [150,180]:",
        time_overlap(
            100,
            200,
            150,
            180,
        ),
    )

    print(
        "[100,200] and [300,400]:",
        time_overlap(
            100,
            200,
            300,
            400,
        ),
    )

    print()

    print(
        "crypto.py completed successfully"
    )


if __name__ == "__main__":
    main()
