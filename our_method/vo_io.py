from __future__ import annotations

import os
import struct
from dataclasses import dataclass


from composite_vo import (
    CompositeVO,

    TOKEN_L0_PRUNED_LEAF,
    TOKEN_L0_PRUNED_INTERNAL,
    TOKEN_L0_LEAF,
    TOKEN_L0_INTERNAL,

    TOKEN_L1_EDGE_OPAQUE,
    TOKEN_L1_EDGE_EXPANDED,
    TOKEN_L1_INTERNAL,

    TOKEN_MMR_PRUNED_LEAF,
    TOKEN_MMR_PRUNED_INTERNAL,
    TOKEN_MMR_RESULT_SLOT,
    TOKEN_MMR_INTERNAL,
    TOKEN_MMR_ROOT,
)


# ============================================================
# File Format
# ============================================================

VO_MAGIC = b"CMVO0001"
VS_MAGIC = b"VSET0001"
ROOT_MAGIC = b"ROOT0001"

VERSION = 1

# magic + version + count
_HEADER = struct.Struct(">8sII")

# token type
_TOKEN_TYPE = struct.Struct(">B")


# ============================================================
# Token Payload
# ============================================================

_L0_PRUNED_LEAF = struct.Struct(
    ">ddddII32s"
)

_L0_PRUNED_INTERNAL = struct.Struct(
    ">ddddII32s32s32s"
)

_L0_NODE = struct.Struct(
    ">ddddII"
)

_L1_EDGE_OPAQUE = struct.Struct(
    ">IIII32s"
)

_L1_EDGE_EXPANDED = struct.Struct(
    ">IIII"
)

_MMR_PRUNED_LEAF = struct.Struct(
    ">32sII"
)

_MMR_PRUNED_INTERNAL = struct.Struct(
    ">II32s32s"
)

_MMR_INTERNAL = struct.Struct(
    ">II"
)

_MMR_ROOT = struct.Struct(
    ">I"
)


# ============================================================
# Verification Set
#
# eid
# trajectory_id
# start
# end
#
# 4 + 32 + 4 + 4 = 44 bytes / Entry
# ============================================================

_VS_ENTRY = struct.Struct(
    ">I32sII"
)


# ============================================================
# Root
# ============================================================

_ROOT = struct.Struct(
    ">32s"
)


# ============================================================
# Loaded Verification Entry
# ============================================================

@dataclass(slots=True)
class VerificationEntryRecord:

    eid: int

    trajectory_id: bytes

    start: int

    end: int


# ============================================================
# Write One VO Token
# ============================================================

def _write_token(
    f,
    token: tuple,
) -> None:

    token_type = token[0]

    f.write(
        _TOKEN_TYPE.pack(
            token_type
        )
    )

    # ========================================================
    # L0 PRUNED LEAF
    # ========================================================

    if (
        token_type
        == TOKEN_L0_PRUNED_LEAF
    ):

        (
            _,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            min_start,
            max_end,
            l1_root,
        ) = token

        f.write(
            _L0_PRUNED_LEAF.pack(
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                min_start,
                max_end,
                l1_root,
            )
        )

        return

    # ========================================================
    # L0 PRUNED INTERNAL
    # ========================================================

    if (
        token_type
        == TOKEN_L0_PRUNED_INTERNAL
    ):

        (
            _,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            min_start,
            max_end,
            left_hash,
            mid_hash,
            right_hash,
        ) = token

        f.write(
            _L0_PRUNED_INTERNAL.pack(
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                min_start,
                max_end,
                left_hash,
                mid_hash,
                right_hash,
            )
        )

        return

    # ========================================================
    # L0 LEAF / INTERNAL
    # ========================================================

    if (
        token_type
        == TOKEN_L0_LEAF
        or
        token_type
        == TOKEN_L0_INTERNAL
    ):

        (
            _,
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            min_start,
            max_end,
        ) = token

        f.write(
            _L0_NODE.pack(
                min_lon,
                min_lat,
                max_lon,
                max_lat,
                min_start,
                max_end,
            )
        )

        return

    # ========================================================
    # L1 EDGE OPAQUE
    # ========================================================

    if (
        token_type
        == TOKEN_L1_EDGE_OPAQUE
    ):

        (
            _,
            eid,
            k_e,
            min_start,
            max_end,
            root_e,
        ) = token

        f.write(
            _L1_EDGE_OPAQUE.pack(
                eid,
                k_e,
                min_start,
                max_end,
                root_e,
            )
        )

        return

    # ========================================================
    # L1 EDGE EXPANDED
    # ========================================================

    if (
        token_type
        == TOKEN_L1_EDGE_EXPANDED
    ):

        (
            _,
            eid,
            k_e,
            min_start,
            max_end,
        ) = token

        f.write(
            _L1_EDGE_EXPANDED.pack(
                eid,
                k_e,
                min_start,
                max_end,
            )
        )

        return

    # ========================================================
    # L1 INTERNAL
    # ========================================================

    if (
        token_type
        == TOKEN_L1_INTERNAL
    ):

        return

    # ========================================================
    # MMR PRUNED LEAF
    # ========================================================

    if (
        token_type
        == TOKEN_MMR_PRUNED_LEAF
    ):

        (
            _,
            trajectory_id,
            start,
            end,
        ) = token

        f.write(
            _MMR_PRUNED_LEAF.pack(
                trajectory_id,
                start,
                end,
            )
        )

        return

    # ========================================================
    # MMR PRUNED INTERNAL
    # ========================================================

    if (
        token_type
        == TOKEN_MMR_PRUNED_INTERNAL
    ):

        (
            _,
            min_start,
            max_end,
            left_hash,
            right_hash,
        ) = token

        f.write(
            _MMR_PRUNED_INTERNAL.pack(
                min_start,
                max_end,
                left_hash,
                right_hash,
            )
        )

        return

    # ========================================================
    # RESULT SLOT
    #
    # No payload.
    # The actual Entry is stored in the Verification Set.
    # ========================================================

    if (
        token_type
        == TOKEN_MMR_RESULT_SLOT
    ):

        return

    # ========================================================
    # MMR INTERNAL
    # ========================================================

    if (
        token_type
        == TOKEN_MMR_INTERNAL
    ):

        (
            _,
            min_start,
            max_end,
        ) = token

        f.write(
            _MMR_INTERNAL.pack(
                min_start,
                max_end,
            )
        )

        return

    # ========================================================
    # MMR ROOT
    # ========================================================

    if (
        token_type
        == TOKEN_MMR_ROOT
    ):

        (
            _,
            k_e,
        ) = token

        f.write(
            _MMR_ROOT.pack(
                k_e
            )
        )

        return

    raise ValueError(
        f"Unknown VO Token type: {token_type}"
    )


# ============================================================
# Save Composite VO
# ============================================================

def save_composite_vo(
    vo: CompositeVO,
    file_path: str,
) -> int:

    os.makedirs(
        os.path.dirname(
            os.path.abspath(
                file_path
            )
        ),
        exist_ok=True,
    )

    with open(
        file_path,
        "wb",
    ) as f:

        f.write(
            _HEADER.pack(
                VO_MAGIC,
                VERSION,
                len(
                    vo.tokens
                ),
            )
        )

        for token in vo.tokens:

            _write_token(
                f,
                token,
            )

    return os.path.getsize(
        file_path
    )


# ============================================================
# Read a Fixed Number of Bytes from File
# ============================================================

def _read_exact(
    f,
    size: int,
) -> bytes:

    data = f.read(
        size
    )

    if len(data) != size:

        raise ValueError(
            "File ended prematurely; the VO file may be corrupted"
        )

    return data


# ============================================================
# Load Composite VO
# ============================================================

def load_composite_vo_tokens(
    file_path: str,
) -> list[tuple]:

    with open(
        file_path,
        "rb",
    ) as f:

        (
            magic,
            version,
            token_count,
        ) = _HEADER.unpack(
            _read_exact(
                f,
                _HEADER.size,
            )
        )

        if magic != VO_MAGIC:

            raise ValueError(
                "Invalid Composite VO file"
            )

        if version != VERSION:

            raise ValueError(
                f"Unsupported VO version: {version}"
            )

        tokens: list[tuple] = []

        for _ in range(
            token_count
        ):

            token_type = (
                _TOKEN_TYPE.unpack(
                    _read_exact(
                        f,
                        _TOKEN_TYPE.size,
                    )
                )[0]
            )

            # ================================================
            # L0 PRUNED LEAF
            # ================================================

            if (
                token_type
                == TOKEN_L0_PRUNED_LEAF
            ):

                values = (
                    _L0_PRUNED_LEAF.unpack(
                        _read_exact(
                            f,
                            _L0_PRUNED_LEAF.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            # ================================================
            # L0 PRUNED INTERNAL
            # ================================================

            if (
                token_type
                == TOKEN_L0_PRUNED_INTERNAL
            ):

                values = (
                    _L0_PRUNED_INTERNAL.unpack(
                        _read_exact(
                            f,
                            _L0_PRUNED_INTERNAL.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            # ================================================
            # L0 LEAF / INTERNAL
            # ================================================

            if (
                token_type
                == TOKEN_L0_LEAF
                or
                token_type
                == TOKEN_L0_INTERNAL
            ):

                values = (
                    _L0_NODE.unpack(
                        _read_exact(
                            f,
                            _L0_NODE.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            # ================================================
            # L1 OPAQUE
            # ================================================

            if (
                token_type
                == TOKEN_L1_EDGE_OPAQUE
            ):

                values = (
                    _L1_EDGE_OPAQUE.unpack(
                        _read_exact(
                            f,
                            _L1_EDGE_OPAQUE.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            # ================================================
            # L1 EXPANDED
            # ================================================

            if (
                token_type
                == TOKEN_L1_EDGE_EXPANDED
            ):

                values = (
                    _L1_EDGE_EXPANDED.unpack(
                        _read_exact(
                            f,
                            _L1_EDGE_EXPANDED.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            # ================================================
            # L1 INTERNAL
            # ================================================

            if (
                token_type
                == TOKEN_L1_INTERNAL
            ):

                tokens.append(
                    (
                        token_type,
                    )
                )

                continue

            # ================================================
            # MMR PRUNED LEAF
            # ================================================

            if (
                token_type
                == TOKEN_MMR_PRUNED_LEAF
            ):

                values = (
                    _MMR_PRUNED_LEAF.unpack(
                        _read_exact(
                            f,
                            _MMR_PRUNED_LEAF.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            # ================================================
            # MMR PRUNED INTERNAL
            # ================================================

            if (
                token_type
                == TOKEN_MMR_PRUNED_INTERNAL
            ):

                values = (
                    _MMR_PRUNED_INTERNAL.unpack(
                        _read_exact(
                            f,
                            _MMR_PRUNED_INTERNAL.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            # ================================================
            # RESULT SLOT
            # ================================================

            if (
                token_type
                == TOKEN_MMR_RESULT_SLOT
            ):

                tokens.append(
                    (
                        token_type,
                    )
                )

                continue

            # ================================================
            # MMR INTERNAL
            # ================================================

            if (
                token_type
                == TOKEN_MMR_INTERNAL
            ):

                values = (
                    _MMR_INTERNAL.unpack(
                        _read_exact(
                            f,
                            _MMR_INTERNAL.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            # ================================================
            # MMR ROOT
            # ================================================

            if (
                token_type
                == TOKEN_MMR_ROOT
            ):

                values = (
                    _MMR_ROOT.unpack(
                        _read_exact(
                            f,
                            _MMR_ROOT.size,
                        )
                    )
                )

                tokens.append(
                    (
                        token_type,
                        *values,
                    )
                )

                continue

            raise ValueError(
                f"Unknown VO Token type: "
                f"{token_type}"
            )

        # ====================================================
        # Do Not Allow Unknown Trailing Data
        # ====================================================

        extra = f.read(1)

        if extra:

            raise ValueError(
                "Extra data found at the end of the VO file"
            )

    return tokens


# ============================================================
# Save Verification Set
#
# Supports:
#
# VerificationEntry objects
#
# Or:
#
# (eid, trajectory_id, start, end)
# ============================================================

def save_verification_set(
    verification_set,
    file_path: str,
) -> int:

    os.makedirs(
        os.path.dirname(
            os.path.abspath(
                file_path
            )
        ),
        exist_ok=True,
    )

    with open(
        file_path,
        "wb",
    ) as f:

        f.write(
            _HEADER.pack(
                VS_MAGIC,
                VERSION,
                len(
                    verification_set
                ),
            )
        )

        for entry in (
            verification_set
        ):

            if hasattr(
                entry,
                "eid",
            ):

                eid = entry.eid

                trajectory_id = (
                    entry.trajectory_id
                )

                start = entry.start
                end = entry.end

            else:

                (
                    eid,
                    trajectory_id,
                    start,
                    end,
                ) = entry

            if (
                len(
                    trajectory_id
                )
                != 32
            ):

                raise ValueError(
                    "trajectory_id must be 32 bytes"
                )

            f.write(
                _VS_ENTRY.pack(
                    eid,
                    trajectory_id,
                    start,
                    end,
                )
            )

    return os.path.getsize(
        file_path
    )


# ============================================================
# Load Verification Set
# ============================================================

def load_verification_set(
    file_path: str,
) -> list[
    VerificationEntryRecord
]:

    with open(
        file_path,
        "rb",
    ) as f:

        (
            magic,
            version,
            entry_count,
        ) = _HEADER.unpack(
            _read_exact(
                f,
                _HEADER.size,
            )
        )

        if magic != VS_MAGIC:

            raise ValueError(
                "Invalid Verification Set file"
            )

        if version != VERSION:

            raise ValueError(
                f"Unsupported VS version: {version}"
            )

        result: list[
            VerificationEntryRecord
        ] = []

        for _ in range(
            entry_count
        ):

            (
                eid,
                trajectory_id,
                start,
                end,
            ) = _VS_ENTRY.unpack(
                _read_exact(
                    f,
                    _VS_ENTRY.size,
                )
            )

            result.append(
                VerificationEntryRecord(
                    eid=
                    eid,

                    trajectory_id=
                    trajectory_id,

                    start=
                    start,

                    end=
                    end,
                )
            )

        if f.read(1):

            raise ValueError(
                "Extra data found at the end of the Verification Set file"
            )

    return result


# ============================================================
# Save trusted root_S
#
# Saved during the testing stage.
#
# In the formal system:
# root_S should be obtained from the blockchain/trusted checkpoint,
# and is not part of the VO returned by the server.
# ============================================================

def save_trusted_root(
    root_hash: bytes,
    file_path: str,
) -> int:

    if len(root_hash) != 32:

        raise ValueError(
            "root_S must be 32 bytes"
        )

    os.makedirs(
        os.path.dirname(
            os.path.abspath(
                file_path
            )
        ),
        exist_ok=True,
    )

    with open(
        file_path,
        "wb",
    ) as f:

        f.write(
            _HEADER.pack(
                ROOT_MAGIC,
                VERSION,
                1,
            )
        )

        f.write(
            _ROOT.pack(
                root_hash
            )
        )

    return os.path.getsize(
        file_path
    )


# ============================================================
# Load trusted root_S
# ============================================================

def load_trusted_root(
    file_path: str,
) -> bytes:

    with open(
        file_path,
        "rb",
    ) as f:

        (
            magic,
            version,
            count,
        ) = _HEADER.unpack(
            _read_exact(
                f,
                _HEADER.size,
            )
        )

        if magic != ROOT_MAGIC:

            raise ValueError(
                "Invalid root_S file"
            )

        if version != VERSION:

            raise ValueError(
                f"Unsupported root version: "
                f"{version}"
            )

        if count != 1:

            raise ValueError(
                "Invalid count field in root file"
            )

        root_hash = (
            _ROOT.unpack(
                _read_exact(
                    f,
                    _ROOT.size,
                )
            )[0]
        )

        if f.read(1):

            raise ValueError(
                "Extra data found at the end of the root file"
            )

    return root_hash
