from __future__ import annotations

from array import array
from time import perf_counter

from road_network import RoadNetwork


# ============================================================
# L0 Node Types
# ============================================================

NODE_LEAF = 0
NODE_INTERNAL = 1


# ============================================================
# Leaf Origin
#
# NORMAL:
#     Normal Leaf formed when recursive partitioning reaches the termination condition
#
# MID:
#     E_mid crossing the current partition boundary,
#     directly forms a Mid Leaf without further recursion
# ============================================================

LEAF_NORMAL = 0
LEAF_MID = 1


# ============================================================
# SpatialSkeleton
# ============================================================

class SpatialSkeleton:
    """
    L0 spatial skeleton.

    This file is responsible only for:

        1. Spatial partitioning
        2. left / mid / right structure
        3. Which eid values belong to each Leaf
        4. The MBR of each node

    Not included yet:

        min_start
        max_end
        L1 root
        L0 hash

    These will be added after L1 is built.


    ----------------------------------------------------------
    Nodes are stored in postorder:

        left
        mid
        right
        parent

    Therefore, when computing the L0 hash later,
    the nodes can be scanned sequentially from node 0 onward.
    ----------------------------------------------------------
    """

    __slots__ = (
        "node_type",
        "leaf_origin",

        "min_lon",
        "min_lat",
        "max_lon",
        "max_lat",

        "left_child",
        "mid_child",
        "right_child",

        "leaf_edge_offset",
        "leaf_edge_count",
        "leaf_edge_ids",

        "root_index",

        "leaf_count",
        "normal_leaf_count",
        "mid_leaf_count",
        "internal_count",
        "empty_leaf_count",
    )

    def __init__(self):

        # ====================================================
        # Node Types
        # ====================================================

        self.node_type = bytearray()

        # Leaf Origin
        self.leaf_origin = bytearray()

        # ====================================================
        # Node MBR
        # ====================================================

        self.min_lon = array("d")
        self.min_lat = array("d")
        self.max_lon = array("d")
        self.max_lat = array("d")

        # ====================================================
        # Three Children of an Internal Node
        #
        # Leaf positions use -1
        # ====================================================

        self.left_child = array("i")
        self.mid_child = array("i")
        self.right_child = array("i")

        # ====================================================
        # Store the eid values of all Leaves in one contiguous array
        #
        # Each Leaf stores only:
        #
        # offset
        # count
        #
        # Avoid:
        #
        # list[list[int]]
        #
        # creating a large number of small objects.
        # ====================================================

        self.leaf_edge_offset = array("Q")
        self.leaf_edge_count = array("I")

        self.leaf_edge_ids = array("I")

        # Root
        self.root_index = -1

        # ====================================================
        # Statistics
        # ====================================================

        self.leaf_count = 0
        self.normal_leaf_count = 0
        self.mid_leaf_count = 0
        self.internal_count = 0
        self.empty_leaf_count = 0

    # ========================================================
    # Total Node Count
    # ========================================================

    @property
    def node_count(self) -> int:

        return len(
            self.node_type
        )

    # ========================================================
    # Add Leaf
    # ========================================================

    def add_leaf(
        self,
        eids: list[int],
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        origin: int,
    ) -> int:

        node_index = len(
            self.node_type
        )

        # ====================================================
        # Road edges inside each Leaf are always sorted by eid
        #
        # This order is used directly when building L1 later.
        # ====================================================

        if len(eids) > 1:

            eids.sort()

        offset = len(
            self.leaf_edge_ids
        )

        count = len(
            eids
        )

        self.leaf_edge_ids.extend(
            eids
        )

        # ====================================================
        # Write Node
        # ====================================================

        self.node_type.append(
            NODE_LEAF
        )

        self.leaf_origin.append(
            origin
        )

        self.min_lon.append(
            min_lon
        )

        self.min_lat.append(
            min_lat
        )

        self.max_lon.append(
            max_lon
        )

        self.max_lat.append(
            max_lat
        )

        self.left_child.append(-1)
        self.mid_child.append(-1)
        self.right_child.append(-1)

        self.leaf_edge_offset.append(
            offset
        )

        self.leaf_edge_count.append(
            count
        )

        # ====================================================
        # Statistics
        # ====================================================

        self.leaf_count += 1

        if origin == LEAF_MID:

            self.mid_leaf_count += 1

        else:

            self.normal_leaf_count += 1

        if count == 0:

            self.empty_leaf_count += 1

        return node_index

    # ========================================================
    # Add Internal
    # ========================================================

    def add_internal(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        left: int,
        mid: int,
        right: int,
    ) -> int:

        node_index = len(
            self.node_type
        )

        self.node_type.append(
            NODE_INTERNAL
        )

        # Internal nodes have no Leaf origin
        self.leaf_origin.append(0)

        self.min_lon.append(
            min_lon
        )

        self.min_lat.append(
            min_lat
        )

        self.max_lon.append(
            max_lon
        )

        self.max_lat.append(
            max_lat
        )

        self.left_child.append(
            left
        )

        self.mid_child.append(
            mid
        )

        self.right_child.append(
            right
        )

        # Internal nodes have no Leaf road edges
        self.leaf_edge_offset.append(0)
        self.leaf_edge_count.append(0)

        self.internal_count += 1

        return node_index

    # ========================================================
    # Get All Road Edges of a Leaf
    # ========================================================

    def get_leaf_edges(
        self,
        node_index: int,
    ) -> list[int]:

        if (
            self.node_type[node_index]
            != NODE_LEAF
        ):

            raise ValueError(
                f"node={node_index} is not a Leaf"
            )

        offset = (
            self.leaf_edge_offset[
                node_index
            ]
        )

        count = (
            self.leaf_edge_count[
                node_index
            ]
        )

        return list(
            self.leaf_edge_ids[
                offset:
                offset + count
            ]
        )


# ============================================================
# Vertex Set MBR
# ============================================================

def vertex_mbr(
    vertices: list[int],
    road: RoadNetwork,
) -> tuple[
    float,
    float,
    float,
    float,
]:

    if not vertices:

        raise ValueError(
            "vertices cannot be empty"
        )

    node_lon = road.node_lon
    node_lat = road.node_lat

    nid = vertices[0]

    min_lon = node_lon[nid]
    max_lon = min_lon

    min_lat = node_lat[nid]
    max_lat = min_lat

    for i in range(
        1,
        len(vertices),
    ):

        nid = vertices[i]

        lon = node_lon[nid]
        lat = node_lat[nid]

        if lon < min_lon:

            min_lon = lon

        elif lon > max_lon:

            max_lon = lon

        if lat < min_lat:

            min_lat = lat

        elif lat > max_lat:

            max_lat = lat

    return (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )


# ============================================================
# Road Edge Set MBR
# ============================================================

def edges_mbr(
    eids: list[int],
    road: RoadNetwork,
) -> tuple[
    float,
    float,
    float,
    float,
]:

    if not eids:

        raise ValueError(
            "eids cannot be empty"
        )

    edge_min_lon = road.edge_min_lon
    edge_min_lat = road.edge_min_lat
    edge_max_lon = road.edge_max_lon
    edge_max_lat = road.edge_max_lat

    eid = eids[0]

    min_lon = edge_min_lon[eid]
    min_lat = edge_min_lat[eid]

    max_lon = edge_max_lon[eid]
    max_lat = edge_max_lat[eid]

    for i in range(
        1,
        len(eids),
    ):

        eid = eids[i]

        value = (
            edge_min_lon[eid]
        )

        if value < min_lon:

            min_lon = value

        value = (
            edge_min_lat[eid]
        )

        if value < min_lat:

            min_lat = value

        value = (
            edge_max_lon[eid]
        )

        if value > max_lon:

            max_lon = value

        value = (
            edge_max_lat[eid]
        )

        if value > max_lat:

            max_lat = value

    return (
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )


# ============================================================
# Weighted Vertex Partition
# ============================================================

def weighted_split_vertices(
    vertices: list[int],
    eids: list[int],
    road: RoadNetwork,
    use_lon: bool,
) -> tuple[
    list[int],
    list[int],
]:
    """
    Find the spatial split position using vertex weights.


    Vertex weight:

        weight(v)

        =

        Number of road edges connected to vertex v in the current Eset


    For example:

        v1 -- v2 -- v3
               |
               v4

    Current region:

        weight(v1) = 1
        weight(v2) = 3
        weight(v3) = 1
        weight(v4) = 1


    ----------------------------------------------------------
    Goal:

        left_weight
        ≈
        right_weight


    That is:

        minimize

        |left_weight - right_weight|


    Equivalent to:

        minimize

        |2 * left_weight - total_weight|
    ----------------------------------------------------------
    """

    vertex_count = len(
        vertices
    )

    if vertex_count < 2:

        raise ValueError(
            "Weighted partitioning requires at least 2 vertices"
        )

    # ========================================================
    # 1. Compute Vertex Weights in the Current Region
    # ========================================================

    weights = {
        nid: 0
        for nid in vertices
    }

    edge_u = road.edge_u
    edge_v = road.edge_v

    for eid in eids:

        u = edge_u[eid]
        v = edge_v[eid]

        weights[u] += 1

        # Normal road edges do not contain self-loops,
        # but still avoid counting the same vertex twice here.
        if v != u:

            weights[v] += 1

    # ========================================================
    # 2. Sort Vertices Along the Longer Axis of the Current Region
    #
    # When coordinates are equal:
    #
    #     nid
    #
    # use nid as the secondary key.
    #
    # This guarantees identical results for the same road network on every build.
    # ========================================================

    ordered = list(
        vertices
    )

    if use_lon:

        node_lon = road.node_lon

        ordered.sort(
            key=lambda nid: (
                node_lon[nid],
                nid,
            )
        )

    else:

        node_lat = road.node_lat

        ordered.sort(
            key=lambda nid: (
                node_lat[nid],
                nid,
            )
        )

    # ========================================================
    # 3. Total Weight
    # ========================================================

    total_weight = 0

    for nid in ordered:

        total_weight += (
            weights[nid]
        )

    # ========================================================
    # Extreme Case:
    #
    # The current region has no edge weight.
    #
    # Fall back to a simple split by vertex count.
    # ========================================================

    if total_weight == 0:

        split_position = (
            vertex_count // 2
        )

        return (
            ordered[
                :split_position
            ],
            ordered[
                split_position:
            ],
        )

    # ========================================================
    # 4. Find the Optimal Weighted Split Position
    #
    # Must guarantee:
    #
    # left has at least 1 vertex
    # right has at least 1 vertex
    # ========================================================

    left_weight = 0

    best_position = 1

    best_difference = None

    for position in range(
        1,
        vertex_count,
    ):

        nid = ordered[
            position - 1
        ]

        left_weight += (
            weights[nid]
        )

        difference = abs(
            2 * left_weight
            - total_weight
        )

        # ----------------------------------------------------
        # Do not update on a tie.
        #
        # Therefore, choose the smaller split position in a tie,
        # ensuring deterministic behavior.
        # ----------------------------------------------------

        if (
            best_difference is None
            or
            difference < best_difference
        ):

            best_difference = (
                difference
            )

            best_position = (
                position
            )

    # ========================================================
    # 5. Obtain Left and Right Vertices
    # ========================================================

    left_vertices = (
        ordered[
            :best_position
        ]
    )

    right_vertices = (
        ordered[
            best_position:
        ]
    )

    if (
        not left_vertices
        or
        not right_vertices
    ):

        raise RuntimeError(
            "Weighted spatial partition produced an empty vertex set"
        )

    return (
        left_vertices,
        right_vertices,
    )


# ============================================================
# L0 Builder
# ============================================================

class SpatialSkeletonBuilder:
    """
    Build the weighted L0 spatial skeleton.
    """

    __slots__ = (
        "road",
        "theta",
        "skeleton",
    )

    def __init__(
        self,
        road: RoadNetwork,
        theta: int,
    ):

        if theta < 2:

            raise ValueError(
                "theta must be >= 2"
            )

        self.road = road

        self.theta = theta

        self.skeleton = (
            SpatialSkeleton()
        )

    # ========================================================
    # Recursive Construction
    # ========================================================

    def _build(
        self,
        vertices: list[int],
        eids: list[int],
    ) -> int:

        road = self.road

        # ====================================================
        # No Road Edges
        #
        # Form an empty Normal Leaf.
        # ====================================================

        if not eids:

            (
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            ) = vertex_mbr(
                vertices,
                road,
            )

            return (
                self.skeleton.add_leaf(
                    eids=[],
                    min_lon=min_lon,
                    min_lat=min_lat,
                    max_lon=max_lon,
                    max_lat=max_lat,
                    origin=LEAF_NORMAL,
                )
            )

        # ====================================================
        # Actual MBR of the Current Eset
        # ====================================================

        (
            current_min_lon,
            current_min_lat,
            current_max_lon,
            current_max_lat,
        ) = edges_mbr(
            eids,
            road,
        )

        # ====================================================
        # Termination Condition
        #
        # Currently still use:
        #
        #     |Vset| <= theta
        #
        # theta controls the granularity of the spatial skeleton.
        #
        # Note:
        #
        # "How to split"
        # has been changed to weighted partitioning.
        #
        # "When to stop"
        # still depends on the number of vertices and theta.
        # ====================================================

        if (
            len(vertices)
            <= self.theta
        ):

            return (
                self.skeleton.add_leaf(
                    eids=eids,
                    min_lon=current_min_lon,
                    min_lat=current_min_lat,
                    max_lon=current_max_lon,
                    max_lat=current_max_lat,
                    origin=LEAF_NORMAL,
                )
            )

        # ====================================================
        # MBR of the Current Vertex Region
        #
        # Used to determine the longer axis.
        # ====================================================

        (
            vertex_min_lon,
            vertex_min_lat,
            vertex_max_lon,
            vertex_max_lat,
        ) = vertex_mbr(
            vertices,
            road,
        )

        lon_span = (
            vertex_max_lon
            - vertex_min_lon
        )

        lat_span = (
            vertex_max_lat
            - vertex_min_lat
        )

        # ====================================================
        # Select the Longer Axis
        #
        # If longitude span is larger:
        #     split along longitude
        #
        # If latitude span is larger:
        #     split along latitude
        #
        # Tie:
        #     longitude
        # ====================================================

        use_lon = (
            lon_span
            >= lat_span
        )

        # ====================================================
        # Core:
        #
        # Weighted Binary Partition by Vertex Weight
        # ====================================================

        (
            left_vertices,
            right_vertices,
        ) = weighted_split_vertices(
            vertices=vertices,
            eids=eids,
            road=road,
            use_lon=use_lon,
        )

        # ====================================================
        # Build the Left Vertex Set
        #
        # Each edge later only needs to check whether its endpoints
        # belong to left.
        #
        # Because endpoints of the current eids must belong to:
        #
        # left ∪ right
        # ====================================================

        left_vertex_set = set(
            left_vertices
        )

        # ====================================================
        # E_left
        # E_mid
        # E_right
        # ====================================================

        left_edges: list[int] = []

        mid_edges: list[int] = []

        right_edges: list[int] = []

        append_left = (
            left_edges.append
        )

        append_mid = (
            mid_edges.append
        )

        append_right = (
            right_edges.append
        )

        edge_u = road.edge_u
        edge_v = road.edge_v

        # ====================================================
        # Road Edge Classification
        # ====================================================

        for eid in eids:

            u = edge_u[eid]
            v = edge_v[eid]

            u_left = (
                u in left_vertex_set
            )

            v_left = (
                v in left_vertex_set
            )

            # ------------------------------------------------
            # Both Endpoints Are on the Left
            # ------------------------------------------------

            if (
                u_left
                and
                v_left
            ):

                append_left(
                    eid
                )

            # ------------------------------------------------
            # Both Endpoints Are on the Right
            # ------------------------------------------------

            elif (
                not u_left
                and
                not v_left
            ):

                append_right(
                    eid
                )

            # ------------------------------------------------
            # One Endpoint Left and One Endpoint Right
            #
            # crossing edge
            #
            # Directly enter E_mid
            # ------------------------------------------------

            else:

                append_mid(
                    eid
                )

        # ====================================================
        # Build Left First
        # ====================================================

        left_index = (
            self._build(
                vertices=left_vertices,
                eids=left_edges,
            )
        )

        # ====================================================
        # Mid
        #
        # E_mid is not recursively partitioned.
        #
        # It directly becomes a Mid Leaf.
        # ====================================================

        if mid_edges:

            (
                mid_min_lon,
                mid_min_lat,
                mid_max_lon,
                mid_max_lat,
            ) = edges_mbr(
                mid_edges,
                road,
            )

        else:

            # ------------------------------------------------
            # Empty Mid Leaf.
            #
            # There is no L0 hash yet,
            # and a unified empty-Leaf time range and L1 root will be defined later.
            #
            # For now, use all zeros as the deterministic MBR placeholder.
            # ------------------------------------------------

            mid_min_lon = 0.0
            mid_min_lat = 0.0
            mid_max_lon = 0.0
            mid_max_lat = 0.0

        mid_index = (
            self.skeleton.add_leaf(
                eids=mid_edges,
                min_lon=mid_min_lon,
                min_lat=mid_min_lat,
                max_lon=mid_max_lon,
                max_lat=mid_max_lat,
                origin=LEAF_MID,
            )
        )

        # ====================================================
        # Right
        # ====================================================

        right_index = (
            self._build(
                vertices=right_vertices,
                eids=right_edges,
            )
        )

        # ====================================================
        # Create the parent node last.
        #
        # Therefore, this naturally forms:
        #
        # left
        # mid
        # right
        # parent
        #
        # postorder storage.
        # ====================================================

        return (
            self.skeleton.add_internal(
                min_lon=current_min_lon,
                min_lat=current_min_lat,
                max_lon=current_max_lon,
                max_lat=current_max_lat,
                left=left_index,
                mid=mid_index,
                right=right_index,
            )
        )

    # ========================================================
    # Main Entry
    # ========================================================

    def build(
        self,
    ) -> SpatialSkeleton:

        road = self.road

        # ====================================================
        # All Real Nodes
        # ====================================================

        vertices = list(
            range(
                road.node_count
            )
        )

        # ====================================================
        # All Real Road Edges
        #
        # eid values may have gaps.
        # ====================================================

        edge_u = road.edge_u

        eids: list[int] = []

        append_eid = (
            eids.append
        )

        for eid in range(
            len(edge_u)
        ):

            if edge_u[eid] != -1:

                append_eid(
                    eid
                )

        # ====================================================
        # Build
        # ====================================================

        root_index = (
            self._build(
                vertices=vertices,
                eids=eids,
            )
        )

        self.skeleton.root_index = (
            root_index
        )

        return self.skeleton


# ============================================================
# Check Whether Each Road Edge Appears in Exactly One Leaf
# ============================================================

def validate_edge_partition(
    skeleton: SpatialSkeleton,
    road: RoadNetwork,
) -> None:
    """
    Must satisfy:

        Every real road edge

        Appears exactly once

    Must not:

        Be omitted
        Be duplicated
    """

    seen = bytearray(
        road.max_eid + 1
    )

    node_type = (
        skeleton.node_type
    )

    leaf_edge_offset = (
        skeleton.leaf_edge_offset
    )

    leaf_edge_count = (
        skeleton.leaf_edge_count
    )

    leaf_edge_ids = (
        skeleton.leaf_edge_ids
    )

    # ========================================================
    # Scan All Leaves
    # ========================================================

    for node_index in range(
        skeleton.node_count
    ):

        if (
            node_type[node_index]
            != NODE_LEAF
        ):

            continue

        offset = (
            leaf_edge_offset[
                node_index
            ]
        )

        count = (
            leaf_edge_count[
                node_index
            ]
        )

        end = (
            offset + count
        )

        for position in range(
            offset,
            end,
        ):

            eid = (
                leaf_edge_ids[
                    position
                ]
            )

            if seen[eid]:

                raise RuntimeError(
                    f"eid={eid} "
                    f"is assigned to multiple Leaves"
                )

            seen[eid] = 1

    # ========================================================
    # Check for Missing Edges
    # ========================================================

    for eid in range(
        road.max_eid + 1
    ):

        if not road.edge_exists(
            eid
        ):

            continue

        if not seen[eid]:

            raise RuntimeError(
                f"eid={eid} "
                f"is not assigned to any Leaf"
            )


# ============================================================
# Leaf Road Edge Count Statistics
# ============================================================

def calculate_leaf_edge_statistics(
    skeleton: SpatialSkeleton,
) -> tuple[
    int,
    int,
    float,
    int,
]:
    """
    Returns:

        Minimum road edge count
        Maximum road edge count
        Average road edge count
        Non-empty Leaf count
    """

    minimum = None

    maximum = 0

    total = 0

    non_empty = 0

    for node_index in range(
        skeleton.node_count
    ):

        if (
            skeleton.node_type[
                node_index
            ]
            != NODE_LEAF
        ):

            continue

        count = (
            skeleton.leaf_edge_count[
                node_index
            ]
        )

        if count == 0:

            continue

        non_empty += 1

        total += count

        if (
            minimum is None
            or
            count < minimum
        ):

            minimum = count

        if count > maximum:

            maximum = count

    if non_empty == 0:

        return (
            0,
            0,
            0.0,
            0,
        )

    average = (
        total / non_empty
    )

    return (
        minimum,
        maximum,
        average,
        non_empty,
    )


# ============================================================
# Run Directly with the PyCharm Green Triangle
# ============================================================

def main():
    NODE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_nodes.txt"
    )

    EDGE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_edges.txt"
    )

    # ========================================================
    # theta
    #
    # The current meaning is still:
    #
    # Stop further partitioning when |Vset| <= theta.
    #
    # The split position already uses vertex weights.
    #
    # Use 64 for now.
    # ========================================================

    THETA = 64

    # ========================================================
    # Load Road Network
    # ========================================================

    print(
        "Loading road network..."
    )

    road = RoadNetwork()

    road.load(
        node_file=NODE_FILE,
        edge_file=EDGE_FILE,
    )

    print(
        "Road network loaded"
    )

    print(
        "Node count:",
        road.node_count,
    )

    print(
        "Road edge count:",
        road.edge_count,
    )

    # ========================================================
    # Build Weighted L0
    # ========================================================

    print()

    print(
        "===== Start Building Weighted L0 Spatial Skeleton ====="
    )

    print(
        "Vertex weight = number of connected road edges in the current region"
    )

    start_time = (
        perf_counter()
    )

    builder = (
        SpatialSkeletonBuilder(
            road=road,
            theta=THETA,
        )
    )

    skeleton = (
        builder.build()
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    # ========================================================
    # Integrity Check
    # ========================================================

    print()

    print(
        "Checking road edge assignment..."
    )

    validate_edge_partition(
        skeleton=skeleton,
        road=road,
    )

    print(
        "Road edge assignment check passed"
    )

    # ========================================================
    # Leaf Road Edge Distribution
    # ========================================================

    (
        min_leaf_edges,
        max_leaf_edges,
        avg_leaf_edges,
        non_empty_leaf_count,
    ) = (
        calculate_leaf_edge_statistics(
            skeleton
        )
    )

    # ========================================================
    # Overall Statistics
    # ========================================================

    print()

    print(
        "===== L0 Spatial Skeleton Construction Completed ====="
    )

    print(
        "theta:",
        THETA,
    )

    print(
        "Total node count:",
        skeleton.node_count,
    )

    print(
        "Internal count:",
        skeleton.internal_count,
    )

    print(
        "Total Leaf count:",
        skeleton.leaf_count,
    )

    print(
        "Normal Leaf count:",
        skeleton.normal_leaf_count,
    )

    print(
        "Mid Leaf count:",
        skeleton.mid_leaf_count,
    )

    print(
        "Empty Leaf count:",
        skeleton.empty_leaf_count,
    )

    print(
        "Non-empty Leaf count:",
        non_empty_leaf_count,
    )

    print(
        "Total road edge count in Leaves:",
        len(
            skeleton.leaf_edge_ids
        ),
    )

    print(
        "Real road edge count:",
        road.edge_count,
    )

    print(
        "root index:",
        skeleton.root_index,
    )

    print(
        "Construction time:",
        f"{elapsed:.6f} s",
    )

    # ========================================================
    # Check Whether Road Edges Are Relatively Balanced
    # ========================================================

    print()

    print(
        "===== Leaf Road Edge Distribution ====="
    )

    print(
        "Minimum road edge count:",
        min_leaf_edges,
    )

    print(
        "Maximum road edge count:",
        max_leaf_edges,
    )

    print(
        "Average road edge count:",
        f"{avg_leaf_edges:.3f}",
    )

    # ========================================================
    # Root MBR
    # ========================================================

    root = (
        skeleton.root_index
    )

    print()

    print(
        "===== Root MBR ====="
    )

    print(
        "min_lon:",
        skeleton.min_lon[
            root
        ],
    )

    print(
        "min_lat:",
        skeleton.min_lat[
            root
        ],
    )

    print(
        "max_lon:",
        skeleton.max_lon[
            root
        ],
    )

    print(
        "max_lat:",
        skeleton.max_lat[
            root
        ],
    )

    # ========================================================
    # Print the First 5 Non-Empty Leaves
    # ========================================================

    print()

    print(
        "===== Sample Leaves ====="
    )

    shown = 0

    for node_index in range(
        skeleton.node_count
    ):

        if (
            skeleton.node_type[
                node_index
            ]
            != NODE_LEAF
        ):

            continue

        count = (
            skeleton.leaf_edge_count[
                node_index
            ]
        )

        if count == 0:

            continue

        origin = (
            skeleton.leaf_origin[
                node_index
            ]
        )

        if origin == LEAF_MID:

            origin_text = "Mid"

        else:

            origin_text = "Normal"

        edges = (
            skeleton.get_leaf_edges(
                node_index
            )
        )

        print()

        print(
            f"Leaf node = {node_index}"
        )

        print(
            "Type:",
            origin_text,
        )

        print(
            "Road edge count:",
            count,
        )

        print(
            "First few eid values:",
            edges[:10],
        )

        print(
            "MBR:",
            (
                skeleton.min_lon[
                    node_index
                ],
                skeleton.min_lat[
                    node_index
                ],
                skeleton.max_lon[
                    node_index
                ],
                skeleton.max_lat[
                    node_index
                ],
            ),
        )

        shown += 1

        if shown >= 5:

            break

    print()

    print(
        "spatial_skeleton.py completed successfully"
    )


if __name__ == "__main__":
    main()
