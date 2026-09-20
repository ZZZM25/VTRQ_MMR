from __future__ import annotations

from time import perf_counter


class RoadNetwork:
    """
    Efficient road network data structure.

    Nodes:
        node_id -> longitude / latitude

    Road edges:
        eid -> u / v

    Undirected edge mapping:
        (u, v) -> eid

    Note:
        eid values do not need to be contiguous; they only need to be unique.
        For fast later access, eid is still used directly as the array index.
        Missing eid values are represented by the placeholder -1.
    """

    __slots__ = (
        "node_lon",
        "node_lat",
        "edge_u",
        "edge_v",
        "edge_min_lon",
        "edge_min_lat",
        "edge_max_lon",
        "edge_max_lat",
        "edge_lookup",
        "_edge_count",
    )

    def __init__(self):

        # ====================================================
        # Nodes
        #
        # node_id is used directly as the array index:
        #
        # node_lon[node_id]
        # node_lat[node_id]
        # ====================================================

        self.node_lon: list[float] = []
        self.node_lat: list[float] = []

        # ====================================================
        # Road Edges
        #
        # eid is used directly as the array index:
        #
        # edge_u[eid]
        # edge_v[eid]
        #
        # If an eid does not exist:
        #
        # edge_u[eid] = -1
        # edge_v[eid] = -1
        # ====================================================

        self.edge_u: list[int] = []
        self.edge_v: list[int] = []

        # ====================================================
        # MBR of Each Road Edge
        #
        # Used directly later for L0 construction and spatial queries,
        # avoiding recalculation from the two endpoints each time.
        # ====================================================

        self.edge_min_lon: list[float] = []
        self.edge_min_lat: list[float] = []
        self.edge_max_lon: list[float] = []
        self.edge_max_lat: list[float] = []

        # ====================================================
        # Undirected Edge Mapping
        #
        # packed(u, v) -> eid
        #
        # Do not use a tuple as the key,
        # instead pack the two node IDs into one integer.
        # ====================================================

        self.edge_lookup: dict[int, int] = {}

        # Number of actual road edges
        self._edge_count = 0

    # ========================================================
    # Undirected Edge Key
    # ========================================================

    @staticmethod
    def edge_key(
        u: int,
        v: int,
    ) -> int:
        """
        Pack the undirected edge (u,v) into one integer.

        For example:

            (10, 20)
            (20, 10)

        will produce exactly the same key.

        Encoding:

            min(u,v) << 32 | max(u,v)
        """

        if u > v:
            u, v = v, u

        return (
            (u << 32)
            | v
        )

    # ========================================================
    # Load Nodes
    # ========================================================

    def load_nodes(
        self,
        filename: str,
    ) -> None:
        """
        Node file format:

            node_id longitude latitude

        The current Chengdu node file has contiguous node_id values,
        so a list is used directly for storage,
        which makes later coordinate access faster than using a dict.
        """

        node_lon = self.node_lon
        node_lat = self.node_lat

        with open(
            filename,
            "r",
            encoding="utf-8",
            buffering=1024 * 1024,
        ) as f:

            for line_no, line in enumerate(
                f,
                start=1,
            ):

                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid node file format at line {line_no}: "
                        f"{line.rstrip()}"
                    )

                nid = int(parts[0])
                lon = float(parts[1])
                lat = float(parts[2])

                # ============================================
                # Node IDs Must Be Contiguous
                #
                # Because we want direct access using:
                #
                # node_lon[nid]
                #
                # If node IDs are found to be non-contiguous in the future,
                # change this to the same placeholder-array structure used for eid.
                # ============================================

                expected_nid = len(
                    node_lon
                )

                if nid != expected_nid:
                    raise ValueError(
                        f"Non-contiguous node ID: "
                        f"expected {expected_nid}, "
                        f"actual {nid}"
                    )

                node_lon.append(
                    lon
                )

                node_lat.append(
                    lat
                )

    # ========================================================
    # Load Road Edges
    # ========================================================

    def load_edges(
        self,
        filename: str,
    ) -> None:
        """
        Road edge file format:

            eid u v

        eid values may be non-contiguous; they only need to be unique.

        To maintain fast later access:

            edge_u[eid]
            edge_v[eid]

        Missing eid positions in the arrays use -1 as a placeholder.
        """

        node_lon = self.node_lon
        node_lat = self.node_lat

        node_count = len(
            node_lon
        )

        edge_u = self.edge_u
        edge_v = self.edge_v

        edge_min_lon = self.edge_min_lon
        edge_min_lat = self.edge_min_lat
        edge_max_lon = self.edge_max_lon
        edge_max_lat = self.edge_max_lat

        edge_lookup = self.edge_lookup

        edge_count = 0

        with open(
            filename,
            "r",
            encoding="utf-8",
            buffering=1024 * 1024,
        ) as f:

            for line_no, line in enumerate(
                f,
                start=1,
            ):

                if not line.strip():
                    continue

                parts = line.split()

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid road edge file format at line {line_no}: "
                        f"{line.rstrip()}"
                    )

                eid = int(
                    parts[0]
                )

                u = int(
                    parts[1]
                )

                v = int(
                    parts[2]
                )

                # ============================================
                # Check eid
                # ============================================

                if eid < 0:
                    raise ValueError(
                        f"eid cannot be negative: {eid}"
                    )

                # ============================================
                # Check Whether Nodes Exist
                # ============================================

                if (
                    u < 0
                    or
                    v < 0
                    or
                    u >= node_count
                    or
                    v >= node_count
                ):
                    raise ValueError(
                        f"eid={eid} uses nonexistent nodes: "
                        f"{u}, {v}"
                    )

                # ============================================
                # Automatically Expand Arrays Based on eid
                #
                # For example:
                #
                # Current maximum eid = 44
                # The next edge directly uses eid = 46
                #
                # Then automatically add:
                #
                # index 45 -> -1
                # index 46 -> current road edge
                # ============================================

                if eid >= len(
                    edge_u
                ):

                    add_count = (
                        eid
                        + 1
                        - len(
                            edge_u
                        )
                    )

                    edge_u.extend(
                        [-1]
                        * add_count
                    )

                    edge_v.extend(
                        [-1]
                        * add_count
                    )

                    edge_min_lon.extend(
                        [0.0]
                        * add_count
                    )

                    edge_min_lat.extend(
                        [0.0]
                        * add_count
                    )

                    edge_max_lon.extend(
                        [0.0]
                        * add_count
                    )

                    edge_max_lat.extend(
                        [0.0]
                        * add_count
                    )

                # ============================================
                # eid Must Be Unique
                # ============================================

                if (
                    edge_u[eid]
                    != -1
                ):
                    raise ValueError(
                        f"Duplicate eid detected: "
                        f"{eid}"
                    )

                # ============================================
                # Save Road Edge
                # ============================================

                edge_u[eid] = u
                edge_v[eid] = v

                # ============================================
                # Build Undirected Edge Mapping
                # ============================================

                key = self.edge_key(
                    u,
                    v,
                )

                if key in edge_lookup:

                    old_eid = (
                        edge_lookup[
                            key
                        ]
                    )

                    raise ValueError(
                        f"Duplicate undirected road edge detected: "
                        f"eid={old_eid} "
                        f"and eid={eid}"
                    )

                edge_lookup[
                    key
                ] = eid

                # ============================================
                # Precompute Road Edge MBR
                # ============================================

                u_lon = (
                    node_lon[u]
                )

                u_lat = (
                    node_lat[u]
                )

                v_lon = (
                    node_lon[v]
                )

                v_lat = (
                    node_lat[v]
                )

                if (
                    u_lon
                    <= v_lon
                ):
                    edge_min_lon[
                        eid
                    ] = u_lon

                    edge_max_lon[
                        eid
                    ] = v_lon

                else:
                    edge_min_lon[
                        eid
                    ] = v_lon

                    edge_max_lon[
                        eid
                    ] = u_lon

                if (
                    u_lat
                    <= v_lat
                ):
                    edge_min_lat[
                        eid
                    ] = u_lat

                    edge_max_lat[
                        eid
                    ] = v_lat

                else:
                    edge_min_lat[
                        eid
                    ] = v_lat

                    edge_max_lat[
                        eid
                    ] = u_lat

                edge_count += 1

        self._edge_count = (
            edge_count
        )

    # ========================================================
    # Load the Complete Road Network
    # ========================================================

    def load(
        self,
        node_file: str,
        edge_file: str,
    ) -> None:

        self.load_nodes(
            node_file
        )

        self.load_edges(
            edge_file
        )

    # ========================================================
    # Check Whether an eid Actually Exists
    # ========================================================

    def edge_exists(
        self,
        eid: int,
    ) -> bool:

        return (
            0
            <= eid
            < len(
                self.edge_u
            )
            and
            self.edge_u[
                eid
            ]
            != -1
        )

    # ========================================================
    # Trajectory Node Pair -> eid
    # ========================================================

    def find_eid(
        self,
        u: int,
        v: int,
    ) -> int | None:
        """
        Undirected lookup.

        For example:

            find_eid(1, 1540)

        and:

            find_eid(1540, 1)

        return exactly the same result.
        """

        key = self.edge_key(
            u,
            v,
        )

        return (
            self.edge_lookup.get(
                key
            )
        )

    # ========================================================
    # Get the Two Nodes of a Road Edge
    # ========================================================

    def get_edge_nodes(
        self,
        eid: int,
    ) -> tuple[
        int,
        int,
    ]:

        if not self.edge_exists(
            eid
        ):
            raise KeyError(
                f"Road edge eid={eid} does not exist"
            )

        return (
            self.edge_u[
                eid
            ],
            self.edge_v[
                eid
            ],
        )

    # ========================================================
    # Get Node Coordinates
    # ========================================================

    def get_node_coordinates(
        self,
        nid: int,
    ) -> tuple[
        float,
        float,
    ]:

        return (
            self.node_lon[
                nid
            ],
            self.node_lat[
                nid
            ],
        )

    # ========================================================
    # Get Road Edge Coordinates
    # ========================================================

    def get_edge_coordinates(
        self,
        eid: int,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Returns:

            u_lon,
            u_lat,
            v_lon,
            v_lat
        """

        if not self.edge_exists(
            eid
        ):
            raise KeyError(
                f"Road edge eid={eid} does not exist"
            )

        u = self.edge_u[
            eid
        ]

        v = self.edge_v[
            eid
        ]

        return (
            self.node_lon[
                u
            ],
            self.node_lat[
                u
            ],
            self.node_lon[
                v
            ],
            self.node_lat[
                v
            ],
        )

    # ========================================================
    # Get Road Edge MBR
    # ========================================================

    def get_edge_mbr(
        self,
        eid: int,
    ) -> tuple[
        float,
        float,
        float,
        float,
    ]:
        """
        Returns:

            min_lon,
            min_lat,
            max_lon,
            max_lat
        """

        if not self.edge_exists(
            eid
        ):
            raise KeyError(
                f"Road edge eid={eid} does not exist"
            )

        return (
            self.edge_min_lon[
                eid
            ],
            self.edge_min_lat[
                eid
            ],
            self.edge_max_lon[
                eid
            ],
            self.edge_max_lat[
                eid
            ],
        )

    # ========================================================
    # Road Network Statistics
    # ========================================================

    @property
    def node_count(
        self,
    ) -> int:
        """
        Actual number of nodes.
        """

        return len(
            self.node_lon
        )

    @property
    def edge_count(
        self,
    ) -> int:
        """
        Actual number of existing road edges.

        Note:

            edge_count

        is not equal to:

            max_eid + 1

        because eid values may have gaps.
        """

        return (
            self._edge_count
        )

    @property
    def max_eid(
        self,
    ) -> int:
        """
        Current maximum eid.
        """

        if not self.edge_u:
            return -1

        return (
            len(
                self.edge_u
            )
            - 1
        )


# ============================================================
# Run Directly with the PyCharm Green Triangle
# ============================================================

def main():

    # ========================================================
    # Replace with your actual filenames
    # ========================================================

    NODE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_nodes.txt"
    )

    EDGE_FILE = (
        "E:\MMR_Trajectory_range\chengdu_edges.txt"
    )

    road = RoadNetwork()

    print(
        "Loading Chengdu road network..."
    )

    start_time = (
        perf_counter()
    )

    road.load(
        node_file=NODE_FILE,
        edge_file=EDGE_FILE,
    )

    elapsed = (
        perf_counter()
        - start_time
    )

    print()

    print(
        "===== Road Network Loading Completed ====="
    )

    print(
        "Node count:",
        road.node_count,
    )

    print(
        "Road edge count:",
        road.edge_count,
    )

    print(
        "Maximum eid:",
        road.max_eid,
    )

    print(
        "Loading time:",
        f"{elapsed:.6f} s",
    )

    # ========================================================
    # Test Existing eid=4
    # ========================================================

    print()

    print(
        "===== Edge 4 Test ====="
    )

    print(
        "Exists:",
        road.edge_exists(
            4
        ),
    )

    print(
        "Nodes:",
        road.get_edge_nodes(
            4
        ),
    )

    print(
        "Coordinates:",
        road.get_edge_coordinates(
            4
        ),
    )

    print(
        "MBR:",
        road.get_edge_mbr(
            4
        ),
    )

    # ========================================================
    # Test Missing eid=45
    # ========================================================

    print()

    print(
        "===== Missing eid Test ====="
    )

    print(
        "Does eid=45 exist:",
        road.edge_exists(
            45
        ),
    )

    print(
        "Does eid=46 exist:",
        road.edge_exists(
            46
        ),
    )

    # ========================================================
    # Undirected Edge Test
    #
    # In your actual data:
    #
    # eid = 4
    # u   = 1
    # v   = 1540
    # ========================================================

    print()

    print(
        "===== Undirected Edge Test ====="
    )

    eid1 = road.find_eid(
        1,
        1540,
    )

    eid2 = road.find_eid(
        1540,
        1,
    )

    print(
        "find_eid(1, 1540) =",
        eid1,
    )

    print(
        "find_eid(1540, 1) =",
        eid2,
    )

    if (
        eid1 != eid2
    ):
        raise RuntimeError(
            "Undirected edge test failed"
        )

    if (
        eid1 != 4
    ):
        raise RuntimeError(
            "Real road edge mapping test failed"
        )

    print()

    print(
        "road_network.py completed successfully"
    )


if __name__ == "__main__":
    main()
