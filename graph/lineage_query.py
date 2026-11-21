
"""
lineage_query.py

数据血缘下游影响查询模块。

核心功能：
1. 查询指定节点的直接和间接下游节点。
2. 使用 BFS 实现广度优先遍历。
3. 支持最大遍历深度和最大结果节点数量。
4. 支持完整血缘路径记录。
5. 支持字段级影响传播。
6. 支持 MultiDiGraph 的多重边。
7. 对环路和路径数量进行安全控制。
8. 严格返回 get_downstream_nodes MCP 工具约定的结构。

依赖：
    networkx
    graph.graph_loader

运行方式（项目根目录）：

    python -m graph.lineage_query

指定节点：

    python -m graph.lineage_query --node-id T01_CUSTOMER_RAW

查询参数：

    python -m graph.lineage_query \
        --node-id T01_CUSTOMER_RAW \
        --max-depth 5 \
        --max-nodes 50

关闭路径：

    python -m graph.lineage_query \
        --node-id T01_CUSTOMER_RAW \
        --no-paths

关闭字段映射：

    python -m graph.lineage_query \
        --node-id T01_CUSTOMER_RAW \
        --no-field-mapping
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from graph.graph_loader import load_graph


# ============================================================
# 1. 查询限制
# ============================================================

# 用于避免复杂图结构中简单路径数量呈指数级增长。
#
# 该值是内部执行保护参数，不属于 MCP API，
# 不会改变 get_downstream_nodes 的输入或输出结构。
MAX_PATH_STATES = 10000


# ============================================================
# 2. 下游血缘查询服务
# ============================================================

class LineageQueryService:
    """
    数据血缘查询服务。

    初始化时注入 NetworkX MultiDiGraph。

    MCP Server 后续只需要持有该 Service 实例，
    即可直接调用 get_downstream_nodes。

    示例：

        graph = load_graph()

        service = LineageQueryService(graph)

        result = service.get_downstream_nodes(
            node_id="T01_CUSTOMER_RAW"
        )
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph
    ) -> None:
        """
        初始化数据血缘查询服务。

        Args:
            graph:
                阶段一 graph_loader 加载的 MultiDiGraph。

        Raises:
            TypeError:
                如果传入的对象不是 MultiDiGraph。
        """

        if not isinstance(graph, nx.MultiDiGraph):

            raise TypeError(
                "graph 必须是 networkx.MultiDiGraph，"
                f"实际类型为 {type(graph).__name__}"
            )

        self.graph = graph

    # --------------------------------------------------------
    # 2.1 参数校验
    # --------------------------------------------------------

    @staticmethod
    def _validate_parameters(
        node_id: str,
        max_depth: int,
        include_paths: bool,
        include_field_mapping: bool,
        max_nodes: int
    ) -> None:
        """
        检查查询参数是否合法。

        参数约束与阶段一 MCP Tool Schema 保持一致。
        """

        if not isinstance(node_id, str) or not node_id.strip():

            raise ValueError(
                "node_id 必须是非空字符串"
            )

        # 使用 type(...) is int，避免将 bool 当作整数。
        if type(max_depth) is not int:

            raise TypeError(
                "max_depth 必须是整数"
            )

        if not 1 <= max_depth <= 20:

            raise ValueError(
                "max_depth 必须处于 1 到 20 之间"
            )

        if type(include_paths) is not bool:

            raise TypeError(
                "include_paths 必须是布尔值"
            )

        if type(include_field_mapping) is not bool:

            raise TypeError(
                "include_field_mapping 必须是布尔值"
            )

        if type(max_nodes) is not int:

            raise TypeError(
                "max_nodes 必须是整数"
            )

        if not 1 <= max_nodes <= 200:

            raise ValueError(
                "max_nodes 必须处于 1 到 200 之间"
            )

    # --------------------------------------------------------
    # 2.2 辅助方法
    # --------------------------------------------------------

    @staticmethod
    def _get_purpose_codes(
        node_attributes: dict[str, Any]
    ) -> list[str]:
        """
        从节点的 original_purposes 中提取 purpose_code。

        MCP get_downstream_nodes 契约中的
        nodes[].original_purposes 使用字符串数组。

        注意：
        get_node_details 返回完整节点对象时，
        仍然保留原始 original_purposes 对象数组。
        """

        purpose_codes = {
            purpose["purpose_code"]
            for purpose in node_attributes.get(
                "original_purposes",
                []
            )
        }

        return sorted(purpose_codes)

    @staticmethod
    def _add_warning(
        warnings: list[str],
        warning_set: set[str],
        message: str
    ) -> None:
        """
        添加去重后的警告信息。
        """

        if message not in warning_set:

            warnings.append(message)

            warning_set.add(message)

    @staticmethod
    def _get_edge_sort_key(
        edge_item: tuple
    ) -> tuple[str, str]:
        """
        生成边的稳定排序键。

        MultiDiGraph.out_edges 返回：

            source_node_id,
            target_node_id,
            edge_key,
            edge_attributes

        阶段一规定 edge_id 同时作为 MultiDiGraph 的 edge key。
        """

        _, target_node_id, edge_key, edge_attributes = edge_item

        return (
            str(edge_attributes.get("edge_id", edge_key)),
            str(target_node_id)
        )

    @staticmethod
    def _get_field_ids(
        field_refs: list[dict[str, Any]]
    ) -> set[str]:
        """
        从 source_field_refs 或 target_field_refs 中
        提取 field_id 集合。

        字段引用原始结构不做修改。
        """

        return {
            field_ref["field_id"]
            for field_ref in field_refs
        }

    # --------------------------------------------------------
    # 2.3 核心接口：get_downstream_nodes
    # --------------------------------------------------------

    def get_downstream_nodes(
        self,
        node_id: str,
        max_depth: int = 10,
        include_paths: bool = True,
        include_field_mapping: bool = True,
        max_nodes: int = 100
    ) -> dict[str, Any]:
        """
        查询指定节点的全部直接和间接下游影响节点。

        这是 MCP 工具 get_downstream_nodes 对应的
        Python Service 方法。

        Args:
            node_id:
                起始节点 ID。

            max_depth:
                最大遍历深度，默认 10，范围 1-20。

            include_paths:
                是否返回完整路径，默认 True。

            include_field_mapping:
                是否执行字段级映射传播，默认 True。

            max_nodes:
                最大返回节点数量，默认 100，范围 1-200。

        Returns:
            dict:
                严格遵守 MCP get_downstream_nodes 输出契约：

                {
                    "is_found": bool,
                    "query_node_id": str,
                    "traversal_direction": str,
                    "traversal_mode": str,
                    "result_count": int,
                    "truncated": bool,
                    "nodes": list,
                    "edges": list,
                    "warnings": list
                }

        Raises:
            ValueError:
                参数值不合法。

            TypeError:
                参数类型不合法。
        """

        # ----------------------------------------------------
        # 第一步：参数校验
        # ----------------------------------------------------

        self._validate_parameters(
            node_id=node_id,
            max_depth=max_depth,
            include_paths=include_paths,
            include_field_mapping=include_field_mapping,
            max_nodes=max_nodes
        )

        # ----------------------------------------------------
        # 第二步：检查起始节点是否存在
        # ----------------------------------------------------

        if not self.graph.has_node(node_id):

            return {
                "is_found": False,
                "query_node_id": node_id,
                "traversal_direction": "downstream",
                "traversal_mode": "transitive",
                "result_count": 0,
                "truncated": False,
                "nodes": [],
                "edges": [],
                "warnings": [
                    f"Node not found: {node_id}"
                ]
            }

        # ----------------------------------------------------
        # 第三步：初始化遍历状态
        # ----------------------------------------------------

        root_attributes = self.graph.nodes[node_id]

        # 如果开启字段映射，初始传播范围为源节点的全部字段。
        #
        # 本工具的输入没有 changed_field_ids 参数，
        # 因此这里按整张源节点的数据资产进行影响分析。
        #
        # 如果关闭字段映射，则不根据字段筛选下游边。
        if include_field_mapping:

            initial_affected_fields = {
                field["field_id"]
                for field in root_attributes.get(
                    "fields",
                    []
                )
            }

        else:

            initial_affected_fields = set()

        # BFS 队列中的每个元素表示一条独立的传播状态：
        #
        # current_node_id:
        #     当前所在节点。
        #
        # current_depth:
        #     从起始节点到当前节点的边数。
        #
        # current_affected_fields:
        #     当前路径到达此节点时，受到影响的字段。
        #
        # path_node_ids:
        #     当前路径已经经过的节点。
        #
        # path_edge_ids:
        #     当前路径已经经过的边。
        #
        # 为了保留多路径和多重边，需要记录完整路径状态，
        # 而不是只维护一个全局 visited_nodes。
        queue = deque([
            (
                node_id,
                0,
                initial_affected_fields,
                (node_id,),
                ()
            )
        ])

        # 记录最终下游节点。
        #
        # key:
        #     下游 node_id。
        #
        # value:
        #     MCP 输出中对应的节点信息。
        node_records: dict[str, dict[str, Any]] = {}

        # 记录所有实际参与下游传播的边。
        #
        # 使用 edge_id 去重，防止同一条边经过不同状态时重复输出。
        edge_records: dict[str, dict[str, Any]] = {}

        # 保存每个节点已记录的唯一路径。
        path_keys_by_node: dict[
            str,
            set[tuple]
        ] = {}

        # 记录警告并去重。
        warnings: list[str] = []

        warning_set: set[str] = set()

        # 是否发生结果截断。
        truncated = False

        # 当前已经处理的路径状态数量。
        processed_path_states = 0

        # ----------------------------------------------------
        # 第四步：BFS 广度优先遍历
        # ----------------------------------------------------

        while queue:

            # 内部路径数量保护。
            if processed_path_states >= MAX_PATH_STATES:

                truncated = True

                self._add_warning(
                    warnings,
                    warning_set,
                    (
                        "Maximum internal path state limit "
                        f"reached: {MAX_PATH_STATES}. "
                        "Some downstream paths may be incomplete."
                    )
                )

                break

            (
                current_node_id,
                current_depth,
                current_affected_fields,
                path_node_ids,
                path_edge_ids
            ) = queue.popleft()

            processed_path_states += 1

            # 获取当前节点的所有出边。
            outgoing_edges = list(
                self.graph.out_edges(
                    current_node_id,
                    keys=True,
                    data=True
                )
            )

            # 排序，保证每次执行得到稳定的输出顺序。
            outgoing_edges.sort(
                key=self._get_edge_sort_key
            )

            # ------------------------------------------------
            # 第五步：逐条检查出边
            # ------------------------------------------------

            for edge_item in outgoing_edges:

                (
                    source_node_id,
                    target_node_id,
                    edge_key,
                    edge_attributes
                ) = edge_item

                edge_id = edge_attributes["edge_id"]

                # --------------------------------------------
                # 5.1 环路检查
                # --------------------------------------------

                # 当前目标节点如果已经出现在同一路径中，
                # 则该边会形成循环路径。
                #
                # 不允许同一路径无限重复经过同一节点。
                if target_node_id in path_node_ids:

                    self._add_warning(
                        warnings,
                        warning_set,
                        (
                            "Cycle-closing edges were ignored "
                            "while enumerating simple paths."
                        )
                    )

                    continue

                # --------------------------------------------
                # 5.2 字段映射检查
                # --------------------------------------------

                if include_field_mapping:

                    source_field_ids = self._get_field_ids(
                        edge_attributes["source_field_refs"]
                    )

                    # 当前传播字段必须与边的输入字段存在交集。
                    #
                    # 如果没有交集，说明当前路径所传播的数据
                    # 不在该处理任务的输入字段范围内。
                    if not (
                        source_field_ids
                        & current_affected_fields
                    ):

                        continue

                # --------------------------------------------
                # 5.3 遍历深度限制
                # --------------------------------------------

                if current_depth >= max_depth:

                    truncated = True

                    self._add_warning(
                        warnings,
                        warning_set,
                        (
                            f"Maximum traversal depth reached: "
                            f"{max_depth}. Some downstream nodes "
                            "were not explored."
                        )
                    )

                    continue

                new_depth = current_depth + 1

                # --------------------------------------------
                # 5.4 最大节点数量限制
                # --------------------------------------------

                is_new_node = (
                    target_node_id not in node_records
                )

                if (
                    is_new_node
                    and len(node_records) >= max_nodes
                ):

                    truncated = True

                    self._add_warning(
                        warnings,
                        warning_set,
                        (
                            f"Maximum result node limit reached: "
                            f"{max_nodes}. Additional nodes "
                            "were omitted."
                        )
                    )

                    continue

                # --------------------------------------------
                # 5.5 获取目标节点属性
                # --------------------------------------------

                target_attributes = self.graph.nodes[
                    target_node_id
                ]

                # --------------------------------------------
                # 5.6 获取目标节点的受影响字段
                # --------------------------------------------

                if include_field_mapping:

                    target_field_ids = self._get_field_ids(
                        edge_attributes["target_field_refs"]
                    )

                else:

                    target_field_ids = set()

                # --------------------------------------------
                # 5.7 创建或更新下游节点记录
                # --------------------------------------------

                if is_new_node:

                    node_records[target_node_id] = {
                        "node_id": target_node_id,
                        "name": target_attributes["name"],
                        "node_type": target_attributes["node_type"],
                        "sensitivity_labels": sorted(
                            target_attributes["sensitivity_labels"]
                        ),
                        "original_purposes": (
                            self._get_purpose_codes(
                                target_attributes
                            )
                        ),
                        "min_depth": new_depth,
                        "affected_fields": set(),
                        "paths": []
                    }

                    path_keys_by_node[target_node_id] = set()

                node_record = node_records[
                    target_node_id
                ]

                # 保留最短路径深度。
                node_record["min_depth"] = min(
                    node_record["min_depth"],
                    new_depth
                )

                # 合并不同路径传递到该节点的受影响字段。
                node_record["affected_fields"].update(
                    target_field_ids
                )

                # --------------------------------------------
                # 5.8 记录完整路径
                # --------------------------------------------

                new_path_node_ids = (
                    path_node_ids + (target_node_id,)
                )

                new_path_edge_ids = (
                    path_edge_ids + (edge_id,)
                )

                if include_paths:

                    path_key = (
                        new_path_node_ids,
                        new_path_edge_ids
                    )

                    if path_key not in path_keys_by_node[
                        target_node_id
                    ]:

                        path_keys_by_node[
                            target_node_id
                        ].add(path_key)

                        node_record["paths"].append({
                            "node_ids": list(
                                new_path_node_ids
                            ),
                            "edge_ids": list(
                                new_path_edge_ids
                            )
                        })

                # --------------------------------------------
                # 5.9 记录实际参与传播的边
                # --------------------------------------------

                if edge_id not in edge_records:

                    edge_records[edge_id] = copy.deepcopy(
                        edge_attributes
                    )

                # --------------------------------------------
                # 5.10 将下一层传播状态加入 BFS 队列
                # --------------------------------------------

                queue.append((
                    target_node_id,
                    new_depth,
                    target_field_ids,
                    new_path_node_ids,
                    new_path_edge_ids
                ))

        # ====================================================
        # 第六步：整理最终节点结果
        # ====================================================

        nodes = []

        # 按最短路径深度、node_id 排序。
        sorted_node_records = sorted(
            node_records.values(),
            key=lambda item: (
                item["min_depth"],
                item["node_id"]
            )
        )

        for node_record in sorted_node_records:

            # 将内部 set 转换为标准 JSON 数组。
            node_record["affected_fields"] = sorted(
                node_record["affected_fields"]
            )

            if include_paths:

                # 先按路径长度排序，再按节点和边 ID 排序。
                node_record["paths"].sort(
                    key=lambda path: (
                        len(path["edge_ids"]),
                        tuple(path["node_ids"]),
                        tuple(path["edge_ids"])
                    )
                )

            else:

                # 即使不返回路径，字段映射仍可正常工作。
                node_record["paths"] = []

            nodes.append(node_record)

        # ====================================================
        # 第七步：整理边结果
        # ====================================================

        edges = [
            copy.deepcopy(edge_records[edge_id])
            for edge_id in sorted(edge_records)
        ]

        # ====================================================
        # 第八步：返回 MCP Tool Contract
        # ====================================================

        return {
            "is_found": True,
            "query_node_id": node_id,
            "traversal_direction": "downstream",
            "traversal_mode": "transitive",
            "result_count": len(nodes),
            "truncated": truncated,
            "nodes": nodes,
            "edges": edges,
            "warnings": warnings
        }


# ============================================================
# 3. 命令行测试入口
# ============================================================

def main() -> int:
    """
    加载模拟数据并执行下游血缘查询。

    该入口用于阶段二的独立测试。
    """

    parser = argparse.ArgumentParser(
        description="Data Lineage Downstream Query"
    )

    parser.add_argument(
        "--node-id",
        type=str,
        default="T01_CUSTOMER_RAW",
        help="起始节点 ID"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="最大遍历深度，范围 1-20"
    )

    parser.add_argument(
        "--max-nodes",
        type=int,
        default=100,
        help="最大返回节点数量，范围 1-200"
    )

    parser.add_argument(
        "--no-paths",
        action="store_true",
        help="关闭完整路径输出"
    )

    parser.add_argument(
        "--no-field-mapping",
        action="store_true",
        help="关闭字段级传播筛选"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("数据血缘下游影响查询")
    print("=" * 65)

    try:

        # 使用阶段一的图加载器。
        graph = load_graph()

        # 初始化查询 Service。
        service = LineageQueryService(graph)

        # 执行下游查询。
        result = service.get_downstream_nodes(
            node_id=args.node_id,
            max_depth=args.max_depth,
            include_paths=not args.no_paths,
            include_field_mapping=(
                not args.no_field_mapping
            ),
            max_nodes=args.max_nodes
        )

        print("\n查询结果：\n")

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

        print("\n查询统计：")
        print(f"  起始节点：{result['query_node_id']}")
        print(f"  下游节点数量：{result['result_count']}")
        print(f"  实际边数量：{len(result['edges'])}")
        print(f"  是否截断：{result['truncated']}")
        print(f"  警告数量：{len(result['warnings'])}")

        if not result["is_found"]:

            return 1

        print("\n下游血缘查询完成。")

        return 0

    except (
        FileNotFoundError,
        ValueError,
        TypeError
    ) as exc:

        print(f"\n[失败] {exc}")

        return 1

    except Exception as exc:

        print(
            f"\n[失败] 未预期异常："
            f"{type(exc).__name__}: {exc}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
