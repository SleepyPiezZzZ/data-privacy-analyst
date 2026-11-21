
"""
node_metadata.py

数据资产节点元数据查询模块。

核心功能：
1. 根据 node_id 查询节点。
2. 获取节点敏感度标签。
3. 获取节点原始使用目的。
4. 获取字段级详细元数据。
5. 获取直接上游节点和下游节点。
6. 获取与当前节点直接关联的 Edge ID。
7. 支持根据参数控制字段和关联关系的返回。

严格遵守 MCP get_node_details 工具输出契约。

依赖：
    networkx
    graph.graph_loader

运行方式（项目根目录）：

    python -m graph.node_metadata

指定节点：

    python -m graph.node_metadata --node-id T01_CUSTOMER_RAW

不返回字段详情：

    python -m graph.node_metadata \
        --node-id T01_CUSTOMER_RAW \
        --no-fields

不返回上下游关系：

    python -m graph.node_metadata \
        --node-id T01_CUSTOMER_RAW \
        --no-related-edges
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

from typing import Any

import networkx as nx

from graph.graph_loader import load_graph


# ============================================================
# 1. 节点元数据查询服务
# ============================================================

class NodeMetadataService:
    """
    数据资产节点元数据查询服务。

    初始化时注入 NetworkX MultiDiGraph。

    MCP Server 后续可以直接调用：

        metadata_service.get_node_details(
            node_id="T01_CUSTOMER_RAW"
        )

    本服务只负责单节点元数据和直接相邻关系，
    不负责递归查询整个下游数据链路。
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph
    ) -> None:
        """
        初始化节点元数据查询服务。

        Args:
            graph:
                阶段一 graph_loader 加载的 MultiDiGraph。

        Raises:
            TypeError:
                如果传入对象不是 MultiDiGraph。
        """

        if not isinstance(graph, nx.MultiDiGraph):

            raise TypeError(
                "graph 必须是 networkx.MultiDiGraph，"
                f"实际类型为 {type(graph).__name__}"
            )

        self.graph = graph

    # --------------------------------------------------------
    # 1.1 参数校验
    # --------------------------------------------------------

    @staticmethod
    def _validate_parameters(
        node_id: str,
        include_fields: bool,
        include_related_edges: bool
    ) -> None:
        """
        校验 get_node_details 的输入参数。
        """

        if not isinstance(node_id, str) or not node_id.strip():

            raise ValueError(
                "node_id 必须是非空字符串"
            )

        if type(include_fields) is not bool:

            raise TypeError(
                "include_fields 必须是布尔值"
            )

        if type(include_related_edges) is not bool:

            raise TypeError(
                "include_related_edges 必须是布尔值"
            )

    # --------------------------------------------------------
    # 1.2 辅助方法
    # --------------------------------------------------------

    @staticmethod
    def _get_edge_sort_key(
        edge_item: tuple
    ) -> tuple[str, str, str]:
        """
        生成 Edge 的稳定排序键。

        适用于 MultiDiGraph 的 incoming edges
        和 outgoing edges。

        Edge ID 是跨模块统一使用的边标识。
        """

        source_node_id, target_node_id, edge_key, edge_attributes = (
            edge_item
        )

        edge_id = edge_attributes.get(
            "edge_id",
            edge_key
        )

        return (
            str(edge_id),
            str(source_node_id),
            str(target_node_id)
        )

    @staticmethod
    def _get_lineage_summary(
        graph: nx.MultiDiGraph,
        node_id: str
    ) -> dict[str, list[str]]:
        """
        获取节点的直接上下游关系。

        注意：
        这里只返回直接相邻节点，不进行递归查询。

        同一个相邻节点可能存在多条 Edge，
        因此节点 ID 使用集合去重，
        Edge ID 则全部保留。
        """

        # ----------------------------------------------------
        # 获取直接入边
        # ----------------------------------------------------

        incoming_edges = list(
            graph.in_edges(
                node_id,
                keys=True,
                data=True
            )
        )

        incoming_edges.sort(
            key=NodeMetadataService._get_edge_sort_key
        )

        # ----------------------------------------------------
        # 获取直接出边
        # ----------------------------------------------------

        outgoing_edges = list(
            graph.out_edges(
                node_id,
                keys=True,
                data=True
            )
        )

        outgoing_edges.sort(
            key=NodeMetadataService._get_edge_sort_key
        )

        # ----------------------------------------------------
        # 提取上游节点
        # ----------------------------------------------------

        upstream_node_ids = sorted({
            source_node_id
            for source_node_id, _, _, _
            in incoming_edges
        })

        # ----------------------------------------------------
        # 提取下游节点
        # ----------------------------------------------------

        downstream_node_ids = sorted({
            target_node_id
            for _, target_node_id, _, _
            in outgoing_edges
        })

        # ----------------------------------------------------
        # 提取入边 ID
        # ----------------------------------------------------

        incoming_edge_ids = sorted({
            edge_attributes["edge_id"]
            for _, _, _, edge_attributes
            in incoming_edges
        })

        # ----------------------------------------------------
        # 提取出边 ID
        # ----------------------------------------------------

        outgoing_edge_ids = sorted({
            edge_attributes["edge_id"]
            for _, _, _, edge_attributes
            in outgoing_edges
        })

        return {
            "upstream_node_ids": upstream_node_ids,
            "downstream_node_ids": downstream_node_ids,
            "incoming_edge_ids": incoming_edge_ids,
            "outgoing_edge_ids": outgoing_edge_ids
        }

    # --------------------------------------------------------
    # 1.3 核心接口：get_node_details
    # --------------------------------------------------------

    def get_node_details(
        self,
        node_id: str,
        include_fields: bool = True,
        include_related_edges: bool = True
    ) -> dict[str, Any]:
        """
        获取指定节点的详细元数据。

        这是 MCP get_node_details 工具对应的
        Python Service 方法。

        Args:
            node_id:
                目标数据资产节点 ID。

            include_fields:
                是否返回字段级元数据，默认 True。

            include_related_edges:
                是否返回直接上下游关系，默认 True。

        Returns:
            dict:
                严格遵守 MCP get_node_details 输出契约：

                {
                    "is_found": bool,
                    "node": object | None,
                    "lineage_summary": {
                        "upstream_node_ids": [],
                        "downstream_node_ids": [],
                        "incoming_edge_ids": [],
                        "outgoing_edge_ids": []
                    },
                    "warnings": []
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
            include_fields=include_fields,
            include_related_edges=include_related_edges
        )

        # ----------------------------------------------------
        # 第二步：检查节点是否存在
        # ----------------------------------------------------

        if not self.graph.has_node(node_id):

            return {
                "is_found": False,
                "node": None,
                "lineage_summary": {
                    "upstream_node_ids": [],
                    "downstream_node_ids": [],
                    "incoming_edge_ids": [],
                    "outgoing_edge_ids": []
                },
                "warnings": [
                    f"Node not found: {node_id}"
                ]
            }

        # ----------------------------------------------------
        # 第三步：读取节点属性
        # ----------------------------------------------------

        node_attributes = self.graph.nodes[node_id]

        # 使用深拷贝，避免修改原始 NetworkX 节点属性。
        node_details = copy.deepcopy(
            node_attributes
        )

        warnings: list[str] = []

        # ----------------------------------------------------
        # 第四步：根据 include_fields 决定是否返回字段
        # ----------------------------------------------------

        if not include_fields:

            # 保留 fields 键，但返回空数组。
            #
            # 这样返回结构保持稳定，
            # 前端和 LLM 不需要额外判断字段是否存在。
            node_details["fields"] = []

            warnings.append(
                "Field-level metadata was omitted "
                "because include_fields is false."
            )

        # ----------------------------------------------------
        # 第五步：获取直接上下游关系
        # ----------------------------------------------------

        if include_related_edges:

            lineage_summary = (
                self._get_lineage_summary(
                    graph=self.graph,
                    node_id=node_id
                )
            )

        else:

            # 调用者明确要求不返回相邻关系时，
            # 所有数组保持为空。
            lineage_summary = {
                "upstream_node_ids": [],
                "downstream_node_ids": [],
                "incoming_edge_ids": [],
                "outgoing_edge_ids": []
            }

            warnings.append(
                "Related lineage edges were omitted "
                "because include_related_edges is false."
            )

        # ----------------------------------------------------
        # 第六步：返回 MCP Tool Contract
        # ----------------------------------------------------

        return {
            "is_found": True,
            "node": node_details,
            "lineage_summary": lineage_summary,
            "warnings": warnings
        }


# ============================================================
# 2. 命令行测试入口
# ============================================================

def main() -> int:
    """
    加载模拟图并执行节点元数据查询。

    该入口用于阶段二的独立测试。
    """

    parser = argparse.ArgumentParser(
        description="Data Lineage Node Metadata Query"
    )

    parser.add_argument(
        "--node-id",
        type=str,
        default="T01_CUSTOMER_RAW",
        help="需要查询的节点 ID"
    )

    parser.add_argument(
        "--no-fields",
        action="store_true",
        help="不返回字段级元数据"
    )

    parser.add_argument(
        "--no-related-edges",
        action="store_true",
        help="不返回直接上下游关系"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("数据资产节点元数据查询")
    print("=" * 65)

    try:

        # 加载阶段一的数据血缘图。
        graph = load_graph()

        # 初始化元数据查询服务。
        service = NodeMetadataService(graph)

        # 执行节点元数据查询。
        result = service.get_node_details(
            node_id=args.node_id,
            include_fields=not args.no_fields,
            include_related_edges=(
                not args.no_related_edges
            )
        )

        print("\n查询结果：\n")

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

        if not result["is_found"]:

            print("\n[失败] 未找到指定节点。")

            return 1

        node = result["node"]

        print("\n节点统计：")

        print(
            f"  节点 ID：{node['node_id']}"
        )

        print(
            f"  节点名称：{node['name']}"
        )

        print(
            f"  节点类型：{node['node_type']}"
        )

        print(
            f"  资产类型：{node['asset_kind']}"
        )

        print(
            f"  敏感度标签：{len(node['sensitivity_labels'])}"
        )

        print(
            f"  登记用途数量：{len(node['original_purposes'])}"
        )

        print(
            f"  返回字段数量：{len(node['fields'])}"
        )

        print("\n血缘关系统计：")

        print(
            "  直接上游节点："
            f"{len(result['lineage_summary']['upstream_node_ids'])}"
        )

        print(
            "  直接下游节点："
            f"{len(result['lineage_summary']['downstream_node_ids'])}"
        )

        print(
            "  入边数量："
            f"{len(result['lineage_summary']['incoming_edge_ids'])}"
        )

        print(
            "  出边数量："
            f"{len(result['lineage_summary']['outgoing_edge_ids'])}"
        )

        print("\n节点元数据查询完成。")

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
