
"""
graph_loader.py

数据血缘图加载模块。

功能：
1. 读取 financial_lineage_demo.json。
2. 调用 graph_validator.py 进行数据校验。
3. 将 nodes 转换为 NetworkX 节点。
4. 将 edges 转换为 NetworkX 有向多重边。
5. 保留原始字段和元数据。
6. 提供命令行加载及统计功能。

运行方式（项目根目录）：
    python -m graph.graph_loader

默认返回：
    networkx.MultiDiGraph
"""

from __future__ import annotations

import argparse
import copy
import sys

from pathlib import Path
from typing import Any

import networkx as nx

from graph.graph_validator import (
    DEFAULT_DATA_PATH,
    DEFAULT_SCHEMA_PATH,
    load_json_file,
    validate_graph_data
)


# ============================================================
# 1. 自定义异常
# ============================================================

class GraphValidationError(Exception):
    """
    图数据验证失败异常。

    当 JSON Schema 或 Graph Semantic Validation
    未通过时，禁止将数据加载到 NetworkX。
    """

    def __init__(
        self,
        errors: list[str]
    ) -> None:

        self.errors = errors

        message = (
            "数据血缘图校验失败，无法创建 NetworkX 图：\n"
            + "\n".join(errors)
        )

        super().__init__(message)


# ============================================================
# 2. JSON 数据获取
# ============================================================

def _resolve_graph_data(
    source: dict[str, Any] | str | Path
) -> dict[str, Any]:
    """
    统一处理图数据来源。

    支持：
    1. 直接传入 Python dict。
    2. 传入 JSON 文件路径。

    Args:
        source:
            JSON 字典或文件路径。

    Returns:
        dict:
            图数据。
    """

    if isinstance(source, dict):

        # 深拷贝，避免后续图操作意外修改原始输入。
        return copy.deepcopy(source)

    if isinstance(source, (str, Path)):

        return load_json_file(source)

    raise TypeError(
        "source 必须是 dict、str 或 pathlib.Path"
    )


# ============================================================
# 3. 创建 NetworkX 图
# ============================================================

def load_graph(
    source: dict[str, Any] | str | Path = DEFAULT_DATA_PATH,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    validate: bool = True
) -> nx.MultiDiGraph:
    """
    加载数据血缘图并创建 NetworkX MultiDiGraph。

    Args:
        source:
            图数据字典，或者 JSON 文件路径。

        schema_path:
            JSON Schema 文件路径。

        validate:
            是否执行完整数据校验。
            默认 True，生产和正式分析场景不建议关闭。

    Returns:
        nx.MultiDiGraph:
            构建完成的数据血缘图。

    Raises:
        GraphValidationError:
            当图数据未通过校验时抛出。

        FileNotFoundError:
            当 JSON 文件不存在时抛出。

        ValueError:
            当 JSON 文件格式错误时抛出。
    """

    # --------------------------------------------------------
    # 第一步：取得原始 JSON 数据
    # --------------------------------------------------------

    graph_data = _resolve_graph_data(source)

    # --------------------------------------------------------
    # 第二步：执行 Graph Validation
    # --------------------------------------------------------

    if validate:

        validation_errors = validate_graph_data(
            graph_data=graph_data,
            schema_path=schema_path
        )

        if validation_errors:

            raise GraphValidationError(
                validation_errors
            )

    # --------------------------------------------------------
    # 第三步：创建 NetworkX MultiDiGraph
    # --------------------------------------------------------

    graph = nx.MultiDiGraph()

    # 将整个图的元信息放入 graph.graph。
    graph.graph.update({
        "schema_version": graph_data["schema_version"],
        "graph_id": graph_data["graph_id"],
        "platform": graph_data["platform"],
        "snapshot_at": graph_data["snapshot_at"],
        "classification_policy_version": (
            graph_data["classification_policy_version"]
        )
    })

    # --------------------------------------------------------
    # 第四步：加载节点
    # --------------------------------------------------------

    for node in graph_data["nodes"]:

        node_id = node["node_id"]

        # 保留节点所有原始数据。
        # 包含：
        # name、asset_kind、node_type、description、
        # sensitivity_labels、original_purposes、fields，
        # 以及 table/API 对应的其他属性。
        node_attributes = copy.deepcopy(node)

        graph.add_node(
            node_id,
            **node_attributes
        )

    # --------------------------------------------------------
    # 第五步：加载边
    # --------------------------------------------------------

    for edge in graph_data["edges"]:

        source_node_id = edge["source_node_id"]

        target_node_id = edge["target_node_id"]

        edge_id = edge["edge_id"]

        edge_attributes = copy.deepcopy(edge)

        # 将 edge_id 直接作为 MultiDiGraph 的边 key。
        #
        # 这样可以确保：
        # 1. 每条边都有稳定的唯一标识。
        # 2. 同一个源节点和目标节点之间可以存在多条边。
        # 3. 后续可以直接通过 edge_id 查询具体任务。
        graph.add_edge(
            source_node_id,
            target_node_id,
            key=edge_id,
            **edge_attributes
        )

    # --------------------------------------------------------
    # 第六步：记录加载统计信息
    # --------------------------------------------------------

    graph.graph["node_count"] = graph.number_of_nodes()

    graph.graph["edge_count"] = graph.number_of_edges()

    graph.graph["field_count"] = sum(
        len(node["fields"])
        for node in graph_data["nodes"]
    )

    return graph


# ============================================================
# 4. 图统计信息
# ============================================================

def get_graph_statistics(
    graph: nx.MultiDiGraph
) -> dict[str, Any]:
    """
    获取 NetworkX 图的基本统计信息。

    Args:
        graph:
            已加载的 MultiDiGraph。

    Returns:
        dict:
            图结构统计数据。
    """

    node_count = graph.number_of_nodes()

    edge_count = graph.number_of_edges()

    table_count = 0
    api_count = 0
    field_count = 0

    node_type_counts: dict[str, int] = {}

    sensitivity_counts: dict[str, int] = {}

    # 统计节点、字段和敏感度。
    for node_id, attributes in graph.nodes(data=True):

        asset_kind = attributes["asset_kind"]

        node_type = attributes["node_type"]

        if asset_kind == "table":
            table_count += 1

        elif asset_kind == "api":
            api_count += 1

        node_type_counts[node_type] = (
            node_type_counts.get(node_type, 0) + 1
        )

        fields = attributes["fields"]

        field_count += len(fields)

        # 统计字段敏感度类别的出现次数。
        for field in fields:

            label = field["sensitivity_label"]

            sensitivity_counts[label] = (
                sensitivity_counts.get(label, 0) + 1
            )

    # 统计节点入度和出度。
    indegree = dict(graph.in_degree())

    outdegree = dict(graph.out_degree())

    # 找到没有上游输入的起始节点。
    source_nodes = [
        node_id
        for node_id in graph.nodes
        if indegree[node_id] == 0
    ]

    # 找到没有下游输出的终端节点。
    sink_nodes = [
        node_id
        for node_id in graph.nodes
        if outdegree[node_id] == 0
    ]

    return {
        "graph_id": graph.graph.get("graph_id"),
        "platform": graph.graph.get("platform"),
        "node_count": node_count,
        "edge_count": edge_count,
        "table_count": table_count,
        "api_count": api_count,
        "field_count": field_count,
        "node_type_counts": node_type_counts,
        "sensitivity_counts": sensitivity_counts,
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
        "is_directed": graph.is_directed(),
        "is_multigraph": graph.is_multigraph()
    }


# ============================================================
# 5. 命令行入口
# ============================================================

def main() -> int:
    """
    加载图文件并打印统计信息。

    返回：
        0：加载成功。
        1：加载失败。
    """

    parser = argparse.ArgumentParser(
        description="Financial Data Lineage Graph Loader"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="数据血缘 JSON 文件路径"
    )

    parser.add_argument(
        "--schema",
        type=str,
        default=str(DEFAULT_SCHEMA_PATH),
        help="Graph JSON Schema 文件路径"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("NetworkX 数据血缘图加载器")
    print("=" * 65)

    print(f"数据文件：{args.data}")
    print(f"Schema：{args.schema}")

    try:

        graph = load_graph(
            source=args.data,
            schema_path=args.schema,
            validate=True
        )

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        GraphValidationError
    ) as exc:

        print("\n[失败] 图数据加载失败：")
        print(exc)

        return 1

    except Exception as exc:

        print("\n[失败] 加载过程中发生未预期错误：")
        print(f"{type(exc).__name__}: {exc}")

        return 1

    statistics = get_graph_statistics(graph)

    print("\n[成功] NetworkX 图加载完成。")

    print("\n图的基本信息：")

    print(
        f"  Graph ID：{statistics['graph_id']}"
    )

    print(
        f"  Platform：{statistics['platform']}"
    )

    print(
        f"  节点数量：{statistics['node_count']}"
    )

    print(
        f"  数据表数量：{statistics['table_count']}"
    )

    print(
        f"  API 数量：{statistics['api_count']}"
    )

    print(
        f"  边数量：{statistics['edge_count']}"
    )

    print(
        f"  字段数量：{statistics['field_count']}"
    )

    print("\n节点类型分布：")

    for node_type, count in sorted(
        statistics["node_type_counts"].items()
    ):

        print(f"  {node_type}: {count}")

    print("\n字段敏感度分布：")

    for label, count in sorted(
        statistics["sensitivity_counts"].items()
    ):

        print(f"  {label}: {count}")

    print("\n图的拓扑信息：")

    print(
        f"  是否为有向图：{statistics['is_directed']}"
    )

    print(
        f"  是否为多重图：{statistics['is_multigraph']}"
    )

    print(
        f"  起始节点：{statistics['source_nodes']}"
    )

    print(
        f"  终止节点：{statistics['sink_nodes']}"
    )

    print("\n阶段一 NetworkX 加载完成。")

    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
