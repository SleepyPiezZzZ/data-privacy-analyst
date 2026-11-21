
"""
graph_validator.py

数据血缘图校验模块。

包含：
1. JSON 文件读取。
2. JSON Schema Draft 2020-12 校验。
3. 节点、字段、边的唯一性校验。
4. 跨节点字段引用校验。
5. 敏感度标签及 Purpose 一致性校验。

运行方式（项目根目录）：
    python -m graph.graph_validator

指定文件：
    python -m graph.graph_validator --data data/financial_lineage_demo.json

    python -m graph.graph_validator \
        --data data/financial_lineage_demo.json \
        --schema schemas/data_lineage_graph.schema.json
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


# ============================================================
# 1. 项目路径配置
# ============================================================

# 当前文件位于 project_root/graph/graph_validator.py
# parents[1] 对应项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "financial_lineage_demo.json"
)

DEFAULT_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "data_lineage_graph.schema.json"
)


# ============================================================
# 2. JSON 文件读取
# ============================================================

def load_json_file(file_path: str | Path) -> dict[str, Any]:
    """
    读取 JSON 文件并返回字典。

    Args:
        file_path:
            JSON 文件路径。

    Returns:
        dict:
            JSON 根对象。

    Raises:
        FileNotFoundError:
            文件不存在。

        ValueError:
            JSON 格式错误，或者根对象不是 JSON Object。
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON 文件不存在：{path}"
        )

    try:
        with path.open(
            mode="r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 文件格式错误：{path}\n"
            f"错误位置：第 {exc.lineno} 行，第 {exc.colno} 列\n"
            f"错误信息：{exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON 文件根对象必须是 Object：{path}"
        )

    return data


# ============================================================
# 3. 辅助函数
# ============================================================

def _format_schema_error(error: Any) -> str:
    """
    将 JSON Schema 错误转换成方便阅读的路径格式。

    例如：
        $.nodes[0].fields[2].sensitivity_label

    Args:
        error:
            jsonschema.ValidationError 对象。

    Returns:
        str:
            格式化后的错误描述。
    """

    location = "$"

    for part in error.absolute_path:
        if isinstance(part, int):
            location += f"[{part}]"
        else:
            location += f".{part}"

    return (
        f"[JSON Schema] {location}: {error.message}"
    )


def _check_unique_values(
    values: list[Any],
    object_name: str,
    error_list: list[str]
) -> None:
    """
    检查指定列表中的字符串是否重复。
    """

    seen: set[Any] = set()

    for value in values:
        if value in seen:
            error_list.append(
                f"[唯一性错误] {object_name} 重复：{value}"
            )
        else:
            seen.add(value)


# ============================================================
# 4. 图结构语义校验
# ============================================================

def _validate_graph_semantics(
    graph_data: dict[str, Any]
) -> list[str]:
    """
    执行 JSON Schema 无法完整表达的业务关系校验。

    检查内容：

    1. Node ID 唯一性。
    2. Edge ID 唯一性。
    3. 字段 ID 在节点内唯一。
    4. 节点敏感度标签覆盖全部字段标签。
    5. 字段 Purpose 属于节点已登记 Purpose。
    6. 表节点只能使用 column 字段角色。
    7. API 节点只能使用 request / response 字段角色。
    8. Edge 两端节点必须存在。
    9. Edge 字段引用必须指向正确节点。
    10. Edge 引用字段必须真实存在。
    11. Edge 处理目的必须得到源节点和目标节点支持。
    12. 不允许边连接同一个节点。

    Args:
        graph_data:
            已经通过 JSON Schema 验证的图数据。

    Returns:
        list[str]:
            所有语义校验错误。空列表表示通过。
    """

    errors: list[str] = []

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    # --------------------------------------------------------
    # 4.1 节点 ID 唯一性
    # --------------------------------------------------------

    node_ids = [
        node["node_id"]
        for node in nodes
    ]

    _check_unique_values(
        node_ids,
        "node_id",
        errors
    )

    # 保存节点 ID 到节点数据的映射。
    # 后续 Edge 可以通过该字典快速定位节点。
    node_map = {
        node["node_id"]: node
        for node in nodes
    }

    # 保存每个节点内部的字段 ID。
    field_ids_by_node: dict[str, set[str]] = {}

    # --------------------------------------------------------
    # 4.2 节点内部字段及元数据校验
    # --------------------------------------------------------

    for node in nodes:

        node_id = node["node_id"]
        asset_kind = node["asset_kind"]

        fields = node["fields"]

        # 字段 ID 在单个节点内必须唯一。
        field_ids = [
            field["field_id"]
            for field in fields
        ]

        _check_unique_values(
            field_ids,
            f"{node_id}.field_id",
            errors
        )

        field_ids_by_node[node_id] = set(field_ids)

        # 获取节点级敏感度标签。
        node_sensitivity_labels = set(
            node["sensitivity_labels"]
        )

        # 获取节点登记的 Purpose。
        original_purposes = {
            purpose["purpose_code"]
            for purpose in node["original_purposes"]
        }

        # 检查字段敏感度是否被节点级标签覆盖。
        for field in fields:

            field_id = field["field_id"]

            sensitivity_label = (
                field["sensitivity_label"]
            )

            if sensitivity_label not in node_sensitivity_labels:

                errors.append(
                    "[敏感度不一致] "
                    f"节点 {node_id} 未声明字段 "
                    f"{field_id} 的敏感度标签 "
                    f"{sensitivity_label}"
                )

            # 字段 Purpose 必须在节点登记的 Purpose 中。
            field_purposes = set(
                field["purpose_codes"]
            )

            unknown_purposes = (
                field_purposes - original_purposes
            )

            if unknown_purposes:

                errors.append(
                    "[Purpose 不一致] "
                    f"节点 {node_id} 的字段 {field_id} "
                    f"引用了未登记的用途："
                    f"{sorted(unknown_purposes)}"
                )

            # 表节点只能包含 column 类型字段。
            if (
                asset_kind == "table"
                and field["field_role"] != "column"
            ):

                errors.append(
                    "[字段角色错误] "
                    f"表节点 {node_id} 的字段 "
                    f"{field_id} 必须使用 column"
                )

            # API 节点必须是 request 或 response。
            if (
                asset_kind == "api"
                and field["field_role"] not in (
                    "request",
                    "response"
                )
            ):

                errors.append(
                    "[字段角色错误] "
                    f"API 节点 {node_id} 的字段 "
                    f"{field_id} 必须使用 request 或 response"
                )

    # --------------------------------------------------------
    # 4.3 Edge ID 唯一性
    # --------------------------------------------------------

    edge_ids = [
        edge["edge_id"]
        for edge in edges
    ]

    _check_unique_values(
        edge_ids,
        "edge_id",
        errors
    )

    # --------------------------------------------------------
    # 4.4 Edge 和字段引用校验
    # --------------------------------------------------------

    for edge in edges:

        edge_id = edge["edge_id"]

        source_node_id = edge["source_node_id"]

        target_node_id = edge["target_node_id"]

        # 检查是否存在自环。
        if source_node_id == target_node_id:

            errors.append(
                "[数据流向错误] "
                f"边 {edge_id} 的起点和终点不能相同："
                f"{source_node_id}"
            )

        # 检查源节点是否存在。
        if source_node_id not in node_map:

            errors.append(
                "[节点引用错误] "
                f"边 {edge_id} 引用了不存在的源节点："
                f"{source_node_id}"
            )

        # 检查目标节点是否存在。
        if target_node_id not in node_map:

            errors.append(
                "[节点引用错误] "
                f"边 {edge_id} 引用了不存在的目标节点："
                f"{target_node_id}"
            )

        # 如果源节点或目标节点不存在，则无法继续
        # 对该条边进行完整的字段和 Purpose 检查。
        if (
            source_node_id not in node_map
            or target_node_id not in node_map
        ):
            continue

        # ----------------------------------------------------
        # 检查源字段引用
        # ----------------------------------------------------

        for field_ref in edge["source_field_refs"]:

            ref_node_id = field_ref["node_id"]
            ref_field_id = field_ref["field_id"]

            # 源字段引用必须属于 source_node_id。
            if ref_node_id != source_node_id:

                errors.append(
                    "[源字段引用错误] "
                    f"边 {edge_id} 的 source_field_refs "
                    f"引用节点 {ref_node_id}，"
                    f"但实际源节点是 {source_node_id}"
                )

                continue

            # 检查字段是否存在。
            if (
                ref_field_id
                not in field_ids_by_node.get(
                    source_node_id,
                    set()
                )
            ):

                errors.append(
                    "[字段引用错误] "
                    f"边 {edge_id} 的源节点 "
                    f"{source_node_id} 不存在字段 "
                    f"{ref_field_id}"
                )

        # ----------------------------------------------------
        # 检查目标字段引用
        # ----------------------------------------------------

        for field_ref in edge["target_field_refs"]:

            ref_node_id = field_ref["node_id"]
            ref_field_id = field_ref["field_id"]

            # 目标字段引用必须属于 target_node_id。
            if ref_node_id != target_node_id:

                errors.append(
                    "[目标字段引用错误] "
                    f"边 {edge_id} 的 target_field_refs "
                    f"引用节点 {ref_node_id}，"
                    f"但实际目标节点是 {target_node_id}"
                )

                continue

            # 检查字段是否存在。
            if (
                ref_field_id
                not in field_ids_by_node.get(
                    target_node_id,
                    set()
                )
            ):

                errors.append(
                    "[字段引用错误] "
                    f"目标节点 {target_node_id} "
                    f"不存在字段 {ref_field_id}"
                )

        # ----------------------------------------------------
        # 检查处理用途
        # ----------------------------------------------------

        source_purposes = {
            purpose["purpose_code"]
            for purpose in node_map[
                source_node_id
            ]["original_purposes"]
        }

        target_purposes = {
            purpose["purpose_code"]
            for purpose in node_map[
                target_node_id
            ]["original_purposes"]
        }

        processing_purposes = set(
            edge["processing_purpose_codes"]
        )

        # 处理用途必须同时被源节点和目标节点登记。
        unsupported_purposes = (
            processing_purposes
            - (source_purposes & target_purposes)
        )

        if unsupported_purposes:

            errors.append(
                "[Edge Purpose 不一致] "
                f"边 {edge_id} 包含未被源节点或目标节点共同登记的用途："
                f"{sorted(unsupported_purposes)}"
            )

    return errors


# ============================================================
# 5. 对外公开的验证接口
# ============================================================

def validate_graph_data(
    graph_data: dict[str, Any],
    schema_path: str | Path | None = None
) -> list[str]:
    """
    对数据血缘图执行完整校验。

    校验顺序：

    1. 加载 JSON Schema。
    2. 校验 Schema 自身是否合法。
    3. 使用 Draft202012Validator 校验 JSON。
    4. 执行跨节点语义校验。

    Args:
        graph_data:
            待校验的 JSON 字典。

        schema_path:
            JSON Schema 路径。
            默认使用项目 schemas 目录中的定义。

    Returns:
        list[str]:
            错误列表。
            空列表代表校验成功。
    """

    if not isinstance(graph_data, dict):
        return [
            "[类型错误] graph_data 必须是 Python dict"
        ]

    schema_file = (
        Path(schema_path)
        if schema_path is not None
        else DEFAULT_SCHEMA_PATH
    )

    # 读取 Schema。
    schema = load_json_file(schema_file)

    # 先检查 Schema 自身是否合法。
    Draft202012Validator.check_schema(schema)

    # 显式启用格式检查。
    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker()
    )

    # --------------------------------------------------------
    # 第一层：JSON Schema Validation
    # --------------------------------------------------------

    schema_errors = list(
        validator.iter_errors(graph_data)
    )

    schema_errors.sort(
        key=lambda error: (
            list(map(str, error.absolute_path)),
            error.message
        )
    )

    if schema_errors:

        return [
            _format_schema_error(error)
            for error in schema_errors
        ]

    # --------------------------------------------------------
    # 第二层：Graph Semantic Validation
    # --------------------------------------------------------

    semantic_errors = _validate_graph_semantics(
        graph_data
    )

    return semantic_errors


# ============================================================
# 6. 文件级校验接口
# ============================================================

def validate_graph_file(
    data_path: str | Path,
    schema_path: str | Path | None = None
) -> list[str]:
    """
    从磁盘加载图 JSON 并执行完整校验。

    Args:
        data_path:
            数据文件路径。

        schema_path:
            Schema 文件路径。

    Returns:
        list[str]:
            校验错误列表。
    """

    graph_data = load_json_file(data_path)

    return validate_graph_data(
        graph_data=graph_data,
        schema_path=schema_path
    )


# ============================================================
# 7. 命令行入口
# ============================================================

def main() -> int:
    """
    命令行校验入口。

    返回：
        0：验证成功。
        1：验证失败。
    """

    parser = argparse.ArgumentParser(
        description="Financial Data Lineage Graph Validator"
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

    data_path = Path(args.data)
    schema_path = Path(args.schema)

    print("=" * 65)
    print("数据血缘图校验")
    print("=" * 65)

    print(f"数据文件：{data_path}")
    print(f"Schema：{schema_path}")

    try:

        graph_data = load_json_file(data_path)

        errors = validate_graph_data(
            graph_data,
            schema_path
        )

    except (
        FileNotFoundError,
        ValueError,
        OSError
    ) as exc:

        print("\n[失败] 文件读取错误：")
        print(exc)

        return 1

    except Exception as exc:

        print("\n[失败] 校验过程出现异常：")
        print(f"{type(exc).__name__}: {exc}")

        return 1

    print("-" * 65)

    if errors:

        print(f"[失败] 检测到 {len(errors)} 个问题：\n")

        for index, error in enumerate(
            errors,
            start=1
        ):
            print(f"{index}. {error}")

        return 1

    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    table_count = sum(
        node["asset_kind"] == "table"
        for node in nodes
    )

    api_count = sum(
        node["asset_kind"] == "api"
        for node in nodes
    )

    field_count = sum(
        len(node["fields"])
        for node in nodes
    )

    print("[成功] JSON Schema 校验通过。")
    print("[成功] Graph Semantic Validation 通过。")

    print("\n数据统计：")
    print(f"  节点总数：{len(nodes)}")
    print(f"  数据表数量：{table_count}")
    print(f"  API 节点数量：{api_count}")
    print(f"  字段总数：{field_count}")
    print(f"  边总数：{len(edges)}")

    print("\n阶段一数据校验完成。")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
