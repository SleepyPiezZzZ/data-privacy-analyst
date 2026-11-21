
"""
output_validator.py

Privacy Impact Report 输出验证模块。

核心功能：

1. 加载 privacy_impact_report.schema.json。
2. 执行 JSON Schema Draft 2020-12 校验。
3. 构建 Groq Strict Structured Output 兼容 Schema。
4. 根据 changed_field_ids 追踪精确字段传播路径。
5. 从真实 MCP 结果建立可信证据目录。
6. 构建真实的 impacted_nodes。
7. 根据确定性规则计算 risk_level 和 decision。
8. 验证 LLM 输出的节点、字段、政策和证据引用。
9. 验证所有 risk_factors 和 evidence_ids 的关联。

特别说明：

本模块不会让 LLM 自行决定最终风险等级。
风险等级由可信图数据和项目内的确定性规则计算。

运行方式：

    python -m llm.output_validator
"""

from __future__ import annotations

import argparse
import copy
import json
import re

from collections import defaultdict
from pathlib import Path
from typing import Any

from jsonschema import (
    Draft202012Validator,
    FormatChecker
)


# ============================================================
# 1. 项目路径与常量
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPORT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "privacy_impact_report.schema.json"
)

HIGH_SENSITIVITY_LABELS = {
    "high_sensitive_pii",
    "sensitive_financial"
}

SUPPORTED_SENSITIVITY_LABELS = {
    "high_sensitive_pii",
    "sensitive_pii",
    "pseudonymous_pii",
    "sensitive_financial",
    "derived_sensitive",
    "non_sensitive"
}

REQUIRED_LIMITATIONS = [
    (
        "本报告使用 NIST Privacy Framework 1.0 "
        "作为自愿性隐私风险管理参考，不构成特定司法辖区的法律合规结论。"
    ),
    (
        "analysis_confidence 为模型自评值，未经统计校准，"
        "不能解释为风险发生概率。"
    )
]


# ============================================================
# 2. 自定义异常
# ============================================================

class ReportValidationError(ValueError):
    """
    最终报告验证失败。

    errors 属性保存全部验证错误。
    """

    def __init__(
        self,
        errors: list[str]
    ) -> None:

        self.errors = errors

        super().__init__(
            "Privacy Impact Report validation failed:\n"
            + "\n".join(errors)
        )


# ============================================================
# 3. JSON Schema 加载
# ============================================================

def load_report_schema(
    schema_path: str | Path = DEFAULT_REPORT_SCHEMA_PATH
) -> dict[str, Any]:
    """
    加载 Privacy Impact Report Schema。

    Args:
        schema_path:
            Schema 文件位置。

    Returns:
        dict:
            JSON Schema。
    """

    schema_path = Path(schema_path)

    if not schema_path.exists():

        raise FileNotFoundError(
            f"Privacy Impact Report Schema 不存在："
            f"{schema_path}"
        )

    try:

        with schema_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            schema = json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"Report Schema JSON 格式错误：{exc}"
        ) from exc

    if not isinstance(schema, dict):

        raise ValueError(
            "Report Schema 必须是 JSON Object。"
        )

    Draft202012Validator.check_schema(
        schema
    )

    return schema


# ============================================================
# 4. Schema Reference 解析
# ============================================================

def _resolve_local_reference(
    root_schema: dict[str, Any],
    reference: str
) -> Any:
    """
    解析 JSON Schema 内部 $ref。

    当前 Report Schema 使用：

        #/$defs/impacted_node
        #/$defs/risk_factor
        #/$defs/compliance_reference
        #/$defs/evidence
        #/$defs/suggestion

    只允许内部引用，不允许从外部 URL 下载 Schema。
    """

    if not reference.startswith("#/"):

        raise ValueError(
            f"不支持的外部 JSON Schema Reference：{reference}"
        )

    current: Any = root_schema

    parts = reference[2:].split("/")

    for part in parts:

        part = (
            part.replace("~1", "/")
            .replace("~0", "~")
        )

        if not isinstance(current, dict) or part not in current:

            raise ValueError(
                f"无法解析 JSON Schema Reference：{reference}"
            )

        current = current[part]

    return current


# ============================================================
# 5. 转换 Groq Strict Structured Output Schema
# ============================================================

def build_provider_strict_schema(
    schema_path: str | Path = DEFAULT_REPORT_SCHEMA_PATH
) -> dict[str, Any]:
    """
    将本地完整 JSON Schema 转换为 Groq Strict
    Structured Outputs 所需的兼容子集。

    原始契约：

        schemas/privacy_impact_report.schema.json

    保留：
        type
        properties
        required
        additionalProperties
        items
        enum
        anyOf

    处理：
        1. 内联 $ref。
        2. 移除 $defs。
        3. 移除 format。
        4. 移除 minLength/maxLength。
        5. 移除 minimum/maximum。
        6. 移除 pattern/uniqueItems。
        7. 将 type=["string","null"] 转换成 anyOf。

    目的：
        让模型 API 使用可接受的 Strict Schema，
        同时由本地完整 Schema 执行最终验证。

    注意：
        本函数只生成 API 适配 Schema，
        不会覆盖原始 Report Schema。
    """

    root_schema = load_report_schema(
        schema_path
    )

    supported_keys = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "anyOf"
    }

    def transform(node: Any) -> Any:

        if isinstance(node, list):

            return [
                transform(item)
                for item in node
            ]

        if not isinstance(node, dict):

            return copy.deepcopy(node)

        # ----------------------------------------------------
        # 解析 $ref
        # ----------------------------------------------------

        if "$ref" in node:

            resolved = _resolve_local_reference(
                root_schema,
                node["$ref"]
            )

            # 保留可能存在的其他 Schema 约束。
            combined = dict(resolved)

            for key, value in node.items():

                if key != "$ref":
                    combined[key] = value

            return transform(combined)

        result: dict[str, Any] = {}

        # ----------------------------------------------------
        # Nullable 类型处理
        # ----------------------------------------------------

        node_type = node.get("type")

        if isinstance(node_type, list):

            result["anyOf"] = [
                {
                    "type": item
                }
                for item in node_type
            ]

        # ----------------------------------------------------
        # 标准字段递归处理
        # ----------------------------------------------------

        for key, value in node.items():

            if key not in supported_keys:
                continue

            if key == "type" and isinstance(node_type, list):
                continue

            if key == "properties":

                result[key] = {
                    name: transform(value)
                    for name, value in value.items()
                }

            elif key == "items":

                result[key] = transform(value)

            elif key == "anyOf":

                result[key] = [
                    transform(item)
                    for item in value
                ]

            else:

                result[key] = copy.deepcopy(value)

        # ----------------------------------------------------
        # Strict 模式下检查对象
        # ----------------------------------------------------

        if result.get("type") == "object":

            properties = result.get(
                "properties",
                {}
            )

            required = set(
                result.get("required", [])
            )

            if set(properties.keys()) != required:

                raise ValueError(
                    "Strict Structured Output 要求所有对象属性"
                    "均属于 required。\n"
                    f"属性：{sorted(properties.keys())}\n"
                    f"required：{sorted(required)}"
                )

            if result.get("additionalProperties") is not False:

                raise ValueError(
                    "Strict Structured Output 要求 "
                    "additionalProperties=false。"
                )

        return result

    strict_schema = transform(
        root_schema
    )

    if strict_schema.get("type") != "object":

        raise ValueError(
            "最终 Strict Schema 根类型必须是 object。"
        )

    Draft202012Validator.check_schema(
        strict_schema
    )

    return strict_schema


# ============================================================
# 6. 字段引用辅助方法
# ============================================================

def _extract_field_ids(
    edge: dict[str, Any],
    reference_key: str
) -> set[str]:
    """
    从 Edge 的 source_field_refs 或 target_field_refs
    提取 field_id。

    不修改阶段一定义的原始字段名称。
    """

    return {
        field_ref["field_id"]
        for field_ref in edge.get(
            reference_key,
            []
        )
    }


# ============================================================
# 7. 精确字段传播路径计算
# ============================================================

def derive_impact_descriptors(
    analysis_context: dict[str, Any],
    source_details_result: dict[str, Any],
    downstream_result: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """
    根据 changed_field_ids 和真实的 MCP paths/edges，
    精确识别受影响的节点与字段。

    输入：
        analysis_context
        source_details_result
        downstream_result

    处理逻辑：

        初始字段 = changed_field_ids

        如果 changed_field_ids 为空：
            使用源节点全部字段。

        对每条路径：
            检查当前字段是否与 source_field_refs 有交集。
            如果有交集，将 target_field_refs 的字段
            作为下一层传播范围。

        如果不存在交集：
            当前路径不属于该变更的影响路径。

    返回：
        {
            node_id: {
                "field_ids": set,
                "path_depth": int,
                "edge_ids": set
            }
        }

    该计算采用阶段二定义的转换任务组级保守映射策略，
    不会声称 source_field_refs 和 target_field_refs
    存在尚未定义的一对一关系。
    """

    if not source_details_result.get("is_found"):

        raise ValueError(
            "源节点不存在，无法计算字段影响范围。"
        )

    if not downstream_result.get("is_found"):

        raise ValueError(
            "下游查询起始节点不存在。"
        )

    source_node_id = (
        analysis_context["source_node_id"]
    )

    source_node = source_details_result["node"]

    if source_node["node_id"] != source_node_id:

        raise ValueError(
            "源节点详情与 analysis_context 不一致。"
        )

    source_fields = {
        field["field_id"]: field
        for field in source_node["fields"]
    }

    changed_field_ids = set(
        analysis_context["changed_field_ids"]
    )

    if changed_field_ids:

        unknown_fields = (
            changed_field_ids
            - set(source_fields.keys())
        )

        if unknown_fields:

            raise ValueError(
                "用户指定的 changed_field_ids "
                "在源节点中不存在："
                f"{sorted(unknown_fields)}"
            )

    else:

        # 未指定字段时按整个源数据资产进行分析。
        changed_field_ids = set(
            source_fields.keys()
        )

    if not changed_field_ids:

        raise ValueError(
            "源节点没有可供分析的字段。"
        )

    # --------------------------------------------------------
    # 建立 Edge ID 索引
    # --------------------------------------------------------

    edges_by_id = {
        edge["edge_id"]: edge
        for edge in downstream_result.get(
            "edges",
            []
        )
    }

    # --------------------------------------------------------
    # 起始节点始终属于分析范围
    # --------------------------------------------------------

    descriptors: dict[str, dict[str, Any]] = {
        source_node_id: {
            "field_ids": set(changed_field_ids),
            "path_depth": 0,
            "edge_ids": set()
        }
    }

    # --------------------------------------------------------
    # 遍历下游节点路径
    # --------------------------------------------------------

    for downstream_node in downstream_result.get(
        "nodes",
        []
    ):

        target_node_id = downstream_node["node_id"]

        # 使用独立的临时集合，汇总所有有效路径。
        affected_fields: set[str] = set()

        relevant_edge_ids: set[str] = set()

        minimum_depth: int | None = None

        for path in downstream_node.get(
            "paths",
            []
        ):

            path_node_ids = path["node_ids"]

            path_edge_ids = path["edge_ids"]

            if not path_edge_ids:
                continue

            if path_node_ids[0] != source_node_id:
                continue

            if path_node_ids[-1] != target_node_id:
                continue

            if len(path_node_ids) != len(path_edge_ids) + 1:
                continue

            # 每一条路径单独传播。
            current_fields = set(
                changed_field_ids
            )

            current_edge_ids: set[str] = set()

            path_is_affected = True

            for edge_index, edge_id in enumerate(
                path_edge_ids
            ):

                edge = edges_by_id.get(
                    edge_id
                )

                if edge is None:

                    path_is_affected = False

                    break

                # 检查路径节点与 Edge 的起点、终点。
                if (
                    edge["source_node_id"]
                    != path_node_ids[edge_index]
                ):

                    path_is_affected = False

                    break

                if (
                    edge["target_node_id"]
                    != path_node_ids[edge_index + 1]
                ):

                    path_is_affected = False

                    break

                source_field_ids = _extract_field_ids(
                    edge,
                    "source_field_refs"
                )

                # 当前被传播的字段必须进入此转换任务。
                if not (
                    current_fields
                    & source_field_ids
                ):

                    path_is_affected = False

                    break

                target_field_ids = _extract_field_ids(
                    edge,
                    "target_field_refs"
                )

                if not target_field_ids:

                    path_is_affected = False

                    break

                # 按转换任务组级策略传播目标字段。
                current_fields = target_field_ids

                current_edge_ids.add(
                    edge_id
                )

            if (
                not path_is_affected
                or not current_fields
            ):

                continue

            affected_fields.update(
                current_fields
            )

            relevant_edge_ids.update(
                current_edge_ids
            )

            path_depth = len(
                path_edge_ids
            )

            if minimum_depth is None:

                minimum_depth = path_depth

            else:

                minimum_depth = min(
                    minimum_depth,
                    path_depth
                )

        # 如果没有任何路径能够传播本次变更字段，
        # 则不将此节点列入 impacted_nodes。
        if not affected_fields:
            continue

        descriptors[target_node_id] = {
            "field_ids": affected_fields,
            "path_depth": minimum_depth,
            "edge_ids": relevant_edge_ids
        }

    return descriptors


# ============================================================
# 8. 风险规则计算
# ============================================================

def _evaluate_risk(
    analysis_context: dict[str, Any],
    source_node: dict[str, Any],
    source_fields: list[dict[str, Any]],
    expected_impacted_nodes: list[dict[str, Any]],
    evidence_mapping: dict[str, Any],
    compliance_catalog: list[dict[str, Any]],
    downstream_truncated: bool
) -> dict[str, Any]:
    """
    根据项目自定义规则计算风险等级。

    注意：
    这是本项目内部的演示规则，不是 NIST 官方评分公式，
    也不是法律风险的自动定量评估。

    返回：
        is_risk_detected
        risk_level
        decision
        risk_factors
    """

    change_type = analysis_context["change_type"]

    requested_purpose_code = (
        analysis_context["requested_purpose_code"]
    )

    # --------------------------------------------------------
    # 获取原始节点用途审批状态
    # --------------------------------------------------------

    original_purposes = {
        purpose["purpose_code"]: purpose["approval_status"]
        for purpose in source_node["original_purposes"]
    }

    purpose_mismatch = False

    if requested_purpose_code is not None:

        # 必须已经登记并被批准。
        purpose_approved = (
            original_purposes.get(
                requested_purpose_code
            ) == "approved"
        )

        # 每个涉及的字段都需要登记该用途。
        fields_allow_purpose = all(
            requested_purpose_code in field["purpose_codes"]
            for field in source_fields
        )

        purpose_mismatch = not (
            purpose_approved
            and fields_allow_purpose
        )

    # --------------------------------------------------------
    # 敏感度判断
    # --------------------------------------------------------

    source_sensitivity_labels = {
        field["sensitivity_label"]
        for field in source_fields
    }

    has_high_sensitivity = bool(
        source_sensitivity_labels
        & HIGH_SENSITIVITY_LABELS
    )

    # --------------------------------------------------------
    # 检查分析信息是否完整
    # --------------------------------------------------------

    metadata_complete = bool(
        source_node
        and source_fields
        and expected_impacted_nodes
    )

    # --------------------------------------------------------
    # 计算初始风险级别
    # --------------------------------------------------------

    if purpose_mismatch:

        if has_high_sensitivity:

            risk_level = "high"

        else:

            risk_level = "medium"

    elif change_type == "purpose_change":

        # Purpose 变更必须明确提供新用途。
        if requested_purpose_code is None:

            risk_level = "undetermined"

        elif downstream_truncated:

            risk_level = "undetermined"

        else:

            risk_level = "low"

    elif change_type in (
        "field_change",
        "schema_change"
    ):

        if not analysis_context["changed_field_ids"]:

            risk_level = "undetermined"

        elif downstream_truncated:

            risk_level = "undetermined"

        elif has_high_sensitivity:

            risk_level = "medium"

        else:

            risk_level = "low"

    elif change_type in (
        "access_change",
        "new_downstream_consumer"
    ):

        # 当前 Graph Schema 没有完整访问控制主体、
        # 接收方权限和访问策略定义。
        # 因此这些变更不能仅根据数据血缘确定安全性。
        risk_level = "undetermined"

    else:

        risk_level = "undetermined"

    if not metadata_complete:

        risk_level = "undetermined"

    # --------------------------------------------------------
    # 是否发现风险
    # --------------------------------------------------------

    is_risk_detected = (
        risk_level in ("high", "medium")
    )

    # --------------------------------------------------------
    # 决策映射
    # --------------------------------------------------------

    decision_mapping = {
        "high": "block_until_review",
        "medium": "human_review_required",
        "low": "monitor",
        "undetermined": "insufficient_evidence"
    }

    decision = decision_mapping[
        risk_level
    ]

    # --------------------------------------------------------
    # 构造证据 ID 索引
    # --------------------------------------------------------

    node_evidence_ids = (
        evidence_mapping["node"]
    )

    field_evidence_ids = (
        evidence_mapping["field"]
    )

    edge_evidence_ids = (
        evidence_mapping["edge"]
    )

    source_node_id = (
        analysis_context["source_node_id"]
    )

    source_evidence_ids = [
        node_evidence_ids[source_node_id]
    ]

    for field in source_fields:

        field_key = (
            source_node_id,
            field["field_id"]
        )

        evidence_id = field_evidence_ids.get(
            field_key
        )

        if evidence_id:

            source_evidence_ids.append(
                evidence_id
            )

    # --------------------------------------------------------
    # 确定性风险因素
    # --------------------------------------------------------

    risk_factors = []

    if purpose_mismatch:

        risk_factors.append({
            "risk_factor_id": "RF-001",
            "factor_type": "purpose_mismatch",
            "severity": (
                "high"
                if risk_level == "high"
                else "medium"
            ),
            "description": (
                "用户请求的新用途不属于当前数据节点"
                "和所涉及字段的已登记且批准的用途范围。"
            ),
            "evidence_ids": sorted(
                set(source_evidence_ids)
            )
        })

    if has_high_sensitivity:

        risk_factors.append({
            "risk_factor_id": "RF-002",
            "factor_type": "high_sensitivity_data",
            "severity": "high",
            "description": (
                "本次分析涉及 high_sensitive_pii "
                "或 sensitive_financial 字段。"
            ),
            "evidence_ids": sorted(
                set(source_evidence_ids)
            )
        })

    downstream_count = max(
        len(expected_impacted_nodes) - 1,
        0
    )

    if downstream_count >= 5:

        relevant_edges = set()

        for descriptor in expected_impacted_nodes:

            relevant_edges.update(
                descriptor["_edge_ids"]
            )

        risk_factors.append({
            "risk_factor_id": "RF-003",
            "factor_type": "broad_downstream_propagation",
            "severity": "medium",
            "description": (
                f"变更具有 {downstream_count} 个下游影响节点，"
                "需要检查多个处理环节。"
            ),
            "evidence_ids": sorted({
                edge_evidence_ids[edge_id]
                for edge_id in relevant_edges
                if edge_id in edge_evidence_ids
            })
        })

    if not compliance_catalog:

        risk_factors.append({
            "risk_factor_id": "RF-004",
            "factor_type": "insufficient_policy_evidence",
            "severity": "unknown",
            "description": (
                "NIST Privacy Framework 检索未返回"
                "可用于本次分析的政策条目。"
            ),
            "evidence_ids": []
        })

    # --------------------------------------------------------
    # 输出规则评估结果
    # --------------------------------------------------------

    return {
        "is_risk_detected": is_risk_detected,
        "risk_level": risk_level,
        "decision": decision,
        "risk_factors": risk_factors
    }


# ============================================================
# 9. 构建可信事实与证据目录
# ============================================================

def build_trusted_facts(
    analysis_context: dict[str, Any],
    source_details_result: dict[str, Any],
    downstream_result: dict[str, Any],
    node_details_by_id: dict[str, dict[str, Any]],
    policy_result: dict[str, Any],
    report_id: str,
    generated_at: str
) -> dict[str, Any]:
    """
    根据真实 MCP 返回结果构建可信事实。

    主要输出：
        expected_impacted_nodes
        risk_evaluation
        evidence_catalog
        compliance_catalog
        required_limitations

    不依赖 LLM 生成事实数据。
    """

    source_node_id = (
        analysis_context["source_node_id"]
    )

    if not source_details_result.get("is_found"):

        raise ValueError(
            f"源节点不存在：{source_node_id}"
        )

    if not downstream_result.get("is_found"):

        raise ValueError(
            "下游查询没有找到正确的源节点。"
        )

    if source_node_id not in node_details_by_id:

        raise ValueError(
            "缺少源节点的详细元数据。"
        )

    # --------------------------------------------------------
    # 第一步：计算真实受影响路径
    # --------------------------------------------------------

    descriptors = derive_impact_descriptors(
        analysis_context=analysis_context,
        source_details_result=source_details_result,
        downstream_result=downstream_result
    )

    # --------------------------------------------------------
    # 第二步：准备完整节点详情
    # --------------------------------------------------------

    for node_id in descriptors:

        result = node_details_by_id.get(
            node_id
        )

        if not result or not result.get("is_found"):

            raise ValueError(
                f"缺少受影响节点的详细数据：{node_id}"
            )

        if result["node"]["node_id"] != node_id:

            raise ValueError(
                f"节点详情 ID 与请求 ID 不匹配：{node_id}"
            )

    # --------------------------------------------------------
    # 第三步：构造 Evidence Catalog
    # --------------------------------------------------------

    evidence_catalog: list[dict[str, Any]] = []

    node_evidence_ids: dict[str, str] = {}

    field_evidence_ids: dict[
        tuple[str, str],
        str
    ] = {}

    edge_evidence_ids: dict[str, str] = {}

    policy_evidence_ids: dict[str, str] = {}

    def add_evidence(
        source_type: str,
        source_tool: str,
        source_pointer: str,
        observed_fact: str,
        supports_claim: str
    ) -> str:

        evidence_id = (
            f"EV-{len(evidence_catalog) + 1:04d}"
        )

        evidence_catalog.append({
            "evidence_id": evidence_id,
            "source_type": source_type,
            "source_tool": source_tool,
            "source_pointer": source_pointer,
            "observed_fact": observed_fact,
            "supports_claim": supports_claim
        })

        return evidence_id

    # --------------------------------------------------------
    # 3.1 节点和字段证据
    # --------------------------------------------------------

    expected_impacted_nodes = []

    for node_id in sorted(
        descriptors,
        key=lambda item: (
            descriptors[item]["path_depth"],
            item
        )
    ):

        descriptor = descriptors[node_id]

        node = node_details_by_id[
            node_id
        ]["node"]

        node_fields = {
            field["field_id"]: field
            for field in node["fields"]
        }

        affected_field_ids = sorted(
            descriptor["field_ids"]
        )

        for field_id in affected_field_ids:

            if field_id not in node_fields:

                raise ValueError(
                    f"受影响节点 {node_id} "
                    f"不存在字段 {field_id}"
                )

        # ----------------------------------------------------
        # 节点级证据
        # ----------------------------------------------------

        node_fact = {
            "node_id": node["node_id"],
            "name": node["name"],
            "node_type": node["node_type"],
            "asset_kind": node["asset_kind"],
            "sensitivity_labels": sorted(
                node["sensitivity_labels"]
            ),
            "original_purposes": node["original_purposes"]
        }

        node_evidence_id = add_evidence(
            source_type="graph_node",
            source_tool="get_node_details",
            source_pointer=node_id,
            observed_fact=json.dumps(
                node_fact,
                ensure_ascii=False,
                sort_keys=True
            ),
            supports_claim="支持核查该数据资产的类型、敏感度与登记用途。"
        )

        node_evidence_ids[node_id] = (
            node_evidence_id
        )

        impact_evidence_ids = [
            node_evidence_id
        ]

        affected_fields = []

        # ----------------------------------------------------
        # 字段级证据
        # ----------------------------------------------------

        requested_purpose_code = (
            analysis_context["requested_purpose_code"]
        )

        all_fields_allow_purpose = True

        for field_id in affected_field_ids:

            field = node_fields[field_id]

            field_fact = {
                "node_id": node_id,
                "field_id": field_id,
                "name": field["name"],
                "data_type": field["data_type"],
                "sensitivity_label": (
                    field["sensitivity_label"]
                ),
                "purpose_codes": field["purpose_codes"]
            }

            field_evidence_id = add_evidence(
                source_type="graph_field",
                source_tool="get_node_details",
                source_pointer=(
                    f"{node_id}.fields.{field_id}"
                ),
                observed_fact=json.dumps(
                    field_fact,
                    ensure_ascii=False,
                    sort_keys=True
                ),
                supports_claim="支持核查具体字段的敏感度及允许用途。"
            )

            field_evidence_ids[
                (node_id, field_id)
            ] = field_evidence_id

            impact_evidence_ids.append(
                field_evidence_id
            )

            affected_fields.append({
                "field_id": field_id,
                "field_name": field["name"],
                "sensitivity_label": (
                    field["sensitivity_label"]
                )
            })

            if (
                requested_purpose_code is not None
                and requested_purpose_code
                not in field["purpose_codes"]
            ):

                all_fields_allow_purpose = False

        # ----------------------------------------------------
        # 用途匹配
        # ----------------------------------------------------

        original_purposes = {
            purpose["purpose_code"]: (
                purpose["approval_status"]
            )
            for purpose in node["original_purposes"]
        }

        if requested_purpose_code is None:

            purpose_match = "unknown"

        else:

            purpose_is_approved = (
                original_purposes.get(
                    requested_purpose_code
                ) == "approved"
            )

            if (
                purpose_is_approved
                and all_fields_allow_purpose
            ):

                purpose_match = "within_original_purpose"

            else:

                purpose_match = "outside_original_purpose"

        # ----------------------------------------------------
        # 路径边证据
        # ----------------------------------------------------

        for edge_id in sorted(
            descriptor["edge_ids"]
        ):

            if edge_id not in edge_evidence_ids:

                continue

            impact_evidence_ids.append(
                edge_evidence_ids[edge_id]
            )

        expected_node = {
            "node_id": node_id,
            "node_name": node["name"],
            "node_type": node["node_type"],
            "impact_type": (
                "source"
                if descriptor["path_depth"] == 0
                else (
                    "direct"
                    if descriptor["path_depth"] == 1
                    else "transitive"
                )
            ),
            "path_depth": descriptor["path_depth"],
            "sensitivity_labels": sorted(
                node["sensitivity_labels"]
            ),
            "original_purpose_codes": sorted(
                original_purposes.keys()
            ),
            "purpose_match": purpose_match,
            "affected_fields": affected_fields,
            "evidence_ids": sorted(
                set(impact_evidence_ids)
            ),
            "explanation": "",
            # 内部使用，生成最终报告前移除。
            "_edge_ids": sorted(
                descriptor["edge_ids"]
            )
        }

        expected_impacted_nodes.append(
            expected_node
        )

    # --------------------------------------------------------
    # 第四步：构造 Edge 证据
    # --------------------------------------------------------

    all_edges = {
        edge["edge_id"]: edge
        for edge in downstream_result.get(
            "edges",
            []
        )
    }

    relevant_edge_ids = set()

    for descriptor in descriptors.values():

        relevant_edge_ids.update(
            descriptor["edge_ids"]
        )

    for edge_id in sorted(relevant_edge_ids):

        edge = all_edges.get(
            edge_id
        )

        if edge is None:

            raise ValueError(
                f"下游查询缺少相关 Edge：{edge_id}"
            )

        edge_fact = {
            "edge_id": edge["edge_id"],
            "source_node_id": edge["source_node_id"],
            "target_node_id": edge["target_node_id"],
            "task_id": edge["task_id"],
            "task_name": edge["task_name"],
            "transform_type": edge["transform_type"],
            "description": edge["description"],
            "source_field_refs": edge["source_field_refs"],
            "target_field_refs": edge["target_field_refs"],
            "processing_purpose_codes": (
                edge["processing_purpose_codes"]
            )
        }

        evidence_id = add_evidence(
            source_type="graph_edge",
            source_tool="get_downstream_nodes",
            source_pointer=edge_id,
            observed_fact=json.dumps(
                edge_fact,
                ensure_ascii=False,
                sort_keys=True
            ),
            supports_claim="支持核查数据处理任务及其字段传播关系。"
        )

        edge_evidence_ids[edge_id] = evidence_id

    # --------------------------------------------------------
    # 第五步：构造政策引用目录
    # --------------------------------------------------------

    compliance_catalog = []

    seen_reference_ids = set()

    for match in policy_result.get(
        "matches",
        []
    ):

        document_id = match["document_id"]

        if document_id in seen_reference_ids:

            continue

        seen_reference_ids.add(
            document_id
        )

        evidence_id = add_evidence(
            source_type="policy_record",
            source_tool="match_privacy_policy",
            source_pointer=document_id,
            observed_fact=(
                f"{match['subcategory_code']}: "
                f"{match['statement']}"
            ),
            supports_claim="支持将本次分析与 NIST 框架条目建立关联。"
        )

        policy_evidence_ids[
            document_id
        ] = evidence_id

        compliance_catalog.append({
            "reference_id": document_id,
            "framework_name": policy_result["framework_name"],
            "framework_version": policy_result["framework_version"],
            "function_code": match["function_code"],
            "category_code": match["category_code"],
            "subcategory_code": match["subcategory_code"],
            "title": match["title"],
            "retrieved_text": match["statement"],
            "source_url": match["source_url"],
            "relevance_score": match["retrieval_score"],
            "relevance_explanation": ""
        })

    # --------------------------------------------------------
    # 第六步：计算风险规则
    # --------------------------------------------------------

    source_node = node_details_by_id[
        analysis_context["source_node_id"]
    ]["node"]

    source_field_ids = (
        descriptors[
            analysis_context["source_node_id"]
        ]["field_ids"]
    )

    source_fields = [
        field
        for field in source_node["fields"]
        if field["field_id"] in source_field_ids
    ]

    # 临时移除内部辅助键。
    public_impacted_nodes = []

    for node in expected_impacted_nodes:

        public_node = {
            key: copy.deepcopy(value)
            for key, value in node.items()
            if not key.startswith("_")
        }

        public_impacted_nodes.append(
            public_node
        )

    risk_evaluation = _evaluate_risk(
        analysis_context=analysis_context,
        source_node=source_node,
        source_fields=source_fields,
        expected_impacted_nodes=expected_impacted_nodes,
        evidence_mapping={
            "node": node_evidence_ids,
            "field": field_evidence_ids,
            "edge": edge_evidence_ids,
            "policy": policy_evidence_ids
        },
        compliance_catalog=compliance_catalog,
        downstream_truncated=bool(
            downstream_result.get("truncated", False)
        )
    )

    # --------------------------------------------------------
    # 第七步：限制说明
    # --------------------------------------------------------

    required_limitations = list(
        REQUIRED_LIMITATIONS
    )

    if not analysis_context["changed_field_ids"]:

        required_limitations.append(
            "用户未指定 changed_field_ids，"
            "本次按源节点全部字段进行数据血缘追踪。"
        )

    if downstream_result.get("truncated", False):

        required_limitations.append(
            "下游血缘查询结果存在截断，"
            "未返回的节点和路径不在本次报告覆盖范围内。"
        )

    if not compliance_catalog:

        required_limitations.append(
            "本次 NIST RAG 检索没有返回可引用的条目，"
            "因此不能形成基于已检索 NIST 条目的具体引用。"
        )

    if not relevant_edge_ids:

        required_limitations.append(
            "当前源字段没有通过已登记 Edge 找到下游传播路径，"
            "不能据此证明其他未登记的数据流转绝对不存在。"
        )

    # --------------------------------------------------------
    # 第八步：最终可信事实
    # --------------------------------------------------------

    return {
        "report_id": report_id,
        "generated_at": generated_at,
        "analysis_context": copy.deepcopy(
            analysis_context
        ),
        "expected_impacted_nodes": (
            public_impacted_nodes
        ),
        "risk_evaluation": risk_evaluation,
        "evidence_catalog": evidence_catalog,
        "compliance_catalog": compliance_catalog,
        "required_limitations": required_limitations,
        "retrieval_summary": {
            "policy_result_count": len(
                compliance_catalog
            ),
            "policy_retrieval_method": (
                policy_result.get(
                    "matches",
                    [{}]
                )[0].get("retrieval_method")
                if policy_result.get("matches")
                else "none"
            ),
            "downstream_truncated": bool(
                downstream_result.get("truncated", False)
            )
        }
    }



def canonicalize_report(
    report: dict[str, Any],
    trusted_facts: dict[str, Any]
) -> dict[str, Any]:
    """
    将 LLM 生成的报告与 Python 确定性事实进行合并。

    设计原则：

    1. 业务事实由 Python 负责，LLM 不得覆盖。
    2. LLM 负责自然语言解释和审计建议。
    3. 所有 Evidence ID 均由程序统一提供。
    4. 所有 NIST 元数据均来自真实 RAG 结果。
    5. 风险等级和决策由 Python 规则决定。

    该函数不修改原始 Report Schema。

    Args:
        report:
            LLM 生成的初始 JSON 报告。

        trusted_facts:
            build_trusted_facts() 生成的可信事实。

    Returns:
        dict:
            回填后的标准化报告。

    Raises:
        ValueError:
            LLM 返回结果不是合法的 JSON Object。
    """

    if not isinstance(report, dict):

        raise ValueError(
            "LLM report 必须是 JSON Object。"
        )

    # 深拷贝，避免修改模型原始输出。
    canonical_report = copy.deepcopy(
        report
    )

    # ========================================================
    # 1. 固定报告基本信息
    # ========================================================

    canonical_report["report_id"] = (
        trusted_facts["report_id"]
    )

    canonical_report["generated_at"] = (
        trusted_facts["generated_at"]
    )

    # ========================================================
    # 2. 固定 Analysis Context
    # ========================================================

    canonical_report["analysis_context"] = (
        copy.deepcopy(
            trusted_facts["analysis_context"]
        )
    )

    # ========================================================
    # 3. 固定确定性风险判断
    # ========================================================

    risk_evaluation = (
        trusted_facts["risk_evaluation"]
    )

    canonical_report["is_risk_detected"] = (
        risk_evaluation["is_risk_detected"]
    )

    canonical_report["risk_level"] = (
        risk_evaluation["risk_level"]
    )

    canonical_report["decision"] = (
        risk_evaluation["decision"]
    )

    # ========================================================
    # 4. 处理 impacted_nodes
    # ========================================================

    # 获取 LLM 生成的解释。
    llm_impacted_nodes = (
        canonical_report.get(
            "impacted_nodes",
            []
        )
    )

    if not isinstance(
        llm_impacted_nodes,
        list
    ):

        llm_impacted_nodes = []

    llm_nodes_by_id = {}

    for node in llm_impacted_nodes:

        if not isinstance(node, dict):
            continue

        node_id = node.get("node_id")

        if isinstance(node_id, str) and node_id:

            llm_nodes_by_id[node_id] = node

    canonical_impacted_nodes = []

    # 以程序计算结果为准，不能使用 LLM 自行创建的节点。
    for expected_node in trusted_facts[
        "expected_impacted_nodes"
    ]:

        node_id = expected_node["node_id"]

        # 获取模型针对当前节点生成的解释。
        llm_node = llm_nodes_by_id.get(
            node_id,
            {}
        )

        explanation = llm_node.get(
            "explanation",
            ""
        )

        if (
            not isinstance(explanation, str)
            or not explanation.strip()
        ):

            # 如果模型没有提供解释，则使用确定性描述。
            impact_type = expected_node["impact_type"]

            if impact_type == "source":

                explanation = (
                    "该节点是本次分析的源数据资产，"
                    "本次影响范围根据用户指定字段确定。"
                )

            elif impact_type == "direct":

                explanation = (
                    "该节点通过直接数据血缘关系接收了"
                    "本次变更所涉及字段的传播。"
                )

            else:

                explanation = (
                    "该节点通过间接数据血缘路径接收了"
                    "本次变更所涉及字段的传播。"
                )

        # 完整复制 Python 计算的可信数据。
        canonical_node = copy.deepcopy(
            expected_node
        )

        # 只有 explanation 允许使用 LLM 生成的内容。
        canonical_node["explanation"] = (
            explanation.strip()
        )

        canonical_impacted_nodes.append(
            canonical_node
        )

    canonical_report["impacted_nodes"] = (
        canonical_impacted_nodes
    )

    # ========================================================
    # 5. 固定 Risk Factors
    # ========================================================

    # 风险因素及其严重程度来自 Python 规则计算。
    # 不使用 LLM 生成的风险 ID、severity 或 evidence_ids。

    canonical_report["risk_factors"] = (
        copy.deepcopy(
            risk_evaluation["risk_factors"]
        )
    )

    # ========================================================
    # 6. 固定 Evidence
    # ========================================================

    # Evidence ID 必须由程序生成。
    # 不能让 LLM 复制、修改或自行生成 ID。

    canonical_report["evidence"] = (
        copy.deepcopy(
            trusted_facts["evidence_catalog"]
        )
    )

    # ========================================================
    # 7. 处理 Compliance Basis
    # ========================================================

    llm_compliance_basis = (
        canonical_report.get(
            "compliance_basis",
            []
        )
    )

    if not isinstance(
        llm_compliance_basis,
        list
    ):

        llm_compliance_basis = []

    llm_compliance_by_id = {}

    for item in llm_compliance_basis:

        if not isinstance(item, dict):
            continue

        reference_id = item.get(
            "reference_id"
        )

        if isinstance(reference_id, str) and reference_id:

            llm_compliance_by_id[
                reference_id
            ] = item

    canonical_compliance_basis = []

    # 只保留真实 RAG 检索到的条目。
    for expected_reference in trusted_facts[
        "compliance_catalog"
    ]:

        reference_id = (
            expected_reference["reference_id"]
        )

        llm_reference = (
            llm_compliance_by_id.get(
                reference_id,
                {}
            )
        )

        relevance_explanation = (
            llm_reference.get(
                "relevance_explanation",
                ""
            )
        )

        if (
            not isinstance(
                relevance_explanation,
                str
            )
            or not relevance_explanation.strip()
        ):

            relevance_explanation = (
                f"该条目 {expected_reference['subcategory_code']} "
                "由 NIST RAG 检索返回，"
                "其主题与本次数据处理、数据用途或隐私风险问题相关。"
                "具体适用性仍需结合组织政策和适用法律进行判断。"
            )

        canonical_reference = copy.deepcopy(
            expected_reference
        )

        # 只允许 LLM 提供关联解释。
        # reference_id、条款内容、来源、版本和得分均保持原样。
        canonical_reference[
            "relevance_explanation"
        ] = relevance_explanation.strip()

        canonical_compliance_basis.append(
            canonical_reference
        )

    canonical_report["compliance_basis"] = (
        canonical_compliance_basis
    )

    # ========================================================
    # 8. 固定 Limitations
    # ========================================================

    current_limitations = (
        canonical_report.get(
            "limitations",
            []
        )
    )

    if not isinstance(
        current_limitations,
        list
    ):

        current_limitations = []

    required_limitations = list(
        trusted_facts["required_limitations"]
    )

    # 合并模型额外提出的限制条件。
    all_limitations = (
        required_limitations
        + current_limitations
    )

    canonical_report["limitations"] = list(
        dict.fromkeys(
            limitation.strip()
            for limitation in all_limitations
            if (
                isinstance(limitation, str)
                and limitation.strip()
            )
        )
    )

    return canonical_report

# ============================================================
# 10. JSON Schema Validation
# ============================================================

def _format_validation_error(
    error: Any
) -> str:
    """
    将 jsonschema 错误转换成可读路径。
    """

    location = "$"

    for part in error.absolute_path:

        if isinstance(part, int):

            location += f"[{part}]"

        else:

            location += f".{part}"

    return (
        f"{location}: {error.message}"
    )


def _validate_json_schema(
    report: dict[str, Any],
    schema: dict[str, Any]
) -> list[str]:
    """
    执行完整的 Report JSON Schema 验证。
    """

    validator = Draft202012Validator(
        schema,
        format_checker=FormatChecker()
    )

    errors = list(
        validator.iter_errors(report)
    )

    errors.sort(
        key=lambda error: (
            list(map(str, error.absolute_path)),
            error.message
        )
    )

    return [
        _format_validation_error(error)
        for error in errors
    ]


# ============================================================
# 11. 证据引用验证
# ============================================================

def _validate_evidence(
    report: dict[str, Any],
    trusted_facts: dict[str, Any]
) -> list[str]:
    """
    检查报告中的 Evidence 是否来自真实 MCP 工具数据。

    验证：
        1. evidence_id 必须存在。
        2. evidence_id 不得重复。
        3. source_type 必须一致。
        4. source_tool 必须一致。
        5. source_pointer 必须一致。
        6. observed_fact 必须与工具原始事实完全一致。
        7. supports_claim 不得为空。
    """

    errors = []

    catalog = {
        item["evidence_id"]: item
        for item in trusted_facts["evidence_catalog"]
    }

    report_evidence = report["evidence"]

    seen_ids = set()

    for item in report_evidence:

        evidence_id = item["evidence_id"]

        # 必须是非空字符串。
        if (
            not isinstance(evidence_id, str)
            or not evidence_id.strip()
        ):

            errors.append(
                "evidence 数组中存在空 evidence_id。"
            )

            continue

        if evidence_id in seen_ids:

            errors.append(
                f"Evidence ID 重复：{evidence_id}"
            )

            continue

        seen_ids.add(evidence_id)

        trusted_item = catalog.get(evidence_id)

        if trusted_item is None:

            errors.append(
                f"报告引用了不存在的 evidence_id：{evidence_id}"
            )

            continue

        immutable_fields = [
            "source_type",
            "source_tool",
            "source_pointer",
            "observed_fact"
        ]

        for field in immutable_fields:

            if item[field] != trusted_item[field]:

                errors.append(
                    f"Evidence {evidence_id} 的 {field} "
                    "与真实 MCP 数据不一致。"
                )

        if not item["supports_claim"].strip():

            errors.append(
                f"Evidence {evidence_id} 缺少 supports_claim。"
            )

    reported_ids = {
        item["evidence_id"]
        for item in report_evidence
        if (
            isinstance(item.get("evidence_id"), str)
            and item["evidence_id"].strip()
        )
    }

    # 分别检查节点和风险因素中的 Evidence ID。
    for node in report["impacted_nodes"]:

        for evidence_id in node["evidence_ids"]:

            if (
                not isinstance(evidence_id, str)
                or not evidence_id.strip()
            ):

                errors.append(
                    f"节点 {node['node_id']} "
                    "包含空 evidence_id。"
                )

                continue

            if evidence_id not in reported_ids:

                errors.append(
                    f"节点 {node['node_id']} "
                    f"引用了未声明的 Evidence ID："
                    f"{evidence_id}"
                )

    for factor in report["risk_factors"]:

        for evidence_id in factor["evidence_ids"]:

            if (
                not isinstance(evidence_id, str)
                or not evidence_id.strip()
            ):

                errors.append(
                    f"风险因素 {factor['risk_factor_id']} "
                    "包含空 evidence_id。"
                )

                continue

            if evidence_id not in reported_ids:

                errors.append(
                    f"风险因素 {factor['risk_factor_id']} "
                    f"引用了未声明的 Evidence ID："
                    f"{evidence_id}"
                )

    return errors


# ============================================================
# 12. NIST 政策依据验证
# ============================================================

def _validate_compliance_basis(
    report: dict[str, Any],
    trusted_facts: dict[str, Any]
) -> list[str]:
    """
    检查 compliance_basis 是否引用真实的 RAG 结果。

    可以选择使用部分检索结果，
    但不能创建未经检索的条目。
    """

    errors = []

    catalog = {
        item["reference_id"]: item
        for item in trusted_facts["compliance_catalog"]
    }

    seen_references = set()

    immutable_fields = [
        "framework_name",
        "framework_version",
        "function_code",
        "category_code",
        "subcategory_code",
        "title",
        "retrieved_text",
        "source_url"
    ]

    for item in report["compliance_basis"]:

        reference_id = item["reference_id"]

        if reference_id in seen_references:

            errors.append(
                f"compliance_basis 出现重复引用：{reference_id}"
            )

            continue

        seen_references.add(
            reference_id
        )

        trusted_item = catalog.get(
            reference_id
        )

        if trusted_item is None:

            errors.append(
                f"引用了未检索到的 NIST 条目：{reference_id}"
            )

            continue

        for field in immutable_fields:

            if item[field] != trusted_item[field]:

                errors.append(
                    f"NIST 引用 {reference_id} 的 "
                    f"{field} 与 RAG 返回结果不一致。"
                )

        try:

            score_matches = (
                abs(
                    float(item["relevance_score"])
                    - float(trusted_item["relevance_score"])
                ) <= 1e-6
            )

        except (TypeError, ValueError):

            score_matches = False

        if not score_matches:

            errors.append(
                f"NIST 引用 {reference_id} 的检索得分不一致。"
            )

        if not item["relevance_explanation"].strip():

            errors.append(
                f"NIST 引用 {reference_id} 缺少关联说明。"
            )

    return errors


# ============================================================
# 13. 受影响节点验证
# ============================================================

def _validate_impacted_nodes(
    report: dict[str, Any],
    trusted_facts: dict[str, Any]
) -> list[str]:
    """
    验证 impacted_nodes 与程序计算的真实影响范围一致。

    以下事实不得被 LLM 修改：

        node_id
        node_name
        node_type
        impact_type
        path_depth
        sensitivity_labels
        original_purpose_codes
        purpose_match
        affected_fields

    explanation 允许 LLM 用中文解释。
    """

    errors = []

    expected_nodes = {
        node["node_id"]: node
        for node in trusted_facts[
            "expected_impacted_nodes"
        ]
    }

    actual_nodes = {
        node["node_id"]: node
        for node in report["impacted_nodes"]
    }

    if len(actual_nodes) != len(
        report["impacted_nodes"]
    ):

        errors.append(
            "impacted_nodes 中存在重复 node_id。"
        )

    if set(expected_nodes) != set(actual_nodes):

        missing = (
            set(expected_nodes)
            - set(actual_nodes)
        )

        extra = (
            set(actual_nodes)
            - set(expected_nodes)
        )

        if missing:

            errors.append(
                f"报告遗漏了真实受影响节点：{sorted(missing)}"
            )

        if extra:

            errors.append(
                f"报告添加了不存在的受影响节点：{sorted(extra)}"
            )

    immutable_fields = [
        "node_name",
        "node_type",
        "impact_type",
        "path_depth",
        "sensitivity_labels",
        "original_purpose_codes",
        "purpose_match"
    ]

    for node_id, expected in expected_nodes.items():

        actual = actual_nodes.get(node_id)

        if actual is None:
            continue

        for field in immutable_fields:

            actual_value = actual[field]

            expected_value = expected[field]

            # 敏感度标签和 Purpose 是集合语义，
            # 元素顺序不代表任何业务含义。
            #
            # 排序后比较，同时保留对重复元素的检查。
            if field in (
                "sensitivity_labels",
                "original_purpose_codes"
            ):

                actual_value = sorted(actual_value)

                expected_value = sorted(expected_value)

            if actual_value != expected_value:

                errors.append(
                    f"节点 {node_id} 的 {field} "
                    "与可信图数据不一致。"
                    f"预期：{expected_value}；"
                    f"实际：{actual_value}"
                )

        expected_fields = {
            field["field_id"]: field
            for field in expected["affected_fields"]
        }

        actual_fields = {
            field["field_id"]: field
            for field in actual["affected_fields"]
        }

        if set(expected_fields) != set(actual_fields):

            errors.append(
                f"节点 {node_id} 的 affected_fields "
                "与字段传播结果不一致。"
            )

        for field_id, expected_field in expected_fields.items():

            actual_field = actual_fields.get(
                field_id
            )

            if actual_field is None:
                continue

            for field in (
                "field_name",
                "sensitivity_label"
            ):

                if actual_field[field] != expected_field[field]:

                    errors.append(
                        f"节点 {node_id} 字段 {field_id} "
                        f"的 {field} 不一致。"
                    )

        expected_evidence_ids = set(
            expected["evidence_ids"]
        )

        actual_evidence_ids = set(
            actual["evidence_ids"]
        )

        # 每个影响节点必须带有其对应的可信证据。
        if not expected_evidence_ids.issubset(
            actual_evidence_ids
        ):

            errors.append(
                f"节点 {node_id} 缺少必要的证据引用。"
            )

    return errors


# ============================================================
# 14. 风险因素验证
# ============================================================

def _validate_risk_factors(
    report: dict[str, Any],
    trusted_facts: dict[str, Any]
) -> list[str]:
    """
    验证模型输出的风险因素与程序规则一致。

    必须包含程序已经检测到的风险因素。
    不允许修改严重程度或伪造 evidence_ids。
    """

    errors = []

    expected_factors = {
        factor["factor_type"]: factor
        for factor in trusted_facts[
            "risk_evaluation"
        ]["risk_factors"]
    }

    actual_factors = {
        factor["factor_type"]: factor
        for factor in report["risk_factors"]
    }

    if len(actual_factors) != len(
        report["risk_factors"]
    ):

        errors.append(
            "risk_factors 中存在重复 factor_type。"
        )

    for factor_type, expected in expected_factors.items():

        actual = actual_factors.get(
            factor_type
        )

        if actual is None:

            errors.append(
                f"报告遗漏确定性风险因素：{factor_type}"
            )

            continue

        if actual["severity"] != expected["severity"]:

            errors.append(
                f"风险因素 {factor_type} 的严重程度不一致。"
            )

        expected_evidence_ids = set(
            expected["evidence_ids"]
        )

        actual_evidence_ids = set(
            actual["evidence_ids"]
        )

        if not expected_evidence_ids.issubset(
            actual_evidence_ids
        ):

            errors.append(
                f"风险因素 {factor_type} 缺少必要证据。"
            )

    # 检查其他额外风险因素的证据是否存在。
    catalog_ids = {
        item["evidence_id"]
        for item in trusted_facts["evidence_catalog"]
    }

    for factor in report["risk_factors"]:

        for evidence_id in factor["evidence_ids"]:

            if evidence_id not in catalog_ids:

                errors.append(
                    f"风险因素 {factor['factor_type']} "
                    f"引用了无效 Evidence：{evidence_id}"
                )

    return errors


# ============================================================
# 15. 完整报告验证
# ============================================================

def validate_report(
    report: dict[str, Any],
    trusted_facts: dict[str, Any],
    schema_path: str | Path = DEFAULT_REPORT_SCHEMA_PATH
) -> None:
    """
    对最终 LLM 报告执行完整校验。

    验证顺序：

        1. JSON 类型检查。
        2. JSON Schema。
        3. report_id 和 generated_at。
        4. analysis_context。
        5. is_risk_detected / risk_level / decision。
        6. impacted_nodes。
        7. evidence。
        8. compliance_basis。
        9. risk_factors。
        10. required_limitations。

    若失败，抛出 ReportValidationError。

    成功时不修改 report。
    """

    if not isinstance(report, dict):

        raise ReportValidationError([
            "最终报告必须是 JSON Object。"
        ])

    schema = load_report_schema(
        schema_path
    )

    # --------------------------------------------------------
    # 第一层：JSON Schema
    # --------------------------------------------------------

    errors = _validate_json_schema(
        report,
        schema
    )

    if errors:

        raise ReportValidationError(
            errors
        )

    semantic_errors = []

    # --------------------------------------------------------
    # 第二层：元数据及分析上下文
    # --------------------------------------------------------

    if report["report_id"] != trusted_facts["report_id"]:

        semantic_errors.append(
            "report_id 与程序生成的报告 ID 不一致。"
        )

    if report["generated_at"] != trusted_facts["generated_at"]:

        semantic_errors.append(
            "generated_at 与程序生成时间不一致。"
        )

    if report["analysis_context"] != trusted_facts["analysis_context"]:

        semantic_errors.append(
            "analysis_context 与用户原始输入不一致。"
        )

    # --------------------------------------------------------
    # 第三层：风险等级与程序规则
    # --------------------------------------------------------

    expected_evaluation = (
        trusted_facts["risk_evaluation"]
    )

    for field in (
        "is_risk_detected",
        "risk_level",
        "decision"
    ):

        if report[field] != expected_evaluation[field]:

            semantic_errors.append(
                f"{field} 与确定性风险规则计算结果不一致。"
                f"预期：{expected_evaluation[field]}，"
                f"实际：{report[field]}"
            )

    # --------------------------------------------------------
    # 第四层：节点和字段验证
    # --------------------------------------------------------

    semantic_errors.extend(
        _validate_impacted_nodes(
            report,
            trusted_facts
        )
    )

    # --------------------------------------------------------
    # 第五层：Evidence 验证
    # --------------------------------------------------------

    semantic_errors.extend(
        _validate_evidence(
            report,
            trusted_facts
        )
    )

    # --------------------------------------------------------
    # 第六层：RAG 引用验证
    # --------------------------------------------------------

    semantic_errors.extend(
        _validate_compliance_basis(
            report,
            trusted_facts
        )
    )

    # --------------------------------------------------------
    # 第七层：风险因素验证
    # --------------------------------------------------------

    semantic_errors.extend(
        _validate_risk_factors(
            report,
            trusted_facts
        )
    )

    # --------------------------------------------------------
    # 第八层：限制条件验证
    # --------------------------------------------------------

    limitations = set(
        report["limitations"]
    )

    for limitation in trusted_facts[
        "required_limitations"
    ]:

        if limitation not in limitations:

            semantic_errors.append(
                "报告缺少必要限制说明："
                f"{limitation}"
            )

    if semantic_errors:

        raise ReportValidationError(
            semantic_errors
        )


# ============================================================
# 16. 独立测试入口
# ============================================================

def main() -> int:
    """
    测试 Report Schema 和 Groq Strict Schema 兼容转换。

    完整 Report 还需要 MCP 真实查询结果才能执行语义校验。
    """

    print("=" * 65)
    print("Privacy Impact Report Validator")
    print("=" * 65)

    try:

        schema = load_report_schema()

        strict_schema = build_provider_strict_schema()

        Draft202012Validator.check_schema(
            schema
        )

        Draft202012Validator.check_schema(
            strict_schema
        )

        print("[成功] 原始 Report Schema 合法。")

        print("[成功] Groq Strict Schema 转换完成。")

        print(
            f"  顶层必填属性数量："
            f"{len(schema['required'])}"
        )

        print(
            f"  Strict Schema 顶层属性数量："
            f"{len(strict_schema['properties'])}"
        )

        print(
            "\n说明：完整报告验证需要在 "
            "analysis_agent.py 中结合真实 MCP 查询结果执行。"
        )

        return 0

    except Exception as exc:

        print(
            f"\n[失败] {type(exc).__name__}: {exc}"
        )

        return 1


if __name__ == "__main__":

    import sys

    sys.exit(main())
