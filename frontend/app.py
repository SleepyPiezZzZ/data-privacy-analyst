
"""
app.py

Data Privacy Analyst - Interactive Frontend

阶段六：Streamlit 前端及系统集成。

主要功能：
1. 数据血缘图概览。
2. 用户隐私变更请求输入。
3. 调用阶段五 PrivacyImpactAnalysisAgent。
4. 显示风险等级和系统决策。
5. 展示受影响数据节点。
6. 使用 Plotly 展示交互式数据血缘图。
7. 展示受影响字段。
8. 展示风险因素和审计建议。
9. 展示真实 NIST RAG 检索结果。
10. 展示 Evidence 证据链。
11. 下载完整 JSON 分析报告。

运行方式（项目根目录）：

    streamlit run frontend/app.py

WSL 环境下也可以使用：

    streamlit run frontend/app.py --server.address 0.0.0.0

默认地址：

    http://localhost:8501

依赖：
    streamlit
    plotly
    networkx
    python-dotenv
    前五阶段全部依赖
"""

from __future__ import annotations

# ============================================================
# 1. 标准库
# ============================================================

import asyncio
import json
import logging
import os
import sys

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# 2. 项目路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 保证通过 streamlit run frontend/app.py 启动时，
# 仍然可以正确 import 项目根目录下的 graph、llm 等 Package。
if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# 3. 环境变量
# ============================================================

from dotenv import load_dotenv

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# 4. 第三方依赖
# ============================================================

try:

    import streamlit as st

    import networkx as nx

    import plotly.graph_objects as go

except ImportError as exc:

    raise ImportError(
        "前端依赖缺失，请执行：\n"
        "pip install -r requirements.txt"
    ) from exc


# ============================================================
# 5. 项目内部模块
# ============================================================

from graph.graph_loader import (
    load_graph,
    DEFAULT_DATA_PATH,
    DEFAULT_SCHEMA_PATH
)

from llm.analysis_agent import (
    PrivacyImpactAnalysisAgent,
    create_analysis_context,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_TOOL_ROUNDS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_VALIDATION_RETRIES,
    DEFAULT_LLM_TIMEOUT
)


# ============================================================
# 6. 页面配置
# ============================================================

st.set_page_config(
    page_title="Data Privacy Analyst",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 7. 项目常量
# ============================================================

REPORT_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "privacy_impact_report.schema.json"
)

NIST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "nist_privacy_core.json"
)

GRAPH_SCHEMA_PATH = DEFAULT_SCHEMA_PATH

GRAPH_DATA_PATH = DEFAULT_DATA_PATH

RISK_LABELS = {
    "high": "高风险",
    "medium": "中风险",
    "low": "低风险",
    "undetermined": "待补充证据"
}

DECISION_LABELS = {
    "block_until_review": "审核完成前暂停处理",
    "human_review_required": "需要人工审核",
    "monitor": "持续监控",
    "no_material_risk_found": "未发现实质性风险",
    "insufficient_evidence": "证据不足，无法判定"
}

CHANGE_TYPE_LABELS = {
    "purpose_change": "数据使用目的变更",
    "field_change": "数据字段变更",
    "schema_change": "数据结构变更",
    "access_change": "数据访问权限变更",
    "new_downstream_consumer": "新增下游数据使用方",
    "unspecified": "未指定变更类型"
}

SENSITIVITY_LABELS = {
    "high_sensitive_pii": "高敏感个人身份信息",
    "sensitive_pii": "敏感个人信息",
    "pseudonymous_pii": "假名化个人信息",
    "sensitive_financial": "敏感财务信息",
    "derived_sensitive": "衍生敏感信息",
    "non_sensitive": "非敏感信息"
}

RISK_FACTOR_LABELS = {
    "purpose_mismatch": "用途不匹配",
    "high_sensitivity_data": "高敏感数据",
    "broad_downstream_propagation": "广泛的下游传播",
    "external_data_disclosure": "对外数据披露",
    "missing_metadata": "缺失数据元信息",
    "insufficient_policy_evidence": "缺少充分的政策依据"
}

PRIORITY_LABELS = {
    "high": "高优先级",
    "medium": "中优先级",
    "low": "低优先级"
}


# ============================================================
# 8. 日志配置
# ============================================================

logger = logging.getLogger("privacy_frontend")

logger.setLevel(
    logging.INFO
)

if not logger.handlers:

    handler = logging.StreamHandler(
        sys.stderr
    )

    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s"
        )
    )

    logger.addHandler(handler)

logger.propagate = False


# ============================================================
# 9. Session State 初始化
# ============================================================

def initialize_session_state() -> None:
    """
    初始化 Streamlit Session State。

    Streamlit 每次用户交互后都会重新执行脚本，
    因此使用 Session State 保存分析报告。
    """

    default_values = {
        "analysis_report": None,
        "analysis_context": None,
        "analysis_tool_summary": None,
        "analysis_completed_at": None,
        "analysis_error": None
    }

    for key, value in default_values.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_session_state()


# ============================================================
# 10. NetworkX 图加载与缓存
# ============================================================

@st.cache_resource(
    show_spinner="正在加载金融平台数据血缘图..."
)
def get_graph() -> nx.MultiDiGraph:
    """
    加载经过阶段一校验的 NetworkX MultiDiGraph。

    使用 cache_resource：
        用户切换控件或刷新页面时，
        不会重复读取和构建图结构。

    图对象在前端只读使用。
    """

    if not GRAPH_DATA_PATH.exists():

        raise FileNotFoundError(
            f"数据血缘文件不存在：{GRAPH_DATA_PATH}"
        )

    graph = load_graph(
        source=GRAPH_DATA_PATH,
        schema_path=GRAPH_SCHEMA_PATH,
        validate=True
    )

    return graph


# ============================================================
# 11. 通用辅助函数
# ============================================================

def get_node_label(
    graph: nx.MultiDiGraph,
    node_id: str
) -> str:
    """
    获取节点展示名称。
    """

    if not graph.has_node(node_id):

        return node_id

    attributes = graph.nodes[node_id]

    return (
        f"{attributes['name']} "
        f"({node_id})"
    )


def get_changed_field_options(
    graph: nx.MultiDiGraph,
    node_id: str
) -> list[dict[str, Any]]:
    """
    获取源节点的字段选项。

    每个选项包含：
        field_id
        name
        data_type
        sensitivity_label
        purpose_codes
    """

    if not graph.has_node(node_id):
        return []

    fields = graph.nodes[node_id].get(
        "fields",
        []
    )

    return sorted(
        fields,
        key=lambda item: item["field_id"]
    )


def get_api_key() -> str | None:
    """
    获取 API Key。

    优先使用 LLM_API_KEY，
    其次使用 GROQ_API_KEY。

    当前项目使用百炼时，
    建议在 .env 中配置：

        LLM_API_KEY=your_api_key
        LLM_BASE_URL=...
        LLM_MODEL=...
    """

    return (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )


def get_model_configuration() -> dict[str, Any]:
    """
    获取当前 LLM 配置。

    模型配置与阶段五保持一致。
    """

    return {
        "api_key": get_api_key(),
        "base_url": os.getenv(
            "LLM_BASE_URL",
            DEFAULT_LLM_BASE_URL
        ),
        "model": os.getenv(
            "LLM_MODEL",
            DEFAULT_LLM_MODEL
        ),
        "timeout": int(
            os.getenv(
                "LLM_TIMEOUT",
                str(DEFAULT_LLM_TIMEOUT)
            )
        ),
        "max_tool_rounds": int(
            os.getenv(
                "LLM_MAX_TOOL_ROUNDS",
                str(DEFAULT_MAX_TOOL_ROUNDS)
            )
        ),
        "max_tool_calls": int(
            os.getenv(
                "LLM_MAX_TOOL_CALLS",
                str(DEFAULT_MAX_TOOL_CALLS)
            )
        ),
        "max_validation_retries": int(
            os.getenv(
                "LLM_MAX_VALIDATION_RETRIES",
                str(DEFAULT_MAX_VALIDATION_RETRIES)
            )
        )
    }


def run_async_task(
    coroutine
) -> Any:
    """
    在独立线程中运行 asyncio 协程。

    Streamlit 主线程负责 UI。
    Agent 在独立线程和独立事件循环中执行。

    这样可以避免 Streamlit 运行环境中
    可能存在的事件循环冲突。
    """

    def execute():

        return asyncio.run(
            coroutine
        )

    with ThreadPoolExecutor(
        max_workers=1
    ) as executor:

        future = executor.submit(
            execute
        )

        return future.result()


# ============================================================
# 12. 字段级影响边计算
# ============================================================

def compute_affected_edges(
    graph: nx.MultiDiGraph,
    source_node_id: str,
    changed_field_ids: list[str],
    impacted_node_ids: set[str],
    max_depth: int = 10
) -> set[str]:
    """
    根据变更字段计算需要高亮的血缘边。

    该计算遵守阶段二和阶段五的字段传播规则：

    1. 从源节点指定字段开始。
    2. 只有当前字段与 Edge source_field_refs 存在交集时，
       才认为该 Edge 传播了本次变更。
    3. 目标节点字段使用 target_field_refs。
    4. 只能传播到报告中真实存在的 impacted_node_id。
    5. 不允许同一条简单路径重复经过节点。

    注意：
    该函数只用于前端显示，不改变分析报告。
    """

    if not graph.has_node(source_node_id):
        return set()

    source_fields = graph.nodes[
        source_node_id
    ].get(
        "fields",
        []
    )

    if changed_field_ids:

        initial_fields = set(
            changed_field_ids
        )

    else:

        # 用户未指定字段时，分析整个源节点。
        initial_fields = {
            field["field_id"]
            for field in source_fields
        }

    if not initial_fields:
        return set()

    if source_node_id not in impacted_node_ids:
        return set()

    # BFS 状态：
    # current_node_id
    # current_fields
    # current_depth
    # path_node_ids
    queue = [(
        source_node_id,
        frozenset(initial_fields),
        0,
        (source_node_id,)
    )]

    affected_edge_ids: set[str] = set()

    visited_states = set()

    while queue:

        (
            current_node_id,
            current_fields,
            current_depth,
            path_node_ids
        ) = queue.pop(0)

        if current_depth >= max_depth:
            continue

        outgoing_edges = list(
            graph.out_edges(
                current_node_id,
                keys=True,
                data=True
            )
        )

        outgoing_edges.sort(
            key=lambda item: item[3].get(
                "edge_id",
                str(item[2])
            )
        )

        for (
            source,
            target,
            edge_key,
            edge_attributes
        ) in outgoing_edges:

            # 只高亮报告中的实际受影响节点。
            if target not in impacted_node_ids:
                continue

            # 简单路径检查。
            if target in path_node_ids:
                continue

            source_field_ids = {
                item["field_id"]
                for item in edge_attributes.get(
                    "source_field_refs",
                    []
                )
            }

            # 当前字段必须进入该处理任务。
            if not (
                set(current_fields)
                & source_field_ids
            ):
                continue

            target_field_ids = {
                item["field_id"]
                for item in edge_attributes.get(
                    "target_field_refs",
                    []
                )
            }

            if not target_field_ids:
                continue

            edge_id = edge_attributes["edge_id"]

            affected_edge_ids.add(
                edge_id
            )

            next_state = (
                target,
                frozenset(target_field_ids),
                current_depth + 1,
                path_node_ids + (target,)
            )

            if next_state in visited_states:
                continue

            visited_states.add(
                next_state
            )

            queue.append(
                next_state
            )

    return affected_edge_ids


# ============================================================
# 13. Plotly 数据血缘图
# ============================================================

def build_lineage_figure(
    graph: nx.MultiDiGraph,
    report: dict[str, Any] | None = None
) -> go.Figure:
    """
    构建交互式数据血缘图。

    颜色语义：

        源节点：
            蓝色。

        本次受影响节点：
            橙色。

        其他节点：
            灰色。

        本次受影响的 Edge：
            强调显示。

        其他 Edge：
            淡化显示。

    支持：
        鼠标悬停查看节点信息。
        缩放图形。
        平移图形。
        图形工具栏操作。
    """

    if graph.number_of_nodes() == 0:

        return go.Figure()

    # --------------------------------------------------------
    # 1. 确定本次影响范围
    # --------------------------------------------------------

    source_node_id = None

    impacted_node_ids: set[str] = set()

    changed_field_ids: list[str] = []

    if report:

        context = report.get(
            "analysis_context",
            {}
        )

        source_node_id = context.get(
            "source_node_id"
        )

        changed_field_ids = context.get(
            "changed_field_ids",
            []
        )

        impacted_node_ids = {
            node["node_id"]
            for node in report.get(
                "impacted_nodes",
                []
            )
        }

    # --------------------------------------------------------
    # 2. 计算真正受影响的边
    # --------------------------------------------------------

    if (
        source_node_id
        and impacted_node_ids
    ):

        affected_edge_ids = compute_affected_edges(
            graph=graph,
            source_node_id=source_node_id,
            changed_field_ids=changed_field_ids,
            impacted_node_ids=impacted_node_ids,
            max_depth=10
        )

    else:

        affected_edge_ids = set()

    # --------------------------------------------------------
    # 3. 生成节点位置
    # --------------------------------------------------------

    positions = nx.spring_layout(
        graph,
        seed=42,
        k=1.1,
        iterations=100
    )

    # --------------------------------------------------------
    # 4. 初始化 Figure
    # --------------------------------------------------------

    fig = go.Figure()

    # --------------------------------------------------------
    # 5. 绘制 Edge
    # --------------------------------------------------------

    sorted_edges = sorted(
        graph.edges(
            keys=True,
            data=True
        ),
        key=lambda item: item[3]["edge_id"]
    )

    for (
        source,
        target,
        edge_key,
        edge_attributes
    ) in sorted_edges:

        x0, y0 = positions[source]

        x1, y1 = positions[target]

        edge_id = edge_attributes["edge_id"]

        edge_is_affected = (
            edge_id in affected_edge_ids
        )

        edge_color = (
            "#dc704f"
            if edge_is_affected
            else "#a7adb7"
        )

        edge_width = (
            3.4
            if edge_is_affected
            else 1.3
        )

        edge_opacity = (
            0.95
            if edge_is_affected
            else 0.45
        )

        # 绘制连接线。
        fig.add_trace(
            go.Scatter(
                x=[
                    x0,
                    x1
                ],
                y=[
                    y0,
                    y1
                ],
                mode="lines",
                line={
                    "width": edge_width,
                    "color": edge_color
                },
                opacity=edge_opacity,
                hoverinfo="text",
                text=(
                    f"Edge ID: {edge_id}<br>"
                    f"Task: {edge_attributes['task_name']}<br>"
                    f"Transform: {edge_attributes['transform_type']}<br>"
                    f"Source: {source}<br>"
                    f"Target: {target}<br>"
                    f"Affected: {edge_is_affected}"
                ),
                showlegend=False
            )
        )

        # 绘制方向箭头。
        fig.add_annotation(
            x=float(x1),
            y=float(y1),
            ax=float(x0),
            ay=float(y0),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=edge_width,
            arrowcolor=edge_color,
            opacity=edge_opacity
        )

    # --------------------------------------------------------
    # 6. 按状态分组绘制节点
    # --------------------------------------------------------

    node_groups = {
        "源数据节点": [],
        "受影响节点": [],
        "其他数据节点": []
    }

    for node_id, attributes in graph.nodes(data=True):

        if node_id == source_node_id:

            group = "源数据节点"

        elif node_id in impacted_node_ids:

            group = "受影响节点"

        else:

            group = "其他数据节点"

        node_groups[group].append(
            (
                node_id,
                attributes
            )
        )

    node_styles = {
        "源数据节点": {
            "color": "#3975b8",
            "size": 25
        },
        "受影响节点": {
            "color": "#dc704f",
            "size": 23
        },
        "其他数据节点": {
            "color": "#929ba8",
            "size": 17
        }
    }

    for group_name, nodes in node_groups.items():

        if not nodes:
            continue

        style = node_styles[group_name]

        x_values = []
        y_values = []

        labels = []
        hover_texts = []

        for node_id, attributes in nodes:

            x, y = positions[node_id]

            x_values.append(x)

            y_values.append(y)

            labels.append(
                node_id
            )

            hover_texts.append(
                "<br>".join([
                    f"Node ID: {node_id}",
                    f"Name: {attributes['name']}",
                    f"Asset Kind: {attributes['asset_kind']}",
                    f"Node Type: {attributes['node_type']}",
                    (
                        "Sensitivity: "
                        + ", ".join(
                            attributes["sensitivity_labels"]
                        )
                    ),
                    f"Field Count: {len(attributes['fields'])}"
                ])
            )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                name=group_name,
                text=labels,
                textposition="bottom center",
                textfont={
                    "size": 9,
                    "color": "#333333"
                },
                marker={
                    "size": style["size"],
                    "color": style["color"],
                    "line": {
                        "width": 1.5,
                        "color": "#ffffff"
                    }
                },
                hovertext=hover_texts,
                hovertemplate=(
                    "%{hovertext}<extra></extra>"
                )
            )
        )

    # --------------------------------------------------------
    # 7. 布局配置
    # --------------------------------------------------------

    fig.update_layout(
        title={
            "text": "Financial Data Lineage Graph",
            "x": 0.02,
            "xanchor": "left"
        },
        height=700,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20
        },
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1
        },
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False
        },
        yaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False
        }
    )

    return fig


# ============================================================
# 14. 分析结果展示辅助函数
# ============================================================

def show_risk_badge(
    risk_level: str
) -> None:
    """
    显示风险等级。
    """

    label = RISK_LABELS.get(
        risk_level,
        risk_level
    )

    if risk_level == "high":

        st.error(
            f"风险等级：{label}"
        )

    elif risk_level == "medium":

        st.warning(
            f"风险等级：{label}"
        )

    elif risk_level == "low":

        st.success(
            f"风险等级：{label}"
        )

    else:

        st.info(
            f"风险等级：{label}"
        )


def show_risk_factors(
    risk_factors: list[dict[str, Any]]
) -> None:
    """
    展示风险因素。
    """

    if not risk_factors:

        st.info(
            "当前规则未识别出明确风险因素。"
        )

        return

    st.subheader("风险因素")

    for factor in risk_factors:

        factor_label = RISK_FACTOR_LABELS.get(
            factor["factor_type"],
            factor["factor_type"]
        )

        severity_label = RISK_LABELS.get(
            factor["severity"],
            factor["severity"]
        )

        title_text = (
            f"{factor_label} · {severity_label}"
        )

        with st.expander(
            title_text,
            expanded=True
        ):

            st.write(
                factor["description"]
            )

            st.caption(
                "Evidence IDs: "
                + ", ".join(
                    factor["evidence_ids"]
                )
            )


def show_suggestions(
    suggestions: list[dict[str, Any]]
) -> None:
    """
    展示审计建议。
    """

    st.subheader("可执行审计建议")

    if not suggestions:

        st.info(
            "模型没有返回具体审计建议。"
        )

        return

    for index, suggestion in enumerate(
        suggestions,
        start=1
    ):

        priority = PRIORITY_LABELS.get(
            suggestion["priority"],
            suggestion["priority"]
        )

        with st.expander(
            f"{index}. {suggestion['action']} · {priority}",
            expanded=True
        ):

            st.write(
                "**建议原因：** "
                + suggestion["rationale"]
            )

            st.write(
                "**责任角色：** "
                + suggestion["owner_role"]
            )

            st.write(
                "**完成标准：** "
                + suggestion["completion_criteria"]
            )

            if suggestion["blocks_processing"]:

                st.warning(
                    "该建议要求在完成审查之前暂停相关数据处理。"
                )

            else:

                st.caption(
                    "该建议不直接要求阻断数据处理。"
                )


def show_impacted_nodes(
    impacted_nodes: list[dict[str, Any]]
) -> None:
    """
    展示受影响节点摘要。
    """

    st.subheader("受影响数据节点")

    if not impacted_nodes:

        st.info(
            "没有发现可以通过当前数据血缘确认的受影响节点。"
        )

        return

    rows = []

    for node in impacted_nodes:

        rows.append({
            "Node ID": node["node_id"],
            "Node Name": node["node_name"],
            "Node Type": node["node_type"],
            "Impact": node["impact_type"],
            "Path Depth": node["path_depth"],
            "Purpose Match": node["purpose_match"],
            "Affected Fields": len(
                node["affected_fields"]
            )
        })

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True
    )


# ============================================================
# 15. 报告概览页面
# ============================================================

def render_overview(
    report: dict[str, Any],
    graph: nx.MultiDiGraph,
    tool_summary: dict[str, Any] | None
) -> None:
    """
    展示报告概要、风险结果和审计建议。
    """

    context = report["analysis_context"]

    risk_level = report["risk_level"]

    decision = report["decision"]

    impacted_nodes = report["impacted_nodes"]

    evidence = report["evidence"]

    policies = report["compliance_basis"]

    changed_fields = context["changed_field_ids"]

    # --------------------------------------------------------
    # 1. 报告标题和元数据
    # --------------------------------------------------------

    st.markdown(
        "### 隐私影响分析结果"
    )

    st.caption(
        f"Report ID: {report['report_id']}"
    )

    st.caption(
        f"Generated At: {report['generated_at']}"
    )

    # --------------------------------------------------------
    # 2. 风险指标
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "风险等级",
            RISK_LABELS.get(
                risk_level,
                risk_level
            )
        )

    with col2:

        st.metric(
            "影响节点",
            len(impacted_nodes)
        )

    with col3:

        st.metric(
            "证据数量",
            len(evidence)
        )

    with col4:

        st.metric(
            "NIST 参考条目",
            len(policies)
        )

    st.divider()

    # --------------------------------------------------------
    # 3. 风险摘要
    # --------------------------------------------------------

    show_risk_badge(
        risk_level
    )

    st.markdown(
        "#### 分析摘要"
    )

    st.write(
        report["summary"]
    )

    decision_label = DECISION_LABELS.get(
        decision,
        decision
    )

    if decision == "block_until_review":

        st.error(
            f"系统建议：{decision_label}"
        )

    elif decision in (
        "human_review_required",
        "insufficient_evidence"
    ):

        st.warning(
            f"系统建议：{decision_label}"
        )

    else:

        st.info(
            f"系统建议：{decision_label}"
        )

    # --------------------------------------------------------
    # 4. 请求上下文
    # --------------------------------------------------------

    with st.expander(
        "查看本次变更请求",
        expanded=False
    ):

        st.write(
            "**源节点：** "
            + get_node_label(
                graph,
                context["source_node_id"]
            )
        )

        st.write(
            "**变更类型：** "
            + CHANGE_TYPE_LABELS.get(
                context["change_type"],
                context["change_type"]
            )
        )

        st.write(
            "**申请用途：** "
            + str(
                context["requested_purpose_code"]
            )
        )

        st.write(
            "**涉及字段：** "
            + (
                ", ".join(changed_fields)
                if changed_fields
                else "源节点全部字段"
            )
        )

        st.write(
            "**变更描述：** "
            + context["change_description"]
        )

    # --------------------------------------------------------
    # 5. 风险因素
    # --------------------------------------------------------

    show_risk_factors(
        report["risk_factors"]
    )

    st.divider()

    # --------------------------------------------------------
    # 6. 受影响节点
    # --------------------------------------------------------

    show_impacted_nodes(
        impacted_nodes
    )

    st.divider()

    # --------------------------------------------------------
    # 7. 审计建议
    # --------------------------------------------------------

    show_suggestions(
        report["actionable_suggestions"]
    )

    # --------------------------------------------------------
    # 8. 系统执行记录
    # --------------------------------------------------------

    if tool_summary:

        with st.expander(
            "查看 MCP 工具调用记录",
            expanded=False
        ):

            st.metric(
                "工具调用总次数",
                tool_summary["tool_call_count"]
            )

            st.write(
                "已执行工具："
            )

            for tool_name in sorted(
                tool_summary["executed_tool_names"]
            ):

                st.code(
                    tool_name,
                    language=None
                )

    # --------------------------------------------------------
    # 9. 系统限制
    # --------------------------------------------------------

    with st.expander(
        "分析限制与注意事项",
        expanded=False
    ):

        for limitation in report["limitations"]:

            st.write(
                f"- {limitation}"
            )

        st.caption(
            "本系统为数据治理和隐私风险审查辅助工具，"
            "不构成自动法律合规认定。"
        )


# ============================================================
# 16. 数据血缘可视化页面
# ============================================================

def render_lineage(
    report: dict[str, Any],
    graph: nx.MultiDiGraph
) -> None:
    """
    展示完整图结构和字段级影响高亮。
    """

    st.subheader(
        "数据血缘网络"
    )

    st.caption(
        "悬停节点可以查看节点详情，"
        "悬停连线可以查看对应的数据转换任务。"
    )

    impacted_nodes = report[
        "impacted_nodes"
    ]

    source_node_id = report[
        "analysis_context"
    ]["source_node_id"]

    impact_node_ids = {
        node["node_id"]
        for node in impacted_nodes
    }

    changed_field_ids = report[
        "analysis_context"
    ]["changed_field_ids"]

    affected_edge_ids = compute_affected_edges(
        graph=graph,
        source_node_id=source_node_id,
        changed_field_ids=changed_field_ids,
        impacted_node_ids=impact_node_ids
    )

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "图节点总数",
            graph.number_of_nodes()
        )

    with c2:

        st.metric(
            "受影响节点",
            len(impact_node_ids)
        )

    with c3:

        st.metric(
            "字段影响路径边",
            len(affected_edge_ids)
        )

    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------

    figure = build_lineage_figure(
        graph=graph,
        report=report
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False
        }
    )

    # --------------------------------------------------------
    # 图例
    # --------------------------------------------------------

    st.markdown(
        """
        **图例**

        - 蓝色节点：本次分析的源数据节点
        - 橙色节点：本次字段级影响分析确认的受影响节点
        - 灰色节点：其他数据资产
        - 强调色连线：实际参与本次字段传播的转换边
        - 灰色连线：平台中登记的其他数据处理关系
        """
    )

    # --------------------------------------------------------
    # 检查是否存在传播边
    # --------------------------------------------------------

    if not affected_edge_ids:

        st.info(
            "当前涉及字段没有在已登记的 Edge 中找到"
            "可以确认的下游传播路径。"
            "这不代表系统能够证明不存在其他未登记的数据流转。"
        )

    else:

        edges = [
            data
            for _, _, _, data in graph.edges(
                keys=True,
                data=True
            )
            if data["edge_id"] in affected_edge_ids
        ]

        with st.expander(
            "查看本次影响路径上的转换任务"
        ):

            for edge in edges:

                st.write(
                    f"**{edge['edge_id']}**："
                    f"{edge['task_name']}"
                )

                st.caption(
                    f"{edge['source_node_id']} → "
                    f"{edge['target_node_id']}"
                )

                st.write(
                    edge["description"]
                )

    # --------------------------------------------------------
    # 展示完整受影响节点路径深度
    # --------------------------------------------------------

    with st.expander(
        "查看受影响节点路径信息"
    ):

        rows = []

        for node in impacted_nodes:

            rows.append({
                "node_id": node["node_id"],
                "impact_type": node["impact_type"],
                "path_depth": node["path_depth"],
                "affected_fields": ", ".join(
                    field["field_id"]
                    for field in node["affected_fields"]
                )
            })

        st.dataframe(
            rows,
            width="stretch",
            hide_index=True
        )


# ============================================================
# 17. 受影响字段页面
# ============================================================

def render_affected_fields(
    report: dict[str, Any]
) -> None:
    """
    展示本次分析中所有受影响字段。
    """

    st.subheader(
        "字段级影响分析"
    )

    rows = []

    for node in report["impacted_nodes"]:

        for field in node["affected_fields"]:

            sensitivity_label = (
                field["sensitivity_label"]
            )

            rows.append({
                "node_id": node["node_id"],
                "node_name": node["node_name"],
                "field_id": field["field_id"],
                "field_name": field["field_name"],
                "sensitivity_label": sensitivity_label,
                "sensitivity": SENSITIVITY_LABELS.get(
                    sensitivity_label,
                    sensitivity_label
                ),
                "impact_type": node["impact_type"],
                "path_depth": node["path_depth"],
                "purpose_match": node["purpose_match"]
            })

    if not rows:

        st.info(
            "没有发现能够确认的字段级影响。"
        )

        return

    st.metric(
        "受影响字段总数",
        len(rows)
    )

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # 按敏感度统计
    # --------------------------------------------------------

    sensitivity_counts = {}

    for row in rows:

        label = row["sensitivity"]

        sensitivity_counts[label] = (
            sensitivity_counts.get(label, 0) + 1
        )

    st.markdown(
        "#### 字段敏感度分布"
    )

    st.bar_chart(
        sensitivity_counts,
        horizontal=True,
        x_label="字段数量",
        y_label="敏感度标签",
        width="stretch"
    )

    # --------------------------------------------------------
    # 逐节点查看字段
    # --------------------------------------------------------

    st.markdown(
        "#### 节点字段详情"
    )

    for node in report["impacted_nodes"]:

        with st.expander(
            f"{node['node_name']} ({node['node_id']})",
            expanded=False
        ):

            st.write(
                f"影响类型：{node['impact_type']}"
            )

            st.write(
                f"Purpose Match：{node['purpose_match']}"
            )

            for field in node["affected_fields"]:

                st.write(
                    f"- **{field['field_name']}** "
                    f"(`{field['field_id']}`)"
                )

                st.caption(
                    SENSITIVITY_LABELS.get(
                        field["sensitivity_label"],
                        field["sensitivity_label"]
                    )
                )


# ============================================================
# 18. NIST RAG 和 Evidence 页面
# ============================================================

def render_evidence_and_policies(
    report: dict[str, Any]
) -> None:
    """
    展示 RAG 检索结果和证据链。
    """

    evidence_tab, policy_tab = st.tabs([
        "Evidence 证据链",
        "NIST 合规依据"
    ])

    # ========================================================
    # Evidence
    # ========================================================

    with evidence_tab:

        evidence_list = report[
            "evidence"
        ]

        st.metric(
            "证据总数量",
            len(evidence_list)
        )

        if not evidence_list:

            st.warning(
                "当前报告没有可用 Evidence。"
            )

        else:

            evidence_rows = []

            for item in evidence_list:

                evidence_rows.append({
                    "Evidence ID": item["evidence_id"],
                    "Source Type": item["source_type"],
                    "Source Tool": item["source_tool"],
                    "Source Pointer": item["source_pointer"]
                })

            st.dataframe(
                evidence_rows,
                width="stretch",
                hide_index=True
            )

            st.markdown(
                "#### 证据详细内容"
            )

            for item in evidence_list:

                with st.expander(
                    (
                        f"{item['evidence_id']} · "
                        f"{item['source_pointer']}"
                    ),
                    expanded=False
                ):

                    st.write(
                        "**来源类型：** "
                        + item["source_type"]
                    )

                    st.write(
                        "**来源工具：** "
                        + item["source_tool"]
                    )

                    st.write(
                        "**数据路径：** "
                        + item["source_pointer"]
                    )

                    st.write(
                        "**观察到的事实：**"
                    )

                    st.code(
                        item["observed_fact"],
                        language="json"
                    )

                    st.write(
                        "**支持的判断：**"
                    )

                    st.write(
                        item["supports_claim"]
                    )

    # ========================================================
    # NIST Compliance
    # ========================================================

    with policy_tab:

        compliance_basis = report[
            "compliance_basis"
        ]

        st.metric(
            "NIST 检索条目数量",
            len(compliance_basis)
        )

        if not compliance_basis:

            st.warning(
                "当前 RAG 没有返回可引用的 NIST 条目。"
            )

        else:

            for item in compliance_basis:

                title = (
                    f"{item['subcategory_code']} · "
                    f"{item['title']}"
                )

                with st.expander(
                    title,
                    expanded=True
                ):

                    c1, c2 = st.columns(2)

                    with c1:

                        st.caption(
                            "Function"
                        )

                        st.code(
                            item["function_code"],
                            language=None
                        )

                    with c2:

                        st.caption(
                            "Category"
                        )

                        st.code(
                            item["category_code"],
                            language=None
                        )

                    st.markdown(
                        "**NIST 原始声明**"
                    )

                    st.info(
                        item["retrieved_text"]
                    )

                    st.markdown(
                        "**与本次分析的关联**"
                    )

                    st.write(
                        item["relevance_explanation"]
                    )

                    st.write(
                        f"检索相关度："
                        f"{item['relevance_score']:.4f}"
                    )

                    st.write(
                        f"Reference ID："
                        f"`{item['reference_id']}`"
                    )

                    st.markdown(
                        f"[查看 NIST 官方来源]({item['source_url']})"
                    )

                    st.caption(
                        f"Framework Version: "
                        f"{item['framework_version']}"
                    )

        st.caption(
            "RAG 检索相关度表示语义匹配程度，"
            "不代表法律适用性或风险发生概率。"
        )


# ============================================================
# 19. JSON 报告页面
# ============================================================

def render_json_report(
    report: dict[str, Any]
) -> None:
    """
    展示完整 JSON 报告并提供下载。
    """

    st.subheader(
        "完整 Structured Output"
    )

    report_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    )

    st.download_button(
        label="下载完整 JSON 报告",
        data=report_text,
        file_name=(
            f"{report['report_id']}.json"
        ),
        mime="application/json",
        width="stretch",
        icon=":material/download:"
    )

    st.download_button(
        label="下载报告 TXT 副本",
        data=report_text,
        file_name=(
            f"{report['report_id']}.txt"
        ),
        mime="text/plain",
        width="stretch",
        icon=":material/description:"
    )

    with st.expander(
        "展开完整 JSON 内容",
        expanded=False
    ):

        st.json(
            report,
            expanded=True
        )


# ============================================================
# 20. 分析报告总展示入口
# ============================================================

def render_analysis_report(
    graph: nx.MultiDiGraph
) -> None:
    """
    根据 Session State 展示已经完成的分析结果。
    """

    report = st.session_state.analysis_report

    if report is None:

        return

    st.divider()

    st.header(
        "分析报告"
    )

    tool_summary = (
        st.session_state.analysis_tool_summary
    )

    # --------------------------------------------------------
    # 切换不同报告页面
    # --------------------------------------------------------

    overview_tab, lineage_tab, field_tab, evidence_tab, json_tab = (
        st.tabs([
            "📋 分析概览",
            "🔗 数据血缘",
            "🔬 字段影响",
            "📚 证据与政策",
            "📄 JSON 报告"
        ])
    )

    with overview_tab:

        render_overview(
            report=report,
            graph=graph,
            tool_summary=tool_summary
        )

    with lineage_tab:

        render_lineage(
            report=report,
            graph=graph
        )

    with field_tab:

        render_affected_fields(
            report
        )

    with evidence_tab:

        render_evidence_and_policies(
            report
        )

    with json_tab:

        render_json_report(
            report
        )


# ============================================================
# 21. 系统概览
# ============================================================

def render_system_overview(
    graph: nx.MultiDiGraph
) -> None:
    """
    展示系统数据资产和运行状态。
    """

    st.title(
        "🔐 Data Privacy Analyst"
    )

    st.caption(
        "LLM-powered Data Lineage & Privacy Impact Analysis System"
    )

    # --------------------------------------------------------
    # 图数据统计
    # --------------------------------------------------------

    table_count = sum(
        attributes["asset_kind"] == "table"
        for _, attributes in graph.nodes(data=True)
    )

    api_count = sum(
        attributes["asset_kind"] == "api"
        for _, attributes in graph.nodes(data=True)
    )

    field_count = sum(
        len(attributes["fields"])
        for _, attributes in graph.nodes(data=True)
    )

    graph_col, table_col, api_col, field_col = st.columns(4)

    with graph_col:

        st.metric(
            "数据节点",
            graph.number_of_nodes()
        )

    with table_col:

        st.metric(
            "数据表",
            table_count
        )

    with api_col:

        st.metric(
            "API",
            api_count
        )

    with field_col:

        st.metric(
            "数据字段",
            field_count
        )

    # --------------------------------------------------------
    # 系统资源状态
    # --------------------------------------------------------

    st.caption(
        f"Graph ID: {graph.graph.get('graph_id')} "
        f"| Platform: {graph.graph.get('platform')} "
        f"| Edges: {graph.number_of_edges()}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        if NIST_DATA_PATH.exists():

            st.success(
                "NIST 知识库：已就绪"
            )

        else:

            st.error(
                "NIST 知识库：未找到"
            )

    with col2:

        if get_api_key():

            st.success(
                "LLM API Key：已配置"
            )

        else:

            st.error(
                "LLM API Key：未配置"
            )

    with col3:

        st.success(
            "NetworkX Graph：已加载"
        )

    st.divider()


# ============================================================
# 22. 分析请求配置
# ============================================================

def render_analysis_form(
    graph: nx.MultiDiGraph
) -> None:
    """
    展示用户隐私变更请求表单。

    源节点选择放在表单外面，
    保证改变源节点时，字段列表能够立即更新。
    """

    st.header(
        "隐私变更分析请求"
    )

    st.caption(
        "指定需要变更的数据资产、处理用途和涉及字段，"
        "系统将自动调用 MCP 和 LLM 进行分析。"
    )

    node_ids = sorted(
        graph.nodes()
    )

    # --------------------------------------------------------
    # 1. 源节点选择
    # --------------------------------------------------------

    default_source = (
        "T01_CUSTOMER_RAW"
        if "T01_CUSTOMER_RAW" in node_ids
        else node_ids[0]
    )

    if "selected_source_node" not in st.session_state:

        st.session_state.selected_source_node = (
            default_source
        )

    source_node_id = st.selectbox(
        "源数据节点",
        options=node_ids,
        key="selected_source_node",
        format_func=lambda node_id: get_node_label(
            graph,
            node_id
        ),
        help="选择此次变更的起始数据资产。"
    )

    source_attributes = graph.nodes[
        source_node_id
    ]

    st.caption(
        f"节点类型：{source_attributes['node_type']} "
        f"| 资产类型：{source_attributes['asset_kind']} "
        f"| 字段数量：{len(source_attributes['fields'])}"
    )

    # --------------------------------------------------------
    # 2. 字段选择
    # --------------------------------------------------------

    field_options = get_changed_field_options(
        graph,
        source_node_id
    )

    field_ids = [
        field["field_id"]
        for field in field_options
    ]

    field_lookup = {
        field["field_id"]: field
        for field in field_options
    }

    # --------------------------------------------------------
    # 3. 变更请求表单
    # --------------------------------------------------------

    with st.form(
        "privacy_analysis_form",
        clear_on_submit=False
    ):

        change_type = st.selectbox(
            "变更类型",
            options=list(
                CHANGE_TYPE_LABELS.keys()
            ),
            format_func=lambda value: (
                CHANGE_TYPE_LABELS[value]
            ),
            index=0,
            help="选择本次需要评估的数据变更类型。"
        )

        change_description = st.text_area(
            "变更描述",
            placeholder=(
                "例如：申请将客户原始数据中的姓名和电话号码"
                "用于营销个性化推荐。"
            ),
            height=110,
            help="尽可能详细地说明变更内容和业务目的。"
        )

        requested_purpose_code = st.text_input(
            "申请使用目的代码",
            value="marketing_personalization",
            placeholder="例如：marketing_personalization",
            help=(
                "对于 purpose_change 必须填写。"
                "其他变更类型可根据情况填写。"
            )
        )

        st.markdown(
            "#### 本次涉及的数据字段"
        )

        selected_fields = st.multiselect(
            "选择字段",
            options=field_ids,
            default=[],
            format_func=lambda field_id: (
                f"{field_lookup[field_id]['name']} "
                f"({field_id}) · "
                f"{SENSITIVITY_LABELS.get(field_lookup[field_id]['sensitivity_label'], field_lookup[field_id]['sensitivity_label'])}"
            ),
            help=(
                "如果不选择任何字段，系统将分析整个源节点的全部字段。"
            )
        )

        # 显示源节点所有字段信息，帮助用户判断。
        with st.expander(
            "查看源节点完整字段元数据"
        ):

            field_rows = []

            for field in field_options:

                field_rows.append({
                    "field_id": field["field_id"],
                    "name": field["name"],
                    "data_type": field["data_type"],
                    "sensitivity": SENSITIVITY_LABELS.get(
                        field["sensitivity_label"],
                        field["sensitivity_label"]
                    ),
                    "purpose_codes": ", ".join(
                        field["purpose_codes"]
                    )
                })

            st.dataframe(
                field_rows,
                width="stretch",
                hide_index=True
            )

        st.divider()

        submitted = st.form_submit_button(
            "开始隐私影响分析",
            type="primary",
            width="stretch",
            icon=":material/analytics:"
        )

    # --------------------------------------------------------
    # 4. 提交后运行 Agent
    # --------------------------------------------------------

    if submitted:

        if not change_description.strip():

            st.error(
                "请输入本次数据变更描述。"
            )

            return

        purpose_code = (
            requested_purpose_code.strip()
            or None
        )

        if (
            change_type == "purpose_change"
            and not purpose_code
        ):

            st.error(
                "数据使用目的变更必须指定 requested_purpose_code。"
            )

            return

        # ----------------------------------------------------
        # 获取 API Configuration
        # ----------------------------------------------------

        config = get_model_configuration()

        if not config["api_key"]:

            st.error(
                "没有检测到 LLM API Key。"
                "请在项目根目录 .env 中配置 LLM_API_KEY。"
            )

            return

        # ----------------------------------------------------
        # 创建标准 Analysis Context
        # ----------------------------------------------------

        try:

            analysis_context = create_analysis_context(
                source_node_id=source_node_id,
                change_type=change_type,
                change_description=change_description,
                requested_purpose_code=purpose_code,
                changed_field_ids=selected_fields
            )

        except Exception as exc:

            st.error(
                f"分析请求参数不合法：{exc}"
            )

            return

        # ----------------------------------------------------
        # 初始化 Agent
        # ----------------------------------------------------

        agent = PrivacyImpactAnalysisAgent(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            timeout=config["timeout"],
            max_tool_rounds=config["max_tool_rounds"],
            max_tool_calls=config["max_tool_calls"],
            max_validation_retries=(
                config["max_validation_retries"]
            )
        )

        # ----------------------------------------------------
        # 执行分析
        # ----------------------------------------------------

        st.session_state.analysis_error = None

        try:

            with st.spinner(
                "正在执行隐私影响分析……"
                "系统将依次调用 MCP 工具、检索 NIST 政策并生成结构化报告。"
            ):

                report = run_async_task(
                    agent.analyze(
                        analysis_context=analysis_context,
                        schema_path=REPORT_SCHEMA_PATH
                    )
                )

            # ------------------------------------------------
            # 保存报告到 Session State
            # ------------------------------------------------

            st.session_state.analysis_report = report

            st.session_state.analysis_context = (
                analysis_context
            )

            st.session_state.analysis_tool_summary = {
                "tool_call_count": (
                    agent.tool_results.tool_call_count
                ),
                "executed_tool_names": sorted(
                    agent.tool_results.executed_tool_names
                )
            }

            st.session_state.analysis_completed_at = (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            )

            st.success(
                "隐私影响分析完成，报告已通过结构和事实一致性验证。"
            )

            # 切换为默认页面，在下次渲染时查看最新报告。
            st.rerun()

        except Exception as exc:

            st.session_state.analysis_report = None

            st.session_state.analysis_context = None

            st.session_state.analysis_tool_summary = None

            st.session_state.analysis_error = (
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "Privacy impact analysis failed."
            )

            st.error(
                "隐私影响分析执行失败。"
            )

            st.code(
                st.session_state.analysis_error,
                language=None
            )


# ============================================================
# 23. 侧边栏
# ============================================================

def render_sidebar() -> None:
    """
    显示运行配置和报告管理。
    """

    with st.sidebar:

        st.title(
            "🔐 Privacy Analyst"
        )

        st.caption(
            "LLM + MCP + Graph RAG"
        )

        st.divider()

        st.subheader(
            "模型配置"
        )

        model = os.getenv(
            "LLM_MODEL",
            DEFAULT_LLM_MODEL
        )

        base_url = os.getenv(
            "LLM_BASE_URL",
            DEFAULT_LLM_BASE_URL
        )

        st.text_input(
            "LLM Model",
            value=model,
            disabled=True
        )

        st.text_input(
            "API Base URL",
            value=base_url,
            disabled=True
        )

        api_key = get_api_key()

        if api_key:

            st.success(
                "API Key 已配置"
            )

        else:

            st.error(
                "API Key 缺失"
            )

        st.divider()

        st.subheader(
            "数据资源"
        )

        st.caption(
            f"Graph Data:\n{GRAPH_DATA_PATH.name}"
        )

        st.caption(
            f"NIST Dataset:\n{NIST_DATA_PATH.name}"
        )

        st.caption(
            "Vector Store: ChromaDB"
        )

        st.divider()

        if st.session_state.analysis_report:

            st.subheader(
                "报告管理"
            )

            if st.session_state.analysis_completed_at:

                st.caption(
                    "最近分析时间："
                    + st.session_state.analysis_completed_at
                )

            if st.button(
                "清除当前分析报告",
                width="stretch"
            ):

                st.session_state.analysis_report = None

                st.session_state.analysis_context = None

                st.session_state.analysis_tool_summary = None

                st.session_state.analysis_completed_at = None

                st.rerun()

        st.divider()

        st.caption(
            "Demo Environment · Financial Data Platform"
        )

        st.caption(
            "NIST Privacy Framework 1.0"
        )


# ============================================================
# 24. 主页面
# ============================================================

def main() -> None:
    """
    Streamlit 应用主入口。
    """

    # --------------------------------------------------------
    # 初始化界面
    # --------------------------------------------------------

    render_sidebar()

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.06);
            padding: 12px;
            border-radius: 8px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # 加载图数据
    # --------------------------------------------------------

    try:

        graph = get_graph()

    except Exception as exc:

        st.error(
            "数据血缘图加载失败。"
        )

        st.exception(exc)

        return

    # --------------------------------------------------------
    # 系统概览
    # --------------------------------------------------------

    render_system_overview(
        graph
    )

    # --------------------------------------------------------
    # 分析表单
    # --------------------------------------------------------

    render_analysis_form(
        graph
    )

    # --------------------------------------------------------
    # 分析错误
    # --------------------------------------------------------

    if st.session_state.analysis_error:

        st.divider()

        st.error(
            "最近一次分析执行失败："
        )

        st.code(
            st.session_state.analysis_error,
            language=None
        )

    # --------------------------------------------------------
    # 分析报告
    # --------------------------------------------------------

    render_analysis_report(
        graph
    )

    # --------------------------------------------------------
    # 页脚
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "Data Privacy Analyst | "
        "Python + NetworkX + MCP + ChromaDB + LLM"
    )

    st.caption(
        "本系统生成的结果用于辅助数据治理和隐私风险审查，"
        "不替代人工审计或专业法律意见。"
    )


# ============================================================
# 25. 程序入口
# ============================================================

if __name__ == "__main__":

    main()
