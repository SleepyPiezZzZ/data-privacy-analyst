
"""
tools.py

MCP Tools 注册模块。

本模块负责将前三阶段的 Python Service
封装成标准 MCP Tools。

注册的工具：

1. get_downstream_nodes
2. get_node_details
3. match_privacy_policy

技术：
    MCP Python SDK v1
    FastMCP
    Pydantic Field
    Python typing

依赖：
    graph.lineage_query.LineageQueryService
    graph.node_metadata.NodeMetadataService
    rag.policy_retriever.PrivacyPolicyRetriever

注意：
1. Tool 名称必须与 mcp_tool_schemas.json 一致。
2. Tool 输入参数必须与既定契约一致。
3. Tool 返回结果保持原 Service 的输出结构。
4. 预期输入错误通过 ToolError 返回。
5. 非预期内部异常记录到日志并返回安全错误信息。
6. 处理阶段三的 print 输出，避免污染 MCP stdio 通信。
"""

from __future__ import annotations

import logging
import sys
import threading

from collections.abc import Callable
from contextlib import redirect_stdout
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from graph.lineage_query import LineageQueryService
from graph.node_metadata import NodeMetadataService
from rag.policy_retriever import PrivacyPolicyRetriever


# ============================================================
# 1. 日志配置
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# 2. MCP Function Code 类型定义
# ============================================================

# 与 schemas/mcp_tool_schemas.json 中的 Function 枚举一致。
#
# 使用 Literal 让 MCP 自动生成的输入 JSON Schema
# 也能包含对应的 enum，而不仅仅是普通字符串数组。

FunctionCode = Literal[
    "IDENTIFY-P",
    "GOVERN-P",
    "CONTROL-P",
    "COMMUNICATE-P",
    "PROTECT-P"
]


# ============================================================
# 3. 标准输出保护
# ============================================================

# Stage 3 的部分模块会通过 print 输出索引及模型加载信息。
#
# stdio MCP 将 stdout 用作 JSON-RPC 通信通道。
# 因此在调用 Service 时，将 Python 层面的标准输出转发至 stderr。
#
# redirect_stdout 修改的是进程级 sys.stdout。
# 使用线程锁保证三个同步工具不会同时执行该重定向。
#
# 这项保护针对当前阶段三代码中的 Python print 输出；
# 它不替代底层文件描述符级的 stdout 隔离。

_STDOUT_REDIRECT_LOCK = threading.RLock()


# ============================================================
# 4. 通用 Service 执行器
# ============================================================

def _execute_service(
    tool_name: str,
    operation: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """
    执行底层 Python Service 并统一处理异常。

    Args:
        tool_name:
            当前 MCP 工具名称。

        operation:
            实际的 Service 调用函数。

    Returns:
        dict:
            底层 Service 返回的 JSON 字典。

    异常规则：

        ValueError / TypeError:
            作为可识别的输入错误返回给 MCP Client。

        其他异常:
            记录完整服务端日志。
            向 MCP Client 返回不包含内部堆栈的错误消息。
    """

    try:

        # ----------------------------------------------------
        # 保护 stdio JSON-RPC 通信通道
        # ----------------------------------------------------

        with _STDOUT_REDIRECT_LOCK:

            with redirect_stdout(sys.stderr):

                result = operation()

        # ----------------------------------------------------
        # 校验 Service 返回类型
        # ----------------------------------------------------

        if not isinstance(result, dict):

            raise RuntimeError(
                f"{tool_name} 返回值必须是 dict。"
            )

        return result

    except ToolError:

        # 避免对已经转换为 MCP 错误的异常重复包装。
        raise

    except (ValueError, TypeError) as exc:

        # 输入参数不合法，或者底层 Service 参数校验失败。
        logger.warning(
            "%s input validation failed: %s",
            tool_name,
            exc
        )

        raise ToolError(
            f"{tool_name}: {exc}"
        ) from exc

    except Exception as exc:

        # 内部错误保留完整日志，客户端只获得简洁信息。
        logger.exception(
            "%s execution failed",
            tool_name
        )

        raise ToolError(
            f"{tool_name} execution failed. "
            "Please check the MCP server logs."
        ) from exc


# ============================================================
# 5. MCP Tools 注册入口
# ============================================================

def register_tools(
    mcp: FastMCP,
    lineage_service: LineageQueryService,
    metadata_service: NodeMetadataService,
    policy_retriever: PrivacyPolicyRetriever
) -> dict[str, Callable[..., dict[str, Any]]]:
    """
    将三个 Python Service 注册为 MCP Tools。

    Args:
        mcp:
            FastMCP Server 实例。

        lineage_service:
            阶段二的下游血缘查询服务。

        metadata_service:
            阶段二的节点元数据查询服务。

        policy_retriever:
            阶段三的 NIST RAG 检索服务。

    Returns:
        dict:
            已注册的 Python Tool Handler 映射。

    注意：
        三个工具名称、输入参数名称和默认值
        严格遵循阶段一定义的 MCP API Contract。
    """

    if not isinstance(mcp, FastMCP):

        raise TypeError(
            "mcp 必须是 FastMCP 实例。"
        )

    if not isinstance(
        lineage_service,
        LineageQueryService
    ):

        raise TypeError(
            "lineage_service 必须是 LineageQueryService 实例。"
        )

    if not isinstance(
        metadata_service,
        NodeMetadataService
    ):

        raise TypeError(
            "metadata_service 必须是 NodeMetadataService 实例。"
        )

    if not isinstance(
        policy_retriever,
        PrivacyPolicyRetriever
    ):

        raise TypeError(
            "policy_retriever 必须是 PrivacyPolicyRetriever 实例。"
        )

    # ========================================================
    # Tool 1: get_downstream_nodes
    # ========================================================

    @mcp.tool()
    def get_downstream_nodes(
        node_id: Annotated[
            str,
            Field(
                min_length=1,
                description="Starting data asset node ID"
            )
        ],
        max_depth: Annotated[
            int,
            Field(
                ge=1,
                le=20,
                description="Maximum downstream traversal depth"
            )
        ] = 10,
        include_paths: bool = True,
        include_field_mapping: bool = True,
        max_nodes: Annotated[
            int,
            Field(
                ge=1,
                le=200,
                description="Maximum number of downstream nodes"
            )
        ] = 100
    ) -> dict[str, Any]:
        """
        查询指定节点的全部直接和间接下游受影响节点。

        功能：
        1. 从指定 node_id 开始遍历。
        2. 查询所有可达下游节点。
        3. 返回完整血缘路径。
        4. 返回字段级影响映射。
        5. 支持最大深度及最大节点数量限制。

        返回：
            is_found
            query_node_id
            traversal_direction
            traversal_mode
            result_count
            truncated
            nodes
            edges
            warnings
        """

        return _execute_service(
            tool_name="get_downstream_nodes",
            operation=lambda: lineage_service.get_downstream_nodes(
                node_id=node_id,
                max_depth=max_depth,
                include_paths=include_paths,
                include_field_mapping=include_field_mapping,
                max_nodes=max_nodes
            )
        )

    # ========================================================
    # Tool 2: get_node_details
    # ========================================================

    @mcp.tool()
    def get_node_details(
        node_id: Annotated[
            str,
            Field(
                min_length=1,
                description="Target data asset node ID"
            )
        ],
        include_fields: bool = True,
        include_related_edges: bool = True
    ) -> dict[str, Any]:
        """
        获取指定数据资产节点的详细元数据。

        功能：
        1. 查询节点基本信息。
        2. 获取节点级敏感度标签。
        3. 获取原始用途登记。
        4. 获取字段级敏感度和用途。
        5. 获取直接上下游节点和边。

        返回：
            is_found
            node
            lineage_summary
            warnings
        """

        return _execute_service(
            tool_name="get_node_details",
            operation=lambda: metadata_service.get_node_details(
                node_id=node_id,
                include_fields=include_fields,
                include_related_edges=include_related_edges
            )
        )

    # ========================================================
    # Tool 3: match_privacy_policy
    # ========================================================

    @mcp.tool()
    def match_privacy_policy(
        keyword: Annotated[
            str,
            Field(
                min_length=2,
                max_length=1000,
                description="Natural language privacy policy query"
            )
        ],
        framework_version: Literal["1.0"] = "1.0",
        top_k: Annotated[
            int,
            Field(
                ge=1,
                le=10,
                description="Number of policy records to return"
            )
        ] = 5,
        function_filter: Annotated[
            list[FunctionCode],
            Field(
                json_schema_extra={
                    "uniqueItems": True
                },
                description=(
                    "Optional NIST Function filters. "
                    "An empty array means no Function filtering."
                )
            )
        ] = [],
        include_full_text: bool = True
    ) -> dict[str, Any]:
        """
        从 NIST Privacy Framework 1.0 知识库中
        检索与用户问题相关的隐私框架条目。

        检索技术：
            1. Multilingual Embedding
            2. ChromaDB Dense Retrieval
            3. BM25 Sparse Retrieval
            4. Reciprocal Rank Fusion

        返回：
            query
            framework_name
            framework_version
            result_count
            matches

        注意：
            retrieval_score 仅表示检索相关性，
            不能解释为隐私风险概率。
        """

        # Function Filter 必须是去重后的列表。
        if len(function_filter) != len(set(function_filter)):

            raise ToolError(
                "function_filter must contain unique values."
            )

        return _execute_service(
            tool_name="match_privacy_policy",
            operation=lambda: policy_retriever.match_privacy_policy(
                keyword=keyword,
                framework_version=framework_version,
                top_k=top_k,
                function_filter=function_filter,
                include_full_text=include_full_text
            )
        )

    # ========================================================
    # 6. 返回工具映射
    # ========================================================

    # 函数名称与 MCP Tool Name 一致。
    # 同时让 server.py 能够检查注册数量和名称。
    registered_tools = {
        "get_downstream_nodes": get_downstream_nodes,
        "get_node_details": get_node_details,
        "match_privacy_policy": match_privacy_policy
    }

    logger.info(
        "Registered %d MCP tools: %s",
        len(registered_tools),
        ", ".join(registered_tools.keys())
    )

    return registered_tools
