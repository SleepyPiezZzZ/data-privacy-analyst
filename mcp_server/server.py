
"""
server.py

Data Privacy Analyst MCP Server 主入口。

本模块负责：

1. 加载并验证阶段一的数据契约。
2. 初始化 NetworkX 数据血缘图。
3. 初始化阶段二的两个图查询 Service。
4. 初始化阶段三的 NIST RAG Service。
5. 创建 FastMCP Server。
6. 注册三个 MCP Tools。
7. 启动 MCP 传输服务。

默认传输：
    stdio

可选传输：
    streamable-http

运行方式：

    python -m mcp_server.server

HTTP 模式：

    python -m mcp_server.server --transport streamable-http

指定数据文件：

    python -m mcp_server.server \
        --graph-data data/financial_lineage_demo.json \
        --nist-data data/nist_privacy_core.json

依赖：

    mcp < 2.0
    networkx
    graph.graph_loader
    graph.lineage_query
    graph.node_metadata
    rag.policy_retriever
    mcp_server.tools
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from graph.graph_loader import (
    DEFAULT_DATA_PATH,
    DEFAULT_SCHEMA_PATH,
    load_graph
)

from graph.lineage_query import LineageQueryService

from graph.node_metadata import NodeMetadataService

from rag.policy_retriever import PrivacyPolicyRetriever

from rag.document_loader import DEFAULT_OUTPUT_JSON

from mcp_server.tools import register_tools


# ============================================================
# 1. 项目路径及默认配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# MCP 工具契约。
DEFAULT_TOOL_CONTRACT_PATH = (
    PROJECT_ROOT / "schemas" / "mcp_tool_schemas.json"
)

# NIST 知识库。
DEFAULT_NIST_DATA_PATH = DEFAULT_OUTPUT_JSON

# ChromaDB 本地持久化目录。
DEFAULT_CHROMA_DIRECTORY = (
    PROJECT_ROOT / "vector_store" / "chroma"
)

# ChromaDB Collection。
DEFAULT_COLLECTION_NAME = (
    "nist_privacy_framework_v1"
)

# MCP Server 基本信息。
SERVER_NAME = "Data Privacy Lineage Analyst"

SERVER_VERSION = "1.0.0"

# 固定需要注册的三个工具。
EXPECTED_TOOL_NAMES = {
    "get_downstream_nodes",
    "get_node_details",
    "match_privacy_policy"
}


# ============================================================
# 2. 日志配置
# ============================================================

def configure_logging(
    log_level: str = "INFO"
) -> None:
    """
    配置 MCP Server 日志。

    重要：
    stdio 模式下 stdout 专门用于 MCP JSON-RPC 通信。

    所有项目日志必须写入 stderr，
    不能使用 print 输出服务器运行信息。
    """

    numeric_level = getattr(
        logging,
        log_level.upper(),
        None
    )

    if not isinstance(numeric_level, int):

        raise ValueError(
            f"不支持的日志级别：{log_level}"
        )

    logging.basicConfig(
        level=numeric_level,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "- %(message)s"
        ),
        stream=sys.stderr,
        force=True
    )


logger = logging.getLogger(__name__)


# ============================================================
# 3. MCP Tool Contract 文件校验
# ============================================================

def load_tool_contract(
    contract_path: str | Path = DEFAULT_TOOL_CONTRACT_PATH
) -> dict[str, Any]:
    """
    加载并验证阶段一定义的 MCP 工具契约文件。

    检查：

    1. 文件是否存在。
    2. JSON 格式是否合法。
    3. schema_version 是否为 1.0.0。
    4. 三个 MCP Tool 是否全部存在。
    5. 每个 Tool 的 inputSchema 是否存在。
    6. 参数名称是否与 Python Tool Handler 一致。

    该函数不会修改契约文件。
    """

    contract_path = Path(contract_path)

    if not contract_path.exists():

        raise FileNotFoundError(
            f"MCP 工具契约文件不存在：{contract_path}"
        )

    try:

        with contract_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            contract = json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"MCP 工具契约 JSON 格式错误：{exc}"
        ) from exc

    if not isinstance(contract, dict):

        raise ValueError(
            "MCP 工具契约根对象必须是 Object。"
        )

    if contract.get("schema_version") != "1.0.0":

        raise ValueError(
            "MCP 工具契约版本不匹配。"
            "当前要求 schema_version='1.0.0'。"
        )

    tools = contract.get("tools")

    if not isinstance(tools, list):

        raise ValueError(
            "MCP 工具契约中的 tools 必须是数组。"
        )

    # --------------------------------------------------------
    # 预期参数名称
    # --------------------------------------------------------

    expected_parameters = {
        "get_downstream_nodes": {
            "node_id",
            "max_depth",
            "include_paths",
            "include_field_mapping",
            "max_nodes"
        },
        "get_node_details": {
            "node_id",
            "include_fields",
            "include_related_edges"
        },
        "match_privacy_policy": {
            "keyword",
            "framework_version",
            "top_k",
            "function_filter",
            "include_full_text"
        }
    }

    tool_names = []

    for tool in tools:

        if not isinstance(tool, dict):

            raise ValueError(
                "tools 数组中的每个元素必须是 Object。"
            )

        name = tool.get("name")

        if not isinstance(name, str):

            raise ValueError(
                "MCP Tool 缺少合法的 name。"
            )

        tool_names.append(name)

        input_schema = tool.get("inputSchema")

        if not isinstance(input_schema, dict):

            raise ValueError(
                f"Tool {name} 缺少 inputSchema。"
            )

        properties = input_schema.get(
            "properties",
            {}
        )

        if not isinstance(properties, dict):

            raise ValueError(
                f"Tool {name} 的 inputSchema.properties 必须是 Object。"
            )

        actual_parameters = set(
            properties.keys()
        )

        expected = expected_parameters.get(name)

        if expected is None:

            raise ValueError(
                f"发现未定义的 MCP Tool：{name}"
            )

        if actual_parameters != expected:

            raise ValueError(
                f"Tool {name} 的参数名称与代码契约不一致。\n"
                f"预期：{sorted(expected)}\n"
                f"实际：{sorted(actual_parameters)}"
            )

    # --------------------------------------------------------
    # 检查工具唯一性
    # --------------------------------------------------------

    if len(tool_names) != len(set(tool_names)):

        raise ValueError(
            "MCP 工具契约中存在重复 Tool Name。"
        )

    if set(tool_names) != EXPECTED_TOOL_NAMES:

        raise ValueError(
            "MCP 工具集合不完整。\n"
            f"预期：{sorted(EXPECTED_TOOL_NAMES)}\n"
            f"实际：{sorted(tool_names)}"
        )

    logger.info(
        "MCP Tool Contract validated: %d tools",
        len(tool_names)
    )

    return contract


# ============================================================
# 4. MCP Server 创建函数
# ============================================================

def create_server(
    graph_data_path: str | Path = DEFAULT_DATA_PATH,
    graph_schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    tool_contract_path: str | Path = DEFAULT_TOOL_CONTRACT_PATH,
    nist_records_path: str | Path = DEFAULT_NIST_DATA_PATH,
    chroma_directory: str | Path = DEFAULT_CHROMA_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> FastMCP:
    """
    创建并初始化完整 MCP Server。

    Args:
        graph_data_path:
            金融平台数据血缘 JSON 路径。

        graph_schema_path:
            Data Lineage Graph JSON Schema 路径。

        tool_contract_path:
            MCP Tool Contract 路径。

        nist_records_path:
            NIST Privacy Framework 标准化 JSON 路径。

        chroma_directory:
            ChromaDB 持久化目录。

        collection_name:
            ChromaDB Collection 名称。

    Returns:
        FastMCP:
            已注册三个工具的 MCP Server。

    注意：
        该函数不启动传输服务。
        只有 main() 调用 server.run() 才会启动服务。
    """

    logger.info(
        "Initializing %s ...",
        SERVER_NAME
    )

    # ========================================================
    # 第一步：检查 MCP Tool Contract
    # ========================================================

    logger.info(
        "Step 1/6: Validating MCP Tool Contract."
    )

    load_tool_contract(
        tool_contract_path
    )

    # ========================================================
    # 第二步：加载 NetworkX 图
    # ========================================================

    logger.info(
        "Step 2/6: Loading NetworkX data lineage graph."
    )

    graph = load_graph(
        source=graph_data_path,
        schema_path=graph_schema_path,
        validate=True
    )

    logger.info(
        "Graph loaded: %d nodes, %d edges.",
        graph.number_of_nodes(),
        graph.number_of_edges()
    )

    # ========================================================
    # 第三步：初始化图查询 Service
    # ========================================================

    logger.info(
        "Step 3/6: Initializing graph query services."
    )

    lineage_service = LineageQueryService(
        graph=graph
    )

    metadata_service = NodeMetadataService(
        graph=graph
    )

    logger.info(
        "LineageQueryService initialized."
    )

    logger.info(
        "NodeMetadataService initialized."
    )

    # ========================================================
    # 第四步：初始化 NIST RAG Service
    # ========================================================

    logger.info(
        "Step 4/6: Initializing NIST RAG service."
    )

    if not Path(nist_records_path).exists():

        raise FileNotFoundError(
            "NIST 知识库文件不存在："
            f"{nist_records_path}\n"
            "请先完成阶段三："
            "python -m rag.document_loader"
        )

    policy_retriever = PrivacyPolicyRetriever(
        records_path=nist_records_path,
        persist_directory=chroma_directory,
        collection_name=collection_name
    )

    logger.info(
        "PrivacyPolicyRetriever initialized."
    )

    logger.info(
        "Chroma collection: %s",
        collection_name
    )

    # ========================================================
    # 第五步：创建 FastMCP Server
    # ========================================================

    logger.info(
        "Step 5/6: Creating FastMCP Server."
    )

    mcp = FastMCP(
        name=SERVER_NAME,
        instructions=(
            "You are connected to a financial data lineage "
            "and privacy impact analysis system. "
            "Use get_node_details to inspect data assets, "
            "get_downstream_nodes to trace downstream impacts, "
            "and match_privacy_policy to retrieve NIST Privacy "
            "Framework 1.0 references. "
            "Do not invent node IDs, field metadata, or policy "
            "references. Distinguish retrieved facts from "
            "inferences. NIST references are framework guidance, "
            "not automatic legal determinations."
        ),
        json_response=True
    )

    # ========================================================
    # 第六步：注册 MCP Tools
    # ========================================================

    logger.info(
        "Step 6/6: Registering MCP Tools."
    )

    registered_tools = register_tools(
        mcp=mcp,
        lineage_service=lineage_service,
        metadata_service=metadata_service,
        policy_retriever=policy_retriever
    )

    # 再检查一次实际注册的 Python Handler 名称。
    actual_tool_names = set(
        registered_tools.keys()
    )

    if actual_tool_names != EXPECTED_TOOL_NAMES:

        raise RuntimeError(
            "MCP Tool 注册不完整。\n"
            f"预期：{sorted(EXPECTED_TOOL_NAMES)}\n"
            f"实际：{sorted(actual_tool_names)}"
        )

    logger.info(
        "MCP Server initialization completed."
    )

    logger.info(
        "Server name: %s",
        SERVER_NAME
    )

    logger.info(
        "Server version: %s",
        SERVER_VERSION
    )

    logger.info(
        "Registered tools: %s",
        ", ".join(sorted(actual_tool_names))
    )

    return mcp


# ============================================================
# 5. 命令行参数
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    解析 MCP Server 启动参数。

    允许用户在启动时切换数据文件和传输方式，
    不改变三个 MCP Tools 的输入契约。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Data Privacy Analyst MCP Server"
        )
    )

    parser.add_argument(
        "--graph-data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="数据血缘 JSON 文件路径"
    )

    parser.add_argument(
        "--graph-schema",
        type=str,
        default=str(DEFAULT_SCHEMA_PATH),
        help="Data Lineage Graph Schema 路径"
    )

    parser.add_argument(
        "--tool-contract",
        type=str,
        default=str(DEFAULT_TOOL_CONTRACT_PATH),
        help="MCP Tool Contract JSON 路径"
    )

    parser.add_argument(
        "--nist-data",
        type=str,
        default=str(DEFAULT_NIST_DATA_PATH),
        help="NIST Privacy Framework JSON 数据路径"
    )

    parser.add_argument(
        "--chroma-directory",
        type=str,
        default=str(DEFAULT_CHROMA_DIRECTORY),
        help="ChromaDB 本地持久化目录"
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB Collection 名称"
    )

    parser.add_argument(
        "--transport",
        type=str,
        choices=[
            "stdio",
            "streamable-http"
        ],
        default="stdio",
        help="MCP 传输方式，默认 stdio"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR"
        ],
        default="INFO",
        help="日志级别，日志输出到 stderr"
    )

    return parser.parse_args()


# ============================================================
# 6. 主启动函数
# ============================================================

def main() -> int:
    """
    MCP Server 主入口。

    默认启动 stdio 传输。

    注意：
        stdio 模式启动后不会提供普通命令行交互界面。
        MCP Client 需要通过标准输入输出发送 JSON-RPC 请求。
    """

    args = parse_arguments()

    try:

        # ----------------------------------------------------
        # 配置日志
        # ----------------------------------------------------

        configure_logging(
            log_level=args.log_level
        )

        logger.info(
            "=" * 65
        )

        logger.info(
            "Starting Data Privacy Analyst MCP Server."
        )

        logger.info(
            "Transport: %s",
            args.transport
        )

        # ----------------------------------------------------
        # 创建 Server
        # ----------------------------------------------------

        mcp = create_server(
            graph_data_path=args.graph_data,
            graph_schema_path=args.graph_schema,
            tool_contract_path=args.tool_contract,
            nist_records_path=args.nist_data,
            chroma_directory=args.chroma_directory,
            collection_name=args.collection_name
        )

        # ----------------------------------------------------
        # 启动 MCP 传输
        # ----------------------------------------------------

        logger.info(
            "Starting MCP transport: %s",
            args.transport
        )

        # 所有 print 和 logging 均不应该写入 stdout。
        # FastMCP 负责标准 MCP 协议读写。
        mcp.run(
            transport=args.transport
        )

        logger.info(
            "MCP Server stopped normally."
        )

        return 0

    except KeyboardInterrupt:

        logger.info(
            "MCP Server interrupted by user."
        )

        return 0

    except Exception as exc:

        # 启动错误输出到 stderr。
        #
        # 不使用 print，防止破坏 stdio 协议。
        logger.exception(
            "MCP Server startup or runtime failed: %s",
            exc
        )

        return 1


# ============================================================
# 7. 程序入口
# ============================================================

if __name__ == "__main__":

    sys.exit(main())
