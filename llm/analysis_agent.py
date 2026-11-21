
"""
analysis_agent.py

Data Privacy Analyst - LLM Analysis Agent。

核心功能：

1. 加载 Groq API 配置。
2. 启动并连接阶段四 MCP Server。
3. 动态发现 MCP Tools。
4. 将 MCP Tools 转换为 OpenAI Function Calling 格式。
5. 运行 LLM Tool Calling Loop。
6. 通过 MCP Client 执行工具调用。
7. 确保三个必要 MCP Tools 均执行成功。
8. 根据 changed_field_ids 补充查询受影响节点详情。
9. 构建可信事实和证据目录。
10. 使用 Groq Strict Structured Outputs 生成最终报告。
11. 调用 output_validator.py 验证报告。
12. 对验证失败的报告进行有限次数修复。

运行方式：

    python -m llm.analysis_agent \
        --source-node-id T01_CUSTOMER_RAW \
        --change-type purpose_change \
        --requested-purpose-code marketing_personalization \
        --changed-field-ids full_name,phone_number \
        --change-description "申请将客户姓名和电话号码用于营销个性化推荐"

指定输出文件：

    python -m llm.analysis_agent \
        --source-node-id T01_CUSTOMER_RAW \
        --change-type purpose_change \
        --requested-purpose-code marketing_personalization \
        --changed-field-ids full_name,phone_number \
        --change-description "申请将客户姓名和电话号码用于营销个性化推荐" \
        --output reports/privacy_impact_report.json

环境变量：

    GROQ_API_KEY
    LLM_API_KEY
    LLM_BASE_URL
    LLM_MODEL
    LLM_TIMEOUT
    LLM_MAX_TOOL_ROUNDS
    LLM_MAX_TOOL_CALLS
    LLM_MAX_VALIDATION_RETRIES

默认模型：

    openai/gpt-oss-20b

默认 MCP 传输：

    stdio
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from jsonschema import Draft202012Validator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import AsyncOpenAI
from openai import BadRequestError
from openai import APIError

from graph.graph_loader import (
    DEFAULT_DATA_PATH,
    DEFAULT_SCHEMA_PATH
)

from rag.document_loader import DEFAULT_OUTPUT_JSON

from llm.prompt_templates import (
    SYSTEM_TOOL_PROMPT,
    SYSTEM_REPORT_PROMPT,
    build_initial_user_message,
    build_report_user_message,
    build_validation_repair_message
)

from llm.output_validator import (
    derive_impact_descriptors,
    build_trusted_facts,
    load_report_schema,
    build_provider_strict_schema,
    canonicalize_report,
    validate_report,
    ReportValidationError
)


# ============================================================
# 1. 项目路径及环境配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TOOL_CONTRACT_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "mcp_tool_schemas.json"
)

DEFAULT_CHROMA_DIRECTORY = (
    PROJECT_ROOT
    / "vector_store"
    / "chroma"
)

DEFAULT_COLLECTION_NAME = (
    "nist_privacy_framework_v1"
)

DEFAULT_LLM_BASE_URL = (
    "https://api.groq.com/openai/v1"
)

DEFAULT_LLM_MODEL = (
    "openai/gpt-oss-20b"
)

DEFAULT_LLM_TIMEOUT = 120

DEFAULT_MAX_TOOL_ROUNDS = 6

DEFAULT_MAX_TOOL_CALLS = 12

DEFAULT_MAX_VALIDATION_RETRIES = 2

MAX_TOOL_RESULT_LENGTH = 100_000

EXPECTED_TOOL_NAMES = {
    "get_downstream_nodes",
    "get_node_details",
    "match_privacy_policy"
}

SUPPORTED_CHANGE_TYPES = {
    "purpose_change",
    "field_change",
    "schema_change",
    "access_change",
    "new_downstream_consumer",
    "unspecified"
}


# 加载项目根目录下的 .env。
load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# 2. 日志配置
# ============================================================

logger = logging.getLogger(__name__)


def configure_logging(
    log_level: str = "INFO"
) -> None:
    """
    所有日志输出至 stderr。

    本模块作为 MCP Client 时，
    日志不能影响其他标准输入输出通信。
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


# ============================================================
# 3. Analysis Context 构建
# ============================================================

def create_analysis_context(
    source_node_id: str,
    change_type: str,
    change_description: str,
    requested_purpose_code: str | None = None,
    changed_field_ids: list[str] | None = None,
    request_id: str | None = None
) -> dict[str, Any]:
    """
    根据用户输入构建标准化分析上下文。

    字段严格遵守 Report Schema：

        request_id
        source_node_id
        change_type
        change_description
        requested_purpose_code
        changed_field_ids
    """

    if not source_node_id.strip():

        raise ValueError(
            "source_node_id 不能为空。"
        )

    if change_type not in SUPPORTED_CHANGE_TYPES:

        raise ValueError(
            f"不支持的 change_type：{change_type}"
        )

    if not change_description.strip():

        raise ValueError(
            "change_description 不能为空。"
        )

    if (
        change_type == "purpose_change"
        and not requested_purpose_code
    ):

        raise ValueError(
            "purpose_change 必须指定 requested_purpose_code。"
        )

    changed_field_ids = (
        changed_field_ids or []
    )

    # 去重且保持顺序。
    changed_field_ids = list(
        dict.fromkeys(changed_field_ids)
    )

    if any(
        not isinstance(field_id, str)
        or not field_id.strip()
        for field_id in changed_field_ids
    ):

        raise ValueError(
            "changed_field_ids 中不能包含空字段 ID。"
        )

    if request_id is None:

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d")

        request_id = (
            f"REQ-{timestamp}-"
            f"{uuid.uuid4().hex[:8].upper()}"
        )

    return {
        "request_id": request_id,
        "source_node_id": source_node_id,
        "change_type": change_type,
        "change_description": change_description,
        "requested_purpose_code": requested_purpose_code,
        "changed_field_ids": changed_field_ids
    }


# ============================================================
# 4. MCP Server 子进程环境
# ============================================================

def build_mcp_environment() -> dict[str, str]:
    """
    为 MCP Server 子进程构建必要环境变量。

    不向 MCP 子进程传递 GROQ_API_KEY 或 LLM_API_KEY。

    MCP Server 只需要：
        Python 环境
        项目根目录
        本地模型及 Chroma 路径相关环境
    """

    allowed_environment_keys = [
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "APPDATA",
        "PROGRAMDATA",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "HF_HOME",
        "SENTENCE_TRANSFORMERS_HOME",
        "TRANSFORMERS_CACHE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR"
    ]

    child_env = {
        key: os.environ[key]
        for key in allowed_environment_keys
        if key in os.environ
    }

    current_pythonpath = os.environ.get(
        "PYTHONPATH",
        ""
    )

    python_paths = [
        str(PROJECT_ROOT)
    ]

    if current_pythonpath:

        python_paths.append(
            current_pythonpath
        )

    child_env["PYTHONPATH"] = (
        os.pathsep.join(python_paths)
    )

    child_env["PYTHONUNBUFFERED"] = "1"

    return child_env


def build_mcp_server_parameters() -> StdioServerParameters:
    """
    构建阶段四 MCP Server 启动参数。

    使用当前 Python 解释器，
    避免 MCP Client 调用到系统中其他 Python 环境。
    """

    arguments = [
        "-m",
        "mcp_server.server",
        "--transport",
        "stdio",
        "--log-level",
        "WARNING",
        "--graph-data",
        str(DEFAULT_DATA_PATH),
        "--graph-schema",
        str(DEFAULT_SCHEMA_PATH),
        "--tool-contract",
        str(DEFAULT_TOOL_CONTRACT_PATH),
        "--nist-data",
        str(DEFAULT_OUTPUT_JSON),
        "--chroma-directory",
        str(DEFAULT_CHROMA_DIRECTORY),
        "--collection-name",
        DEFAULT_COLLECTION_NAME
    ]

    return StdioServerParameters(
        command=sys.executable,
        args=arguments,
        env=build_mcp_environment()
    )


# ============================================================
# 5. MCP 工具发现
# ============================================================

def get_tool_input_schema(
    tool: Any
) -> dict[str, Any]:
    """
    获取 MCP Tool 的 inputSchema。

    MCP Python SDK 使用 inputSchema 字段。
    同时兼容可能出现的 Python snake_case 属性形式。
    """

    input_schema = getattr(
        tool,
        "inputSchema",
        None
    )

    if input_schema is None:

        input_schema = getattr(
            tool,
            "input_schema",
            None
        )

    if not isinstance(input_schema, dict):

        raise ValueError(
            f"MCP Tool {tool.name} 缺少有效 inputSchema。"
        )

    return input_schema


def build_openai_tool_definitions(
    mcp_tools: list[Any]
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]]
]:
    """
    将 MCP Tools 转换成 OpenAI Function Calling Schema。

    输入：
        MCP ListToolsResult.tools

    输出：
        openai_tools
        tool_schemas

    OpenAI 工具格式：

        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }
    """

    openai_tools = []

    tool_schemas = {}

    for tool in mcp_tools:

        name = tool.name

        if name not in EXPECTED_TOOL_NAMES:

            raise ValueError(
                f"MCP Server 注册了未定义工具：{name}"
            )

        input_schema = get_tool_input_schema(
            tool
        )

        # 检查 Tool Input Schema 自身是否合法。
        Draft202012Validator.check_schema(
            input_schema
        )

        tool_schemas[name] = input_schema

        openai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (
                    tool.description
                    or f"MCP Tool: {name}"
                ),
                "parameters": input_schema
            }
        })

    actual_names = set(
        tool_schemas.keys()
    )

    if actual_names != EXPECTED_TOOL_NAMES:

        raise ValueError(
            "MCP Server 工具集合与项目契约不一致。\n"
            f"预期：{sorted(EXPECTED_TOOL_NAMES)}\n"
            f"实际：{sorted(actual_names)}"
        )

    return openai_tools, tool_schemas


# ============================================================
# 6. MCP Tool Result 解码
# ============================================================

def decode_mcp_result(
    result: Any
) -> dict[str, Any]:
    """
    将 MCP CallToolResult 解析成 Python JSON 字典。

    优先读取 structuredContent。
    如果没有结构化内容，则尝试读取 TextContent。

    注意：
    必须得到字典，否则认为工具调用失败。
    """

    is_error = getattr(
        result,
        "isError",
        getattr(result, "is_error", False)
    )

    if is_error:

        error_texts = []

        for block in getattr(
            result,
            "content",
            []
        ):

            text = getattr(
                block,
                "text",
                None
            )

            if text:
                error_texts.append(text)

        raise RuntimeError(
            "MCP Tool 执行失败："
            + " | ".join(error_texts)
        )

    # --------------------------------------------------------
    # 首选 Structured Content
    # --------------------------------------------------------

    structured_content = getattr(
        result,
        "structuredContent",
        None
    )

    if structured_content is None:

        structured_content = getattr(
            result,
            "structured_content",
            None
        )

    if structured_content is not None:

        if isinstance(
            structured_content,
            dict
        ):

            return structured_content

        if hasattr(
            structured_content,
            "model_dump"
        ):

            return structured_content.model_dump(
                by_alias=True
            )

    # --------------------------------------------------------
    # 兼容 TextContent
    # --------------------------------------------------------

    text_parts = []

    for block in getattr(
        result,
        "content",
        []
    ):

        text = getattr(
            block,
            "text",
            None
        )

        if text:
            text_parts.append(text)

    if text_parts:

        raw_text = "\n".join(
            text_parts
        )

        try:

            data = json.loads(
                raw_text
            )

            if isinstance(data, dict):

                return data

        except json.JSONDecodeError:

            pass

    raise RuntimeError(
        "MCP Tool 返回值不是有效的 JSON Object。"
    )


# ============================================================
# 7. 工具结果存储
# ============================================================

class MCPToolResultStore:
    """
    保存一次分析任务中所有 MCP Tool 的成功结果。

    节点详情：
        node_details_by_id[node_id]

    下游查询：
        downstream_result

    政策检索：
        policy_result

    不修改 MCP 原始输出结构。
    """

    def __init__(self) -> None:

        self.node_details_by_id: dict[
            str,
            dict[str, Any]
        ] = {}

        self.downstream_result: dict[str, Any] | None = None

        self.policy_result: dict[str, Any] | None = None

        self.executed_tool_names: set[str] = set()

        self.tool_call_count = 0

    def record(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any]
    ) -> None:
        """
        保存一次 MCP 工具调用结果。
        """

        if tool_name == "get_node_details":

            node_id = arguments["node_id"]

            if (
                result.get("is_found")
                and result.get("node") is not None
            ):

                self.node_details_by_id[
                    node_id
                ] = result

        elif tool_name == "get_downstream_nodes":

            self.downstream_result = result

        elif tool_name == "match_privacy_policy":

            self.policy_result = result

        self.executed_tool_names.add(
            tool_name
        )

        self.tool_call_count += 1


# ============================================================
# 8. Agent 主类
# ============================================================

class PrivacyImpactAnalysisAgent:
    """
    数据血缘与隐私影响分析 LLM Agent。

    整合：
        1. Groq / OpenAI-compatible API。
        2. MCP Client。
        3. 三个 MCP Tools。
        4. Prompt Templates。
        5. Trusted Facts。
        6. Structured Output。
        7. Output Validator。

    对外主要方法：

        analyze()
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_LLM_BASE_URL,
        model: str = DEFAULT_LLM_MODEL,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_validation_retries: int = DEFAULT_MAX_VALIDATION_RETRIES,
        timeout: int = DEFAULT_LLM_TIMEOUT
    ) -> None:
        """
        初始化 Agent。

        Args:
            api_key:
                LLM Provider API Key。

            base_url:
                OpenAI-compatible API 地址。

            model:
                LLM Model ID。

            max_tool_rounds:
                最大模型工具调用轮次。

            max_tool_calls:
                最大总 MCP Tool 调用次数。

            max_validation_retries:
                最大报告验证重试次数。

            timeout:
                LLM 请求超时。
        """

        if not api_key:

            raise ValueError(
                "LLM API Key 不能为空。"
            )

        self.api_key = api_key

        self.base_url = base_url

        self.model = model

        self.max_tool_rounds = max_tool_rounds

        self.max_tool_calls = max_tool_calls

        self.max_validation_retries = max_validation_retries

        self.timeout = timeout

        self.tool_results = MCPToolResultStore()

        self.output_mode = "strict_json_schema"

        # 报告生成使用的报告元数据。
        self.report_id = None

        self.generated_at = None

    # --------------------------------------------------------
    # 8.1 MCP Tool 实际调用
    # --------------------------------------------------------

    async def _call_mcp_tool(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: dict[str, Any],
        tool_schemas: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        调用实际 MCP Server 中的 Tool。

        这里不会直接调用 Graph Service 或 RAG Service。

        所有工具执行必须经过 MCP Client。
        """

        if tool_name not in EXPECTED_TOOL_NAMES:

            raise ValueError(
                f"Agent 不允许执行未注册的工具：{tool_name}"
            )

        if self.tool_results.tool_call_count >= self.max_tool_calls:

            raise RuntimeError(
                "已达到 MAX_TOOL_CALLS 限制。"
            )

        if not isinstance(arguments, dict):

            raise TypeError(
                "MCP Tool arguments 必须是 JSON Object。"
            )

        # ----------------------------------------------------
        # 根据 MCP Tool Input Schema 验证参数
        # ----------------------------------------------------

        input_schema = tool_schemas[
            tool_name
        ]

        validator = Draft202012Validator(
            input_schema
        )

        validation_errors = list(
            validator.iter_errors(arguments)
        )

        if validation_errors:

            raise ValueError(
                "MCP Tool 输入参数未通过 Schema 验证："
                + "; ".join(
                    error.message
                    for error in validation_errors
                )
            )

        # ----------------------------------------------------
        # 执行 MCP 调用
        # ----------------------------------------------------

        logger.info(
            "Calling MCP Tool: %s",
            tool_name
        )

        logger.debug(
            "Arguments: %s",
            json.dumps(
                arguments,
                ensure_ascii=False
            )
        )

        result = await session.call_tool(
            tool_name,
            arguments=arguments
        )

        decoded_result = decode_mcp_result(
            result
        )

        # ----------------------------------------------------
        # 保存 MCP 返回结果
        # ----------------------------------------------------

        self.tool_results.record(
            tool_name=tool_name,
            arguments=arguments,
            result=decoded_result
        )

        logger.info(
            "MCP Tool completed: %s",
            tool_name
        )

        return decoded_result

    # --------------------------------------------------------
    # 8.2 获取 MCP Tool Result 的聊天消息
    # --------------------------------------------------------

    @staticmethod
    def _tool_result_message(
        tool_call_id: str,
        tool_name: str,
        result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        将 MCP 结果转换成 OpenAI Chat Completion 的 Tool Message。
        """

        result_text = json.dumps(
            result,
            ensure_ascii=False
        )

        if len(result_text) > MAX_TOOL_RESULT_LENGTH:

            raise ValueError(
                f"MCP Tool {tool_name} 返回内容超过允许长度。"
            )

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result_text
        }

    # --------------------------------------------------------
    # 8.3 运行 LLM Tool Calling Loop
    # --------------------------------------------------------

    async def _run_tool_calling_loop(
        self,
        llm_client: AsyncOpenAI,
        session: ClientSession,
        openai_tools: list[dict[str, Any]],
        tool_schemas: dict[str, dict[str, Any]],
        analysis_context: dict[str, Any]
    ) -> None:
        """
        运行 LLM 自主工具调用循环。

        工作过程：

            User Request
                ↓
            Groq Chat Completion
                ↓
            tool_calls
                ↓
            MCP Server
                ↓
            Tool Results
                ↓
            Groq Chat Completion

        直到模型不再产生 tool_calls，
        或达到最大循环次数。

        Groq Strict Structured Output 不在此阶段启用。
        """

        user_request = (
            analysis_context["change_description"]
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_TOOL_PROMPT
            },
            {
                "role": "user",
                "content": build_initial_user_message(
                    user_request=user_request,
                    analysis_context=analysis_context
                )
            }
        ]

        logger.info(
            "Starting LLM Tool Calling Loop."
        )

        for round_index in range(
            self.max_tool_rounds
        ):

            logger.info(
                "LLM tool round: %d/%d",
                round_index + 1,
                self.max_tool_rounds
            )

            completion = (
                await llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    max_completion_tokens=4000,
                    extra_body={"enable_thinking": True}
                )
            )

            choice = completion.choices[0]

            assistant_message = choice.message

            tool_calls = (
                assistant_message.tool_calls
                or []
            )

            # ------------------------------------------------
            # 模型不再要求调用工具
            # ------------------------------------------------

            if not tool_calls:

                if choice.finish_reason == "length":

                    raise RuntimeError(
                        "LLM Tool Calling 输出达到 Token 长度限制。"
                    )

                logger.info(
                    "LLM Tool Calling Loop completed."
                )

                break

            # ------------------------------------------------
            # 保存 Assistant Tool Call Message
            # ------------------------------------------------

            assistant_tool_message = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": []
            }

            for tool_call in tool_calls:

                assistant_tool_message[
                    "tool_calls"
                ].append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": (
                            tool_call.function.arguments
                        )
                    }
                })

            messages.append(
                assistant_tool_message
            )

            # ------------------------------------------------
            # 逐个执行 MCP Tool
            # ------------------------------------------------

            for tool_call in tool_calls:

                if self.tool_results.tool_call_count >= self.max_tool_calls:

                    raise RuntimeError(
                        "Agent 达到最大 MCP Tool 调用次数。"
                    )

                tool_name = (
                    tool_call.function.name
                )

                arguments_text = (
                    tool_call.function.arguments
                )

                tool_call_id = (
                    tool_call.id
                )

                try:

                    # 解析 Function Arguments。
                    arguments = json.loads(
                        arguments_text
                    )

                    if not isinstance(arguments, dict):

                        raise ValueError(
                            "Function Arguments 必须是 JSON Object。"
                        )

                    # 执行 MCP Server 工具。
                    result = await self._call_mcp_tool(
                        session=session,
                        tool_name=tool_name,
                        arguments=arguments,
                        tool_schemas=tool_schemas
                    )

                    tool_message = (
                        self._tool_result_message(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            result=result
                        )
                    )

                except Exception as exc:

                    logger.warning(
                        "MCP Tool execution failed: %s",
                        tool_name
                    )

                    # 通过 Tool Message 将错误返回 LLM，
                    # 允许模型重新组织参数或尝试其他查询。
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps({
                            "error": str(exc),
                            "tool_name": tool_name
                        }, ensure_ascii=False)
                    }

                messages.append(
                    tool_message
                )

        logger.info(
            "Completed model-selected tool calls: %s",
            sorted(self.tool_results.executed_tool_names)
        )

    # --------------------------------------------------------
    # 8.4 确保必要 MCP Tool 执行成功
    # --------------------------------------------------------

    async def _ensure_required_tool_results(
        self,
        session: ClientSession,
        tool_schemas: dict[str, dict[str, Any]],
        analysis_context: dict[str, Any]
    ) -> None:
        """
        确保三个核心工具都成功返回有效结果。

        如果模型没有主动调用某项必要工具，
        则程序使用已知的分析上下文通过 MCP Client 补充调用。

        这是可靠性保护措施，不会直接访问底层 Graph 或 ChromaDB。
        """

        source_node_id = (
            analysis_context["source_node_id"]
        )

        # ----------------------------------------------------
        # 1. 源节点详情
        # ----------------------------------------------------

        source_details = (
            self.tool_results.node_details_by_id.get(
                source_node_id
            )
        )

        if (
            source_details is None
            or not source_details.get("is_found")
        ):

            logger.info(
                "补充执行 get_node_details：%s",
                source_node_id
            )

            result = await self._call_mcp_tool(
                session=session,
                tool_name="get_node_details",
                arguments={
                    "node_id": source_node_id,
                    "include_fields": True,
                    "include_related_edges": True
                },
                tool_schemas=tool_schemas
            )

            if not result.get("is_found"):

                raise ValueError(
                    f"无法找到源节点：{source_node_id}"
                )

        # ----------------------------------------------------
        # 2. 下游数据血缘
        # ----------------------------------------------------

        downstream_result = (
            self.tool_results.downstream_result
        )

        if (
            downstream_result is None
            or not downstream_result.get("is_found")
            or downstream_result.get("query_node_id")
            != source_node_id
        ):

            logger.info(
                "补充执行 get_downstream_nodes：%s",
                source_node_id
            )

            downstream_result = await self._call_mcp_tool(
                session=session,
                tool_name="get_downstream_nodes",
                arguments={
                    "node_id": source_node_id,
                    "max_depth": 10,
                    "include_paths": True,
                    "include_field_mapping": True,
                    "max_nodes": 100
                },
                tool_schemas=tool_schemas
            )

            if not downstream_result.get("is_found"):

                raise ValueError(
                    "无法获取源节点的下游血缘信息。"
                )

        # ----------------------------------------------------
        # 3. NIST 政策检索
        # ----------------------------------------------------

        if self.tool_results.policy_result is None:

            description = (
                analysis_context["change_description"]
            )

            purpose_code = (
                analysis_context["requested_purpose_code"]
                or ""
            )

            keyword = (
                f"{description} "
                f"{purpose_code} "
                "privacy purpose data actions "
                "data elements data processing mapping"
            )

            logger.info(
                "补充执行 match_privacy_policy。"
            )

            await self._call_mcp_tool(
                session=session,
                tool_name="match_privacy_policy",
                arguments={
                    "keyword": keyword,
                    "framework_version": "1.0",
                    "top_k": 5,
                    "function_filter": [],
                    "include_full_text": True
                },
                tool_schemas=tool_schemas
            )

        if self.tool_results.policy_result is None:

            raise RuntimeError(
                "match_privacy_policy 没有成功执行。"
            )

    # --------------------------------------------------------
    # 8.5 补充查询实际受影响的节点
    # --------------------------------------------------------

    async def _enrich_impacted_node_details(
        self,
        session: ClientSession,
        tool_schemas: dict[str, dict[str, Any]],
        analysis_context: dict[str, Any]
    ) -> None:
        """
        根据真实字段传播结果，
        获取所有实际受影响下游节点的完整字段详情。

        get_downstream_nodes 只返回节点汇总，
        为了让最终报告中的 affected_fields
        拥有真实 field_name 和 sensitivity_label，
        需要再调用 get_node_details。

        每次调用都必须经过 MCP。
        """

        source_node_id = (
            analysis_context["source_node_id"]
        )

        source_details = (
            self.tool_results.node_details_by_id[
                source_node_id
            ]
        )

        downstream_result = (
            self.tool_results.downstream_result
        )

        descriptors = derive_impact_descriptors(
            analysis_context=analysis_context,
            source_details_result=source_details,
            downstream_result=downstream_result
        )

        impacted_node_ids = sorted(
            descriptors.keys()
        )

        logger.info(
            "Field-level affected nodes: %s",
            impacted_node_ids
        )

        for node_id in impacted_node_ids:

            if node_id in self.tool_results.node_details_by_id:

                continue

            if self.tool_results.tool_call_count >= self.max_tool_calls:

                raise RuntimeError(
                    "补充节点详情查询时达到最大 MCP Tool 调用次数。"
                )

            logger.info(
                "Enriching impacted node: %s",
                node_id
            )

            result = await self._call_mcp_tool(
                session=session,
                tool_name="get_node_details",
                arguments={
                    "node_id": node_id,
                    "include_fields": True,
                    "include_related_edges": False
                },
                tool_schemas=tool_schemas
            )

            if not result.get("is_found"):

                raise ValueError(
                    f"受影响节点不存在：{node_id}"
                )

    # --------------------------------------------------------
    # 8.6 最终 Structured Output 生成
    # --------------------------------------------------------

    async def _generate_structured_report(
        self,
        llm_client: AsyncOpenAI,
        analysis_context: dict[str, Any],
        trusted_facts: dict[str, Any],
        schema_path: str | Path
    ) -> dict[str, Any]:
        """
        调用模型生成最终 Privacy Impact Report。

        流程：

            1. 加载原始 Report Schema。
            2. 转换为 Groq Strict Schema。
            3. 使用 json_schema + strict=true。
            4. 解析 JSON。
            5. 执行本地完整验证。
            6. 失败时要求 LLM 修复。
            7. 最多重试指定次数。

        Structured Output 请求与 Tool Calling 分开执行。
        """

        # ----------------------------------------------------
        # 第一步：加载 Schema
        # ----------------------------------------------------

        original_schema = load_report_schema(
            schema_path
        )

        strict_schema = build_provider_strict_schema(
            schema_path
        )

        # ----------------------------------------------------
        # 第二步：构建独立报告生成消息
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": SYSTEM_REPORT_PROMPT
            },
            {
                "role": "user",
                "content": build_report_user_message(
                    trusted_facts
                )
            }
        ]

        # ----------------------------------------------------
        # 第三步：Strict Structured Output 参数
        # ----------------------------------------------------

        strict_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "privacy_impact_report",
                "description": (
                    "Evidence-grounded Privacy Impact Analysis Report"
                ),
                "strict": True,
                "schema": strict_schema
            }
        }

        # ----------------------------------------------------
        # 第四步：生成与校验
        # ----------------------------------------------------

        fallback_to_json_object = False

        last_validation_errors: list[str] = []

        total_attempts = (
            self.max_validation_retries + 1
        )

        for attempt in range(
            total_attempts
        ):

            logger.info(
                "Generating Structured Report: %d/%d",
                attempt + 1,
                total_attempts
            )

            response_format = (
                {"type": "json_object"}
                if fallback_to_json_object
                else strict_response_format
            )

            try:

                completion = (
                    await llm_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format=response_format,
                        max_completion_tokens=32768,
                        extra_body={"enable_thinking": True}
                    )
                )

            except BadRequestError as exc:

                # 如果 Provider 不接受 Strict JSON Schema，
                # 降级为 JSON Object Mode。
                #
                # 本地 JSON Schema 和语义校验仍然是强制执行的。
                if not fallback_to_json_object:

                    logger.warning(
                        "Strict JSON Schema request rejected. "
                        "Fallback to JSON Object Mode: %s",
                        exc
                    )

                    fallback_to_json_object = True

                    self.output_mode = "json_object_fallback"

                    # 本次请求没有成功生成报告，
                    # 不消耗 validation retry 计数。
                    continue

                raise

            # ------------------------------------------------
            # 第五步：读取模型响应
            # ------------------------------------------------

            choice = completion.choices[0]

            message = choice.message

            if getattr(message, "refusal", None):

                raise RuntimeError(
                    "模型拒绝生成报告："
                    f"{message.refusal}"
                )

            report_text = message.content

            # --- 新增：记录 Token 消耗与结束原因诊断日志 ---
            safe_report_text = report_text or ""
            usage = completion.usage
            completion_tokens = (
                usage.completion_tokens if usage else None
            )
            completion_details = (
                getattr(usage, "completion_tokens_details", None)
                if usage else None
            )
            reasoning_tokens = (
                getattr(completion_details, "reasoning_tokens", None)
                if completion_details else None
            )

            logger.info(
                "[Structured Report] finish_reason=%s, "
                "completion_tokens=%s, reasoning_tokens=%s, "
                "output_chars=%d",
                choice.finish_reason,
                completion_tokens,
                reasoning_tokens,
                len(safe_report_text)
            )

            if choice.finish_reason == "length":
                logger.error(
                    "Structured report generation reached the token limit. "
                    "Current output tail: %r",
                    safe_report_text[-300:]
                )
            # ------------------------------------------------

            if not report_text:


                last_validation_errors = [
                    "LLM 没有返回有效的 JSON 内容。"
                ]

            else:

                try:

                    report = json.loads(
                        report_text
                    )

                    if not isinstance(report, dict):

                        raise ValueError(
                            "模型输出根对象不是 JSON Object。"
                        )

                    # ----------------------------------------------------
                    # 1. 将程序确定的事实回填到 LLM 报告
                    # ----------------------------------------------------

                    report = canonicalize_report(
                        report=report,
                        trusted_facts=trusted_facts
                    )

                    logger.info(
                        "LLM report canonicalization completed."
                    )

                    # ----------------------------------------------------
                    # 2. 执行完整本地验证
                    # ----------------------------------------------------

                    validate_report(
                        report=report,
                        trusted_facts=trusted_facts,
                        schema_path=schema_path
                    )

                    logger.info(
                        "Privacy Impact Report validation passed."
                    )

                    logger.info(
                        "Structured Output Mode: %s",
                        self.output_mode
                    )

                    return report

                except json.JSONDecodeError as exc:

                    start = max(0, exc.pos - 150)
                    end = min(len(report_text), exc.pos + 150)

                    logger.error(
                        "JSON parsing failed: line=%d, column=%d, "
                        "position=%d, output_length=%d",
                        exc.lineno,
                        exc.colno,
                        exc.pos,
                        len(report_text)
                    )

                    logger.error(
                        "Invalid JSON context: %r",
                        report_text[start:end]
                    )

                    logger.error(
                        "JSON output tail: %r",
                        report_text[-500:]
                    )                    

                    last_validation_errors = [
                        f"JSON 解析失败：{exc}"
                    ]

                except ReportValidationError as exc:

                    last_validation_errors = (
                        exc.errors
                    )

                except ValueError as exc:

                    last_validation_errors = [
                        str(exc)
                    ]

            # ------------------------------------------------
            # 第六步：失败修复
            # ------------------------------------------------

            if attempt < total_attempts - 1:

                logger.warning(
                    "Report validation failed. "
                    "Requesting model repair."
                )

                if report_text:

                    messages.append({
                        "role": "assistant",
                        "content": report_text
                    })

                messages.append({
                    "role": "user",
                    "content": build_validation_repair_message(
                        validation_errors=(
                            last_validation_errors
                        ),
                        trusted_facts=trusted_facts
                    )
                })

        raise ReportValidationError(
            last_validation_errors
        )

    # --------------------------------------------------------
    # 8.7 完整分析接口
    # --------------------------------------------------------

    async def analyze(
        self,
        analysis_context: dict[str, Any],
        schema_path: str | Path
    ) -> dict[str, Any]:
        """
        执行一份完整的隐私影响分析。

        Args:
            analysis_context:
                用户变更请求上下文。

            schema_path:
                Report JSON Schema 路径。

        Returns:
            dict:
                完整且通过验证的 Privacy Impact Report。
        """

        # ----------------------------------------------------
        # 初始化运行状态
        # ----------------------------------------------------

        self.tool_results = MCPToolResultStore()

        self.report_id = (
            f"RPT-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d')}-"
            f"{uuid.uuid4().hex[:8].upper()}"
        )

        self.generated_at = (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

        # ----------------------------------------------------
        # 初始化 OpenAI-compatible Client
        # ----------------------------------------------------

        async with AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=2
        ) as llm_client:

            # ------------------------------------------------
            # 建立 MCP stdio 连接
            # ------------------------------------------------

            server_parameters = (
                build_mcp_server_parameters()
            )

            logger.info(
                "Starting MCP Server subprocess."
            )

            async with stdio_client(
                server_parameters
            ) as (
                read_stream,
                write_stream
            ):

                async with ClientSession(
                    read_stream,
                    write_stream
                ) as session:

                    # ----------------------------------------
                    # MCP 初始化
                    # ----------------------------------------

                    await session.initialize()

                    logger.info(
                        "MCP Client session initialized."
                    )

                    # ----------------------------------------
                    # 获取 Tool Registry
                    # ----------------------------------------

                    tools_response = (
                        await session.list_tools()
                    )

                    openai_tools, tool_schemas = (
                        build_openai_tool_definitions(
                            tools_response.tools
                        )
                    )

                    logger.info(
                        "Discovered %d MCP Tools.",
                        len(openai_tools)
                    )

                    # ----------------------------------------
                    # 执行模型自主 Tool Calling
                    # ----------------------------------------

                    await self._run_tool_calling_loop(
                        llm_client=llm_client,
                        session=session,
                        openai_tools=openai_tools,
                        tool_schemas=tool_schemas,
                        analysis_context=analysis_context
                    )

                    # ----------------------------------------
                    # 确保三个核心工具已经执行
                    # ----------------------------------------

                    await self._ensure_required_tool_results(
                        session=session,
                        tool_schemas=tool_schemas,
                        analysis_context=analysis_context
                    )

                    # ----------------------------------------
                    # 获取下游节点字段详情
                    # ----------------------------------------

                    await self._enrich_impacted_node_details(
                        session=session,
                        tool_schemas=tool_schemas,
                        analysis_context=analysis_context
                    )

                    # ----------------------------------------
                    # 构建可信事实
                    # ----------------------------------------

                    trusted_facts = build_trusted_facts(
                        analysis_context=analysis_context,
                        source_details_result=(
                            self.tool_results.node_details_by_id[
                                analysis_context["source_node_id"]
                            ]
                        ),
                        downstream_result=(
                            self.tool_results.downstream_result
                        ),
                        node_details_by_id=(
                            self.tool_results.node_details_by_id
                        ),
                        policy_result=(
                            self.tool_results.policy_result
                        ),
                        report_id=self.report_id,
                        generated_at=self.generated_at
                    )

                    logger.info(
                        "Trusted analysis facts constructed."
                    )

                    logger.info(
                        "Expected impacted nodes: %d",
                        len(trusted_facts[
                            "expected_impacted_nodes"
                        ])
                    )

                    logger.info(
                        "Trusted evidence count: %d",
                        len(trusted_facts[
                            "evidence_catalog"
                        ])
                    )

                    logger.info(
                        "Policy reference count: %d",
                        len(trusted_facts[
                            "compliance_catalog"
                        ])
                    )

                    # ----------------------------------------
                    # Structured Output 报告生成
                    # ----------------------------------------

                    report = (
                        await self._generate_structured_report(
                            llm_client=llm_client,
                            analysis_context=analysis_context,
                            trusted_facts=trusted_facts,
                            schema_path=schema_path
                        )
                    )

                    logger.info(
                        "Privacy Impact Analysis completed."
                    )

                    return report


# ============================================================
# 9. 命令行参数
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """
    解析 Agent 命令行参数。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Data Lineage & Privacy Impact Analysis Agent"
        )
    )

    parser.add_argument(
        "--source-node-id",
        type=str,
        required=True,
        help="源数据节点 ID"
    )

    parser.add_argument(
        "--change-type",
        type=str,
        required=True,
        choices=sorted(
            SUPPORTED_CHANGE_TYPES
        ),
        help="数据变更类型"
    )

    parser.add_argument(
        "--change-description",
        type=str,
        required=True,
        help="数据变更的自然语言描述"
    )

    parser.add_argument(
        "--requested-purpose-code",
        type=str,
        default=None,
        help="申请使用的目的代码"
    )

    parser.add_argument(
        "--changed-field-ids",
        type=str,
        default="",
        help=(
            "需要分析的字段 ID，"
            "逗号分隔；留空表示分析整个源节点"
        )
    )

    parser.add_argument(
        "--request-id",
        type=str,
        default=None,
        help="自定义分析请求 ID"
    )

    parser.add_argument(
        "--schema",
        type=str,
        default=str(
            PROJECT_ROOT
            / "schemas"
            / "privacy_impact_report.schema.json"
        ),
        help="Privacy Impact Report Schema 路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选的 JSON 报告保存路径"
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
        help="日志级别"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv(
            "LLM_MODEL",
            DEFAULT_LLM_MODEL
        ),
        help="LLM Model ID"
    )

    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=int(
            os.getenv(
                "LLM_MAX_TOOL_ROUNDS",
                DEFAULT_MAX_TOOL_ROUNDS
            )
        ),
        help="最大模型 Tool Calling 轮次"
    )

    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=int(
            os.getenv(
                "LLM_MAX_TOOL_CALLS",
                DEFAULT_MAX_TOOL_CALLS
            )
        ),
        help="最大 MCP Tool 调用次数"
    )

    parser.add_argument(
        "--max-validation-retries",
        type=int,
        default=int(
            os.getenv(
                "LLM_MAX_VALIDATION_RETRIES",
                DEFAULT_MAX_VALIDATION_RETRIES
            )
        ),
        help="输出验证失败后的最大修复次数"
    )

    return parser.parse_args()


# ============================================================
# 10. CLI 主函数
# ============================================================

async def async_main(
    args: argparse.Namespace
) -> int:
    """
    异步命令行执行入口。
    """

    configure_logging(
        args.log_level
    )

    # --------------------------------------------------------
    # 获取 API Key
    # --------------------------------------------------------

    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )

    if not api_key:

        raise ValueError(
            "没有检测到 LLM API Key。\n"
            "请在项目根目录 .env 中配置 GROQ_API_KEY。"
        )

    base_url = os.getenv(
        "LLM_BASE_URL",
        DEFAULT_LLM_BASE_URL
    )

    timeout = int(
        os.getenv(
            "LLM_TIMEOUT",
            DEFAULT_LLM_TIMEOUT
        )
    )

    # --------------------------------------------------------
    # 解析 changed_field_ids
    # --------------------------------------------------------

    changed_field_ids = [
        field_id.strip()
        for field_id in args.changed_field_ids.split(",")
        if field_id.strip()
    ]

    # --------------------------------------------------------
    # 构建分析上下文
    # --------------------------------------------------------

    analysis_context = create_analysis_context(
        source_node_id=args.source_node_id,
        change_type=args.change_type,
        change_description=args.change_description,
        requested_purpose_code=(
            args.requested_purpose_code
        ),
        changed_field_ids=changed_field_ids,
        request_id=args.request_id
    )

    # --------------------------------------------------------
    # 初始化 Agent
    # --------------------------------------------------------

    agent = PrivacyImpactAnalysisAgent(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        max_tool_rounds=args.max_tool_rounds,
        max_tool_calls=args.max_tool_calls,
        max_validation_retries=(
            args.max_validation_retries
        ),
        timeout=timeout
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Data Privacy Analyst - Phase 5"
    )

    logger.info(
        "Model: %s",
        args.model
    )

    logger.info(
        "Source Node: %s",
        args.source_node_id
    )

    logger.info(
        "Change Type: %s",
        args.change_type
    )

    logger.info(
        "Requested Purpose: %s",
        args.requested_purpose_code
    )

    logger.info(
        "Changed Fields: %s",
        changed_field_ids
    )

    logger.info(
        "=" * 70
    )

    # --------------------------------------------------------
    # 执行完整分析
    # --------------------------------------------------------

    report = await agent.analyze(
        analysis_context=analysis_context,
        schema_path=args.schema
    )

    # --------------------------------------------------------
    # 序列化最终报告
    # --------------------------------------------------------

    report_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    )

    print("\n" + "=" * 70)

    print(
        "Privacy Impact Analysis Report"
    )

    print("=" * 70)

    print(report_text)

    print("=" * 70)

    # --------------------------------------------------------
    # 可选：保存 JSON 报告
    # --------------------------------------------------------

    if args.output:

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                report_text
            )

        temporary_path.replace(
            output_path
        )

        logger.info(
            "Report saved: %s",
            output_path.resolve()
        )

    logger.info(
        "Final risk level: %s",
        report["risk_level"]
    )

    logger.info(
        "Final decision: %s",
        report["decision"]
    )

    return 0


# ============================================================
# 11. 程序入口
# ============================================================

def main() -> int:
    """
    同步 CLI 入口。
    """

    parser_args = parse_arguments()

    try:

        return asyncio.run(
            async_main(parser_args)
        )

    except KeyboardInterrupt:

        logger.warning(
            "Analysis interrupted by user."
        )

        return 130

    except (
        ValueError,
        TypeError,
        FileNotFoundError,
        RuntimeError,
        APIError,
        ReportValidationError
    ) as exc:

        logger.error(
            "Analysis failed: %s",
            exc
        )

        return 1

    except Exception as exc:

        logger.exception(
            "Unexpected Agent error."
        )

        return 1


if __name__ == "__main__":

    sys.exit(main())
