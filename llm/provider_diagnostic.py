
"""
provider_diagnostic.py

Gemini API 网络与工具调用诊断。

测试内容：
1. DNS 解析。
2. TCP 443 连接。
3. Gemini Models API 鉴权。
4. Gemini Chat Completions。
5. Gemini Function Calling。

运行方式：

    python -m llm.provider_diagnostic

只检查网络，不调用模型：

    python -m llm.provider_diagnostic --network-only

只检查 API，不测试 Function Calling：

    python -m llm.provider_diagnostic --skip-tool-test

注意：
不会打印 API Key。
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

from urllib.parse import quote

import httpx
import requests

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
)

load_dotenv(
    PROJECT_ROOT / ".env"
)

DEFAULT_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)

DEFAULT_MODEL = "gemini-3.8-flash"


def get_config() -> tuple[str, str, str]:
    """
    读取 Gemini 配置。
    """

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("LLM_API_KEY")
    )

    base_url = os.getenv(
        "LLM_BASE_URL",
        DEFAULT_BASE_URL
    )

    model = os.getenv(
        "LLM_MODEL",
        DEFAULT_MODEL
    )

    if not api_key:

        raise ValueError(
            "缺少 GEMINI_API_KEY，请检查项目根目录 .env"
        )

    if "generativelanguage.googleapis.com" not in base_url:

        raise ValueError(
            "LLM_BASE_URL 不是 Gemini Developer API Endpoint。\n"
            f"当前值：{base_url}"
        )

    return api_key, base_url.rstrip("/") + "/", model


def check_dns() -> bool:
    """
    检查 Gemini API 域名能否解析。
    """

    hostname = "generativelanguage.googleapis.com"

    print("\n[1] DNS 检查")

    try:

        addresses = socket.getaddrinfo(
            hostname,
            443,
            type=socket.SOCK_STREAM
        )

        unique_addresses = sorted({
            item[4][0]
            for item in addresses
        })

        print("[成功] DNS 解析成功")

        for address in unique_addresses:

            print(f"  IP: {address}")

        return True

    except OSError as exc:

        print("[失败] DNS 解析失败")

        print(
            f"  {type(exc).__name__}: {exc}"
        )

        return False


def check_tcp() -> bool:
    """
    检查能否连接 Google Gemini API 的 443 端口。
    """

    hostname = "generativelanguage.googleapis.com"

    print("\n[2] TCP 连接检查")

    start = time.perf_counter()

    try:

        with socket.create_connection(
            (hostname, 443),
            timeout=10
        ):

            elapsed = (
                time.perf_counter() - start
            )

            print("[成功] TCP 连接成功")

            print(
                f"  连接耗时：{elapsed:.2f}s"
            )

        return True

    except OSError as exc:

        print("[失败] TCP 连接失败")

        print(
            f"  {type(exc).__name__}: {exc}"
        )

        return False


def check_proxy_configuration() -> None:
    """
    检查 WSL 是否存在 HTTP/S 代理环境变量。

    不输出代理地址，避免意外泄露代理凭证。
    """

    print("\n[3] HTTP Proxy 环境检查")

    keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy"
    ]

    configured = False

    for key in keys:

        if os.getenv(key):

            print(f"  {key}: configured")

            configured = True

        else:

            print(f"  {key}: not configured")

    if not configured:

        print(
            "  未检测到显式 HTTP/S 代理。"
            "这不代表 Proton VPN 未生效，"
            "因为 VPN 也可能通过系统网络路由工作。"
        )


def check_model_endpoint(
    api_key: str,
    base_url: str,
    model: str
) -> bool:
    """
    使用 Gemini Models Endpoint 检查 API 鉴权和连通性。
    """

    print("\n[4] Gemini Models API 检查")

    endpoint = (
        base_url
        + "models/"
        + quote(model, safe="-_.")
    )

    start = time.perf_counter()

    try:

        response = requests.get(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            timeout=(15, 30)
        )

        elapsed = (
            time.perf_counter() - start
        )

        print(
            f"  HTTP Status: {response.status_code}"
        )

        print(
            f"  Response Time: {elapsed:.2f}s"
        )

        if response.status_code == 200:

            print("[成功] 模型 API 和鉴权正常")

            return True

        if response.status_code == 401:

            print(
                "[失败] API Key 无效或认证失败"
            )

        elif response.status_code == 403:

            print(
                "[失败] 请求被拒绝。"
                "请检查 API 项目资格、地区支持及权限。"
            )

        elif response.status_code == 404:

            print(
                "[失败] API Endpoint 或 Model ID 不存在。"
            )

        elif response.status_code == 429:

            print(
                "[失败] Gemini Rate Limit。"
                "请检查 RPM、TPM 和每日配额。"
            )

        elif response.status_code >= 500:

            print(
                "[失败] Google API 服务端异常。"
            )

        else:

            print(
                "[失败] API 返回非预期状态。"
            )

        # 不打印完整响应正文，避免输出敏感信息。
        return False

    except requests.Timeout:

        print(
            "[失败] Models API 请求超时。"
        )

        return False

    except requests.RequestException as exc:

        print(
            f"[失败] 网络请求错误：{type(exc).__name__}"
        )

        return False


def check_chat_completion(
    api_key: str,
    base_url: str,
    model: str,
    connect_timeout: int = 15,
    read_timeout: int = 90
) -> bool:
    """
    测试最简单的 Gemini Chat Completion。

    如果该测试也超时：
        优先检查网络、地区和 API Endpoint。
    """

    print("\n[5] Gemini Chat Completion 检查")

    try:

        timeout = httpx.Timeout(
            timeout=read_timeout,
            connect=connect_timeout,
            read=read_timeout,
            write=30,
            pool=15
        )

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0
        )

        start = time.perf_counter()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly: GEMINI_OK"
                    )
                }
            ],
            reasoning_effort="low",
            max_tokens=64
        )

        elapsed = (
            time.perf_counter() - start
        )

        content = (
            response.choices[0].message.content
            or ""
        )

        print("[成功] Chat Completion 正常返回")

        print(
            f"  Model: {response.model}"
        )

        print(
            f"  耗时：{elapsed:.2f}s"
        )

        print(
            f"  Response: {content[:100]}"
        )

        return True

    except Exception as exc:

        print("[失败] Chat Completion 请求失败")

        print(
            f"  Error Type: {type(exc).__name__}"
        )

        print(
            f"  Error: {str(exc)[:500]}"
        )

        return False


def check_function_calling(
    api_key: str,
    base_url: str,
    model: str
) -> bool:
    """
    使用一个最简单的 Function Schema 测试 Gemini Tool Calling。

    本测试不启动 MCP Server。
    只验证 Gemini API 是否支持当前 OpenAI-compatible
    Function Calling 请求。
    """

    print("\n[6] Gemini Function Calling 检查")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "probe_ping",
                "description": (
                    "A simple connectivity test function. "
                    "Call it with message='ok'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "message"
                    ],
                    "additionalProperties": False
                }
            }
        }
    ]

    try:

        timeout = httpx.Timeout(
            timeout=120,
            connect=15,
            read=120
        )

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0
        )

        start = time.perf_counter()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You must call probe_ping exactly once "
                        "with message set to 'ok'."
                    )
                }
            ],
            tools=tools,
            tool_choice="auto",
            reasoning_effort="low",
            max_tokens=128
        )

        elapsed = (
            time.perf_counter() - start
        )

        message = response.choices[0].message

        tool_calls = (
            message.tool_calls or []
        )

        print("[成功] Function Calling API 已返回")

        print(
            f"  Response Time: {elapsed:.2f}s"
        )

        print(
            f"  Tool Calls: {len(tool_calls)}"
        )

        if tool_calls:

            for call in tool_calls:

                print(
                    f"  Tool: {call.function.name}"
                )

                print(
                    f"  Arguments: {call.function.arguments}"
                )

        else:

            print(
                "  模型没有返回 Tool Call。"
                "请检查其返回内容及 Finish Reason。"
            )

            print(
                f"  Finish Reason: "
                f"{response.choices[0].finish_reason}"
            )

        return True

    except Exception as exc:

        print(
            f"[失败] Function Calling 请求失败："
            f"{type(exc).__name__}"
        )

        print(
            str(exc)[:500]
        )

        return False


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Gemini API Connectivity Diagnostic"
    )

    parser.add_argument(
        "--network-only",
        action="store_true"
    )

    parser.add_argument(
        "--skip-tool-test",
        action="store_true"
    )

    args = parser.parse_args()

    print("=" * 65)

    print("Gemini API Diagnostic")

    print("=" * 65)

    try:

        api_key, base_url, model = get_config()

    except Exception as exc:

        print(
            f"[配置错误] {exc}"
        )

        return 1

    print(
        f"Model: {model}"
    )

    print(
        f"Endpoint: {base_url}"
    )

    print(
        "API Key: Configured"
    )

    check_proxy_configuration()

    dns_ok = check_dns()

    tcp_ok = check_tcp()

    if not dns_ok or not tcp_ok:

        print(
            "\n[结论] 当前 WSL 无法稳定访问 Gemini API 网络入口。"
        )

        return 1

    if args.network_only:

        print(
            "\n[结论] DNS 和 TCP 网络检查通过。"
        )

        return 0

    endpoint_ok = check_model_endpoint(
        api_key,
        base_url,
        model
    )

    if not endpoint_ok:

        print(
            "\n[结论] Gemini 模型端点或权限检查未通过。"
        )

        return 1

    chat_ok = check_chat_completion(
        api_key,
        base_url,
        model
    )

    if not chat_ok:

        print(
            "\n[结论] 网络和鉴权正常，但模型生成请求失败。"
        )

        return 1

    if not args.skip_tool_test:

        tool_ok = check_function_calling(
            api_key,
            base_url,
            model
        )

        if not tool_ok:

            return 1

    print("\n" + "=" * 65)

    print("[成功] Gemini API 诊断完成。")

    print("=" * 65)

    return 0


if __name__ == "__main__":

    sys.exit(main())
