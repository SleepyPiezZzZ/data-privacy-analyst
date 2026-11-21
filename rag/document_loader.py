
"""
document_loader.py

NIST Privacy Framework 1.0 Core 数据导入模块。

核心功能：
1. 下载 NIST 官方 Core XLSX。
2. 读取 Excel 工作簿。
3. 解析 Function、Category、Subcategory 三层结构。
4. 处理合并单元格造成的空白值。
5. 提取官方 Subcategory ID 和原始声明。
6. 生成标准化 nist_privacy_core.json。
7. 提供统一的知识库 JSON 加载接口。

数据来源：
    https://www.nist.gov/privacy-framework/privacy-framework

官方 XLSX：
    https://www.nist.gov/document/nist-privacy-framework-v10-core

生成文件：
    data/nist_privacy_core.json

运行方式：
    python -m rag.document_loader

指定本地 XLSX：
    python -m rag.document_loader \
        --source-xlsx data/source/nist_privacy_framework.xlsx

允许非标准数量的记录（仅用于调试）：
    python -m rag.document_loader --allow-count-mismatch
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ============================================================
# 1. 路径及资源配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRECTORY = PROJECT_ROOT / "data"

SOURCE_DIRECTORY = DATA_DIRECTORY / "source"

DEFAULT_SOURCE_XLSX = (
    SOURCE_DIRECTORY / "nist-privacy-framework-v10-core.xlsx"
)

DEFAULT_OUTPUT_JSON = (
    DATA_DIRECTORY / "nist_privacy_core.json"
)

OFFICIAL_FRAMEWORK_PAGE = (
    "https://www.nist.gov/privacy-framework/privacy-framework"
)

OFFICIAL_CORE_XLSX_URL = (
    "https://www.nist.gov/document/nist-privacy-framework-v10-core"
)

OFFICIAL_CORE_PDF_URL = (
    "https://www.nist.gov/document/nist-privacy-framework-version-1-core-pdf"
)

FRAMEWORK_NAME = "NIST Privacy Framework"

FRAMEWORK_VERSION = "1.0"

EXPECTED_SUBCATEGORY_COUNT = 100


# ============================================================
# 2. Function 标准名称
# ============================================================

FUNCTION_NAMES = {
    "IDENTIFY-P": "Identify-P",
    "GOVERN-P": "Govern-P",
    "CONTROL-P": "Control-P",
    "COMMUNICATE-P": "Communicate-P",
    "PROTECT-P": "Protect-P"
}


# 当 Excel 的 Function 单元格为空时，
# 可以根据 Category ID 的前缀恢复 Function。
CATEGORY_FUNCTION_MAP = {
    "ID": "IDENTIFY-P",
    "GV": "GOVERN-P",
    "CT": "CONTROL-P",
    "CM": "COMMUNICATE-P",
    "PR": "PROTECT-P"
}


# ============================================================
# 3. 正则表达式
# ============================================================

FUNCTION_PATTERN = re.compile(
    r"\b(IDENTIFY-P|GOVERN-P|CONTROL-P|"
    r"COMMUNICATE-P|PROTECT-P)\b",
    re.IGNORECASE
)

CATEGORY_PATTERN = re.compile(
    r"\b([A-Z]{2}\.[A-Z]{2}-P)\b",
    re.IGNORECASE
)

SUBCATEGORY_PATTERN = re.compile(
    r"\b([A-Z]{2}\.[A-Z]{2}-P\d+)"
    r"\s*:\s*(.*)",
    re.IGNORECASE | re.DOTALL
)


# ============================================================
# 4. 通用文本处理
# ============================================================

def normalize_text(value: Any) -> str:
    """
    将 Excel 单元格转换为标准字符串。

    处理：
    1. None。
    2. 数值。
    3. 单元格换行。
    4. 多余空格。
    """

    if value is None:
        return ""

    text = str(value)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_header(value: Any) -> str:
    """
    标准化 Excel 表头。

    例如：
        Function -> function
        Sub-Category -> subcategory
    """

    text = normalize_text(value).lower()

    return re.sub(
        r"[^a-z]",
        "",
        text
    )


def extract_function(
    text: str
) -> tuple[str | None, str]:
    """
    从 Function 单元格提取 Function Code 和描述。

    示例：
        IDENTIFY-P (ID-P): Develop the organizational...

    返回：
        IDENTIFY-P
        Develop the organizational...
    """

    match = FUNCTION_PATTERN.search(text)

    if not match:
        return None, ""

    function_code = match.group(1).upper()

    description = text[match.end():].strip()

    # 移除例如 (ID-P) 这样的辅助缩写。
    description = re.sub(
        r"^\([^)]*\)\s*",
        "",
        description
    )

    description = description.lstrip(":：- ").strip()

    return function_code, description


def extract_category(
    text: str
) -> tuple[str | None, str, str]:
    """
    提取 Category Code、名称和描述。

    示例：
        Inventory and Mapping (ID.IM-P):
        Data processing by systems...
    """

    match = CATEGORY_PATTERN.search(text)

    if not match:
        return None, "", ""

    category_code = match.group(1).upper()

    category_name = (
        text[:match.start()]
        .strip()
        .rstrip(":： ")
    )

    category_description = (
        text[match.end():]
        .strip()
        .lstrip(":：- ")
    )

    return (
        category_code,
        category_name,
        category_description
    )


def extract_subcategory(
    text: str
) -> tuple[str | None, str]:
    """
    提取 Subcategory Code 和官方原文。

    示例：
        ID.IM-P5: The purposes for the data actions are inventoried.

    返回：
        ID.IM-P5
        The purposes for the data actions are inventoried.
    """

    match = SUBCATEGORY_PATTERN.search(text)

    if not match:
        return None, ""

    subcategory_code = match.group(1).upper()

    statement = normalize_text(
        match.group(2)
    )

    return subcategory_code, statement


def generate_title(statement: str) -> str:
    """
    根据官方 Subcategory Statement 生成短显示标题。

    注意：
    NIST Core 的 Subcategory 以 outcome statement 为核心，
    这里生成的 title 是本项目的展示字段，
    不是声称 NIST 原始文件提供了独立标题。
    """

    title = statement.strip()

    title = re.sub(
        r"^the\s+",
        "",
        title,
        flags=re.IGNORECASE
    )

    title = title.rstrip(" .;")

    if title:
        title = title[0].upper() + title[1:]

    # 防止异常长文本影响界面展示。
    if len(title) > 120:
        title = title[:117].rstrip() + "..."

    return title


# ============================================================
# 5. 下载官方 XLSX
# ============================================================

def download_official_core_xlsx(
    output_path: str | Path
) -> Path:
    """
    从 NIST 官方下载 Privacy Framework 1.0 Core XLSX。

    下载时验证 ZIP 文件头，避免把 HTML 错误页面
    错误地保存为 XLSX。

    Args:
        output_path:
            XLSX 保存位置。

    Returns:
        Path:
            下载后的文件路径。
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    headers = {
        "User-Agent": "DataPrivacyAnalyst/1.0",
        "Accept": (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet,application/octet-stream,*/*"
        )
    }

    print("正在下载 NIST Privacy Framework 1.0 Core XLSX...")

    try:

        with requests.get(
            OFFICIAL_CORE_XLSX_URL,
            headers=headers,
            timeout=(15, 120),
            stream=True
        ) as response:

            response.raise_for_status()

            first_chunk = None

            with temporary_path.open("wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    if first_chunk is None:
                        first_chunk = chunk[:4]

                    file.write(chunk)

            # XLSX 本质上是 ZIP 文件。
            if first_chunk != b"PK\x03\x04":

                raise ValueError(
                    "下载结果不是有效的 XLSX/ZIP 文件。"
                    "可能是 NIST 返回了 HTML 页面或下载被拦截。"
                )

        temporary_path.replace(output_path)

    except Exception:

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    print(f"下载完成：{output_path}")

    return output_path


# ============================================================
# 6. Excel 表头识别
# ============================================================

def find_header_mapping(
    rows: list[list[Any]]
) -> tuple[int, dict[str, int]]:
    """
    自动识别 Function、Category、Subcategory 所在列。

    不依赖固定 Sheet 名称或固定列位置。

    返回：
        header_row_index
        {
            "function": column_index,
            "category": column_index,
            "subcategory": column_index
        }
    """

    aliases = {
        "function": {
            "function",
            "functions"
        },
        "category": {
            "category",
            "categories"
        },
        "subcategory": {
            "subcategory",
            "subcategories"
        }
    }

    for row_index, row in enumerate(rows[:40]):

        header_map: dict[str, int] = {}

        for column_index, value in enumerate(row):

            header = normalize_header(value)

            for field_name, candidates in aliases.items():

                if header in candidates:

                    header_map[field_name] = column_index

        if len(header_map) == 3:

            return row_index, header_map

    raise ValueError(
        "无法识别 XLSX 工作表中的 Function、Category、"
        "Subcategory 三列。\n"
        "请确认下载的是 NIST Privacy Framework 1.0 Core XLSX。"
    )


# ============================================================
# 7. 单个工作表解析
# ============================================================

def parse_core_sheet(
    rows: list[list[Any]]
) -> list[dict[str, Any]]:
    """
    解析单个 NIST Core 工作表。

    主要处理：

    1. 向下填充 Function。
    2. 向下填充 Category。
    3. 根据编码提取层级关系。
    4. 提取全部 Subcategory。
    5. 使用标准化字段生成记录。
    """

    header_row_index, header_mapping = (
        find_header_mapping(rows)
    )

    current_function_code = None
    current_function_description = ""

    current_category_code = None
    current_category_name = ""
    current_category_description = ""

    records: list[dict[str, Any]] = []

    seen_subcategory_codes: set[str] = set()

    for row_index in range(
        header_row_index + 1,
        len(rows)
    ):

        row = rows[row_index]

        # 防止不完整行产生索引错误。
        def get_cell(field_name: str) -> str:

            column_index = header_mapping[field_name]

            if column_index >= len(row):
                return ""

            return normalize_text(
                row[column_index]
            )

        function_raw = get_cell("function")

        category_raw = get_cell("category")

        subcategory_raw = get_cell("subcategory")

        # ----------------------------------------------------
        # 更新 Function
        # ----------------------------------------------------

        if function_raw:

            parsed_code, parsed_description = (
                extract_function(function_raw)
            )

            if parsed_code:

                current_function_code = parsed_code

                current_function_description = (
                    parsed_description
                )

        # ----------------------------------------------------
        # 更新 Category
        # ----------------------------------------------------

        if category_raw:

            (
                parsed_category_code,
                parsed_category_name,
                parsed_category_description
            ) = extract_category(category_raw)

            if parsed_category_code:

                current_category_code = (
                    parsed_category_code
                )

                current_category_name = (
                    parsed_category_name
                )

                current_category_description = (
                    parsed_category_description
                )

        # ----------------------------------------------------
        # 解析 Subcategory
        # ----------------------------------------------------

        if not subcategory_raw:
            continue

        (
            subcategory_code,
            statement
        ) = extract_subcategory(subcategory_raw)

        # 不是标准 Subcategory 行则跳过。
        if not subcategory_code:
            continue

        if not statement:

            raise ValueError(
                f"Subcategory {subcategory_code} 缺少官方声明。"
            )

        # ----------------------------------------------------
        # 唯一性检查
        # ----------------------------------------------------

        if subcategory_code in seen_subcategory_codes:

            raise ValueError(
                f"发现重复的 Subcategory ID："
                f"{subcategory_code}"
            )

        seen_subcategory_codes.add(
            subcategory_code
        )

        # ----------------------------------------------------
        # 从 Subcategory ID 提取 Category ID
        # ----------------------------------------------------

        category_prefix_match = re.match(
            r"([A-Z]{2}\.[A-Z]{2}-P)\d+$",
            subcategory_code
        )

        if not category_prefix_match:

            raise ValueError(
                f"无法从 Subcategory ID 提取 Category："
                f"{subcategory_code}"
            )

        derived_category_code = (
            category_prefix_match.group(1)
        )

        # 如果 Category 单元格合并或为空，
        # 可以根据 Subcategory ID 恢复 Category ID。
        if current_category_code is None:

            current_category_code = (
                derived_category_code
            )

        if (
            current_category_code
            != derived_category_code
        ):

            raise ValueError(
                f"层级不匹配：{subcategory_code} "
                f"应属于 {derived_category_code}，"
                f"但当前 Category 为 {current_category_code}"
            )

        # ----------------------------------------------------
        # 恢复 Function
        # ----------------------------------------------------

        prefix = subcategory_code[:2]

        derived_function_code = (
            CATEGORY_FUNCTION_MAP.get(prefix)
        )

        if derived_function_code is None:

            raise ValueError(
                f"未知的 NIST Function 前缀：{prefix}"
            )

        if current_function_code is None:

            current_function_code = (
                derived_function_code
            )

        if (
            current_function_code
            != derived_function_code
        ):

            raise ValueError(
                f"Function 层级不匹配："
                f"{subcategory_code} 应属于 "
                f"{derived_function_code}，"
                f"但当前 Function 为 "
                f"{current_function_code}"
            )

        # ----------------------------------------------------
        # 生成标准记录
        # ----------------------------------------------------

        document_id = (
            f"NIST-PF-{FRAMEWORK_VERSION}-"
            f"{subcategory_code}"
        )

        record = {
            "document_id": document_id,
            "framework_name": FRAMEWORK_NAME,
            "framework_version": FRAMEWORK_VERSION,
            "function_code": current_function_code,
            "function_name": FUNCTION_NAMES[
                current_function_code
            ],
            "function_description": (
                current_function_description
            ),
            "category_code": current_category_code,
            "category_name": (
                current_category_name
                or current_category_code
            ),
            "category_description": (
                current_category_description
            ),
            "subcategory_code": subcategory_code,
            "title": generate_title(statement),
            "statement": statement,
            "source_title": "NIST Privacy Framework Core",
            "source_url": OFFICIAL_FRAMEWORK_PAGE,
            "source_document_url": OFFICIAL_CORE_PDF_URL
        }

        records.append(record)

    # 最终按照官方 Subcategory ID 排序。
    records.sort(
        key=lambda record: (
            record["function_code"],
            record["category_code"],
            record["subcategory_code"]
        )
    )

    return records


# ============================================================
# 8. Excel 工作簿读取
# ============================================================

def extract_nist_records(
    source_xlsx: str | Path
) -> list[dict[str, Any]]:
    """
    从 NIST XLSX 中提取 Subcategory 数据。

    使用 python-calamine 读取，不对源 Excel 进行修改。
    """

    source_xlsx = Path(source_xlsx)

    if not source_xlsx.exists():

        raise FileNotFoundError(
            f"NIST XLSX 文件不存在：{source_xlsx}"
        )

    if source_xlsx.stat().st_size == 0:

        raise ValueError(
            "NIST XLSX 文件为空。"
        )

    try:

        from python_calamine import CalamineWorkbook

    except ImportError as exc:

        raise ImportError(
            "缺少 python-calamine，请执行："
            "pip install python-calamine"
        ) from exc

    workbook = CalamineWorkbook.from_path(
        str(source_xlsx)
    )

    sheet_names = workbook.sheet_names

    if not sheet_names:

        raise ValueError(
            "NIST XLSX 中没有可读取的工作表。"
        )

    parsing_errors = []

    # 不依赖固定工作表名称，
    # 自动寻找包含三层数据结构的工作表。
    for sheet_name in sheet_names:

        try:

            sheet = workbook.get_sheet_by_name(
                sheet_name
            )

            rows = sheet.to_python(
                skip_empty_area=False
            )

            records = parse_core_sheet(rows)

            if records:

                print(f"成功解析工作表：{sheet_name}")

                return records

        except Exception as exc:

            parsing_errors.append(
                f"{sheet_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    raise ValueError(
        "无法从 NIST XLSX 中解析有效 Subcategory 数据。\n"
        "工作表解析信息：\n"
        + "\n".join(parsing_errors)
    )


# ============================================================
# 9. 构建知识库 JSON
# ============================================================

def build_core_dataset(
    source_xlsx: str | Path,
    expected_count: int = EXPECTED_SUBCATEGORY_COUNT,
    allow_count_mismatch: bool = False
) -> dict[str, Any]:
    """
    构建完整的 NIST Core 标准化数据集。

    Args:
        source_xlsx:
            官方 XLSX 路径。

        expected_count:
            预期 Subcategory 数量，默认 100。

        allow_count_mismatch:
            是否允许数量不一致。

    Returns:
        dict:
            完整的标准化知识库数据。
    """

    records = extract_nist_records(
        source_xlsx
    )

    record_count = len(records)

    if (
        expected_count > 0
        and record_count != expected_count
        and not allow_count_mismatch
    ):

        raise ValueError(
            f"Subcategory 数量异常："
            f"预期 {expected_count} 条，"
            f"实际提取 {record_count} 条。\n"
            "为保证知识库完整性，本次拒绝生成文件。"
            "\n请检查 XLSX 来源、工作表和解析结果。"
        )

    # 每个 Function 的记录数。
    function_counts: dict[str, int] = {}

    for record in records:

        code = record["function_code"]

        function_counts[code] = (
            function_counts.get(code, 0) + 1
        )

    dataset = {
        "dataset_id": "NIST-PF-CORE-1.0",
        "framework_name": FRAMEWORK_NAME,
        "framework_version": FRAMEWORK_VERSION,
        "source_title": "NIST Privacy Framework Core",
        "source_url": OFFICIAL_FRAMEWORK_PAGE,
        "source_file_url": OFFICIAL_CORE_XLSX_URL,
        "source_format": "xlsx",
        "source_file": str(
            Path(source_xlsx).resolve()
        ),
        "retrieved_at": (
            datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        ),
        "record_count": record_count,
        "function_counts": function_counts,
        "records": records
    }

    return dataset


# ============================================================
# 10. 保存知识库 JSON
# ============================================================

def save_core_dataset(
    dataset: dict[str, Any],
    output_path: str | Path
) -> Path:
    """
    将知识库 JSON 保存到磁盘。

    使用临时文件加 rename，
    避免程序中断造成原始文件损坏。
    """

    output_path = Path(output_path)

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

        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2
        )

    temporary_path.replace(output_path)

    return output_path


# ============================================================
# 11. 知识库加载接口
# ============================================================

def load_core_dataset(
    file_path: str | Path = DEFAULT_OUTPUT_JSON
) -> dict[str, Any]:
    """
    加载已经生成的 NIST Core JSON。

    后续 embedding_service 和 policy_retriever
    通过该方法使用同一份标准化知识源。
    """

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"NIST JSON 知识库不存在：{file_path}\n"
            "请先运行：python -m rag.document_loader"
        )

    try:

        with file_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            dataset = json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            f"NIST JSON 格式错误：{exc}"
        ) from exc

    if not isinstance(dataset, dict):

        raise ValueError(
            "NIST JSON 根对象必须为 Object。"
        )

    required_dataset_keys = {
        "dataset_id",
        "framework_name",
        "framework_version",
        "record_count",
        "records"
    }

    missing_keys = (
        required_dataset_keys
        - set(dataset.keys())
    )

    if missing_keys:

        raise ValueError(
            f"NIST 数据集缺少字段：{sorted(missing_keys)}"
        )

    if dataset["framework_version"] != "1.0":

        raise ValueError(
            "当前项目只允许 NIST Privacy Framework 1.0。"
        )

    if not isinstance(dataset["records"], list):

        raise ValueError(
            "records 必须是数组。"
        )

    if dataset["record_count"] != len(dataset["records"]):

        raise ValueError(
            "record_count 与 records 实际数量不一致。"
        )

    seen_ids = set()

    required_record_keys = {
        "document_id",
        "framework_name",
        "framework_version",
        "function_code",
        "function_name",
        "function_description",
        "category_code",
        "category_name",
        "category_description",
        "subcategory_code",
        "title",
        "statement",
        "source_title",
        "source_url",
        "source_document_url"
    }

    for record in dataset["records"]:

        if not isinstance(record, dict):

            raise ValueError(
                "records 中的每个元素必须是 Object。"
            )

        missing_record_keys = (
            required_record_keys
            - set(record.keys())
        )

        if missing_record_keys:

            raise ValueError(
                "NIST 记录缺少字段："
                f"{sorted(missing_record_keys)}"
            )

        document_id = record["document_id"]

        if document_id in seen_ids:

            raise ValueError(
                f"发现重复 document_id：{document_id}"
            )

        seen_ids.add(document_id)

        if not record["statement"].strip():

            raise ValueError(
                f"NIST 记录为空：{document_id}"
            )

    return dataset


# ============================================================
# 12. 命令行入口
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description="NIST Privacy Framework 1.0 Core Loader"
    )

    parser.add_argument(
        "--source-xlsx",
        type=str,
        default=None,
        help="指定本地 NIST Core XLSX"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="生成 JSON 的保存位置"
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_SUBCATEGORY_COUNT,
        help="预期 Subcategory 数量"
    )

    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="调试时允许记录数量与预期不一致"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("NIST Privacy Framework 1.0 Core Loader")
    print("=" * 65)

    try:

        if args.source_xlsx:

            source_xlsx = Path(
                args.source_xlsx
            )

        elif DEFAULT_SOURCE_XLSX.exists():

            # 已经下载过则优先复用本地文件。
            source_xlsx = DEFAULT_SOURCE_XLSX

        else:

            source_xlsx = download_official_core_xlsx(
                DEFAULT_SOURCE_XLSX
            )

        dataset = build_core_dataset(
            source_xlsx=source_xlsx,
            expected_count=args.expected_count,
            allow_count_mismatch=args.allow_count_mismatch
        )

        output_path = save_core_dataset(
            dataset=dataset,
            output_path=args.output
        )

        # 保存后重新读取，验证实际文件内容。
        verified_dataset = load_core_dataset(
            output_path
        )

        print("\n[成功] NIST 知识库生成完成。")

        print(
            f"  Framework：{verified_dataset['framework_name']}"
        )

        print(
            f"  Version：{verified_dataset['framework_version']}"
        )

        print(
            f"  Subcategory 总数："
            f"{verified_dataset['record_count']}"
        )

        print("\nFunction 分布：")

        for function_code, count in sorted(
            verified_dataset["function_counts"].items()
        ):

            print(
                f"  {function_code}: {count}"
            )

        print(f"\nJSON 路径：{output_path}")

        print("\nNIST 数据导入完成。")

        return 0

    except Exception as exc:

        print("\n[失败] NIST 数据导入失败：")

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "\n若自动下载失败，请从以下官方页面下载"
            " V1.0 Core (XLSX)："
        )

        print(OFFICIAL_FRAMEWORK_PAGE)

        return 1


if __name__ == "__main__":
    sys.exit(main())
