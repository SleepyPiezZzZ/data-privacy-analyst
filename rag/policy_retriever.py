
"""
policy_retriever.py

NIST Privacy Framework 混合 RAG 检索模块。

核心功能：
1. 加载标准化 NIST Core JSON。
2. 构建 BM25 稀疏检索索引。
3. 使用 SentenceTransformer 生成查询向量。
4. 使用 ChromaDB 执行 Dense Retrieval。
5. 使用 Reciprocal Rank Fusion 融合检索结果。
6. 支持中英文查询。
7. 支持 Function 过滤。
8. 返回 match_privacy_policy MCP 工具规定的 JSON 结构。

MCP 工具：
    match_privacy_policy(
        keyword,
        framework_version,
        top_k,
        function_filter,
        include_full_text
    )

运行方式：
    python -m rag.policy_retriever \
        --query "检查数据使用目的是否已经登记"

英文查询：
    python -m rag.policy_retriever \
        --query "Are data processing purposes inventoried?"

指定 Function：
    python -m rag.policy_retriever \
        --query "数据处理目的登记" \
        --function-filter IDENTIFY-P

重建索引：
    python -m rag.policy_retriever \
        --query "数据使用目的" \
        --rebuild-index
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from rank_bm25 import BM25Okapi

from rag.document_loader import (
    DEFAULT_OUTPUT_JSON,
    load_core_dataset
)

from rag.embedding_service import EmbeddingService

from rag.vector_store import ChromaVectorStore


# ============================================================
# 1. 项目路径及检索配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHROMA_DIRECTORY = (
    PROJECT_ROOT / "vector_store" / "chroma"
)

DEFAULT_COLLECTION_NAME = (
    "nist_privacy_framework_v1"
)

SUPPORTED_FRAMEWORK_VERSION = "1.0"

SUPPORTED_FUNCTIONS = {
    "IDENTIFY-P",
    "GOVERN-P",
    "CONTROL-P",
    "COMMUNICATE-P",
    "PROTECT-P"
}

DEFAULT_TOP_K = 5

DEFAULT_BM25_CANDIDATE_COUNT = 30

DEFAULT_RRF_K = 60

DOCUMENT_EMBEDDING_BATCH_SIZE = 32


# ============================================================
# 2. 中英文领域词典
# ============================================================

# 这里不使用 LLM 翻译。
#
# 目的是将常见中文查询表达映射为 NIST Core
# 中实际使用的英文关键词。
#
# 词典是可扩展的，但不会修改 NIST 原始知识内容。

QUERY_SYNONYMS = {
    "使用目的": [
        "purpose",
        "purposes",
        "data actions",
        "processing purpose"
    ],
    "数据用途": [
        "purpose",
        "purposes",
        "data uses"
    ],
    "数据处理": [
        "data processing",
        "data actions",
        "processing"
    ],
    "数据流转": [
        "data processing mapping",
        "data processing",
        "mapped",
        "mapping"
    ],
    "数据血缘": [
        "data processing mapping",
        "data elements",
        "data flow"
    ],
    "字段": [
        "data elements",
        "data element"
    ],
    "个人信息": [
        "individuals",
        "personal data",
        "privacy"
    ],
    "敏感信息": [
        "data sensitivity",
        "sensitive data",
        "privacy risk"
    ],
    "第三方": [
        "third parties",
        "data processing ecosystem",
        "service providers",
        "partners"
    ],
    "授权": [
        "authorization",
        "authorizing data processing"
    ],
    "同意": [
        "consent",
        "authorization"
    ],
    "访问控制": [
        "access control",
        "identity management",
        "authentication"
    ],
    "风险": [
        "privacy risk",
        "risk assessment",
        "problematic data actions"
    ],
    "数据删除": [
        "data deletion",
        "destroyed",
        "retention"
    ],
    "数据共享": [
        "data sharing",
        "disclosure",
        "transmission"
    ],
    "审计": [
        "audit",
        "audit records",
        "assessment"
    ],
    "数据最小化": [
        "data minimization",
        "disassociability",
        "selective collection"
    ],
    "数据映射": [
        "mapping",
        "mapped",
        "data processing mapping"
    ]
}


# ============================================================
# 3. 文本 Tokenizer
# ============================================================

def tokenize_text(text: str) -> list[str]:
    """
    将中英文混合文本转换为 BM25 Tokens。

    英文：
        使用单词和数字切分。

    中文：
        提取连续中文字符。
        同时生成相邻双字组合。

    例如：
        数据使用目的

    会产生：
        数
        据
        使
        用
        目
        的
        数据
        据使
        使用
        用目
        目的

    这样可以避免必须依赖额外的中文分词模型。
    """

    if not isinstance(text, str):
        return []

    text = text.lower()

    # --------------------------------------------------------
    # 英文和数字
    # --------------------------------------------------------

    tokens = re.findall(
        r"[a-z0-9]+",
        text
    )

    # --------------------------------------------------------
    # 中文连续片段
    # --------------------------------------------------------

    chinese_segments = re.findall(
        r"[\u4e00-\u9fff]+",
        text
    )

    for segment in chinese_segments:

        # 添加单字 token。
        tokens.extend(
            list(segment)
        )

        # 添加相邻双字 token。
        if len(segment) >= 2:

            tokens.extend([
                segment[index:index + 2]
                for index in range(
                    len(segment) - 1
                )
            ])

    return tokens


def expand_query(keyword: str) -> str:
    """
    对用户查询执行固定词典扩展。

    不改变原始 keyword。

    例如：
        数据使用目的

    扩展为：
        数据使用目的 purpose purposes
        data actions processing purpose
    """

    normalized_keyword = keyword.lower()

    expansion_terms = []

    for chinese_phrase, synonyms in QUERY_SYNONYMS.items():

        if chinese_phrase in normalized_keyword:

            expansion_terms.extend(synonyms)

    # 去重并保持顺序。
    expansion_terms = list(
        dict.fromkeys(expansion_terms)
    )

    if not expansion_terms:
        return keyword

    return (
        keyword
        + " "
        + " ".join(expansion_terms)
    )


# ============================================================
# 4. NIST Policy Retriever
# ============================================================

class PrivacyPolicyRetriever:
    """
    NIST Privacy Framework 1.0 混合检索服务。

    依赖：

        load_core_dataset
        EmbeddingService
        ChromaVectorStore
        BM25Okapi

    对外主要接口：

        match_privacy_policy()

    本 Service 的最终返回结构与阶段一
    mcp_tool_schemas.json 中的输出契约保持一致。
    """

    def __init__(
        self,
        records_path: str | Path = DEFAULT_OUTPUT_JSON,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaVectorStore | None = None,
        persist_directory: str | Path = DEFAULT_CHROMA_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> None:
        """
        初始化检索器。

        Args:
            records_path:
                标准化 NIST JSON 路径。

            embedding_service:
                可选的 Embedding Service 实例。

            vector_store:
                可选的 ChromaDB Service 实例。

            persist_directory:
                ChromaDB 数据目录。

            collection_name:
                ChromaDB Collection 名称。
        """

        # ----------------------------------------------------
        # 加载标准化 NIST 知识数据
        # ----------------------------------------------------

        self.dataset = load_core_dataset(
            records_path
        )

        if (
            self.dataset["framework_version"]
            != SUPPORTED_FRAMEWORK_VERSION
        ):

            raise ValueError(
                "当前检索模块只支持 NIST Framework 1.0。"
            )

        self.records = self.dataset["records"]

        if not self.records:

            raise ValueError(
                "NIST 知识库为空，无法执行检索。"
            )

        # document_id -> record
        self.records_by_id = {
            record["document_id"]: record
            for record in self.records
        }

        # ----------------------------------------------------
        # 初始化 Embedding Service
        # ----------------------------------------------------

        self.embedding_service = (
            embedding_service
            or EmbeddingService()
        )

        # ----------------------------------------------------
        # 初始化 ChromaDB
        # ----------------------------------------------------

        self.vector_store = (
            vector_store
            or ChromaVectorStore(
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        )

        # ----------------------------------------------------
        # 构建 BM25 索引
        # ----------------------------------------------------

        self._build_bm25_index()

        # 记录 Chroma 是否已经与 NIST 数据同步。
        self._index_checked = False

    # --------------------------------------------------------
    # 4.1 构建 BM25 索引
    # --------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """
        使用完整 NIST 知识库构建 BM25 索引。

        每个 Subcategory 对应一个文档。
        """

        self.document_texts = [
            ChromaVectorStore.build_document_text(record)
            for record in self.records
        ]

        self.tokenized_documents = [
            tokenize_text(document)
            for document in self.document_texts
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        # 记录每个文档的位置。
        self.record_positions = {
            record["document_id"]: index
            for index, record in enumerate(self.records)
        }

    # --------------------------------------------------------
    # 4.2 Function 过滤
    # --------------------------------------------------------

    @staticmethod
    def _filter_records(
        records: Sequence[dict[str, Any]],
        function_filter: list[str] | None
    ) -> list[dict[str, Any]]:
        """
        根据 Function Code 筛选知识记录。
        """

        if not function_filter:
            return list(records)

        allowed = set(function_filter)

        return [
            record
            for record in records
            if record["function_code"] in allowed
        ]

    # --------------------------------------------------------
    # 4.3 查询参数校验
    # --------------------------------------------------------

    @staticmethod
    def _validate_query_parameters(
        keyword: str,
        framework_version: str,
        top_k: int,
        function_filter: list[str] | None,
        include_full_text: bool
    ) -> None:
        """
        按照 MCP match_privacy_policy 输入契约验证参数。
        """

        if not isinstance(keyword, str):

            raise TypeError(
                "keyword 必须是字符串"
            )

        keyword = keyword.strip()

        if len(keyword) < 2:

            raise ValueError(
                "keyword 长度不能小于 2"
            )

        if len(keyword) > 1000:

            raise ValueError(
                "keyword 长度不能超过 1000"
            )

        if framework_version != "1.0":

            raise ValueError(
                "当前只支持 framework_version='1.0'"
            )

        if type(top_k) is not int:

            raise TypeError(
                "top_k 必须是整数"
            )

        if not 1 <= top_k <= 10:

            raise ValueError(
                "top_k 必须处于 1 到 10 之间"
            )

        if function_filter is not None:

            if not isinstance(function_filter, list):

                raise TypeError(
                    "function_filter 必须是 list"
                )

            invalid_functions = (
                set(function_filter)
                - SUPPORTED_FUNCTIONS
            )

            if invalid_functions:

                raise ValueError(
                    "存在不支持的 Function Code："
                    f"{sorted(invalid_functions)}"
                )

        if type(include_full_text) is not bool:

            raise TypeError(
                "include_full_text 必须是布尔值"
            )

    # --------------------------------------------------------
    # 4.4 构建或重建 Dense Index
    # --------------------------------------------------------

    def build_index(
        self,
        force_rebuild: bool = False
    ) -> int:
        """
        将完整 NIST 知识库写入 ChromaDB。

        Args:
            force_rebuild:
                True：删除旧 Collection 并重新建立。
                False：如果 ID 已经一致，则跳过重复计算。

        Returns:
            int:
                本次索引写入的记录数。
        """

        expected_ids = set(
            self.records_by_id.keys()
        )

        existing_ids = (
            self.vector_store.get_all_ids()
        )

        # ----------------------------------------------------
        # 检查索引是否需要重建
        # ----------------------------------------------------

        if force_rebuild:

            print("检测到 force_rebuild=True，重建 Chroma Index。")

            self.vector_store.reset_collection()

            existing_ids = set()

        elif existing_ids != expected_ids:

            print(
                "Chroma Index 与 NIST JSON 不一致，"
                "将自动重建。"
            )

            self.vector_store.reset_collection()

            existing_ids = set()

        else:

            print(
                "Chroma Index 已存在且 document_id 一致，"
                "跳过重复 Embedding。"
            )

            self._index_checked = True

            return 0

        # ----------------------------------------------------
        # 批量生成向量并写入 Chroma
        # ----------------------------------------------------

        total_written = 0

        for start_index in range(
            0,
            len(self.records),
            DOCUMENT_EMBEDDING_BATCH_SIZE
        ):

            batch_records = self.records[
                start_index:
                start_index + DOCUMENT_EMBEDDING_BATCH_SIZE
            ]

            batch_documents = [
                ChromaVectorStore.build_document_text(record)
                for record in batch_records
            ]

            print(
                f"正在处理文档："
                f"{start_index + 1}-"
                f"{start_index + len(batch_records)}"
                f"/{len(self.records)}"
            )

            # 使用本地 Embedding 模型。
            embeddings = (
                self.embedding_service.encode_documents(
                    batch_documents,
                    batch_size=DOCUMENT_EMBEDDING_BATCH_SIZE
                )
            )

            # 写入 Chroma。
            written = self.vector_store.upsert_records(
                records=batch_records,
                embeddings=embeddings
            )

            total_written += written

        if (
            self.vector_store.count()
            != len(self.records)
        ):

            raise RuntimeError(
                "Chroma 索引数量校验失败。\n"
                f"预期：{len(self.records)}\n"
                f"实际：{self.vector_store.count()}"
            )

        self._index_checked = True

        return total_written

    # --------------------------------------------------------
    # 4.5 确保 Dense Index 可用
    # --------------------------------------------------------

    def _ensure_index(self) -> None:
        """
        查询之前检查 Chroma 索引是否完整。

        第一次查询时自动初始化索引。
        """

        if self._index_checked:
            return

        expected_ids = set(
            self.records_by_id.keys()
        )

        existing_ids = (
            self.vector_store.get_all_ids()
        )

        if existing_ids != expected_ids:

            self.build_index(
                force_rebuild=True
            )

        else:

            self._index_checked = True

    # --------------------------------------------------------
    # 4.6 BM25 Sparse Retrieval
    # --------------------------------------------------------

    def _bm25_search(
        self,
        expanded_query: str,
        allowed_document_ids: set[str],
        candidate_count: int
    ) -> list[str]:
        """
        使用 BM25 得到稀疏检索排名。

        返回：
            按相关性由高到低排列的 document_id。
        """

        query_tokens = tokenize_text(
            expanded_query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_candidates = []

        for index, score in enumerate(scores):

            record = self.records[index]

            document_id = record["document_id"]

            # Function 过滤。
            if document_id not in allowed_document_ids:
                continue

            score = float(score)

            if not math.isfinite(score):
                continue

            # BM25 得分为零的记录不作为稀疏命中。
            if score <= 0:
                continue

            ranked_candidates.append(
                (document_id, score)
            )

        ranked_candidates.sort(
            key=lambda item: (
                -item[1],
                item[0]
            )
        )

        return [
            document_id
            for document_id, _
            in ranked_candidates[:candidate_count]
        ]

    # --------------------------------------------------------
    # 4.7 Dense Retrieval
    # --------------------------------------------------------

    def _dense_search(
        self,
        keyword: str,
        candidate_count: int,
        function_filter: list[str] | None
    ) -> list[str]:
        """
        使用 Embedding + Chroma 获取向量检索排名。
        """

        query_embedding = (
            self.embedding_service.encode_query(
                keyword
            )
        )

        dense_results = (
            self.vector_store.query_by_embedding(
                query_embedding=query_embedding,
                n_results=candidate_count,
                function_filter=function_filter
            )
        )

        return [
            result["document_id"]
            for result in dense_results
        ]

    # --------------------------------------------------------
    # 4.8 RRF 排名融合
    # --------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_ranked_ids: list[str],
        bm25_ranked_ids: list[str],
        rrf_k: int = DEFAULT_RRF_K
    ) -> tuple[list[tuple[str, float]], str]:
        """
        使用 Reciprocal Rank Fusion 融合 Dense 和 BM25 排名。

        公式：
            RRF(d) = sum(1 / (k + rank))

        其中：
            k = 60

        分数归一化方式：

            normalized_score =
                RRF score / maximum_possible_score

        该分数用于排序展示，
        不代表概率或模型置信度。

        返回：
            [
                (document_id, normalized_score)
            ]

            retrieval_method
        """

        rankings = []

        if dense_ranked_ids:

            rankings.append(
                dense_ranked_ids
            )

        if bm25_ranked_ids:

            rankings.append(
                bm25_ranked_ids
            )

        if not rankings:

            return [], "none"

        fused_scores = defaultdict(float)

        # ----------------------------------------------------
        # 计算 RRF
        # ----------------------------------------------------

        for ranked_ids in rankings:

            for rank, document_id in enumerate(
                ranked_ids,
                start=1
            ):

                fused_scores[document_id] += (
                    1.0 / (rrf_k + rank)
                )

        # ----------------------------------------------------
        # 归一化
        # ----------------------------------------------------

        max_possible_score = (
            len(rankings) / (rrf_k + 1)
        )

        if max_possible_score <= 0:

            return [], "none"

        ranked_results = []

        for document_id, score in fused_scores.items():

            normalized_score = (
                score / max_possible_score
            )

            ranked_results.append((
                document_id,
                min(max(normalized_score, 0.0), 1.0)
            ))

        ranked_results.sort(
            key=lambda item: (
                -item[1],
                item[0]
            )
        )

        # ----------------------------------------------------
        # 判断实际检索方式
        # ----------------------------------------------------

        if dense_ranked_ids and bm25_ranked_ids:

            retrieval_method = "hybrid"

        elif dense_ranked_ids:

            retrieval_method = "dense"

        elif bm25_ranked_ids:

            retrieval_method = "bm25"

        else:

            retrieval_method = "none"

        return ranked_results, retrieval_method

    # --------------------------------------------------------
    # 4.9 核心接口：match_privacy_policy
    # --------------------------------------------------------

    def match_privacy_policy(
        self,
        keyword: str,
        framework_version: str = "1.0",
        top_k: int = 5,
        function_filter: list[str] | None = None,
        include_full_text: bool = True
    ) -> dict[str, Any]:
        """
        根据自然语言查询 NIST Privacy Framework 条目。

        这是 MCP 工具 match_privacy_policy
        对应的 Python Service 方法。

        Args:
            keyword:
                用户自然语言查询。

            framework_version:
                固定使用 1.0。

            top_k:
                返回前 K 条结果，1-10。

            function_filter:
                可选的 Function 筛选条件。

            include_full_text:
                是否完整返回官方 Subcategory Statement。

        Returns:
            dict:

                {
                    "query": str,
                    "framework_name": str,
                    "framework_version": str,
                    "result_count": int,
                    "matches": [
                        {
                            "document_id": str,
                            "function_code": str,
                            "category_code": str,
                            "subcategory_code": str,
                            "title": str,
                            "statement": str,
                            "retrieval_score": float,
                            "retrieval_method": str,
                            "source_url": str
                        }
                    ]
                }
        """

        # ----------------------------------------------------
        # 第一步：输入检查
        # ----------------------------------------------------

        self._validate_query_parameters(
            keyword=keyword,
            framework_version=framework_version,
            top_k=top_k,
            function_filter=function_filter,
            include_full_text=include_full_text
        )

        keyword = keyword.strip()

        # ----------------------------------------------------
        # 第二步：确保向量索引已经构建
        # ----------------------------------------------------

        self._ensure_index()

        # ----------------------------------------------------
        # 第三步：Function 过滤
        # ----------------------------------------------------

        filtered_records = self._filter_records(
            records=self.records,
            function_filter=function_filter
        )

        if not filtered_records:

            return {
                "query": keyword,
                "framework_name": (
                    self.dataset["framework_name"]
                ),
                "framework_version": "1.0",
                "result_count": 0,
                "matches": []
            }

        allowed_document_ids = {
            record["document_id"]
            for record in filtered_records
        }

        # ----------------------------------------------------
        # 第四步：中英文查询扩展
        # ----------------------------------------------------

        expanded_query = expand_query(
            keyword
        )

        # ----------------------------------------------------
        # 第五步：准备候选数量
        # ----------------------------------------------------

        candidate_count = min(
            max(
                top_k * 4,
                DEFAULT_BM25_CANDIDATE_COUNT
            ),
            len(filtered_records)
        )

        # ----------------------------------------------------
        # 第六步：执行 Dense Retrieval
        # ----------------------------------------------------

        dense_ranked_ids = self._dense_search(
            keyword=expanded_query,
            candidate_count=candidate_count,
            function_filter=function_filter
        )

        # ----------------------------------------------------
        # 第七步：执行 BM25 Retrieval
        # ----------------------------------------------------

        bm25_ranked_ids = self._bm25_search(
            expanded_query=expanded_query,
            allowed_document_ids=allowed_document_ids,
            candidate_count=candidate_count
        )

        # ----------------------------------------------------
        # 第八步：RRF 排名融合
        # ----------------------------------------------------

        fused_results, retrieval_method = (
            self._reciprocal_rank_fusion(
                dense_ranked_ids=dense_ranked_ids,
                bm25_ranked_ids=bm25_ranked_ids
            )
        )

        # ----------------------------------------------------
        # 第九步：选择 Top-K
        # ----------------------------------------------------

        top_results = fused_results[:top_k]

        matches = []

        for document_id, relevance_score in top_results:

            record = self.records_by_id.get(
                document_id
            )

            # 只允许从真实 NIST JSON 数据集中取回记录。
            if record is None:

                raise RuntimeError(
                    "检索结果引用了不存在的 document_id："
                    f"{document_id}"
                )

            statement = record["statement"]

            # include_full_text=False 时，
            # 对异常长的文本进行展示截断。
            #
            # NIST Subcategory Statement 通常较短，
            # 正常情况下不会损失原文。
            if not include_full_text and len(statement) > 240:

                statement = (
                    statement[:237] + "..."
                )

            matches.append({
                "document_id": record["document_id"],
                "function_code": record["function_code"],
                "category_code": record["category_code"],
                "subcategory_code": record["subcategory_code"],
                "title": record["title"],
                "statement": statement,
                "retrieval_score": round(
                    relevance_score,
                    6
                ),
                "retrieval_method": retrieval_method,
                "source_url": record["source_url"]
            })

        # ----------------------------------------------------
        # 第十步：返回 MCP Tool Contract
        # ----------------------------------------------------

        return {
            "query": keyword,
            "framework_name": (
                self.dataset["framework_name"]
            ),
            "framework_version": "1.0",
            "result_count": len(matches),
            "matches": matches
        }


# ============================================================
# 5. CLI 测试入口
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description="NIST Privacy Framework Hybrid RAG Retriever"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="需要检索的自然语言问题"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="返回的 Top-K 记录数量"
    )

    parser.add_argument(
        "--function-filter",
        type=str,
        nargs="*",
        choices=sorted(SUPPORTED_FUNCTIONS),
        default=None,
        help="按 NIST Function 筛选"
    )

    parser.add_argument(
        "--no-full-text",
        action="store_true",
        help="限制超长 Subcategory Statement"
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制删除并重建 Chroma 索引"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("NIST Privacy Framework Hybrid RAG")
    print("=" * 65)

    try:

        retriever = PrivacyPolicyRetriever()

        # ----------------------------------------------------
        # 可选：手动重建向量数据库
        # ----------------------------------------------------

        if args.rebuild_index:

            written = retriever.build_index(
                force_rebuild=True
            )

            print(
                f"\n索引重建完成，写入 {written} 条记录。"
            )

        # ----------------------------------------------------
        # 执行检索
        # ----------------------------------------------------

        result = retriever.match_privacy_policy(
            keyword=args.query,
            framework_version="1.0",
            top_k=args.top_k,
            function_filter=args.function_filter,
            include_full_text=not args.no_full_text
        )

        print("\n检索结果：\n")

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

        # ----------------------------------------------------
        # 打印摘要
        # ----------------------------------------------------

        print("\n检索统计：")

        print(
            f"  查询：{result['query']}"
        )

        print(
            f"  Framework：{result['framework_name']}"
        )

        print(
            f"  Version：{result['framework_version']}"
        )

        print(
            f"  返回数量：{result['result_count']}"
        )

        if result["matches"]:

            print("\nTop-K 条目：")

            for index, item in enumerate(
                result["matches"],
                start=1
            ):

                print(
                    f"{index}. {item['subcategory_code']} "
                    f"| {item['retrieval_score']:.4f}"
                )

        print("\nNIST RAG 检索完成。")

        return 0

    except Exception as exc:

        print(
            f"\n[失败] {type(exc).__name__}: {exc}"
        )

        return 1


if __name__ == "__main__":

    sys.exit(main())
