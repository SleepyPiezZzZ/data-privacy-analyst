
"""
vector_store.py

ChromaDB 向量数据库管理模块。

核心功能：
1. 创建 PersistentClient。
2. 创建或获取 NIST Privacy Framework Collection。
3. 将 NIST 记录和 Embedding 写入 ChromaDB。
4. 通过 Embedding 执行向量检索。
5. 支持 Function 元数据过滤。
6. 支持读取已有 document_id。
7. 支持重建 Collection。

默认数据库：
    vector_store/chroma/

默认 Collection：
    nist_privacy_framework_v1

距离度量：
    cosine

运行方式：
    python -m rag.vector_store
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


# ============================================================
# 1. 路径及 Collection 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHROMA_DIRECTORY = (
    PROJECT_ROOT / "vector_store" / "chroma"
)

DEFAULT_COLLECTION_NAME = (
    "nist_privacy_framework_v1"
)

COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "framework_name": "NIST Privacy Framework",
    "framework_version": "1.0",
    "dataset_id": "NIST-PF-CORE-1.0"
}


# ============================================================
# 2. ChromaDB 向量数据库服务
# ============================================================

class ChromaVectorStore:
    """
    ChromaDB Persistent Vector Store。

    不在该类内部生成 Embedding。

    所有 Embedding 必须由 EmbeddingService 生成，
    然后通过 upsert_records 写入。

    示例：

        store = ChromaVectorStore()

        store.upsert_records(
            records=records,
            embeddings=embeddings
        )

        results = store.query_by_embedding(
            query_embedding=query_vector,
            n_results=5
        )
    """

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_CHROMA_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> None:
        """
        初始化 ChromaDB。

        Args:
            persist_directory:
                本地持久化数据库目录。

            collection_name:
                Collection 名称。
        """

        self.persist_directory = Path(
            persist_directory
        )

        self.collection_name = collection_name

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        # Chroma Collection 名称的简单合法性检查。
        if not isinstance(collection_name, str):

            raise TypeError(
                "collection_name 必须是字符串"
            )

        if not collection_name.strip():

            raise ValueError(
                "collection_name 不能为空"
            )

        try:

            import chromadb

        except ImportError as exc:

            raise ImportError(
                "未安装 chromadb。\n"
                "请执行：pip install chromadb"
            ) from exc

        # ----------------------------------------------------
        # 创建本地持久化客户端
        # ----------------------------------------------------

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # ----------------------------------------------------
        # 创建或读取 Collection
        # ----------------------------------------------------

        # 显式指定 embedding_function=None。
        # 由 EmbeddingService 提供向量，
        # 避免 ChromaDB 隐式使用其他默认 Embedding 模型。
        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=COLLECTION_METADATA,
                embedding_function=None
            )
        )

        # 检查现有 Collection 的距离度量。
        existing_metadata = (
            self.collection.metadata or {}
        )

        existing_space = existing_metadata.get(
            "hnsw:space"
        )

        if (
            existing_space is not None
            and existing_space != "cosine"
        ):

            raise ValueError(
                "当前 Collection 的距离度量不是 cosine。\n"
                "请使用新的 Collection 名称，"
                "或者删除旧 Collection 后重新创建。"
            )

    # --------------------------------------------------------
    # 2.1 生成 Chroma Document
    # --------------------------------------------------------

    @staticmethod
    def build_document_text(
        record: dict[str, Any]
    ) -> str:
        """
        将一条 NIST 记录转换为适合语义检索的文本。

        注意：
        不修改原始 JSON 记录。

        同时保留：
        1. Function。
        2. Category。
        3. Subcategory ID。
        4. 官方原文 Statement。
        """

        document_text = "\n".join([
            f"Framework: {record['framework_name']}",
            f"Version: {record['framework_version']}",
            f"Function: {record['function_code']}",
            f"Function Name: {record['function_name']}",
            (
                "Function Description: "
                f"{record['function_description']}"
            ),
            f"Category: {record['category_code']}",
            f"Category Name: {record['category_name']}",
            (
                "Category Description: "
                f"{record['category_description']}"
            ),
            (
                "Subcategory: "
                f"{record['subcategory_code']}"
            ),
            f"Title: {record['title']}",
            f"Statement: {record['statement']}"
        ])

        return document_text

    # --------------------------------------------------------
    # 2.2 构建 Chroma Metadata
    # --------------------------------------------------------

    @staticmethod
    def build_metadata(
        record: dict[str, Any]
    ) -> dict[str, Any]:
        """
        将 NIST 记录转换为 ChromaDB 兼容的 Metadata。

        Chroma Metadata 使用标量字段，
        不直接存储 Python list 或 dict。
        """

        return {
            "framework_name": record["framework_name"],
            "framework_version": record["framework_version"],
            "function_code": record["function_code"],
            "function_name": record["function_name"],
            "category_code": record["category_code"],
            "category_name": record["category_name"],
            "subcategory_code": record["subcategory_code"],
            "title": record["title"],
            "source_url": record["source_url"]
        }

    # --------------------------------------------------------
    # 2.3 写入文档和向量
    # --------------------------------------------------------

    def upsert_records(
        self,
        records: Sequence[dict[str, Any]],
        embeddings: Sequence[Sequence[float]]
    ) -> int:
        """
        将 NIST 记录写入 ChromaDB。

        Args:
            records:
                标准化 NIST 记录列表。

            embeddings:
                对应记录的 Embedding 列表。

        Returns:
            int:
                本次写入的文档数量。
        """

        if not isinstance(records, (list, tuple)):

            raise TypeError(
                "records 必须是 list 或 tuple"
            )

        if not isinstance(embeddings, (list, tuple)):

            raise TypeError(
                "embeddings 必须是 list 或 tuple"
            )

        if len(records) != len(embeddings):

            raise ValueError(
                "records 数量与 embeddings 数量不一致。"
            )

        if len(records) == 0:
            return 0

        ids = []
        documents = []
        metadatas = []
        vectors = []

        embedding_dimension = None

        for index, (record, embedding) in enumerate(
            zip(records, embeddings)
        ):

            document_id = record["document_id"]

            if not isinstance(document_id, str):

                raise TypeError(
                    f"records[{index}].document_id 必须是字符串"
                )

            if document_id in ids:

                raise ValueError(
                    f"批次中出现重复 document_id：{document_id}"
                )

            vector = [
                float(value)
                for value in embedding
            ]

            if not vector:

                raise ValueError(
                    f"文档 {document_id} 的 Embedding 为空"
                )

            if embedding_dimension is None:

                embedding_dimension = len(vector)

            elif len(vector) != embedding_dimension:

                raise ValueError(
                    "同一批次 Embedding 维度不一致。"
                )

            if not all(
                value == value
                and abs(value) != float("inf")
                for value in vector
            ):

                raise ValueError(
                    f"文档 {document_id} 包含非法向量值"
                )

            ids.append(document_id)

            documents.append(
                self.build_document_text(record)
            )

            metadatas.append(
                self.build_metadata(record)
            )

            vectors.append(vector)

        # upsert 可以安全地重复写入相同 ID，
        # 不会因为重复执行而新增重复文档。
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas
        )

        return len(ids)

    # --------------------------------------------------------
    # 2.4 向量查询
    # --------------------------------------------------------

    def query_by_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 5,
        function_filter: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        根据查询向量执行 ChromaDB 相似度搜索。

        Args:
            query_embedding:
                用户查询对应的向量。

            n_results:
                需要返回的候选文档数量。

            function_filter:
                Function 过滤列表。

                例如：
                    ["IDENTIFY-P", "GOVERN-P"]

        Returns:
            list[dict]:
                按向量距离升序排列的候选记录。

                每条记录包含：
                    document_id
                    distance
                    metadata
                    document
        """

        if type(n_results) is not int or n_results < 1:

            raise ValueError(
                "n_results 必须是正整数"
            )

        if self.collection.count() == 0:
            return []

        query_vector = [
            float(value)
            for value in query_embedding
        ]

        if not query_vector:

            raise ValueError(
                "query_embedding 不能为空"
            )

        query_dimension = len(query_vector)

        if not all(
            value == value
            and abs(value) != float("inf")
            for value in query_vector
        ):

            raise ValueError(
                "query_embedding 包含非法数值"
            )

        # ----------------------------------------------------
        # 构建 Function 过滤条件
        # ----------------------------------------------------

        where = None

        if function_filter:

            where = {
                "function_code": {
                    "$in": function_filter
                }
            }

        # ----------------------------------------------------
        # 执行 Chroma 向量查询
        # ----------------------------------------------------

        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where,
            include=[
                "metadatas",
                "documents",
                "distances"
            ]
        )

        # ----------------------------------------------------
        # 获取第一条查询的结果
        # ----------------------------------------------------

        ids = result["ids"][0]

        metadatas = result["metadatas"][0]

        documents = result["documents"][0]

        distances = result["distances"][0]

        candidates = []

        for document_id, metadata, document, distance in zip(
            ids,
            metadatas,
            documents,
            distances
        ):

            candidates.append({
                "document_id": document_id,
                "distance": float(distance),
                "metadata": metadata,
                "document": document
            })

        return candidates

    # --------------------------------------------------------
    # 2.5 查询已有 ID
    # --------------------------------------------------------

    def get_all_ids(self) -> set[str]:
        """
        返回 Collection 中所有 document_id。

        用于检查当前向量索引与 NIST JSON 是否一致。
        """

        result = self.collection.get(
            include=["metadatas"]
        )

        return set(
            result["ids"]
        )

    # --------------------------------------------------------
    # 2.6 获取记录数量
    # --------------------------------------------------------

    def count(self) -> int:
        """
        获取 Collection 中的记录总数。
        """

        return self.collection.count()

    # --------------------------------------------------------
    # 2.7 重建 Collection
    # --------------------------------------------------------

    def reset_collection(self) -> None:
        """
        删除旧 Collection 并重新创建。

        注意：
        这是破坏性操作，会删除当前 Collection 全部向量。
        仅在重建索引时调用。
        """

        self.client.delete_collection(
            name=self.collection_name
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=COLLECTION_METADATA,
                embedding_function=None
            )
        )

    # --------------------------------------------------------
    # 2.8 数据库信息
    # --------------------------------------------------------

    def get_collection_info(self) -> dict[str, Any]:
        """
        返回 Collection 基本信息。
        """

        return {
            "collection_name": self.collection_name,
            "persist_directory": str(
                self.persist_directory
            ),
            "document_count": self.collection.count(),
            "metadata": self.collection.metadata
        }


# ============================================================
# 3. 独立测试
# ============================================================

def main() -> int:

    print("=" * 65)
    print("ChromaDB Vector Store Test")
    print("=" * 65)

    try:

        store = ChromaVectorStore()

        information = store.get_collection_info()

        print("\n[成功] ChromaDB 已连接。")

        print(
            f"  Collection：{information['collection_name']}"
        )

        print(
            f"  数据目录：{information['persist_directory']}"
        )

        print(
            f"  当前记录数量：{information['document_count']}"
        )

        print(
            "\n注意：尚未导入 NIST 数据时，"
            "记录数量为 0 是正常情况。"
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
