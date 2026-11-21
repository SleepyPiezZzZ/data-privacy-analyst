
"""
embedding_service.py

本地多语言 Embedding 服务。

核心功能：
1. 加载多语言 Sentence Transformer。
2. 将 NIST 文档转换为向量。
3. 将用户查询转换为向量。
4. 对输出向量进行 L2 归一化。
5. 检查向量是否包含 NaN 或 Inf。

默认模型：
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

模型信息：
    384 dimensions
    50 languages
    Apache-2.0

模型首次运行时自动从 Hugging Face 下载。

默认 CPU 推理，不需要 NVIDIA GPU。

环境变量：
    EMBEDDING_MODEL
    EMBEDDING_DEVICE

运行方式：
    python -m rag.embedding_service
"""

from __future__ import annotations

import os

from pathlib import Path
from typing import Sequence

import numpy as np


# ============================================================
# 1. 模型配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

DEFAULT_MODEL_CACHE = PROJECT_ROOT / "models"

DEFAULT_DEVICE = "cpu"

EXPECTED_EMBEDDING_DIMENSION = 384


# ============================================================
# 2. Embedding 服务
# ============================================================

class EmbeddingService:
    """
    本地文本向量化服务。

    模型采用延迟加载：
    只有首次调用 encode_query 或 encode_documents 时，
    才会下载或加载模型权重。

    示例：

        service = EmbeddingService()

        vector = service.encode_query(
            "检查数据使用目的是否经过登记"
        )

        vectors = service.encode_documents([
            "The purposes for the data actions are inventoried.",
            "Data processing is mapped."
        ])
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cache_folder: str | Path | None = None
    ) -> None:
        """
        初始化 Embedding 服务。

        Args:
            model_name:
                Hugging Face 上的模型名称。

            device:
                cpu / cuda / mps 等设备。

            cache_folder:
                模型下载缓存目录。
        """

        self.model_name = (
            model_name
            or os.getenv(
                "EMBEDDING_MODEL",
                DEFAULT_MODEL_NAME
            )
        )

        self.device = (
            device
            or os.getenv(
                "EMBEDDING_DEVICE",
                DEFAULT_DEVICE
            )
        )

        self.cache_folder = Path(
            cache_folder or DEFAULT_MODEL_CACHE
        )

        self.cache_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # 不在初始化阶段立即下载模型。
        self._model = None

    # --------------------------------------------------------
    # 2.1 延迟加载模型
    # --------------------------------------------------------

    def _load_model(self):
        """
        加载 SentenceTransformer。

        模型只加载一次，后续调用复用内存中的模型。
        """

        if self._model is not None:
            return self._model

        try:

            from sentence_transformers import SentenceTransformer

        except ImportError as exc:

            raise ImportError(
                "未安装 sentence-transformers。\n"
                "请执行：pip install sentence-transformers"
            ) from exc

        print(
            f"正在加载 Embedding 模型：{self.model_name}"
        )

        print(
            f"运行设备：{self.device}"
        )

        try:

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(self.cache_folder)
            )

        except Exception as exc:

            raise RuntimeError(
                "Embedding 模型加载失败。\n"
                f"模型：{self.model_name}\n"
                f"设备：{self.device}\n"
                f"原始错误：{exc}"
            ) from exc

        dimension = (
            self._model.get_sentence_embedding_dimension()
        )

        if dimension != EXPECTED_EMBEDDING_DIMENSION:

            raise RuntimeError(
                "Embedding 向量维度与项目配置不一致。\n"
                f"预期：{EXPECTED_EMBEDDING_DIMENSION}\n"
                f"实际：{dimension}"
            )

        print(
            f"Embedding 模型加载完成，维度：{dimension}"
        )

        return self._model

    # --------------------------------------------------------
    # 2.2 通用编码接口
    # --------------------------------------------------------

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32
    ) -> list[list[float]]:
        """
        将多个文本转换为向量。

        Args:
            texts:
                文本序列。

            batch_size:
                推理批量大小。

        Returns:
            list[list[float]]:
                每条文本对应一个 384 维向量。

        注意：
            使用 normalize_embeddings=True，
            使向量具有统一的归一化尺度。
        """

        if not isinstance(texts, (list, tuple)):

            raise TypeError(
                "texts 必须是 list 或 tuple"
            )

        if type(batch_size) is not int or batch_size < 1:

            raise ValueError(
                "batch_size 必须是正整数"
            )

        if len(texts) == 0:
            return []

        normalized_texts = []

        for index, text in enumerate(texts):

            if not isinstance(text, str):

                raise TypeError(
                    f"texts[{index}] 必须是字符串"
                )

            text = text.strip()

            if not text:

                raise ValueError(
                    f"texts[{index}] 不能为空"
                )

            normalized_texts.append(text)

        model = self._load_model()

        embeddings = model.encode(
            normalized_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # 检查输出维度
        # ----------------------------------------------------

        expected_shape = (
            len(normalized_texts),
            EXPECTED_EMBEDDING_DIMENSION
        )

        if embeddings.shape != expected_shape:

            raise RuntimeError(
                "Embedding 输出形状错误。\n"
                f"预期：{expected_shape}\n"
                f"实际：{embeddings.shape}"
            )

        # ----------------------------------------------------
        # 检查数值有效性
        # ----------------------------------------------------

        if not np.isfinite(embeddings).all():

            raise RuntimeError(
                "Embedding 输出中存在 NaN 或 Inf。"
            )

        return embeddings.tolist()

    # --------------------------------------------------------
    # 2.3 文档编码接口
    # --------------------------------------------------------

    def encode_documents(
        self,
        documents: Sequence[str],
        batch_size: int = 32
    ) -> list[list[float]]:
        """
        对知识库文档进行 Embedding。

        documents 中的每个元素对应一条 NIST Subcategory。
        """

        return self.encode_texts(
            texts=documents,
            batch_size=batch_size
        )

    # --------------------------------------------------------
    # 2.4 查询编码接口
    # --------------------------------------------------------

    def encode_query(
        self,
        query: str
    ) -> list[float]:
        """
        对用户自然语言查询进行 Embedding。

        支持中文和英文输入。
        """

        if not isinstance(query, str):

            raise TypeError(
                "query 必须是字符串"
            )

        query = query.strip()

        if not query:

            raise ValueError(
                "query 不能为空"
            )

        embeddings = self.encode_texts(
            texts=[query],
            batch_size=1
        )

        return embeddings[0]

    # --------------------------------------------------------
    # 2.5 模型信息
    # --------------------------------------------------------

    def get_model_info(self) -> dict:

        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": (
                EXPECTED_EMBEDDING_DIMENSION
            ),
            "cache_folder": str(self.cache_folder),
            "normalized_embeddings": True
        }


# ============================================================
# 3. 独立测试
# ============================================================

def main() -> int:

    print("=" * 65)
    print("Embedding Service Test")
    print("=" * 65)

    try:

        service = EmbeddingService()

        test_documents = [
            (
                "The purposes for the data actions "
                "are inventoried."
            ),
            (
                "Data processing is mapped, illustrating "
                "the data actions and associated data elements."
            )
        ]

        query = "检查数据处理目的是否已经登记"

        print("\n生成文档向量...")

        document_vectors = service.encode_documents(
            test_documents
        )

        print("\n生成查询向量...")

        query_vector = service.encode_query(
            query
        )

        print("\n[成功] Embedding 测试完成。")

        print(
            f"  文档数量：{len(document_vectors)}"
        )

        print(
            f"  文档向量维度：{len(document_vectors[0])}"
        )

        print(
            f"  查询向量维度：{len(query_vector)}"
        )

        print(
            f"  模型：{service.model_name}"
        )

        print(
            f"  设备：{service.device}"
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
