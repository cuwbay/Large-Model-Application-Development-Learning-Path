# 03: LLM 相关的核心 NLP 概念 (Key NLP Concepts for LLMs)

在了解了大型语言模型 (LLM) 的概貌之后，我们需要掌握一些核心的自然语言处理 (NLP) 概念。这些概念是理解 LLM 如何处理文本、学习语言模式以及生成内容的基础。

## 1. Tokens 和 Tokenization (分词/词元化)

**什么是 Token？**

在 NLP 中，**Token (词元)** 是文本处理的基本单位。它可以是一个单词、一个子词 (subword)，甚至是一个字符。LLMs 将输入的文本分割成一系列 Tokens 来进行处理。

**什么是 Tokenization (分词/词元化)？**

**Tokenization** 是将原始文本字符串分解为 Tokens 列表的过程。这个过程由一个称为 **Tokenizer (分词器)** 的组件完成。

**示例：**

*   输入文本: `"Hello world, how are you?"`
*   可能的 Tokens (取决于分词器): `["Hello", "world", ",", "how", "are", "y","ou", "?"]`

**为什么需要 Tokenization？**

*   **结构化输入：** 模型无法直接处理原始文本字符串，需要将其转换为结构化的、可量化的单元。
*   **词汇表管理 (Vocabulary Management)：** Tokenizer 通常会维护一个词汇表，将每个 Token 映射到一个唯一的 ID (数字)。模型实际处理的是这些 ID。
*   **处理未登录词 (Out-of-Vocabulary, OOV)：** 现代 Tokenizer (尤其是子词分词器) 能够通过将未知单词分解为已知的子词片段来处理词汇表中不存在的单词，从而减少 OOV 问题。

**常见的 Tokenization 方法：**

*   **基于空格/标点符号：** 简单的方法，但处理复杂语言现象（如连字符、缩写）和 OOV 时效果不佳。
*   **子词分词 (Subword Tokenization)：** 这是现代 LLMs 中最常用的方法。
    *   **目标：** 在常见词作为整体 Token 和将罕见词分解为有意义的子词片段之间取得平衡。
    *   **优点：**
        *   有效控制词汇表大小。
        *   显著减少 OOV 问题。
        *   能够表示形态相似的词（如 "learn", "learning", "learned" 可能共享 "learn" 这个子词）。
    *   **常见算法：**
        *   **Byte Pair Encoding (BPE):** 从单个字符开始，迭代地合并最常出现的相邻字节对。
        *   **WordPiece:** 类似 BPE，但合并决策基于最大化语言模型概率。常用于 BERT 等模型。
        *   **SentencePiece:** 将文本视为一系列 Unicode 字符，并将空格也视为一种特殊符号进行处理，不依赖于预分词。常用于 T5, Llama 等模型。

**Tokenization 的重要性：**

*   **输入长度限制：** LLMs 有一个最大上下文窗口 (Context Window)，即它们一次能处理的 Token 数量是有限的。理解 Tokenization 如何影响输入文本的 Token 数量非常重要。
*   **模型行为：** 不同的 Tokenization 策略会影响模型的学习方式和最终性能。通常，模型的 Tokenizer 是在预训练阶段就已经确定并一同发布的。

## 2. Embeddings (词嵌入/文本嵌入)

**什么是 Embedding？**

**Embedding (嵌入)** 是一种将离散的符号（如 Tokens 或整个文本片段）映射到低维、稠密的连续向量空间中的技术。这些向量被称为 **嵌入向量 (Embedding Vectors)**。

简单来说，Embedding 就是用一组数字（向量）来表示一个词或一段文本的语义含义。

**为什么需要 Embedding？**

*   **语义表示：** 纯粹的 Token ID 无法捕捉词与词之间的语义关系（例如，“king” 和 “queen” 在语义上比 “king” 和 “apple” 更接近）。Embedding 向量旨在捕捉这些语义相似性。在嵌入空间中，语义相近的词或文本，其对应的向量在空间中的距离也更近。
*   **模型输入：** 神经网络（LLMs 的核心）通常期望数值化的输入。Embedding 将文本数据转换成了模型可以处理的数值形式。

**Embedding 的类型：**

*   **词嵌入 (Word Embeddings)：** 为词汇表中的每个词学习一个固定的向量表示。
    *   **早期方法 (历史背景)：**
        *   **Word2Vec (Skip-gram, CBOW):** 通过预测上下文词或基于上下文词预测当前词来学习词向量。
        *   **GloVe (Global Vectors for Word Representation):** 基于全局词共现统计来学习词向量。
    *   **缺点：** 无法处理一词多义（例如 "bank" 可以指银行或河岸，但只有一个向量表示），并且对于上下文信息不敏感。
*   **上下文嵌入 (Contextual Embeddings)：** 现代 LLMs 使用的主要是上下文嵌入。一个词的嵌入向量会根据其在具体句子或文本中的上下文而动态变化。
    *   **来源：** 通常由预训练的 Transformer 模型（如 BERT, RoBERTa, GPT 等）的中间层或输出层产生。
    *   **优点：** 能够很好地处理一词多义，并捕捉更丰富的上下文语义信息。
*   **句子/文档嵌入 (Sentence/Document Embeddings)：** 将整个句子或文档映射到一个单一的向量表示，用于衡量句子间或文档间的语义相似性。
    *   **常见方法：**
        *   对句子中所有 Token 的上下文嵌入进行平均池化 (Average Pooling) 或最大池化 (Max Pooling)。
        *   使用专门的句子嵌入模型，如 Sentence-Transformers (它通常基于预训练的 Transformer 模型进行微调)。

**Embedding 在 LLM 应用中的作用：**

*   **语义搜索/相似性比较 (Semantic Search / Similarity Comparison)：** 在 RAG (检索增强生成) 系统中，将用户查询和文档块都转换为 Embedding 向量，然后通过计算向量间的相似度（如余弦相似度）来检索最相关的文档。
*   **分类与聚类 (Classification & Clustering)：** 文本的 Embedding 向量可以作为下游分类或聚类任务的特征输入。
*   **LLM 输入层：** LLM 的第一层通常是一个 Embedding 层，它将输入的 Token IDs 转换为 Embedding 向量，作为后续 Transformer 层的输入。

# 03: LLM 相关的核心 NLP 概念 (Key NLP Concepts for LLMs)

在了解了大型语言模型 (LLM) 的概貌之后，我们需要掌握一些核心的自然语言处理 (NLP) 概念。这些概念是理解 LLM 如何处理文本、学习语言模式以及生成内容的基础。

## 1. Tokens 和 Tokenization (分词/词元化)

**什么是 Token？**

在 NLP 中，**Token (词元)** 是文本处理的基本单位。它可以是一个单词、一个子词 (subword)，甚至是一个字符。LLMs 将输入的文本分割成一系列 Tokens 来进行处理。

**什么是 Tokenization (分词/词元化)？**

**Tokenization** 是将原始文本字符串分解为 Tokens 列表的过程。这个过程由一个称为 **Tokenizer (分词器)** 的组件完成。

**示例：**

*   输入文本: `"Hello world, how are you?"`
*   可能的 Tokens (取决于分词器): `["Hello", "world", ",", "how", "are", "you", "?"]`

**为什么需要 Tokenization？**

*   **结构化输入：** 模型无法直接处理原始文本字符串，需要将其转换为结构化的、可量化的单元。
*   **词汇表管理 (Vocabulary Management)：** Tokenizer 通常会维护一个词汇表，将每个 Token 映射到一个唯一的 ID (数字)。模型实际处理的是这些 ID。
*   **处理未登录词 (Out-of-Vocabulary, OOV)：** 现代 Tokenizer (尤其是子词分词器) 能够通过将未知单词分解为已知的子词片段来处理词汇表中不存在的单词，从而减少 OOV 问题。

**常见的 Tokenization 方法：**

*   **基于空格/标点符号：** 简单的方法，但处理复杂语言现象（如连字符、缩写）和 OOV 时效果不佳。
*   **子词分词 (Subword Tokenization)：** 这是现代 LLMs 中最常用的方法。
    *   **目标：** 在常见词作为整体 Token 和将罕见词分解为有意义的子词片段之间取得平衡。
    *   **优点：**
        *   有效控制词汇表大小。
        *   显著减少 OOV 问题。
        *   能够表示形态相似的词（如 "learn", "learning", "learned" 可能共享 "learn" 这个子词）。
    *   **常见算法：**
        *   **Byte Pair Encoding (BPE):** 从单个字符开始，迭代地合并最常出现的相邻字节对。
        *   **WordPiece:** 类似 BPE，但合并决策基于最大化语言模型概率。常用于 BERT 等模型。
        *   **SentencePiece:** 将文本视为一系列 Unicode 字符，并将空格也视为一种特殊符号进行处理，不依赖于预分词。常用于 T5, Llama 等模型。

**Tokenization 的重要性：**

*   **输入长度限制：** LLMs 有一个最大上下文窗口 (Context Window)，即它们一次能处理的 Token 数量是有限的。理解 Tokenization 如何影响输入文本的 Token 数量非常重要。
*   **模型行为：** 不同的 Tokenization 策略会影响模型的学习方式和最终性能。通常，模型的 Tokenizer 是在预训练阶段就已经确定并一同发布的。

## 2. Embeddings (词嵌入/文本嵌入)

**什么是 Embedding？**

**Embedding (嵌入)** 是一种将离散的符号（如 Tokens 或整个文本片段）映射到低维、稠密的连续向量空间中的技术。这些向量被称为 **嵌入向量 (Embedding Vectors)**。

简单来说，Embedding 就是用一组数字（向量）来表示一个词或一段文本的语义含义。

**为什么需要 Embedding？**

*   **语义表示：** 纯粹的 Token ID 无法捕捉词与词之间的语义关系（例如，“king” 和 “queen” 在语义上比 “king” 和 “apple” 更接近）。Embedding 向量旨在捕捉这些语义相似性。在嵌入空间中，语义相近的词或文本，其对应的向量在空间中的距离也更近。
*   **模型输入：** 神经网络（LLMs 的核心）通常期望数值化的输入。Embedding 将文本数据转换成了模型可以处理的数值形式。

**Embedding 的类型：**

*   **词嵌入 (Word Embeddings)：** 为词汇表中的每个词学习一个固定的向量表示。
    *   **早期方法 (历史背景)：**
        *   **Word2Vec (Skip-gram, CBOW):** 通过预测上下文词或基于上下文词预测当前词来学习词向量。
        *   **GloVe (Global Vectors for Word Representation):** 基于全局词共现统计来学习词向量。
    *   **缺点：** 无法处理一词多义（例如 "bank" 可以指银行或河岸，但只有一个向量表示），并且对于上下文信息不敏感。
*   **上下文嵌入 (Contextual Embeddings)：** 现代 LLMs 使用的主要是上下文嵌入。一个词的嵌入向量会根据其在具体句子或文本中的上下文而动态变化。
    *   **来源：** 通常由预训练的 Transformer 模型（如 BERT, RoBERTa, GPT 等）的中间层或输出层产生。
    *   **优点：** 能够很好地处理一词多义，并捕捉更丰富的上下文语义信息。
*   **句子/文档嵌入 (Sentence/Document Embeddings)：** 将整个句子或文档映射到一个单一的向量表示，用于衡量句子间或文档间的语义相似性。
    *   **常见方法：**
        *   对句子中所有 Token 的上下文嵌入进行平均池化 (Average Pooling) 或最大池化 (Max Pooling)。
        *   使用专门的句子嵌入模型，如 Sentence-Transformers (它通常基于预训练的 Transformer 模型进行微调)。

**Embedding 在 LLM 应用中的作用：**

*   **语义搜索/相似性比较 (Semantic Search / Similarity Comparison)：** 在 RAG (检索增强生成) 系统中，将用户查询和文档块都转换为 Embedding 向量，然后通过计算向量间的相似度（如余弦相似度）来检索最相关的文档。
*   **分类与聚类 (Classification & Clustering)：** 文本的 Embedding 向量可以作为下游分类或聚类任务的特征输入。
*   **LLM 输入层：** LLM 的第一层通常是一个 Embedding 层，它将输入的 Token IDs 转换为 Embedding 向量，作为后续 Transformer 层的输入。

## 3. Transformer 架构简介

**Transformer 模型** 是现代 LLMs 的核心架构，由 Vaswani 等人在 2017 年的论文 "Attention Is All You Need" 中提出，最初用于机器翻译任务。它的出现彻底改变了 NLP 领域。

**为什么 Transformer 如此重要？**

*   **并行处理能力：** 与早期依赖循环结构 (RNN, LSTM) 的模型相比，Transformer 能够更好地并行处理输入序列中的所有 Tokens，极大地提高了训练效率，使得训练更大规模的模型成为可能。
*   **长距离依赖捕捉：** 通过其核心的自注意力机制 (Self-Attention)，Transformer 能够有效地捕捉文本中相距较远的词之间的依赖关系。

**Transformer 的核心组件 (简化版)：**

1.  **输入嵌入 (Input Embedding) 和位置编码 (Positional Encoding)：**
    *   **输入嵌入：** 将输入的 Token IDs 转换为词嵌入向量。
    *   **位置编码：** 由于 Transformer 的并行特性本身不包含序列顺序信息，需要显式地向嵌入向量中添加位置信息（通常是固定的或可学习的向量），让模型知道每个 Token在序列中的位置。

2.  **多头自注意力机制 (Multi-Head Self-Attention)：**
    *   **核心思想：** 对于序列中的每个 Token，自注意力机制会计算该 Token 与序列中所有其他 Tokens (包括其自身) 的“关注度”或“相关性得分”。然后，基于这些得分，对所有 Tokens 的向量表示进行加权求和，得到该 Token 的新的、富含上下文信息的表示。
    *   **Query, Key, Value (QKV)：** 每个输入 Token 的嵌入向量会生成三个不同的向量：查询向量 (Query)、键向量 (Key) 和值向量 (Value)。Query 用于与所有 Key 进行匹配计算相似度，相似度得分用于对 Value 进行加权。
    *   **多头 (Multi-Head)：** 将注意力机制并行地执行多次（每个“头”学习不同的注意力模式），然后将结果拼接起来。这允许模型从不同的表示子空间学习信息，捕捉更丰富的依赖关系。

3.  **前馈神经网络 (Feed-Forward Neural Network, FFN)：**
    *   在自注意力层之后，每个位置的输出会独立地通过一个简单的前馈神经网络（通常包含两个线性层和一个激活函数，如 ReLU 或 GELU）。
    *   FFN 用于对自注意力层的输出进行进一步的非线性变换和特征提取。

4.  **残差连接 (Residual Connections) 和层归一化 (Layer Normalization)：**
    *   **残差连接：** 将每一层的输入直接加到该层的输出上（类似于跳跃连接）。这有助于缓解梯度消失问题，使得训练更深层的网络成为可能。
    *   **层归一化：** 对每一层的输出进行归一化，稳定训练过程，加速收敛。

5.  **编码器 (Encoder) 和解码器 (Decoder) 结构：**
    *   **编码器：** 由 N 个相同的编码器层堆叠而成，每个编码器层包含一个多头自注意力子层和一个前馈神经网络子层。主要用于理解和表征输入序列。
    *   **解码器：** 也由 N 个相同的解码器层堆叠而成，但每个解码器层除了包含编码器中的两个子层外，还有一个额外的多头注意力子层，用于关注编码器的输出（这在机器翻译等序列到序列任务中很重要）。解码器用于生成输出序列。
    *   **LLM 架构变体：**
        *   **Encoder-Only (例如 BERT, RoBERTa):** 擅长理解任务，如文本分类、命名实体识别、句子相似度。输出的是文本的上下文表示。
        *   **Decoder-Only (例如 GPT 系列, Llama, Qwen):** 擅长生成任务，如文本续写、对话。它们通常采用自回归 (Autoregressive) 的方式逐个 Token 生成文本。这是当前主流 LLM 的主要架构。
        *   **Encoder-Decoder (例如 T5, BART):** 同时包含编码器和解码器，适用于序列到序列任务，如翻译、摘要。

**LLM 通常是基于 Decoder-Only 或 Encoder-Decoder 架构的扩展和变体。**

## 4. Attention 机制 (注意力机制)

虽然上面提到了自注意力，但广义的 **Attention 机制** 是一种让神经网络在处理序列数据时，能够“关注”输入序列中不同部分的重要程度，并据此进行信息加权的思想。

**核心思想：** 不是平等地对待输入序列中的所有部分，而是有选择地分配“注意力资源”。

*   **自注意力 (Self-Attention)：** 如上所述，是序列内部元素之间的注意力计算，用于捕捉序列内部的依赖关系。Transformer 的核心。
*   **交叉注意力 (Cross-Attention)：** （在 Encoder-Decoder 架构的 Decoder 中使用）允许 Decoder 在生成输出序列的每一步时，关注 Encoder 输出的输入序列的不同部分。例如，在翻译时，翻译目标语言的某个词时，模型会关注源语言句子中的相关词。

## 5. 上下文窗口 (Context Window / Context Length)

**上下文窗口** 是指 LLM 在一次处理或生成文本时能够考虑的输入文本的最大长度，通常以 **Tokens 的数量** 来衡量。

**示例：**

*   一个模型的上下文窗口是 4096 Tokens，意味着它最多能处理包含 4096 个 Tokens 的输入序列。

**重要性：**

*   **信息容量：** 上下文窗口的大小直接决定了模型能“记住”和利用多少上下文信息。窗口越大，模型在处理长文档、进行长对话或执行需要广泛背景知识的任务时表现可能越好。
*   **应用限制：** 如果输入的文本长度超过了模型的上下文窗口，通常需要进行截断或采用其他策略（如滑动窗口、分块处理后再聚合等），这可能会丢失信息。
*   **成本与效率：** 更大的上下文窗口通常也意味着更高的计算成本和更慢的推理速度。

**发展趋势：** 新的 LLM 普遍在努力扩大上下文窗口的长度，例如从早期的几千 Tokens 发展到现在的数万甚至数十万 Tokens (如 Claude 的某些版本)。
