# 00: 入门介绍 - 开启你的大模型应用开发之旅

本模块是整个学习旅程的基石。无论你是对大型语言模型 (LLM) 充满好奇的开发者，还是希望系统学习如何利用 LLM 构建创新应用的探索者，这里都将为你提供必要的起点。

## 🎯 本模块学习目标

完成本入门模块后，你将能够：

1.  **清晰理解什么是大型语言模型 (LLM)**：了解它们的核心能力、关键特征以及它们在人工智能领域所扮演的革命性角色。
2.  **掌握与 LLM 密切相关的核心自然语言处理 (NLP) 概念**：如分词 (Tokenization)、文本嵌入 (Embeddings) 以及 Transformer 架构和注意力机制的基本思想，这些是理解 LLM 工作原理的关键。
3.  **洞察 LLM 应用的广阔前景与常见模式**：了解当前 LLM 技术催生了哪些令人兴奋的应用，并初步认识到提示工程 (Prompt Engineering)、检索增强生成 (RAG) 和智能体 (Agent) 等关键应用范式。
4.  **成功搭建并配置你的本地开发环境**：包括安装必要的 Python 库。

## 📖 本模块内容概览

为了达成以上目标，本模块将包含以下主要内容：

*   **`01-Deep-Dive-into-LLMs.md`**:
    *   深入探讨 LLM 的核心能力（文本生成、理解、问答等）。
    *   解析 LLM 之所以“大”的关键因素（参数量、数据量）。
    *   概览市面上主流的 LLM 及其特点。
    *   正视 LLM 当前面临的局限性与挑战 (如幻觉、偏见)。
*   **`02-Key-NLP-Concepts-for-LLMs.md`**:
    *   **分词 (Tokens & Tokenization)**：文本如何被模型理解的第一步。
    *   **文本嵌入 (Embeddings)**：赋予文本语义向量表示的魔法。
    *   **Transformer 架构简介**：支撑现代 LLM 的核心引擎 (重点理解其设计理念和自注意力机制的作用)。
    *   **注意力机制 (Attention Mechanism)**：LLM 如何聚焦关键信息。
    *   **上下文窗口 (Context Window)**：LLM 的“记忆”边界。
*   **`03-The-Rise-of-LLM-Applications.md`**:
    *   探讨 LLM 应用的爆发原因。
    *   介绍聊天机器人、内容创作、代码辅助、RAG、Agent 等典型应用场景。
    *   强调提示工程 (Prompt Engineering) 在释放 LLM 能力中的核心地位。
*   **`04-Setting-up-Your-Development-Environment.md`**:
    *   Python 环境与虚拟环境管理 (Conda/venv)。
    *   核心库安装 (`transformers`, `langchain` 等)。
    *   IDE/编辑器推荐与配置。
    *   API 密钥的安全获取、存储与使用 (`.env` 文件)。
    *   (可选) GPU 环境的基本认知。
*   **`05-Git-and-GitHub-Quickstart-for-this-Project.md`**:
    *   克隆本项目、浏览 Jupyter Notebooks。
    *   (可选) 个人学习分支的简单指引。

ps：现有的LLMs能力已经非常强悍了，上面的全部内容都是由2025年3月发布的Gemini 2.5pro模型生成的，而且是全部对的。
本项目侧重于大模型应用开发，所以对大模型的算法的内容只停留在知道是什么的层面。本章节只是为了快速理解LLMs大概包括什么。后续如果我有精力会编写的更加详细