# 大模型应用开发学习路径

👋 欢迎来到我的大模型 (LLM) 应用开发学习之旅！本仓库旨在记录我的学习进度、分享见解，并为所有对使用 Hugging Face 和 Langchain 构建应用程序感兴趣的朋友提供一个结构化的学习路径。

## 🎯 项目目标

本项目的主要目标是：

1. **系统学习** 开发大语言模型应用所需的核心概念和实践技能。
2. **记录学习过程**，包括笔记、代码示例、遇到的挑战及解决方案。
3. **创建一个可共享的资源**，帮助其他初学者在这个快速发展的领域中导航。
4. 探索关键框架，如用于模型交互的 **Hugging Face 🤗** 和用于构建复杂 LLM 驱动应用（如 RAG 系统和智能体）的 **Langchain 🦜🔗**。

## 🗺️ 学习路径 / 仓库结构

本仓库按模块组织，建议按顺序学习，但你也可以随时跳转到感兴趣的主题：

- **`00-Introduction/` (入门介绍)**:
  - 什么是 LLM？核心自然语言处理概念 (Transformer, Embeddings)。
  - 搭建你的开发环境 (Python, Conda/Venv, API 密钥)。
- **`01-HuggingFace-Basics/` (Hugging Face 基础)**:
  - Hugging Face 生态系统简介 (Hub, Transformers, Datasets)。
  - 加载预训练模型和分词器。
  - 执行基本的推理任务 (文本生成、分类等)。
- **`02-Langchain-Core-Concepts/` (Langchain 核心概念)**:
  - 理解 Langchain 的架构 (模型, 提示, 链, 索引, 记忆, 智能体)。
  - 在 Langchain 中使用 LLM 和聊天模型。
  - 通过 PromptTemplates掌握提示工程。
  - 构建基础链和顺序链。
  - LangChain 表达式语言 (LCEL) 入门。
- **`03-Building-RAG-with-Langchain/` (使用 Langchain 构建 RAG)**:
  - 深入理解检索增强生成 (RAG)。
  - 文档加载器 (PDF, TXT, 网页等)。
  - 用于高效分块的文本分割器。
  - 嵌入模型和向量存储 (例如 Chroma, FAISS)。
  - 从零开始构建完整的 RAG 流水线。
- **`04-Building-Agents-with-Langchain/` (使用 Langchain 构建智能体)**:
  - 理解智能体的概念和架构 (例如 ReAct)。
  - 在 Langchain 智能体中使用工具。
  - 构建能够推理和行动的简单智能体。
- **`05-Advanced-Topics-and-Projects/` (进阶主题与项目)**:
  - 探索 Langchain 中的记忆功能以支持有状态应用。
  - 评估和调试技术 (例如 LangSmith 简介)。
  - 集成各种概念的示例项目。
- **`Cheatsheets-and-Resources/` (速查表与资源)**:
  - Hugging Face 和 Langchain 的快速参考指南。
  - 指向有用的文档、文章、教程和工具的链接。

## 🚀 开始使用

1. **克隆仓库：**
   
   ```bash
   git clone https://github.com/cuwbay/Large-Model-Application-Development-Learning-Path.git
   cd Large-Model-Application-Development-Learning-Path
   ```

2. **搭建环境：**
   
   - 强烈建议使用虚拟环境 (例如 Conda 或 venv)。
   - 有关搭建环境和安装必要库的详细说明，请参见 `00-Introduction/03-Setting-up-Environment.md`。
   - 你可能需要某些服务的 API 密钥 (例如 OpenAI, Dashscope)。有关如何管理这些密钥的说明也将在设置指南中提供。

3. **浏览模块：**
   
   - 从 `00-Introduction` 开始，并按顺序浏览各个文件夹。
   - 大多数代码示例将采用 Jupyter Notebooks (`.ipynb`) 的形式，以便交互式学习。

## 🛠️ 技术与工具

- Python
- Jupyter Notebooks
- Git & GitHub
- Hugging Face (Transformers, Hub)
- Langchain
- 多种向量数据库 (Chroma, FAISS 等)
- LLM APIs (OpenAI, Dashscope 等作为示例)

## 🤝 贡献 (可选)

这主要是一个个人学习项目，但欢迎提出建议、修正或贡献！请随时：

- 提交 issue 来报告错误、提出改进建议或进行提问。
- 提交 pull request 并附带你建议的更改 (对于重大更改，请先提交 issue 进行讨论)。

## 📜 开源许可

本项目采用 MIT 许可证 - 详情请参见 [LICENSE](LICENSE) 文件。

---

学习愉快！让我们一起探索大语言模型的神奇能力吧。✨
