# 05: 搭建你的开发环境 (Setting up Your Development Environment)

在正式开始编写代码探索大型语言模型 (LLM) 应用之前，我们需要搭建一个稳定、高效的开发环境。本节将指导你完成必要的软件安装和配置。

## 1. 构建虚拟环境

在进行 Python 项目开发时，强烈建议为每个项目使用独立的虚拟环境。

*   **为什么需要虚拟环境？**
    *   **依赖隔离：** 不同项目可能需要不同版本的库。虚拟环境可以防止不同项目之间的库版本冲突。
    *   **环境一致性：** 确保你的项目在不同开发或部署环境中使用相同的依赖版本。
    *   **保持全局 Python 环境整洁。**

*   **常用的虚拟环境工具：**
    *   **`conda` (Anaconda / Miniconda)：**
        如果你安装了 Anaconda 或 Miniconda (一个更轻量级的 Anaconda 版本，推荐初学者使用)，`conda` 是一个强大的包管理和环境管理工具。
        1.  **安装 Miniconda (如果尚未安装)：**
            访问 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html) 下载并安装，简单百度一下即可找到各种安装Anaconda的教程。
        2.  **创建虚拟环境：**
            ```bash
            conda create --name llm_env python=3.10
            ```
            (将 `"llm_env"` 替换为你喜欢的环境名称，`"python=3.10"` 指定 Python 版本。)
        3.  **激活虚拟环境：**
            ```bash
            conda activate llm_env
            ```
        4.  **退出虚拟环境：**
            ```bash
            conda deactivate
            ```
            
## 2. 安装核心 Python 库

激活虚拟环境后，我们使用 `pip` (Python 的包安装器) 来安装 LLM 应用开发所需的核心库。

创建一个名为 `requirements.txt` 的文件在你的项目根目录下，并填入以下内容 ：

```bash
# requirements.txt

# Jupyter Environment
jupyterlab>=3.0.0
notebook>=6.0.0
ipykernel>=6.0.0

# Hugging Face Core Libraries
transformers>=4.30.0
datasets>=2.0.0
evaluate>=0.4.0
accelerate>=0.20.0 # For efficient model loading and training on multi-GPU/TPU
sentence-transformers>=2.2.0 # For easy sentence embeddings

# Langchain Core Libraries
langchain>=0.1.0
langchain-openai>=0.0.1 # For OpenAI models
langchain-community>=0.0.1 # For community integrations (vector stores, other LLMs, etc.)
# langchain-huggingface # (可选) 更方便地集成Hugging Face本地模型到Langchain

# API Key Management
python-dotenv>=1.0.0

# Common Vector Stores (选择性安装，根据后续使用)
# chromadb>=0.4.0 # ChromaDB vector store
# faiss-cpu>=1.7.0 # FAISS vector store (CPU version). For GPU: faiss-gpu
# tiktoken>=0.4.0 # Used by Langchain for OpenAI token counting

# Other useful libraries
requests>=2.20.0 # For making HTTP requests
beautifulsoup4>=4.0.0 # For web scraping (if needed for document loaders)
pypdf>=3.0.0 # For loading PDF documents
```

然后在终端中运行 (确保虚拟环境已激活)：
```
pip install -r requirements.txt
```
## 3. IDE / 代码编辑器推荐
1. PyCharm (JetBrains):功能非常强大的 Python IDE，有免费的社区版和付费的专业版。
对虚拟环境、调试、版本控制等支持良好。如果是小白/学生，建议使用PyCharm,可申请高校免费试用PyCharm。


2. JupyterLab / Jupyter Notebook:
如果你主要进行探索性数据分析和快速原型验证，直接在浏览器中使用 JupyterLab 或 Jupyter Notebook 也是一个不错的选择。
```bash
pip install jupyterlab 
```
在终端运行 jupyter lab 即可启动。

## 4.GPU支持
如果你计划在本地运行一些中大型的 Hugging Face 模型，可以选择。本项目侧重于应用开发，所以暂时不计划补全这部分内容