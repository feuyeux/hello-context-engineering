# Hello Context Engineering

- <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents>
- <https://rlancemartin.github.io/2025/06/23/context_engineering/>
- <https://github.com/langchain-ai/context_engineering>

```sh
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Context Engineering Strategies | 上下文工程策略

In this repo, we cover some common strategies — write, select, compress, and isolate — for agent context engineering by reviewing various popular agents and papers. We then explain how LangGraph is designed to support them!

在本仓库中,我们通过回顾各种流行的智能体和论文,介绍了智能体上下文工程的一些常见策略——写入、选择、压缩和隔离。然后我们解释了LangGraph是如何设计来支持这些策略的!

- **Writing context | 写入上下文** - saving it outside the context window to help an agent perform a task. | 将信息保存在上下文窗口之外以帮助智能体执行任务。
- **Selecting context | 选择上下文** - pulling it into the context window to help an agent perform a task. | 将信息拉入上下文窗口以帮助智能体执行任务。
- **Compressing context | 压缩上下文** - retaining only the tokens required to perform a task. | 仅保留执行任务所需的令牌。
- **Isolating context | 隔离上下文** - splitting it up to help an agent perform a task. | 将上下文拆分以帮助智能体执行任务。

### 1. Write Context | 写入上下文

**Description | 描述**: Saving information outside the context window to help an agent perform a task. | 将信息保存在上下文窗口之外以帮助智能体执行任务。

### 📚 **What's Covered in [1_write_context.ipynb](context_engineering/1_write_context.ipynb) | 内容概览**

- **Scratchpads in LangGraph | LangGraph中的草稿本**: Using state objects to persist information during agent sessions | 使用状态对象在智能体会话期间持久化信息
  - StateGraph implementation with TypedDict for structured data | 使用TypedDict实现StateGraph以支持结构化数据
  - Writing context to state and accessing it across nodes | 将上下文写入状态并在节点间访问
  - Checkpointing for fault tolerance and pause/resume workflows | 检查点机制用于容错和暂停/恢复工作流
- **Memory Systems | 记忆系统**: Long-term persistence across multiple sessions | 跨多个会话的长期持久化
  - InMemoryStore for storing memories with namespaces | 使用命名空间的InMemoryStore存储记忆
  - Integration with checkpointing for comprehensive memory management | 与检查点集成以实现全面的记忆管理
  - Examples of storing and retrieving jokes with user context | 存储和检索带有用户上下文的笑话示例

## 2. Select Context | 选择上下文

**Description | 描述**: Pulling information into the context window to help an agent perform a task. | 将信息拉入上下文窗口以帮助智能体执行任务。

### 📚 **What's Covered in [2_select_context.ipynb](context_engineering/2_select_context.ipynb) | 内容概览**

- **Scratchpad Selection | 草稿本选择**: Fetching specific context from agent state | 从智能体状态中获取特定上下文
  - Selective state access in LangGraph nodes | 在LangGraph节点中选择性访问状态
  - Multi-step workflows with context passing between nodes | 节点间传递上下文的多步骤工作流
- **Memory Retrieval | 记忆检索**: Selecting relevant memories for current tasks | 为当前任务选择相关记忆
  - Namespace-based memory retrieval | 基于命名空间的记忆检索
  - Context-aware memory selection to avoid irrelevant information | 上下文感知的记忆选择以避免无关信息
- **Tool Selection | 工具选择**: RAG-based tool retrieval for large tool sets | 基于RAG的大型工具集检索
  - LangGraph Bigtool library for semantic tool search | 用于语义工具搜索的LangGraph Bigtool库
  - Embedding-based tool description matching | 基于嵌入的工具描述匹配
  - Examples with math library functions and semantic retrieval | 数学库函数和语义检索示例
- **Knowledge Retrieval | 知识检索**: RAG implementation for external knowledge | 外部知识的RAG实现
  - Vector store creation with document splitting | 文档拆分的向量存储创建
  - Retriever tools integrated with LangGraph agents | 与LangGraph智能体集成的检索工具
  - Multi-turn conversations with context-aware retrieval | 具有上下文感知检索的多轮对话

## 3. Compress Context | 压缩上下文

**Description | 描述**: Retaining only the tokens required to perform a task. | 仅保留执行任务所需的令牌。

### 📚 **What's Covered in [3_compress_context.ipynb](context_engineering/3_compress_context.ipynb) | 内容概览**

- **Conversation Summarization | 对话摘要**: Managing long agent trajectories | 管理长智能体轨迹
  - End-to-end conversation summarization after task completion | 任务完成后的端到端对话摘要
  - Token usage optimization (demonstrated reduction from 115k to 60k tokens) | 令牌使用优化(演示了从115k到60k令牌的减少)
- **Tool Output Compression | 工具输出压缩**: Reducing token-heavy tool responses | 减少令牌密集的工具响应
  - Summarization of RAG retrieval results | RAG检索结果的摘要
  - Integration with LangGraph tool nodes | 与LangGraph工具节点集成
  - Practical examples with blog post retrieval and summarization | 博客文章检索和摘要的实际示例
- **State-based Compression | 基于状态的压缩**: Using LangGraph state for context management | 使用LangGraph状态进行上下文管理
  - Custom state schemas with summary fields | 带摘要字段的自定义状态模式
  - Conditional summarization based on context length | 基于上下文长度的条件摘要

## 4. Isolate Context | 隔离上下文

**Description | 描述**: Splitting up context to help an agent perform a task. | 将上下文拆分以帮助智能体执行任务。

### 📚 **What's Covered in [4_isolate_context.ipynb](context_engineering/4_isolate_context.ipynb) | 内容概览**

- **Multi-Agent Systems | 多智能体系统**: Separating concerns across specialized agents | 跨专业智能体分离关注点
  - Supervisor architecture for task delegation | 用于任务委派的监督者架构
  - Specialized agents with isolated context windows (math expert, research expert) | 具有隔离上下文窗口的专业智能体(数学专家、研究专家)
  - LangGraph Supervisor library implementation | LangGraph Supervisor库实现
- **Sandboxed Environments | 沙盒环境**: Isolating context in execution environments | 在执行环境中隔离上下文
  - PyodideSandboxTool for secure code execution | 用于安全代码执行的PyodideSandboxTool
  - State isolation outside the LLM context window | LLM上下文窗口之外的状态隔离
  - Examples of context storage in sandbox variables | 沙盒变量中的上下文存储示例
- **State-based Isolation | 基于状态的隔离**: Using LangGraph state schemas for context separation | 使用LangGraph状态模式进行上下文分离
  - Structured state design for selective context exposure | 用于选择性上下文暴露的结构化状态设计
  - Field-based isolation within agent state objects | 智能体状态对象内的基于字段的隔离
