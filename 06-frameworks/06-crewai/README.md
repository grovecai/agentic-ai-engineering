<!-- ---
title: "CrewAI"
description: "Role-based multi-agent collaboration framework"
icon: "users"
--- -->

# CrewAI

Role-based multi-agent collaboration. Define agents with roles, goals, and backstories that work together on complex tasks.

## 🎯 What You'll Learn

- Define agents with distinct roles, goals, and backstories
- Orchestrate crews for collaborative task completion
- Configure agent communication and delegation patterns
- Build multi-agent workflows with clear responsibilities

## DeepSeek 示例

CrewAI 的角色、任务和工具。

从仓库根目录运行：

```bash
uv run --directory 06-frameworks/06-crewai python demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
All nine framework examples use the same question and `lookup_document` implementation from `../reference.py`. Framework dependencies are isolated in this folder.

### 实现说明

CrewAI 的首次 trace 上传提示在本地教学脚本中关闭；任务结果不依赖远端 trace 服务。 框架升级后接口可能变化；本目录的 `uv.lock` 固定了验证过的依赖组合。参见[官方文档](https://docs.crewai.com/en/concepts/llms)。
