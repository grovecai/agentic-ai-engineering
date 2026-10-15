<!-- ---
title: "Security & Guardrails"
description: "Authentication, sandboxing, and prompt injection defense"
icon: "lock"
--- -->

# Security & Guardrails

Authentication, sandboxing, prompt injection defense, and tool-use permissions. Build agents that are safe to expose to untrusted input.

## 🎯 What You'll Learn

- Implement authentication and authorization for agent access
- Sandbox tool execution to limit blast radius
- Defend against prompt injection attacks
- Design permission systems for tool-use safety

## DeepSeek 示例

限制输入大小和工具目标，把检索文本作为数据处理。

从仓库根目录运行：

```bash
uv run --directory 07-production python 05-security-guardrails/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

HTTP 边界校验应用密钥；问答服务还限制输入长度和工具 ID。模型会被要求把检索文本当作数据。这些措施不能单独抵御所有提示注入，部署时还需平台级隔离。
