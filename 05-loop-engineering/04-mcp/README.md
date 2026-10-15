<!-- ---
title: "MCP Integration"
description: "Discover tools from Model Context Protocol servers and publish your own"
icon: "plug"
--- -->

# MCP Integration

Stop hand-coding every tool. Connect the agent loop to Model Context Protocol servers to discover tools at runtime — and build a small server of your own to expose capabilities to any MCP client.

## 🎯 What You'll Learn

- Use the `mcp` Python SDK client to list and call tools over a session
- Bridge MCP tool schemas into the agent loop for both Anthropic and OpenAI
- Build a small MCP server with `FastMCP`
- Compare the SDK-bridge approach with Anthropic's native MCP connector

## DeepSeek 示例

启动本地 MCP stdio server，发现 lookup_document 并交给 DeepSeek 调用。

从仓库根目录运行：

```bash
uv run --directory 05-loop-engineering python 04-mcp/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。
The MCP server is local and launched over stdio. The client discovers its schema before exposing the tool to the model.

### 实现说明

本地 stdio server 暴露文档工具，客户端通过 MCP 发现 schema 后桥接到 DeepSeek。此教程固定 `mcp<2`。
