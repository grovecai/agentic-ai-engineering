<!-- ---
title: "Deployment Strategies"
description: "Containers, serverless, and scaling patterns for agents"
icon: "upload-cloud"
--- -->

# Deployment Strategies

Containers, serverless, and scaling patterns. How to package and ship agents that handle variable load and long-running tasks.

## 🎯 What You'll Learn

- Package agents in containers for consistent deployment
- Choose between serverless and long-running deployments
- Scale agents to handle variable traffic loads
- Handle long-running tasks without timeouts

## DeepSeek 示例

FastAPI 健康检查和问答端点，以及容器构建文件。

从仓库根目录运行：

```bash
uv run --directory 07-production uvicorn app:app --app-dir 02-deployment-strategies --port 8000
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）和 `DEMO_SERVER_API_KEY`；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" -H "x-demo-key: $DEMO_SERVER_API_KEY" \
  -d '{"text":"What is the refund window?"}'
```

从 `07-production` 目录构建镜像：`docker build -f 02-deployment-strategies/Dockerfile -t deepseek-support .`。

### 实现说明

`GET /health` 不调用模型。`POST /answer` 需要 `x-demo-key` 请求头，服务端通过 `DEMO_SERVER_API_KEY` 验证；API 密钥由 `DEEPSEEK_API_KEY` 提供。容器镜像不包含密钥。
