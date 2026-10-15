<!-- ---
title: "Monitoring & Observability"
description: "Metrics, structured logging, and distributed tracing for production agents"
icon: "activity"
--- -->

# Monitoring & Observability

Metrics, structured logging, and distributed tracing for production agents. Know when things break before your users tell you.

## 🎯 What You'll Learn

- Instrument agents with meaningful metrics
- Implement structured logging for agent actions
- Set up distributed tracing across agent calls
- Build dashboards and alerts for production health

## DeepSeek 示例

结构化请求日志、追踪 ID、时延和 token 计数。

从仓库根目录运行：

```bash
uv run --directory 07-production python 03-monitoring-observability/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

每次请求输出 trace ID、延迟与聚合 token 计数；真实部署可将相同字段接入 OpenTelemetry 和指标系统。
