<!-- ---
title: "Cost Optimization"
description: "Token budgets, caching strategies, and model routing for cost control"
icon: "dollar-sign"
--- -->

# Cost Optimization

Token budgets, caching strategies, model routing, and knowing when a smaller model is the right call. Keep costs predictable as usage grows.

## 🎯 What You'll Learn

- Set and enforce token budgets per request
- Implement caching to reduce redundant API calls
- Route requests to cheaper models when appropriate
- Track and forecast costs as usage scales

## DeepSeek 示例

限制输出 token，缓存重复问题并展示真实用量。

从仓库根目录运行：

```bash
uv run --directory 07-production python 04-cost-optimization/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

短问题路由至 Flash，超过 120 字符的复杂问题可路由至 `DEEPSEEK_COMPLEX_MODEL`。这是透明的教学规则，生产中应由评测数据决定路由阈值。真实 token 用量来自 API；设置 `DEEPSEEK_INPUT_USD_PER_MILLION` 与 `DEEPSEEK_OUTPUT_USD_PER_MILLION` 可按当前价格估算美元成本。
