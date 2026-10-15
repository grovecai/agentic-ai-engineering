# Error Handling & Resilience

对连接、超时、限流错误做有界退避重试。

## DeepSeek 示例

对连接、超时、限流错误做有界退避重试。

从仓库根目录运行：

```bash
uv run --directory 07-production python 06-error-handling-resilience/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

连接、超时和限流错误最多重试三次；仅在设置 `DEEPSEEK_FALLBACK_MODEL` 时尝试另一个模型。无效输入与逻辑错误不会无限重试。
