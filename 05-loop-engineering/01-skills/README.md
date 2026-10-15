<!-- ---
title: "Skills"
description: "Inject specialized capabilities on demand with filesystem-based Agent Skills and progressive disclosure"
icon: "sparkles"
--- -->

# Skills

Inject specialized capabilities into the agent on demand with filesystem-based Agent Skills. A lightweight `SKILL.md` catalog stays cheap in the system prompt, while full instructions and bundled assets load only when the agent actually needs them.

## 🎯 What You'll Learn

- Structure a `SKILL.md` with YAML front-matter (name, description) and a Markdown body
- Implement three-tier progressive disclosure: catalog → load skill → read bundled files
- Give the agent `load_skill` and file-reading tools to discover capabilities at runtime
- Guard skill loading against path traversal and keep the base prompt lean

## DeepSeek 示例

按需加载本地 SKILL.md，再由 DeepSeek 使用文档工具回答。

从仓库根目录运行：

```bash
uv run --directory 05-loop-engineering python 01-skills/demo.py
```

设置 `DEEPSEEK_API_KEY`（可放在根目录 `.env`）；可选 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。模型调用会产生 API 费用。

### 实现说明

目录只向模型暴露名称与描述；加载时校验名称并限制读取路径。
