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

`skills.py` 扫描 `skills/*/SKILL.md` 的 front matter，在 system prompt 中只放名称和描述。目录有 `incident-summary` 和 `deployment-checklist` 两个技能，agent 根据任务选择。选中后调用 `load_skill` 获取完整指令；当指令引用附属文件时，再调用 `read_skill_file`。事故总结技能提供 `references/template.md`，因此可以观察三个层级的加载。未加载技能时，附属文件不可读；所有读取都被限制在该技能目录内。

增加技能时，新建 `skills/<name>/SKILL.md`，front matter 的 `name` 必须与目录名一致，并提供 `description`。运行时目录会自动出现在目录提示和工具参数枚举中，无需修改 agent 代码。
