# DeepSeek framework comparison design

Nine implementations answer the same question from the same two-document
knowledge base. Each offers a `lookup` tool; a successful answer cites a
document ID. The raw SDK version is the baseline. The others use each
framework's native agent/tool abstraction, with DeepSeek configured through
its OpenAI-compatible endpoint. Each lesson has an isolated dependency set,
because these frameworks have substantial and sometimes conflicting trees.

The comparison records the code needed to configure the model, register a
tool, execute a task, and extract the result. It does not claim one framework
is universally better from a single example. `DEEPSEEK_API_KEY` is required
for live calls; `DEEPSEEK_BASE_URL` and `DEEPSEEK_MODEL` may override defaults.

The README in each folder includes installation and run commands and any
provider compatibility caveat. The testable shared `reference.py` contains
the exact task and deterministic lookup behavior, so results are comparable.
