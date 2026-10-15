"""Explicit LangGraph nodes, conditional tool edge, and checkpointed state."""

import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference import BASE_URL, INSTRUCTIONS, MODEL, QUESTION, key, lookup_document


@tool
def lookup(doc_id: str) -> str:
    """Read the local runbook or incident document by ID."""
    return lookup_document(doc_id)


if __name__ == "__main__":
    model = ChatOpenAI(
        model=MODEL, api_key=key(), base_url=BASE_URL, extra_body={"thinking": {"type": "disabled"}}
    )
    bound_model = model.bind_tools([lookup])

    def call_model(state: MessagesState) -> dict:
        return {"messages": [bound_model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode([lookup]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    graph = builder.compile(checkpointer=MemorySaver())
    result = graph.invoke(
        {
            "messages": [
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": QUESTION},
            ]
        },
        config={"configurable": {"thread_id": "reference-demo"}, "recursion_limit": 10},
    )
    print(result["messages"][-1].content)
