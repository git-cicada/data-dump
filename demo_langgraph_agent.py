"""
=============================================================================
DEMO: LangGraph Stateful Graph Agent
=============================================================================
Pattern  : Stateful Graph with conditional routing, checkpointing,
           and an optional human-in-the-loop interrupt
Framework: LangGraph + LangChain-OpenAI
LLM      : GPT-4o

Install deps:
    pip install langgraph langchain langchain-openai langchain-community duckduckgo-search

Set env var:
    export OPENAI_API_KEY="sk-..."
=============================================================================
"""

import os
from typing import TypedDict, Annotated, List
import operator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import DuckDuckGoSearchRun, tool


# ─── 1. Define the typed shared state ────────────────────────────────────────
#   Everything the graph nodes read and write lives here.
#   Annotated[List, operator.add] means new messages are APPENDED, not replaced.

class AgentState(TypedDict):
    messages:  Annotated[List, operator.add]   # full conversation history
    query:     str                              # original user query
    attempts:  int                              # how many LLM calls so far
    final_ans: str                              # populated on completion


# ─── 2. Initialise LLM and bind tools ────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

search = DuckDuckGoSearchRun()


@tool
def calculate(expression: str) -> float:
    """
    Safely evaluate a mathematical expression and return the result.
    Example: calculate("(42 * 1.5) + 100")
    """
    try:
        # Only allow safe characters for arithmetic
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: unsafe characters in expression"
        return eval(expression)  # noqa: S307 (demo only — use sympy in production)
    except Exception as e:
        return f"Error: {e}"


tools = [search, calculate]
llm_with_tools = llm.bind_tools(tools)


# ─── 3. Graph NODES — each is a pure function on AgentState ──────────────────

def call_model(state: AgentState) -> AgentState:
    """
    LLM reasoning node.
    Reads the current messages, calls the LLM (with tools bound),
    and appends the response to messages.
    """
    system = SystemMessage(
        content=(
            "You are a helpful research assistant. "
            "Use the search tool to find current information. "
            "Use the calculate tool for any arithmetic. "
            "When you have enough information, provide a final answer."
        )
    )
    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "attempts": state.get("attempts", 0) + 1,
    }


# ToolNode automatically executes any tool_calls found in the last AI message
tool_node = ToolNode(tools)


# ─── 4. Routing logic (conditional edge) ─────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """
    Decide what happens after the LLM responds:
      - If the LLM requested tool calls → go to 'tools' node
      - If max attempts reached         → go to END
      - Otherwise                       → END (final answer ready)
    """
    last_msg = state["messages"][-1]
    if state.get("attempts", 0) >= 6:
        return END   # hard cap — prevents runaway loops

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return END


# ─── 5. Build the graph ───────────────────────────────────────────────────────

def build_graph(interrupt_for_human: bool = False):
    """
    Assemble and compile the agent graph.

    Args:
        interrupt_for_human: If True, the graph pauses BEFORE executing tools,
                             allowing a human to inspect/approve the tool call.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    # Entry point
    graph.set_entry_point("agent")

    # Conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END:     END,
        }
    )

    # Tool node always loops back to agent
    graph.add_edge("tools", "agent")

    # Compile with in-memory checkpointer (swap for SqliteSaver / RedisSaver in prod)
    checkpointer = MemorySaver()

    compile_kwargs = {"checkpointer": checkpointer}
    if interrupt_for_human:
        compile_kwargs["interrupt_before"] = ["tools"]  # human-in-the-loop

    return graph.compile(**compile_kwargs)


# ─── 6. DEMO A — Basic stateful run ──────────────────────────────────────────

print("\n" + "=" * 60)
print("DEMO A — Stateful agent run with tool use")
print("=" * 60)

app = build_graph()

# thread_id scopes the checkpoint — same ID = same persistent memory
config = {"configurable": {"thread_id": "session_001"}}

initial_state = {
    "messages":  [HumanMessage(content="What are the latest AI agent frameworks in 2024?")],
    "query":     "AI agent frameworks 2024",
    "attempts":  0,
    "final_ans": "",
}

result = app.invoke(initial_state, config=config)
final = result["messages"][-1].content
print(f"\n[FINAL ANSWER]\n{final}")


# ─── 7. DEMO B — Resume same session (state persisted) ───────────────────────

print("\n" + "=" * 60)
print("DEMO B — Follow-up in same thread (state persisted)")
print("=" * 60)

follow_up_state = {
    "messages":  [HumanMessage(content="Which of those frameworks is easiest to learn?")],
    "query":     "easiest AI framework",
    "attempts":  0,
    "final_ans": "",
}

# Same config / thread_id — the agent has access to the previous conversation
result2 = app.invoke(follow_up_state, config=config)
print(f"\n[FOLLOW-UP ANSWER]\n{result2['messages'][-1].content}")


# ─── 8. DEMO C — Human-in-the-loop ───────────────────────────────────────────

print("\n" + "=" * 60)
print("DEMO C — Human-in-the-loop (interrupt_before tools)")
print("=" * 60)

hil_app = build_graph(interrupt_for_human=True)
hil_config = {"configurable": {"thread_id": "hil_session_001"}}

hil_state = {
    "messages":  [HumanMessage(content="Calculate: (1024 * 3.5) + 512")],
    "query":     "math calculation",
    "attempts":  0,
    "final_ans": "",
}

# Run until the first interrupt (just before the tools node)
partial = hil_app.invoke(hil_state, hil_config)
print("\n[AGENT IS PAUSED — pending tool call inspection]")

# Inspect the pending tool call
last = partial["messages"][-1]
if hasattr(last, "tool_calls") and last.tool_calls:
    call = last.tool_calls[0]
    print(f"  Tool requested : {call['name']}")
    print(f"  Arguments      : {call['args']}")

# Simulate human approval: resume by passing None as input
print("\n[HUMAN APPROVED — resuming graph …]")
final_state = hil_app.invoke(None, hil_config)  # None = resume from checkpoint
print(f"\n[CALCULATION RESULT]\n{final_state['messages'][-1].content}")


# ─── 9. Inspect graph structure ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("DEMO D — Graph structure (nodes and edges)")
print("=" * 60)
print("Nodes:", list(app.nodes.keys()))


# ─── Key concepts recap ──────────────────────────────────────────────────────
print("""
─── KEY CONCEPTS ────────────────────────────────────────────────────
• AgentState (TypedDict)        — typed shared state across all nodes
• Annotated[List, operator.add] — append-only message list
• add_node / add_edge           — build the directed graph explicitly
• add_conditional_edges         — route based on state (e.g. tool_calls present?)
• ToolNode                      — auto-executes tool_calls from AI messages
• MemorySaver / SqliteSaver     — checkpointer for session persistence
• thread_id in config           — scopes state to a specific conversation
• interrupt_before=["tools"]    — pause for human review before tool execution
• app.invoke(None, config)      — resume from a checkpoint after interrupt
─────────────────────────────────────────────────────────────────────
""")
