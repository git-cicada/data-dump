"""
=============================================================================
DEMO: LangChain ReAct Agent
=============================================================================
Pattern  : ReAct (Reason → Act → Observe loop)
Framework: LangChain + LangChain-OpenAI
LLM      : GPT-4o  (swap for any ChatOpenAI-compatible model)

Install deps:
    pip install langchain langchain-openai langchain-community duckduckgo-search wikipedia

Set env var:
    export OPENAI_API_KEY="sk-..."
=============================================================================
"""

import os
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import WikipediaQueryRun
from langchain.tools.wikipedia.api import WikipediaAPIWrapper
from langchain.tools import tool
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage


# ─── 1. Initialise the LLM ───────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,        # deterministic for demos
    max_tokens=2000,
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# ─── 2. Define tools ─────────────────────────────────────────────────────────

# Built-in tools
search = DuckDuckGoSearchRun()
wiki   = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


# Custom tool — any Python function decorated with @tool becomes a tool
@tool
def word_count(text: str) -> int:
    """
    Returns the exact word count of a given piece of text.
    Use this when you need to count words in a response.
    """
    return len(text.split())


tools = [search, wiki, word_count]


# ─── 3. Load the standard ReAct prompt from LangChain Hub ────────────────────
#   The prompt shapes the Thought/Action/Observation trace

prompt = hub.pull("hwchase17/react")


# ─── 4. Build the agent ──────────────────────────────────────────────────────

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


# ─── 5. Wrap in an executor (manages the Reason→Act→Observe loop) ────────────

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,              # prints full thought trace
    max_iterations=6,          # prevents infinite loops
    handle_parsing_errors=True,
)


# ─── 6. Run a single query ───────────────────────────────────────────────────

print("\n" + "="*60)
print("DEMO 1 — Single query with tool use")
print("="*60)

result = executor.invoke({
    "input": (
        "What are the top 3 AI agent frameworks in 2024? "
        "For each one, give a one-sentence summary."
    )
})
print("\n[FINAL ANSWER]\n", result["output"])


# ─── 7. Multi-turn conversation with memory ──────────────────────────────────

print("\n" + "="*60)
print("DEMO 2 — Multi-turn with ConversationBufferMemory")
print("="*60)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

# For multi-turn you typically use a different prompt that includes chat_history.
# Here we demonstrate memory object usage directly.
memory.chat_memory.add_user_message("Tell me about LangChain in one sentence.")
memory.chat_memory.add_ai_message(
    "LangChain is an open-source framework for building LLM-powered applications "
    "with chains, agents, and a vast tool ecosystem."
)

# Retrieve history to show it works
history = memory.load_memory_variables({})
print("\n[Stored memory]\n", history)

follow_up = executor.invoke({
    "input": "Now search the web for the latest LangChain release notes and summarise."
})
print("\n[FOLLOW-UP ANSWER]\n", follow_up["output"])


# ─── Key concepts recap ──────────────────────────────────────────────────────
print("""
─── KEY CONCEPTS ────────────────────────────────────────────────────
• ChatOpenAI     — wraps any OpenAI model; swap model= for another
• @tool          — decorator that turns any Python function into a tool
• create_react_agent — binds LLM + tools + ReAct prompt
• AgentExecutor  — manages the Thought → Action → Observation loop
• verbose=True   — see every reasoning step in the terminal
• max_iterations — guard against runaway loops
• ConversationBufferMemory — keeps the full chat history in RAM
─────────────────────────────────────────────────────────────────────
""")
