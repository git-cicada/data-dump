"""
=============================================================================
DEMO: CrewAI Multi-Agent Crew
=============================================================================
Pattern  : Role-based multi-agent collaboration (Sequential + Hierarchical)
Framework: CrewAI
LLM      : GPT-4o

Install deps:
    pip install crewai crewai-tools

Set env var:
    export OPENAI_API_KEY="sk-..."
    export SERPER_API_KEY="..."   # optional — for SerperDevTool web search
                                  # get a free key at serper.dev
=============================================================================
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool


# ─── 1. Initialise tools ─────────────────────────────────────────────────────

search_tool = SerperDevTool()    # requires SERPER_API_KEY
file_tool   = FileWriterTool()   # writes output to local files


# ─── 2. Define specialist agents with personas ───────────────────────────────

researcher = Agent(
    role="Senior Research Analyst",
    goal=(
        "Find accurate, up-to-date information about AI agent frameworks "
        "and produce a structured, evidence-based report."
    ),
    backstory=(
        "You are a seasoned technology analyst with 10 years of experience "
        "benchmarking developer tools and open-source frameworks. "
        "You are known for concise, data-driven insights."
    ),
    tools=[search_tool],
    llm="gpt-4o",
    verbose=True,
    memory=True,           # agent remembers context across tasks
    max_iter=5,            # limit reasoning iterations
)

writer = Agent(
    role="Technical Content Strategist",
    goal=(
        "Transform raw research into a clear, well-structured article "
        "that a senior software engineer can act on immediately."
    ),
    backstory=(
        "Award-winning technical writer with a background in software engineering. "
        "Specialises in making complex AI topics accessible without dumbing them down."
    ),
    llm="gpt-4o",
    verbose=True,
    allow_delegation=False,  # writer works independently
)

reviewer = Agent(
    role="Quality Assurance Lead",
    goal="Ensure all content is factually accurate, well-structured and free of jargon.",
    backstory=(
        "Former principal engineer turned editor. Expert at spotting technical inaccuracies "
        "and improving clarity without changing the author's voice."
    ),
    llm="gpt-4o",
    verbose=True,
)


# ─── 3. Define tasks ─────────────────────────────────────────────────────────

research_task = Task(
    description=(
        "Research the top 4 AI agent frameworks in 2024: LangChain, CrewAI, "
        "Google ADK, and LangGraph. For each: GitHub stars, primary use case, "
        "key strengths, and one notable production deployment."
    ),
    expected_output=(
        "A structured report with four sections (one per framework), "
        "each containing: overview, GitHub popularity, strengths, and a real-world example."
    ),
    agent=researcher,
    output_file="research_output.md",    # saves result to a file
)

write_task = Task(
    description=(
        "Using the research report, write a 400-word blog post titled "
        "'Choosing the Right AI Agent Framework in 2024'. "
        "Include an intro, a comparison paragraph, and a decision guide."
    ),
    expected_output=(
        "A polished blog post: intro paragraph, 2-3 body paragraphs, "
        "and a clear decision guide. Approx 400 words."
    ),
    agent=writer,
    context=[research_task],    # feeds the researcher's output as context
    output_file="blog_post.md",
)

review_task = Task(
    description=(
        "Review the blog post for factual accuracy, readability, and structure. "
        "Provide a brief review note and suggest up to 3 specific improvements."
    ),
    expected_output="A review note (max 150 words) with up to 3 numbered improvement suggestions.",
    agent=reviewer,
    context=[write_task],
)


# ─── 4. DEMO A — Sequential crew ─────────────────────────────────────────────

print("\n" + "="*60)
print("DEMO A — Sequential process (research → write → review)")
print("="*60)

crew_sequential = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, write_task, review_task],
    process=Process.sequential,   # tasks run one after another in order
    verbose=True,
    memory=True,                  # shared crew-level memory across agents
)

result_seq = crew_sequential.kickoff(
    inputs={
        "topic": "AI agent frameworks 2026",
        "audience": "senior software engineers",
    }
)

print("\n[SEQUENTIAL RESULT — final task output]\n")
print(result_seq.raw)


# ─── 5. DEMO B — Hierarchical crew (manager delegates) ───────────────────────

print("\n" + "="*60)
print("DEMO B — Hierarchical process (manager delegates to agents)")
print("="*60)

# In hierarchical mode CrewAI automatically creates a manager LLM
# that reads all task descriptions and decides which agent to delegate to.
crew_hierarchical = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o",          # the manager uses GPT-4o to coordinate
    verbose=True,
    memory=True,
)

result_hier = crew_hierarchical.kickoff(
    inputs={"topic": "LangGraph vs CrewAI for enterprise", "audience": "CTOs"}
)

print("\n[HIERARCHICAL RESULT]\n")
print(result_hier.raw)


# ─── Key concepts recap ──────────────────────────────────────────────────────
print("""
─── KEY CONCEPTS ────────────────────────────────────────────────────
• Agent         — role + goal + backstory + tools = a specialist worker
• Task          — description + expected_output + agent assignment
• context=[]    — pass output of previous tasks as input to the next
• output_file=  — automatically saves task output to a local file
• Process.sequential   — tasks run in strict order, output flows forward
• Process.hierarchical — manager LLM reads tasks and delegates dynamically
• memory=True   — crew shares a common memory store across all agents
• max_iter      — limits per-agent reasoning loops to prevent runaway calls
─────────────────────────────────────────────────────────────────────
""")
