import os
import sys
from typing import TypedDict, List, Optional
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END

# Reuse the existing planner, researcher, critic, and writer functions
from pipeline import (
    planner,
    researcher,
    critic,
    writer,
    ResearchFindings,
    CriticReview
)

# Reconfigure stdout to use UTF-8 to prevent encoding crashes on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# 2. Define a shared state schema called ResearchState
class ResearchState(TypedDict):
    research_question: str
    sub_questions: List[str]
    findings: List[ResearchFindings]
    critic_review: Optional[CriticReview]
    loop_count: int
    final_report: Optional[str]

# 3. Define the nodes of the graph

def plan_node(state: ResearchState) -> dict:
    """Planner node: calls planner() and populates sub_questions in the state."""
    question = state["research_question"]
    sub_questions_obj = planner(question)
    return {"sub_questions": sub_questions_obj.sub_questions}

def research_node(state: ResearchState) -> dict:
    """Researcher node: loops through sub-questions not yet researched,

    calls researcher() on each, and appends to findings.
    """
    sub_qs = state.get("sub_questions") or []
    current_findings = state.get("findings") or []
    
    # Identify which sub-questions have already been researched
    researched_qs = {f.sub_question for f in current_findings}
    
    new_findings = list(current_findings)
    for idx, sub_q in enumerate(sub_qs, 1):
        if sub_q not in researched_qs:
            try:
                finding = researcher(sub_q)
                new_findings.append(finding)
            except Exception as e:
                print(f"Error researching sub-question '{sub_q}': {e}")
                
    return {"findings": new_findings}

def critique_node(state: ResearchState) -> dict:
    """Critique node: calls critic(), stores review, and increments loop_count.

    If gaps are found and we haven't reached loop limits, we also append
    the gaps to sub_questions.
    """
    question = state["research_question"]
    current_findings = state.get("findings") or []
    current_loop = state.get("loop_count", 0)
    
    review = critic(question, current_findings)
    
    # Determine updated sub-questions
    new_sub_qs = list(state.get("sub_questions") or [])
    if not review.is_sufficient and review.gaps:
        # Only add gaps if we are going to loop (loop_count < 2 before incrementing)
        if current_loop < 2:
            for gap in review.gaps:
                if gap not in new_sub_qs:
                    new_sub_qs.append(gap)
                    
    return {
        "critic_review": review,
        "loop_count": current_loop + 1,
        "sub_questions": new_sub_qs
    }

def write_node(state: ResearchState) -> dict:
    """Writer node: calls writer() and stores final report in the state."""
    question = state["research_question"]
    current_findings = state.get("findings") or []
    
    report = writer(question, current_findings)
    return {"final_report": report}

# 4. Add conditional edges routing logic
def route_after_critique(state: ResearchState) -> str:
    """Routes to 'write' if the findings are sufficient or loop count is >= 2,

    otherwise routes back to 'research'.
    """
    review = state.get("critic_review")
    loop_count = state.get("loop_count", 0)
    
    if review is None or review.is_sufficient or loop_count >= 2:
        return "write"
    return "research"

# Build LangGraph StateGraph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("plan", plan_node)
workflow.add_node("research", research_node)
workflow.add_node("critique", critique_node)
workflow.add_node("write", write_node)

# Add static transitions
workflow.add_edge(START, "plan")
workflow.add_edge("plan", "research")
workflow.add_edge("research", "critique")

# Add conditional transitions from critique
workflow.add_conditional_edges(
    "critique",
    route_after_critique,
    {
        "write": "write",
        "research": "research"
    }
)
workflow.add_edge("write", END)

# Compile graph
app = workflow.compile()

# 5. Write run_graph_pipeline function
def run_graph_pipeline(research_question: str) -> str:
    """Initialize state, execute compiled graph, and return final report."""
    initial_state: ResearchState = {
        "research_question": research_question,
        "sub_questions": [],
        "findings": [],
        "critic_review": None,
        "loop_count": 0,
        "final_report": None
    }
    
    final_report = ""
    # Stream the graph execution step-by-step
    for chunk in app.stream(initial_state):
        for node, output in chunk.items():
            print(f"\n>>> Executing Node: '{node}'")
            if "sub_questions" in output:
                print(f"    Sub-questions planned/updated: {len(output['sub_questions'])} total")
            if "critic_review" in output:
                review = output["critic_review"]
                print(f"    Critic Review: is_sufficient={review.is_sufficient}, gaps={len(review.gaps)}")
                print(f"    Reasoning: {review.reasoning}")
            if "loop_count" in output:
                print(f"    Loop Count: {output['loop_count']}")
            if "final_report" in output:
                final_report = output["final_report"]
                
    return final_report

# 6. main execution block
if __name__ == "__main__":
    print("=========================================")
    print("Welcome to the LangGraph Research Pipeline")
    print("=========================================\n")
    
    try:
        research_question = input("Enter your research question: ").strip()
        if not research_question:
            print("Error: Research question cannot be empty.")
            sys.exit(1)
            
        final_report = run_graph_pipeline(research_question)
        
        # Save the report to report_v2.md
        with open("report_v2.md", "w", encoding="utf-8") as f:
            f.write(final_report)
            
        print("\nReport saved to report_v2.md")
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as err:
        print(f"\nPipeline failed: {err}")
        sys.exit(1)
