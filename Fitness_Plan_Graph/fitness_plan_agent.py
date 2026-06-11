import sys
import operator
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# -----------------------------------
# CONFIG
# -----------------------------------

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)


# -----------------------------------
# STATE
# -----------------------------------

class FitnessState(BaseModel):

    user_goal: str = ""
    user_level: str = ""

    cardio_plan: str = ""
    strength_plan: str = ""
    recovery_plan: str = ""

    beginner_plan: bool = False

    final_plan: str = ""

    messages: Annotated[list, operator.add] = []


# -----------------------------------
# NODE 1
# CARDIO SPECIALIST
# -----------------------------------

def suggest_cardio_plan(state: FitnessState) -> dict:

    response = llm.invoke(
        f"You are a cardio fitness expert. "
        f"The user's goal is: '{state.user_goal}'. "
        f"The fitness level is: '{state.user_level}'. "
        f"Suggest ONE cardio workout plan. "
        f"Include exercise name, duration, and frequency. "
        f"Keep it under 5 sentences."
    )

    return {
        "cardio_plan": response.content,
        "messages": ["[suggest_cardio_plan] Done"]
    }


# -----------------------------------
# NODE 2
# STRENGTH SPECIALIST
# -----------------------------------

def suggest_strength_plan(state: FitnessState) -> dict:

    response = llm.invoke(
        f"You are a strength training coach. "
        f"The user's goal is: '{state.user_goal}'. "
        f"The fitness level is: '{state.user_level}'. "
        f"Suggest ONE strength workout plan. "
        f"Include sets, reps, and exercises. "
        f"Keep it under 5 sentences."
    )

    return {
        "strength_plan": response.content,
        "messages": ["[suggest_strength_plan] Done"]
    }


# -----------------------------------
# NODE 3
# RECOVERY SPECIALIST
# -----------------------------------

def suggest_recovery_plan(state: FitnessState) -> dict:

    response = llm.invoke(
        f"You are a recovery and wellness coach. "
        f"The user's goal is: '{state.user_goal}'. "
        f"Suggest recovery tips, stretching, hydration, and rest advice. "
        f"Keep it under 5 sentences."
    )

    return {
        "recovery_plan": response.content,
        "messages": ["[suggest_recovery_plan] Done"]
    }


# -----------------------------------
# DECISION NODE
# -----------------------------------

def decide_fitness_level(state: FitnessState) -> dict:

    if state.user_level.lower() == "beginner":

        return {
            "beginner_plan": True,
            "messages": ["[decide_fitness_level] Beginner selected"]
        }

    return {
        "beginner_plan": False,
        "messages": ["[decide_fitness_level] Intermediate selected"]
    }


# -----------------------------------
# BEGINNER PLAN
# -----------------------------------

def beginner_fitness_plan(state: FitnessState) -> dict:

    response = llm.invoke(
        f"""
        Create a BEGINNER FITNESS PLAN.

        USER GOAL:
        {state.user_goal}

        CARDIO PLAN:
        {state.cardio_plan}

        STRENGTH PLAN:
        {state.strength_plan}

        RECOVERY PLAN:
        {state.recovery_plan}

        Create a simple beginner-friendly weekly plan.
        Keep it motivating and easy to follow.
        """
    )

    return {
        "final_plan": f"BEGINNER FITNESS PLAN\n\n{'='*50}\n{response.content}",
        "messages": ["[beginner_fitness_plan] Generated"]
    }


# -----------------------------------
# INTERMEDIATE PLAN
# -----------------------------------

def intermediate_fitness_plan(state: FitnessState) -> dict:

    response = llm.invoke(
        f"""
        Create an INTERMEDIATE FITNESS PLAN.

        USER GOAL:
        {state.user_goal}

        CARDIO PLAN:
        {state.cardio_plan}

        STRENGTH PLAN:
        {state.strength_plan}

        RECOVERY PLAN:
        {state.recovery_plan}

        Create a challenging but balanced weekly plan.
        Keep it motivating and structured.
        """
    )

    return {
        "final_plan": f"INTERMEDIATE FITNESS PLAN\n\n{'='*50}\n{response.content}",
        "messages": ["[intermediate_fitness_plan] Generated"]
    }


# -----------------------------------
# ROUTER
# -----------------------------------

def route_after_decision(state: FitnessState) -> str:

    if state.beginner_plan:
        return "beginner"

    return "intermediate"


# -----------------------------------
# BUILD GRAPH
# -----------------------------------

graph = StateGraph(FitnessState)

graph.add_node("suggest_cardio_plan", suggest_cardio_plan)

graph.add_node("suggest_strength_plan", suggest_strength_plan)

graph.add_node("suggest_recovery_plan", suggest_recovery_plan)

graph.add_node("decide_fitness_level", decide_fitness_level)

graph.add_node("beginner_fitness_plan", beginner_fitness_plan)

graph.add_node("intermediate_fitness_plan", intermediate_fitness_plan)


# -----------------------------------
# FLOW
# -----------------------------------

graph.add_edge(START, "suggest_cardio_plan")

graph.add_edge(
    "suggest_cardio_plan",
    "suggest_strength_plan"
)

graph.add_edge(
    "suggest_strength_plan",
    "suggest_recovery_plan"
)

graph.add_edge(
    "suggest_recovery_plan",
    "decide_fitness_level"
)


# -----------------------------------
# CONDITIONAL ROUTING
# -----------------------------------

graph.add_conditional_edges(
    "decide_fitness_level",
    route_after_decision,
    {
        "beginner": "beginner_fitness_plan",
        "intermediate": "intermediate_fitness_plan"
    }
)


graph.add_edge("beginner_fitness_plan", END)

graph.add_edge("intermediate_fitness_plan", END)


# -----------------------------------
# COMPILE
# -----------------------------------

app = graph.compile()


# -----------------------------------
# RUN FUNCTION
# -----------------------------------

def run_fitness_graph(goal: str, level: str):

    print("=" * 55)
    print("        FITNESS PLAN GENERATOR")
    print("=" * 55)

    print(f"\nGoal  : {goal}")
    print(f"Level : {level}")

    result = app.invoke({

        "user_goal": goal,
        "user_level": level,
        "messages": []

    })

    print("\n" + "=" * 55)
    print("         YOUR FITNESS PLAN")
    print("=" * 55)

    print(f"\n{result['final_plan']}")

    print("\n" + "=" * 55)
    print("            MESSAGE LOG")
    print("=" * 55)

    for msg in result["messages"]:
        print(f" {msg}")

    return result


# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("        AI FITNESS PLAN GRAPH")
    print("=" * 55)

    goal = input("\nEnter your fitness goal: ").strip()

    level = input(
        "Enter your level (Beginner/Intermediate): "
    ).strip()

    run_fitness_graph(goal, level)

    print("hello")