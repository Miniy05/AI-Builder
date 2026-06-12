import os
import sys
import operator
from pathlib import Path
from typing import Annotated
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# -----------------------------------
# CONFIG
# -----------------------------------

sys.stdout.reconfigure(encoding="utf-8")

env_path = Path(__file__).parent / ".env.fitness"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to your environment or .env.fitness file. "
        "Example: OPENAI_API_KEY=your_key"
    )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
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
# INPUT GUARDRAIL
# -----------------------------------

VALID_LEVELS = ["beginner", "intermediate"]

def validate_user_input(goal: str, level: str):

    if not goal.strip():
        raise ValueError("Fitness goal cannot be empty.")

    if level.lower() not in VALID_LEVELS:
        raise ValueError(
            "Level must be Beginner or Intermediate."
        )

    banned_keywords = [
        "steroids",
        "starvation",
        "extreme weight loss"
    ]

    for keyword in banned_keywords:
        if keyword in goal.lower():
            raise ValueError(
                f"Unsafe goal detected: {keyword}"
            )

    return True


# -----------------------------------
# OUTPUT GUARDRAIL
# -----------------------------------

def output_guardrail(plan: str):

    banned_terms = [
        "steroid",
        "starvation",
        "skip meals",
        "dangerous workout"
    ]

    for term in banned_terms:
        if term.lower() in plan.lower():
            raise ValueError(
                f"Unsafe output detected: {term}"
            )

    return plan


# -----------------------------------
# CARDIO SPECIALIST
# -----------------------------------

def suggest_cardio_plan(state: FitnessState) -> dict:

    response = llm.invoke(
        f"""
        You are a certified fitness coach.

        Rules:
        - No medical advice
        - No steroids
        - No starvation diets
        - No dangerous workouts

        User Goal: {state.user_goal}
        User Level: {state.user_level}

        Suggest ONE cardio workout plan.
        Include exercise name, duration, and frequency.
        Keep it under 5 sentences.
        """
    )

    return {
        "cardio_plan": response.content,
        "messages": ["[suggest_cardio_plan] Done"]
    }


# -----------------------------------
# STRENGTH SPECIALIST
# -----------------------------------

def suggest_strength_plan(state: FitnessState) -> dict:

    response = llm.invoke(
        f"""
        You are a certified strength coach.

        Rules:
        - No medical advice
        - No steroids
        - No dangerous workouts

        User Goal: {state.user_goal}
        User Level: {state.user_level}

        Suggest ONE strength workout plan.
        Include sets, reps and exercises.
        Keep it under 5 sentences.
        """
    )

    return {
        "strength_plan": response.content,
        "messages": ["[suggest_strength_plan] Done"]
    }


# -----------------------------------
# RECOVERY SPECIALIST
# -----------------------------------

def suggest_recovery_plan(state: FitnessState) -> dict:

    response = llm.invoke(
        f"""
        You are a recovery and wellness coach.

        Rules:
        - No medical advice
        - No unsafe health claims

        User Goal: {state.user_goal}

        Suggest recovery tips, stretching,
        hydration and rest advice.
        Keep it under 5 sentences.
        """
    )

    return {
        "recovery_plan": response.content,
        "messages": ["[suggest_recovery_plan] Done"]
    }


# -----------------------------------
# HALLUCINATION / SAFETY GUARDRAIL
# -----------------------------------

def hallucination_guardrail(state: FitnessState):

    combined_plan = f"""
    CARDIO:
    {state.cardio_plan}

    STRENGTH:
    {state.strength_plan}

    RECOVERY:
    {state.recovery_plan}
    """

    review = llm.invoke(
        f"""
        You are a fitness safety auditor.

        Review the workout plan.

        Check:
        - Unsafe exercises
        - Harmful advice
        - Medical claims
        - Unrealistic schedules

        Return only:
        SAFE
        or
        UNSAFE

        PLAN:
        {combined_plan}
        """
    )

    verdict = review.content.strip().upper()

    if "SAFE" not in verdict:
        raise ValueError(
            "Guardrail blocked unsafe fitness plan."
        )

    return {
        "messages": ["[hallucination_guardrail] SAFE"]
    }


# -----------------------------------
# DECISION NODE
# -----------------------------------

def decide_fitness_level(state: FitnessState):

    if state.user_level.lower() == "beginner":

        return {
            "beginner_plan": True,
            "messages": [
                "[decide_fitness_level] Beginner selected"
            ]
        }

    return {
        "beginner_plan": False,
        "messages": [
            "[decide_fitness_level] Intermediate selected"
        ]
    }


# -----------------------------------
# BEGINNER PLAN
# -----------------------------------

def beginner_fitness_plan(state: FitnessState):

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

        Create a simple weekly plan.
        """
    )

    safe_plan = output_guardrail(response.content)

    return {
        "final_plan":
        f"BEGINNER FITNESS PLAN\n\n{'='*50}\n{safe_plan}",
        "messages": ["[beginner_fitness_plan] Generated"]
    }


# -----------------------------------
# INTERMEDIATE PLAN
# -----------------------------------

def intermediate_fitness_plan(state: FitnessState):

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

        Create a balanced weekly plan.
        """
    )

    safe_plan = output_guardrail(response.content)

    return {
        "final_plan":
        f"INTERMEDIATE FITNESS PLAN\n\n{'='*50}\n{safe_plan}",
        "messages": ["[intermediate_fitness_plan] Generated"]
    }


# -----------------------------------
# ROUTER
# -----------------------------------

def route_after_decision(state: FitnessState):

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
graph.add_node("hallucination_guardrail", hallucination_guardrail)
graph.add_node("decide_fitness_level", decide_fitness_level)
graph.add_node("beginner_fitness_plan", beginner_fitness_plan)
graph.add_node("intermediate_fitness_plan", intermediate_fitness_plan)

graph.add_edge(START, "suggest_cardio_plan")
graph.add_edge("suggest_cardio_plan", "suggest_strength_plan")
graph.add_edge("suggest_strength_plan", "suggest_recovery_plan")
graph.add_edge("suggest_recovery_plan", "hallucination_guardrail")
graph.add_edge("hallucination_guardrail", "decide_fitness_level")

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

app = graph.compile()


# -----------------------------------
# RUN FUNCTION
# -----------------------------------

def run_fitness_graph(goal: str, level: str):

    validate_user_input(goal, level)

    result = app.invoke({
        "user_goal": goal,
        "user_level": level,
        "messages": []
    })

    return result


# -----------------------------------
# MAIN
# -----------------------------------

if __name__ == "__main__":

    goal = input("Enter your fitness goal: ").strip()

    level = input(
        "Enter your level (Beginner/Intermediate): "
    ).strip()

    result = run_fitness_graph(goal, level)

    print("\n")
    print(result["final_plan"])

    print("\nMESSAGE LOG:")

    for msg in result["messages"]:
        print(msg)
