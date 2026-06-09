import os

def load_prompt(filename: str) -> str:
    """
    Utility to load prompt templates from the prompts directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Dynamically load the prompt contents
GOAL_SELECTOR_SYSTEM_PROMPT = load_prompt("goal_system_prompt.md")
GOAL_SELECTOR_USER_PROMPT = load_prompt("goal_user_prompt.md")
PLANNER_SYSTEM_PROMPT = load_prompt("planner_system_prompt.md")
PLANNER_USER_PROMPT = load_prompt("planner_user_prompt.md")
REFLECTION_SYSTEM_PROMPT = load_prompt("reflection_system_prompt.md")
REFLECTION_USER_PROMPT = load_prompt("reflection_user_prompt.md")

__all__ = [
    "GOAL_SELECTOR_SYSTEM_PROMPT",
    "GOAL_SELECTOR_USER_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "PLANNER_USER_PROMPT",
    "REFLECTION_SYSTEM_PROMPT",
    "REFLECTION_USER_PROMPT"
]
