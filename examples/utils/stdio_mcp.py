from typing import List

from mcp.types import PromptArgument


def get_prompt_arguments(prompt_arguments: List[PromptArgument]) -> dict[str, str]:
    """Ask user for prompt arguments interactively."""
    if not prompt_arguments:
        return {}
    print("Please provide values for the following arguments:")
    arguments: dict[str, str] = {}
    for arg in prompt_arguments:
        required_text = "(required)" if arg.required else "(optional)"
        user_input = input(f"Enter {arg.name} {required_text}: ").strip()
        if user_input or arg.required:
            arguments[arg.name] = user_input
    return arguments


def get_template_variables(variables: List[str]) -> dict[str, str]:
    """Ask user for values for variables extracted from URI template"""
    print("Please provide values for the following variables:")
    values = {}
    for var in variables:
        value = input(f"Enter value for {var}: ").strip()
        values[var] = value
    return values
