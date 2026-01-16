import logging
from typing import List

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("MCP Todo Server")


todos: List[str] = []
completed: List[bool] = []


def show(text: str) -> None:
    try:
        import sys
        from rich.console import Console

        Console(file=sys.stderr).print(text)
    except Exception:
        logging.info(text)


@mcp.tool(name="get_todo_report")
def get_todo_report() -> str:
    """Generate a formatted report of all todos with their completion status.

    Creates a numbered list of all todos, with completed items shown using
    green strikethrough formatting (Rich markup). The report is displayed
    via the console and returned as a string.

    Returns:
        str: Formatted todo list with each todo numbered starting from 1.
            Completed todos are marked with [green][strike] Rich markup.
            Returns empty string if there are no todos.
    """
    result = ""
    for index, todo in enumerate(todos):
        if completed[index]:
            result += f"Todo #{index + 1}: [green][strike]{todo}[/strike][/green]\n"
        else:
            result += f"Todo #{index + 1}: {todo}\n"
    show(result)
    return result


@mcp.tool(name="create_todos")
def create_todos(descriptions: List[str]) -> str:
    """Add multiple new todos to the list.

    Appends the provided todo descriptions to the existing todo list and
    initializes them all as incomplete. After adding the todos, displays
    and returns the complete updated todo report.

    Args:
        descriptions: List of todo description strings to add to the list.
            Each string becomes a separate todo item.

    Returns:
        str: Complete formatted todo report including all existing and newly
            added todos. Same format as get_todo_report().
    """
    todos.extend(descriptions)
    completed.extend([False] * len(descriptions))
    return get_todo_report()


@mcp.tool(name="mark_complete")
def mark_complete(index: int, completion_notes: str) -> str:
    """Mark a specific todo as complete with completion notes.

    Marks the todo at the given 1-based index as complete and displays the
    provided completion notes. After marking complete, displays and returns
    the updated todo report.

    Args:
        index: 1-based index of the todo to mark complete. Use 1 for the first
            todo, 2 for the second todo, etc. (NOT 0-based indexing).
        completion_notes: Notes about the completion to display to the user.
            This message is shown via the console output.

    Returns:
        str: Complete formatted todo report if the index is valid, or an error
            message "No todo at this index." if the index is out of range.
    """
    if 1 <= index <= len(todos):
        completed[index - 1] = True
    else:
        return "No todo at this index."
    show(completion_notes)
    return get_todo_report()


@mcp.tool(name="clear_todos")
def clear_todos() -> str:
    """Clear all todos and reset the list to empty.

    Removes all todos and their completion status, resetting the todo list
    to an empty state. Displays a confirmation message via the console.

    Returns:
        str: Confirmation message "All todos cleared."
    """
    global todos, completed

    todos = []
    completed = []
    show("All todos cleared.")
    return "All todos cleared."


if __name__ == "__main__":
    logging.info("Starting MCP Todo Server...")
    mcp.run()
