from typing import List


class ToDo:
    """A todo list manager with methods that can be registered as LLM tools.

    This class maintains a list of todos with completion tracking. All public methods
    return formatted strings suitable for LLM tool responses and display output using
    Rich Console formatting when available.

    Attributes:
        _todos: List of todo description strings
        _completed: List of boolean completion status for each todo
    """

    def __init__(self) -> None:
        """Initialize an empty todo list with no todos."""
        self._todos: List[str] = []
        self._completed: List[bool] = []

    @staticmethod
    def _show(text: str) -> None:
        try:
            from rich.console import Console

            Console().print(text)
        except Exception:
            print(text)

    def get_todo_report(self) -> str:
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
        for index, todo in enumerate(self._todos):
            if self._completed[index]:
                result += f"Todo #{index + 1}: [green][strike]{todo}[/strike][/green]\n"
            else:
                result += f"Todo #{index + 1}: {todo}\n"
        self._show(result)
        return result

    def create_todos(self, descriptions: List[str]) -> str:
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
        self._todos.extend(descriptions)
        self._completed.extend([False] * len(descriptions))
        return self.get_todo_report()

    def mark_complete(self, index: int, completion_notes: str) -> str:
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
        if 1 <= index <= len(self._todos):
            self._completed[index - 1] = True
        else:
            return "No todo at this index."
        self._show(completion_notes)
        return self.get_todo_report()

    def clear_todos(self) -> str:
        """Clear all todos and reset the list to empty.

        Removes all todos and their completion status, resetting the todo list
        to an empty state. Displays a confirmation message via the console.

        Returns:
            str: Confirmation message "All todos cleared."
        """
        self._todos = []
        self._completed = []
        self._show("All todos cleared.")
        return "All todos cleared."
