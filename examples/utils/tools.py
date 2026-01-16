import os
import requests  # type: ignore
from typing import Dict

from dotenv import (
    find_dotenv,
    load_dotenv,
)


load_dotenv(find_dotenv(), override=True)


def push(text: str) -> None:
    """Send a push notification via Pushover.

    Args:
        text: The message text to send

    Returns:
        None
    """
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> Dict[str, str]:
    """Record user details for follow-up contact.

    Args:
        email: The email address of this user
        name: The user's name, if they provided it
        notes: Any additional information about the conversation

    Returns:
        dict: Confirmation of recording
    """
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> Dict[str, str]:
    """Record questions that couldn't be answered.

    Args:
        question: The question that couldn't be answered

    Returns:
        dict: Confirmation of recording
    """
    push(f"Recording {question}")
    return {"recorded": "ok"}


# Tool definitions - schemas auto-generated from type hints and docstrings!
tools = [record_user_details, record_unknown_question]
