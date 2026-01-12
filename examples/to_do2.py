"""TodoManager class (in-memory, optional JSON persistence)

Uses a TodoItem dataclass (id, text, done, timestamps, due date, tags)
Supports: add, get, update, delete, complete/reopen/toggle, clear, list (filter/sort)
Optional: save/load to JSON file

Minimal usage example:
todos = TodoManager()
a = todos.add("Buy milk", tags=["errands"])
b = todos.add("Finish report", tags=["work"])

todos.complete(a.id)

open_items = todos.list(done=False)          # only not done
work_items = todos.list(tag="work")          # filter by tag

todos.save("todos.json")
todos2 = TodoManager.load("todos.json")
print([t.text for t in todos2.list()])
"""

from __future__ import annotations

import json
from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Iterable,
    Optional,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TodoItem:
    id: int
    text: str
    done: bool = False
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)
    due_at: Optional[datetime] = None
    tags: set[str] = field(default_factory=set)

    def touch(self) -> None:
        self.updated_at = _now_utc()


class TodoManager:
    def __init__(self) -> None:
        self._items: dict[int, TodoItem] = {}
        self._next_id: int = 1

    # ---------- CRUD ----------
    def add(
        self,
        text: str,
        *,
        due_at: Optional[datetime] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> TodoItem:
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")

        item = TodoItem(
            id=self._next_id,
            text=text.strip(),
            due_at=due_at,
            tags=set(tags) if tags else set(),
        )
        self._items[item.id] = item
        self._next_id += 1
        return item

    def get(self, todo_id: int) -> TodoItem:
        try:
            return self._items[todo_id]
        except KeyError:
            raise KeyError(f"Todo id {todo_id} not found") from None

    def update(
        self,
        todo_id: int,
        *,
        text: Optional[str] = None,
        due_at: Optional[datetime] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> TodoItem:
        item = self.get(todo_id)

        if text is not None:
            if not text.strip():
                raise ValueError("text must be non-empty when provided")
            item.text = text.strip()

        if due_at is not None:
            item.due_at = due_at

        if tags is not None:
            item.tags = set(tags)

        item.touch()
        return item

    def delete(self, todo_id: int) -> None:
        if todo_id not in self._items:
            raise KeyError(f"Todo id {todo_id} not found")
        del self._items[todo_id]

    def clear(self) -> None:
        self._items.clear()
        self._next_id = 1

    # ---------- status ----------
    def complete(self, todo_id: int) -> TodoItem:
        item = self.get(todo_id)
        item.done = True
        item.touch()
        return item

    def reopen(self, todo_id: int) -> TodoItem:
        item = self.get(todo_id)
        item.done = False
        item.touch()
        return item

    def toggle(self, todo_id: int) -> TodoItem:
        item = self.get(todo_id)
        item.done = not item.done
        item.touch()
        return item

    # ---------- listing ----------
    def list(
        self,
        *,
        done: Optional[bool] = None,  # None = all, True = only done, False = only open
        tag: Optional[str] = None,  # filter items containing this tag
        search: Optional[str] = None,  # substring match in text (case-insensitive)
        sort_by: str = "id",  # "id", "created_at", "updated_at", "due_at"
        reverse: bool = False,
    ) -> list[TodoItem]:
        items = list(self._items.values())

        if done is not None:
            items = [i for i in items if i.done == done]

        if tag is not None:
            items = [i for i in items if tag in i.tags]

        if search is not None:
            q = search.casefold()
            items = [i for i in items if q in i.text.casefold()]

        key_funcs = {
            "id": lambda i: i.id,
            "created_at": lambda i: i.created_at,
            "updated_at": lambda i: i.updated_at,
            "due_at": lambda i: (i.due_at is None, i.due_at),  # due dates first, None last
        }
        if sort_by not in key_funcs:
            raise ValueError(f"sort_by must be one of {sorted(key_funcs)}")

        items.sort(key=key_funcs[sort_by], reverse=reverse)
        return items

    # ---------- persistence (optional) ----------
    def to_json_str(self) -> str:
        def dt_to_str(dt: Optional[datetime]) -> Optional[str]:
            return None if dt is None else dt.astimezone(timezone.utc).isoformat()

        payload = {
            "next_id": self._next_id,
            "items": [
                {
                    "id": i.id,
                    "text": i.text,
                    "done": i.done,
                    "created_at": dt_to_str(i.created_at),
                    "updated_at": dt_to_str(i.updated_at),
                    "due_at": dt_to_str(i.due_at),
                    "tags": sorted(i.tags),
                }
                for i in self.list(sort_by="id")
            ],
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json_str(cls, s: str) -> "TodoManager":
        def str_to_dt(v: Optional[str]) -> Optional[datetime]:
            return None if v is None else datetime.fromisoformat(v)

        data = json.loads(s)
        mgr = cls()
        mgr._next_id = int(data.get("next_id", 1))

        for raw in data.get("items", []):
            item = TodoItem(
                id=int(raw["id"]),
                text=str(raw["text"]),
                done=bool(raw.get("done", False)),
                created_at=str_to_dt(raw.get("created_at")) or _now_utc(),
                updated_at=str_to_dt(raw.get("updated_at")) or _now_utc(),
                due_at=str_to_dt(raw.get("due_at")),
                tags=set(raw.get("tags", [])),
            )
            mgr._items[item.id] = item

        # Ensure next id is safe even if the file omitted/incorrectly set it
        if mgr._items:
            mgr._next_id = max(mgr._next_id, max(mgr._items) + 1)
        return mgr

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json_str())

    @classmethod
    def load(cls, path: str) -> "TodoManager":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json_str(f.read())
