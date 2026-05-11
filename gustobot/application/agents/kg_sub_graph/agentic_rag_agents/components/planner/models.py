from typing import Any

from pydantic import BaseModel, Field, field_validator

from ...components.models import Task


class PlannerOutput(BaseModel):
    tasks: list[Task] = Field(
        default_factory=list,
        description="A list of tasks that must be complete to satisfy the input question.",
    )

    @field_validator("tasks", mode="before")
    @classmethod
    def coerce_string_tasks(cls, value: Any) -> Any:
        """Some LLMs return tasks as strings; convert them to Task payloads."""
        if value is None:
            return []
        if not isinstance(value, list):
            return value

        normalized: list[Any] = []
        for item in value:
            if isinstance(item, str):
                question = item.strip()
                if question:
                    normalized.append(
                        {
                            "question": question,
                            "parent_task": question,
                        }
                    )
                continue
            normalized.append(item)
        return normalized
