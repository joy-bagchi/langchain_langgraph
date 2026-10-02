"""Harness notification port for operator-facing workflow events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True, slots=True)
class WorkflowNotification:
    notification_type: str
    message: str
    run_id: str | None = None
    workflow_id: str | None = None
    step_id: str | None = None
    metadata: dict[str, Any] | None = None


class NotificationService(Protocol):
    def notify(self, notification: WorkflowNotification) -> None: ...


class HarnessNotificationService:
    """Optional delivery adapter; workflow state remains the durable inbox."""

    def __init__(self, handler: Callable[[WorkflowNotification], None] | None = None) -> None:
        self._handler = handler
        self.notifications: list[WorkflowNotification] = []

    def notify(self, notification: WorkflowNotification) -> None:
        self.notifications.append(notification)
        if self._handler is not None:
            try:
                self._handler(notification)
            except Exception:
                # Delivery failures must not replace the durable workflow pause.
                return
