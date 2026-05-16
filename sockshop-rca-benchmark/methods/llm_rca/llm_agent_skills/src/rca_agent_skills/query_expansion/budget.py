from __future__ import annotations

from dataclasses import dataclass

from rca_agent_skills.common.exceptions import BudgetExceededError


@dataclass
class QueryBudget:
    skill: str
    limit: int
    used: int = 0

    def consume(self, amount: int = 1) -> None:
        if self.used + amount > self.limit:
            raise BudgetExceededError(f"{self.skill} follow-up budget exceeded: {self.used + amount}>{self.limit}")
        self.used += amount

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

