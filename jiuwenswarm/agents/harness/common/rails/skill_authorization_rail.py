# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Expose the skill gate's scope resolution to the permission rail."""

from openjiuwen.core.single_agent.rail.base import AgentCallbackContext
from openjiuwen.harness.rails.skills.skill_authorization_rail import (
    SkillAuthorizationRail as CoreSkillAuthorizationRail,
)


class SkillAuthorizationRail(CoreSkillAuthorizationRail):
    """Skill gate with a public scope resolver for host integration."""

    def resolve_scope(self, ctx: AgentCallbackContext) -> tuple[str, str] | None:
        """Reuse the gate's scope resolution without changing authorization rules."""
        return self._resolve_scope(ctx)
