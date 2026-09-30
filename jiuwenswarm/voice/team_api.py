"""Public integration API for connecting voice clients to Team mode.

The realtime (Qwen) and browser session implementations are intentionally
kept behind this small interface.  Team-mode callers only need a dispatcher
that accepts normalized :class:`VoiceCommand` objects; they do not need to
know about websocket envelopes or the voice provider in use.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from jiuwenswarm.voice.models import LeaderInstruction, VoiceCommand


@runtime_checkable
class TeamVoiceInterface(Protocol):
    """Minimal contract exposed to a Team-mode integration."""

    async def submit_instruction(
        self, voice_turn_id: str, instruction: LeaderInstruction
    ) -> dict[str, Any]:
        """Send a normalized instruction to the Team Leader."""

    async def dispatch_command(
        self, voice_turn_id: str, command: VoiceCommand
    ) -> dict[str, Any]:
        """Execute one normalized voice control in Team mode."""


class TeamVoiceAdapter:
    """Adapt an existing ``LeaderGatewayBridge`` to ``TeamVoiceInterface``.

    Keeping this adapter tiny makes it straightforward for a native Team
    runtime (or tests) to provide its own implementation without importing
    Qwen or Gateway client details.
    """

    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge

    async def submit_instruction(
        self, voice_turn_id: str, instruction: LeaderInstruction
    ) -> dict[str, Any]:
        return await self.bridge.submit(voice_turn_id, instruction)

    async def dispatch_command(
        self, voice_turn_id: str, command: VoiceCommand
    ) -> dict[str, Any]:
        return await self.bridge.execute(voice_turn_id, command)


def create_team_voice_interface(bridge: Any) -> TeamVoiceInterface:
    """Return the stable Team-facing interface for a voice bridge."""

    # LeaderGatewayBridge already implements the contract.  Return it
    # directly so exactly-once state and lifecycle remain in one object.
    if isinstance(bridge, TeamVoiceInterface):
        return bridge
    return TeamVoiceAdapter(bridge)
