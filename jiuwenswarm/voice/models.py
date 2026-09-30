"""Data models shared by the realtime voice bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast


VoiceCommandName = Literal[
    "submit_task",
    "supplement_task",
    "pause_task",
    "resume_task",
    "cancel_task",
    "get_task_status",
    "ask_leader",
]

MAX_TARGET_TASKS = 16

SUBMIT_TASK_TOOL = "submit_task"
SUPPLEMENT_TASK_TOOL = "supplement_task"
PAUSE_TASK_TOOL = "pause_task"
RESUME_TASK_TOOL = "resume_task"
CANCEL_TASK_TOOL = "cancel_task"
GET_TASK_STATUS_TOOL = "get_task_status"
ASK_LEADER_TOOL = "ask_leader"
LEGACY_SUBMIT_VOICE_REQUEST_TOOL = "submit_voice_request"
LEGACY_SUBMIT_TO_LEADER_TOOL = "submit_to_leader"

SUPPORTED_VOICE_TOOLS = frozenset(
    {
        SUBMIT_TASK_TOOL,
        SUPPLEMENT_TASK_TOOL,
        PAUSE_TASK_TOOL,
        RESUME_TASK_TOOL,
        CANCEL_TASK_TOOL,
        GET_TASK_STATUS_TOOL,
        ASK_LEADER_TOOL,
        LEGACY_SUBMIT_VOICE_REQUEST_TOOL,
        LEGACY_SUBMIT_TO_LEADER_TOOL,
    }
)

# Short Chinese verb for each action. The Leader identifies an intent block by
# its ``动作：`` line, so these must be mutually unambiguous.
_ACTION_LABELS: dict[str, str] = {
    SUBMIT_TASK_TOOL: "新建任务",
    SUPPLEMENT_TASK_TOOL: "修改任务",
    PAUSE_TASK_TOOL: "暂停任务",
    RESUME_TASK_TOOL: "恢复任务",
    CANCEL_TASK_TOOL: "取消任务",
    GET_TASK_STATUS_TOOL: "查询状态",
    ASK_LEADER_TOOL: "询问Leader",
}


def _normalize_target_task_ids(raw_task_ids: Any) -> tuple[str, ...]:
    if raw_task_ids in (None, "", []):
        return ()
    if not isinstance(raw_task_ids, list):
        raise ValueError("target_task_ids must be an array")
    if len(raw_task_ids) > MAX_TARGET_TASKS:
        raise ValueError(
            f"target_task_ids cannot contain more than {MAX_TARGET_TASKS} items"
        )
    task_ids = tuple(
        dict.fromkeys(
            task_id
            for item in raw_task_ids
            if (task_id := str(item or "").strip())
        )
    )
    if any(len(task_id) > 128 for task_id in task_ids):
        raise ValueError("target task id is too long")
    return task_ids


@dataclass(frozen=True, slots=True)
class LeaderInstruction:
    """A voice request normalized before it enters the team leader runtime."""

    summary: str
    instruction: str
    constraints: tuple[str, ...] = ()
    expected_output: str = ""

    @classmethod
    def from_arguments(cls, arguments: dict[str, Any]) -> "LeaderInstruction":
        summary = str(arguments.get("summary") or "").strip()
        instruction = str(arguments.get("instruction") or "").strip()
        if not summary:
            raise ValueError("summary is required")
        if not instruction:
            raise ValueError("instruction is required")

        raw_constraints = arguments.get("constraints")
        constraints: tuple[str, ...] = ()
        if isinstance(raw_constraints, list):
            constraints = tuple(
                value for item in raw_constraints if (value := str(item or "").strip())
            )
        elif raw_constraints:
            value = str(raw_constraints).strip()
            constraints = (value,) if value else ()

        return cls(
            summary=summary,
            instruction=instruction,
            constraints=constraints,
            expected_output=str(arguments.get("expected_output") or "").strip(),
        )

    def render_dispatch_text(self) -> str:
        """Render the full instruction body (requirement + constraints + delivery)."""
        lines = [self.instruction]
        if self.constraints:
            lines.extend(("", "约束条件：", *(f"- {item}" for item in self.constraints)))
        if self.expected_output:
            lines.extend(("", f"期望交付：{self.expected_output}"))
        return "\n".join(lines)

    def render_for_leader(self, voice_turn_id: str) -> str:
        lines = [
            f"[语音请求 {voice_turn_id}]",
            "",
            f"任务摘要：{self.summary}",
            f"完整要求：{self.render_dispatch_text()}",
        ]
        lines.extend(
            (
                "",
                "你已经通过语音向用户完成接收确认。请不要重复寒暄，直接按照团队规则拆解任务并调度成员；"
                "只有存在会显著改变执行结果的关键歧义时才向用户追问。",
            )
        )
        return "\n".join(lines)

    def render_intent_block(self, idx: int) -> str:
        """Render this instruction as one unified intent block.

        A multi-intent voice turn is a single chat.send; each instruction
        becomes one ``意图N`` block that the Leader can identify by the
        ``动作`` field instead of by a per-action header.
        """
        lines = [f"意图{idx} 动作：{self.summary or '新建任务'}"]
        lines.append(f"完整要求：{self.render_dispatch_text()}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class VoiceCommand:
    """One validated control decision produced by Qwen Realtime."""

    name: VoiceCommandName
    instruction: LeaderInstruction | None = None
    reason: str = ""
    query: str = ""
    target_task_ids: tuple[str, ...] = ()

    @classmethod
    def from_tool_call(
        cls,
        name: str,
        arguments: dict[str, Any],
    ) -> "VoiceCommand":
        normalized_name = {
            LEGACY_SUBMIT_VOICE_REQUEST_TOOL: SUBMIT_TASK_TOOL,
            LEGACY_SUBMIT_TO_LEADER_TOOL: SUBMIT_TASK_TOOL,
        }.get(name, name)
        if normalized_name not in SUPPORTED_VOICE_TOOLS:
            raise ValueError(f"unsupported tool: {name or '<empty>'}")

        if normalized_name == SUBMIT_TASK_TOOL:
            return cls(
                name=SUBMIT_TASK_TOOL,
                instruction=LeaderInstruction.from_arguments(arguments),
            )

        if normalized_name == SUPPLEMENT_TASK_TOOL:
            instruction_text = str(arguments.get("instruction") or "").strip()
            if not instruction_text:
                raise ValueError("instruction is required")
            adjustment_arguments = dict(arguments)
            adjustment_arguments["summary"] = str(
                arguments.get("summary") or "修改当前任务"
            ).strip()
            target_task_ids = _normalize_target_task_ids(
                arguments.get("target_task_ids")
            )
            if not target_task_ids:
                raise ValueError("supplement_task requires target_task_ids")
            return cls(
                name=SUPPLEMENT_TASK_TOOL,
                instruction=LeaderInstruction.from_arguments(adjustment_arguments),
                reason=str(arguments.get("reason") or "").strip(),
                target_task_ids=target_task_ids,
            )

        if normalized_name == GET_TASK_STATUS_TOOL:
            return cls(
                name=GET_TASK_STATUS_TOOL,
                query=str(
                    arguments.get("query") or "请汇报当前任务的真实状态和进度"
                ).strip(),
                target_task_ids=_normalize_target_task_ids(
                    arguments.get("target_task_ids")
                ),
            )

        if normalized_name == ASK_LEADER_TOOL:
            question = str(
                arguments.get("question") or arguments.get("query") or ""
            ).strip()
            if not question:
                raise ValueError("question is required")
            return cls(name=ASK_LEADER_TOOL, query=question)

        target_task_ids = _normalize_target_task_ids(
            arguments.get("target_task_ids")
        )
        if normalized_name == CANCEL_TASK_TOOL and not target_task_ids:
            raise ValueError("cancel_task requires target_task_ids")
        return cls(
            name=cast(VoiceCommandName, normalized_name),
            reason=str(arguments.get("reason") or "").strip(),
            target_task_ids=target_task_ids,
        )

    def _action_label(self) -> str:
        return _ACTION_LABELS.get(self.name, self.name)

    def _constraints_for_action(self) -> tuple[str, ...]:
        """Hard constraints that pin this action's blast radius.

        These were previously free-floating tail prompts (``只取消不得停止
        Team`` etc.). In a unified multi-intent request they must ride inside
        the intent block so they only apply to their own action and do not
        leak into sibling intents.
        """
        if self.name == SUPPLEMENT_TASK_TOOL and self.target_task_ids:
            return (
                "只修改这些任务，不要暂停、取消或重建其他并行任务。",
            )
        if self.name == CANCEL_TASK_TOOL:
            return (
                "只调用单任务取消能力取消上述任务，不得停止整个 Team，也不得影响其他并行任务。",
            )
        if self.name == GET_TASK_STATUS_TOOL:
            return (
                "请根据真实的团队成员、子任务和运行状态直接回答，不要根据对话记忆猜测。",
            )
        if self.name == RESUME_TASK_TOOL:
            return ("保持原有目标、已完成工作和用户约束。",)
        return ()

    def render_intent_block(self, idx: int) -> str:
        """Render this command as one unified intent block.

        A multi-intent voice turn is a single chat.send whose body lists
        ``意图N 动作：xxx`` blocks. Every action renders with the same
        field names so the Leader identifies each block by its ``动作`` line
        instead of by a per-action header (``[语音任务取消]`` etc.) — that
        per-action header is what welded heterogeneous blocks together and
        made the Leader drop the second intent.
        """
        lines = [f"意图{idx} 动作：{self._action_label()}"]
        if self.instruction is not None:
            lines.append(f"任务摘要：{self.instruction.summary}")
            lines.append(f"完整要求：{self.instruction.render_dispatch_text()}")
        elif self.query:
            lines.append(f"问题：{self.query}")
        if self.target_task_ids:
            lines.append(f"目标任务：{'、'.join(self.target_task_ids)}")
        if self.reason:
            lines.append(f"原因：{self.reason}")
        constraints = self._constraints_for_action()
        if self.instruction is not None and self.instruction.constraints:
            constraints = (*self.instruction.constraints, *constraints)
        if constraints:
            lines.append("约束：")
            for item in constraints:
                lines.append(f"- {item}")
        return "\n".join(lines)

    @staticmethod
    def render_voice_batch(
        commands: "tuple[VoiceCommand, ...]",
        voice_turn_id: str,
    ) -> str:
        """Render one or more commands as a single unified chat.send body.

        One request header ``[语音请求 turn_id]`` covers the whole turn;
        the dispatch posture (``不要寒暄、直接调度``) is stated once at the
        end for every intent — never repeated per block. Single-intent turns
        use the exact same shape so the Leader sees one consistent format.
        """
        blocks = [
            cmd.render_intent_block(idx)
            for idx, cmd in enumerate(commands, start=1)
        ]
        intent_word = "意图" if len(blocks) == 1 else "多个意图"
        footer = (
            "你已经通过语音向用户完成接收确认。请不要重复寒暄，直接按照团队规则"
            "拆解任务并调度成员；以上为同一轮语音的"
            f"{intent_word}，请分别独立处理、互不影响，每个意图只作用于各自的"
            "目标任务，不要相互覆盖；只有存在会显著改变执行结果的关键歧义时才向用户追问。"
        )
        # Mark the payload as machine-generated so the downstream Team Leader
        # never mistakes the dispatch template for a second user utterance.
        blocks.insert(0, "[VOICE_DISPATCH_MACHINE_ONLY]")
        return "\n\n".join(
            [f"[语音请求 {voice_turn_id}]", *blocks, footer]
        )

    def dispatch_text(self, voice_turn_id: str) -> str:
        return self.render_voice_batch((self,), voice_turn_id)


_POSITIVE_CONTINUE_PHRASES = (
    "继续执行",
    "继续做",
    "继续处理",
    "接着执行",
    "接着做",
    "接着处理",
    "照常执行",
    "恢复执行",
    "不要停",
    "别停",
    "不用停",
    "无需暂停",
)
_EXPLICIT_CANCEL_OR_NEGATED_CONTINUE_PHRASES = (
    "不要继续",
    "别继续",
    "不用继续",
    "无需继续",
    "不再继续",
    "停止执行",
    "终止",
    "取消",
    "放弃",
    "任务不要了",
)

def reconcile_voice_command_with_transcript(
    command: VoiceCommand,
    transcript: str,
) -> VoiceCommand:
    """Prevent an affirmative continuation from being executed as cancellation.

    Qwen may anchor on a leading conversational negation such as ``不用`` and
    ignore the correction that follows (``继续执行``).  Cancellation is
    destructive, so a clear positive continuation wins unless the same utterance
    also contains an explicit cancel or a negated continuation.
    """
    if command.name != CANCEL_TASK_TOOL:
        return command
    normalized = "".join(str(transcript or "").lower().split())
    if not normalized:
        return command
    wants_to_continue = any(
        phrase in normalized for phrase in _POSITIVE_CONTINUE_PHRASES
    )
    explicitly_cancels = any(
        phrase in normalized
        for phrase in _EXPLICIT_CANCEL_OR_NEGATED_CONTINUE_PHRASES
    )
    if not wants_to_continue or explicitly_cancels:
        return command
    return VoiceCommand(
        name=RESUME_TASK_TOOL,
        reason="用户明确要求继续执行",
    )


def reconcile_supplement_with_team_info(
    command: VoiceCommand,
    team_info: dict[str, Any] | None,
) -> VoiceCommand:
    """Bind a supplement to authoritative task metadata when possible.

    Never relabel an explicit supplement as ``submit_task``: doing so silently
    creates a second task instead of modifying the existing one. Known task
    IDs remain targeted supplements; an ambiguous target is preserved for the
    backend/user clarification path.
    """
    if command.name != SUPPLEMENT_TASK_TOOL or command.instruction is None:
        return command
    if not isinstance(team_info, dict):
        return command

    raw_tasks = team_info.get("tasks")
    # An empty directory has no authoritative target. Preserve the model's
    # command and let the downstream clarification path handle it.
    if isinstance(raw_tasks, list) and not raw_tasks:
        return command
    known_task_ids = {
        str(item.get("task_id") or "").strip()
        for item in raw_tasks
        if isinstance(item, dict)
    } if isinstance(raw_tasks, list) else set()
    known_targets = tuple(
        task_id for task_id in command.target_task_ids if task_id in known_task_ids
    )
    if known_targets:
        # Keep the complete explicit list; do not silently drop IDs that are
        # not present in a possibly stale directory.
        return command

    # No authoritative target was found. Never infer one from a similar title
    # or from a one-task roster.
    return command
