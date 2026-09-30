"""Qwen-Audio realtime microphone client used by the voice sidecar."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any

from jiuwenswarm.voice.coordinator import VoiceDispatchCoordinator
from jiuwenswarm.voice.models import (
    ASK_LEADER_TOOL,
    CANCEL_TASK_TOOL,
    GET_TASK_STATUS_TOOL,
    PAUSE_TASK_TOOL,
    RESUME_TASK_TOOL,
    SUBMIT_TASK_TOOL,
    SUPPLEMENT_TASK_TOOL,
)

logger = logging.getLogger(__name__)

INPUT_RATE = 16_000
OUTPUT_RATE = 24_000
CHUNK_FRAMES = 3_200

VOICE_INSTRUCTIONS = """你就是 JiuwenSwarm 团队的 Team Leader 本人，正在通过实时语音和用户直接对话。
语音模型与后端 Leader 是同一个角色的两种能力：你负责实时理解和口语回应，工具负责让你的后端执行体
继续拆解、调度和执行。这是你自己的内部控制过程，不是把请求转交给另一个 Leader。

始终以 Leader 第一人称和用户说话：
- 用“我”表示 Leader，用“我们”表示你管理的团队，用“你/您”称呼当前用户。
- 禁止说“我会转交给 Leader”“我会反馈给 Leader”“我会通知 Leader”“交给相关成员处理”。
- 如果需要成员执行，可以说“我来安排成员处理”，但不得虚构成员、进度或已完成结果。

对每个有意义的用户语音请求：
1. 先用一句简短、自然的中文，以 Leader 本人的口吻确认你理解的操作。
2. 确认说完后，把工具调用当作 Leader 自己的内部动作，从下列工具中选择：
   - 独立的新目标：submit_task。凡是需要 Team 实际执行、调用能力、访问外部来源或获取当前状态的工作，都属于新目标；不要因为用户使用“查询”“帮我看看”等措辞就选择 ask_leader。判断依据是是否要产生可交付的执行结果，而不是主题领域或关键词。
   对用户提出的每项要求分别选择最匹配的工具，并让每次调用的参数只描述该项要求；不要替用户安排执行顺序、并行关系或依赖。
   - 修改、补充或纠正已经存在的具体任务：supplement_task，并填写 target_task_ids。只有能够从任务目录确定被修改的任务时才使用它；独立的新要求仍使用 submit_task。
   - 暂时停止整个 Team 的全部任务但以后可继续：pause_task；这是全局控制，不填写任务 ID。
   - 恢复整个 Team 中被全局暂停的任务：resume_task；这是全局控制，不填写任务 ID。
   - 明确永久放弃一个或多个具体任务：cancel_task，并填写 target_task_ids。
   - 询问具体任务或全部任务的进度、完成情况或执行成员：get_task_status。
   - 仅当用户询问 Team 的状态、计划、解释或决策，且问题本身不要求 Team 再执行一项工作时，才使用 ask_leader。ask_leader 只获取后端已有上下文中的权威回答；一旦需要新的执行、访问外部来源或生成交付物，必须使用 submit_task。
3. 任务如何拆分、串行或并行执行，交给后端 Team Leader 根据团队能力和任务依赖决定。
4. “别说了”只是停止语音播报，不得调用 pause_task 或 cancel_task。
5. “算了”“不用了”“停一下”等语义不明确时，只追问是停止播报、暂停任务还是取消任务，此时不调用工具。
   如果一句话前面有“不了/不用”等口头否定，但后面明确说“继续执行/接着做/不要停”，以后半句的明确动作为准，必须选择 resume_task，绝对不得选择 cancel_task。取消属于破坏性操作，只有用户明确说取消、终止、放弃具体任务时才能调用。
6. 危险操作或会显著改变结果的关键约束不清楚时，只追问一个问题，不调用工具。
7. 回答普通问题时，以“是否需要后端权威数据”为边界，不要只根据主观置信度决定：
   - 通用且不随当前任务变化的知识，可以直接回答且不调用工具。
   - 当前语音会话中已经明确说过的内容，可以直接回答且不调用工具。
   - 系统注入的 Team Leader 汇报已经明确给出的事实，可以直接回答且不调用工具。
   - 系统注入的团队成员目录能够直接回答的问题，可以直接回答且不调用工具；但该目录不代表成员状态。
   - 不涉及真实执行状态的解释、建议和澄清，可以直接回答且不调用工具。
   - 依赖后端对话历史、项目文件、工具结果、实际执行情况或尚未同步信息的问题，调用 ask_leader。
   - 当前上下文没有明确答案、答案可能已经变化或无法确定上下文是否足够时，调用 ask_leader。
8. “你好”“谢谢”“嗯”等纯寒暄、感谢、填充词和无意义声音直接回答且不调用工具。
9. 任务进度、完成情况、执行成员、运行状态必须调用 get_task_status，即使上下文中存在旧状态也不得直接回答。
   调用时只能说“我查一下”之类的过程确认，真实进度必须等待后端汇报，不得根据语音上下文猜测任务已完成。
10. 系统注入的任务目录包含真实 task_id、创建顺序、标题和状态：
   - supplement_task 和 cancel_task 必须从目录中选择准确的 target_task_ids，不得编造 ID。
   - get_task_status 查询具体任务时填写 target_task_ids；用户明确查询全部任务时可留空。
   - “第一个/第二个任务”按目录中的创建顺序解析；按标题能唯一匹配时也可以解析。
   - 存在多个可能目标、目录没有对应任务或用户所说“删除”无法区分取消任务与删除交付物时，先追问，不调用工具。
   - submit_task 创建新任务，不填写 target_task_ids；任务创建后系统会把真实 ID 更新到目录。
11. 新的 submit_task 不携带语音侧并行模式或依赖图，具体调度统一交给后端 Team Leader。
12. pause_task 和 resume_task 始终作用于整个 Team，不得用于单个任务。单个任务永久停止使用 cancel_task。
13. 调用 ask_leader 时不要自行回答问题，只可简短说“我想一下”或“我结合当前情况看一下”，等待后端执行结果。
14. 回复适合直接朗读，不使用 Markdown，不声称任何后端操作已经成功。

系统有时会注入两种 Team Leader 汇报：
1. 以“JiuwenSwarm Team Leader 汇报：”开头时，把冒号后的汇报忠实、自然地直接说给用户。
2. 以“JiuwenSwarm Team Leader 简短汇报：”开头时，把冒号后的内容概括成1到2句、
最多80个中文字符的口语回复。优先保留任务状态、核心结果、失败原因和需要用户决定的事项；
不逐字朗读路径、代码、日志和过程细节，不添加原文中没有的结论。
处理这两种汇报时，它们是你自己的后端执行结果，而不是别人发给你的消息。
始终以 Team Leader 本人的身份和用户直接对话：用“我/我们”表达 Leader 或团队，用“你/您”称呼当前用户。
禁止把当前用户称为“用户”，禁止用“他/她/对方”指代当前用户，也不要说“Leader 表示”“他说”
“团队让我转告”“我收到 Leader 的消息”等旁观者式转述。
如果原文使用第三人称，在不改变事实、状态和结论的前提下改写成自然的第一、第二人称表达。
处理上述两种汇报时，都不要确认、不要转交，也绝对不要调用工具。
"""

TEAM_DIRECTORY_INSTRUCTIONS = """
以下是 JiuwenSwarm 系统同步的当前团队目录。成员目录不包含你自己（Team Leader）。
成员信息只表示身份，不表示忙闲、进度、在线状态或任务分配。
{directory}
回答成员和任务指代问题时只能使用该目录；目录中没有的信息不得编造。
"""

LEADER_REPORT_PREFIX = "JiuwenSwarm Team Leader 汇报："
LEADER_BRIEF_REPORT_PREFIX = "JiuwenSwarm Team Leader 简短汇报："


def _one_line(value: Any, max_chars: int = 100) -> str:
    return " ".join(str(value or "").split())[:max_chars]


def build_voice_instructions(team_info: dict[str, Any] | None = None) -> str:
    """Attach member identity and the addressable task directory."""
    if team_info is None:
        return VOICE_INSTRUCTIONS

    team_id = _one_line(team_info.get("team_id"), 128)
    members = team_info.get("members")
    lines: list[str] = []
    if isinstance(members, list):
        for item in members[:64]:
            if not isinstance(item, dict):
                continue
            member_id = _one_line(item.get("member_id"), 100)
            if not member_id:
                continue
            name = _one_line(item.get("name"), 100) or member_id
            role = _one_line(item.get("role"), 60) or "teammate"
            lines.append(f"- 成员ID：{member_id}；名称：{name}；角色：{role}")

    member_directory = "\n".join(lines) if lines else "当前暂无已知团队成员。"
    task_lines: list[str] = []
    tasks = team_info.get("tasks")
    if isinstance(tasks, list):
        for index, item in enumerate(tasks[:64], start=1):
            if not isinstance(item, dict):
                continue
            task_id = _one_line(item.get("task_id"), 128)
            if not task_id:
                continue
            try:
                order = max(1, int(item.get("created_order")))
            except (TypeError, ValueError):
                order = index
            title = _one_line(item.get("title"), 160) or "未命名任务"
            status = _one_line(item.get("status"), 40) or "unknown"
            task_lines.append(
                f"- 创建顺序：{order}；任务ID：{task_id}；标题：{title}；状态：{status}"
            )
    task_directory = "\n".join(task_lines) if task_lines else "当前暂无已知团队任务。"
    heading = f"团队：{team_id}\n" if team_id else ""
    directory = (
        f"{heading}成员目录：\n{member_directory}\n"
        f"任务目录：\n{task_directory}"
    )
    return (
        VOICE_INSTRUCTIONS
        + "\n"
        + TEAM_DIRECTORY_INSTRUCTIONS.format(directory=directory).strip()
    )


def _task_instruction_properties() -> dict[str, Any]:
    return {
        "summary": {"type": "string", "description": "简短任务摘要"},
        "instruction": {"type": "string", "description": "完整任务要求"},
        "constraints": {
            "type": "array",
            "items": {"type": "string"},
            "description": "用户明确提出的限制条件",
        },
        "expected_output": {
            "type": "string",
            "description": "用户期望得到的最终交付",
        },
    }


def submit_task_tool_schema() -> dict[str, Any]:
    schema = {
        "type": "function",
        "function": {
            "name": SUBMIT_TASK_TOOL,
            "description": "用户提出一个独立的新任务；“再做一个”属于新任务",
            "parameters": {
                "type": "object",
                "properties": _task_instruction_properties(),
                "required": ["summary", "instruction"],
            },
        },
    }
    schema["function"]["description"] = (
        "创建独立的新目标并让 Team 实际执行。凡是需要新的执行、外部访问、当前信息或可交付结果的请求，"
        "都属于 submit_task；不要把执行请求当作 ask_leader。"
    )
    return schema


def _target_task_ids_schema(*, required: bool) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1 if required else 0,
        "maxItems": 16,
        "description": (
            "从系统任务目录中选择的真实任务 ID；不得使用标题、序号或编造的 ID"
        ),
    }


def supplement_task_tool_schema() -> dict[str, Any]:
    properties = _task_instruction_properties()
    properties["reason"] = {"type": "string", "description": "修改原因，可留空"}
    properties["target_task_ids"] = _target_task_ids_schema(required=True)
    schema = {
        "type": "function",
        "function": {
            "name": SUPPLEMENT_TASK_TOOL,
            "description": "修改、补充或纠正当前任务，不创建独立新目标",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["instruction", "target_task_ids"],
            },
        },
    }
    schema["function"]["description"] = (
        "修改、补充或纠正已有任务，不创建独立新目标；例如修改已有任务的目标、参数、范围或交付要求。"
        "必须填写任务目录中的准确 target_task_ids。"
    )
    return schema


def _reason_tool_schema(name: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "用户给出的原因，可留空"}
                },
            },
        },
    }


def voice_tool_schemas() -> list[dict[str, Any]]:
    schemas = [
        submit_task_tool_schema(),
        supplement_task_tool_schema(),
        _reason_tool_schema(PAUSE_TASK_TOOL, "全局暂停整个 Team 的全部任务，以后可以恢复"),
        _reason_tool_schema(RESUME_TASK_TOOL, "全局恢复整个 Team 中被暂停的任务"),
        {
            "type": "function",
            "function": {
                "name": CANCEL_TASK_TOOL,
                "description": "永久取消任务目录中的一个或多个具体任务，不停止整个 Team",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_task_ids": _target_task_ids_schema(required=True),
                        "reason": {"type": "string", "description": "用户给出的取消原因"},
                    },
                    "required": ["target_task_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": GET_TASK_STATUS_TOOL,
                "description": "查询当前任务进度、完成情况、执行成员或运行状态",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "用户想了解的状态问题"},
                        "target_task_ids": _target_task_ids_schema(required=False),
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": ASK_LEADER_TOOL,
                "description": (
                    "当前语音上下文没有权威答案，且问题需要 Leader 结合后端对话历史、"
                    "项目文件、工具结果或实际执行情况回答；上下文足以回答时不要调用"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "用户的完整问题，不要改写为第三人称",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
    ]
    # Keep the model-facing boundary explicit even if the shared prompt is
    # changed later: ask_leader answers Team state/plan/explanation/decision
    # questions, while real external or up-to-date work is submit_task.
    for schema in schemas:
        function = schema.get("function", {})
        if function.get("name") == ASK_LEADER_TOOL:
            function["description"] = (
                "仅用于询问 Team 的状态、计划、解释或决策，且只读取后端已有上下文。"
                "如果问题要求新的执行、外部访问、当前信息或可交付结果，必须调用 submit_task。"
                "上下文足以回答时不要调用；调用 ask_leader 时不要自行回答问题。"
            )
    return schemas


@dataclass(frozen=True, slots=True)
class QwenRealtimeConfig:
    api_key: str
    workspace_id: str
    model: str = "qwen-audio-3.0-realtime-plus"
    voice: str = "longanqian"
    turn_detection: str = "server_vad"
    silence_duration_ms: int = 800

    @property
    def url(self) -> str:
        return (
            f"wss://{self.workspace_id}.cn-beijing.maas.aliyuncs.com/"
            f"api-ws/v1/realtime?model={self.model}"
        )


def build_session_update(
    config: QwenRealtimeConfig,
    *,
    team_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config.turn_detection == "smart_turn":
        turn_detection: dict[str, Any] = {"type": "smart_turn"}
    else:
        turn_detection = {
            "type": "server_vad",
            "threshold": 0.5,
            "silence_duration_ms": config.silence_duration_ms,
        }
    return {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "voice": config.voice,
            "instructions": build_voice_instructions(team_info),
            "turn_detection": turn_detection,
            "tools": voice_tool_schemas(),
        },
    }


class QwenRealtimeVoiceClient:
    """One full-duplex Qwen connection for input and Leader speech."""

    def __init__(
        self,
        config: QwenRealtimeConfig,
        coordinator: VoiceDispatchCoordinator,
    ) -> None:
        self._config = config
        self._coordinator = coordinator
        self._leader_reply_queue: asyncio.Queue[str] = asyncio.Queue()
        self._audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._realtime_idle = asyncio.Event()
        self._realtime_idle.set()
        self._leader_response_requested = False
        self._active_response_origin: str | None = None
        self._user_turn_active = False
        self._response_interrupted_by_new_speech = False
        self._drop_audio_until_done = False

    async def enqueue_leader_reply(self, text: str) -> None:
        """Queue one Leader report for playback on the existing Realtime socket."""
        normalized = text.strip()
        if normalized:
            await self._leader_reply_queue.put(normalized)

    async def run(self) -> None:
        try:
            import pyaudio
        except ImportError as exc:
            raise RuntimeError(
                "Voice mode requires PyAudio; install with: pip install -e '.[voice]'"
            ) from exc
        import websockets

        audio = pyaudio.PyAudio()
        microphone = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )
        speaker = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=OUTPUT_RATE,
            output=True,
        )

        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        header_key = (
            "additional_headers"
            if "additional_headers" in inspect.signature(websockets.connect).parameters
            else "extra_headers"
        )
        try:
            async with websockets.connect(
                self._config.url,
                **{header_key: headers},
            ) as websocket:
                await websocket.send(
                    json.dumps(build_session_update(self._config), ensure_ascii=False)
                )
                await asyncio.wait_for(
                    self._wait_for_session_updated(websocket),
                    timeout=15,
                )
                print(
                    "语音会话已就绪。所有请求只交给 Team Leader；"
                    "Leader 汇报时可直接开口打断。按 Ctrl+C 结束。",
                    flush=True,
                )
                await asyncio.gather(
                    self._send_audio(websocket, microphone),
                    self._receive_events(websocket),
                    self._play_audio(speaker),
                    self._speak_leader_replies(websocket),
                )
        finally:
            microphone.close()
            speaker.close()
            audio.terminate()

    @staticmethod
    async def _send_audio(websocket: Any, microphone: Any) -> None:
        while True:
            pcm = await asyncio.to_thread(microphone.read, CHUNK_FRAMES, False)
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode("ascii"),
                    }
                )
            )

    async def _receive_events(self, websocket: Any) -> None:
        async for message in websocket:
            event = json.loads(message)
            event_type = str(event.get("type") or "")
            if event_type == "response.created":
                self._realtime_idle.clear()
                if self._leader_response_requested:
                    self._active_response_origin = "leader-report"
                    self._leader_response_requested = False
                else:
                    self._active_response_origin = "user-turn"
                self._drop_audio_until_done = False
            elif event_type == "response.audio.delta":
                if not self._drop_audio_until_done:
                    pcm = base64.b64decode(event.get("delta") or "")
                    if pcm:
                        self._audio_output_queue.put_nowait(pcm)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                user_transcript = str(event.get("transcript") or "").strip()
                self._coordinator.record_user_transcript(user_transcript)
                print(f"\n[你] {user_transcript}", flush=True)
            elif event_type == "input_audio_buffer.speech_started":
                # Qwen VAD cancels an active generated response on the server.
                # Drop already-received PCM as well so local playback stops now.
                interrupted_response = self._active_response_origin is not None
                self._realtime_idle.clear()
                self._user_turn_active = True
                self._response_interrupted_by_new_speech = interrupted_response
                self._drop_audio_until_done = interrupted_response
                if interrupted_response:
                    self._clear_audio_output_queue()
                    origin = (
                        "Leader 播报"
                        if self._active_response_origin == "leader-report"
                        else "语音入口播报"
                    )
                    print(f"\n[麦克风] 检测到说话，已打断{origin}", flush=True)
                else:
                    print("\n[麦克风] 检测到说话", flush=True)
            elif event_type == "input_audio_buffer.speech_stopped":
                print("[麦克风] 检测到停顿，等待 Qwen 回复……", flush=True)
                if event.get("reason") == "turn_invalid":
                    self._user_turn_active = False
                    if self._active_response_origin is None:
                        self._realtime_idle.set()
            elif event_type in {"response.audio_transcript.done", "response.text.done"}:
                transcript = str(event.get("transcript") or event.get("text") or "")
                if transcript:
                    label = (
                        "Leader 语音"
                        if self._active_response_origin == "leader-report"
                        else "语音入口"
                    )
                    print(f"[{label}] {transcript}", flush=True)
                if self._active_response_origin != "leader-report":
                    self._coordinator.mark_ack_completed(transcript)
            elif event_type == "response.function_call_arguments.done":
                if self._active_response_origin == "leader-report":
                    logger.warning("Ignoring function call generated while reading Leader report")
                else:
                    self._coordinator.record_function_call(event)
            elif event_type == "response.done":
                response = event.get("response")
                status = response.get("status") if isinstance(response, dict) else None
                origin = self._active_response_origin
                logger.info(
                    "[VoiceTrace] response.done origin=%s status=%s",
                    origin,
                    status,
                )
                self._active_response_origin = None
                self._drop_audio_until_done = False
                interrupted_by_new_speech = self._response_interrupted_by_new_speech
                self._response_interrupted_by_new_speech = False
                if origin != "leader-report":
                    if status == "cancelled":
                        self._coordinator.cancel_pending()
                    else:
                        logger.info(
                            "[VoiceTrace] turn completed: %s",
                            "tool selected" if getattr(self._coordinator, "_pending", {}) else "no tool selected",
                        )
                        await self._coordinator.flush(
                            write_tool_output=lambda call_id, output: (
                                self._write_tool_output(websocket, call_id, output)
                            ),
                            fallback_acknowledge=self._fallback_acknowledge,
                        )
                if not interrupted_by_new_speech:
                    self._user_turn_active = False
                if not self._user_turn_active:
                    self._realtime_idle.set()
            elif event_type == "error":
                error = event.get("error")
                message_text = error.get("message") if isinstance(error, dict) else error
                logger.error("Qwen realtime error: %s", message_text)

    async def _speak_leader_replies(self, websocket: Any) -> None:
        """Inject Leader text and trigger speech on this same Qwen connection."""
        while True:
            text = await self._leader_reply_queue.get()
            try:
                await self._realtime_idle.wait()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": f"{LEADER_REPORT_PREFIX}{text}",
                                    }
                                ],
                            },
                        },
                        ensure_ascii=False,
                    )
                )
                self._leader_response_requested = True
                self._realtime_idle.clear()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {"modalities": ["audio", "text"]},
                        }
                    )
                )
                # Preserve FIFO ordering. A user interruption still produces
                # response.done(status=cancelled), which releases this wait.
                await self._realtime_idle.wait()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to inject Team Leader report")
            finally:
                self._leader_reply_queue.task_done()

    def _clear_audio_output_queue(self) -> None:
        while True:
            try:
                self._audio_output_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._audio_output_queue.task_done()

    @staticmethod
    async def _wait_for_session_updated(websocket: Any) -> None:
        while True:
            event = json.loads(await websocket.recv())
            event_type = str(event.get("type") or "")
            if event_type == "session.updated":
                return
            if event_type == "error":
                error = event.get("error")
                message = error.get("message") if isinstance(error, dict) else error
                raise RuntimeError(str(message or "Qwen speech session error"))

    async def _play_audio(self, speaker: Any) -> None:
        while True:
            pcm = await self._audio_output_queue.get()
            try:
                await asyncio.to_thread(speaker.write, pcm)
            finally:
                self._audio_output_queue.task_done()

    @staticmethod
    async def _write_tool_output(
        websocket: Any,
        call_id: str,
        output: dict[str, Any],
    ) -> None:
        await websocket.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(output, ensure_ascii=False),
                    },
                },
                ensure_ascii=False,
            )
        )

    @staticmethod
    async def _fallback_acknowledge(message: str) -> None:
        print(f"[语音入口] {message}", flush=True)
