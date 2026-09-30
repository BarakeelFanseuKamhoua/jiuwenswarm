"""Command-line entry point for the Qwen-Audio Team Leader sidecar."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from jiuwenswarm.voice.coordinator import VoiceDispatchCoordinator
from jiuwenswarm.voice.leader_gateway import LeaderGatewayBridge
from jiuwenswarm.voice.qwen_realtime import QwenRealtimeConfig, QwenRealtimeVoiceClient

logger = logging.getLogger(__name__)


def _default_gateway_url() -> str:
    return os.getenv("JIUWENSWARM_GATEWAY_URL") or (
        f"ws://127.0.0.1:{os.getenv('GATEWAY_PORT', '19001')}/tui"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Talk to a persistent JiuwenSwarm Team Leader through Qwen-Audio."
    )
    parser.add_argument("--session", required=True, help="JiuwenSwarm session id")
    parser.add_argument("--gateway-url", default=_default_gateway_url())
    parser.add_argument("--mode", default="team")
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--api-key", default=os.getenv("DASHSCOPE_API_KEY"))
    parser.add_argument("--workspace-id", default=os.getenv("DASHSCOPE_WORKSPACE_ID"))
    parser.add_argument("--model", default="qwen-audio-3.0-realtime-plus")
    parser.add_argument("--voice", default="longanqian")
    parser.add_argument(
        "--turn-detection",
        choices=("server_vad", "smart_turn"),
        default="server_vad",
    )
    parser.add_argument("--silence-duration-ms", type=int, default=800)
    return parser


async def _run(args: argparse.Namespace) -> int:
    if not args.api_key:
        logger.error("DASHSCOPE_API_KEY is not set")
        return 2
    if not args.workspace_id:
        logger.error("DASHSCOPE_WORKSPACE_ID is not set")
        return 2
    if not 200 <= args.silence_duration_ms <= 6000:
        logger.error("--silence-duration-ms must be between 200 and 6000")
        return 2

    project_dir = str(Path(args.project_dir or args.cwd).resolve())
    bridge = LeaderGatewayBridge(
        gateway_url=args.gateway_url,
        session_id=args.session,
        mode=args.mode,
        cwd=args.cwd,
        project_dir=project_dir,
    )
    coordinator = VoiceDispatchCoordinator(bridge)
    voice_client = QwenRealtimeVoiceClient(
        QwenRealtimeConfig(
            api_key=args.api_key,
            workspace_id=args.workspace_id,
            model=args.model,
            voice=args.voice,
            turn_detection=args.turn_detection,
            silence_duration_ms=args.silence_duration_ms,
        ),
        coordinator,
    )
    bridge.set_leader_reply_handler(voice_client.enqueue_leader_reply)
    await bridge.connect()
    try:
        await voice_client.run()
    finally:
        await bridge.close()
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _build_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        print("\n语音会话结束")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
