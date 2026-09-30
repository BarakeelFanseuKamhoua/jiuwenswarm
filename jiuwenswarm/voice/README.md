# Voice / Team integration

语音相关能力统一放在本目录，按职责分层：

- `models.py`：语音工具调用和 Team 指令的数据模型。
- `team_api.py`：Team 模式唯一依赖的稳定接口（`TeamVoiceInterface`）。
- `leader_gateway.py`：把指令转换为现有 Gateway 的 `chat.send` / `chat.interrupt`。
- `coordinator.py`：语音确认完成后，负责 exactly-once 调度。
- `qwen_realtime.py`：Qwen Realtime provider 及音频收发。
- `web_session.py`：浏览器 WebSocket 会话和事件协议。
- `cli.py`：本地命令行入口。

Team runtime 或其它语音 provider 不需要依赖 Qwen 实现，只需提供或使用：

```python
from jiuwenswarm.voice import TeamVoiceInterface, create_team_voice_interface

# bridge 可以是 LeaderGatewayBridge，也可以是实现该协议的原生 Team 适配器。
voice: TeamVoiceInterface = create_team_voice_interface(bridge)
await voice.dispatch_command(turn_id, command)
```

`jiuwenswarm.gateway.voice_mirror` 保留为兼容入口；新的语音代码应从
`jiuwenswarm.voice` 导入。
