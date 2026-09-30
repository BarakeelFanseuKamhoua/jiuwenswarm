import { useCallback, useEffect, useRef, useState } from 'react';
import { webClient } from '../services/webClient';
import { onLeaderMessageCommitted } from '../features/leaderMessageCommitted';

type VoiceState = 'idle' | 'connecting' | 'listening' | 'thinking' | 'speaking' | 'error';

type UseRealtimeVoiceOptions = {
  sessionId?: string;
  enabled?: boolean;
  onVoiceCommand: (command: VoiceCommand) => void | Promise<void>;
  onVoiceCommandBatch?: (batch: VoiceCommandBatch) => void | Promise<void>;
  onSpeechStarted?: () => void | Promise<void>;
  onTranscriptCompleted?: (text: string) => void | Promise<void>;
  // A voice turn ended without dispatching any command (no tool selected,
  // or an invalid/noise turn).
  onTurnCompletedWithoutCommand?: () => void | Promise<void>;
  onError?: (message: string) => void;
};

export type VoiceCommandName =
  | 'submit_task'
  | 'supplement_task'
  | 'pause_task'
  | 'resume_task'
  | 'cancel_task'
  | 'get_task_status'
  | 'ask_leader';

export type VoiceCommand = {
  name: VoiceCommandName;
  text: string;
  dispatchText: string;
  summary?: string;
  reason?: string;
  query?: string;
  targetTaskIds: string[];
};

export type VoiceCommandBatch = {
  callIds: string[];
  text: string;
  dispatchText: string;
  name: VoiceCommandName;
  summary?: string;
  // Re-dispatch of an utterance whose realtime response was cut off by the
  // next utterance; it is not tied to the live speech turn.
  replayed?: boolean;
  commands: Array<{
    callId: string;
    name: VoiceCommandName;
    reason?: string;
    query?: string;
    targetTaskIds: string[];
    summary?: string;
    instruction?: string;
  }>;
};

type VoiceEventPayload = {
  session_id?: string;
  audio?: string;
  sample_rate?: number;
  text?: string;
  name?: VoiceCommandName;
  dispatch_text?: string;
  instruction?: string;
  summary?: string;
  state?: VoiceState;
  message?: string;
  /** voice.error only: the realtime connection is gone and cannot be reused. */
  closed?: boolean;
  reason?: string;
  query?: string;
  target_task_ids?: string[];
  role?: string;
  member_name?: string;
  source_member?: string;
  content?: string;
  is_complete?: boolean;
  interrupted?: boolean;
  call_ids?: string[];
  count?: number;
  replayed?: boolean;
  commands?: Array<{
    call_id?: string;
    name?: VoiceCommandName;
    reason?: string;
    query?: string;
    target_task_ids?: string[];
    summary?: string;
    instruction?: string;
  }>;
};

type VoiceTeamMember = {
  member_id: string;
  name?: string;
  role?: string;
};

type VoiceTeamInfo = {
  team_id: string;
  members: VoiceTeamMember[];
  tasks: VoiceTeamTask[];
};

type VoiceTeamTask = {
  task_id: string;
  title: string;
  status: string;
  created_order: number;
};

type VoiceTeamMemberMutation = {
  sessionId: string;
  teamId: string;
  type: 'upsert' | 'remove';
  member: VoiceTeamMember;
};

type VoiceTeamTaskMutation = {
  sessionId: string;
  teamId: string;
  task: Omit<VoiceTeamTask, 'created_order'>;
};

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function cleanContextText(value: unknown, maxLength = 100): string {
  return typeof value === 'string'
    ? value.replace(/\s+/g, ' ').trim().slice(0, maxLength)
    : '';
}

function normalizeVoiceTeamMember(value: unknown): VoiceTeamMember | undefined {
  const member = asRecord(value);
  if (!member) return undefined;
  const memberId = cleanContextText(member.member_id);
  if (!memberId) return undefined;
  const name = cleanContextText(member.name);
  const role = cleanContextText(member.role ?? member.mode, 60);
  return {
    member_id: memberId,
    ...(name ? { name } : {}),
    ...(role ? { role } : {}),
  };
}

function normalizeVoiceTeamInfo(value: unknown): VoiceTeamInfo {
  const snapshot = asRecord(value);
  const members = new Map<string, VoiceTeamMember>();
  if (Array.isArray(snapshot?.members)) {
    snapshot.members.forEach(item => {
      const member = normalizeVoiceTeamMember(item);
      if (member) members.set(member.member_id, member);
    });
  }
  const tasks: VoiceTeamTask[] = [];
  if (Array.isArray(snapshot?.tasks)) {
    snapshot.tasks.forEach((item, index) => {
      const task = asRecord(item);
      if (!task) return;
      const taskId = cleanContextText(task.task_id, 128);
      if (!taskId) return;
      tasks.push({
        task_id: taskId,
        title: cleanContextText(task.title ?? task.name ?? task.content, 160) || '未命名任务',
        status: cleanContextText(task.status, 40) || 'unknown',
        created_order: index + 1,
      });
    });
  }
  return {
    team_id: cleanContextText(snapshot?.team_id, 128),
    members: [...members.values()].sort((left, right) => left.member_id.localeCompare(right.member_id)),
    tasks,
  };
}

function readVoiceTeamMemberMutation(payload: unknown): VoiceTeamMemberMutation | undefined {
  const outer = asRecord(payload);
  if (!outer) return undefined;
  const nested = asRecord(outer.payload) ?? outer;
  const event = asRecord(nested.event) ?? asRecord(outer.event);
  if (!event) return undefined;
  const sessionId = cleanContextText(outer.session_id ?? nested.session_id, 160);
  const eventType = cleanContextText(event.type, 80);
  const member = normalizeVoiceTeamMember(event);
  if (!sessionId || !member) return undefined;
  if (eventType === 'team.member.shutdown') {
    return {
      sessionId,
      teamId: cleanContextText(event.team_id, 128),
      type: 'remove',
      member,
    };
  }
  if (![
    'team.member.registered',
    'team.member.spawned',
    'team.member.restarted',
  ].includes(eventType)) {
    return undefined;
  }
  return {
    sessionId,
    teamId: cleanContextText(event.team_id, 128),
    type: 'upsert',
    member,
  };
}

function readVoiceTeamTaskMutation(payload: unknown): VoiceTeamTaskMutation | undefined {
  const outer = asRecord(payload);
  if (!outer) return undefined;
  const nested = asRecord(outer.payload) ?? outer;
  const event = asRecord(nested.event) ?? asRecord(outer.event);
  if (!event) return undefined;
  const sessionId = cleanContextText(outer.session_id ?? nested.session_id, 160);
  const eventType = cleanContextText(event.type, 80);
  const taskId = cleanContextText(event.task_id, 128);
  if (!sessionId || !taskId || !eventType.startsWith('team.task.')) return undefined;
  const terminalStatus = eventType === 'team.task.cancelled'
    ? 'cancelled'
    : eventType === 'team.task.completed'
      ? 'completed'
      : '';
  return {
    sessionId,
    teamId: cleanContextText(event.team_id ?? event.team_name, 128),
    task: {
      task_id: taskId,
      title: cleanContextText(
        event.title ?? event.name ?? event.description ?? event.content,
        160,
      ),
      status: cleanContextText(event.status, 40) || terminalStatus || 'unknown',
    },
  };
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const stride = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += stride) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + stride));
  }
  return btoa(binary);
}

function base64ToBytes(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function resampleToPcm16(input: Float32Array, sourceRate: number, targetRate = 16_000): Uint8Array {
  const ratio = sourceRate / targetRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Int16Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const sourceIndex = index * ratio;
    const left = Math.floor(sourceIndex);
    const right = Math.min(left + 1, input.length - 1);
    const fraction = sourceIndex - left;
    const sample = input[left] * (1 - fraction) + input[right] * fraction;
    output[index] = Math.max(-1, Math.min(1, sample)) * 0x7fff;
  }
  return new Uint8Array(output.buffer);
}

export function useRealtimeVoice({
  sessionId,
  enabled = true,
  onVoiceCommand,
  onVoiceCommandBatch,
  onSpeechStarted,
  onTranscriptCompleted,
  onTurnCompletedWithoutCommand,
  onError,
}: UseRealtimeVoiceOptions) {
  const [state, setState] = useState<VoiceState>('idle');
  const [isActive, setIsActive] = useState(false);
  const [isMicrophoneEnabled, setIsMicrophoneEnabled] = useState(false);
  const [transcript, setTranscript] = useState('');
  const captureContextRef = useRef<AudioContext | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const workletUrlRef = useRef<string | null>(null);
  const playbackTimeRef = useRef(0);
  const playbackSourcesRef = useRef(new Set<AudioBufferSourceNode>());
  const playbackGenerationRef = useRef(0);
  const activeRef = useRef(false);
  const microphoneEnabledRef = useRef(false);
  // Serializes microphone starts: getUserMedia/addModule are async, so a second
  // start during that window would otherwise build a parallel capture pipeline
  // that stop can no longer reach, and the server would receive duplicated audio.
  const microphoneStartRef = useRef<Promise<void> | null>(null);
  const captureGenerationRef = useRef(0);
  const captureChunksRef = useRef<Float32Array[]>([]);
  const captureFramesRef = useRef(0);
  const teamInfoRef = useRef<VoiceTeamInfo>({ team_id: '', members: [], tasks: [] });
  const spokenLeaderMessageIdsRef = useRef(new Set<string>());
  const commandHandlerRef = useRef(onVoiceCommand);
  const commandBatchHandlerRef = useRef(onVoiceCommandBatch);
  const speechStartedHandlerRef = useRef(onSpeechStarted);
  const transcriptCompletedHandlerRef = useRef(onTranscriptCompleted);
  const turnCompletedWithoutCommandHandlerRef = useRef(onTurnCompletedWithoutCommand);
  const errorHandlerRef = useRef(onError);

  useEffect(() => {
    commandHandlerRef.current = onVoiceCommand;
    commandBatchHandlerRef.current = onVoiceCommandBatch;
    speechStartedHandlerRef.current = onSpeechStarted;
    transcriptCompletedHandlerRef.current = onTranscriptCompleted;
    turnCompletedWithoutCommandHandlerRef.current = onTurnCompletedWithoutCommand;
    errorHandlerRef.current = onError;
  }, [
    onError,
    onSpeechStarted,
    onTranscriptCompleted,
    onTurnCompletedWithoutCommand,
    onVoiceCommand,
    onVoiceCommandBatch,
  ]);

  const clearPlayback = useCallback(() => {
    playbackGenerationRef.current += 1;
    playbackSourcesRef.current.forEach(source => {
      try {
        source.stop();
      } catch {
        // Already stopped.
      }
    });
    playbackSourcesRef.current.clear();
    playbackTimeRef.current = playbackContextRef.current?.currentTime ?? 0;
  }, []);

  const playPcm = useCallback(async (audioBase64: string, sampleRate = 24_000) => {
    const generation = playbackGenerationRef.current;
    let context = playbackContextRef.current;
    if (!context) {
      context = new AudioContext({ sampleRate });
      playbackContextRef.current = context;
    }
    await context.resume();
    if (generation !== playbackGenerationRef.current) return;
    const bytes = base64ToBytes(audioBase64);
    const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
    const buffer = context.createBuffer(1, pcm.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let index = 0; index < pcm.length; index += 1) {
      channel[index] = pcm[index] / 0x8000;
    }
    const source = context.createBufferSource();
    if (generation !== playbackGenerationRef.current) return;
    source.buffer = buffer;
    source.connect(context.destination);
    const startAt = Math.max(context.currentTime + 0.02, playbackTimeRef.current);
    source.start(startAt);
    playbackTimeRef.current = startAt + buffer.duration;
    playbackSourcesRef.current.add(source);
    source.onended = () => playbackSourcesRef.current.delete(source);
    setState('speaking');
  }, []);

  const preparePlayback = useCallback(async () => {
    let context = playbackContextRef.current;
    if (!context) {
      context = new AudioContext({ sampleRate: 24_000 });
      playbackContextRef.current = context;
    }
    await context.resume();
  }, []);

  const stopMicrophone = useCallback(
    async (finalizeTurn = true) => {
      const wasEnabled = microphoneEnabledRef.current;
      microphoneEnabledRef.current = false;
      captureGenerationRef.current += 1;
      setIsMicrophoneEnabled(false);
      const context = captureContextRef.current;

      try {
        if (wasEnabled && finalizeTurn && activeRef.current && sessionId) {
          const pendingFrames = captureFramesRef.current;
          if (context && pendingFrames > 0) {
            const merged = new Float32Array(pendingFrames);
            let offset = 0;
            captureChunksRef.current.forEach(chunk => {
              merged.set(chunk, offset);
              offset += chunk.length;
            });
            const pcm = resampleToPcm16(merged, context.sampleRate);
            await webClient.sendFireAndForget('voice.session.audio', {
              session_id: sessionId,
              audio: bytesToBase64(pcm),
            });
          }

          // Let server_vad observe enough trailing silence to close an utterance
          // even when the user mutes immediately after finishing a sentence.
          const trailingSilence = new Uint8Array(Math.ceil(16_000 * 0.9) * 2);
          await webClient.sendFireAndForget('voice.session.audio', {
            session_id: sessionId,
            audio: bytesToBase64(trailingSilence),
          });
        }
      } finally {
        captureChunksRef.current = [];
        captureFramesRef.current = 0;
        workletRef.current?.disconnect();
        sourceRef.current?.disconnect();
        streamRef.current?.getTracks().forEach(track => track.stop());
        workletRef.current = null;
        sourceRef.current = null;
        streamRef.current = null;
        if (captureContextRef.current) {
          await captureContextRef.current.close();
          captureContextRef.current = null;
        }
        if (workletUrlRef.current) {
          URL.revokeObjectURL(workletUrlRef.current);
          workletUrlRef.current = null;
        }
      }
    },
    [sessionId],
  );

  const startMicrophoneOnce = useCallback(async () => {
    if (!activeRef.current || microphoneEnabledRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      throw new Error('当前浏览器不支持实时语音所需的 AudioWorklet。');
    }
    const generation = captureGenerationRef.current;

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    const context = new AudioContext();
    const workletSource = `
      class JiuwenVoiceCapture extends AudioWorkletProcessor {
        process(inputs) {
          const channel = inputs[0] && inputs[0][0];
          if (channel) this.port.postMessage(channel.slice());
          return true;
        }
      }
      registerProcessor('jiuwen-voice-capture', JiuwenVoiceCapture);
    `;
    const workletUrl = URL.createObjectURL(new Blob([workletSource], { type: 'text/javascript' }));

    try {
      await context.audioWorklet.addModule(workletUrl);
      if (generation !== captureGenerationRef.current || !activeRef.current) {
        // Muted or stopped while the pipeline was being built.
        stream.getTracks().forEach(track => track.stop());
        await context.close();
        URL.revokeObjectURL(workletUrl);
        return;
      }
      const source = context.createMediaStreamSource(stream);
      const worklet = new AudioWorkletNode(context, 'jiuwen-voice-capture');
      const muted = context.createGain();
      muted.gain.value = 0;
      source.connect(worklet);
      worklet.connect(muted).connect(context.destination);
      microphoneEnabledRef.current = true;
      worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
        if (!activeRef.current || !microphoneEnabledRef.current) return;
        if (workletRef.current !== worklet) return;
        captureChunksRef.current.push(event.data);
        captureFramesRef.current += event.data.length;
        if (captureFramesRef.current < context.sampleRate / 10) return;
        const merged = new Float32Array(captureFramesRef.current);
        let offset = 0;
        captureChunksRef.current.forEach(chunk => {
          merged.set(chunk, offset);
          offset += chunk.length;
        });
        captureChunksRef.current = [];
        captureFramesRef.current = 0;
        const pcm = resampleToPcm16(merged, context.sampleRate);
        void webClient.sendFireAndForget('voice.session.audio', {
          session_id: sessionId,
          audio: bytesToBase64(pcm),
        });
      };
      captureContextRef.current = context;
      streamRef.current = stream;
      sourceRef.current = source;
      workletRef.current = worklet;
      workletUrlRef.current = workletUrl;
      setIsMicrophoneEnabled(true);
    } catch (error) {
      microphoneEnabledRef.current = false;
      stream.getTracks().forEach(track => track.stop());
      await context.close();
      URL.revokeObjectURL(workletUrl);
      throw error;
    }
  }, [sessionId]);

  const startMicrophone = useCallback(() => {
    if (microphoneStartRef.current) return microphoneStartRef.current;
    const pending = startMicrophoneOnce().finally(() => {
      if (microphoneStartRef.current === pending) microphoneStartRef.current = null;
    });
    microphoneStartRef.current = pending;
    return pending;
  }, [startMicrophoneOnce]);

  const stop = useCallback(async () => {
    activeRef.current = false;
    setIsActive(false);
    setTranscript('');
    await stopMicrophone(false);
    clearPlayback();
    if (playbackContextRef.current) {
      await playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
    if (sessionId) {
      try {
        await webClient.request('voice.session.stop', { session_id: sessionId });
      } catch {
        // Closing an already-disconnected voice session is harmless.
      }
    }
    setState('idle');
    teamInfoRef.current = { team_id: '', members: [], tasks: [] };
  }, [clearPlayback, sessionId, stopMicrophone]);
  const stopRef = useRef(stop);
  stopRef.current = stop;

  const start = useCallback(async () => {
    if (!enabled || !sessionId || activeRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      const message = '当前浏览器不支持实时语音所需的 AudioWorklet。';
      setState('error');
      errorHandlerRef.current?.(message);
      return;
    }
    setTranscript('');
    setState('connecting');
    try {
      await preparePlayback();
      let teamInfo: VoiceTeamInfo = { team_id: '', members: [], tasks: [] };
      try {
        const snapshot = await webClient.request<Record<string, unknown>>(
          'team.snapshot',
          { session_id: sessionId },
          { timeoutMs: 5_000 },
        );
        teamInfo = normalizeVoiceTeamInfo(snapshot);
      } catch (error) {
        console.warn('[voice] Failed to load initial team directory:', error);
      }
      teamInfoRef.current = teamInfo;
      await webClient.request(
        'voice.session.start',
        {
          session_id: sessionId,
          turn_detection: 'server_vad',
          team_info: teamInfo,
        },
        { timeoutMs: 20_000 },
      );
      activeRef.current = true;
      setIsActive(true);
      await startMicrophone();
      setState('listening');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await stop();
      setState('error');
      errorHandlerRef.current?.(message);
    }
  }, [enabled, preparePlayback, sessionId, startMicrophone, stop]);

  const toggleMicrophone = useCallback(async () => {
    if (!activeRef.current) return;
    try {
      if (microphoneEnabledRef.current) {
        await stopMicrophone(true);
      } else {
        await startMicrophone();
        setState('listening');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      errorHandlerRef.current?.(message);
    }
  }, [startMicrophone, stopMicrophone]);

  useEffect(() => {
    const offAudio = webClient.on<VoiceEventPayload>('voice.audio', event => {
      if (event.payload.session_id !== sessionId || !event.payload.audio) return;
      void playPcm(event.payload.audio, event.payload.sample_rate);
    });
    const offInterrupted = webClient.on<VoiceEventPayload>('voice.interrupted', event => {
      if (event.payload.session_id !== sessionId) return;
      clearPlayback();
      // A transcript belongs only to the utterance that produced it. Clear the
      // previous turn as soon as VAD detects a new one so the status bar never
      // presents stale speech as the current input.
      setTranscript('');
      setState('listening');
      void speechStartedHandlerRef.current?.();
    });
    const offStatus = webClient.on<VoiceEventPayload>('voice.status', event => {
      if (event.payload.session_id !== sessionId || !event.payload.state) return;
      // `listening` after a completed response marks the end of the previous
      // utterance. The user message is already committed to chat history, so
      // the status bar must not keep presenting its transcript as current.
      if (event.payload.state === 'listening') {
        setTranscript('');
      }
      setState(event.payload.state);
    });
    const offTranscript = webClient.on<VoiceEventPayload>('voice.transcript', event => {
      if (event.payload.session_id !== sessionId || !event.payload.text) return;
      setTranscript(event.payload.text);
      void transcriptCompletedHandlerRef.current?.(event.payload.text);
    });
    const buildCommand = (payload: VoiceEventPayload): VoiceCommand | null => {
      if (!payload.text || !payload.name) return null;
      return {
        name: payload.name,
        text: payload.text,
        dispatchText: payload.dispatch_text?.trim() || payload.instruction?.trim() || payload.text,
        summary: payload.summary?.trim(),
        reason: payload.reason?.trim(),
        query: payload.query?.trim(),
        targetTaskIds: Array.isArray(payload.target_task_ids)
          ? payload.target_task_ids.filter((taskId): taskId is string => typeof taskId === 'string')
          : [],
      };
    };
    const dispatchCommand = (command: VoiceCommand) => {
      // The transcript has now been converted into a durable user bubble and
      // a structured command. Keep it out of the transient voice status bar
      // while Leader handles the request.
      setTranscript('');
      void commandHandlerRef.current(command);
    };
    // Unified batch path: the backend renders one voice turn (even a single
    // intent) as one ``voice.command_batch`` whose ``dispatch_text`` already
    // collapses every intent into a single Leader request through
    // render_voice_batch. The frontend dispatches that one text once — never
    // N serial chat.send frames, which the Gateway cancels pairwise.
    const offBatch = webClient.on<VoiceEventPayload>('voice.command_batch', event => {
      if (event.payload.session_id !== sessionId) return;
      const callIds = Array.isArray(event.payload.call_ids)
        ? event.payload.call_ids.filter((id): id is string => typeof id === 'string')
        : [];
      const rawCommands = Array.isArray(event.payload.commands) ? event.payload.commands : [];
      const dispatchText = event.payload.dispatch_text?.trim() || event.payload.text || '';
      if (!dispatchText || callIds.length === 0) return;
      const batch: VoiceCommandBatch = {
        callIds,
        text: event.payload.text || dispatchText,
        dispatchText,
        name: event.payload.name || (rawCommands[0]?.name ?? 'submit_task'),
        summary: event.payload.summary?.trim(),
        replayed: event.payload.replayed === true,
        commands: rawCommands.map(cmd => ({
          callId: cmd.call_id || '',
          name: cmd.name || 'submit_task',
          reason: cmd.reason?.trim(),
          query: cmd.query?.trim(),
          targetTaskIds: Array.isArray(cmd.target_task_ids)
            ? cmd.target_task_ids.filter((id): id is string => typeof id === 'string')
            : [],
          summary: cmd.summary?.trim(),
          instruction: cmd.instruction?.trim(),
        })),
      };
      setTranscript('');
      // Prefer the dedicated batch handler; fall back to the single-command
      // handler (legacy ChatPanel) by synthesizing one VoiceCommand whose
      // dispatchText is the fully aggregated batch text.
      if (commandBatchHandlerRef.current) {
        void commandBatchHandlerRef.current(batch);
      } else {
        dispatchCommand({
          name: batch.name,
          text: batch.text,
          dispatchText: batch.dispatchText,
          summary: batch.summary,
          targetTaskIds: batch.commands[0]?.targetTaskIds ?? [],
        });
      }
    });
    const offCommand = webClient.on<VoiceEventPayload>('voice.command', event => {
      const payload = event.payload;
      if (payload.session_id !== sessionId) return;
      const command = buildCommand(payload);
      if (!command) return;
      // Legacy backend without the unified batch event: behave as a
      // single-call turn and dispatch the pre-aggregated text directly.
      dispatchCommand(command);
    });
    const offError = webClient.on<VoiceEventPayload>('voice.error', event => {
      if (event.payload.session_id !== sessionId) return;
      const message = event.payload.message || '实时语音连接错误';
      if (event.payload.closed) {
        // The realtime connection ended (e.g. the provider's idle timeout).
        // End this voice session so turning the microphone back on starts a
        // new one rather than streaming into the closed connection.
        void stopRef.current().finally(() => {
          setState('error');
          errorHandlerRef.current?.(message);
        });
        return;
      }
      setState('error');
      errorHandlerRef.current?.(message);
    });
    const offTurnCompleted = webClient.on<VoiceEventPayload>('voice.turn_completed', event => {
      if (event.payload.session_id !== sessionId) return;
      void turnCompletedWithoutCommandHandlerRef.current?.();
    });
    const offLeaderMessageCommitted = onLeaderMessageCommitted(message => {
      if (!activeRef.current || message.sessionId !== sessionId) return;
      const dedupKey = `${message.sessionId}\u0000${message.messageId}`;
      if (spokenLeaderMessageIdsRef.current.has(dedupKey)) return;
      spokenLeaderMessageIdsRef.current.add(dedupKey);
      if (spokenLeaderMessageIdsRef.current.size > 200) {
        const oldest = spokenLeaderMessageIdsRef.current.values().next().value;
        if (oldest) spokenLeaderMessageIdsRef.current.delete(oldest);
      }
      void webClient.sendFireAndForget('voice.session.leader_reply', {
        session_id: sessionId,
        message_id: message.messageId,
        text: message.content,
      });
    });
    const offTeamMember = webClient.on<Record<string, unknown>>('team.member', event => {
      if (!activeRef.current || !sessionId) return;
      const mutation = readVoiceTeamMemberMutation(event.payload);
      if (!mutation || mutation.sessionId !== sessionId) return;

      const previous = teamInfoRef.current;
      const members = new Map(previous.members.map(member => [member.member_id, member]));
      if (mutation.type === 'remove') {
        if (!members.delete(mutation.member.member_id)) return;
      } else {
        const existing = members.get(mutation.member.member_id);
        const merged: VoiceTeamMember = {
          member_id: mutation.member.member_id,
          name: mutation.member.name || existing?.name,
          role: mutation.member.role || existing?.role,
        };
        if (existing && JSON.stringify(existing) === JSON.stringify(merged)) return;
        members.set(merged.member_id, merged);
      }

      const next: VoiceTeamInfo = {
        team_id: mutation.teamId || previous.team_id,
        members: [...members.values()].sort((left, right) => left.member_id.localeCompare(right.member_id)),
        tasks: previous.tasks,
      };
      teamInfoRef.current = next;
      void webClient.sendFireAndForget('voice.session.team_info', {
        session_id: sessionId,
        team_info: next,
      });
    });
    const offTeamTask = webClient.on<Record<string, unknown>>('team.task', event => {
      if (!activeRef.current || !sessionId) return;
      const mutation = readVoiceTeamTaskMutation(event.payload);
      if (!mutation || mutation.sessionId !== sessionId) return;

      const previous = teamInfoRef.current;
      const tasks = new Map(previous.tasks.map(task => [task.task_id, task]));
      const existing = tasks.get(mutation.task.task_id);
      const createdOrder = existing?.created_order
        ?? Math.max(0, ...previous.tasks.map(task => task.created_order)) + 1;
      tasks.set(mutation.task.task_id, {
        task_id: mutation.task.task_id,
        title: mutation.task.title || existing?.title || '未命名任务',
        status: mutation.task.status === 'unknown'
          ? existing?.status || 'unknown'
          : mutation.task.status,
        created_order: createdOrder,
      });
      const next: VoiceTeamInfo = {
        team_id: mutation.teamId || previous.team_id,
        members: previous.members,
        tasks: [...tasks.values()].sort((left, right) => left.created_order - right.created_order),
      };
      teamInfoRef.current = next;
      void webClient.sendFireAndForget('voice.session.team_info', {
        session_id: sessionId,
        team_info: next,
      });
    });
    return () => {
      offAudio();
      offInterrupted();
      offStatus();
      offTranscript();
      offCommand();
      offBatch();
      offError();
      offTurnCompleted();
      offLeaderMessageCommitted();
      offTeamMember();
      offTeamTask();
    };
  }, [clearPlayback, playPcm, sessionId]);

  useEffect(
    () => () => {
      if (activeRef.current) void stop();
    },
    [stop],
  );

  useEffect(() => {
    if (!enabled && activeRef.current) void stop();
  }, [enabled, stop]);

  return {
    state,
    transcript,
    isActive,
    isMicrophoneEnabled,
    isSupported: typeof navigator !== 'undefined' && 'mediaDevices' in navigator && typeof window !== 'undefined' && 'AudioWorkletNode' in window,
    preparePlayback,
    toggleMicrophone,
    start,
    stop,
  };
}

export type RealtimeVoiceController = ReturnType<typeof useRealtimeVoice>;
