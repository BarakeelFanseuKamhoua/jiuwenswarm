import { useCallback, useEffect, useRef, useState } from 'react';
import { Radio, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { webClient } from '../../services/webClient';
import { useSessionStore } from '../../stores';

interface VoiceMirrorControlProps {
  sessionId: string;
  compact?: boolean;
  onBindingChange: (voiceSessionId: string | null) => void;
}

interface VoiceMirrorResponse {
  bound: boolean;
  read_only: boolean;
  voice_session_id?: string;
  web_session_id?: string;
}

const STORAGE_PREFIX = 'jiuwenswarm.voiceMirror.';
const DEFAULT_VOICE_SESSION_ID = 'voice-leader-test';

function storedVoiceSessionId(sessionId: string): string {
  try {
    return window.localStorage.getItem(`${STORAGE_PREFIX}${sessionId}`)?.trim() ?? '';
  } catch {
    return '';
  }
}

function persistVoiceSessionId(sessionId: string, voiceSessionId: string | null): void {
  try {
    const key = `${STORAGE_PREFIX}${sessionId}`;
    if (voiceSessionId) {
      window.localStorage.setItem(key, voiceSessionId);
    } else {
      window.localStorage.removeItem(key);
    }
  } catch {
    // Storage can be unavailable in a restricted browser profile; the live binding still works.
  }
}

export function VoiceMirrorControl({ sessionId, compact = false, onBindingChange }: VoiceMirrorControlProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [source, setSource] = useState(DEFAULT_VOICE_SESSION_ID);
  const [boundSource, setBoundSource] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const mountedRef = useRef(true);

  const bind = useCallback(
    async (candidate: string, quiet = false) => {
      const voiceSessionId = candidate.trim();
      if (!voiceSessionId) {
        setError(t('voiceMirror.required'));
        return;
      }
      setPending(true);
      if (!quiet) setError('');
      try {
        const response = await webClient.request<VoiceMirrorResponse>('voice.mirror.bind', {
          session_id: sessionId,
          voice_session_id: voiceSessionId,
          mode: 'team',
        });
        if (!mountedRef.current || !response.bound) return;
        persistVoiceSessionId(sessionId, voiceSessionId);
        setSource(voiceSessionId);
        setBoundSource(voiceSessionId);
        const sessionStore = useSessionStore.getState();
        sessionStore.ensureRuntime(sessionId);
        sessionStore.setMode(sessionId, 'team');
        onBindingChange(voiceSessionId);
        setOpen(false);
      } catch (reason) {
        if (!mountedRef.current || quiet) return;
        setError(reason instanceof Error ? reason.message : t('voiceMirror.bindFailed'));
      } finally {
        if (mountedRef.current) setPending(false);
      }
    },
    [onBindingChange, sessionId, t],
  );

  const unbind = useCallback(async () => {
    setPending(true);
    setError('');
    try {
      await webClient.request<VoiceMirrorResponse>('voice.mirror.unbind', {
        session_id: sessionId,
      });
      persistVoiceSessionId(sessionId, null);
      setBoundSource(null);
      onBindingChange(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : t('voiceMirror.unbindFailed'));
    } finally {
      setPending(false);
    }
  }, [onBindingChange, sessionId, t]);

  useEffect(() => {
    mountedRef.current = true;
    const stored = storedVoiceSessionId(sessionId);
    setSource(stored || DEFAULT_VOICE_SESSION_ID);
    setBoundSource(null);
    onBindingChange(null);

    if (stored && webClient.getState() === 'ready') {
      void bind(stored, true);
    }
    const unsubscribe = webClient.onStateChange(state => {
      if (state !== 'ready') return;
      const current = storedVoiceSessionId(sessionId);
      if (current) void bind(current, true);
    });
    return () => {
      mountedRef.current = false;
      unsubscribe();
    };
  }, [bind, onBindingChange, sessionId]);

  return (
    <div className={`voice-mirror${compact ? ' voice-mirror--compact' : ''}`}>
      <button
        type="button"
        className={`chat-header-icon-btn voice-mirror__trigger${boundSource ? ' voice-mirror__trigger--bound' : ''}`}
        title={boundSource ? t('voiceMirror.boundTitle', { sessionId: boundSource }) : t('voiceMirror.title')}
        aria-label={t('voiceMirror.title')}
        aria-expanded={open}
        onClick={() => setOpen(value => !value)}
      >
        <Radio size={16} strokeWidth={2} aria-hidden="true" />
      </button>
      {open && (
        <div className="voice-mirror__panel">
          <div className="voice-mirror__panel-header">
            <div>
              <div className="voice-mirror__title">{t('voiceMirror.title')}</div>
              <div className="voice-mirror__description">{t('voiceMirror.description')}</div>
            </div>
            <button type="button" className="chat-header-icon-btn" aria-label={t('common.close')} onClick={() => setOpen(false)}>
              <X size={15} strokeWidth={2} aria-hidden="true" />
            </button>
          </div>
          <label className="voice-mirror__label" htmlFor={`voice-mirror-source-${sessionId}`}>
            {t('voiceMirror.sourceLabel')}
          </label>
          <input
            id={`voice-mirror-source-${sessionId}`}
            className="voice-mirror__input"
            value={source}
            disabled={pending}
            onChange={event => setSource(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') void bind(source);
            }}
          />
          {error && <div className="voice-mirror__error">{error}</div>}
          {boundSource && <div className="voice-mirror__status">{t('voiceMirror.boundStatus', { sessionId: boundSource })}</div>}
          <div className="voice-mirror__actions">
            {boundSource && (
              <button type="button" className="voice-mirror__button" disabled={pending} onClick={() => void unbind()}>
                {t('voiceMirror.unbind')}
              </button>
            )}
            <button type="button" className="voice-mirror__button voice-mirror__button--primary" disabled={pending} onClick={() => void bind(source)}>
              {pending ? t('voiceMirror.binding') : t('voiceMirror.bind')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export function VoiceMirrorReadOnlyNotice({ source }: { source: string }) {
  const { t } = useTranslation();
  return (
    <div className="voice-mirror-readonly" role="status">
      <Radio size={16} strokeWidth={2} aria-hidden="true" />
      <div>
        <div className="voice-mirror-readonly__title">{t('voiceMirror.readOnlyTitle')}</div>
        <div className="voice-mirror-readonly__description">{t('voiceMirror.readOnlyDescription', { sessionId: source })}</div>
      </div>
    </div>
  );
}
