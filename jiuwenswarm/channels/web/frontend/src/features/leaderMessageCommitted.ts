export type LeaderMessageCommitted = {
  sessionId: string;
  messageId: string;
  content: string;
};

type LeaderMessageCommittedListener = (message: LeaderMessageCommitted) => void;

const listeners = new Set<LeaderMessageCommittedListener>();

/**
 * Notify consumers only after the same complete Leader message shown by the
 * chat UI has been committed to the local message store.
 */
export function emitLeaderMessageCommitted(message: LeaderMessageCommitted): void {
  const content = message.content.trim();
  if (!message.sessionId || !message.messageId || !content) return;
  listeners.forEach(listener => listener({ ...message, content }));
}

export function onLeaderMessageCommitted(listener: LeaderMessageCommittedListener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
