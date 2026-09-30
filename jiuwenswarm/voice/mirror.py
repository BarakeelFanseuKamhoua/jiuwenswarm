"""Voice-session event mirroring.

Implementation lives in the voice package so all voice concerns are grouped
together.  ``jiuwenswarm.gateway.voice_mirror`` remains as a compatibility
module for existing imports.
"""

from jiuwenswarm.gateway.voice_mirror import VoiceMirrorBinding, VoiceMirrorRegistry

__all__ = ["VoiceMirrorBinding", "VoiceMirrorRegistry"]
