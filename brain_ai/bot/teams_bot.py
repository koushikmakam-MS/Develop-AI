"""
BCDR DeveloperAI Teams Bot — Microsoft Teams integration using Bot Framework SDK.

Wraps the existing BrainAgent so every Teams message goes through
the same routing + agent pipeline as the CLI chat.

Features:
  - Responds when @mentioned in a channel
  - Monitors channel activity; if nobody responds within 10 minutes,
    the bot proactively answers the unanswered question
  - Each Teams conversation gets its own BrainAgent session
  - Handles personal (1:1), group chat, and Teams channel scopes
"""

import asyncio
import logging
import time
from typing import Dict, Optional

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import (
    Activity,
    ActivityTypes,
    ConversationReference,
)

from brain_ai.agents.brain_agent import BrainAgent
from brain_ai.config import get_config

log = logging.getLogger(__name__)


# ── Pending message tracker (for 10-minute auto-reply) ──────────────────

class _PendingMessage:
    """Tracks an unanswered channel message."""
    __slots__ = ("text", "conv_ref", "timestamp", "answered")

    def __init__(self, text: str, conv_ref: ConversationReference):
        self.text = text
        self.conv_ref = conv_ref
        self.timestamp = time.time()
        self.answered = False


class BrainAIBot(ActivityHandler):
    """Microsoft Teams / Bot Framework bot backed by BrainAgent."""

    AGENT_EMOJI = {
        "knowledge": "📚",
        "debug": "🔍",
        "coder": "💻",
        "knowledge_updater": "📝",
        "brain": "🧠",
    }

    def __init__(self, cfg: dict | None = None, adapter=None):
        super().__init__()
        if cfg is None:
            cfg = get_config()
        self.cfg = cfg
        self._adapter = adapter  # needed for proactive messaging

        # Read auto-reply delay from config (default 10 minutes)
        bot_cfg = cfg.get("teams_bot", {})
        self.AUTO_REPLY_DELAY = bot_cfg.get("auto_reply_delay_minutes", 10) * 60

        # One BrainAgent per conversation-id so sessions don't bleed
        self._sessions: Dict[str, BrainAgent] = {}

        # Channel messages awaiting a human reply (key = activity-id)
        self._pending: Dict[str, _PendingMessage] = {}

        # Background task ref
        self._monitor_task: Optional[asyncio.Task] = None

        log.info("BrainAIBot initialized (auto-reply delay: %ds)", self.AUTO_REPLY_DELAY)

    def _get_brain(self, conversation_id: str) -> BrainAgent:
        """Return (or create) a BrainAgent for this conversation."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = BrainAgent(self.cfg)
            log.info("New session created for conversation %s", conversation_id)
        return self._sessions[conversation_id]

    # ── Start / stop the background monitor ─────────────────────────

    def start_monitor(self):
        """Start the background loop that checks for unanswered messages."""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.ensure_future(self._monitor_unanswered())
            log.info("Unanswered-message monitor started")

    async def _monitor_unanswered(self):
        """Background loop: every 60s, check for messages past the auto-reply window."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired = [
                (mid, pm) for mid, pm in list(self._pending.items())
                if not pm.answered and (now - pm.timestamp) >= self.AUTO_REPLY_DELAY
            ]
            for msg_id, pm in expired:
                pm.answered = True
                log.info("Auto-replying to unanswered message %s (%.0fs old)",
                         msg_id, now - pm.timestamp)
                try:
                    await self._proactive_reply(pm)
                except Exception as e:
                    log.error("Auto-reply failed for %s: %s", msg_id, e, exc_info=True)

            # Clean up old entries (>1 hour)
            stale = [mid for mid, pm in self._pending.items()
                     if (now - pm.timestamp) > 3600]
            for mid in stale:
                del self._pending[mid]

    async def _proactive_reply(self, pm: _PendingMessage):
        """Send a BrainAgent reply as a proactive message to the channel."""
        if not self._adapter:
            log.warning("No adapter set — cannot send proactive message")
            return

        brain = self._get_brain(pm.conv_ref.conversation.id)
        result = brain.chat(pm.text)
        agent_name = result["agent"]
        response = result["response"]
        emoji = self.AGENT_EMOJI.get(agent_name, "🧠")

        reply_text = (
            f"💡 *No one responded in {self.AUTO_REPLY_DELAY // 60} minutes "
            f"— here's what I found:*\n\n"
            f"{emoji} **{agent_name.capitalize()} Agent**\n\n{response}"
        )
        chunks = self._split_message(reply_text)

        async def callback(turn_context: TurnContext):
            for chunk in chunks:
                await turn_context.send_activity(chunk)

        await self._adapter.continue_conversation(
            pm.conv_ref, callback, self.cfg.get("teams_bot", {}).get("app_id", "")
        )

    # ── Bot Framework event handlers ────────────────────────────────

    async def on_members_added_activity(self, members_added, turn_context: TurnContext):
        """Send a welcome message when the bot is added to a chat."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                welcome = (
                    "👋 **Hello! I'm BCDR DeveloperAI** — your Azure Backup Management assistant.\n\n"
                    "I can help you with:\n"
                    "- 📚 **Project questions** — architecture, features, docs\n"
                    "- 🔍 **Debugging** — error analysis, KQL queries, Kusto\n"
                    "- 💻 **Code tracing** — source code flows, root cause analysis\n"
                    "- 📝 **Doc updates** — correct docs & create PRs automatically\n\n"
                    "**In channels:** @mention me to ask a question. "
                    "If nobody responds to a question within 10 minutes, "
                    "I'll try to help automatically.\n\n"
                    "**Commands:** `help`, `clear`, `agents`\n\n"
                    "Just type your question to get started!"
                )
                await turn_context.send_activity(welcome)

    async def on_message_activity(self, turn_context: TurnContext):
        """Handle an incoming user message."""
        activity = turn_context.activity
        text = (activity.text or "").strip()
        if not text:
            return

        conversation_id = activity.conversation.id
        is_channel = self._is_channel_message(activity)

        # ── Channel message handling ────────────────────────────────
        if is_channel:
            # Check if this is a human reply to a tracked pending message
            if activity.reply_to_id and activity.reply_to_id in self._pending:
                # A human replied — mark as answered so bot doesn't auto-reply
                self._pending[activity.reply_to_id].answered = True
                log.info("Message %s answered by human, cancelling auto-reply",
                         activity.reply_to_id)

            # Check if the bot is @mentioned
            is_mentioned = self._is_bot_mentioned(activity)
            if not is_mentioned:
                # Not mentioned — track the message for potential auto-reply
                clean_text = text
                if clean_text and not clean_text.startswith("/"):
                    conv_ref = TurnContext.get_conversation_reference(activity)
                    self._pending[activity.id] = _PendingMessage(clean_text, conv_ref)
                    log.info("Tracking channel message %s for auto-reply", activity.id)
                return  # Don't respond yet — wait for @mention or timeout

            # Bot is @mentioned — respond immediately
            text = self._strip_mention(activity, text)

        if not text:
            return

        brain = self._get_brain(conversation_id)

        # Handle simple commands
        cmd = text.lower()
        if cmd in ("help", "/help"):
            await self.on_members_added_activity(
                [activity.from_property], turn_context
            )
            return
        if cmd in ("clear", "/clear"):
            brain.reset_conversation()
            await turn_context.send_activity("🗑️ Conversation history cleared.")
            return
        if cmd in ("agents", "/agents"):
            agent_list = ", ".join(brain._agents.keys())
            await turn_context.send_activity(f"Available agents: {agent_list}")
            return

        # Show typing indicator while processing
        await turn_context.send_activity(Activity(type=ActivityTypes.typing))

        # Dispatch to BrainAgent
        try:
            result = brain.chat(text)
            agent_name = result["agent"]
            response = result["response"]

            emoji = self.AGENT_EMOJI.get(agent_name, "🧠")
            header = f"{emoji} **{agent_name.capitalize()} Agent**\n\n"

            # Mark this as answered if it was a tracked channel message
            if is_channel and activity.reply_to_id and activity.reply_to_id in self._pending:
                self._pending[activity.reply_to_id].answered = True

            full_response = header + response
            chunks = self._split_message(full_response)
            for chunk in chunks:
                await turn_context.send_activity(chunk)

        except Exception as e:
            log.error("Bot processing error: %s", e, exc_info=True)
            await turn_context.send_activity(
                f"❌ Sorry, something went wrong: {e}\n\nPlease try again or type `clear` to reset."
            )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_channel_message(activity: Activity) -> bool:
        """Check if the message is from a Teams channel (not 1:1 or group chat)."""
        conv = activity.conversation
        if conv and hasattr(conv, "conversation_type"):
            return getattr(conv, "conversation_type", None) == "channel"
        # Also check channel_data for Teams
        channel_data = activity.channel_data or {}
        return "teamsChannelId" in channel_data

    @staticmethod
    def _is_bot_mentioned(activity: Activity) -> bool:
        """Check if the bot is @mentioned in the message."""
        if not activity.entities:
            return False
        bot_id = activity.recipient.id if activity.recipient else None
        for entity in activity.entities:
            if entity.type == "mention":
                mentioned = getattr(entity, "mentioned", None)
                if mentioned and getattr(mentioned, "id", None) == bot_id:
                    return True
        return False

    @staticmethod
    def _strip_mention(activity: Activity, text: str) -> str:
        """Remove the bot's @mention from the message text."""
        if activity.entities:
            for entity in activity.entities:
                if entity.type == "mention":
                    mention_text = getattr(entity, "text", None)
                    if mention_text:
                        text = text.replace(mention_text, "").strip()
        return text

    @staticmethod
    def _split_message(text: str, max_len: int = 25000) -> list[str]:
        """Split a long message into chunks that fit Teams' 28KB limit."""
        if len(text) <= max_len:
            return [text]
        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            split_at = text.rfind("\n\n", 0, max_len)
            if split_at == -1:
                split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks
