"""
Bot Framework adapter configuration.

Creates the adapter with error handling and authentication
for both local development and Azure deployment.
"""

import logging
import traceback

from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
)

log = logging.getLogger(__name__)


def create_adapter(app_id: str = "", app_password: str = "") -> BotFrameworkAdapter:
    """
    Create a configured Bot Framework adapter.

    For local development, leave app_id and app_password empty.
    For Azure deployment, provide the Bot registration credentials.
    """
    settings = BotFrameworkAdapterSettings(
        app_id=app_id,
        app_password=app_password,
    )
    adapter = BotFrameworkAdapter(settings)

    async def on_error(context: TurnContext, error: Exception):
        log.error("Unhandled bot error: %s", error, exc_info=True)
        traceback.print_exc()
        await context.send_activity(
            "❌ An internal error occurred. Please try again or type `clear` to reset."
        )

    adapter.on_turn_error = on_error
    return adapter
