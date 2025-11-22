import logging
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

ORDERS_FILE = Path("orders.json")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly coffee shop barista.\n"
                "Your job is to take the user's coffee order using voice.\n\n"
                "Maintain an internal JSON order_state with keys:\n"
                "drinkType, size, milk, extras, name.\n"
                "Ask clarifying questions until all fields are filled.\n"
                "When everything is collected, call the submit_coffee_order tool.\n"
                "After that, speak a neat summary of the order.\n"
                "No emojis or special formatting."
            ),
        )

    @function_tool
    async def submit_coffee_order(
        self,
        context: RunContext,
        drinkType: str,
        size: str,
        milk: str,
        extras: list[str],
        name: str,
    ) -> str:
        logger.info(
            "submit_coffee_order called: "
            f"{drinkType=}, {size=}, {milk=}, {extras=}, {name=}"
        )

        order = {
            "drinkType": drinkType,
            "size": size,
            "milk": milk,
            "extras": extras or [],
            "name": name,
            "timestamp": datetime.now().isoformat(),
        }

        if ORDERS_FILE.exists():
            try:
                with ORDERS_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        else:
            data = []

        data.append(order)

        with ORDERS_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        extras_text = ", ".join(extras) if extras else "no extras"
        return (
            f"Order saved for {name}: {size} {drinkType} with {milk} and {extras_text}."
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info("Usage summary: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
