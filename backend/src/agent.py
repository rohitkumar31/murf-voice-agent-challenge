import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    MetricsCollectedEvent,
)
from livekit.plugins import silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# -----------------------------
# Day 8 – Voice Game Master
# -----------------------------

GM_SYSTEM_PROMPT = """
You are **Aldric the Eternal**, a dramatic and immersive **Fantasy Game Master**.

UNIVERSE:
- A dark medieval fantasy world named **Eryndor**
- Dragons, ruined kingdoms, lost magic, cursed forests, ancient runes
- Tone: cinematic, mysterious, adventurous

YOUR ROLE:
- You are the Game Master (GM)
- You **describe scenes visually**, with sound cues, atmosphere, tension
- You **never control the player**; you only guide the world around them
- Always end your turn with **“What do you do?”**

STORY RULES:
- Keep responses short, clear, and playable for voice (6–10 sentences)
- Track story continuity using conversation memory
- Remember:
    - Player choices
    - Items collected
    - NPCs met
    - Places visited
- Do NOT end the story too fast
- Create a mini-arc:
    - A discovery
    - A threat
    - A challenge
    - A turning point
- Keep the story highly interactive

START:
You must greet the player as the GM and set the opening scene:
Something like…

“The torch flickers. The ancient stones tremble. A rumble echoes beneath the old fortress… What do you do?”

Do NOT ask for character creation unless the user requests it.
"""

class GameMasterAgent(Agent):
    def __init__(self):
        super().__init__(instructions=GM_SYSTEM_PROMPT)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts = google.beta.GeminiTTS(
    model="gemini-2.5-flash-preview-tts",
    voice_name="zephyr",  # valid voice name
    instructions="Speak like a dramatic fantasy narrator. Deep, atmospheric, and immersive.",
),

        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def log_usage():
        logger.info(usage.get_summary())

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    # Start game
    await session.generate_reply(
        instructions="Begin the adventure with a dramatic opening scene and ask: 'What do you do?'"
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
