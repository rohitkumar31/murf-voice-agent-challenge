import logging
import json
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

WELLNESS_LOG_FILE = Path("wellness_log.json")


class WellnessCompanion(Agent):
    """
    Day 3 – Health & Wellness Voice Companion

    - Supportive, non-clinical daily check-in agent
    - Asks about mood, energy, stress, and simple goals
    - Stores each check-in in a JSON file (wellness_log.json)
    - Uses past data to lightly reference previous check-ins
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a calm, supportive health and wellness voice companion.\n"
                "You are NOT a doctor, therapist, or clinician.\n"
                "You must NOT give medical, diagnostic, or treatment advice.\n\n"
                "Your main job is to do a short daily check-in with the user.\n"
                "Follow this structure:\n\n"
                "1) If possible, briefly recall previous check-ins.\n"
                "   - You may call the `get_last_checkin` tool at the start of a session\n"
                "     to see the previous entry (if any).\n"
                "   - Use it to say one small, grounded reference like:\n"
                "     'Last time you mentioned feeling low energy. How is it today?'\n\n"
                "2) Ask about MOOD and ENERGY:\n"
                "   - Examples: 'How are you feeling today?',\n"
                "               'What is your energy like right now?',\n"
                "               'Is anything stressing you out at the moment?'\n"
                "   - Let them describe things in their own words.\n"
                "   - Keep your tone gentle, non-judgmental, and grounded.\n"
                "   - Do NOT label, diagnose, or mention disorders or illnesses.\n\n"
                "3) Ask about INTENTIONS / GOALS for today:\n"
                "   - Ask for 1–3 simple things they want to do today.\n"
                "   - Include both tasks and self-care if possible.\n"
                "   - Examples: 'What are 1–3 things you'd like to get done today?',\n"
                "               'Is there anything you want to do just for yourself,\n"
                "                like rest, exercise, or a hobby?'\n\n"
                "4) Offer SMALL, REALISTIC, NON-MEDICAL suggestions:\n"
                "   - Break big goals into smaller steps.\n"
                "   - Encourage short breaks or simple actions\n"
                "     (e.g., a 5-minute walk, a glass of water, stretching, breathing).\n"
                "   - Never mention treatment, prescriptions, diagnosis, or illness names.\n\n"
                "5) Close with a brief RECAP and CONFIRMATION:\n"
                "   - Summarize today's mood in 1 short phrase.\n"
                "   - List the main 1–3 goals in a clear sentence.\n"
                "   - Example: 'So today you are feeling a bit tired but motivated,\n"
                "              and your goals are X, Y, and one small self-care step Z.\n"
                "              Does that sound right?'\n\n"
                "6) IMPORTANT: Logging the check-in\n"
                "   - Once the conversation feels complete and you have:\n"
                "       * mood (user's own words or a short phrase),\n"
                "       * a basic sense of energy level,\n"
                "       * a list of 1–3 goals/intentions,\n"
                "       * and you have spoken a recap,\n"
                "     then you MUST call the `log_wellness_checkin` tool exactly once.\n"
                "   - Pass the mood, energy, goals (as a list of short strings),\n"
                "     and a short 1–2 sentence summary of the check-in.\n"
                "   - After the tool returns, you may say a brief closing line of support.\n\n"
                "General style:\n"
                "- Keep responses short, conversational, and grounded.\n"
                "- No emojis or special formatting.\n"
                "- Do NOT pretend to be a medical professional.\n"
                "- Focus on listening, reflecting, and suggesting small, gentle actions.\n"
            ),
        )

    @function_tool
    async def get_last_checkin(self, context: RunContext) -> dict:
        """
        Fetch the most recent wellness check-in from wellness_log.json.

        Use this at the start of a new conversation to lightly reference
        how the user was doing last time.

        Returns:
            A dict with the latest entry, or a dict with `available=False`
            if no log exists yet.
        """
        if not WELLNESS_LOG_FILE.exists():
            return {"available": False, "message": "No previous check-ins found."}

        try:
            with WELLNESS_LOG_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read wellness_log.json: {e}")
            return {"available": False, "message": "Could not read previous entries."}

        if not isinstance(data, list) or len(data) == 0:
            return {"available": False, "message": "No previous check-ins found."}

        last_entry = data[-1]
        last_entry["available"] = True
        return last_entry

    @function_tool
    async def log_wellness_checkin(
        self,
        context: RunContext,
        mood: str,
        energy: str,
        goals: list[str],
        summary: str,
    ) -> str:
        """
        Save a wellness check-in to a JSON file.

        Args:
            mood: User's self-reported mood in their own words or a short phrase.
            energy: Short description of energy level (e.g., 'low', 'okay', 'high').
            goals: List of 1–3 small goals or intentions for the day.
            summary: A short 1–2 sentence summary of the overall check-in.

        Behavior:
            - Appends an entry to wellness_log.json with timestamp and given fields.
            - Keeps a human-readable, consistent schema.
        """
        logger.info(
            "log_wellness_checkin called with: "
            f"mood={mood!r}, energy={energy!r}, goals={goals!r}, summary={summary!r}"
        )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "energy": energy,
            "goals": goals or [],
            "summary": summary,
        }

        if WELLNESS_LOG_FILE.exists():
            try:
                with WELLNESS_LOG_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception as e:
                logger.warning(f"Failed to read wellness_log.json: {e}")
                data = []
        else:
            data = []

        data.append(entry)

        try:
            with WELLNESS_LOG_FILE.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Wellness check-in saved to wellness_log.json")
        except Exception as e:
            logger.error(f"Failed to write wellness_log.json: {e}")
            return (
                "I tried to log this check-in, but something went wrong on my side. "
                "You might want to manually note this somewhere for today."
            )

        return "Your check-in has been saved successfully."



def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice pipeline setup (STT + LLM + TTS + turn detection)
       # Voice pipeline setup (STT + LLM + TTS + turn detection)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=google.beta.GeminiTTS(
            model="gemini-2.5-flash-preview-tts",
            voice_name="Zephyr",
            instructions="Speak in a calm, friendly, supportive tone.",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )






    # Metrics collection (optional but useful)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the wellness companion session
    await session.start(
        agent=WellnessCompanion(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the room / user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
