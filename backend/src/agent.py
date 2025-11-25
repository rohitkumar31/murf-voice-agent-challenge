import logging
import json
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
)
from livekit.plugins import silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# -----------------------------
# Day 4 – Teach-the-Tutor: content loading
# -----------------------------


def _load_course_content() -> list[dict]:
    """
    Load small course content JSON for the tutor.

    We try a couple of common paths:
      - <repo_root>/shared-data/day4_tutor_content.json
      - <backend_root>/shared-data/day4_tutor_content.json

    If nothing is found, we fall back to a small built-in default.
    """
    # backend/src/agent.py -> backend
    backend_dir = Path(__file__).resolve().parents[1]
    candidate_paths = [
        backend_dir.parent / "shared-data" / "day4_tutor_content.json",
        backend_dir / "shared-data" / "day4_tutor_content.json",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    logger.info(f"Loaded course content from {path}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to read course content from {path}: {e}")

    # Fallback: small inline default so agent still works
    logger.warning("Falling back to built-in course content.")
    return [
        {
            "id": "variables",
            "title": "Variables",
            "summary": "Variables store values in memory so you can reuse or change them later. Each variable has a name and holds a value, like a labelled box. You can assign numbers, text, or other data types to variables and then use those names instead of repeating the values everywhere.",
            "sample_question": "What is a variable and why is it useful in a program?",
        },
        {
            "id": "loops",
            "title": "Loops",
            "summary": "Loops let you repeat an action multiple times without copying the same code. A for loop usually runs a known number of times, while a while loop keeps running as long as a condition stays true.",
            "sample_question": "Explain the difference between a for loop and a while loop with an example.",
        },
        {
            "id": "conditions",
            "title": "Conditions",
            "summary": "Conditions let your code make decisions. Using if, elif, and else, the program can choose different paths based on whether an expression is true or false.",
            "sample_question": "What is an if-else statement and when would you use it?",
        },
    ]


COURSE_CONTENT: list[dict] = _load_course_content()


def _build_course_block() -> str:
    """
    Turn COURSE_CONTENT into a readable block that we can inject into the LLM
    system prompt so it always teaches only from this mini-course.
    """
    lines: list[str] = []
    for concept in COURSE_CONTENT:
        cid = concept.get("id", "")
        title = concept.get("title", "")
        summary = concept.get("summary", "")
        q = concept.get("sample_question", "")
        lines.append(
            f"- id: {cid}\n"
            f"  title: {title}\n"
            f"  summary: {summary}\n"
            f"  sample_question: {q}"
        )
    return "\n".join(lines)


BASE_INSTRUCTIONS = """
You are an ACTIVE RECALL COACH called "Teach-the-Tutor".

Your job is to help the learner understand basic programming concepts by:
1) Explaining them (learn mode),
2) Quizzing them (quiz mode),
3) Asking them to teach the concept back to you (teach_back mode) and
   giving gentle, qualitative feedback.

VERY IMPORTANT RULES:
- You ONLY teach from the small course content given below.
- All examples and questions must stay close to that content.
- You must always remember and respect the current MODE:
    * learn      → you explain
    * quiz       → you ask questions
    * teach_back → the user explains, you listen and then give feedback
- The user can switch mode at any time by saying things like:
    "learn mode", "quiz mode", "teach back", "switch to quiz", etc.
- When they switch modes:
    1. Briefly confirm the new mode.
    2. Continue in that new mode.

COURSE CONTENT (you must follow this closely and not invent new topics):
{course_block}

--------------------------------
MODE BEHAVIOR
--------------------------------

1) LEARN MODE
   - Explain ONE concept at a time using its "summary".
   - Use simple language and short explanations.
   - After explaining, ask a small check question like:
       "Does this make sense?"
       "Want a tiny example, or should we move to quiz or teach_back for this concept?"

2) QUIZ MODE
   - Use the "sample_question" for that concept as a base.
   - Ask ONE question at a time.
   - Keep questions short and focused on the concept.
   - When the user answers:
       * Say if the answer is roughly correct or what is missing.
       * Add 1–2 lines of correction or extra intuition.
       * Then either ask another question OR offer to switch to teach_back.

3) TEACH_BACK MODE
   - Say something like:
       "Now explain this concept to me in your own words, like you are teaching a friend."
   - Let the user speak for a while.
   - After they explain:
       * Briefly summarize what they said.
       * Give qualitative feedback with 1 of 3 levels:
           - "Strong understanding"
           - "Okay but needs a bit more clarity"
           - "Needs more work"
       * Point out 1–3 things they did well.
       * Point out 1–3 small improvements or missing pieces.
   - Then ask if they want:
       - another teach_back on the same concept,
       - to switch to quiz,
       - or to learn a new concept.

--------------------------------
MODE SWITCHING
--------------------------------
- If the user clearly asks to change mode (e.g., "quiz karo", "teach_back mode on",
  "ab learn mode"), then:
    * Confirm: "Okay, switching to QUIZ mode for <concept>."
    * Immediately behave according to that mode.

--------------------------------
CONCEPT CHOOSING
--------------------------------
- The user can say things like:
    "variables padhna hai", "loops sikhao", "conditions pe quiz karo".
- Map this to the closest concept id from:
    - variables
    - loops
    - conditions
- Always confirm:
    "Great, we'll work on <title> (id: <id>)."

--------------------------------
MASTERY (internal sense)
--------------------------------
- Internally, based on quiz answers and teach_back quality, think of the learner as:
    - BEGINNER
    - INTERMEDIATE
    - CONFIDENT
- You do NOT need to show a numeric score.
- Use this only to adjust difficulty of questions and explanations.

--------------------------------
STYLE
--------------------------------
- Friendly, encouraging, patient.
- Short, clear paragraphs (good for voice).
- Use explicit instructions like:
    "Now, answer this question:"
    "Now, explain in your own words:"
    "Say 'switch to quiz mode' if you want me to quiz you."

--------------------------------
SESSION START (VERY IMPORTANT)
--------------------------------
At the beginning of the conversation you MUST:
1) Briefly introduce yourself as an active recall coach.
2) Mention the three modes: learn, quiz, teach_back.
3) List available concepts by id and title (from the course content).
4) Ask the user:
     a) Which concept they want to start with.
     b) Which mode they want to start in.
"""


def build_instructions() -> str:
    """Inject the course content block into the base instructions."""
    course_block = _build_course_block()
    return BASE_INSTRUCTIONS.format(course_block=course_block)


class TeachTheTutor(Agent):
    """
    Day 4 – Teach-the-Tutor: Active Recall Coach
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(),
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Pehle room se connect karo
    await ctx.connect()

    # Voice pipeline setup (STT + LLM + TTS + turn detection)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=google.beta.GeminiTTS(
            model="gemini-2.5-flash-preview-tts",
            voice_name="Zephyr",
            instructions="Speak like a friendly programming tutor, clear and encouraging.",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,  # quota bachane ke liye
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

    # Start the Teach-the-Tutor session
    await session.start(
        agent=TeachTheTutor(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Proactive greeting: ask for concept + mode
    await session.generate_reply(
        instructions=(
            "Greet the learner warmly in one or two short sentences. "
            "Explain that you are an active recall coach with three modes: "
            "learn, quiz, and teach_back. "
            "Mention the available concepts with their ids and titles based on the "
            "course content. Then ask them: "
            "1) Which concept they want to start with, and "
            "2) Which mode they want to begin in. "
            "Keep it concise and friendly for a voice conversation."
        )
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
