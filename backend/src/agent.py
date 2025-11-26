import logging
import json
from pathlib import Path
from datetime import datetime

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
    function_tool,
    RunContext,
)
from livekit.plugins import silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# -----------------------------
# Day 5 – SDR for an Indian startup (Razorpay-style)
# -----------------------------

# Lead log file (simple JSON array of leads)
LEADS_LOG_FILE = Path("sdr_leads_log.json")


def _load_company_content() -> dict:
    """
    Load basic company info + FAQ from a JSON file if available.

    Expected file path (any one of these):
      - <repo_root>/shared-data/day5_sdr_content.json
      - <backend_root>/shared-data/day5_sdr_content.json

    If file is missing, we fall back to built-in Razorpay-style content.
    """
    backend_dir = Path(__file__).resolve().parents[1]
    candidate_paths = [
        backend_dir.parent / "shared-data" / "day5_sdr_content.json",
        backend_dir / "shared-data" / "day5_sdr_content.json",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"Loaded SDR company content from {path}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to read SDR company content from {path}: {e}")

    logger.warning("Falling back to built-in SDR company content.")

    # Built-in sample for an Indian fintech SaaS (Razorpay-style)
    return {
        "company_name": "Razorpay",
        "tagline": "Accept online payments and simplify business banking for Indian businesses.",
        "website": "https://razorpay.com",
        "product_summary": (
            "Razorpay is a full-stack payments and business banking platform for Indian businesses. "
            "You can accept payments online via UPI, cards, netbanking, and wallets; "
            "set up subscription billing; automate payouts; and manage your business finances in one place."
        ),
        "ideal_customers": (
            "Startups, D2C brands, SaaS companies, online marketplaces, and any Indian business "
            "that wants to accept online payments or automate payouts."
        ),
        "pricing_basics": (
            "Razorpay typically charges per-transaction fees for the payment gateway with no setup fees "
            "for standard plans. For enterprise pricing or latest offers, merchants usually contact the sales team."
        ),
        "faqs": [
            {
                "question": "What does your product do?",
                "answer": (
                    "We provide a full-stack payments and business banking platform. "
                    "You can accept online payments via UPI, cards, netbanking, and wallets, "
                    "set up subscription billing, automate payouts, and manage business finances in one place."
                ),
            },
            {
                "question": "Who is this for?",
                "answer": (
                    "We are built for Indian businesses of all sizes — from early-stage startups and D2C brands, "
                    "to large marketplaces and SaaS companies that need reliable payments and payouts."
                ),
            },
            {
                "question": "Do you have a free tier?",
                "answer": (
                    "For the payment gateway, there is generally no setup cost or monthly fee for standard plans — "
                    "you pay per successful transaction. Some advanced products may have custom or enterprise pricing."
                ),
            },
            {
                "question": "What are your pricing basics?",
                "answer": (
                    "Pricing is usually a simple per-transaction fee for the payment gateway, and custom pricing "
                    "for advanced products. For exact, up-to-date pricing, it's best to connect with our sales team."
                ),
            },
            {
                "question": "How do I get started?",
                "answer": (
                    "You can sign up online with your business details, complete basic verification, and integrate "
                    "our APIs, plugins, or no-code payment pages. Our team can guide you if you need help choosing "
                    "the right product."
                ),
            },
            {
                "question": "Do you support international payments?",
                "answer": (
                    "Yes, many merchants use us to accept international payments, subject to eligibility and "
                    "compliance checks. The sales team can confirm the best setup for your use case."
                ),
            },
        ],
    }


COMPANY_CONTENT: dict = _load_company_content()


def _build_faq_block() -> str:
    """
    Convert company content into a plain-text block we can embed
    into the system prompt so the agent can answer from it.
    """
    name = COMPANY_CONTENT.get("company_name", "Our company")
    tagline = COMPANY_CONTENT.get("tagline", "")
    website = COMPANY_CONTENT.get("website", "")
    product_summary = COMPANY_CONTENT.get("product_summary", "")
    ideal_customers = COMPANY_CONTENT.get("ideal_customers", "")
    pricing_basics = COMPANY_CONTENT.get("pricing_basics", "")
    faqs = COMPANY_CONTENT.get("faqs", [])

    lines: list[str] = []
    lines.append(f"COMPANY NAME: {name}")
    if tagline:
        lines.append(f"TAGLINE: {tagline}")
    if website:
        lines.append(f"WEBSITE: {website}")
    if product_summary:
        lines.append(f"PRODUCT SUMMARY: {product_summary}")
    if ideal_customers:
        lines.append(f"IDEAL CUSTOMERS: {ideal_customers}")
    if pricing_basics:
        lines.append(f"PRICING BASICS: {pricing_basics}")

    lines.append("\nFAQ ENTRIES:")
    for idx, item in enumerate(faqs, start=1):
        q = item.get("question", "")
        a = item.get("answer", "")
        lines.append(f"{idx}. Q: {q}\n   A: {a}")

    return "\n".join(lines)


BASE_INSTRUCTIONS = """
You are a friendly, focused SALES DEVELOPMENT REPRESENTATIVE (SDR) for the company described below.

Your job:
1) Greet visitors warmly.
2) Ask what brought them here and what they are working on.
3) Keep the conversation focused on understanding the user's needs.
4) Answer basic product / company / pricing questions from the FAQ content.
5) Politely and naturally collect LEAD details.
6) At the end, save a lead summary using the `log_lead` tool.

--------------------------------
COMPANY & FAQ (SOURCE OF TRUTH)
--------------------------------
You must treat the following company content as your ground truth.
Do NOT invent features, pricing, or policies beyond what is stated.
If you are unsure, say that details depend on the latest pricing and the sales team
can share exact numbers.

{faq_block}

--------------------------------
WHAT YOU CAN ANSWER
--------------------------------
From this content, you can answer questions like:
- "What does your product do?"
- "Who is this for?"
- "Do you have a free tier?"
- "How does pricing work?"
- "Do you support international payments?"
- "How do I get started?"

If they ask about something NOT covered in this content:
- Be honest: say you don't have that exact detail.
- Offer to connect them with the sales team and continue to collect lead info.

--------------------------------
LEAD CAPTURE – FIELDS TO COLLECT
--------------------------------
Over the course of the conversation, you want to gently collect:

- Name
- Company
- Email
- Role
- Use case (what they want to use this for)
- Team size
- Timeline (now / soon / later)

Do NOT interrogate them. Collect these fields naturally by:
- Asking follow-up questions when they describe their project.
- Saying things like:
    "Can I grab your work email so our team can follow up?"
    "What does your company do?"
    "Roughly how big is your team?"
    "When are you hoping to go live — now, soon, or a bit later?"

As you learn these details, keep them in mind. At the end of the call
you will use them to call the `log_lead` tool.

--------------------------------
CALL ENDING & SUMMARY
--------------------------------
Detect when the user is done (they might say:
  "That's all", "I'm done", "Thanks, this was helpful", etc.)

When you feel the conversation is wrapping up:

1) Give a short verbal summary:
   - Who they are (name, role, company)
   - What they want to use the product for (use case)
   - Rough team size and timeline.

2) Then call the `log_lead` tool EXACTLY ONCE with:
   - name
   - company
   - email
   - role
   - use_case
   - team_size
   - timeline
   - summary (1–3 sentence description of the lead and their needs)

If you are missing some fields, it's okay:
- Politely ask once more, e.g. "Before we wrap up, could I just grab your email?"
- If they still don't give it, call the tool anyway with what you have
  and leave missing fields as empty strings.

3) After the tool returns, say a short closing line:
   - Thank them for their time.
   - Mention that the team will follow up.

--------------------------------
CONVERSATION STYLE
--------------------------------
- Warm, concise, and business-casual.
- Ask one clear question at a time.
- Keep answers short for voice.
- Do NOT talk about being an AI or that you are reading from a file.
- Stay focused on:
    - Their business,
    - Their use case,
    - How this product can help,
    - Capturing lead info.

--------------------------------
IMPORTANT – TOOL USAGE
--------------------------------
You have access to the `log_lead` tool.

- Use it ONLY near the end of the conversation after you give a verbal summary.
- Call it exactly once per call.
- After calling it, you may say a brief goodbye and end the conversation.
"""


def build_instructions() -> str:
    """Inject the company FAQ block into the base instructions."""
    faq_block = _build_faq_block()
    return BASE_INSTRUCTIONS.format(faq_block=faq_block)


class SdrAgent(Agent):
    """
    Day 5 – SDR / FAQ / Lead Capture Agent
    """

    def __init__(self) -> None:
        super().__init__(instructions=build_instructions())

    @function_tool
    async def log_lead(
        self,
        context: RunContext,
        name: str = "",
        company: str = "",
        email: str = "",
        role: str = "",
        use_case: str = "",
        team_size: str = "",
        timeline: str = "",
        summary: str = "",
    ) -> str:
        """
        Save a qualified lead to a JSON log file.

        Call this exactly once near the end of the conversation, after:
          - You have given the user a short verbal summary of who they are
            and what they want,
          - You have collected as many of these fields as possible:

            - name
            - company
            - email
            - role
            - use_case
            - team_size
            - timeline (e.g. "now", "soon", "later")
            - summary (1–3 sentence description of their needs)

        Behavior:
          - Appends a JSON entry to sdr_leads_log.json with timestamp and fields.
          - If the file does not exist yet, it will be created.
        """
        logger.info(
            "log_lead called with: "
            f"name={name!r}, company={company!r}, email={email!r}, role={role!r}, "
            f"use_case={use_case!r}, team_size={team_size!r}, timeline={timeline!r}, "
            f"summary={summary!r}"
        )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "company": company,
            "email": email,
            "role": role,
            "use_case": use_case,
            "team_size": team_size,
            "timeline": timeline,
            "summary": summary,
        }

        # Load existing log
        if LEADS_LOG_FILE.exists():
            try:
                with LEADS_LOG_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception as e:
                logger.warning(f"Failed to read {LEADS_LOG_FILE}: {e}")
                data = []
        else:
            data = []

        data.append(entry)

        # Write back
        try:
            with LEADS_LOG_FILE.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Lead saved to {LEADS_LOG_FILE}")
        except Exception as e:
            logger.error(f"Failed to write {LEADS_LOG_FILE}: {e}")
            return (
                "I tried to save this lead, but something went wrong on my side. "
                "Please make sure the details are captured manually."
            )

        return "Lead saved successfully."


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
            instructions=(
                "Speak like a friendly Indian SDR: clear, concise, and helpful. "
                "Sound professional but warm."
            ),
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

    # Start the SDR agent session
    await session.start(
        agent=SdrAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Initial greeting and opening questions
    await session.generate_reply(
        instructions=(
            "Greet the visitor warmly as an SDR for the company. "
            "Briefly mention what the company does in one sentence, "
            "then ask: 'What brings you here today?' or "
            "'Tell me a bit about what you're working on.' "
            "Keep it short and friendly."
        )
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
