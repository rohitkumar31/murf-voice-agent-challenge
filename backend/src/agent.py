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
# Day 6 – Fraud Alert Voice Agent
# -----------------------------

FRAUD_DB_FILE = Path("fraud_case.json")


def _load_fraud_case() -> dict:
    """
    Load a single fake fraud case from JSON file, or fallback to a built-in case.

    Structure example:
    {
      "userName": "Aarav",
      "securityIdentifier": "12345",
      "cardEnding": "4242",
      "transactionName": "ABC Online Store",
      "transactionAmount": "₹4,999.00",
      "transactionTime": "2025-11-20 14:35 IST",
      "transactionCategory": "e-commerce",
      "transactionLocation": "Mumbai, India",
      "transactionSource": "abcstore.example.com",
      "securityQuestion": "What is the name of your first school?",
      "securityAnswer": "Green Valley School",
      "status": "pending_review",
      "outcomeNote": ""
    }
    """
    if FRAUD_DB_FILE.exists():
        try:
            with FRAUD_DB_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                logger.info(f"Loaded fraud case from {FRAUD_DB_FILE}")
                return data
        except Exception as e:
            logger.warning(f"Failed to read fraud case from {FRAUD_DB_FILE}: {e}")

    logger.warning("Falling back to built-in fake fraud case.")
    return {
        "userName": "Aarav",
        "securityIdentifier": "12345",
        "cardEnding": "4242",
        "transactionName": "ABC Online Store",
        "transactionAmount": "₹4,999.00",
        "transactionTime": "2025-11-20 14:35 IST",
        "transactionCategory": "e-commerce",
        "transactionLocation": "Mumbai, India",
        "transactionSource": "abcstore.example.com",
        "securityQuestion": "What is the name of your first school?",
        "securityAnswer": "Green Valley School",
        "status": "pending_review",
        "outcomeNote": "",
    }


FRAUD_CASE: dict = _load_fraud_case()


def _build_case_block() -> str:
    """
    Turn the fraud case into a readable block for the system prompt,
    so the agent knows the exact fake details.
    """
    c = FRAUD_CASE
    lines = [
        f"Customer (fake) name: {c.get('userName', '')}",
        f"Security identifier (fake): {c.get('securityIdentifier', '')}",
        f"Masked card ending: **** {c.get('cardEnding', '')}",
        f"Suspicious merchant: {c.get('transactionName', '')}",
        f"Amount: {c.get('transactionAmount', '')}",
        f"Category: {c.get('transactionCategory', '')}",
        f"Transaction time: {c.get('transactionTime', '')}",
        f"Location (fake): {c.get('transactionLocation', '')}",
        f"Source (website/app): {c.get('transactionSource', '')}",
        "",
        f"Security question (fake): {c.get('securityQuestion', '')}",
        f"Correct security answer (fake, internal only): {c.get('securityAnswer', '')}",
        f"Current status: {c.get('status', '')}",
    ]
    return "\n".join(lines)


BASE_INSTRUCTIONS = """
You are a FRAUD PREVENTION AGENT for a fictional bank called "SecureBank".

IMPORTANT:
- This is a DEMO / SANDBOX. All data is fake.
- Do NOT ask for real card numbers, PINs, passwords, OTPs, or any sensitive credentials.
- Only use the fake data and security question provided below.

-------------------------------
FAKE FRAUD CASE (INTERNAL DATA)
-------------------------------
The current fraud case you are investigating is:

{case_block}

Treat this as internal knowledge. You must NOT read the correct security answer directly;
you should only ASK the question and compare the user's answer logically.

-------------------------------
CALL GOAL
-------------------------------
When a fraud alert session starts:

1) Introduce yourself clearly as SecureBank's fraud prevention team.
   Example:
   "Hello, this is the SecureBank fraud prevention team. We detected a suspicious transaction
    on your card and would like to confirm a few details."

2) Explain that:
   - This is about a single suspicious transaction.
   - You will ask a basic security question (NON-sensitive).
   - You will then describe the transaction and ask if it was made by the customer.

3) Do NOT ask for:
   - Full card number
   - CVV
   - PIN
   - Passwords
   - OTP
   - Netbanking credentials
   Only use the fake security question provided.

-------------------------------
VERIFICATION FLOW
-------------------------------
You MUST perform a basic verification before discussing transaction details:

a) Ask for the customer's FIRST NAME.
   - Treat it as correct if it matches the internal name approximately:
       "{user_name}"
     Small spelling differences are okay.

b) Ask the fake security question:
   "{security_question}"
   - Treat it as correct ONLY if the user's answer roughly matches the internal answer:
       "{security_answer}"

c) Allow at most 2 attempts at the security question.
   - If they fail twice or give obviously wrong answers:
       - Politely say you cannot proceed for security reasons.
       - Ask no more sensitive details.
       - Then call the `update_fraud_case` tool with:
           status = "verification_failed"
           outcome_note = short explanation, e.g.
                          "Verification failed; customer could not answer security question."
       - After the tool returns, briefly say goodbye and end.

-------------------------------
IF VERIFICATION PASSES
-------------------------------
1) Briefly confirm:
   - "Thank you, your verification is complete."

2) Then read out the suspicious transaction based on the fake case:
   - Merchant
   - Amount
   - Masked card ending (e.g. "ending with 4242")
   - Approximate time
   - Location / category

3) Ask clearly:
   - "Did you make this transaction?" or
   - "Was this purchase done by you?"

Interpret answers:
- If the user confirms → treat as YES (legitimate).
- If the user denies → treat as NO (fraudulent).
- If they are unsure, gently help them recall; if still unsure, treat as suspicious.

-------------------------------
UPDATING THE CASE
-------------------------------
At the END of the conversation, after you have:

1) Completed verification (pass or fail),
2) And, if verified, asked whether the transaction is legitimate,

You MUST:

- Summarize verbally what happened, for example:
    - "You confirmed that the transaction at ABC Online Store for ₹4,999.00 was legitimate.
       We will mark this as safe and keep your card active."
    - OR
      "You denied the transaction, so we will treat it as fraudulent. In this demo, we will
       block the card and raise a mock dispute on your behalf."

- Then call the `update_fraud_case` tool EXACTLY ONCE with:
    * status:
        - "confirmed_safe"    if user confirmed the transaction
        - "confirmed_fraud"   if user denied the transaction
        - "verification_failed" if verification did not pass
    * outcome_note: 1–3 sentence description of the outcome.

If you are in the verification_failed branch:
- You must still call `update_fraud_case` with status = "verification_failed".

After the tool returns:
- Say a short closing line and end the call.

-------------------------------
CONVERSATION STYLE
-------------------------------
- Tone: calm, professional, reassuring.
- Speak slowly and clearly (for voice).
- Ask one question at a time.
- If the user says things like "That's all", "I'm done", "Thanks", and you've already handled
  verification and the yes/no decision, you should move toward closing and updating the case.
- Do not mention tools, JSON, files, or internal structures.
- Always remember: this is a demo; all data is fake, for a fictional bank.
"""


def build_instructions() -> str:
    """Inject the fraud case block and key fields into the base instructions."""
    case_block = _build_case_block()
    return BASE_INSTRUCTIONS.format(
        case_block=case_block,
        user_name=FRAUD_CASE.get("userName", ""),
        security_question=FRAUD_CASE.get("securityQuestion", ""),
        security_answer=FRAUD_CASE.get("securityAnswer", ""),
    )


class FraudAgent(Agent):
    """
    Day 6 – Fraud Alert Voice Agent
    """

    def __init__(self) -> None:
        super().__init__(instructions=build_instructions())

    @function_tool
    async def update_fraud_case(
        self,
        context: RunContext,
        status: str,
        outcome_note: str,
    ) -> str:
        """
        Update the fraud case status and outcome note in the local JSON "database".

        Args:
            status: One of "confirmed_safe", "confirmed_fraud", "verification_failed",
                    or any other short status string describing the result.
            outcome_note: 1–3 sentence description of what happened in the call.

        Behavior:
            - Loads the current fraud_case (from file if exists, otherwise fallback).
            - Updates the "status" and "outcomeNote" fields.
            - Adds/updates a "lastUpdated" timestamp.
            - Writes the updated case back to fraud_case.json.
        """
        logger.info(
            "update_fraud_case called with: status=%r, outcome_note=%r",
            status,
            outcome_note,
        )

        # Start from existing case (from disk if possible)
        if FRAUD_DB_FILE.exists():
            try:
                with FRAUD_DB_FILE.open("r", encoding="utf-8") as f:
                    current = json.load(f)
                if not isinstance(current, dict):
                    current = dict(FRAUD_CASE)
            except Exception as e:
                logger.warning(f"Failed to read {FRAUD_DB_FILE} in update_fraud_case: {e}")
                current = dict(FRAUD_CASE)
        else:
            current = dict(FRAUD_CASE)

        current["status"] = status
        current["outcomeNote"] = outcome_note
        current["lastUpdated"] = datetime.now().isoformat()

        try:
            with FRAUD_DB_FILE.open("w", encoding="utf-8") as f:
                json.dump(current, f, indent=2, ensure_ascii=False)
            logger.info(f"Fraud case updated and saved to {FRAUD_DB_FILE}")
        except Exception as e:
            logger.error(f"Failed to write {FRAUD_DB_FILE}: {e}")
            return (
                "I tried to update the fraud case, but something went wrong on my side. "
                "Please ensure the outcome is recorded manually."
            )

        return "Fraud case updated successfully."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Connect to room first
    await ctx.connect()

    # Voice pipeline setup (STT + LLM + TTS + turn detection)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=google.beta.GeminiTTS(
            model="gemini-2.5-flash-preview-tts",
            voice_name="Zephyr",
            instructions=(
                "Speak like a calm, professional Indian bank representative from the fraud prevention team. "
                "Be clear, reassuring, and concise."
            ),
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,  # reduce TTS calls to help with quota
    )

    # Metrics collection (optional)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the FraudAgent session
    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Proactive opening message
    await session.generate_reply(
        instructions=(
            "Introduce yourself as SecureBank's fraud prevention team. "
            "Explain in 1–2 short sentences that there is a suspicious card transaction "
            "and you need to verify a few details. "
            "Then ask politely for the customer's first name to begin verification."
        )
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
