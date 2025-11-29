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
    function_tool,
    RunContext,
)
from livekit.plugins import silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# -----------------------------
# Day 9 – ACP-style E-commerce Agent
# -----------------------------

ORDERS_FILE = Path("orders_day9.json")

# Small in-memory catalog (ACP-style: structured objects)
PRODUCTS: list[dict] = [
    {
        "id": "mug-001",
        "name": "Stoneware Coffee Mug",
        "description": "Matte finish stoneware mug with a wide handle.",
        "price": 799,
        "currency": "INR",
        "category": "mug",
        "color": "white",
        "sizes": [],
    },
    {
        "id": "mug-002",
        "name": "Travel Coffee Tumbler",
        "description": "Insulated stainless steel tumbler with lid.",
        "price": 1199,
        "currency": "INR",
        "category": "mug",
        "color": "black",
        "sizes": [],
    },
    {
        "id": "tee-001",
        "name": "Basic Cotton T-shirt",
        "description": "Unisex cotton t-shirt, regular fit.",
        "price": 699,
        "currency": "INR",
        "category": "tshirt",
        "color": "black",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "tee-002",
        "name": "Graphic Tee – Sunset",
        "description": "Soft tee with minimal sunset print.",
        "price": 999,
        "currency": "INR",
        "category": "tshirt",
        "color": "white",
        "sizes": ["S", "M", "L"],
    },
    {
        "id": "hoodie-001",
        "name": "Cozy Fleece Hoodie",
        "description": "Pullover hoodie with kangaroo pocket.",
        "price": 1599,
        "currency": "INR",
        "category": "hoodie",
        "color": "black",
        "sizes": ["M", "L", "XL"],
    },
    {
        "id": "hoodie-002",
        "name": "Zip-Up Hoodie",
        "description": "Lightweight zip hoodie for everyday wear.",
        "price": 1799,
        "currency": "INR",
        "category": "hoodie",
        "color": "blue",
        "sizes": ["S", "M", "L"],
    },
    {
        "id": "bottle-001",
        "name": "Stainless Steel Water Bottle",
        "description": "750ml insulated bottle, keeps drinks cold.",
        "price": 899,
        "currency": "INR",
        "category": "bottle",
        "color": "silver",
        "sizes": [],
    },
    {
        "id": "cap-001",
        "name": "Minimal Logo Cap",
        "description": "Adjustable cotton cap with small logo.",
        "price": 499,
        "currency": "INR",
        "category": "cap",
        "color": "black",
        "sizes": ["Free"],
    },
]

# Orders will be kept in memory for the session,
# and persisted to ORDERS_FILE as an ACP-style log.
ORDERS: list[dict] = []


def _load_orders_from_file() -> list[dict]:
    if not ORDERS_FILE.exists():
        return []
    try:
        with ORDERS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.warning(f"Failed to read {ORDERS_FILE}: {e}")
    return []


def _save_orders_to_file():
    try:
        with ORDERS_FILE.open("w", encoding="utf-8") as f:
            json.dump(ORDERS, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(ORDERS)} orders to {ORDERS_FILE}")
    except Exception as e:
        logger.error(f"Failed to write {ORDERS_FILE}: {e}")


# Initialize orders from file (if any)
ORDERS.extend(_load_orders_from_file())


def _build_catalog_block() -> str:
    """Human-readable snapshot for the system prompt."""
    lines: list[str] = ["CATALOG SNAPSHOT:"]
    for p in PRODUCTS:
        lines.append(
            f"- {p['id']}: {p['name']} "
            f"(₹{p['price']} {p['currency']}, category: {p['category']}, color: {p['color']})"
        )
    return "\n".join(lines)


BASE_INSTRUCTIONS = """
You are a voice-first E-COMMERCE SHOPPING ASSISTANT for a fictional brand called "VocalCart".

You follow ACP-style separation:
- Conversation layer: you (LLM + voice)
- Merchant layer: Python tools (catalog + orders)

--------------------------------
YOUR JOB
--------------------------------
- Understand what the user wants to buy.
- Use tools to:
    1) Browse the product catalog,
    2) Filter products by category, price, color, etc.,
    3) Create structured orders.
- Speak like a friendly shopping assistant (think Amazon / Flipkart style).
- Keep answers short and clear for voice.

--------------------------------
CATALOG (SOURCE OF TRUTH)
--------------------------------
You MUST NOT invent products.
You may only offer or sell items that exist in the catalog below:

{catalog_block}

Each product has:
- id
- name
- description
- price + currency
- category (mug, tshirt, hoodie, etc.)
- color
- optional sizes (S, M, L, etc.)

If a user asks for something not in the catalog:
- Suggest the closest match from the existing products instead of inventing new ones.

--------------------------------
TOOLS (MERCHANT LAYER)
--------------------------------
You have these tools:

1) list_products(filters: dict | None = None) -> list[product]
   - Use this to browse or filter products.
   - Filters may include:
        - "category" (e.g., "mug", "tshirt", "hoodie")
        - "max_price" (e.g., 1000)
        - "color" (e.g., "black", "blue")
        - "query" (free-text search in name/description)
   - Example queries you should handle by calling this tool:
        - "Show me all coffee mugs."
        - "Do you have t-shirts under 1000?"
        - "I'm looking for a black hoodie."
        - "Does this coffee mug come in blue?"
   - After calling:
        * Read out 2–4 relevant products with index numbers:
          "First, Stoneware Coffee Mug for ₹799..."
        * Let the user refer to them by index or name.

2) create_order(line_items: list[{{product_id, quantity}}]) -> order
   - Use this when the user decides to buy.
   - Example user lines:
        - "I'll buy the second hoodie you mentioned in size M."
        - "Add the black t-shirt and the silver bottle."
   - You must resolve:
        * Which product(s) they refer to (based on recent product listing or catalog).
        * Quantities (default to 1 if not specified).
   - Then call this tool with the structured line_items.
   - After the tool returns:
        * Read back the order summary (items + total + currency + order id).

3) get_last_order() -> order or null
   - Use when the user asks:
        * "What did I just buy?"
        * "Show my last order."
   - Read a short summary:
        * product names, quantities, total, currency, created_at.

--------------------------------
CONVERSATION STYLE
--------------------------------
- Start by introducing yourself:
    "Hi, I'm your VocalCart shopping assistant. I can help you browse our catalog and place an order."
- Ask what they are looking for:
    "What are you shopping for today?"
- For browsing:
    - Use list_products with appropriate filters instead of guessing.
    - Present 2–4 options with index numbers.
- For selecting:
    - Allow phrases like:
        * "I'll take the second one."
        * "Buy the black hoodie you mentioned."
- For ordering:
    - Once they're clear about what to buy, call create_order.
    - Confirm order details and total before closing.
- For last order:
    - Call get_last_order and summarize if available.

--------------------------------
IMPORTANT
--------------------------------
- Never mention tools, functions, JSON, or ACP by name.
- Think in terms of:
    "Browsing products" → list_products
    "Placing order" → create_order
    "Checking last purchase" → get_last_order
"""


def build_instructions() -> str:
    return BASE_INSTRUCTIONS.format(catalog_block=_build_catalog_block())


class EcommerceAgent(Agent):
    """
    Day 9 – ACP-inspired E-commerce Agent
    """

    def __init__(self) -> None:
        super().__init__(instructions=build_instructions())

    def _apply_filters(self, filters: dict | None) -> list[dict]:
        if not filters:
            return PRODUCTS

        results = PRODUCTS
        category = filters.get("category")
        max_price = filters.get("max_price")
        color = filters.get("color")
        query = filters.get("query")

        if category:
            category_l = str(category).lower()
            results = [p for p in results if p["category"].lower() == category_l]

        if max_price is not None:
            try:
                max_p = float(max_price)
                results = [p for p in results if float(p["price"]) <= max_p]
            except Exception:
                pass

        if color:
            color_l = str(color).lower()
            results = [p for p in results if p["color"].lower() == color_l]

        if query:
            q = str(query).lower()
            def matches(p: dict) -> bool:
                text = (p["name"] + " " + p["description"] + " " + p["category"]).lower()
                return q in text
            results = [p for p in results if matches(p)]

        return results

    @function_tool
    async def list_products(
        self,
        context: RunContext,
        filters: dict | None = None,
    ) -> dict:
        """
        List products from the catalog with optional filters.

        filters may include:
          - "category": e.g., "mug", "tshirt", "hoodie"
          - "max_price": e.g., 1000
          - "color": e.g., "black"
          - "query": free-text search in name/description/category

        Returns a dict:
          {
            "products": [...],
            "count": <int>
          }
        """
        results = self._apply_filters(filters or {})
        logger.info(f"list_products called with filters={filters}, found {len(results)} items")
        return {
            "products": results,
            "count": len(results),
        }

    @function_tool
    async def create_order(
        self,
        context: RunContext,
        line_items: list[dict],
        currency: str = "INR",
    ) -> dict:
        """
        Create an order from given line_items.

        line_items example:
          [
            {"product_id": "hoodie-001", "quantity": 1},
            {"product_id": "mug-001", "quantity": 2}
          ]

        Behavior:
          - Looks up products by id.
          - Ignores invalid product_ids.
          - Computes total.
          - Generates order_id and created_at.
          - Appends to ORDERS and persists to JSON file.

        Returns:
          {
            "ok": bool,
            "order": {
              "id": "...",
              "items": [...],
              "total": ...,
              "currency": "...",
              "created_at": "..."
            } | null,
            "message": "..."
          }
        """
        if not line_items:
            return {
                "ok": False,
                "order": None,
                "message": "No line items provided.",
            }

        items: list[dict] = []
        total = 0.0

        for li in line_items:
            pid = li.get("product_id")
            qty = li.get("quantity", 1)
            if not pid:
                continue
            try:
                qty = int(qty)
            except Exception:
                qty = 1
            if qty <= 0:
                continue

            product = next((p for p in PRODUCTS if p["id"] == pid), None)
            if not product:
                logger.warning(f"Unknown product_id in line_items: {pid}")
                continue

            price = float(product["price"])
            line_total = price * qty
            total += line_total

            items.append(
                {
                    "product_id": product["id"],
                    "name": product["name"],
                    "quantity": qty,
                    "unit_price": price,
                    "line_total": line_total,
                }
            )

        if not items:
            return {
                "ok": False,
                "order": None,
                "message": "No valid products found in line items.",
            }

        created_at = datetime.now().isoformat()
        order_id = f"ORD-{int(datetime.now().timestamp())}"

        order = {
            "id": order_id,
            "items": items,
            "total": total,
            "currency": currency,
            "created_at": created_at,
        }

        ORDERS.append(order)
        _save_orders_to_file()
        logger.info(f"Created order {order_id} with {len(items)} items, total={total} {currency}")

        return {
            "ok": True,
            "order": order,
            "message": "Order created successfully.",
        }

    @function_tool
    async def get_last_order(self, context: RunContext) -> dict:
        """
        Return the most recent order created in this backend,
        or indicate that no orders exist.

        Returns:
          {
            "has_order": bool,
            "order": {...} | null
          }
        """
        if not ORDERS:
            return {
                "has_order": False,
                "order": None,
            }
        last = ORDERS[-1]
        return {
            "has_order": True,
            "order": last,
        }


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Connect to LiveKit room
    await ctx.connect()

    # Voice pipeline setup
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=google.beta.GeminiTTS(
            model="gemini-2.5-flash-preview-tts",
            voice_name="zephyr",  # valid voice
            instructions=(
                "Speak like a friendly Indian shopping assistant. "
                "Keep responses short, clear, and conversational."
            ),
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=False,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the EcommerceAgent session
    await session.start(
        agent=EcommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Proactive greeting
    await session.generate_reply(
        instructions=(
            "Introduce yourself as VocalCart's voice shopping assistant. "
            "Say that you can help browse products like mugs, t-shirts, hoodies, and bottles, "
            "and help place an order. Then ask the user: "
            "'What are you looking for today?'"
        )
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
