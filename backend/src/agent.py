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
# Day 7 – Food & Grocery Ordering Voice Agent
# -----------------------------

CATALOG_FILE = Path("shared-data/day7_catalog.json")
ORDERS_DIR = Path("orders")
ORDERS_DIR.mkdir(exist_ok=True)


def _load_catalog() -> list[dict]:
    """
    Load catalog from JSON file if present, otherwise fallback to built-in sample.
    Expected structure per item:
    {
      "id": "bread_whole_wheat",
      "name": "Whole Wheat Bread",
      "category": "groceries",
      "price": 45.0,
      "unit": "loaf",
      "brand": "HealthyBite",
      "tags": ["bread", "sandwich", "vegan"]
    }
    """
    # Try project-root/shared-data/day7_catalog.json OR backend/shared-data/day7_catalog.json
    backend_dir = Path(__file__).resolve().parents[1]
    candidate_paths = [
        backend_dir.parent / "shared-data" / "day7_catalog.json",
        backend_dir / "shared-data" / "day7_catalog.json",
    ]

    for path in candidate_paths:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    logger.info(f"Loaded catalog from {path}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to read catalog from {path}: {e}")

    logger.warning("Falling back to built-in catalog for Day 7.")
    return [
        {
            "id": "bread_whole_wheat",
            "name": "Whole Wheat Bread",
            "category": "groceries",
            "price": 45.0,
            "unit": "loaf",
            "brand": "HealthyBite",
            "tags": ["bread", "sandwich", "vegan"],
        },
        {
            "id": "bread_white",
            "name": "White Bread",
            "category": "groceries",
            "price": 40.0,
            "unit": "loaf",
            "brand": "SoftLoaf",
            "tags": ["bread", "sandwich"],
        },
        {
            "id": "eggs_12",
            "name": "Eggs (12 pcs)",
            "category": "groceries",
            "price": 75.0,
            "unit": "dozen",
            "brand": "FarmFresh",
            "tags": ["eggs", "protein"],
        },
        {
            "id": "milk_1l",
            "name": "Toned Milk 1L",
            "category": "groceries",
            "price": 60.0,
            "unit": "1L",
            "brand": "DailyDairy",
            "tags": ["milk"],
        },
        {
            "id": "peanut_butter",
            "name": "Peanut Butter (Crunchy)",
            "category": "snacks",
            "price": 180.0,
            "unit": "340g jar",
            "brand": "NuttySpread",
            "tags": ["peanut butter", "sandwich"],
        },
        {
            "id": "pasta_500g",
            "name": "Pasta 500g",
            "category": "groceries",
            "price": 90.0,
            "unit": "500g",
            "brand": "PastaBox",
            "tags": ["pasta"],
        },
        {
            "id": "pasta_sauce",
            "name": "Tomato Pasta Sauce",
            "category": "groceries",
            "price": 120.0,
            "unit": "420g jar",
            "brand": "SaucyChef",
            "tags": ["pasta", "sauce"],
        },
        {
            "id": "chips_masala",
            "name": "Masala Potato Chips",
            "category": "snacks",
            "price": 30.0,
            "unit": "70g pack",
            "brand": "CrunchMate",
            "tags": ["chips", "snacks"],
        },
        {
            "id": "cola_1_25l",
            "name": "Cola 1.25L",
            "category": "beverages",
            "price": 70.0,
            "unit": "1.25L bottle",
            "brand": "FizzUp",
            "tags": ["cold drink", "cola"],
        },
        {
            "id": "margherita_pizza",
            "name": "Margherita Pizza (Medium)",
            "category": "prepared_food",
            "price": 299.0,
            "unit": "1 pizza",
            "brand": "PizzaPal",
            "tags": ["pizza", "veg"],
        },
        {
            "id": "veg_sandwich",
            "name": "Veg Sandwich",
            "category": "prepared_food",
            "price": 120.0,
            "unit": "1 sandwich",
            "brand": "CaféBite",
            "tags": ["sandwich", "veg"],
        },
    ]


CATALOG: list[dict] = _load_catalog()

# Simple recipes mapping: dish → list of item IDs
RECIPES: dict[str, list[str]] = {
    "peanut butter sandwich": ["bread_whole_wheat", "peanut_butter"],
    "pb sandwich": ["bread_whole_wheat", "peanut_butter"],
    "pasta for two": ["pasta_500g", "pasta_sauce"],
    "simple pasta": ["pasta_500g", "pasta_sauce"],
}


def _build_catalog_block() -> str:
    """Builds a human-readable catalog block for instructions."""
    lines: list[str] = ["CATALOG ITEMS:"]
    for item in CATALOG:
        lines.append(
            f"- {item['id']}: {item['name']} "
            f"(category: {item['category']}, price: ₹{item['price']}, brand: {item.get('brand','')})"
        )
    lines.append("\nRECIPES (dish → items):")
    for dish, ids in RECIPES.items():
        names = [i["name"] for i in CATALOG if i["id"] in ids]
        lines.append(f"- {dish}: {', '.join(names)}")
    return "\n".join(lines)


BASE_INSTRUCTIONS = """
You are a friendly FOOD & GROCERY ORDERING ASSISTANT for a fictional store called "QuickCart".

Your job:
1) Greet the user and explain that you can help order groceries, snacks, and simple meal ingredients.
2) Understand what they want to order:
   - Specific items (e.g., "2 loaves of whole wheat bread")
   - Quantities
   - Higher-level requests like "ingredients for a peanut butter sandwich" or "pasta for two".
3) Manage a CART in memory by calling the cart tools.
4) When the user is done, confirm their order and place it by calling `place_order`.

--------------------------------
CATALOG (SOURCE OF TRUTH)
--------------------------------
You MUST only offer items that exist in this catalog:

{catalog_block}

- If a user asks for something that isn't available, politely suggest the closest match from the catalog.
- Feel free to ask small clarifying questions (brand, size, quantity) if ambiguous.

--------------------------------
CART MANAGEMENT (TOOLS)
--------------------------------
You have access to these tools:

1) add_item_to_cart(item_id: str, quantity: int, notes: str = "")
   - Use when the user requests to add an item.
   - item_id must be one of the IDs in the catalog (e.g., "bread_whole_wheat").
   - After calling, briefly confirm to the user what was added.

2) remove_item_from_cart(item_id: str)
   - Remove that item completely from the cart.
   - After calling, tell the user what was removed, or if it wasn't in the cart.

3) update_item_quantity(item_id: str, quantity: int)
   - Change the quantity for an existing item.
   - If quantity becomes 0, remove it.
   - Confirm the new quantity to the user.

4) list_cart()
   - Use when the user asks "what's in my cart", "show my order", etc.
   - Read out items and total value.

5) add_recipe_to_cart(dish: str, servings: int = 1)
   - For higher-level requests like "ingredients for a peanut butter sandwich" or "pasta for two":
   - Map the dish to the recipe list given in the instructions (internal mapping).
   - For example:
       * "peanut butter sandwich" → bread + peanut butter
       * "pasta for two" → pasta + pasta sauce
   - Add the relevant items to the cart and confirm to the user.
   - If the dish is unknown, apologize and ask them to order items directly.

6) place_order(customer_name: str, address: str, notes: str = "")
   - Use when the user says "place my order", "that's all", "I'm done", etc.
   - Before calling:
       * Make sure the cart is not empty.
       * Read a short summary of the final cart and estimated total.
       * Ask for any minimal details you want (e.g., name and address as free text).
   - After the tool returns successfully:
       * Tell the user that the order has been placed and saved.
       * You can mention a simple order ID from the tool response.

--------------------------------
RECIPES / DISH REQUESTS
--------------------------------
If the user says:
- "I need ingredients for a peanut butter sandwich."
- "Get me what I need for making pasta for two people."

You should:
1) Interpret which recipe is closest (e.g., "peanut butter sandwich", "pasta for two").
2) Call `add_recipe_to_cart` with that dish name and servings (2 for pasta for two, etc.).
3) Then tell the user exactly which items you added.

--------------------------------
CONVERSATION STYLE
--------------------------------
- Tone: warm, simple, and efficient — like a Swiggy Instamart / Blinkit style assistant.
- Ask one question at a time.
- Always confirm cart changes after using a cart tool.
- When unclear, ask clarifying questions instead of guessing.

--------------------------------
ORDER COMPLETION
--------------------------------
When the user indicates they are done ordering:
- Confirm cart contents using `list_cart`.
- Ask for their name and address in a simple way.
- Then call `place_order`.
- After placing the order, give a short summary with:
    - Number of items,
    - Order total,
    - A simple order ID (from the tool response).
- Then say a polite goodbye.
"""


def build_instructions() -> str:
    catalog_block = _build_catalog_block()
    return BASE_INSTRUCTIONS.format(catalog_block=catalog_block)


class GroceryAgent(Agent):
    """
    Day 7 – Food & Grocery Ordering Assistant
    """

    def __init__(self) -> None:
        super().__init__(instructions=build_instructions())
        # in-memory cart: list of {item_id, name, unit_price, quantity, notes}
        self.cart: list[dict] = []

    def _find_item_by_id(self, item_id: str) -> dict | None:
        for item in CATALOG:
            if item["id"] == item_id:
                return item
        return None

    def _cart_summary(self) -> dict:
        total = 0.0
        items_summary = []
        for entry in self.cart:
            line_total = entry["unit_price"] * entry["quantity"]
            total += line_total
            items_summary.append(
                {
                    "item_id": entry["item_id"],
                    "name": entry["name"],
                    "quantity": entry["quantity"],
                    "unit_price": entry["unit_price"],
                    "line_total": line_total,
                    "notes": entry.get("notes", ""),
                }
            )
        return {"items": items_summary, "total": total}

    @function_tool
    async def add_item_to_cart(
        self,
        context: RunContext,
        item_id: str,
        quantity: int = 1,
        notes: str = "",
    ) -> dict:
        """
        Add an item from the catalog to the cart.

        Args:
          item_id: Catalog item id (e.g., "bread_whole_wheat").
          quantity: How many units to add (default 1).
          notes: Optional free-text notes (e.g., "extra fresh", "if unavailable, skip").

        Returns:
          A dict with the current cart summary: {items: [...], total: float}.
        """
        if quantity <= 0:
            quantity = 1

        item = self._find_item_by_id(item_id)
        if not item:
            return {"ok": False, "message": f"Item id {item_id!r} not found in catalog."}

        # Check if already in cart
        for entry in self.cart:
            if entry["item_id"] == item_id:
                entry["quantity"] += quantity
                if notes:
                    entry["notes"] = (entry.get("notes", "") + " " + notes).strip()
                break
        else:
            self.cart.append(
                {
                    "item_id": item_id,
                    "name": item["name"],
                    "unit_price": float(item["price"]),
                    "quantity": quantity,
                    "notes": notes,
                }
            )

        summary = self._cart_summary()
        logger.info(f"Item {item_id} added to cart. Cart now: {summary}")
        return {"ok": True, "cart": summary}

    @function_tool
    async def remove_item_from_cart(
        self,
        context: RunContext,
        item_id: str,
    ) -> dict:
        """
        Remove an entire line item from the cart by its item_id.

        Returns:
          Dict with ok flag, and updated cart summary.
        """
        original_len = len(self.cart)
        self.cart = [entry for entry in self.cart if entry["item_id"] != item_id]
        removed = len(self.cart) < original_len
        summary = self._cart_summary()
        return {"ok": removed, "cart": summary}

    @function_tool
    async def update_item_quantity(
        self,
        context: RunContext,
        item_id: str,
        quantity: int,
    ) -> dict:
        """
        Update the quantity of an item in the cart.

        - If quantity <= 0, the item is removed.
        - Returns updated cart summary.
        """
        if quantity <= 0:
            self.cart = [entry for entry in self.cart if entry["item_id"] != item_id]
        else:
            for entry in self.cart:
                if entry["item_id"] == item_id:
                    entry["quantity"] = quantity
                    break

        summary = self._cart_summary()
        return {"ok": True, "cart": summary}

    @function_tool
    async def list_cart(self, context: RunContext) -> dict:
        """
        Return the current cart contents and total.
        """
        summary = self._cart_summary()
        return summary

    @function_tool
    async def add_recipe_to_cart(
        self,
        context: RunContext,
        dish: str,
        servings: int = 1,
    ) -> dict:
        """
        Add ingredients for a simple dish into the cart.

        dish:
          e.g., "peanut butter sandwich", "pasta for two"
        servings:
          Rough multiplier for quantity (e.g., 2 for 2 people).

        Behavior:
          - Looks up the dish in an internal recipes mapping.
          - Adds each mapped item to the cart.
        """
        key = dish.strip().lower()
        if key not in RECIPES:
            return {
                "ok": False,
                "message": f"No recipe found for {dish!r}.",
                "cart": self._cart_summary(),
            }

        multiplier = max(servings, 1)
        for item_id in RECIPES[key]:
            item = self._find_item_by_id(item_id)
            if not item:
                continue
            base_qty = 1
            qty = base_qty * multiplier
            # Reuse add_item_to_cart logic
            for entry in self.cart:
                if entry["item_id"] == item_id:
                    entry["quantity"] += qty
                    break
            else:
                self.cart.append(
                    {
                        "item_id": item_id,
                        "name": item["name"],
                        "unit_price": float(item["price"]),
                        "quantity": qty,
                        "notes": f"For dish: {dish}",
                    }
                )

        summary = self._cart_summary()
        return {
            "ok": True,
            "dish": dish,
            "cart": summary,
        }

    @function_tool
    async def place_order(
        self,
        context: RunContext,
        customer_name: str,
        address: str,
        notes: str = "",
    ) -> dict:
        """
        Place the current order by saving it to a JSON file.

        Args:
          customer_name: Name of the customer (free text).
          address: Delivery address (free text).
          notes: Optional notes or instructions.

        Behavior:
          - Uses the current cart to create an order object:
                {
                  "order_id": "...",
                  "timestamp": "...",
                  "customer_name": "...",
                  "address": "...",
                  "notes": "...",
                  "items": [...],
                  "total": ...
                }
          - Writes to orders/order_<timestamp>.json
          - Returns order_id and total.
        """
        summary = self._cart_summary()
        if not summary["items"]:
            return {
                "ok": False,
                "message": "Cart is empty, cannot place order.",
            }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        order_id = f"ORD_{timestamp}"
        order_obj = {
            "order_id": order_id,
            "timestamp": datetime.now().isoformat(),
            "customer_name": customer_name,
            "address": address,
            "notes": notes,
            "items": summary["items"],
            "total": summary["total"],
        }

        try:
            out_path = ORDERS_DIR / f"order_{timestamp}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(order_obj, f, indent=2, ensure_ascii=False)
            logger.info(f"Order {order_id} saved to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write order file: {e}")
            return {
                "ok": False,
                "message": "Failed to save order on server side.",
            }

        # Clear cart after placing order
        self.cart.clear()

        return {
            "ok": True,
            "order_id": order_id,
            "total": order_obj["total"],
        }


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Connect to room first
    await ctx.connect()

    # Voice pipeline setup
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=google.beta.GeminiTTS(
            model="gemini-2.5-flash-preview-tts",
            voice_name="Zephyr",
            instructions=(
                "Speak like a friendly Indian food & grocery assistant, "
                "short and clear, like Swiggy Instamart or Blinkit support."
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

    # Start the GroceryAgent session
    await session.start(
        agent=GroceryAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Initial greeting
    await session.generate_reply(
        instructions=(
            "Greet the user as QuickCart's voice assistant. "
            "Explain in one short sentence that you can help them order groceries, snacks, "
            "and simple meal ingredients. Then ask what they would like to order today."
        )
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

