from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
from pathlib import Path

CATALOG = [
    {
        "id": "mug-001",
        "name": "Stoneware Coffee Mug",
        "price": 799,
        "currency": "INR",
        "category": "mug",
        "color": "white",
    },
    {
        "id": "tee-001",
        "name": "Basic Cotton T-shirt",
        "price": 699,
        "currency": "INR",
        "category": "tshirt",
        "color": "black",
    },
    {
        "id": "hoodie-001",
        "name": "Cozy Fleece Hoodie",
        "price": 1599,
        "currency": "INR",
        "category": "hoodie",
        "color": "black",
    },
]

ORDERS_FILE = Path("orders_day9.json")

def load_orders():
    if not ORDERS_FILE.exists():
        return []
    try:
        with ORDERS_FILE.open("r") as f:
            return json.load(f)
    except:
        return []

def save_orders(data):
    with ORDERS_FILE.open("w") as f:
        json.dump(data, f, indent=2)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/acp/catalog")
def get_catalog():
    return {"products": CATALOG}

@app.post("/acp/orders")
def create_order(payload: dict):
    """
    payload format:
    {
      "items": [
         { "product_id": "hoodie-001", "quantity": 1 }
      ]
    }
    """
    items = payload.get("items", [])
    orders = load_orders()

    processed_items = []
    total = 0

    for it in items:
        pid = it["product_id"]
        qty = int(it.get("quantity", 1))
        product = next((p for p in CATALOG if p["id"] == pid), None)
        if product:
            line_total = product["price"] * qty
            total += line_total
            processed_items.append({
                "product_id": pid,
                "name": product["name"],
                "quantity": qty,
                "unit_price": product["price"],
                "line_total": line_total,
            })

    order_obj = {
        "id": f"ORD-{int(datetime.now().timestamp())}",
        "items": processed_items,
        "total": total,
        "currency": "INR",
        "created_at": datetime.now().isoformat(),
    }

    orders.append(order_obj)
    save_orders(orders)

    return {"ok": True, "order": order_obj}
