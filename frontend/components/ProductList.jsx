import { useEffect, useState } from "react";

export default function ProductList() {
  const [products, setProducts] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/acp/catalog")
      .then(res => res.json())
      .then(data => setProducts(data.products));
  }, []);

  const buyNow = async (productId) => {
    const payload = {
      items: [
        { product_id: productId, quantity: 1 }
      ]
    };

    const res = await fetch("http://localhost:8000/acp/orders", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    alert("Order Placed! ID: " + data.order.id);
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>🛒 Product Catalog</h2>
      <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
        {products.map(p => (
          <div key={p.id} style={{
            width: "200px",
            border: "1px solid #ddd",
            padding: "10px",
            borderRadius: "8px",
          }}>
            <h4>{p.name}</h4>
            <p>₹{p.price}</p>
            <p>Category: {p.category}</p>
            <button
              onClick={() => buyNow(p.id)}
              style={{
                width: "100%",
                padding: "8px",
                background: "#4CAF50",
                color: "white",
                border: "none",
                borderRadius: "5px",
              }}
            >
              Buy Now
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
