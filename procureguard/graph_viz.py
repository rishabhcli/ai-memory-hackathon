"""
ProcureGuard Graph Visualization — extract focused subgraphs from Qdrant.

Builds a small D3-compatible {nodes, links} JSON for the knowledge graph panel.
"""

import json
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Entity type → colour mapping
TYPE_COLORS = {
    "Vendor": "#ff6b6b",
    "Invoice": "#ffd93d",
    "Transaction": "#4ecdc4",
    "Product": "#6bcb77",
    "LineItem": "#a5b4fc",
    "TotalAmount": "#f59e0b",
    "Date": "#94a3b8",
    "Quantity": "#c084fc",
    "default": "#7c3aed",
}


def _parse_text(payload: dict):
    text = payload.get("text", "")
    if isinstance(text, dict):
        return text
    if isinstance(text, str):
        try:
            return json.loads(text.replace("'", '"'))
        except Exception:
            try:
                return eval(text)
            except Exception:
                return {}
    return {}


def extract_subgraph(qdrant: QdrantClient, query: str, embed_fn, limit: int = 30) -> dict:
    """
    Build a focused subgraph from a query.

    Strategy:
    1. Search DocumentChunk_text for matching invoices/transactions
    2. Parse out vendor_id, products, amounts to build nodes
    3. Create relationship edges between them
    """
    vec = embed_fn(query)

    # Search for relevant document chunks
    results = qdrant.query_points(
        collection_name="DocumentChunk_text",
        query=vec,
        using="text",
        limit=limit,
        with_payload=True,
    )

    nodes = {}
    links = []

    for point in results.points:
        payload = point.payload or {}
        data = _parse_text(payload)
        pid = str(point.id)

        inv_num = data.get("invoice_number")
        tx_id = data.get("transaction_id")
        vendor_id = data.get("vendor_id")
        record_type = "Invoice" if inv_num else "Transaction" if tx_id else "Document"
        record_name = inv_num or tx_id or pid[:8]

        # Add the record node
        nodes[pid] = {
            "id": pid,
            "name": record_name,
            "type": record_type,
            "color": TYPE_COLORS.get(record_type, TYPE_COLORS["default"]),
            "amount": data.get("total") or data.get("amount"),
        }

        # Add vendor node + edge
        if vendor_id is not None:
            vid = f"vendor_{vendor_id}"
            if vid not in nodes:
                nodes[vid] = {
                    "id": vid,
                    "name": f"Vendor {vendor_id}",
                    "type": "Vendor",
                    "color": TYPE_COLORS["Vendor"],
                }
            rel = "issued_by" if inv_num else "paid_to"
            links.append({"source": pid, "target": vid, "relation": rel})

        # Add product nodes + edges from line items
        items_raw = data.get("items", [])
        if isinstance(items_raw, str):
            try:
                items_raw = json.loads(items_raw.replace("'", '"'))
            except Exception:
                try:
                    items_raw = eval(items_raw)
                except Exception:
                    items_raw = []
        if isinstance(items_raw, list):
            for item in items_raw[:5]:  # cap at 5 per record
                sku = item.get("sku", "")
                product_name = item.get("product", sku)
                prod_id = f"product_{sku}" if sku else f"product_{product_name}"
                if prod_id not in nodes:
                    nodes[prod_id] = {
                        "id": prod_id,
                        "name": product_name,
                        "type": "Product",
                        "color": TYPE_COLORS["Product"],
                    }
                links.append({"source": pid, "target": prod_id, "relation": "contains_item"})

        # Add total amount node
        total = data.get("total") or data.get("amount")
        if total is not None:
            total_id = f"total_{total}"
            if total_id not in nodes:
                nodes[total_id] = {
                    "id": total_id,
                    "name": f"${float(total):,.2f}",
                    "type": "TotalAmount",
                    "color": TYPE_COLORS["TotalAmount"],
                }
            links.append({"source": pid, "target": total_id, "relation": "has_total"})

    return {"nodes": list(nodes.values()), "links": links}
