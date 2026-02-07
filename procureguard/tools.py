"""
ProcureGuard Tools — 7 tools for the agent to call.

Each tool is an async function returning a structured dict.
Tools use Qdrant directly (via shared/ modules) and cognee for memory.
"""

import os
import sys
import json
import time
import statistics
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Prefetch,
    Fusion,
    FusionQuery,
    DiscoverQuery,
    DiscoverInput,
    ContextPair,
    RecommendQuery,
    RecommendInput,
    RecommendStrategy,
    QueryRequest,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.llm import get_llm_response, get_model_name
from shared.embeddings import get_embedding

# ── globals set by init_tools() ──────────────────────────────────────────────

qdrant: QdrantClient | None = None
_analytics_cache: dict = {}
_anomaly_cache: dict = {}
_pricing_cache: dict = {}
_records_cache: list = []

TOOL_DESCRIPTIONS = {
    "search_knowledge": {
        "description": "Search invoices, transactions, vendors, and products semantically. Uses Qdrant Prefetch + RRF Fusion for high-quality retrieval.",
        "params": "query (str, required), collection (str, default DocumentChunk_text), limit (int, default 10)",
    },
    "analyze_spend": {
        "description": "Get spend analytics: vendor totals, monthly trends, product breakdown, comparisons between vendors. Use for any financial analysis question.",
        "params": "vendor_id (int, optional — filter to one vendor), compare_vendors (comma-separated ids, optional)",
    },
    "detect_anomalies": {
        "description": "Check for anomalies: amount outliers, near-duplicates, embedding outliers, vendor variance. Use when asked about suspicious activity or discrepancies.",
        "params": "vendor_id (int, optional — filter to one vendor), anomaly_type (str, optional — amount_outlier|near_duplicate|embedding_outlier|vendor_variance)",
    },
    "investigate_record": {
        "description": "Deep-dive a specific record: find similar records via Qdrant Recommend API. Use for drill-down on a flagged invoice or transaction.",
        "params": "record_id (str, required — Qdrant point UUID)",
    },
    "recall_memory": {
        "description": "Search past investigations and findings stored in the knowledge graph via cognee. Use when referencing previous conversations or stored audit notes.",
        "params": "query (str, required)",
    },
    "store_finding": {
        "description": "Store an important audit finding or user note in the knowledge graph via cognee for future reference.",
        "params": "text (str, required — the finding to store)",
    },
    "ingest_document": {
        "description": "Ingest a new procurement document (invoice, transaction, vendor profile) through cognee's entity extraction pipeline into the knowledge graph.",
        "params": "raw_text (str, required — the raw document text)",
    },
    "analyze_pricing": {
        "description": "Get dynamic pricing intelligence: vendor price comparisons per product, price trends over time, overpriced items, and savings potential. Use for pricing questions, vendor cost comparisons, or optimization.",
        "params": "product_name (str, optional — filter to one product), vendor_id (int, optional — filter to one vendor)",
    },
}


def get_tool_descriptions_text() -> str:
    """Format tool descriptions for the SLM prompt."""
    lines = []
    for name, info in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name}: {info['description']}\n  Parameters: {info['params']}")
    return "\n".join(lines)


# ── initialisation ───────────────────────────────────────────────────────────

def init_tools(qdrant_client: QdrantClient):
    global qdrant
    qdrant = qdrant_client


def _parse_record(payload: dict):
    """Parse a Qdrant payload's text field into a Python dict."""
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
                return None
    return None


# ── data loading (called once at startup) ────────────────────────────────────

def load_all_records():
    """Load all DocumentChunk_text records via Qdrant Scroll API."""
    global _records_cache
    records = []
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name="DocumentChunk_text",
            limit=250,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            data = _parse_record(p.payload)
            if data:
                records.append({"id": str(p.id), "data": data, "payload": p.payload})
        if offset is None:
            break
    _records_cache = records
    return records


def compute_analytics():
    """Build analytics cache from loaded records."""
    global _analytics_cache
    records = _records_cache
    invoices = [r["data"] for r in records if "invoice_number" in r["data"]]
    transactions = [r["data"] for r in records if "transaction_id" in r["data"]]

    vendor_spend = defaultdict(float)
    monthly_spend = defaultdict(float)
    product_qty = defaultdict(int)
    product_revenue = defaultdict(float)
    vendor_invoice_count = defaultdict(int)

    for inv in invoices:
        vid = inv.get("vendor_id", "unknown")
        total = float(inv.get("total", 0))
        vendor_spend[f"Vendor {vid}"] += total
        vendor_invoice_count[f"Vendor {vid}"] += 1
        date = inv.get("date", "")
        if date:
            monthly_spend[date[:7]] += total
        items_raw = inv.get("items", "[]")
        items = items_raw if isinstance(items_raw, list) else []
        if isinstance(items_raw, str):
            try:
                items = json.loads(items_raw.replace("'", '"'))
            except Exception:
                try:
                    items = eval(items_raw)
                except Exception:
                    items = []
        for item in items:
            name = item.get("product", "Unknown")
            product_qty[name] += int(item.get("qty", 0))
            product_revenue[name] += float(item.get("total", 0))

    for tx in transactions:
        vid = tx.get("vendor_id", "unknown")
        amt = float(tx.get("amount", 0))
        vendor_spend[f"Vendor {vid}"] += amt
        date = tx.get("date", "")
        if date:
            monthly_spend[date[:7]] += amt

    _analytics_cache = {
        "vendor_spend": dict(sorted(vendor_spend.items(), key=lambda x: -x[1])),
        "vendor_invoice_count": dict(sorted(vendor_invoice_count.items(), key=lambda x: -x[1])),
        "monthly_spend": dict(sorted(monthly_spend.items())),
        "top_products_qty": dict(sorted(product_qty.items(), key=lambda x: -x[1])[:20]),
        "top_products_revenue": dict(sorted(product_revenue.items(), key=lambda x: -x[1])[:20]),
        "total_invoices": len(invoices),
        "total_transactions": len(transactions),
        "total_spend": sum(vendor_spend.values()),
    }
    return _analytics_cache


# ── pricing intelligence (called once at startup) ────────────────────────────

def _parse_items(items_raw):
    """Parse items field from a record."""
    if isinstance(items_raw, list):
        return items_raw
    if isinstance(items_raw, str):
        try:
            return json.loads(items_raw.replace("'", '"'))
        except Exception:
            try:
                return eval(items_raw)
            except Exception:
                return []
    return []


def compute_pricing_intelligence():
    """Build pricing intelligence cache: vendor price comparisons, trends, savings."""
    global _pricing_cache
    records = _records_cache

    # product_key -> vendor_key -> list of {price, qty, month}
    product_vendor_data = defaultdict(lambda: defaultdict(list))

    for r in records:
        data = r["data"]
        vendor_id = data.get("vendor_id")
        date = data.get("date", "")
        month = date[:7] if date else "unknown"
        items = _parse_items(data.get("items", []))

        for item in items:
            product = item.get("product", "")
            sku = item.get("sku", "")
            price = item.get("price")
            qty = item.get("qty", 0)
            if not product or price is None or vendor_id is None:
                continue
            product_vendor_data[product][f"Vendor {vendor_id}"].append({
                "price": float(price),
                "qty": int(qty),
                "total": float(item.get("total", 0)),
                "month": month,
                "sku": sku,
            })

    # Build comparison matrix
    product_vendor_prices = {}
    for product, vendors in product_vendor_data.items():
        vendor_stats = {}
        all_prices = []
        total_qty_all = 0
        for vendor, entries in vendors.items():
            prices = [e["price"] for e in entries]
            qtys = [e["qty"] for e in entries]
            spend = sum(e["total"] for e in entries)
            vendor_stats[vendor] = {
                "avg_price": round(statistics.mean(prices), 2),
                "min_price": round(min(prices), 2),
                "max_price": round(max(prices), 2),
                "total_qty": sum(qtys),
                "total_spend": round(spend, 2),
                "entries": len(entries),
            }
            all_prices.extend(prices)
            total_qty_all += sum(qtys)

        if not vendor_stats:
            continue

        best_vendor = min(vendor_stats, key=lambda v: vendor_stats[v]["avg_price"])
        best_price = vendor_stats[best_vendor]["avg_price"]
        worst_price = max(vs["avg_price"] for vs in vendor_stats.values())

        # Calculate savings if all purchases used best price
        actual_spend = sum(vs["total_spend"] for vs in vendor_stats.values())
        optimal_spend = best_price * total_qty_all
        savings = round(actual_spend - optimal_spend, 2)

        sku = ""
        for v_entries in vendors.values():
            for e in v_entries:
                if e.get("sku"):
                    sku = e["sku"]
                    break
            if sku:
                break

        product_vendor_prices[product] = {
            "sku": sku,
            "vendors": vendor_stats,
            "best_vendor": best_vendor,
            "best_price": best_price,
            "worst_price": worst_price,
            "savings_potential": max(0, savings),
            "total_qty": total_qty_all,
        }

    # Build price trends
    price_trends = {}
    for product, vendors in product_vendor_data.items():
        trends = []
        for vendor, entries in vendors.items():
            month_groups = defaultdict(list)
            for e in entries:
                month_groups[e["month"]].append(e)
            for month, group in sorted(month_groups.items()):
                avg_p = statistics.mean(g["price"] for g in group)
                total_q = sum(g["qty"] for g in group)
                trends.append({
                    "month": month,
                    "vendor": vendor,
                    "avg_price": round(avg_p, 2),
                    "qty": total_q,
                })
        price_trends[product] = sorted(trends, key=lambda x: x["month"])

    # Summary stats
    total_savings = sum(p["savings_potential"] for p in product_vendor_prices.values())
    overpriced = [
        {"product": name, "spread": round(info["worst_price"] - info["best_price"], 2),
         "best_vendor": info["best_vendor"], "best_price": info["best_price"], "worst_price": info["worst_price"]}
        for name, info in product_vendor_prices.items()
        if info["worst_price"] > info["best_price"] * 1.1 and info["best_price"] > 0
    ]
    overpriced.sort(key=lambda x: -x["spread"])

    _pricing_cache = {
        "product_vendor_prices": product_vendor_prices,
        "price_trends": price_trends,
        "total_savings_potential": round(total_savings, 2),
        "products_analyzed": len(product_vendor_prices),
        "overpriced_products": overpriced[:15],
    }
    return _pricing_cache


# ── anomaly detection (called once at startup) ──────────────────────────────

def _detect_amount_outliers(records, field="total", z_threshold=2.5):
    amounts = [(float(r["data"].get(field, 0)), r) for r in records if r["data"].get(field) is not None]
    if len(amounts) < 5:
        return []
    values = [a[0] for a in amounts]
    mean, stdev = statistics.mean(values), statistics.stdev(values)
    if stdev == 0:
        return []
    return sorted(
        [
            {
                "id": r["id"],
                "type": "amount_outlier",
                "severity": "high" if abs(v - mean) / stdev > 4 else "medium",
                "detail": f"{field}=${v:,.2f} (z={abs(v - mean) / stdev:.1f}, mean=${mean:,.2f})",
                "data": r["data"],
            }
            for v, r in amounts
            if abs(v - mean) / stdev > z_threshold
        ],
        key=lambda x: -float(x["detail"].split("z=")[1].split(",")[0]),
    )


def _detect_duplicates(records):
    """Near-duplicate detection via Qdrant Batch Query API."""
    duplicates = []
    seen = set()
    batch_size = 50
    scan = records[:200]
    for batch_start in range(0, len(scan), batch_size):
        batch = scan[batch_start: batch_start + batch_size]
        requests = [
            QueryRequest(query=r["id"], using="text", limit=3, score_threshold=0.99, with_payload=True)
            for r in batch
        ]
        try:
            responses = qdrant.query_batch_points(
                collection_name="DocumentChunk_text", requests=requests, timeout=30
            )
        except Exception:
            continue
        for r, resp in zip(batch, responses):
            for match in resp.points:
                if str(match.id) != r["id"]:
                    pair = tuple(sorted([r["id"], str(match.id)]))
                    if pair not in seen:
                        seen.add(pair)
                        duplicates.append(
                            {
                                "id": r["id"],
                                "type": "near_duplicate",
                                "severity": "high" if match.score > 0.999 else "medium",
                                "detail": f"sim={match.score:.4f} with {match.id}",
                                "data": r["data"],
                            }
                        )
    return duplicates


def _detect_vendor_anomalies(records):
    vendor_totals = defaultdict(list)
    for r in records:
        vid, total = r["data"].get("vendor_id"), r["data"].get("total")
        if vid is not None and total is not None:
            vendor_totals[vid].append(float(total))
    return sorted(
        [
            {
                "id": f"vendor_{vid}",
                "type": "vendor_variance",
                "severity": "medium",
                "detail": f"Vendor {vid}: CV={statistics.stdev(t) / statistics.mean(t):.2f}, mean=${statistics.mean(t):,.0f}, n={len(t)}",
                "data": {"vendor_id": vid, "count": len(t), "spend": sum(t)},
            }
            for vid, t in vendor_totals.items()
            if len(t) >= 3 and statistics.mean(t) > 0 and statistics.stdev(t) / statistics.mean(t) > 0.8
        ],
        key=lambda x: -float(x["detail"].split("CV=")[1].split(",")[0]),
    )


def run_anomaly_detection():
    """Run all anomaly detectors on cached records. Called once at startup."""
    global _anomaly_cache
    records = _records_cache
    all_anomalies = []
    all_anomalies.extend(_detect_amount_outliers(records))
    all_anomalies.extend(_detect_duplicates(records))
    all_anomalies.extend(_detect_vendor_anomalies(records))

    summary = {
        "total": len(all_anomalies),
        "high": sum(1 for a in all_anomalies if a["severity"] == "high"),
        "medium": sum(1 for a in all_anomalies if a["severity"] == "medium"),
        "by_type": {},
    }
    for a in all_anomalies:
        summary["by_type"][a["type"]] = summary["by_type"].get(a["type"], 0) + 1

    _anomaly_cache = {"anomalies": all_anomalies, "summary": summary}
    return _anomaly_cache


# ── TOOL IMPLEMENTATIONS ────────────────────────────────────────────────────


async def search_knowledge(query: str, collection: str = "DocumentChunk_text", limit: int = 10) -> dict:
    """Qdrant Prefetch + RRF Fusion search."""
    t0 = time.time()
    vec = get_embedding(query)
    embed_ms = round((time.time() - t0) * 1000, 1)

    t1 = time.time()
    results = qdrant.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=vec, using="text", limit=100),
            Prefetch(query=vec, using="text", limit=50),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    search_ms = round((time.time() - t1) * 1000, 1)

    items = []
    for p in results.points:
        payload = p.payload or {}
        data = _parse_record(payload)
        items.append({
            "id": str(p.id),
            "score": round(p.score, 4),
            "text": payload.get("text", "")[:300],
            "data": data,
        })

    return {
        "tool": "search_knowledge",
        "results": items,
        "count": len(items),
        "embed_ms": embed_ms,
        "search_ms": search_ms,
        "method": "prefetch_rrf_fusion",
    }


async def analyze_spend(vendor_id: int | None = None, compare_vendors: str | None = None) -> dict:
    """Return analytics from cache, optionally filtered."""
    data = dict(_analytics_cache)
    if vendor_id is not None:
        key = f"Vendor {vendor_id}"
        data = {
            "vendor_spend": {key: data["vendor_spend"].get(key, 0)},
            "vendor_invoice_count": {key: data["vendor_invoice_count"].get(key, 0)},
            "total_spend": data["vendor_spend"].get(key, 0),
        }
    if compare_vendors:
        ids = [f"Vendor {v.strip()}" for v in compare_vendors.split(",")]
        data["comparison"] = {v: _analytics_cache["vendor_spend"].get(v, 0) for v in ids}
    data["tool"] = "analyze_spend"
    return data


async def detect_anomalies(vendor_id: int | None = None, anomaly_type: str | None = None) -> dict:
    """Return anomalies from cache, optionally filtered."""
    anomalies = list(_anomaly_cache.get("anomalies", []))
    if vendor_id is not None:
        anomalies = [a for a in anomalies if a.get("data", {}).get("vendor_id") == vendor_id
                     or str(a.get("data", {}).get("vendor_id")) == str(vendor_id)]
    if anomaly_type:
        anomalies = [a for a in anomalies if a["type"] == anomaly_type]
    return {
        "tool": "detect_anomalies",
        "anomalies": anomalies[:20],
        "count": len(anomalies),
        "summary": _anomaly_cache.get("summary", {}),
    }


async def investigate_record(record_id: str) -> dict:
    """Qdrant Recommend API — find records similar to a given point."""
    t0 = time.time()
    try:
        results = qdrant.query_points(
            collection_name="DocumentChunk_text",
            query=RecommendQuery(
                recommend=RecommendInput(
                    positive=[record_id],
                    strategy=RecommendStrategy.BEST_SCORE,
                )
            ),
            using="text",
            limit=10,
            with_payload=True,
        )
        items = []
        for s in results.points:
            payload = s.payload or {}
            items.append({
                "id": str(s.id),
                "score": round(s.score, 4),
                "text": payload.get("text", "")[:300],
            })
    except Exception as e:
        items = []
    return {
        "tool": "investigate_record",
        "record_id": record_id,
        "similar": items,
        "count": len(items),
        "time_ms": round((time.time() - t0) * 1000, 1),
        "method": "recommend_best_score",
    }


async def recall_memory(query: str) -> dict:
    """Search cognee knowledge graph for past findings."""
    import cognee
    from cognee.api.v1.search import SearchType

    t0 = time.time()
    try:
        results = await cognee.search(query_text=query, query_type=SearchType.CHUNKS)
        items = [str(r)[:300] for r in (results or [])[:5]]
    except Exception as e:
        items = [f"Memory search error: {e}"]
    return {
        "tool": "recall_memory",
        "query": query,
        "results": items,
        "count": len(items),
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


async def store_finding(text: str) -> dict:
    """Store a finding in the cognee knowledge graph."""
    import cognee

    t0 = time.time()
    try:
        await cognee.add(text)
        await cognee.cognify()
        status = "stored"
    except Exception as e:
        status = f"error: {e}"
    return {
        "tool": "store_finding",
        "status": status,
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


async def ingest_document(raw_text: str) -> dict:
    """Ingest a new document through cognee's entity extraction pipeline."""
    import cognee
    from cognee.api.v1.search import SearchType

    t0 = time.time()
    try:
        await cognee.add(raw_text)
        await cognee.cognify()
        results = await cognee.search(query_text=raw_text[:100], query_type=SearchType.CHUNKS)
        count = len(results) if results else 0
        status = "ingested"
    except Exception as e:
        count = 0
        status = f"error: {e}"
    return {
        "tool": "ingest_document",
        "status": status,
        "chunks_found": count,
        "time_ms": round((time.time() - t0) * 1000, 1),
    }


async def analyze_pricing(product_name: str | None = None, vendor_id: int | None = None) -> dict:
    """Return pricing intelligence from cache, optionally filtered."""
    data = dict(_pricing_cache)
    result = {"tool": "analyze_pricing"}

    if product_name:
        # Fuzzy match product name
        matches = {k: v for k, v in data.get("product_vendor_prices", {}).items()
                   if product_name.lower() in k.lower()}
        result["product_vendor_prices"] = matches
        result["price_trends"] = {k: v for k, v in data.get("price_trends", {}).items()
                                  if product_name.lower() in k.lower()}
    elif vendor_id is not None:
        vkey = f"Vendor {vendor_id}"
        filtered = {}
        for prod, info in data.get("product_vendor_prices", {}).items():
            if vkey in info.get("vendors", {}):
                filtered[prod] = info
        result["product_vendor_prices"] = filtered
        result["price_trends"] = {k: [t for t in v if t.get("vendor") == vkey]
                                  for k, v in data.get("price_trends", {}).items()
                                  if any(t.get("vendor") == vkey for t in v)}
    else:
        result["product_vendor_prices"] = dict(list(data.get("product_vendor_prices", {}).items())[:10])
        result["price_trends"] = {}

    result["total_savings_potential"] = data.get("total_savings_potential", 0)
    result["products_analyzed"] = data.get("products_analyzed", 0)
    result["overpriced_products"] = data.get("overpriced_products", [])[:5]
    return result


# ── tool dispatcher ──────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "search_knowledge": search_knowledge,
    "analyze_spend": analyze_spend,
    "analyze_pricing": analyze_pricing,
    "detect_anomalies": detect_anomalies,
    "investigate_record": investigate_record,
    "recall_memory": recall_memory,
    "store_finding": store_finding,
    "ingest_document": ingest_document,
}


async def execute_tool(name: str, params: dict) -> dict:
    """Execute a tool by name with given params."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return await fn(**params)
    except Exception as e:
        return {"error": f"Tool {name} failed: {e}"}
