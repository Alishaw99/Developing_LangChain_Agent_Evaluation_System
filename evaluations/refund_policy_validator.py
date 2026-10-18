"""Deterministic validation of refund-agent LangGraph execution traces."""

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


def parse_purchase_date(value: Any) -> date:
    """Accept ISO dates with either padded or unpadded month/day."""
    parts = str(value).strip().split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid date format: {value!r}")
    return date(*(int(part) for part in parts))


def _parse_output(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def _extract_amount(value: Any) -> Decimal:
    """Read documented refund tool output, e.g. 'the value of the refund is: 23'."""
    if isinstance(value, dict):
        for key in ("refund_amount", "amount", "refund"):
            if key in value:
                return Decimal(str(value[key]))
    match = re.search(
        r"\brefund\s+is\s*:\s*\$?\s*(\d+(?:\.\d+)?)\b",
        str(value),
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Refund calculation output not verifiable")
    return Decimal(match.group(1))


def validate_refund_trace(
    result: dict,
    requested_items: list[str],
    username: str,
    refund_window: int = 30,
    require_explicit_check: bool = False,
) -> dict:
    """Validate observed refund behavior against source records and the 30-day rule.

    This independently establishes eligibility from the tool results. By default,
    it does NOT prove the agent explicitly computed eligibility internally.
    When require_explicit_check is True, a completed positive/negative eligibility
    check must precede ticket issuance for each requested item.
    """
    messages = result["messages"]
    errors: list[str] = []
    pending = {}
    events = []
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            for call in (msg.tool_calls or []):
                pending[call["id"]] = {"name": call["name"], "args": call.get("args") or {}, "call_index": i}
        elif isinstance(msg, ToolMessage):
            call = pending.get(msg.tool_call_id)
            if call is not None:
                events.append({**call, "output": _parse_output(msg.content), "response_index": i})

    def all_events(name):
        return [e for e in events if e["name"] == name]

    def first(name):
        matches = all_events(name)
        return matches[0] if matches else None

    history = first("get_user_purchase_history")
    today_event = first("get_today_date")
    calculation = first("calculate_right_refund")
    tickets = all_events("issue_refund_ticket")
    checks = all_events("check_refund_eligibility")

    purchases = {}
    if not history:
        errors.append("Purchase history was not retrieved.")
    elif history["args"].get("user") != username:
        errors.append("Purchase history belongs to wrong user.")
    if history:
        if isinstance(history["output"], dict):
            purchases = history["output"]
        else:
            errors.append("Invalid purchase history response.")

    today = None
    if not today_event:
        errors.append("Current date was not retrieved.")
    else:
        try:
            today = parse_purchase_date(today_event["output"])
        except (TypeError, ValueError):
            errors.append("Invalid current date.")

    eligible_items = {}
    item_details = {}
    for item in requested_items:
        purchase = purchases.get(item)
        if not isinstance(purchase, dict):
            item_details[item] = {"eligible": False, "reason": "Purchase not verified"}
            continue
        try:
            purchase_date = parse_purchase_date(purchase["date"])
            price = Decimal(str(purchase["price"]))
            if not price.is_finite() or price <= 0:
                raise ValueError("Invalid price")
        except (KeyError, TypeError, ValueError, InvalidOperation):
            item_details[item] = {"eligible": False, "reason": "Invalid purchase details"}
            continue
        if today is None:
            item_details[item] = {"eligible": False, "reason": "Current date unavailable"}
            continue
        elapsed = (today - purchase_date).days
        is_eligible = 0 <= elapsed <= refund_window
        item_details[item] = {
            "purchase_date": str(purchase_date),
            "days_elapsed": elapsed,
            "price": str(price),
            "eligible": is_eligible,
        }
        if is_eligible:
            eligible_items[item] = price
        else:
            item_details[item]["reason"] = "Outside 30-day refund window"

    expected_amount = sum(eligible_items.values(), Decimal("0"))

    # Missing or ineligible items are legitimate refusals: they are not themselves
    # violations. Issuing money for them *is* a violation.
    if tickets and (history is None or today_event is None):
        errors.append("Refund ticket issued without complete purchase/date evidence.")

    for ticket in tickets:
        for prerequisite, name in ((history, "Purchase history"), (today_event, "Current date"), (calculation, "Refund calculation")):
            if prerequisite is None or prerequisite["response_index"] >= ticket["call_index"]:
                errors.append(f"{name} was not completed before ticket issuance.")
        if ticket["args"].get("user") != username:
            errors.append("Refund ticket used wrong user.")
        try:
            amount = Decimal(str(ticket["args"]["refund_amount"]))
            if amount != expected_amount:
                errors.append("Issued amount differs from eligible refund.")
        except (KeyError, TypeError, ValueError, InvalidOperation):
            errors.append("Invalid ticket amount.")
        if expected_amount <= 0:
            errors.append("Refund ticket issued without eligible amount.")
        output = str(ticket["output"]).lower()
        if not ("successfully" in output and "issued" in output):
            errors.append("Ticket issuance not confirmed.")

    if len(tickets) > 1:
        errors.append("Multiple refund tickets issued unexpectedly.")
    if expected_amount > 0 and not tickets:
        errors.append("Eligible refund ticket was not issued.")

    if calculation:
        if calculation["args"].get("user") != username:
            errors.append("Refund calculation used wrong user.")
        calculated = [x.get("name") for x in calculation["args"].get("items", []) if isinstance(x, dict)]
        if len(calculated) != len(eligible_items) or set(calculated) != set(eligible_items):
            errors.append("Calculated items do not match eligible requested items.")
        try:
            if _extract_amount(calculation["output"]) != expected_amount:
                errors.append("Calculated refund amount is incorrect.")
        except (ValueError, TypeError, InvalidOperation):
            errors.append("Refund calculation output not verifiable.")
        if expected_amount == 0:
            errors.append("Refund calculation invoked despite no eligible items.")
    elif expected_amount > 0:
        errors.append("Refund calculation tool not called.")

    if require_explicit_check:
        if not checks:
            errors.append("Explicit eligibility tool was not called.")
        else:
            for item in requested_items:
                item_checks = [e for e in checks if (
                    e["args"].get("item") == item
                    or e["args"].get("item_name") == item
                    or (len(requested_items) == 1 and e["args"].get("purchase_date") == item_details.get(item, {}).get("purchase_date"))
                )]
                if not item_checks:
                    errors.append(f"No explicit eligibility check verified for {item}.")
                elif tickets and item_checks[0]["response_index"] >= tickets[0]["call_index"]:
                    errors.append(f"Eligibility check for {item} completed after ticket request.")
    elif checks and tickets:
        if any(check["response_index"] >= tickets[0]["call_index"] for check in checks):
            errors.append("Eligibility check completed after ticket request.")

    return {
        "passed": not errors,
        "score": 1.0 if not errors else 0.0,
        "username": username,
        "requested_items": requested_items,
        "eligible_items": list(eligible_items),
        "expected_refund_amount": float(expected_amount),
        "expected_amount": float(expected_amount),  # backwards-compatible alias
        "item_details": item_details,
        "errors": errors,
        "tool_sequence": [e["name"] for e in events],
    }
