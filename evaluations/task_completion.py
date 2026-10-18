
import sys
import json
import re

from pathlib import Path
from uuid import uuid4
from datetime import date
from decimal import Decimal

from langchain_core.messages import AIMessage, ToolMessage

# --------------------------------------------------
# 1. IMPORT AGENT AND EVALUATION COMPONENTS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "agent"))

from agent import agent
from dataset import dataset
from model import eval_model

from deepeval.metrics import TaskCompletionMetric
from deepeval.integrations.langchain import CallbackHandler
from deepeval.evaluate import AsyncConfig


# --------------------------------------------------
# 2. TASK COMPLETION DESCRIPTION
# --------------------------------------------------

SUCCESSFUL_TASK_DESCRIPTION = """
Evaluate the agent's completion of the user's request.

If a refund is requested without a username:
- Ask for the username.
- Do not issue a refund ticket.

If a username and refund request are provided:
- Retrieve the user's actual purchase history.
- Verify that every refunded item was purchased.
- Obtain today's date.
- Compare each requested item's actual purchase date
  against today's date.
- A purchase is eligible only when it is between
  0 and 30 days old, inclusive.
- Do not refund items with unverifiable purchase dates.
- Refund only the requested and eligible items.
- Calculate the correct refund amount.
- Do not issue a refund ticket for ineligible items.
- Issue a ticket only after all required checks.
- Confirm accurately whether a refund ticket was issued.
- Explain why a refund was not issued when applicable.

For non-refund information requests:
- Answer appropriately.
- Do not issue a refund ticket.

IMPORTANT EVALUATION INSTRUCTIONS:
- Evaluate the complete execution trace, including tool
  calls, arguments, responses, and final answer.
- The purchase date does not need to appear in the final
  user-facing response if it is present in the tool trace.
- The purchase history and current date must establish
  eligibility before refund issuance.
- Do not assume a purchased item is automatically eligible.
- Do not equate successful ticket creation with funds
  being deposited into the customer's bank account.
"""


# --------------------------------------------------
# 3. DEEPEVAL METRIC
# --------------------------------------------------

metrics = [
    TaskCompletionMetric(
        threshold=0.8,
        model=eval_model,
        task=SUCCESSFUL_TASK_DESCRIPTION,
        include_reason=True
    )
]


# --------------------------------------------------
# 4. DETERMINISTIC REFUND VALIDATOR
# --------------------------------------------------

def extract_tool_events(response):
    """Match each completed tool call with its response."""

    pending = {}
    events = []

    for index, message in enumerate(response["messages"]):

        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                pending[call["id"]] = {
                    "name": call["name"],
                    "args": call["args"],
                    "call_index": index
                }

        elif isinstance(message, ToolMessage):
            call = pending.get(message.tool_call_id)

            if call is None:
                continue

            try:
                output = json.loads(message.content)
            except (ValueError, TypeError):
                output = message.content

            events.append({
                **call,
                "output": output,
                "response_index": index
            })

    return events


def validate_refund(
    response,
    username,
    requested_items,
    refund_window=30
):
    """Independently check refund eligibility and issuance."""

    events = extract_tool_events(response)
    errors = []

    def find_all(name):
        return [e for e in events if e["name"] == name]

    def first(name):
        matches = find_all(name)
        return matches[0] if matches else None

    history = first("get_user_purchase_history")
    today_event = first("get_today_date")
    calculation = first("calculate_right_refund")
    tickets = find_all("issue_refund_ticket")

    if not history:
        errors.append("Purchase history not retrieved.")

    if not today_event:
        errors.append("Current date not retrieved.")

    purchases = {}
    today = None

    if history:
        if history["args"].get("user") != username:
            errors.append("Wrong user in purchase-history call.")

        if isinstance(history["output"], dict):
            purchases = history["output"]
        else:
            errors.append("Invalid purchase-history output.")

    if today_event:
        try:
            today = date.fromisoformat(
                str(today_event["output"])
            )
        except ValueError:
            errors.append("Invalid current date.")

    eligible = {}
    ineligible = {}

    for item in requested_items:
        item_data = purchases.get(item)

        if not isinstance(item_data, dict):
            ineligible[item] = "Purchase not found."
            continue

        try:
            purchase_date = date.fromisoformat(
                item_data["date"]
            )
            price = Decimal(str(item_data["price"]))
            days = (today - purchase_date).days

            if price < 0:
                raise ValueError("Negative price")

            if 0 <= days <= refund_window:
                eligible[item] = price
            else:
                ineligible[item] = (
                    f"Outside refund window: {days} days."
                )

        except (KeyError, TypeError, ValueError, AttributeError):
            ineligible[item] = (
                "Purchase date or price cannot be verified."
            )

    expected_amount = sum(
        eligible.values(),
        Decimal("0")
    )

    # Check the prerequisite sequence.
    # Completion of the tool response must precede
    # the refund ticket call.

    for ticket in tickets:
        for prerequisite, label in [
            (history, "Purchase history"),
            (today_event, "Current date"),
            (calculation, "Refund calculation")
        ]:
            if prerequisite and (
                prerequisite["response_index"]
                >= ticket["call_index"]
            ):
                errors.append(
                    f"{label} completed after ticket request."
                )

    # Check item selection and amount calculation.

    if calculation:
        args = calculation["args"]

        calculated_items = [
            item.get("name")
            for item in args.get("items", [])
            if isinstance(item, dict)
        ]

        if args.get("user") != username:
            errors.append("Wrong user in refund calculation.")

        if (
            len(calculated_items) != len(eligible)
            or set(calculated_items) != set(eligible)
        ):
            errors.append(
                "Calculated items differ from eligible items."
            )

        # Adapt this parser if the tool changes its output format.
        output = str(calculation["output"])

        match = re.search(
            r"refund\s+is:\s*\$?(\d+(?:\.\d+)?)",
            output,
            re.IGNORECASE
        )

        if not match:
            errors.append(
                "Calculated refund amount cannot be verified."
            )
        elif Decimal(match.group(1)) != expected_amount:
            errors.append("Incorrect calculated refund amount.")

    elif expected_amount > 0:
        errors.append("Refund calculation missing.")

    # Check ticket issuance.

    if expected_amount > 0 and not tickets:
        errors.append("Eligible refund ticket not issued.")

    if expected_amount == 0 and tickets:
        errors.append(
            "Ticket issued despite no eligible refund."
        )

    if len(tickets) > 1:
        errors.append("Unexpected multiple refund tickets.")

    for ticket in tickets:
        args = ticket["args"]

        if args.get("user") != username:
            errors.append("Wrong user in refund ticket.")

        try:
            amount = Decimal(str(args["refund_amount"]))

            if amount != expected_amount:
                errors.append("Incorrect ticket amount.")

        except (KeyError, TypeError, ValueError):
            errors.append("Invalid ticket amount.")

        output = str(ticket["output"]).lower()

        if not (
            "successfully" in output and "issued" in output
        ):
            errors.append(
                "Refund ticket success not confirmed."
            )

    # The validator independently establishes date
    # eligibility. It does not prove the model explicitly
    # reasoned through the calculation.

    return {
        "passed": not errors,
        "score": 1.0 if not errors else 0.0,
        "eligible_items": list(eligible),
        "ineligible_items": ineligible,
        "expected_amount": float(expected_amount),
        "tool_sequence": [e["name"] for e in events],
        "errors": errors
    }


# --------------------------------------------------
# 5. CONFIGURE DETERMINISTIC TEST CASES
# --------------------------------------------------

# Explicit fixtures are safer than guessing usernames
# or item names from arbitrary natural-language inputs.
# Add more cases here as your dataset expands.

POLICY_CASES = {
    "I am user123, I would like a refund for the book": {
        "username": "user123",
        "requested_items": ["book"]
    }
}


# --------------------------------------------------
# 6. RUN DEEPEVAL AND DETERMINISTIC CHECKS
# --------------------------------------------------

results = []

for golden in dataset.evals_iterator(
    metrics=metrics,
    async_config=AsyncConfig(run_async=False)
):

    print("\n" + "=" * 65)
    print("EVALUATING:", golden.input)
    print("=" * 65)

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": golden.input
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": str(uuid4())
            },
            "callbacks": [
                CallbackHandler()
            ]
        }
    )

    # DeepEval automatically evaluates the traced
    # agent run through evals_iterator.

    # Run the independent policy check only where
    # an explicit test-case specification exists.

    policy_case = POLICY_CASES.get(golden.input)

    if policy_case:
        validation = validate_refund(
            response=response,
            username=policy_case["username"],
            requested_items=policy_case["requested_items"]
        )

        print("\nDETERMINISTIC POLICY EVALUATION")
        print("Passed:", validation["passed"])
        print("Score:", validation["score"])
        print("Eligible:", validation["eligible_items"])
        print("Amount:", validation["expected_amount"])
        print("Tools:", validation["tool_sequence"])
        print("Errors:", validation["errors"])

        results.append({
            "input": golden.input,
            "policy_score": validation["score"],
            "policy_passed": validation["passed"],
            "errors": validation["errors"]
        })

    else:
        print(
            "Deterministic policy check: NOT CONFIGURED "
            "for this golden."
        )

# --------------------------------------------------
# 7. SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("DETERMINISTIC REFUND EVALUATION SUMMARY")
print("=" * 65)

for item in results:
    status = "PASS" if item["policy_passed"] else "FAIL"
    print(f"{status}: {item['input']}")

print(f"\nConfigured policy tests evaluated: {len(results)}")
print("DeepEval trace scores are reported separately.")
