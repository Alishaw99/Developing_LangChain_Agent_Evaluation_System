
import sys
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "agent")
)

load_dotenv(PROJECT_ROOT / ".env")

from agent import agent
from dataset import dataset
from model import eval_model

from deepeval.metrics import PlanAdherenceMetric
from deepeval.integrations.langchain import CallbackHandler

# Deterministic refund policy validation
from refund_policy_validator import validate_refund_trace


# ==================================================
# 2. PLAN ADHERENCE METRIC
# ==================================================

metrics = [
    PlanAdherenceMetric(
        threshold=0.8,
        model=eval_model,
        include_reason=True
    )
]


# ==================================================
# 3. REFUND TEST CASE CONFIGURATION
# ==================================================

REFUND_TEST_CASES = {
    "I am user123, I would like a refund for the book": {
        "username": "user123",
        "requested_items": ["book"]
    }
}


# ==================================================
# 4. RUN DEEPEVAL EVALUATIONS
# ==================================================

policy_results = []

for golden in dataset.evals_iterator(metrics=metrics):

    print("\n" + "=" * 65)
    print("Evaluating:", golden.input)
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

    # ==================================================
    # 5. DETERMINISTIC REFUND POLICY VALIDATION
    # ==================================================

    refund_case = REFUND_TEST_CASES.get(golden.input.strip())

    if refund_case:

        validation = validate_refund_trace(
            result=response,
            username=refund_case["username"],
            requested_items=refund_case["requested_items"],
            refund_window=30
        )

        print("\nREFUND POLICY VALIDATION")
        print("-" * 45)

        print("Passed:", validation.get("passed"))
        print("Score:", validation.get("score"))
        print("Eligible Items:", validation.get("eligible_items"))
        print(
            "Expected Refund:",
            validation.get(
                "expected_refund_amount",
                validation.get("expected_amount")
            )
        )
        print("Tool Sequence:", validation.get("tool_sequence"))

        print("\nPURCHASE DETAILS")

        for item, details in validation.get(
            "item_details", {}
        ).items():
            print(f"\nItem: {item}")

            for key, value in details.items():
                print(f"  {key}: {value}")

        if validation.get("errors"):

            print("\nPOLICY VIOLATIONS:")

            for error in validation["errors"]:
                print("-", error)

        else:
            print("\nAll refund policy checks passed.")

        policy_results.append({
            "input": golden.input,
            "passed": validation.get("passed", False),
            "score": validation.get("score", 0.0),
            "errors": validation.get("errors", [])
        })

    else:
        print(
            "\nDeterministic refund validation "
            "not configured for this test case."
        )


# ==================================================
# 6. POLICY VALIDATION SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("REFUND POLICY VALIDATION SUMMARY")
print("=" * 65)

for result in policy_results:

    status = "PASS" if result["passed"] else "FAIL"

    print(f"\n{status}: {result['input']}")
    print("Policy Score:", result["score"])

    for error in result["errors"]:
        print(" -", error)

print("\nTotal policy tests:", len(policy_results))
