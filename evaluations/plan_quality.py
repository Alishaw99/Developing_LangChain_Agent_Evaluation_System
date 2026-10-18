import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "agent")
)

import sys
from pathlib import Path
from uuid import uuid4

# ==================================================
# 1. PROJECT PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "agent")
)

from agent import agent
from dataset import dataset
from model import eval_model

from deepeval.metrics import PlanQualityMetric
from deepeval.integrations.langchain import CallbackHandler

# Import deterministic refund policy validator
from refund_policy_validator import validate_refund_trace


# ==================================================
# 2. PLAN QUALITY METRIC
# ==================================================

metrics = [
    PlanQualityMetric(
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
# 4. RUN AGENT EVALUATIONS
# ==================================================

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

    # ==============================================
    # 5. DETERMINISTIC REFUND POLICY VALIDATION
    # ==============================================

    test_case = REFUND_TEST_CASES.get(golden.input)

    if test_case:

        validation = validate_refund_trace(
            response=response,
            username=test_case["username"],
            requested_items=test_case["requested_items"],
            refund_window=30
        )

        print("\nREFUND POLICY VALIDATION")
        print("-" * 45)

        print("Passed:", validation["passed"])
        print("Score:", validation["score"])
        print("Eligible Items:", validation["eligible_items"])
        print("Expected Refund:", validation["expected_amount"])
        print("Tool Sequence:", validation["tool_sequence"])

        if validation["errors"]:
            print("\nPolicy Violations:")

            for error in validation["errors"]:
                print("-", error)

        else:
            print("All refund policy checks passed.")

    else:
        print(
            "\nDeterministic policy validation "
            "not configured for this test case."
        )
