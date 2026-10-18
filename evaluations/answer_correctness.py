
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "agent"))

load_dotenv(PROJECT_ROOT / ".env")

from agent import agent
from dataset import dataset
from model import eval_model

from deepeval.metrics import GEval
from deepeval.integrations.langchain import CallbackHandler
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval import evaluate
from deepeval.evaluate import CacheConfig, AsyncConfig

from refund_policy_validator import validate_refund_trace


# ==================================================
# 2. CONFIGURE REFUND TEST CASES
# ==================================================

REFUND_TEST_CASES = {
    "I am user123, I would like a refund for the book": {
        "username": "user123",
        "requested_items": ["book"]
    }
}


# ==================================================
# 3. CONFIGURE ANSWER CORRECTNESS METRIC
# ==================================================

metrics = [
    GEval(
        name="Answer Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        evaluation_steps=[
            "Identify the user's original request.",
            "Identify the expected answer's main conclusion.",
            "Compare the actual answer with the expected answer.",
            "Determine whether both answers reach the same "
            "logical conclusion.",
            "For refunds, distinguish between a refund ticket "
            "being issued and money reaching a bank account.",
            "Do not penalize the agent for omitting internal "
            "eligibility calculations from its final response.",
            "Penalize unsupported claims about refund approval, "
            "payment completion, or refund arrival dates.",
            "Explain any material contradiction or omission."
        ],
        threshold=0.8,
        model=eval_model,
        verbose_mode=True
    )
]


# ==================================================
# 4. INITIALIZE RESULT COLLECTIONS
# ==================================================

test_cases = []
policy_results = []


# ==================================================
# 5. EXECUTE AGENT FOR EACH GOLDEN
# ==================================================

for golden in dataset.goldens:

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
            "callbacks": [CallbackHandler()]
        }
    )

    # ==================================================
    # 6. EXTRACT ACTUAL RESPONSE
    # ==================================================

    structured_response = response.get(
        "structured_response"
    )

    if structured_response is not None:
        if isinstance(structured_response, dict):
            actual_output = structured_response["answer"]
        else:
            actual_output = structured_response.answer
    else:
        actual_output = str(
            response["messages"][-1].content
        )

    print("\nACTUAL ANSWER:")
    print(actual_output)

    print("\nEXPECTED ANSWER:")
    print(golden.expected_output)

    # ==================================================
    # 7. CREATE DEEPEVAL TEST CASE
    # ==================================================

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output
    )

    test_cases.append(test_case)

    # ==================================================
    # 8. DETERMINISTIC REFUND POLICY VALIDATION
    # ==================================================

    refund_case = REFUND_TEST_CASES.get(
        golden.input.strip()
    )

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
        print(
            "Eligible Items:",
            validation.get("eligible_items")
        )

        print(
            "Expected Refund:",
            validation.get(
                "expected_refund_amount",
                validation.get("expected_amount")
            )
        )

        print(
            "Tool Sequence:",
            validation.get("tool_sequence")
        )

        print("\nPURCHASE DETAILS")
        print("-" * 45)

        for item, details in validation.get(
            "item_details", {}
        ).items():

            print(f"\nItem: {item}")

            for key, value in details.items():
                print(f"  {key}: {value}")

        print("\nPOLICY VALIDATION RESULTS")
        print("-" * 45)

        errors = validation.get("errors", [])

        if errors:
            print("Policy validation FAILED")

            for error in errors:
                print("-", error)
        else:
            print("All refund policy checks passed.")

        policy_results.append({
            "input": golden.input,
            "passed": validation.get("passed", False),
            "score": validation.get("score", 0.0),
            "errors": errors
        })

    else:
        print(
            "\nDeterministic refund validation "
            "not configured for this input."
        )


# ==================================================
# 9. RUN DEEPEVAL ANSWER CORRECTNESS
# ==================================================

print("\n" + "=" * 65)
print("DEEPEVAL ANSWER CORRECTNESS")
print("=" * 65)

# ==================================================
# 9. RUN DEEPEVAL ANSWER CORRECTNESS
# ==================================================

print("\n" + "=" * 65)
print("DEEPEVAL ANSWER CORRECTNESS")
print("=" * 65)

evaluation_results = evaluate(
    test_cases=test_cases,
    metrics=metrics,

    # Disable reading and writing cached results
    cache_config=CacheConfig(
        use_cache=False,
        write_cache=False
    ),

    # Execute evaluations sequentially
    async_config=AsyncConfig(
        run_async=False
    )
)

print("\nEvaluation completed.")


# ==================================================
# 10. REFUND POLICY SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("REFUND POLICY VALIDATION SUMMARY")
print("=" * 65)

for item in policy_results:

    status = "PASS" if item["passed"] else "FAIL"

    print(f"\n{status}: {item['input']}")
    print("Policy Score:", item["score"])

    for error in item["errors"]:
        print(" -", error)

print("\nTotal answer test cases:", len(test_cases))
print("Total deterministic policy tests:", len(policy_results))
