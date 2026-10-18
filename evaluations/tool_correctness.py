
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

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.integrations.langchain import CallbackHandler
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate

# Deterministic refund policy validator
from refund_policy_validator import validate_refund_trace


# ==================================================
# 2. REFUND TEST CASES
# ==================================================

REFUND_TEST_CASES = {
    "I am user123, I would like a refund for the book": {
        "username": "user123",
        "requested_items": ["book"]
    }
}


# ==================================================
# 3. COLLECT AGENT EXECUTIONS
# ==================================================

test_cases = []
policy_results = []

for golden in dataset.goldens:

    print("\n" + "=" * 65)
    print("Evaluating:", golden.input)
    print("=" * 65)

    # Run the agent once per golden
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
    # 4. EXTRACT AGENT OUTPUT
    # ==================================================

    structured_response = response.get(
        "structured_response"
    )

    if structured_response is not None:
        actual_output = structured_response.answer
    else:
        actual_output = str(
            response["messages"][-1].content
        )

    # ==================================================
    # 5. EXTRACT ACTUAL TOOL CALLS
    # ==================================================

    tools_called = []

    for message in response["messages"]:

        for tool_call in (
            getattr(message, "tool_calls", []) or []
        ):
            tools_called.append(
                ToolCall(
                    name=tool_call["name"],
                    input_parameters=tool_call["args"]
                )
            )

    print("\nTools called:")

    for tool in tools_called:
        print("-", tool.name)

    # ==================================================
    # 6. BUILD DEEPEVAL TEST CASE
    # ==================================================

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        tools_called=tools_called,
        expected_tools=golden.expected_tools
    )

    test_cases.append(test_case)

    
    # ==================================================
    # 7. INDEPENDENT REFUND POLICY VALIDATION
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
            validation.get("expected_refund_amount")
        )
        print("Tool Sequence:", validation.get("tool_sequence"))

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

        if validation.get("errors"):

            print("Policy validation FAILED")

            for error in validation["errors"]:
                print(" -", error)

        else:
            print("All refund policy checks passed.")

        policy_results.append({
            "input": golden.input,
            "passed": validation.get("passed", False),
            "score": validation.get("score", 0.0),
            "errors": validation.get("errors", [])
        })

    else:
        print(
            "\nDeterministic refund validation "
            "not configured for this input."
        )

# ==================================================
# 9. POLICY VALIDATION SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("REFUND POLICY VALIDATION SUMMARY")
print("=" * 65)

for result in policy_results:

    status = "PASS" if result["passed"] else "FAIL"

    print(f"\n{status}: {result['input']}")
    print("Policy score:", result["score"])

    for error in result["errors"]:
        print(" -", error)
