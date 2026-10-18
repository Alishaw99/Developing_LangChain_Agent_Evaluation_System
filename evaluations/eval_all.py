
import sys
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import (
    CacheConfig,
    DisplayConfig,
    AsyncConfig,
)
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
    ToolCall,
)

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from agent import agent

if __package__:
    from .dataset import dataset
    from .model import eval_model
    from .refund_policy_validator import validate_refund_trace
else:
    from dataset import dataset
    from model import eval_model
    from refund_policy_validator import validate_refund_trace


# ==================================================
# 2. REFUND TEST CASE CONFIGURATION
# ==================================================

REFUND_TEST_CASES = {
    "I am user123, I would like a refund for the book": {
        "username": "user123",
        "requested_items": ["book"],
    }
}


# ==================================================
# 3. SUCCESSFUL TASK DESCRIPTION
# ==================================================

SUCCESSFUL_TASK_DESCRIPTION = """
Evaluate whether the agent successfully completed
the user's task.

USERNAME REQUIREMENT:
- For a refund request without a username, ask for it.
- Do not issue a refund before identifying the user.

REFUND POLICY:
1. Retrieve the user's purchase history.
2. Verify that the requested item was purchased.
3. Retrieve the current date.
4. Compare the actual purchase date with today's date.
5. A refund is eligible only when the purchase is
   between 0 and 30 days old, inclusive.
6. Do not refund items with missing or invalid dates.
7. Refund only items explicitly requested by the user.
8. Calculate the refund using verified purchase prices.
9. Issue a refund ticket only after confirming eligibility.
10. Accurately communicate whether a ticket was issued.

If a refund is not eligible, explain the reason.

If the user only requests information, answer the
question without issuing a refund ticket.

EVALUATION INSTRUCTIONS:
- Examine the complete execution trace.
- Consider the tool calls, arguments, and responses.
- Do not require internal purchase-date calculations
  to appear in the final customer-facing answer.
- Do not assume a purchase is eligible simply
  because it appears in purchase history.
- Distinguish ticket creation from bank payment.
"""


# ==================================================
# 4. BUILD DEEPEVAL METRICS
# ==================================================

def build_metrics():

    trace_metrics = [
        TaskCompletionMetric(
            threshold=0.8,
            model=eval_model,
            task=SUCCESSFUL_TASK_DESCRIPTION,
            include_reason=True,
        ),

        StepEfficiencyMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),

        PlanQualityMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),

        PlanAdherenceMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),
    ]

    single_turn_metrics = [
        ArgumentCorrectnessMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),

        ToolCorrectnessMetric(
            threshold=0.9,
            model=eval_model,
            include_reason=True,
            should_consider_ordering=True,
        ),

        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Identify the user's request.",
                "Compare the actual output with the expected output.",
                "Verify that both reach the same logical conclusion.",
                "For refunds, distinguish ticket creation from "
                "actual receipt of money.",
                "Penalize unsupported claims about payment timing.",
                "Do not require internal eligibility calculations "
                "to appear in the final response.",
            ],
            threshold=0.8,
            model=eval_model,
            verbose_mode=True,
        ),
    ]

    return trace_metrics, single_turn_metrics


# ==================================================
# 5. EXTRACT FINAL AGENT ANSWER
# ==================================================

def get_actual_output(response):

    structured = response.get("structured_response")

    if structured is not None:
        if isinstance(structured, dict):
            return str(structured.get("answer", ""))

        return str(structured.answer)

    last_message = response["messages"][-1]

    return str(last_message.content)


# ==================================================
# 6. EXTRACT TOOL CALLS
# ==================================================

def get_tool_calls(response):

    tools_called = []

    for message in response["messages"]:

        for call in (
            getattr(message, "tool_calls", []) or []
        ):
            tools_called.append(
                ToolCall(
                    name=call["name"],
                    input_parameters=call["args"],
                )
            )

    return tools_called


# ==================================================
# 7. RUN REFUND POLICY VALIDATION
# ==================================================

def run_refund_policy_validation(golden, response):

    refund_case = REFUND_TEST_CASES.get(
        golden.input.strip()
    )

    if refund_case is None:
        return None

    validation = validate_refund_trace(
        result=response,
        username=refund_case["username"],
        requested_items=refund_case["requested_items"],
        refund_window=30,
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

    for item, details in validation.get(
        "item_details", {}
    ).items():

        print(f"\nItem: {item}")

        for key, value in details.items():
            print(f"  {key}: {value}")

    errors = validation.get("errors", [])

    if errors:
        print("\nPOLICY VIOLATIONS")

        for error in errors:
            print("-", error)

    else:
        print("\nAll refund policy checks passed.")

    return validation


# ==================================================
# 8. MAIN EVALUATION RUNNER
# ==================================================

def main():

    trace_metrics, single_turn_metrics = build_metrics()

    cache_config = CacheConfig(
        use_cache=False,
        write_cache=False,
    )

    async_config = AsyncConfig(
        run_async=False,
    )

    display_config = DisplayConfig(
        inspect_after_run=False,
    )

    test_cases = []
    policy_results = []

    # ----------------------------------------------
    # 8A. AGENT EXECUTION AND TRACE METRICS
    # ----------------------------------------------

    for golden in dataset.evals_iterator(
        metrics=trace_metrics,
        cache_config=cache_config,
        async_config=async_config,
        display_config=display_config,
    ):

        print("\n" + "=" * 70)
        print("EVALUATING:", golden.input)
        print("=" * 70)

        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": golden.input,
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": str(uuid4())
                },
                "callbacks": [CallbackHandler()],
            },
        )

        actual_output = get_actual_output(response)

        tools_called = get_tool_calls(response)

        print("\nACTUAL ANSWER:")
        print(actual_output)

        print("\nTOOLS CALLED:")
        for tool in tools_called:
            print("-", tool.name)

        # ------------------------------------------
        # BUILD SINGLE-TURN TEST CASE
        # ------------------------------------------

        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            expected_output=golden.expected_output,
            tools_called=tools_called,
            expected_tools=golden.expected_tools,
        )

        test_cases.append(test_case)

        # ------------------------------------------
        # DETERMINISTIC POLICY VALIDATION
        # ------------------------------------------

        validation = run_refund_policy_validation(
            golden,
            response
        )

        if validation is not None:
            policy_results.append({
                "input": golden.input,
                "passed": validation.get("passed", False),
                "score": validation.get("score", 0.0),
                "errors": validation.get("errors", []),
            })

    # ----------------------------------------------
    # 8B. SINGLE-TURN DEEPEVAL METRICS
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING SINGLE-TURN METRICS")
    print("=" * 70)

    if test_cases:

        evaluation_results = evaluate(
            test_cases=test_cases,
            metrics=single_turn_metrics,
            cache_config=cache_config,
            async_config=async_config,
            display_config=display_config,
        )

    else:
        evaluation_results = None
        print("No test cases available.")

    # ----------------------------------------------
    # 8C. REFUND POLICY SUMMARY
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print("REFUND POLICY VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(
        1 for item in policy_results
        if item["passed"]
    )

    failed = len(policy_results) - passed

    for item in policy_results:

        status = (
            "PASS"
            if item["passed"]
            else "FAIL"
        )

        print(f"\n{status}: {item['input']}")

        print("Policy Score:", item["score"])

        for error in item["errors"]:
            print(" -", error)

    print("\n" + "=" * 70)
    print("OVERALL EVALUATION SUMMARY")
    print("=" * 70)

    print("Golden test cases:", len(test_cases))
    print("DeepEval metrics configured:", 7)
    print("Policy cases evaluated:", len(policy_results))
    print("Policy cases passed:", passed)
    print("Policy cases failed:", failed)

    print(
        "Other cases may not have deterministic "
        "policy coverage."
    )

    print("\nEvaluation run completed.")

    return evaluation_results


# ==================================================
# 9. ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
