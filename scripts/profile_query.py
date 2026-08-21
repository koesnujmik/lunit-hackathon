import argparse
import asyncio
import json
import logging
import time

from l2_baseline.harness import L2Harness
from l2_baseline.mcp_client import LunitMCPClient
from l2_baseline.ranking import rank_tool_candidates


async def inspect_candidates(
    query: str,
    test_single_selector: bool = False,
    test_auto_selector: bool = False,
) -> None:
    harness = L2Harness()
    assessment_started = time.perf_counter()
    assessment = await harness._assess_retrieval_query(query)
    print(
        "QUERY_SUFFICIENCY duration_sec=",
        f"{time.perf_counter() - assessment_started:.3f}",
    )
    print("QUERY_SUFFICIENT=", assessment.query_sufficient)
    print("QUERY_ASSESSMENT_REASON=", assessment.reason)

    rationale = ""
    if not assessment.query_sufficient:
        rationale_started = time.perf_counter()
        rationale = await harness.create_retrieval_rationale(query)
        print(
            "RETRIEVAL_RATIONALE duration_sec=",
            f"{time.perf_counter() - rationale_started:.3f}",
        )
    print("RETRIEVAL_RATIONALE_USED=", bool(rationale))
    print(f"RETRIEVAL_RATIONALE_WORDS={len(rationale.split())}")

    async with LunitMCPClient(
        harness.settings.mcp_url,
        harness.settings.token,
        harness.settings.request_timeout_sec,
    ) as mcp:
        tools = await mcp.openai_tools()
    candidates = rank_tool_candidates(
        query,
        tools,
        harness.settings.tool_candidate_limit,
        rationale=rationale,
    )
    print(f"TOOL_SCHEMA_BM25_TOP_K={len(candidates)}")
    for tool in candidates:
        print(
            "CANDIDATE",
            tool["function"]["name"],
            "schema_chars=",
            len(json.dumps(tool, ensure_ascii=False)),
        )
    if test_single_selector or test_auto_selector:
        if test_single_selector:
            started = time.perf_counter()
            try:
                actions = await harness._choose_actions(
                    candidates[:1], query, rationale
                )
            finally:
                print(
                    "SINGLE_SELECTOR duration_sec=",
                    f"{time.perf_counter() - started:.3f}",
                )
            print("SINGLE_SELECTOR actions=", [call.function.name for call in actions])
        if test_auto_selector:
            started = time.perf_counter()
            try:
                actions = await harness._choose_actions(candidates, query, rationale)
            finally:
                print(
                    "BATCH_SELECTOR duration_sec=",
                    f"{time.perf_counter() - started:.3f}",
                )
            print(
                "BATCH_SELECTOR actions=",
                [call.function.name for call in actions[:5]],
            )


async def profile(query: str) -> None:
    started = time.perf_counter()
    answer = await L2Harness().chat([{"role": "user", "content": query}])
    print(f"PROFILE total_duration_sec={time.perf_counter() - started:.3f}")
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--inspect-candidates", action="store_true")
    parser.add_argument("--test-single-selector", action="store_true")
    parser.add_argument("--test-auto-selector", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.inspect_candidates:
        asyncio.run(
            inspect_candidates(
                args.query,
                args.test_single_selector,
                args.test_auto_selector,
            )
        )
    else:
        asyncio.run(profile(args.query))


if __name__ == "__main__":
    main()
