import argparse
import asyncio
import json
import logging
import time

from openai import AsyncOpenAI

from l2_baseline.harness import L2Harness
from l2_baseline.mcp_client import LunitMCPClient
from l2_baseline.prompts import TOOL_SELECTOR_SYSTEM_PROMPT
from l2_baseline.ranking import rank_tool_candidates


async def inspect_candidates(
    query: str,
    test_single_selector: bool = False,
    test_auto_selector: bool = False,
) -> None:
    harness = L2Harness()
    passage = await harness.create_hypothetical_passage(query)
    async with LunitMCPClient(
        harness.settings.mcp_url,
        harness.settings.token,
        harness.settings.request_timeout_sec,
    ) as mcp:
        tools = await mcp.openai_tools()
    candidates = rank_tool_candidates(
        f"{query}\n{passage}", tools, harness.settings.tool_candidate_limit
    )
    print(f"HYDE_WORDS={len(passage.split())}")
    for tool in candidates:
        print(
            "CANDIDATE",
            tool["function"]["name"],
            "schema_chars=",
            len(json.dumps(tool, ensure_ascii=False)),
        )
    if test_single_selector or test_auto_selector:
        daily_med_tool = next(
            tool
            for tool in candidates
            if tool["function"]["name"] == "adr_retrieve_drug_info"
        )
        async with AsyncOpenAI(
            api_key=harness.settings.token,
            base_url=harness.settings.api_url.rstrip("/") + "/v1",
            timeout=45,
            max_retries=0,
        ) as client:
            if test_single_selector:
                test_harness = L2Harness(settings=harness.settings, client=client)
                started = time.perf_counter()
                try:
                    actions = await test_harness._choose_actions(
                        [daily_med_tool], query, passage
                    )
                finally:
                    print(
                        "SINGLE_SELECTOR duration_sec=",
                        f"{time.perf_counter() - started:.3f}",
                    )
                print("SINGLE_SELECTOR actions=", [call.function.name for call in actions])
            if test_auto_selector:
                started = time.perf_counter()
                response = await client.chat.completions.create(
                    model=harness.settings.model,
                    messages=[
                        {"role": "system", "content": TOOL_SELECTOR_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"QUERY:\n{query}\n\nHYPOTHETICAL PASSAGE:\n{passage}",
                        },
                    ],
                    tools=[daily_med_tool],
                    tool_choice="auto",
                    temperature=0,
                )
                print(
                    "AUTO_SELECTOR duration_sec=",
                    f"{time.perf_counter() - started:.3f}",
                )
                calls = response.choices[0].message.tool_calls or []
                print("AUTO_SELECTOR actions=", [call.function.name for call in calls])


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
