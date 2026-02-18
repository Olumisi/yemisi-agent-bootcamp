"""Multi-Agent Report Orchestrator.

Orchestrates specialized sub-agents to gather information and produce a
structured markdown report.

Follows the ReAct pattern from app.py.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncGenerator

import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage

from src.prompts import REACT_INSTRUCTIONS
from src.utils.client_manager import AsyncClientManager

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionSystemMessageParam,
        ChatCompletionToolParam,
    )

MAX_TURNS = 10  # More turns needed to orchestrate multiple sub-agents

# ---------------------------------------------------------------------------
# Tool definitions – one per sub-agent
# ---------------------------------------------------------------------------

tools: list["ChatCompletionToolParam"] = [
    {
        "type": "function",
        "function": {
            "name": "overview_agent",
            "description": (
                "Sub-agent that retrieves background context and general information "
                "about a topic. Returns a structured summary suitable for the Overview "
                "section of a report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to research for the overview section.",
                    }
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "data_profile_agent",
            "description": (
                "Sub-agent that retrieves quantitative data, statistics, key metrics, "
                "and structured data characteristics about a topic. Returns a data "
                "profile suitable for the Data Profile section of a report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to profile with data and statistics.",
                    }
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Retrieve references on a specific keyword from English Wikipedia. "
                "Use for fact-checking or supplementing sub-agent outputs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword for the Wikipedia search, e.g. 'GPT-4'.",
                    }
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = (
    "You are a Report Orchestrator Agent. Your job is to produce a comprehensive "
    "markdown report for a given topic by coordinating three specialist sub-agents:\n\n"
    "1. **overview_agent** – gathers background context, history, and general information.\n"
    "2. **data_profile_agent** – gathers quantitative data, metrics, and structured statistics.\n"
    "3. **search_wikipedia** – supplements findings with encyclopedic references.\n\n"
    "Workflow:\n"
    "- Always call overview_agent FIRST, then data_profile_agent, then search_wikipedia "
    "if additional facts are needed.\n"
    "- After all tools have run, synthesize findings into a markdown report with EXACTLY "
    "these sections in order:\n\n"
    "```\n"
    "# Report: <Topic>\n\n"
    "## Metadata\n"
    "## Executive Summary\n"
    "## Overview\n"
    "## Data Profile\n"
    "```\n\n"
    "Guidelines for each section:\n"
    "- **Metadata**: topic, report generated timestamp (UTC), agents used, data sources.\n"
    "- **Executive Summary**: 3-5 sentences synthesising the most important insights.\n"
    "- **Overview**: Background, context, history, and significance of the topic.\n"
    "- **Data Profile**: Key metrics, statistics, and quantitative characteristics in a "
    "markdown table where possible.\n\n"
    + REACT_INSTRUCTIONS
)

system_message: "ChatCompletionSystemMessageParam" = {
    "role": "system",
    "content": REPORT_SYSTEM_PROMPT,
}

# ---------------------------------------------------------------------------
# Sub-agent implementations
# ---------------------------------------------------------------------------


async def run_overview_agent(topic: str, client_manager: AsyncClientManager) -> str:
    """Sub-agent: produces an Overview section using the knowledgebase."""
    results = await client_manager.knowledgebase.search_knowledgebase(topic)
    serialized = json.dumps([r.model_dump() for r in results])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an Overview Agent. Given search results, write 2-4 paragraphs "
                "of background context about the topic. Focus on history, significance, "
                "and general understanding. Return plain prose only."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\n\nSearch results:\n{serialized}",
        },
    ]
    completion = await client_manager.openai_client.chat.completions.create(
        model=client_manager.configs.default_worker_model,
        messages=messages,
    )
    return completion.choices[0].message.content or ""


async def run_data_profile_agent(topic: str, client_manager: AsyncClientManager) -> str:
    """Sub-agent: produces a Data Profile section using the knowledgebase."""
    results = await client_manager.knowledgebase.search_knowledgebase(
        f"{topic} statistics data metrics"
    )
    serialized = json.dumps([r.model_dump() for r in results])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Data Profile Agent. Given search results, extract and present "
                "key quantitative data, metrics, and statistics about the topic. "
                "Format as a markdown table where possible, followed by brief commentary. "
                "If no quantitative data is available, provide qualitative characteristics "
                "in structured form."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {topic}\n\nSearch results:\n{serialized}",
        },
    ]
    completion = await client_manager.openai_client.chat.completions.create(
        model=client_manager.configs.default_worker_model,
        messages=messages,
    )
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def report_orchestrator(
    query: str, history: list[ChatMessage]
) -> AsyncGenerator[list[ChatMessage], Any]:
    """Orchestrate sub-agents and produce a structured markdown report."""
    agent_responded = False
    agents_used: list[str] = []
    collected_sections: dict[str, str] = {}

    oai_messages = [system_message, {"role": "user", "content": query}]

    for turn in range(MAX_TURNS):
        completion = await client_manager.openai_client.chat.completions.create(
            model=client_manager.configs.default_planner_model,
            messages=oai_messages,
            tools=tools,
        )

        message = completion.choices[0].message
        oai_messages.append(message)
        tool_calls = message.tool_calls

        # No tool calls → orchestrator has produced the final report
        if tool_calls is None:
            history.append(
                ChatMessage(content=message.content or "", role="assistant")
            )
            agent_responded = True
            yield history
            break

        # Show orchestrator's reasoning step
        if message.content:
            history.append(
                ChatMessage(
                    role="assistant",
                    content=message.content,
                    metadata={"title": "🧠 Orchestrator Thought"},
                )
            )
            yield history

        # Dispatch each tool call to the appropriate sub-agent
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            agents_used.append(fn_name)

            # Show "running agent" indicator
            history.append(
                ChatMessage(
                    role="assistant",
                    content=f"Running **{fn_name}** with `{arguments}`…",
                    metadata={
                        "title": f"🤖 Sub-Agent: `{fn_name}`",
                        "status": "pending",
                    },
                )
            )
            yield history

            # Execute the sub-agent
            if fn_name == "overview_agent":
                result = await run_overview_agent(
                    arguments["topic"], client_manager
                )
                collected_sections["overview"] = result

            elif fn_name == "data_profile_agent":
                result = await run_data_profile_agent(
                    arguments["topic"], client_manager
                )
                collected_sections["data_profile"] = result

            elif fn_name == "search_wikipedia":
                wiki_results = await client_manager.knowledgebase.search_knowledgebase(
                    arguments["keyword"]
                )
                result = json.dumps([r.model_dump() for r in wiki_results])

            else:
                result = f"Unknown tool: {fn_name}"

            # Feed result back to orchestrator
            oai_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

            # Update UI: collapse the pending indicator with result preview
            history[-1] = ChatMessage(
                role="assistant",
                content=f"```\n{result[:300]}{'...' if len(result) > 300 else ''}\n```",
                metadata={
                    "title": f"✅ `{fn_name}` completed",
                    "log": f"Arguments: {arguments}",
                    "status": "done",
                },
            )
            yield history

    # Fallback: if MAX_TURNS exhausted without a final answer
    if not agent_responded:
        timestamp = datetime.now(timezone.utc).isoformat()
        fallback_prompt = (
            "You have reached the maximum reasoning steps. "
            "Using all gathered information, produce the final markdown report now "
            "with sections: Metadata, Executive Summary, Overview, Data Profile."
        )
        oai_messages.append({"role": "system", "content": fallback_prompt})
        completion = await client_manager.openai_client.chat.completions.create(
            model=client_manager.configs.default_planner_model,
            messages=oai_messages,
        )
        message = completion.choices[0].message
        history.append(ChatMessage(content=message.content or "", role="assistant"))
        yield history


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv(verbose=True)

    client_manager = AsyncClientManager()

    demo = gr.ChatInterface(
        report_orchestrator,
        chatbot=gr.Chatbot(height=700, render_markdown=True),
        textbox=gr.Textbox(
            lines=1,
            placeholder="Enter a topic to generate a report (e.g. 'Quantum Computing')",
        ),
        examples=[
            ["Generate a report on Artificial Intelligence"],
            ["Generate a report on Climate Change"],
            ["Generate a report on the Global Semiconductor Industry"],
            ["Produce a full report on the history and data profile of the Internet"],
        ],
        title="📄 Multi-Agent Report Generator",
        description=(
            "Enter any topic. The orchestrator will coordinate specialist sub-agents "
            "(Overview Agent, Data Profile Agent, Wikipedia Search) and synthesize "
            "a structured markdown report."
        ),
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())
