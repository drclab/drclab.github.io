+++
title = "Tool Calling in Agentic Workflow"
date = "2025-10-25T12:00:00+00:00"
type = "post"
draft = false
tags = ["announcement", "agentic_workflow", "LLM"]
categories = ["posts"]
description = "Walk through porting the email assistant workflow onto Google’s Agent ADK with tool registration and Gemini 2.0 examples."
+++

Welcome back — this update dives into how we're approaching **tool calling in agentic workflows** and what it unlocks for the lab. We've been experimenting with giving our agents programmable interfaces so they can pull the right levers (search, run code, update data) without human babysitting.

{{< post-figure src="images/posts/stone.png" alt="Diagram of the tool registry feeding the agent runtime, which loops outcomes back into future tasks." caption="The agent runtime validates tool calls, executes actions, and feeds results back into the registry loop." >}}

## Why Google's Agent ADK

- Ship agent workflows on Google’s Agent Developer Kit (ADK) so we can reuse policy, auth, and telemetry out of the box.
- Keep every tool contract discoverable through the ADK Tool Registry, which mirrors our HuggingFace schemas.
- Let the runtime gatekeep API access with ADK’s native argument validation and cooldown policies.

The rest of this post walks through the notebook and shows how we implement the email-assistant workflow onto the Agent ADK runtime. Everything below runs against the same simulated inbox that the ungraded lab ships with; the only difference is the platform glue.

## 1. Bootstrap the ADK runtime

The lab still starts by loading environment variables and instantiating the client, but we now make the ADK target explicit. The `aisuite` SDK already bundles an opinionated wrapper for Google’s ADK, so migrating was as simple as passing the `platform="google-agent-adk"` flag.

```python
# ================================
# Imports
# ================================
from dotenv import load_dotenv
import aisuite as ai

import json
import utils
import display_functions
import email_tools

# ================================
# Environment & Client
# ================================
load_dotenv()
client = ai.Client(platform="google-agent-adk")
```

Behind the scenes ADK provisions a dedicated agent runtime, handles service account credentials, and exposes observability hooks straight into Cloud Logging. No extra scaffolding required.

## 2. Keep the simulated email backend handy

The utilities from the notebook still work unchanged. We lean on them during local runs to reset the inbox or pull deterministic fixtures when iterating on tool contracts.

```python
new_email = email_tools.send_email("test@example.com", "Lunch plans", "Shall we meet at noon?")
email_tools.get_email(new_email["id"])
# email_tools.list_all_emails()
# email_tools.search_emails("lunch")
# email_tools.reset_database()
```

These helpers give us fast feedback before we hand control to the agent runtime.

## 3. Register tools with Agent ADK

With ADK, every callable is declared up front. We mirror the notebook’s tool surface—searching, marking read, sending, and deleting mail—but now the registry is managed by the platform.

```python
TOOLS = [
    email_tools.search_unread_from_sender,
    email_tools.list_unread_emails,
    email_tools.search_emails,
    email_tools.get_email,
    email_tools.mark_email_as_read,
    email_tools.send_email,
    email_tools.delete_email,
]

agent = client.agents.register(
    display_name="InboxOps",
    instructions="Manage email end-to-end without human confirmation.",
    tools=TOOLS,
)
```

Each function carries the docstring and JSON schema that ADK uses for argument validation. The runtime blocks malformed invocations before they hit the simulated backend, which keeps the lab deterministic.

## 4. Shape the prompt the same way

Prompt construction stays identical. We wrap the user request in a system-style preamble so the agent knows it can act autonomously. This snippet is a direct lift from the notebook.

```python
def build_prompt(request_: str) -> str:
    return f"""
    - You are an AI assistant specialized in managing emails.
    - You can perform various actions such as listing, searching, filtering, and manipulating emails.
    - Use the provided tools to interact with the email system.
    - Never ask the user for confirmation before performing an action.
    - If needed, my email address is "you@email.com" so you can use it to send emails or perform actions related to my account.
    {request_.strip()}
    """
```

You can preview the wrapped prompt exactly like in the notebook by triggering `utils.print_html`.

## 5. Run the scenario through ADK

Switching to ADK doesn’t impact the high-level flow: we send a request, ADK routes tool calls, and we inspect the trace.

```python
prompt_ = build_prompt("Check for unread emails from boss@email.com, mark them as read, and send a polite follow-up.")

session = client.chat.completions.create(
    agent=agent,
    model="google/generativeai/gemini-2.0-pro-exp",
    messages=[{"role": "user", "content": prompt_}],
    max_turns=5,
)

display_functions.pretty_print_chat_completion(session)
```

Notice the `model` swap: we now default to the Gemini 2.0 experimental checkpoints that ship with ADK. Tool traces stream in alongside the text response, which makes verification instantaneous.

## 6. Spot the missing tool (and fix it)

The notebook highlights what happens when `delete_email` is absent. ADK’s validator surfaces the same failure mode with a structured error, so the agent gracefully degrades instead of hallucinating success:

```python
prompt_ = build_prompt("Delete alice@work.com email")
client.chat.completions.create(
    agent=agent.with_tools(TOOLS[:-1]),  # drop delete_email
    model="google/generativeai/gemini-2.0-pro-exp",
    messages=[{"role": "user", "content": prompt_}],
    max_turns=3,
)
```

Re-enable the tool and the agent completes the task end-to-end:

```python
prompt_ = build_prompt("Delete the happy hour email")
client.chat.completions.create(
    agent=agent,
    model="google/generativeai/gemini-2.0-pro-exp",
    messages=[{"role": "user", "content": prompt_}],
    max_turns=3,
)
```

## 7. Takeaways

- The lab’s email assistant ports cleanly onto Google’s Agent ADK with only minor client changes.
- ADK’s registry enforces tool contracts, preventing silent failures like the missing `delete_email` call.
- Using Gemini 2.0 inside ADK lets the agent chain multiple email operations without extra glue code.
- The simulated backend plus `utils.reset_database()` keeps experiments reproducible as you iterate on tool coverage.
