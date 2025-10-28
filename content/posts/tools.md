+++
title = "MCP 101"
date = "2025-10-04T12:00:00+00:00"
type = "post"
draft = false
tags = ["guide", "mcp", "agentic_workflow"]
categories = ["posts"]
description = "Primer on the Model Context Protocol (MCP), why it matters, and how to try it inside the lab’s agent stack."
+++

Curious about the **Model Context Protocol (MCP)** and what it unlocks for autonomous agents? This primer covers the core ideas, how it differs from ad-hoc tool APIs, and the first steps to integrate it into our lab workflows.

## What Is MCP?

- Shared protocol that lets large language models discover, describe, and call tools in a predictable way.
- Standardizes the contract between the model runtime and external capabilities such as vector stores, automations, and databases.
- Built to keep the transport layer agnostic—use HTTP, WebSockets, or a message bus without rewriting your agent.

At its core, MCP decouples model prompting from tool orchestration. Instead of hard-coding Python functions or SDK calls, you expose capabilities through MCP schemas that any compliant agent can understand.

## Protocol Building Blocks

- **Resources**: Long-lived data the agent can inspect (e.g., documents, calendars). They are addressable and include metadata for filtering.
- **Prompts**: Reusable prompt templates exposed over MCP so agents can parameterize them at runtime.
- **Tools**: The actionable endpoints. Each describes accepted arguments, expected return types, and failure shapes.
- **Events**: Optional streaming channel that surfaces state changes (tool completed, resource updated) back to the agent.

These primitives share the same schema language, which keeps discovery and validation consistent across the stack.

## Why the Lab Cares

- **Portability**: We can swap between hosted LLM providers or local inference without re-plumbing tool wiring.
- **Governance**: Every tool call is schema-validated, logged, and traceable—key for compliance reviews.
- **Velocity**: New automations only require an MCP descriptor, not a bespoke SDK integration.
- **Collaboration**: Teams can publish reusable MCP manifests so agents across projects share the same capabilities.

## MCP vs. Traditional Tool Calling

| Topic | Traditional Implementations | MCP Approach |
| ----- | --------------------------- | ------------ |
| Discovery | Hard-coded in prompts or code | Agents fetch manifests at runtime |
| Validation | Custom argument checks per tool | JSON schema enforced by the protocol |
| Transport | Tied to the client SDK | Pluggable (HTTP, gRPC, message queues) |
| Observability | Often bespoke logging | Standard event stream with typed payloads |

Adopting MCP turns tool calling into a contract-driven interface rather than a pile of glue code.

## Getting Started in the Lab

1. **Model a capability** – Describe inputs, outputs, and error cases for the automation you want to expose.
2. **Author a manifest** – Use the MCP schema (YAML or JSON) to publish the tool, resource, or prompt.
3. **Host the server** – Most teams start with a lightweight FastAPI or Node service that serves descriptors and handles requests.
4. **Register with the agent** – Point the agent runtime at the MCP endpoint; it will negotiate capabilities during startup.
5. **Test flows** – Run through happy paths and failure modes with `hugo server` to confirm prompts render correctly and logs capture MCP events.

Keep the manifests versioned in `data/mcp/` so updates ship alongside code reviews.

## Tooling Tips

- Use the OpenAPI-to-MCP converter when wrapping existing REST automations; it generates starter manifests with schemas populated.
- Leverage the protocol’s `callTool` simulation mode to check payloads before running destructive operations.
- Record traces in our observability stack—MCP payloads are JSON, so dashboards and alerting remain straightforward.

## Roadmap

- Bundle starter manifests for our most-used research automations.
- Add a Hugo shortcode to render MCP manifests for documentation.
- Pilot an MCP gateway that proxies to legacy tools while we migrate natively.

Have a workflow that could benefit from MCP? Drop a note in the #agent-infra channel or open an issue with your proposed manifest so we can help wire it up.
