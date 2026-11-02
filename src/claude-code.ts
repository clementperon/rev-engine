// Runs a review pass in Claude Code (`claude -p`) instead of through the chat
// API, so a Claude Code login (CLAUDE_CODE_OAUTH_TOKEN) can drive the review.
//
// Claude Code runs the tool loop. It gets none of its own tools: it reads the
// checkout only through this reviewer's read tools, served to it over MCP from
// this process, so the same confinement and read tracking apply. The submit
// tool becomes Claude Code's structured output.
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentOptions } from "./agent.ts";
import type { Chat } from "./llm.ts";
import { readTools } from "./tools.ts";

const SERVER = "rev";
const SETTINGS = Symbol("claude-code");

export interface ClaudeCodeSettings {
  effort: string | null;
}

/** A Chat that marks the run for Claude Code; runAgent never calls it. */
export function claudeCodeChat(settings: ClaudeCodeSettings): Chat {
  const chat: Chat = async () => {
    throw new Error("the claude-code engine runs only through the tool loop");
  };
  return Object.assign(chat, { [SETTINGS]: settings });
}

export function claudeCodeSettings(chat: Chat): ClaudeCodeSettings | null {
  return (chat as Chat & { [SETTINGS]?: ClaudeCodeSettings })[SETTINGS] ?? null;
}

/** anthropic/claude-opus-5.5 becomes claude-opus-5-5; aliases such as opus pass through. */
export function claudeModel(model: string): string {
  return model.replace(/^anthropic\//, "").replace(/(\d)\.(\d)/g, "$1-$2");
}

function toolNote(submitName: string): string {
  return `

## Tools in this session

Your tools are named mcp__${SERVER}__<name>. Where these instructions name a tool, such as read_file, use mcp__${SERVER}__read_file. Where they say to call ${submitName}, give that answer as your structured output instead.`;
}

async function body(request: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of request) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

function reply(response: ServerResponse, status: number, value?: unknown): void {
  response.writeHead(status, value === undefined ? {} : { "content-type": "application/json" });
  response.end(value === undefined ? undefined : JSON.stringify(value));
}

/** A streamable-HTTP MCP server on a loopback port that answers with the workspace's read tools. */
async function serveTools(options: AgentOptions, who: string): Promise<{ url: string; close: () => void }> {
  const path = `/${randomUUID()}`;
  const tools = readTools.map((tool) => ({ name: tool.function.name, description: tool.function.description, inputSchema: tool.function.parameters }));
  const server = createServer(async (request, response) => {
    if (request.url !== path) return reply(response, 404);
    if (request.method !== "POST") return reply(response, 405);
    let message: { id?: unknown; method?: string; params?: Record<string, unknown> };
    try {
      message = JSON.parse(await body(request));
    } catch {
      return reply(response, 400);
    }
    if (message.id === undefined) return reply(response, 202);
    const answer = (result: unknown) => reply(response, 200, { jsonrpc: "2.0", id: message.id, result });
    switch (message.method) {
      case "initialize":
        return answer({
          protocolVersion: message.params?.protocolVersion ?? "2025-06-18",
          capabilities: { tools: {} },
          serverInfo: { name: SERVER, version: "1" },
        });
      case "ping":
        return answer({});
      case "tools/list":
        return answer({ tools });
      case "tools/call": {
        const name = String(message.params?.name ?? "");
        const args = JSON.stringify(message.params?.arguments ?? {});
        options.log(`${who}: ${`${name} ${String((message.params?.arguments as Record<string, unknown>)?.path ?? "")}`.trim().slice(0, 80)}`);
        return answer({ content: [{ type: "text", text: await options.workspace.call(name, args) }] });
      }
      default:
        return reply(response, 200, { jsonrpc: "2.0", id: message.id, error: { code: -32601, message: "method not found" } });
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { url: `http://127.0.0.1:${port}${path}`, close: () => server.close() };
}

interface Result {
  session_id?: string;
  is_error?: boolean;
  subtype?: string;
  num_turns?: number;
  total_cost_usd?: number;
  structured_output?: unknown;
  result?: string;
}

function runClaude(args: string[], cwd: string, timeoutMs: number): Promise<Result> {
  return new Promise((resolve, reject) => {
    const child = spawn("claude", args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    const timer = setTimeout(() => child.kill("SIGTERM"), Math.max(timeoutMs, 1_000));
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      if (signal) return reject(new Error(`claude was stopped by ${signal}, usually the time budget`));
      try {
        resolve(JSON.parse(stdout) as Result);
      } catch {
        reject(new Error(`claude exited with code ${code}: ${(stderr || stdout).trim().slice(0, 300)}`));
      }
    });
  });
}

export async function runClaudeCode(options: AgentOptions, settings: ClaudeCodeSettings): Promise<Record<string, unknown>> {
  const { model, submit, deadline, maxTurns, secondLook, log } = options;
  const who = options.caller ?? model;
  const tools = await serveTools(options, who);
  // An empty directory, so Claude Code finds no CLAUDE.md or settings from the checkout.
  const cwd = await mkdtemp(join(tmpdir(), "rev-claude-"));
  const submissions: Array<Record<string, unknown>> = [];
  try {
    const common = [
      "-p", "--output-format", "json",
      "--model", claudeModel(model),
      "--tools", "",
      "--strict-mcp-config", "--mcp-config", JSON.stringify({ mcpServers: { [SERVER]: { type: "http", url: tools.url } } }),
      "--allowedTools", `mcp__${SERVER}`,
      "--setting-sources", "",
      "--json-schema", JSON.stringify(submit.function.parameters),
      "--max-turns", String(maxTurns),
      ...(settings.effort ? ["--effort", settings.effort] : []),
    ];
    let args = [...common, "--system-prompt", options.system + toolNote(submit.function.name), options.prompt];
    for (let round = 1; ; round += 1) {
      const result = await runClaude(args, cwd, deadline - Date.now());
      log(`${who}: round ${round}: ${result.subtype ?? "done"} after ${result.num_turns ?? "?"} turn(s), $${(result.total_cost_usd ?? 0).toFixed(4)} at API prices`);
      const value = result.structured_output;
      if (result.is_error || !value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error(`${who}: claude returned no ${submit.function.name} output (${result.subtype ?? "error"}): ${String(result.result ?? "").slice(0, 300)}`);
      }
      submissions.push(value as Record<string, unknown>);
      const next = secondLook && result.session_id && deadline - Date.now() > 200_000 ? secondLook.message(submissions.length) : null;
      if (!next) break;
      args = [...common, "--resume", result.session_id!, next];
    }
  } catch (error) {
    if (!submissions.length) throw error;
    log(`${who}: a later round failed, keeping the submissions so far: ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    tools.close();
    await rm(cwd, { recursive: true, force: true });
  }
  return secondLook && submissions.length > 1 ? secondLook.combine(submissions) : submissions[0]!;
}
