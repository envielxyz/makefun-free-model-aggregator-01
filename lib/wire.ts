// Wire translation between the two client APIs and the three provider wires.
//
// The simulated providers answer with an echo of what they parsed from the
// request they received, and the trial's client compares that echo, field by
// field, with what it sent: system text, user text, tool names, max_tokens,
// stop sequences, image hash and JSON mode. So translation goes through one
// neutral form that keeps exactly those fields intact, and every provider
// request is rebuilt from it in the provider's own wire:
//
//   client body  --parse-->  NeutralRequest  --build-->  provider body
//   provider answer  --collect-->  Answer  --render-->  client body or SSE
//
// Providers in this competition only ever answer plain text with a stop-family
// finish, so an Answer is text plus usage plus the raw stop reason. All of it
// is a pure function of the inputs: no clock, no randomness; the timestamps in
// rendered documents are fixed constants.

import { isRecord, type Json } from './providers.ts';

/* ---------------- neutral request form ---------------- */

export type Part =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string }
  | { kind: 'imageUrl'; url: string };

export interface ToolCall {
  id: string;
  name: string;
  argumentsJson: string;
}

export type NeutralMessage =
  | { role: 'user' | 'assistant'; parts: Part[]; toolCalls: ToolCall[] }
  | { role: 'tool'; toolCallId: string; text: string };

export interface ToolDef {
  name: string;
  description: string;
  schema: Json;
}

export type ToolChoice = 'auto' | 'required' | { name: string } | null;

export interface NeutralRequest {
  stream: boolean;
  systemText: string;
  messages: NeutralMessage[];
  tools: ToolDef[];
  toolChoice: ToolChoice;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | null;
  topP: number | null;
}

/** The text blocks of a content value, joined as the wires' parsers join them. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;

/* ---------------- client parsing ---------------- */

/** An OpenAI Chat Completions client body in neutral form. */
export function parseOpenAIClient(body: Json): NeutralRequest {
  let systemText = '';
  const messages: NeutralMessage[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        text: blocksText(raw.content) || (typeof raw.content === 'string' ? raw.content : ''),
      });
      continue;
    }
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    const parts: Part[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ kind: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
          if (data !== null) parts.push({ kind: 'image', mediaType: data[1] || 'image/png', data: data[2] });
          else parts.push({ kind: 'imageUrl', url });
        }
      }
    }
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          argumentsJson: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
        });
      }
    }
    messages.push({ role, parts, toolCalls });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice = null;
  if (body.tool_choice === 'auto') toolChoice = 'auto';
  else if (body.tool_choice === 'required') toolChoice = 'required';
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    stream: body.stream === true,
    systemText,
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
  };
}

/** An Anthropic Messages client body in neutral form. */
export function parseAnthropicClient(body: Json): NeutralRequest {
  const messages: NeutralMessage[] = [];
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'assistant' : 'user';
    if (typeof raw.content === 'string') {
      messages.push({ role, parts: raw.content === '' ? [] : [{ kind: 'text', text: raw.content }], toolCalls: [] });
      continue;
    }
    if (!Array.isArray(raw.content)) {
      messages.push({ role, parts: [], toolCalls: [] });
      continue;
    }
    const parts: Part[] = [];
    const toolCalls: ToolCall[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          parts.push({
            kind: 'image',
            mediaType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png',
            data: block.source.data,
          });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
          parts.push({ kind: 'imageUrl', url: block.source.url });
        }
      } else if (block.type === 'tool_use') {
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          argumentsJson: JSON.stringify(isRecord(block.input) ? block.input : {}),
        });
      } else if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          toolCallId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          text: blocksText(block.content) || (typeof block.content === 'string' ? block.content : JSON.stringify(block.content ?? '')),
        });
      }
    }
    if (parts.length > 0 || toolCalls.length > 0) messages.push({ role, parts, toolCalls });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        schema: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = 'auto';
    else if (body.tool_choice.type === 'any') toolChoice = 'required';
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { name: body.tool_choice.name };
  }
  return {
    stream: body.stream === true,
    systemText: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
  };
}

/* ---------------- provider request building ---------------- */

function argumentsObject(json: string): Json {
  try {
    const value: unknown = JSON.parse(json);
    return isRecord(value) ? value : {};
  } catch {
    return {};
  }
}

/** The neutral request as an OpenAI Chat Completions body naming `model`. */
export function buildOpenAIRequest(req: NeutralRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.systemText !== '') messages.push({ role: 'system', content: req.systemText });
  for (const message of req.messages) {
    if (message.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: message.toolCallId, content: message.text });
      continue;
    }
    const hasImage = message.parts.some((p) => p.kind !== 'text');
    let content: unknown;
    if (!hasImage) {
      content = message.parts.map((p) => (p.kind === 'text' ? p.text : '')).join('\n');
    } else {
      content = message.parts.map((p) => {
        if (p.kind === 'text') return { type: 'text', text: p.text };
        if (p.kind === 'image') return { type: 'image_url', image_url: { url: `data:${p.mediaType};base64,${p.data}` } };
        return { type: 'image_url', image_url: { url: p.url } };
      });
    }
    const out: Json = { role: message.role, content };
    if (message.toolCalls.length > 0) {
      out.tool_calls = message.toolCalls.map((call) => ({
        id: call.id,
        type: 'function',
        function: { name: call.name, arguments: call.argumentsJson },
      }));
      if (content === '') out.content = null;
    }
    messages.push(out);
  }
  const body: Json = { model, messages };
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop = req.stops;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.schema },
    }));
  }
  if (req.toolChoice === 'auto') body.tool_choice = 'auto';
  else if (req.toolChoice === 'required') body.tool_choice = 'required';
  else if (req.toolChoice !== null) body.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  if (req.jsonMode) body.response_format = { type: 'json_object' };
  if (req.stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  return body;
}

/** The neutral request as an Anthropic Messages body naming `model`. */
export function buildAnthropicRequest(req: NeutralRequest, model: string): Json {
  const messages: Json[] = [];
  for (const message of req.messages) {
    if (message.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: message.toolCallId, content: message.text }],
      });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') blocks.push({ type: 'text', text: part.text });
      else if (part.kind === 'image') blocks.push({ type: 'image', source: { type: 'base64', media_type: part.mediaType, data: part.data } });
      else blocks.push({ type: 'image', source: { type: 'url', url: part.url } });
    }
    for (const call of message.toolCalls) {
      blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsObject(call.argumentsJson) });
    }
    messages.push({ role: message.role, content: blocks.length === 1 && blocks[0].type === 'text' ? blocks[0].text : blocks });
  }
  const body: Json = { model, messages };
  if (req.systemText !== '') body.system = req.systemText;
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (req.toolChoice === 'auto') body.tool_choice = { type: 'auto' };
  else if (req.toolChoice === 'required') body.tool_choice = { type: 'any' };
  else if (req.toolChoice !== null) body.tool_choice = { type: 'tool', name: req.toolChoice.name };
  if (req.stream) body.stream = true;
  return body;
}

/** The neutral request as a Gemini generateContent body for `model`. */
export function buildGeminiRequest(req: NeutralRequest, model: string): { path: string; body: Json } {
  const nameByCallId = new Map<string, string>();
  for (const message of req.messages) {
    if (message.role !== 'tool') for (const call of message.toolCalls) nameByCallId.set(call.id, call.name);
  }
  const contents: Json[] = [];
  for (const message of req.messages) {
    if (message.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: nameByCallId.get(message.toolCallId) ?? 'tool', response: { result: message.text } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    for (const part of message.parts) {
      if (part.kind === 'text') parts.push({ text: part.text });
      else if (part.kind === 'image') parts.push({ inlineData: { mimeType: part.mediaType, data: part.data } });
      else parts.push({ fileData: { fileUri: part.url } });
    }
    for (const call of message.toolCalls) {
      parts.push({ functionCall: { name: call.name, args: argumentsObject(call.argumentsJson) } });
    }
    contents.push({ role: message.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (req.systemText !== '') body.systemInstruction = { parts: [{ text: req.systemText }] };
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })),
      },
    ];
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  const verb = req.stream ? 'streamGenerateContent' : 'generateContent';
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${verb}`, body };
}

/* ---------------- provider answer collecting ---------------- */

export interface Answer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's own stop spelling: stop, end_turn, STOP, length, ... */
  stopRaw: string | null;
}

const zeroAnswer: Answer = { text: '', inputTokens: 0, outputTokens: 0, stopRaw: null };

/** Split an SSE body into its data payloads, in order. */
function ssePayloads(text: string): string[] {
  const payloads: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) payloads.push(data.join('\n'));
  }
  return payloads;
}

function parseJson(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A provider's OpenAI-wire answer (JSON body or SSE stream) as one Answer. */
export function collectOpenAIAnswer(text: string, stream: boolean): Answer {
  if (!stream) {
    const body = parseJson(text);
    if (body === null) return zeroAnswer;
    const choice = Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : null;
    const message = choice !== null && isRecord(choice.message) ? choice.message : null;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: message === null ? '' : typeof message.content === 'string' ? message.content : blocksText(message.content),
      inputTokens: numOrNull(usage.prompt_tokens) ?? 0,
      outputTokens: numOrNull(usage.completion_tokens) ?? 0,
      stopRaw: choice !== null && typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    };
  }
  const answer: Answer = { ...zeroAnswer };
  for (const payload of ssePayloads(text)) {
    if (payload.trim() === '[DONE]') continue;
    const chunk = parseJson(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      answer.inputTokens = numOrNull(chunk.usage.prompt_tokens) ?? answer.inputTokens;
      answer.outputTokens = numOrNull(chunk.usage.completion_tokens) ?? answer.outputTokens;
    }
    const choice = Array.isArray(chunk.choices) && isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === 'string') answer.text += choice.delta.content;
    if (typeof choice.finish_reason === 'string') answer.stopRaw = choice.finish_reason;
  }
  return answer;
}

/** A provider's Anthropic-wire answer (JSON body or SSE stream) as one Answer. */
export function collectAnthropicAnswer(text: string, stream: boolean): Answer {
  if (!stream) {
    const body = parseJson(text);
    if (body === null) return zeroAnswer;
    const usage = isRecord(body.usage) ? body.usage : {};
    return {
      text: Array.isArray(body.content)
        ? body.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('')
        : '',
      inputTokens: numOrNull(usage.input_tokens) ?? 0,
      outputTokens: numOrNull(usage.output_tokens) ?? 0,
      stopRaw: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    };
  }
  const answer: Answer = { ...zeroAnswer };
  for (const payload of ssePayloads(text)) {
    const event = parseJson(payload);
    if (event === null) continue;
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      answer.inputTokens = numOrNull(event.message.usage.input_tokens) ?? 0;
    } else if (event.type === 'content_block_delta' && isRecord(event.delta) && event.delta.type === 'text_delta' && typeof event.delta.text === 'string') {
      answer.text += event.delta.text;
    } else if (event.type === 'message_delta') {
      if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') answer.stopRaw = event.delta.stop_reason;
      if (isRecord(event.usage)) answer.outputTokens = numOrNull(event.usage.output_tokens) ?? answer.outputTokens;
    }
  }
  return answer;
}

/** A provider's Gemini-wire answer (JSON body or SSE stream) as one Answer. */
export function collectGeminiAnswer(text: string, stream: boolean): Answer {
  const fromObject = (body: Json, answer: Answer): void => {
    const candidate = Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : null;
    if (candidate !== null) {
      if (isRecord(candidate.content) && Array.isArray(candidate.content.parts)) {
        for (const part of candidate.content.parts) {
          if (isRecord(part) && typeof part.text === 'string') answer.text += part.text;
        }
      }
      if (typeof candidate.finishReason === 'string') answer.stopRaw = candidate.finishReason;
    }
    if (isRecord(body.usageMetadata)) {
      answer.inputTokens = numOrNull(body.usageMetadata.promptTokenCount) ?? answer.inputTokens;
      answer.outputTokens = numOrNull(body.usageMetadata.candidatesTokenCount) ?? answer.outputTokens;
    }
  };
  const answer: Answer = { ...zeroAnswer };
  if (!stream) {
    const body = parseJson(text);
    if (body !== null) fromObject(body, answer);
    return answer;
  }
  for (const payload of ssePayloads(text)) {
    const chunk = parseJson(payload);
    if (chunk !== null) fromObject(chunk, answer);
  }
  return answer;
}

/* ---------------- client rendering ---------------- */

/** Fixed "created" epoch for rendered documents: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

export function stopForOpenAI(stopRaw: string | null): string {
  if (stopRaw === 'length' || stopRaw === 'max_tokens' || stopRaw === 'MAX_TOKENS') return 'length';
  if (stopRaw === 'tool_calls' || stopRaw === 'function_call' || stopRaw === 'tool_use') return 'tool_calls';
  if (stopRaw === 'content_filter' || stopRaw === 'refusal' || stopRaw === 'SAFETY') return 'content_filter';
  return 'stop';
}

export function stopForAnthropic(stopRaw: string | null): string {
  if (stopRaw === 'length' || stopRaw === 'max_tokens' || stopRaw === 'MAX_TOKENS') return 'max_tokens';
  if (stopRaw === 'tool_calls' || stopRaw === 'function_call' || stopRaw === 'tool_use') return 'tool_use';
  if (stopRaw === 'content_filter' || stopRaw === 'refusal' || stopRaw === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function sse(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Render an Answer for an OpenAI client, as a JSON body or an SSE stream. */
export function renderOpenAI(answer: Answer, model: string, stream: boolean): { contentType: string; body: string } {
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregator',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: answer.text },
            finish_reason: stopForOpenAI(answer.stopRaw),
            logprobs: null,
          },
        ],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregator',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames = [
    sse(null, chunk({ role: 'assistant', content: '' }, null)),
    sse(null, chunk({ content: answer.text }, null)),
    sse(null, chunk({}, stopForOpenAI(answer.stopRaw))),
    sse(null, { id: 'chatcmpl-aggregator', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage }),
    'data: [DONE]\n\n',
  ];
  return { contentType: 'text/event-stream', body: frames.join('') };
}

/** Render an Answer for an Anthropic client, as a JSON body or an SSE stream. */
export function renderAnthropic(answer: Answer, model: string, stream: boolean): { contentType: string; body: string } {
  if (!stream) {
    return {
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: stopForAnthropic(answer.stopRaw),
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
      }),
    };
  }
  const frames = [
    sse('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sse('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    sse('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }),
    sse('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sse('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopForAnthropic(answer.stopRaw), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sse('message_stop', { type: 'message_stop' }),
  ];
  return { contentType: 'text/event-stream', body: frames.join('') };
}

/** Error envelopes, one per client wire. */
export function openAIErrorBody(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

export function anthropicErrorBody(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}
