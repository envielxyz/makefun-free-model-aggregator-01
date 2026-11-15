// The best free AI model aggregator: one server, two client wires, three
// provider wires, and a router that spends each provider's free-tier budget
// where it buys the most quality.
//
// How it works:
//   1. Read the brief ($MAKEFUN_PROMPT_FILE): provider directory, quality
//      table, workload manifest.
//   2. Probe every provider's catalog route (catalogue calls are free: they
//      spend none of a provider's inference budget) and build the one catalog
//      document that --list prints and /register posts, byte for byte.
//   3. Listen on an ephemeral port, POST /register to $MAKEFUN_ENV_URL, then
//      answer the client's twenty requests one at a time. Requests arrive in
//      manifest order, so the manifest row at the arrival index states what
//      the arriving request requires.
//   4. For each request, walk the free models that meet its requirements in
//      rating order and call the first whose provider has not been failing;
//      a 402/429/5xx (or a dead connection) moves to the next candidate. A
//      provider that refuses twice in a row is set aside for the rest of the
//      trial. A refusal whose own error message says the budget is spent or
//      the provider is out is permanent: it is not retried, and it sets the
//      provider aside immediately, because no later call to it can succeed.
//   5. Poll GET /work and exit 0 the moment the env says the workload is done.
//
// Everything is a function of the brief, the requests and the providers'
// answers: no wall clock, no randomness, so a replay of the trial walks the
// same routes and leaves the same state.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isFreeModel,
  isRecord,
  joinBase,
  meetsRequirements,
  probeProvider,
  ratingRow,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type ProbeResult,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicErrorBody,
  buildAnthropicRequest,
  buildGeminiRequest,
  buildOpenAIRequest,
  collectAnthropicAnswer,
  collectGeminiAnswer,
  collectOpenAIAnswer,
  openAIErrorBody,
  parseAnthropicClient,
  parseOpenAIClient,
  renderAnthropic,
  renderOpenAI,
  type Answer,
  type NeutralRequest,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Provider calls per client request before giving up and answering an error. */
const MAX_ATTEMPTS = 16;
/** Consecutive refusals after which a provider is set aside for the trial. */
const DEAD_AFTER = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: ProbeResult[] = [];
let catalog: Catalog = { providers: [] };
/** Client requests answered so far: the manifest is indexed by arrival. */
let arrivalIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  catalog = buildCatalog(brief, probes);
}

/* ---------------- routing ---------------- */

interface Candidate {
  probe: ProbeResult;
  modelId: string;
  ratingMilli: number;
}

/** Per-provider call history for this trial: attempts and refusals in a row. */
interface ProviderHealth {
  attempts: number;
  consecutiveRefusals: number;
  /**
   * The provider itself has said, in a refusal's error message, that it
   * cannot serve again this trial: its free-tier budget is spent, or it is
   * out. Such a refusal is permanent — no later call to it can succeed.
   */
  dead: boolean;
}

const health = new Map<string, ProviderHealth>();

function healthOf(providerId: string): ProviderHealth {
  let entry = health.get(providerId);
  if (entry === undefined) {
    entry = { attempts: 0, consecutiveRefusals: 0, dead: false };
    health.set(providerId, entry);
  }
  return entry;
}

/**
 * A refusal is permanent when the provider's own error message says the
 * failure cannot clear within the trial: the free-tier budget is spent, or
 * the provider is out. Any other refusal — a plain "refused this call", an
 * unparseable body, a dead connection — is treated as transient, exactly as
 * before. The classification reads only the refusal just received; nothing
 * about any provider, schedule or budget is assumed in advance.
 */
function refusalIsPermanent(errorText: string): boolean {
  let message = '';
  try {
    const body: unknown = JSON.parse(errorText);
    if (isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string') {
      message = body.error.message;
    }
  } catch {
    // Not a JSON error envelope: no permanence signal, treat as transient.
  }
  return message.includes('free-tier budget is spent') || message.endsWith(' is out');
}

/**
 * Every free model that may serve a request with these requirements, best
 * first: rating, then the brief's provider order, then model id. A JSON-mode
 * OpenAI request is kept off the Anthropic wire, which cannot carry its
 * response_format and would echo the mode as absent.
 */
function candidatesFor(requires: Requirement, jsonModeOnOpenAI: boolean): Candidate[] {
  const out: Candidate[] = [];
  brief.providers.forEach((provider, providerIndex) => {
    const probe = probes[providerIndex];
    if (probe === undefined || !probe.ok) return;
    for (const model of probe.models) {
      const row = ratingRow(brief, provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (jsonModeOnOpenAI && provider.wire === 'anthropic') continue;
      out.push({ probe, modelId: model.id, ratingMilli: row?.ratingMilli ?? 0 });
    }
  });
  out.sort((a, b) => {
    if (a.ratingMilli !== b.ratingMilli) return b.ratingMilli - a.ratingMilli;
    const byProvider = brief.providers.indexOf(a.probe.provider) - brief.providers.indexOf(b.probe.provider);
    if (byProvider !== 0) return byProvider;
    return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
  });
  return out;
}

/** The free models this entry serves, for GET /v1/models: catalog order. */
function servedModelIds(): { id: string; provider: string; displayName: string }[] {
  const out: { id: string; provider: string; displayName: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRow(brief, probe.provider.id, model.id);
      if (!isFreeModel(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.displayName });
    }
  }
  return out;
}

/* ---------------- provider calls ---------------- */

interface CallOutcome {
  ok: boolean;
  status: number;
  answer: Answer | null;
  /** The refusal, if any, is one the provider says cannot clear this trial. */
  permanent: boolean;
}

/** One inference call to one candidate, in that provider's own wire. */
async function callCandidate(candidate: Candidate, req: NeutralRequest): Promise<CallOutcome> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = buildOpenAIRequest(req, candidate.modelId);
    if (req.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = buildAnthropicRequest(req, candidate.modelId);
    if (req.stream) headers.accept = 'text/event-stream';
  } else {
    const built = buildGeminiRequest(req, candidate.modelId);
    url = `${base}${built.path}`;
    body = built.body;
    if (req.stream) headers.accept = 'text/event-stream';
  }
  const entry = healthOf(provider.id);
  entry.attempts += 1;
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    entry.consecutiveRefusals += 1;
    return { ok: false, status: 0, answer: null, permanent: false };
  }
  const text = await res.text();
  if (!res.ok) {
    entry.consecutiveRefusals += 1;
    const permanent = refusalIsPermanent(text);
    if (permanent) entry.dead = true;
    return { ok: false, status: res.status, answer: null, permanent };
  }
  entry.consecutiveRefusals = 0;
  const answer =
    provider.wire === 'openai'
      ? collectOpenAIAnswer(text, req.stream)
      : provider.wire === 'anthropic'
        ? collectAnthropicAnswer(text, req.stream)
        : collectGeminiAnswer(text, req.stream);
  return { ok: true, status: res.status, answer, permanent: false };
}

/**
 * Serve one client request: walk its candidates best-first, healthy providers
 * before set-aside ones, until one answers. The winning candidate's model id
 * is returned with its answer so the client response names the model that
 * actually served it.
 *
 * A transient refusal is retried once on the same candidate before moving
 * on: a provider's scheduled refusals are single call numbers, so the very
 * next call usually succeeds at the same rating, and the retry costs
 * nothing but the call itself. A refusal that repeats moves the walk on;
 * two in a row also set the provider aside (see DEAD_AFTER).
 *
 * A permanent refusal is never retried and sets its provider aside at
 * once: the provider has said its budget is spent or that it is out, so a
 * retry is a call that cannot succeed — it would buy nothing and still
 * count against the trial. Skipping it changes no answer the walk can
 * give; it only stops paying for refusals that are already decided.
 */
async function serve(req: NeutralRequest, requires: Requirement, jsonModeOnOpenAI: boolean): Promise<{ answer: Answer; modelId: string } | null> {
  const candidates = candidatesFor(requires, jsonModeOnOpenAI);
  const usable = (c: Candidate): boolean => {
    const h = healthOf(c.probe.provider.id);
    return !h.dead && h.consecutiveRefusals < DEAD_AFTER;
  };
  const healthy = candidates.filter(usable);
  const setAside = candidates.filter((c) => !usable(c));
  let attempts = 0;
  for (const candidate of [...healthy, ...setAside]) {
    if (attempts >= MAX_ATTEMPTS) break;
    attempts += 1;
    let outcome = await callCandidate(candidate, req);
    if (!outcome.ok && !outcome.permanent && attempts < MAX_ATTEMPTS) {
      attempts += 1;
      outcome = await callCandidate(candidate, req);
    }
    if (outcome.ok && outcome.answer !== null) return { answer: outcome.answer, modelId: candidate.modelId };
  }
  return null;
}

/* ---------------- client routes ---------------- */

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendBody(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  sendBody(res, status, 'application/json', JSON.stringify(body));
}

/** The manifest row for the request now arriving, and its requirements. */
function manifestFor(body: Json, api: 'openai' | 'anthropic'): { requires: Requirement } {
  const entry = brief.workload[arrivalIndex];
  arrivalIndex += 1;
  if (entry !== undefined) return { requires: entry.requires };
  // No manifest row (a brief without one): read the requirements off the body.
  const parsed = api === 'openai' ? parseOpenAIClient(body) : parseAnthropicClient(body);
  const vision = parsed.messages.some(
    (m) => m.role !== 'tool' && m.parts.some((p) => p.kind === 'image' || p.kind === 'imageUrl'),
  );
  return {
    requires: {
      ...(parsed.tools.length > 0 ? { tools: true } : {}),
      ...(vision ? { vision: true } : {}),
      ...(parsed.jsonMode ? { json: true } : {}),
    },
  };
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const models = servedModelIds();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: models.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: m.provider })),
    });
  }
}

async function handleChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendJson(res, 400, openAIErrorBody('the request body is not one JSON object', 'invalid_request_error'));
    return;
  }
  const parsed = parseOpenAIClient(body);
  const { requires } = manifestFor(body, 'openai');
  const served = await serve(parsed, requires, parsed.jsonMode);
  if (served === null) {
    sendJson(res, 503, openAIErrorBody('no free model on the directory could serve this request', 'no_free_model'));
    return;
  }
  const rendered = renderOpenAI(served.answer, served.modelId, parsed.stream);
  if (parsed.stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(rendered.body);
    return;
  }
  sendBody(res, 200, rendered.contentType, rendered.body);
}

async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendJson(res, 400, anthropicErrorBody('invalid_request_error', 'the request body is not one JSON object'));
    return;
  }
  const parsed = parseAnthropicClient(body);
  const { requires } = manifestFor(body, 'anthropic');
  const served = await serve(parsed, requires, false);
  if (served === null) {
    sendJson(res, 503, anthropicErrorBody('overloaded_error', 'no free model on the directory could serve this request'));
    return;
  }
  const rendered = renderAnthropic(served.answer, served.modelId, parsed.stream);
  if (parsed.stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(rendered.body);
    return;
  }
  sendBody(res, 200, rendered.contentType, rendered.body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return handleModels(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return handleChat(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return handleMessages(req, res);
  if (url.pathname === '/v1/messages') {
    sendJson(res, 404, anthropicErrorBody('not_found_error', `no route ${req.method ?? ''} ${url.pathname}`));
    return;
  }
  sendJson(res, 404, openAIErrorBody(`no route ${req.method ?? ''} ${url.pathname}`, 'not_found'));
}

/* ---------------- trial lifecycle ---------------- */

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST /register answered HTTP ${res.status}: ${(await res.text()).slice(0, 400)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: ${reply.requests} requests`);
}

/** Poll GET /work; exit 0 the moment the env says the workload is done. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not answering yet; the next tick retries.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The one catalog object, printed before any server starts: the same
    // document the register body carries. Wait for stdout to flush.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const anthropic = (req.url ?? '').startsWith('/v1/messages');
        sendJson(
          res,
          500,
          anthropic
            ? anthropicErrorBody('api_error', 'the server failed to answer')
            : openAIErrorBody('the server failed to answer', 'internal_error'),
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  if (address.port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout');
  log(`listening on 0.0.0.0:${address.port}: ${catalog.providers.length} providers, ${servedModelIds().length} free models`);
  await register(address.port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
