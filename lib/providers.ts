// Brief parsing, provider probing and catalog building for the aggregator.
//
// The trial brief (the file at $MAKEFUN_PROMPT_FILE) is prose with three JSON
// documents embedded in it, identified by their "schema" member:
//   free-model-aggregator-01/providers/v1  the ten providers (wire, auth, key, base path)
//   free-model-aggregator-01/ratings/v1    the quality table (rating, caps, free flag)
//   free-model-aggregator-01/workload/v1   the twenty requests, in arrival order
// This module reads those documents, probes every provider's own catalog route
// in that provider's wire, and folds the two into the single catalog document
// the entry registers (and prints for --list). Everything is a pure function of
// the brief and the providers' answers: no clock, no randomness.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;

export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** Base path ("/p/<id>") or absolute URL, exactly as the brief spells it. */
  base: string;
  keyEnv: string;
  key: string;
}

export interface RatingRow {
  provider: string;
  model: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  bestByRequest: Map<string, number>;
  workload: WorkloadEntry[];
  error: string | null;
}

/** One model as a provider's catalog route listed it; null means "not stated". */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface ProbeResult {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null =>
  typeof v === 'number' && Number.isFinite(v) ? v : null;
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

/** End index of the JSON object opening at `start`, honouring strings; -1 if none. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every embedded JSON document that carries a string `schema`, in file order. */
function embeddedDocuments(text: string): Json[] {
  const docs: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') docs.push(value);
    } catch {
      // Not a standalone document; keep scanning.
    }
  }
  return docs;
}

function providerFrom(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (str(v.id) === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  const base = [v.path, v.base_path, v.base_url, v.base].map(str).find((s) => s !== '') ?? '';
  return {
    id: str(v.id),
    name: str(v.name) !== '' ? str(v.name) : str(v.id),
    wire,
    auth,
    base,
    keyEnv: str(v.key_env),
    key: str(v.key),
  };
}

function ratingFrom(v: unknown): RatingRow | null {
  if (!isRecord(v) || str(v.model) === '' || str(v.provider) === '') return null;
  return {
    provider: str(v.provider),
    model: str(v.model),
    ratingMilli: numOrNull(v.rating_milli) ?? 0,
    contextLength: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadFrom(v: unknown): WorkloadEntry | null {
  if (!isRecord(v) || str(v.id) === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id: str(v.id),
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(req.tools === true ? { tools: true } : {}),
      ...(req.vision === true ? { vision: true } : {}),
      ...(req.json === true ? { json: true } : {}),
      ...(numOrNull(req.min_context) !== null ? { minContext: numOrNull(req.min_context) as number } : {}),
    },
  };
}

/** Read the brief at `path`; on failure the brief is empty and `error` says why. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], bestByRequest: new Map(), workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read the brief: ${(err as Error).message}` };
  }
  const docs = embeddedDocuments(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const ratings = (ratingsDoc?.ratings ?? []).flatMap((v) => {
    const row = ratingFrom(v);
    return row === null ? [] : [row];
  });
  const bestByRequest = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (isRecord(v) && str(v.id) !== '') bestByRequest.set(str(v.id), numOrNull(v.rating_milli) ?? 0);
    }
  }
  return {
    providers: (directory.providers ?? []).flatMap((v) => {
      const p = providerFrom(v);
      return p === null ? [] : [p];
    }),
    ratings,
    bestByRequest,
    workload: (workloadDoc?.manifest ?? []).flatMap((v) => {
      const w = workloadFrom(v);
      return w === null ? [] : [w];
    }),
    error: null,
  };
}

/** Join a provider's base path (or absolute base URL) to the env origin. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim().replace(/\/+$/, '');
  if (/^https?:\/\//i.test(b)) return b;
  const origin = envUrl.replace(/\/+$/, '');
  if (b === '') return origin;
  return `${origin}${b.startsWith('/') ? '' : '/'}${b}`;
}

/** The header(s) carrying a provider's key, per its auth scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '' || auth === 'none') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  return { 'x-goog-api-key': key };
}

export function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Free or not: the quality table wins, then the listing's own word, else paid. */
export function isFreeModel(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

export interface Capabilities {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export function capabilitiesOf(model: ListedModel, row: RatingRow | null): Capabilities {
  return {
    contextLength: row?.contextLength ?? model.contextLength,
    tools: row?.tools ?? model.tools,
    vision: row?.vision ?? model.vision,
    json: row?.json ?? model.json,
  };
}

/** A model may serve a request only when every stated requirement is met. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.minContext !== undefined && (caps.contextLength === null || caps.contextLength < requires.minContext)) return false;
  return true;
}

/** A listing's price block: free when every price in it is zero, unstated otherwise. */
function zeroPriced(v: Json): boolean | null {
  if (!isRecord(v.pricing)) return null;
  const values = Object.values(v.pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '' && Number.isFinite(Number(x))) return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => n === 0);
}

function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? str(v.name).replace(/^models\//, '') : str(v.id);
  if (id === '') return null;
  return {
    id,
    displayName: str(v.display_name) || str(v.displayName) || id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    contextLength: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** Ask one provider for its catalog, in its own wire; failures leave it out. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = 5000): Promise<ProbeResult> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    return {
      provider,
      ok: true,
      models: list.flatMap((v) => {
        const m = listedModel(provider.wire, v);
        return m === null ? [] : [m];
      }),
    };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

/**
 * The registered catalog: one entry per provider that answered its probe, its
 * models sorted by id, free flags and context lengths from the quality table
 * where it has a row and from the listing otherwise. base_url stays the brief's
 * own path: the env's origin differs between a live trial and its replay, and
 * this document is part of what must match byte for byte.
 */
export function buildCatalog(brief: Brief, probes: readonly ProbeResult[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(brief, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFreeModel(model, row),
        context_length: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}
