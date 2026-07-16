/**
 * Typed fetch client for the FastAPI backend.
 *
 * Auth model: the short-lived access token lives ONLY in module memory (never
 * localStorage); the refresh token is an httpOnly cookie managed by the
 * backend, so every request is sent with `credentials: "include"`. On a 401
 * the client attempts POST /auth/refresh exactly once (deduped across
 * concurrent requests) and retries the original request.
 */

// ── wire types (mirror core/schemas.py) ──────────────────────────────────────

export interface GarmentAttribute {
  type: string;
  color: string;
  color_hex: string | null;
  formality: string;
  material: string | null;
}

export interface ImageAttributes {
  image_id?: string;
  garments: GarmentAttribute[];
  scene: string;
  scene_type: string;
  lighting: string;
}

export interface MatchExplanation {
  region_id: string;
  bbox: [number, number, number, number]; // xyxy, absolute pixels in the ORIGINAL image
  label: string;
  query_term: string;
  similarity: number;
}

export interface ComponentScores {
  garment: number;
  scene: number;
  attribute: number;
  rerank: number | null;
}

export interface SearchResult {
  image_id: string;
  url?: string;
  score: number;
  components: ComponentScores;
  matches: MatchExplanation[];
  attributes: ImageAttributes | null;
}

export interface QueryGarment {
  type: string | null;
  color: string | null;
}

export interface Negation {
  term: string;
  type: string | null;
  color: string | null;
  material: string | null;
}

export interface ParsedQuery {
  raw: string;
  garments: QueryGarment[];
  scene: string | null;
  style: string | null;
  negations: Negation[];
}

export interface SearchResponse {
  parsed: ParsedQuery | null;
  results: SearchResult[];
}

export interface MeResponse {
  email: string;
  role: string;
}

// ── plumbing ─────────────────────────────────────────────────────────────────

export const API_URL: string = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

let accessToken: string | null = null;

export function hasAccessToken(): boolean {
  return accessToken !== null;
}

export function clearAccessToken(): void {
  accessToken = null;
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, message: string, detail: unknown = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error && e.message) return e.message;
  return "Something went wrong";
}

interface RequestOptions {
  method?: string;
  json?: unknown;
  form?: FormData;
  /** attach the bearer token (default true) */
  auth?: boolean;
  /** on 401: refresh once, then retry (default true) */
  retry401?: boolean;
}

let refreshInFlight: Promise<boolean> | null = null;

async function doRefresh(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/auth/refresh`, {
      method: "POST",
      credentials: "include",
    });
    if (!res.ok) return false;
    const data = (await res.json().catch(() => null)) as
      | { access_token?: string }
      | null;
    if (data?.access_token) {
      accessToken = data.access_token;
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

/** POST /auth/refresh — deduped so N concurrent 401s trigger one refresh. */
export function refresh(): Promise<boolean> {
  if (!refreshInFlight) {
    refreshInFlight = doRefresh().finally(() => {
      refreshInFlight = null;
    });
  }
  return refreshInFlight;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { method = "GET", json, form, auth = true, retry401 = true } = opts;

  const headers: Record<string, string> = {};
  if (json !== undefined) headers["Content-Type"] = "application/json";
  if (auth && accessToken) headers["Authorization"] = `Bearer ${accessToken}`;

  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      method,
      headers,
      credentials: "include",
      body: json !== undefined ? JSON.stringify(json) : form,
    });
  } catch (e) {
    throw new ApiError(0, "Network error — is the API reachable?", e);
  }

  if (res.status === 401 && retry401 && auth) {
    const refreshed = await refresh();
    if (refreshed) {
      return request<T>(path, { ...opts, retry401: false });
    }
    accessToken = null;
  }

  if (!res.ok) {
    let detail: unknown = null;
    let message = `Request failed (${res.status})`;
    try {
      detail = await res.json();
      const d = (detail as { detail?: unknown } | null)?.detail;
      if (typeof d === "string" && d) message = d;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, message, detail);
  }

  if (res.status === 204) return undefined as unknown as T;
  return (await res.json().catch(() => undefined)) as T;
}

// ── auth ─────────────────────────────────────────────────────────────────────

export async function register(email: string, password: string): Promise<void> {
  await request<unknown>("/auth/register", {
    method: "POST",
    json: { email, password },
    auth: false,
    retry401: false,
  });
}

export async function login(
  email: string,
  password: string,
): Promise<{ role: string }> {
  const data =
    (await request<{ access_token?: string; role?: string }>("/auth/login", {
      method: "POST",
      json: { email, password },
      auth: false,
      retry401: false,
    })) ?? {};
  accessToken = data.access_token ?? null;
  return { role: data.role ?? "user" };
}

export async function logout(): Promise<void> {
  try {
    await request<unknown>("/auth/logout", { method: "POST", retry401: false });
  } finally {
    accessToken = null;
  }
}

export async function me(): Promise<MeResponse> {
  const data = (await request<Record<string, unknown>>("/auth/me")) ?? {};
  const nested = (data.user ?? {}) as Record<string, unknown>;
  return {
    email: String(data.email ?? nested.email ?? ""),
    role: String(data.role ?? nested.role ?? "user"),
  };
}

// ── search ───────────────────────────────────────────────────────────────────

function normalizeSearchResponse(data: unknown): SearchResponse {
  if (Array.isArray(data)) {
    return { parsed: null, results: data as SearchResult[] };
  }
  const obj = (data ?? {}) as Record<string, unknown>;
  const parsed = (obj.parsed ?? obj.parsed_query ?? null) as ParsedQuery | null;
  const results = (obj.results ?? obj.items ?? []) as SearchResult[];
  return { parsed, results };
}

export async function search(
  query: string,
  k: number,
  useRerank: boolean,
): Promise<SearchResponse> {
  const data = await request<unknown>("/search", {
    method: "POST",
    json: { query, k, use_rerank: useRerank },
  });
  return normalizeSearchResponse(data);
}

export async function searchImage(
  file: File,
  refinement: string,
  k: number,
): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file);
  if (refinement.trim()) form.append("refinement", refinement.trim());
  form.append("k", String(k));
  const data = await request<unknown>("/search/image", {
    method: "POST",
    form,
  });
  return normalizeSearchResponse(data);
}

export async function parse(query: string): Promise<ParsedQuery> {
  const data =
    (await request<Record<string, unknown>>("/parse", {
      method: "POST",
      json: { query },
    })) ?? {};
  const candidate = (
    typeof data.raw === "string" ? data : (data.parsed ?? data)
  ) as Partial<ParsedQuery>;
  return {
    raw: candidate.raw ?? query,
    garments: candidate.garments ?? [],
    scene: candidate.scene ?? null,
    style: candidate.style ?? null,
    negations: candidate.negations ?? [],
  };
}

/** Resolve a result to a displayable image URL (absolute, API-relative, or fallback). */
export function imageUrl(result: SearchResult): string {
  const u = result.url ?? "";
  if (/^(https?:|data:|blob:)/.test(u)) return u;
  if (u.startsWith("/")) return `${API_URL}${u}`;
  if (u) return `${API_URL}/${u}`;
  return `${API_URL}/images/${encodeURIComponent(result.image_id)}`;
}

// ── admin ────────────────────────────────────────────────────────────────────

export async function adminStats(): Promise<Record<string, unknown>> {
  return (await request<Record<string, unknown>>("/admin/stats")) ?? {};
}

export async function adminDistribution(): Promise<Record<string, unknown>> {
  return (
    (await request<Record<string, unknown>>("/admin/attributes/distribution")) ??
    {}
  );
}

export async function reindex(): Promise<Record<string, unknown>> {
  return (
    (await request<Record<string, unknown>>("/admin/reindex", {
      method: "POST",
    })) ?? {}
  );
}

// ── health ───────────────────────────────────────────────────────────────────

export async function healthz(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/healthz`, { credentials: "include" });
    return res.ok;
  } catch {
    return false;
  }
}
