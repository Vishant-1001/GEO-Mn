/**
 * GEO-MN — manganese supply decision support — frontend controller.
 *
 * Rules this file follows:
 *  - Every number, decision, target and metric shown comes from the backend.
 *    Missing values render as "N/A" / "unavailable"; nothing is substituted.
 *  - Decisions, portfolio selection and target priority are never computed here.
 *  - Provenance (real / synthetic / simulated / cached) is always displayed.
 *  - Status is never conveyed by colour alone: every tone carries text + an icon.
 */

(function () {
    'use strict';

    const API_BASE = window.location.origin;
    // Selected deterministic demo state (mine_id). Changed only via the demo selector;
    // every result is recomputed by the backend for this id.
    let MINE_ID = 'DEMO_MINE';

    // ── Single application state ────────────────────────────
    const state = {
        currentSection: 'supply-command',
        lastWorkflow: 'exploration',
        gen: 0,                  // bumped on demo-state switch; stale responses are dropped
        apiOnline: null,
        health: null,
        supply: null,
        supplyPromise: null,
        exploration: {
            targets: [],
            targetsMeta: null,
            targetsPromise: null,
            selectedId: null,
            flyOnLoad: false,
            detail: {},          // id -> detail response
            pending: {},         // id -> in-flight detail request
            subsurface: {},      // id -> subsurface-scenarios response
            surface: null,
            query: null,
        },
        production: { forecast: null, history: null },
        recovery: null,
        contingency: null,
        decision: { current: null, previous: null, flipRuns: 0 },
        decisionHistory: [],
        trust: { exploration: null, production: null, provenance: null, reconciliation: null, recovery: null, context: 'exploration' },
        loaded: {},              // section -> true once fetched
        flipPreset: false,
        maps: { exploration: null, baseLayers: {}, currentBase: null, surfaceLayer: null, markers: {}, highlight: null, queryMarker: null },
    };

    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
    const enc = encodeURIComponent;

    // ── Small utilities ─────────────────────────────────────
    function esc(v) {
        return String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    }

    function num(v) {
        if (typeof v === 'number') return Number.isFinite(v) ? v : null;
        if (typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))) return Number(v);
        return null;
    }

    // First present (non-null) value among the given keys.
    function pick(obj, ...keys) {
        if (!obj || typeof obj !== 'object') return undefined;
        for (const k of keys) {
            if (obj[k] !== undefined && obj[k] !== null) return obj[k];
        }
        return undefined;
    }

    // Like pick, but also looks one or two levels down (metrics are often grouped).
    // Nested baseline/naive blocks are skipped unless a baseline metric is being
    // looked up, so a baseline MAE is never displayed as the model's MAE.
    function deepPick(obj, keys, depth = 2) {
        const direct = pick(obj, ...keys);
        if (direct !== undefined || depth === 0 || !obj || typeof obj !== 'object') return direct;
        const wantBaseline = keys.some(k => /baseline|naive/i.test(k));
        for (const [k, v] of Object.entries(obj)) {
            if (!wantBaseline && /baseline|naive/i.test(k)) continue;
            if (v && typeof v === 'object' && !Array.isArray(v)) {
                const found = deepPick(v, keys, depth - 1);
                if (found !== undefined) return found;
            }
        }
        return undefined;
    }

    const human = s => String(s ?? '').replace(/[_-]+/g, ' ').trim();
    const upperHuman = s => human(s).toUpperCase();
    const norm = s => String(s ?? '').trim().toUpperCase().replace(/[^A-Z0-9]+/g, '_').replace(/^_|_$/g, '');
    const clampPct = v => Math.max(0, Math.min(100, num(v) ?? 0));

    function fmtT(v) {
        v = num(v);
        if (v === null) return 'N/A';
        const a = Math.abs(v);
        if (a >= 1e6) return `${(v / 1e6).toFixed(2)} Mt`;
        if (a >= 1000) return `${(v / 1000).toFixed(a < 10000 ? 2 : 1)} kt`;
        return `${Math.round(v).toLocaleString()} t`;
    }

    function fmtSigned(v) {
        v = num(v);
        if (v === null) return 'N/A';
        return `${v > 0 ? '+' : (v < 0 ? '−' : '')}${fmtT(Math.abs(v))}`;
    }

    function fmtNum(v, dp = 3) {
        const n = num(v);
        if (n === null) return null;
        if (Number.isInteger(n)) return n.toLocaleString();
        return String(Number(n.toFixed(dp)));
    }

    const fmtPct = (v, dp = 0) => (num(v) === null ? 'N/A' : `${(num(v) * 100).toFixed(dp)}%`);
    const fmtWin = w => String(w).replace('/', ' → ');

    function asArray(v) {
        if (Array.isArray(v)) return v;
        if (v === undefined || v === null || v === '') return [];
        return [v];
    }

    // ── Toasts & loading ────────────────────────────────────
    function showToast(message, type = 'info') {
        const container = $('#toastContainer');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 250);
        }, 4000);
    }

    function setLoading(btn, loading) {
        if (!btn) return;
        btn.classList.toggle('loading', loading);
        btn.disabled = loading;
        btn.setAttribute('aria-busy', loading ? 'true' : 'false');
    }

    function loadingHTML(label = 'Loading…') {
        return `<div class="loading-state" role="status"><span class="spinner spinner-inline" aria-hidden="true"></span>${esc(label)}</div>`;
    }

    function errorHTML(title, err, retryId) {
        const detail = err && err.userMessage ? err.userMessage : '';
        return `<div class="error-state" role="alert">
            <div class="error-title">${iconHTML('bad')}${esc(title)}</div>
            ${detail ? `<div class="error-detail">${esc(detail)}</div>` : ''}
            ${retryId ? `<button type="button" class="btn btn-ghost btn-sm" data-retry="${esc(retryId)}">Retry</button>` : ''}
        </div>`;
    }

    // ── Resilient API client ────────────────────────────────
    class ApiError extends Error {
        constructor(kind, status, userMessage) {
            super(userMessage);
            this.kind = kind;            // 'timeout' | 'network' | 'http' | 'parse'
            this.status = status;
            this.userMessage = userMessage;
        }
    }

    // FastAPI 422s arrive as detail: [{loc, msg}]; flatten without leaking internals.
    function describeDetail(body, status) {
        if (body && typeof body.message === 'string' && body.message.length < 400) {
            return body.error ? `${body.message} [${body.error}]` : body.message;
        }
        const d = body && body.detail;
        if (Array.isArray(d)) {
            return d.slice(0, 3).map(e => {
                const field = Array.isArray(e.loc) ? e.loc.filter(x => x !== 'body').join('.') : '';
                return `${field ? field + ': ' : ''}${e.msg || 'invalid'}`;
            }).join('; ');
        }
        if (status === 404) return 'Endpoint not available on this backend (HTTP 404).';
        if (typeof d === 'string' && d.length < 240 && !/Traceback|File "/.test(d)) return d;
        if (status >= 500) return `Backend error (HTTP ${status}).`;
        return `Request failed (HTTP ${status}).`;
    }

    function recordLatency(ms, endpoint) {
        const el = $('#latencyPill');
        if (!el) return;
        el.textContent = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
        el.title = `Last call: ${endpoint}`;
    }

    async function request(method, endpoint, { body, timeout = 15000, retries = 0 } = {}) {
        let attempt = 0;
        for (;;) {
            const ctrl = new AbortController();
            const timer = setTimeout(() => ctrl.abort(), timeout);
            const t0 = performance.now();
            try {
                const res = await fetch(`${API_BASE}${endpoint}`, {
                    method,
                    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
                    body: body !== undefined ? JSON.stringify(body) : undefined,
                    signal: ctrl.signal,
                });
                clearTimeout(timer);
                recordLatency(performance.now() - t0, endpoint);
                if (!res.ok) {
                    const errBody = await res.json().catch(() => null);
                    const err = new ApiError('http', res.status, describeDetail(errBody, res.status));
                    // Retry only transient server errors, never 4xx.
                    if (res.status >= 500 && attempt < retries) { attempt++; await wait(600 * attempt); continue; }
                    console.warn(`${method} ${endpoint} -> ${res.status}`);
                    throw err;
                }
                try {
                    return await res.json();
                } catch (_) {
                    throw new ApiError('parse', res.status, 'Backend returned an unreadable response.');
                }
            } catch (e) {
                clearTimeout(timer);
                if (e instanceof ApiError) throw e;
                const kind = e && e.name === 'AbortError' ? 'timeout' : 'network';
                if (attempt < retries) { attempt++; await wait(600 * attempt); continue; }
                console.warn(`${method} ${endpoint} failed: ${kind}`);
                throw new ApiError(kind, 0, kind === 'timeout'
                    ? `No response within ${Math.round(timeout / 1000)}s.`
                    : 'Backend unreachable.');
            }
        }
    }

    const wait = ms => new Promise(r => setTimeout(r, ms));
    const api = {
        get: (ep, opts = {}) => request('GET', ep, { retries: 1, ...opts }),
        post: (ep, body, opts = {}) => request('POST', ep, { body, ...opts }),
    };

    // ── Provenance badges ───────────────────────────────────
    // [full label, css class, short label]
    const MODE_BADGES = {
        REAL_PUBLIC: ['REAL / PUBLIC', 'real', 'REAL · PUBLIC'],
        REAL_GOVERNMENT: ['REAL / GOVERNMENT OF INDIA', 'realgov', 'REAL · GOVT'],
        REAL_MOIL_PUBLIC: ['REAL / MOIL PUBLIC DISCLOSURE', 'realmoil', 'REAL · MOIL'],
        REAL_DERIVED: ['REAL-DERIVED FEATURES', 'derived', 'REAL-DERIVED'],
        REPORTED_BLOCK_LEVEL: ['REPORTED (BLOCK LEVEL)', 'realgov', 'REPORTED'],
        REAL: ['REAL / PUBLIC', 'real', 'REAL'],
        PUBLIC: ['REAL / PUBLIC', 'real', 'REAL · PUBLIC'],
        SYNTHETIC: ['SYNTHETIC DEMONSTRATION DATA', 'synthetic', 'SYNTHETIC DEMO'],
        SYNTHETIC_DEMO: ['SYNTHETIC DEMONSTRATION DATA', 'synthetic', 'SYNTHETIC DEMO'],
        SYNTHETIC_DEMONSTRATION: ['SYNTHETIC DEMONSTRATION DATA', 'synthetic', 'SYNTHETIC DEMO'],
        SIMULATED: ['SIMULATED SCENARIO', 'simulated', 'SIMULATED'],
        SIMULATION: ['SIMULATED SCENARIO', 'simulated', 'SIMULATED'],
        CACHED: ['CACHED / REPLAY', 'cached', 'CACHED'],
        CACHE: ['CACHED / REPLAY', 'cached', 'CACHED'],
        REPLAY: ['CACHED / REPLAY', 'cached', 'CACHED'],
        CACHED_REPLAY: ['CACHED / REPLAY', 'cached', 'CACHED'],
        CACHED_GRID: ['CACHED / REPLAY', 'cached', 'CACHED'],
        LIVE: ['LIVE COORDINATE QUERY', 'live', 'LIVE QUERY'],
        LIVE_SATELLITE: ['LIVE COORDINATE QUERY', 'live', 'LIVE QUERY'],
        LIVE_COORDINATE_QUERY: ['LIVE COORDINATE QUERY', 'live', 'LIVE QUERY'],
        LIVE_QUERY: ['LIVE COORDINATE QUERY', 'live', 'LIVE QUERY'],
        UNAVAILABLE: ['UNAVAILABLE', 'unavailable', 'UNAVAILABLE'],
        NONE: ['UNAVAILABLE', 'unavailable', 'UNAVAILABLE'],
    };

    function modeInfo(value) {
        const key = norm(value);
        const m = MODE_BADGES[key];
        return m ? { label: m[0], cls: m[1], short: m[2], known: true }
            : { label: upperHuman(value), cls: 'neutral', short: upperHuman(value), known: false };
    }

    function badgeHTML(value, prefix) {
        if (value === undefined || value === null || value === '') return '';
        const m = modeInfo(value);
        return `<span class="badge badge-${m.cls}">${prefix ? `<span class="badge-k">${esc(prefix)}:</span> ` : ''}${esc(m.label)}</span>`;
    }

    function fmtWindow(w) {
        if (!w) return null;
        if (typeof w === 'string') return w;
        const s = pick(w, 'start', 'from', 'start_date');
        const e = pick(w, 'end', 'to', 'end_date');
        if (s || e) return `${s || '?'} → ${e || '?'}`;
        return null;
    }

    // Full provenance row (technical views): one badge per key the backend returned.
    function provenanceHTML(prov) {
        if (!prov) return badgeHTML('UNAVAILABLE', 'PROVENANCE');
        if (typeof prov === 'string') return badgeHTML(prov, 'DATA');
        const out = [];
        const win = fmtWindow(pick(prov, 'observation_window'));
        Object.entries(prov).forEach(([k, v]) => {
            if (k === 'observation_window') return;
            if (typeof v === 'string' && modeInfo(v).known) {
                out.push(badgeHTML(v, upperHuman(k.replace(/_?(mode|source)$/i, '') || k)));
            } else if (v && typeof v === 'object' && !Array.isArray(v)) {
                const mode = pick(v, 'mode', 'data_mode', 'type', 'status', 'source');
                if (typeof mode === 'string') out.push(badgeHTML(mode, upperHuman(k)));
            }
        });
        if (win) out.push(`<span class="badge badge-window"><span class="badge-k">WINDOW:</span> ${esc(fmtWin(win))}</span>`);
        return out.length ? out.join('') : badgeHTML('UNAVAILABLE', 'PROVENANCE');
    }

    // Compact provenance (screen headers): one badge per distinct data mode; the keys
    // that carry each mode are listed in the tooltip. Nothing is dropped silently.
    function provChips(prov) {
        if (!prov) return badgeHTML('UNAVAILABLE');
        if (typeof prov === 'string') return badgeHTML(prov);
        const groups = new Map();
        const walk = o => Object.entries(o).forEach(([k, v]) => {
            if (typeof v === 'string' && modeInfo(v).known) {
                const m = modeInfo(v);
                const g = groups.get(m.label) || { m, keys: [] };
                g.keys.push(human(k.replace(/_?(mode|source)$/i, '')) || k);
                groups.set(m.label, g);
            } else if (v && typeof v === 'object' && !Array.isArray(v)) walk(v);
        });
        walk(prov);
        let out = [...groups.values()].map(g => `<span class="badge badge-${g.m.cls}" title="${esc(g.m.label)} — ${esc([...new Set(g.keys)].join(', '))}">${esc(g.m.short)}</span>`).join('');
        const win = fmtWindow(pick(prov, 'observation_window'));
        if (win) out += `<span class="badge badge-window" title="Observation window">${esc(fmtWin(win))}</span>`;
        return out || badgeHTML('UNAVAILABLE');
    }

    // ── Status system: tone + icon + text, used everywhere ──
    const TONE_ICON = { ok: '✓', warn: '!', bad: '✕', info: 'i', neutral: '–' };

    function iconHTML(tone) {
        return `<span class="ico" data-tone="${tone}" aria-hidden="true">${TONE_ICON[tone] || '–'}</span>`;
    }

    function chip(text, tone = 'neutral', title) {
        return `<span class="chip" data-tone="${tone}"${title ? ` title="${esc(title)}"` : ''}><span class="chip-ico" aria-hidden="true">${TONE_ICON[tone] || '–'}</span>${esc(text)}</span>`;
    }

    function tileHTML({ label, value, sub, tone = 'neutral', mono = false, title }) {
        return `<div class="tile" data-tone="${tone}"${title ? ` title="${esc(title)}"` : ''}>
            <span class="tile-k">${iconHTML(tone)}${esc(label)}</span>
            <span class="tile-v${mono ? ' mono' : ''}">${esc(value ?? 'N/A')}</span>
            ${sub ? `<span class="tile-s" title="${esc(sub)}">${esc(sub)}</span>` : ''}
        </div>`;
    }

    function miniHTML(label, value, tone = 'neutral', sub) {
        return `<div class="mini" data-tone="${tone}"><span class="mini-k">${esc(label)}</span><span class="mini-v">${iconHTML(tone)}${esc(value ?? 'N/A')}</span>${sub ? `<span class="mini-s">${esc(sub)}</span>` : ''}</div>`;
    }

    const applTone = l => { const n = norm(l); return !n ? 'neutral' : /HIGH/.test(n) ? 'ok' : /MOD/.test(n) ? 'warn' : /LOW|OUT/.test(n) ? 'bad' : 'neutral'; };
    const uncTone = l => { const n = norm(l); return n === 'LOW' ? 'ok' : /MOD/.test(n) ? 'warn' : n === 'HIGH' ? 'bad' : 'neutral'; };
    const riskTone = r => { const n = norm(r); return n === 'LOW' ? 'ok' : /MED|MOD/.test(n) ? 'warn' : /HIGH|CRIT/.test(n) ? 'bad' : 'neutral'; };
    const stratTone = s => { const n = norm(typeof s === 'object' && s ? pick(s, 'level', 'label') : s); return n === 'HIGH' ? 'info' : n ? 'neutral' : 'neutral'; };
    const evTone = lvl => (lvl === null || lvl === undefined) ? 'neutral' : lvl >= 3 ? 'ok' : lvl === 2 ? 'info' : 'warn';
    const dataTone = mode => { const c = modeInfo(mode).cls; return /real|derived/.test(c) ? 'ok' : c === 'cached' ? 'info' : c === 'unavailable' ? 'bad' : 'warn'; };

    // ── Normalizers (tolerant of minor schema variation) ────
    function normDrivers(raw) {
        if (!raw) return [];
        let list = raw;
        if (!Array.isArray(raw) && typeof raw === 'object') {
            list = Object.entries(raw).map(([name, value]) => ({ name, value }));
        }
        return asArray(list).map(d => {
            if (typeof d === 'string') return { name: d, value: null, key: null };
            const valueKeys = ['contribution_tonnes', 'impact_tonnes', 'contribution_pct', 'share_pct', 'contribution', 'impact', 'share', 'value', 'weight'];
            const key = valueKeys.find(k => num(d[k]) !== null) || null;
            return {
                name: pick(d, 'label', 'name', 'feature', 'driver', 'factor') ?? '—',
                value: key ? num(d[key]) : null,
                key,
                unit: pick(d, 'unit'),
                featureValue: key !== 'value' ? num(d.value) : null,
                direction: pick(d, 'direction', 'effect'),
            };
        });
    }

    function driverValueText(d) {
        if (d.value === null) return '';
        // A contribution in tonnes is a tonnage, whatever the feature's own unit is.
        if (d.key && /tonnes/.test(d.key)) return fmtSigned(d.value);
        if (d.unit) return `${fmtNum(d.value, 2)} ${d.unit}`;
        if (d.key && /pct|percent/.test(d.key)) return `${fmtNum(d.value, 1)}%`;
        return fmtNum(d.value, 3);
    }

    function normLevel(v) {
        // Accepts "HIGH", {level: "HIGH", score: 92}, 92 → {level, score}
        if (v === undefined || v === null) return { level: null, score: null };
        if (typeof v === 'number') return { level: null, score: v };
        if (typeof v === 'string') return { level: v.toUpperCase(), score: null };
        return {
            level: pick(v, 'level', 'class', 'category', 'label') ? String(pick(v, 'level', 'class', 'category', 'label')).toUpperCase() : null,
            score: num(pick(v, 'score', 'value', 'rank', 'percentile')),
            note: pick(v, 'note', 'warning', 'message', 'reason'),
        };
    }

    function normTarget(t) {
        if (!t || typeof t !== 'object') return null;
        const id = pick(t, 'target_id', 'id', 'name');
        const pros = normLevel(pick(t, 'prospectivity', 'relative_prospectivity'));
        if (pros.level === null && pick(t, 'prospectivity_level')) pros.level = String(t.prospectivity_level).toUpperCase();
        if (pros.score === null) pros.score = num(pick(t, 'prospectivity_score', 'prospectivity_rank', 'rank'));
        const appl = normLevel(pick(t, 'applicability', 'applicability_level', 'model_applicability'));
        const unc = normLevel(pick(t, 'uncertainty', 'uncertainty_level'));
        const centroid = pick(t, 'centroid', 'center');
        let lat = num(pick(t, 'lat', 'latitude', 'centroid_lat'));
        let lon = num(pick(t, 'lon', 'lng', 'longitude', 'centroid_lon'));
        if ((lat === null || lon === null) && centroid) {
            if (Array.isArray(centroid)) { lat = num(centroid[0]); lon = num(centroid[1]); }
            else { lat = num(pick(centroid, 'lat', 'latitude')); lon = num(pick(centroid, 'lon', 'lng', 'longitude')); }
        }
        return {
            raw: t,
            id: id !== undefined ? String(id) : null,
            name: pick(t, 'display_name', 'label', 'region_name'),
            lat, lon,
            geometry: pick(t, 'geometry', 'polygon', 'boundary'),
            radiusKm: num(pick(t, 'radius_km')),
            pros, appl, unc,
            status: pick(t, 'status', 'target_status'),
            priority: num(pick(t, 'exploration_priority', 'priority_score', 'priority')),
            strategic: pick(t, 'strategic_relevance'),
            maturity: pick(t, 'evidence_maturity', 'evidence_level', 'maturity_level', 'maturity'),
            evidence: pick(t, 'evidence', 'evidence_layers') ?? ((t.surface_evidence || t.geological_evidence || t.geological_context || t.features || t.subsurface_evidence || t.subsurface_status) ? {
                surface: t.surface_evidence || (t.features ? { status: 'AVAILABLE' } : undefined),
                geology: t.geological_evidence || (t.geological_context ? { status: t.geological_context.unit_name ? 'AVAILABLE' : 'UNAVAILABLE' } : undefined),
                subsurface: (t.subsurface_evidence && !Array.isArray(t.subsurface_evidence) ? t.subsurface_evidence : undefined) || (t.subsurface_status ? { status: t.subsurface_status } : undefined),
            } : undefined),
            geology: t.geological_evidence || pick(t, 'geological_context'),
            distanceKm: num(pick(t, 'distance_to_demo_mine_km')),
            context: pick(t, 'target_context'),
            whyTarget: pick(t, 'why_target', 'why_this_target', 'rationale'),
            whyNow: pick(t, 'why_this_target_now', 'why_target_now', 'why_now'),
            reasonCodes: pick(t, 'reason_codes'),
            nextEvidence: pick(t, 'next_evidence', 'next_evidence_steps'),
            warnings: asArray(pick(t, 'warnings', 'coverage_warning', 'applicability_warning', 'domain_warning')),
            inDomain: pick(t, 'in_domain', 'within_applicability', 'in_applicability_domain'),
            source: pick(t, 'source_mode', 'source', 'data_source', 'satellite_source'),
            window: fmtWindow(pick(t, 'observation_window')) || fmtWindow({ start: t.observation_start, end: t.observation_end }),
            provenance: pick(t, 'provenance'),
        };
    }

    const isLow = lvl => lvl && /LOW|OUT|OOD|UNSUPPORTED|NONE/.test(lvl);
    const isHigh = lvl => lvl && /HIGH|VERY/.test(lvl);

    // Marker class follows backend state: prospectivity × applicability × uncertainty.
    function targetStyle(t) {
        const lowAppl = isLow(t.appl.level) || t.inDomain === false;
        if (lowAppl) return 'caution';
        if ((isHigh(t.pros.level) || (t.raw && t.raw.priority_rank === 1)) && (t.appl.level === null || !isLow(t.appl.level))) return 'priority';
        return 'standard';
    }

    function statusText(t) {
        if (t.status) {
            const s = norm(t.status);
            // Wording rules: never "reserve", never "drill target" unless evidence supports it.
            if (/RESERVE|DEPOSIT/.test(s)) return 'EXPLORATION TARGET';
            if (/DRILL/.test(s)) {
                const lvl = maturityLevel(t.maturity);
                return lvl !== null && lvl >= 4 ? upperHuman(t.status) : 'PRIORITY EXPLORATION TARGET';
            }
            return upperHuman(t.status);
        }
        return 'EXPLORATION TARGET';
    }

    function maturityLevel(m) {
        if (m === undefined || m === null) return null;
        if (typeof m === 'number') return m;
        if (typeof m === 'object') return num(pick(m, 'level', 'value'));
        const match = String(m).match(/\d+/);
        return match ? Number(match[0]) : null;
    }

    // ── Decision vocabulary (display only; states come from the backend) ──
    const DECISIONS = {
        OPERATIONAL_RESPONSE: { title: 'OPERATIONAL RESPONSE', short: 'OPERATIONAL', tone: 'ok' },
        OPERATIONAL_AND_EXPLORATION_CONTINGENCY: { title: 'OPERATIONAL + EXPLORATION CONTINGENCY', short: 'OPS + EXPLORATION', tone: 'warn' },
        REVIEW_REQUIRED: { title: 'REVIEW REQUIRED', short: 'REVIEW REQUIRED', tone: 'bad' },
    };
    const SUPPLY_STATUS = {
        ON_TRACK: ['ON TRACK', 'ok'],
        OPERATIONALLY_RECOVERABLE: ['OPERATIONALLY RECOVERABLE', 'warn'],
        RESIDUAL_GAP: ['RESIDUAL GAP', 'bad'],
    };

    function decisionInfo(ds) {
        if (!ds) return { title: 'UNAVAILABLE', short: 'N/A', tone: 'neutral' };
        return DECISIONS[norm(ds)] || { title: upperHuman(ds), short: upperHuman(ds), tone: 'neutral' };
    }

    function normDecision(src) {
        if (!src) return null;
        const ds = pick(src, 'decision_state', 'decision', 'state');
        if (!ds || typeof ds !== 'string') return null;
        return {
            state: ds,
            horizon: pick(src, 'decision_horizon', 'horizon', 'horizon_class'),
            nextTarget: pick(src, 'next_target', 'next_target_id', 'recommended_target'),
            reasons: asArray(pick(src, 'review_reasons', 'reasons', 'decision_reasons')),
            summary: pick(src, 'decision_summary', 'summary', 'explanation'),
            residual: num(pick(src, 'expected_residual_gap_tonnes', 'residual_gap_tonnes')),
            worst: num(pick(src, 'worst_case_residual_gap_tonnes')),
        };
    }

    function reasonText(r) {
        if (typeof r === 'string') return r;
        if (r && typeof r === 'object') {
            const code = pick(r, 'code', 'reason_code');
            const text = pick(r, 'text', 'message', 'explanation', 'detail', 'description');
            return [code ? `[${code}]` : '', text || ''].join(' ').trim() || JSON.stringify(r);
        }
        return String(r);
    }

    // Backend reason objects {code, text} as a compact list: code chip + one line.
    function reasonListHTML(items, tone = 'neutral') {
        const rows = asArray(items).filter(Boolean);
        if (!rows.length) return '';
        return `<ul class="reason-list">${rows.map(r => {
            const code = typeof r === 'object' ? pick(r, 'code', 'reason_code') : null;
            const text = typeof r === 'object' ? pick(r, 'text', 'message', 'explanation', 'detail', 'description') : r;
            return `<li>${code ? `<span class="code" data-tone="${tone}">${esc(human(code))}</span>` : ''}<span>${esc(text || '')}</span></li>`;
        }).join('')}</ul>`;
    }

    // ── Driver bars (diverging: lowers ← 0 → raises) ────────
    function renderDriverBars(el, raw, { limit = 8 } = {}) {
        if (!el) return;
        const drivers = normDrivers(raw);
        if (!drivers.length) {
            el.innerHTML = '<div class="empty-state">No model contributions returned.</div>';
            return;
        }
        // Largest contributions first; the remainder stays one click away.
        const ranked = drivers.some(d => d.value !== null) ? [...drivers].sort((a, b) => Math.abs(b.value ?? 0) - Math.abs(a.value ?? 0)) : drivers;
        const max = Math.max(...drivers.map(d => Math.abs(d.value ?? 0)), 0);
        const rowsHTML = list => list.map(d => {
            const w = d.value !== null && max > 0 ? Math.max(1.5, Math.abs(d.value) / max * 50) : 0;
            const neg = d.value !== null && d.value < 0;
            const valText = driverValueText(d);
            const fv = d.featureValue !== null ? ` — current ${fmtNum(d.featureValue, 3)}${d.unit ? ' ' + d.unit : ''}` : '';
            return `<div class="drv" title="${esc(human(d.name) + fv)}">
                <span class="drv-name">${esc(human(d.name))}</span>
                <span class="drv-track" role="img" aria-label="${esc(human(d.name))}${valText ? ': ' + esc(valText) : ''}">
                    <span class="drv-axis"></span>
                    ${d.value !== null ? `<span class="drv-bar ${neg ? 'neg' : 'pos'}" style="${neg ? 'right' : 'left'}:50%;width:${w}%"></span>` : '<span class="drv-norank">rank only</span>'}
                </span>
                <span class="drv-val mono ${neg ? 'neg' : 'pos'}">${esc(valText)}</span>
            </div>`;
        }).join('');
        const top = ranked.slice(0, limit), rest = ranked.slice(limit);
        el.innerHTML = `<div class="drivers">${rowsHTML(top)}</div>
        ${rest.length ? `<details class="disclosure"><summary>All ${ranked.length} Contributions</summary><div class="drivers disclosure-body">${rowsHTML(rest)}</div></details>` : ''}
        <div class="chart-legend chart-legend-sm"><span><i class="lg-sw lg-neg"></i>Lowers forecast</span><span><i class="lg-sw lg-pos"></i>Raises forecast</span></div>`;
    }

    // Interval wording is driven only by backend validation metadata.
    function intervalInfo(src) {
        const qv = src && src.quantile_validation;
        const validated = src && src.quantiles_validated === true;
        const obs = qv ? num(qv.observed_coverage) : null;
        const nom = qv ? num(qv.nominal_coverage) : null;
        const pct = v => `${Math.round(v * 100)}%`;
        let text;
        if (!qv || qv.status === 'NOT_AVAILABLE') text = 'P10–P90 interval: validation unavailable';
        else if (validated) text = `P10–P90 prediction interval · observed backtest coverage ${obs !== null ? pct(obs) : 'N/A'} (nominal ${nom !== null ? pct(nom) : 'N/A'})`;
        else text = `P10–P90 interval not validated · observed backtest coverage ${obs !== null ? pct(obs) : 'N/A'} vs nominal ${nom !== null ? pct(nom) : 'N/A'} — indicative only`;
        return { validated, text, window: qv && qv.evaluation_window, obs, nom };
    }

    function riskPolicyText(src) {
        const p = pick(src, 'risk_policy_version', 'risk_policy', 'risk_policy_source');
        if (!p) return 'configured policy';
        if (typeof p === 'string') return p;
        return pick(p, 'name', 'source', 'label') || 'configured policy';
    }

    function dlHTML(rows) {
        const r = rows.filter(([, v]) => v !== undefined && v !== null && v !== '');
        return r.length ? `<dl class="kv">${r.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join('')}</dl>` : '';
    }

    // ═══════════════════════════════════════════════════════
    // HEALTH / GLOBAL STATUS
    // ═══════════════════════════════════════════════════════
    function setPill(id, st, text) {
        const pill = $(id);
        if (!pill) return;
        pill.dataset.state = st;
        const t = $('.pill-text', pill);
        if (t) t.textContent = text;
    }

    function dataModeFrom(...sources) {
        for (const s of sources) {
            if (!s) continue;
            const v = pick(s, 'data_mode', 'production_data', 'operational_data', 'mode');
            if (typeof v === 'string') return v;
        }
        return null;
    }

    function updateDataModePill() {
        const mode = dataModeFrom(state.health, state.supply && state.supply.provenance, state.trust.provenance);
        if (!mode) { setPill('#pillData', 'pending', 'DATA: UNKNOWN'); return; }
        const m = modeInfo(mode);
        setPill('#pillData', m.cls, `DATA: ${m.short}`);
    }

    async function loadHealth() {
        setPill('#pillApi', 'pending', 'API: CHECKING');
        try {
            const h = await api.get('/api/health', { timeout: 8000 });
            state.health = h;
            state.apiOnline = h && String(h.status).toLowerCase() === 'ok';
        } catch (e) {
            state.health = null;
            state.apiOnline = false;
        }
        const h = state.health;
        $('#backendBanner').hidden = state.apiOnline !== false || !!h;
        if (h) {
            setPill('#pillApi', state.apiOnline ? 'ok' : 'warn', state.apiOnline ? 'API ONLINE' : `API ${upperHuman(h.status || 'DEGRADED')}`);
            const exp = h.exploration_model === true, prod = h.production_models === true;
            if (exp && prod) setPill('#pillModels', 'ok', 'MODELS READY');
            else if (exp || prod) setPill('#pillModels', 'warn', `MODELS PARTIAL (${exp ? 'exploration' : 'production'} only)`);
            else setPill('#pillModels', 'bad', 'MODELS UNAVAILABLE');
            $('#telApiVersion').textContent = h.api_version || '—';
            $('#telEarthEngine').textContent = h.live_satellite === true ? 'AVAILABLE (2024 REF.)' : (h.live_satellite === false ? 'OFF → CACHED GRID' : '—');
            $('#telCache').textContent = h.cache === true ? 'LOADED' : (h.cache === false ? 'UNAVAILABLE' : '—');
            $('#statusDot').className = `status-dot ${state.apiOnline ? 'online' : 'warn'}`;
            $('#statusText').textContent = state.apiOnline ? 'API ONLINE' : 'API DEGRADED';
        } else {
            setPill('#pillApi', 'bad', 'API OFFLINE');
            setPill('#pillModels', 'bad', 'MODELS UNKNOWN');
            $('#statusDot').className = 'status-dot offline';
            $('#statusText').textContent = 'BACKEND UNAVAILABLE';
            ['#telApiVersion', '#telEarthEngine', '#telCache'].forEach(s => { $(s).textContent = '—'; });
        }
        updateDataModePill();
        return state.apiOnline;
    }

    // Live state next to each workflow step in the sidebar.
    function setNavState(section, text, tone = 'neutral') {
        const el = $(`[data-nav-state="${section}"]`);
        if (!el) return;
        el.textContent = text || '';
        el.dataset.tone = tone;
        el.hidden = !text;
    }

    function updateNavStates() {
        const s = state.supply;
        if (s) {
            const di = decisionInfo(s.decision_state);
            setNavState('supply-command', di.short, di.tone);
            const gap = num(s.gap_p50_tonnes);
            setNavState('production-risk', gap === null ? '' : (gap > 0 ? `GAP ${fmtT(gap)}` : 'ON TARGET'), gap > 0 ? riskTone(s.risk_state) : 'ok');
            const res = num(s.expected_residual_gap_tonnes);
            if (s.selection_status === 'REVIEW_REQUIRED') setNavState('recovery', 'REVIEW', 'bad');
            else setNavState('recovery', res === null ? '' : `RESIDUAL ${fmtT(res)}`, res > 0 ? 'bad' : 'ok');
            setNavState('targets', s.next_target ? `NEXT ${s.next_target}` : (norm(s.decision_state) === 'REVIEW_REQUIRED' ? 'WITHHELD' : 'NOT ACTIVE'), s.next_target ? 'warn' : 'neutral');
        } else {
            ['supply-command', 'production-risk', 'recovery', 'targets'].forEach(k => setNavState(k, ''));
        }
        const n = state.exploration.targets.length;
        setNavState('exploration', n ? `${n} TARGETS` : '', 'neutral');
    }

    // ═══════════════════════════════════════════════════════
    // OVERVIEW
    // ═══════════════════════════════════════════════════════
    function loadSupply() {
        const btn = $('#btnRefreshSupply');
        setLoading(btn, true);
        const gen = state.gen;
        if (!state.supply) {
            // Fresh state (first load or demo switch): clear anything from a previous state.
            $('#scStatus').dataset.tone = 'neutral';
            $('#scStatusText').textContent = 'Loading…';
            $('#scStatusSub').textContent = '';
            $('#scDecision').dataset.tone = 'neutral';
            $('#scDecision').innerHTML = '<span class="label">Decision</span><div class="hero-val">Loading…</div>';
            $('#scChain').innerHTML = '';
            $('#scBridge').innerHTML = loadingHTML('Loading supply outlook…');
            ['#scBridgeAside', '#scContingencyChip', '#scNextTarget', '#scDrivers', '#scReasons', '#scDecisionDetail'].forEach(s => { $(s).innerHTML = ''; });
            $('#scFlipPanel').hidden = true;
            $('#btnViewTarget').disabled = true;
        }
        const p = api.get(`/api/supply-command?mine_id=${enc(MINE_ID)}`).then(s => {
            if (gen !== state.gen) return;
            state.supply = s;
            renderOverview();
            state.loaded['supply-command'] = true;
            if (state.loaded.exploration) renderFallback();
        }).catch(e => {
            if (gen !== state.gen) return;
            state.supply = null;
            $('#scStatus').dataset.tone = 'bad';
            $('#scStatusText').innerHTML = `${iconHTML('bad')}UNAVAILABLE`;
            $('#scStatusSub').textContent = e.userMessage || 'Supply outlook unavailable.';
            $('#scDecision').dataset.tone = 'neutral';
            $('#scDecision').innerHTML = '<span class="label">Decision</span><div class="hero-val">UNAVAILABLE</div>';
            $('#scChain').innerHTML = '';
            $('#scBridge').innerHTML = errorHTML('Supply outlook unavailable.', e, 'supply-command');
            ['#scNextTarget', '#scDrivers', '#scReasons', '#scDecisionDetail'].forEach(s => { $(s).innerHTML = ''; });
            $('#scProvenance').innerHTML = badgeHTML('UNAVAILABLE');
            $('#btnViewTarget').disabled = true;
        }).finally(() => {
            setLoading(btn, false);
            updateDataModePill();
            updateNavStates();
        });
        state.supplyPromise = p;
        return p;
    }

    function renderOverview() {
        const s = state.supply || {};
        const p50 = num(s.p50_tonnes), target = num(s.target_tonnes), gap = num(s.gap_p50_tonnes);
        const fp = s.forecast_period || {};
        $('#scHorizon').textContent = `${fp.days ? `${fp.days}-DAY FORECAST` : 'FORECAST'}${fp.start ? ` ${fp.start} → ${fp.end || '?'}` : ''} · ${s.decision_horizon ? upperHuman(s.decision_horizon) + ' HORIZON' : 'HORIZON N/A'}`;
        $('#scProvenance').innerHTML = provChips(s.provenance);

        // Supply status (backend vocabulary)
        const ss = SUPPLY_STATUS[norm(s.supply_status)] || [s.supply_status ? upperHuman(s.supply_status) : 'N/A', 'neutral'];
        $('#scStatus').dataset.tone = ss[1];
        $('#scStatusText').innerHTML = `${iconHTML(ss[1])}${esc(ss[0])}`;
        $('#scStatusSub').innerHTML = gap !== null
            ? `P50 <b class="mono">${esc(fmtT(p50))}</b> · required <b class="mono">${esc(fmtT(target))}</b> · gap <b class="mono">${esc(fmtT(gap))}</b>`
            : '';

        // Decision
        const dec = normDecision(s);
        const di = decisionInfo(dec && dec.state);
        const hero = $('#scDecision');
        hero.dataset.tone = di.tone;
        hero.innerHTML = `<span class="label">Decision</span>
            <div class="hero-val">${iconHTML(di.tone)}${esc(di.title)}</div>
            <div class="hero-sub">${s.decision_horizon ? `${esc(upperHuman(s.decision_horizon))} horizon` : ''}${s.human_review_required === true ? ' · human review required' : ''}</div>`;
        if (dec && !state.decision.current) state.decision.current = dec;

        $('#scChain').innerHTML = chainHTML(s);
        renderBridge(s);

        const next = pick(s, 'next_target', 'next_target_id');
        $('#btnViewTarget').disabled = !next;
        $('#btnViewTarget').dataset.whyTarget = next || '';
        renderNextTarget(s);

        renderDriverBars($('#scDrivers'), s.primary_drivers);
        renderOverviewReasons(s);
        renderSupplyFlip(s.decision_flip);
    }

    function chainHTML(s) {
        const p10 = num(s.p10_tonnes), p50 = num(s.p50_tonnes), p90 = num(s.p90_tonnes), target = num(s.target_tonnes), gap = num(s.gap_p50_tonnes);
        const withheld = s.selection_status === 'REVIEW_REQUIRED';
        const rec = num(s.expected_recovery_tonnes), res = num(s.expected_residual_gap_tonnes), worst = num(s.worst_case_residual_gap_tonnes);
        const di = decisionInfo(s.decision_state);
        const next = s.next_target;
        const days = (s.forecast_period && s.forecast_period.days) || null;
        const nodes = [
            { k: 'Forecast P50', v: fmtT(p50), sub: p10 !== null && p90 !== null ? `P10–P90 ${fmtT(p10)} – ${fmtT(p90)}` : '', tone: s.forecast_applicability === 'LOW' ? 'warn' : 'info', go: 'production-risk' },
            { k: 'Required', v: fmtT(target), sub: days ? `${days}-day plan` : 'plan', tone: 'neutral', go: 'production-risk' },
            { k: 'Supply Gap', v: fmtT(gap), sub: gap > 0 ? `${fmtNum(s.gap_pct, 1)}% · risk ${norm(s.risk_state)}` : 'P50 meets required', tone: gap === null ? 'neutral' : gap > 0 ? riskTone(s.risk_state) : 'ok', go: 'production-risk' },
            { k: 'Recovery', v: withheld ? 'WITHHELD' : (rec !== null ? fmtSigned(rec) : 'N/A'), sub: withheld ? 'no eligible portfolio' : (s.selected_portfolio ? `portfolio ${s.selected_portfolio}` : ''), tone: withheld ? 'bad' : 'info', go: 'recovery' },
            { k: 'Residual Gap', v: res !== null ? fmtT(res) : (withheld ? 'NOT COMPUTED' : 'N/A'), sub: worst !== null ? `worst tested ${fmtT(worst)}` : '', tone: res === null ? (withheld ? 'bad' : 'neutral') : (res > 0 ? 'bad' : 'ok'), go: 'recovery' },
            { k: 'Decision', v: di.short, sub: s.decision_horizon ? `${human(s.decision_horizon).toLowerCase()} horizon` : '', tone: di.tone, go: 'recovery' },
            { k: 'Next Target', v: next || 'NONE', sub: next ? `priority ${fmtNum(s.target_priority, 1)}` : 'contingency not active', tone: next ? 'warn' : 'neutral', go: 'targets' },
        ];
        return nodes.map((n, i) => `<li>
            <button type="button" class="chain-node" data-tone="${n.tone}" data-goto="${n.go}">
                <span class="chain-k"><span class="chain-n">${i + 1}</span>${esc(n.k)}</span>
                <span class="chain-v">${iconHTML(n.tone)}<span class="mono">${esc(n.v)}</span></span>
                <span class="chain-sub">${esc(n.sub)}</span>
            </button>
        </li>`).join('');
    }

    // Supply bridge: required vs P50 + expected recovery; the remainder is the residual gap.
    function renderBridge(s) {
        const el = $('#scBridge');
        const t = num(s.target_tonnes), p50 = num(s.p50_tonnes), p10 = num(s.p10_tonnes), p90 = num(s.p90_tonnes);
        const withheld = s.selection_status === 'REVIEW_REQUIRED';
        const rec = withheld ? null : num(s.expected_recovery_tonnes);
        const res = num(s.expected_residual_gap_tonnes);
        if (t === null || p50 === null) { el.innerHTML = '<div class="empty-state">Forecast not returned.</div>'; $('#scBridgeAside').innerHTML = ''; return; }
        const recv = rec !== null && rec > 0 ? rec : 0;
        const after = p50 + recv;
        const max = Math.max(t, p90 ?? 0, after) * 1.08;
        const x = v => `${(v / max * 100).toFixed(2)}%`;
        const w = v => `${(Math.max(0, v) / max * 100).toFixed(2)}%`;
        const gapW = t - after;
        const gapLabel = withheld ? `GAP ${fmtT(s.gap_p50_tonnes)} · recovery withheld` : `RESIDUAL ${fmtT(res ?? gapW)}`;
        el.innerHTML = `<div class="bridge" role="img" aria-label="Required ${esc(fmtT(t))}; forecast P50 ${esc(fmtT(p50))}${recv ? `; expected recovery ${esc(fmtT(recv))}` : ''}; ${esc(gapLabel)}">
            <div class="br-row">
                <span class="br-label">Required</span>
                <div class="br-track">
                    <div class="br-seg br-req" style="left:0;width:${w(t)}"><span>${esc(fmtT(t))}</span></div>
                    <div class="br-line" style="left:${x(t)}"></div>
                </div>
            </div>
            <div class="br-row">
                <span class="br-label">Forecast${recv ? ' + recovery' : ''}</span>
                <div class="br-track">
                    <div class="br-seg br-p50" style="left:0;width:${w(p50)}"><span>P50 ${esc(fmtT(p50))}</span></div>
                    ${recv ? `<div class="br-seg br-rec" style="left:${x(p50)};width:${w(recv)}" title="Expected recovery ${esc(fmtT(recv))}"></div>` : ''}
                    ${gapW > 0 ? `<div class="br-seg br-gap" style="left:${x(after)};width:${w(gapW)}" title="${esc(gapLabel)}"><span>${esc(gapLabel)}</span></div>` : ''}
                    ${p10 !== null && p90 !== null ? `<div class="br-whisker" style="left:${x(p10)};width:${w(p90 - p10)}" title="P10–P90 ${esc(fmtT(p10))} – ${esc(fmtT(p90))}"></div>` : ''}
                    <div class="br-line" style="left:${x(t)}"></div>
                </div>
            </div>
            <div class="br-axis"><span>0</span><span>${esc(fmtT(max / 2))}</span><span>${esc(fmtT(max))}</span></div>
        </div>
        <div class="chart-legend">
            <span><i class="lg-sw lg-p50"></i>Forecast P50</span>
            ${recv ? `<span><i class="lg-sw lg-rec"></i>Expected recovery <b class="mono">${esc(fmtSigned(recv))}</b> (${esc(s.selected_portfolio || '')})</span>` : ''}
            <span><i class="lg-sw lg-gap"></i>${withheld ? 'Supply gap' : 'Residual gap'}</span>
            <span><i class="lg-whisker"></i>P10–P90</span>
            <span><i class="lg-req"></i>Required</span>
        </div>
        <div class="mini-grid mini-4">
            ${miniHTML('Residual · expected', withheld ? 'N/A' : fmtT(res), withheld ? 'bad' : (res > 0 ? 'bad' : 'ok'))}
            ${miniHTML('Residual · worst tested', withheld ? 'N/A' : fmtT(s.worst_case_residual_gap_tonnes), withheld ? 'bad' : (num(s.worst_case_residual_gap_tonnes) > 0 ? 'bad' : 'ok'))}
            ${miniHTML('Worst scenario', s.worst_case_scenario ? upperHuman(s.worst_case_scenario) : 'N/A', 'neutral')}
            ${miniHTML('Portfolio', withheld ? 'WITHHELD' : (s.selected_portfolio || 'N/A'), withheld ? 'bad' : 'info')}
        </div>`;
        $('#scBridgeAside').innerHTML = withheld ? chip('RECOVERY WITHHELD', 'bad')
            : (res !== null ? (res > 0 ? chip(`RESIDUAL ${fmtT(res)}`, 'bad') : chip('GAP CLOSED', 'ok')) : '');
    }

    function priorityBarHTML(v, label = 'Exploration priority') {
        const n = num(v);
        return `<div class="prio"><span class="label">${esc(label)}</span>
            <div class="prio-track"><div class="prio-fill" style="width:${clampPct(n)}%"></div></div>
            <span class="prio-val mono">${n !== null ? esc(fmtNum(n, 1)) : 'N/A'}</span></div>`;
    }

    function renderNextTarget(s) {
        const el = $('#scNextTarget');
        const next = s.next_target;
        const d = s.selected_target_detail || {};
        const review = norm(s.decision_state) === 'REVIEW_REQUIRED';
        const active = s.strategic_requirement && s.strategic_requirement.active;
        $('#scContingencyChip').innerHTML = review ? chip('WITHHELD', 'bad') : (active ? chip('ACTIVE', 'warn') : chip('NOT ACTIVE', 'ok'));
        if (!next) {
            el.innerHTML = `<div class="nt-empty">
                <div class="nt-empty-val">${review ? `${iconHTML('bad')}Review required` : `${iconHTML('ok')}No target nominated`}</div>
                <div class="btn-row"><button type="button" class="btn btn-ghost" data-goto="targets">View Target Ranking</button></div>
            </div>`;
            return;
        }
        const lvl = maturityLevel(d.evidence_level);
        el.innerHTML = `<div class="nt-head">
                <div><span class="label">Next target</span><div class="nt-id mono">${esc(next)}</div></div>
                <div class="nt-meta">${d.target_context ? chip(upperHuman(d.target_context), 'neutral') : ''}${num(d.distance_to_demo_mine_km) !== null ? `<span class="mono muted">${esc(fmtNum(d.distance_to_demo_mine_km, 1))} km to supply point</span>` : ''}</div>
            </div>
            ${priorityBarHTML(s.target_priority ?? d.exploration_priority)}
            <div class="mini-grid">
                ${miniHTML('Prospectivity', fmtNum(d.prospectivity_rank, 1), 'info', 'relative rank')}
                ${miniHTML('Applicability', d.applicability, applTone(d.applicability))}
                ${miniHTML('Uncertainty', d.uncertainty, uncTone(d.uncertainty))}
                ${miniHTML('Evidence', lvl !== null ? `L${lvl}` : 'N/A', evTone(lvl))}
                ${miniHTML('Strategic', d.strategic_relevance, stratTone(d.strategic_relevance))}
                ${miniHTML('Subsurface', d.subsurface_status ? upperHuman(d.subsurface_status) : 'N/A', norm(d.subsurface_status) === 'UNAVAILABLE' ? 'warn' : 'info')}
            </div>
            <div class="btn-row">
                <button type="button" class="btn btn-primary" data-why-target="${esc(next)}">Why This Target?</button>
                <button type="button" class="btn btn-ghost" data-open-target="${esc(next)}">View on Map</button>
            </div>`;
    }

    function renderOverviewReasons(s) {
        const rr = asArray(s.review_reasons);
        if (rr.length) {
            $('#scReasonsTitle').textContent = 'Review Reasons';
            $('#scReasons').innerHTML = reasonListHTML(rr, 'bad');
        } else {
            $('#scReasonsTitle').textContent = 'Decision Basis';
            const codes = asArray(s.reason_codes);
            $('#scReasons').innerHTML = codes.length
                ? `<div class="code-row">${codes.map(c => `<span class="code">${esc(human(c))}</span>`).join('')}</div>`
                : '<div class="empty-state">No reason codes returned.</div>';
        }
        const sr = s.strategic_requirement || {};
        const iv = intervalInfo(s);
        $('#scDecisionDetail').innerHTML = dlHTML([
            ['Decision summary', esc(s.decision_summary ? reasonText(s.decision_summary) : '')],
            ['Portfolio selection', esc(s.selection_explanation || '')],
            ['Action note', esc(s.action_note || '')],
            ['Worst tested scenario', esc(s.worst_case_scenario ? human(s.worst_case_scenario) : '')],
            ['Materiality / worst-case tolerance', num(sr.materiality_tonnes) !== null ? `<span class="mono">${esc(fmtT(sr.materiality_tonnes))} / ${esc(fmtT(sr.worst_case_tolerance_tonnes))}</span>` : ''],
            ['Strategic requirement', sr.active ? `<span class="mono">${esc(fmtT(sr.cumulative_requirement_expected_tonnes))}</span> expected · <span class="mono">${esc(fmtT(sr.cumulative_requirement_worst_case_tonnes))}</span> worst case over ${esc(sr.strategic_periods)} periods` : ''],
            ['Requirement note', esc(sr.requirement_note || '')],
            ['Forecast basis', esc(s.forecast_basis || '')],
            ['Interval', esc(iv.text + (iv.window ? ` (${iv.window})` : ''))],
            ['Risk policy', esc(riskPolicyText(s))],
            ['Result status', esc(s.status ? upperHuman(s.status) : '')],
        ]);
    }

    // ── Decision flip (backend /api/decision/flip result) ───
    function flipSideHTML(label, d) {
        if (!d) return `<div class="flip-side" data-tone="neutral"><span class="label">${esc(label)}</span><div class="flip-title">NOT RETURNED</div></div>`;
        const di = decisionInfo(d.decision_state);
        const port = d.selection_status === 'REVIEW_REQUIRED' ? 'withheld (review)' : (d.selected_portfolio || '—');
        return `<div class="flip-side" data-tone="${di.tone}">
            <span class="label">${esc(label)}</span>
            <div class="flip-title">${iconHTML(di.tone)}${esc(di.title)}</div>
            ${dlHTML([
                ['P50 / required', d.forecast ? `<span class="mono">${esc(fmtT(d.forecast.p50_tonnes))} / ${esc(fmtT(d.target_tonnes))}</span>` : ''],
                ['Supply gap', `<span class="mono">${esc(fmtT(d.baseline_gap_tonnes))}</span>`],
                ['Portfolio', `<span class="mono">${esc(port)}</span>`],
                ['Residual (exp. / worst)', `<span class="mono">${esc(fmtT(d.expected_residual_gap_tonnes))} / ${esc(fmtT(d.worst_case_residual_gap_tonnes))}</span>`],
                ['Exploration contingency', d.exploration_contingency ? chip('ON', 'warn') : chip('OFF', 'neutral')],
                ['Next target', `<span class="mono">${esc(d.next_target || 'none')}</span>`],
            ])}
            ${asArray(d.review_reasons).length ? reasonListHTML(d.review_reasons, 'bad') : ''}
        </div>`;
    }

    function flipResultHTML(f) {
        if (!f) return '<div class="empty-state">No decision flip returned.</div>';
        const changed = asArray(f.changed_inputs);
        const flipped = f.flipped === true;
        const deltas = f.deltas && typeof f.deltas === 'object' ? Object.entries(f.deltas).filter(([, v]) => num(v) !== null) : [];
        const DL = { forecast_p50_tonnes: 'P50', baseline_gap_tonnes: 'Supply gap', expected_residual_gap_tonnes: 'Residual', worst_case_residual_gap_tonnes: 'Worst residual' };
        return `
            <div class="flip-banner" data-tone="${flipped ? 'warn' : 'ok'}" role="status">
                ${iconHTML(flipped ? 'warn' : 'ok')}<b>DECISION ${flipped ? 'FLIPPED' : 'UNCHANGED'}</b>
                ${f.transition ? `<span class="mono muted">${esc(String(f.transition).replace(/_/g, ' '))}</span>` : ''}
            </div>
            ${deltas.length ? `<div class="delta-row">${deltas.map(([k, v]) => `<span class="delta"><span class="label">Δ ${esc(DL[k] || human(k))}</span><b class="mono ${num(v) > 0 ? 'up' : (num(v) < 0 ? 'down' : '')}">${esc(fmtSigned(v))}</b></span>`).join('')}</div>` : ''}
            <div class="flip-sides">
                ${flipSideHTML('Baseline', f.baseline)}
                <div class="flip-arrow" aria-hidden="true">&rarr;</div>
                ${flipSideHTML('Perturbed', f.perturbed)}
            </div>
            <details class="disclosure">
                <summary>Changed Inputs &amp; Priority Changes</summary>
                <div class="disclosure-body">
                    ${changed.length ? `<table class="table"><thead><tr><th>Input</th><th>Baseline</th><th>Perturbed</th></tr></thead><tbody>${changed.map(c => `<tr><td>${esc(human(c.input))}</td><td class="mono">${esc(c.baseline ?? '—')}</td><td class="mono">${esc(c.perturbed ?? '—')}</td></tr>`).join('')}</tbody></table>` : '<div class="muted">No inputs changed.</div>'}
                    ${flipPriorityHTML(f)}
                </div>
            </details>`;
    }

    // Target priority under both supply states (backend-computed; prospectivity itself is unchanged).
    function flipPriorityHTML(f) {
        const rows = asArray(f.exploration_priority_changes);
        if (!rows.length) return '';
        const mv = r => r.rank_perturbed < r.rank_baseline ? `<span class="up">▲${r.rank_baseline - r.rank_perturbed}</span>`
            : (r.rank_perturbed > r.rank_baseline ? `<span class="down">▼${r.rank_perturbed - r.rank_baseline}</span>` : '=');
        return `<div class="table-wrap"><table class="table"><thead><tr><th>Target</th><th>Prospectivity</th><th>Priority</th><th>Rank</th><th>Strategic relevance</th></tr></thead><tbody>
            ${rows.map(r => `<tr><td class="mono">${esc(r.target_id)}</td><td class="mono">${fmtNum(r.prospectivity_rank, 1)}</td>
                <td class="mono">${fmtNum(r.priority_baseline, 1)} → ${fmtNum(r.priority_perturbed, 1)}</td>
                <td class="mono">${r.rank_baseline} → ${r.rank_perturbed} ${mv(r)}</td>
                <td>${esc(r.strategic_relevance_baseline)} → ${esc(r.strategic_relevance_perturbed)}</td></tr>`).join('')}
            </tbody></table></div>
            ${f.priority_note ? `<div class="note">${esc(f.priority_note)}</div>` : ''}`;
    }

    function renderSupplyFlip(f) {
        const panel = $('#scFlipPanel');
        if (!panel) return;
        panel.hidden = !f;
        if (!f) return;
        $('#scFlip').innerHTML = flipResultHTML(f);
        // Start the Recovery-screen sliders at the backend's documented demo perturbation.
        if (!state.decision.flipRuns) {
            const map = { rainfall_7d_mm: '#flipRainfall', equipment_availability: '#flipEquip', blast_delay_h: '#flipBlast' };
            asArray(f.changed_inputs).forEach(c => {
                const el = map[c.input] && $(map[c.input]);
                if (el && num(c.perturbed) !== null) { el.value = c.perturbed; el.dispatchEvent(new Event('input')); }
            });
        }
    }

    // ── Demo-state selector (backend list; no client logic) ─
    async function loadDemoStates() {
        const sel = $('#demoSelect');
        try {
            const d = await api.get('/api/demo/scenarios', { timeout: 8000 });
            const items = asArray(pick(d, 'scenarios'));
            if (!items.length) return;
            sel.innerHTML = items.map(x => `<option value="${esc(x.mine_id)}">${esc(x.mine_id)}${x.demo_state ? ` · ${esc(human(x.demo_state))}` : ' · current synthetic state'}</option>`).join('');
            sel.value = MINE_ID;
            items.forEach(x => { const o = sel.querySelector(`option[value="${CSS.escape(x.mine_id)}"]`); if (o && x.label) o.title = x.label; });
        } catch (_) {
            sel.title = 'Demo state list unavailable; showing DEMO_MINE.';
        }
    }

    function switchMine(id) {
        if (!id || id === MINE_ID) return;
        MINE_ID = id;
        state.gen++;
        $('#telMine').textContent = id;
        // Drop every mine-dependent result so nothing stale is shown for the new state.
        state.supply = null;
        state.supplyPromise = null;
        state.recovery = null;
        state.contingency = null;
        state.decision = { current: null, previous: null, flipRuns: 0 };
        Object.assign(state.exploration, { targets: [], targetsMeta: null, targetsPromise: null, selectedId: null, detail: {}, pending: {}, subsurface: {}, query: null });
        state.production = { forecast: null, history: null };
        state.trust.recovery = null;
        state.flipPreset = false;
        state.loaded = {};
        $('#flipResult').innerHTML = '<div class="empty-state">Set conditions, then compare with the baseline decision.</div>';
        $('#targetCard').innerHTML = '<div class="empty-state">Select a target on the map or in the list.</div>';
        $('#evidenceBody').innerHTML = '<div class="empty-state">Select a target to inspect its evidence.</div>';
        $('#evidenceTargetId').textContent = '—';
        $('#evidenceAside').innerHTML = '';
        $('#whyPanel').innerHTML = loadingHTML();
        $('#tpContext').innerHTML = '';
        $('#rankList').innerHTML = loadingHTML('Loading targets…');
        $('#targetList').innerHTML = loadingHTML('Loading targets…');
        $('#prRangeChart').innerHTML = '';
        $('#prGapAside').innerHTML = '';
        $('#fallbackPanel').hidden = true;
        showToast(`Demo state ${id} — recomputed by the backend (synthetic data).`, 'info');
        updateNavStates();
        if (state.currentSection !== 'supply-command') loadSupply();
        SECTIONS[state.currentSection].load();
    }

    // ═══════════════════════════════════════════════════════
    // SHARED TARGET DATA (Exploration + Target Priority)
    // ═══════════════════════════════════════════════════════
    function ensureTargets() {
        const ex = state.exploration;
        if (ex.targetsPromise) return ex.targetsPromise;
        const gen = state.gen;
        ex.targetsPromise = api.get(`/api/exploration/targets?mine_id=${enc(MINE_ID)}`).then(resp => {
            if (gen !== state.gen) return;
            const rawList = Array.isArray(resp) ? resp : asArray(pick(resp, 'targets', 'items', 'results'));
            ex.targetsMeta = Array.isArray(resp) ? null : resp;
            ex.targets = rawList.map(normTarget).filter(t => t && t.id)
                .sort((a, b) => (num(a.raw.priority_rank) ?? 1e9) - (num(b.raw.priority_rank) ?? 1e9));
            updateNavStates();
        }).catch(e => {
            if (gen === state.gen) ex.targetsPromise = null;
            throw e;
        });
        return ex.targetsPromise;
    }

    function getDetail(id) {
        const ex = state.exploration;
        if (ex.detail[id]) return Promise.resolve(ex.detail[id]);
        if (!ex.pending[id]) {
            const gen = state.gen;
            ex.pending[id] = api.get(`/api/exploration/targets/${enc(id)}?mine_id=${enc(MINE_ID)}`)
                .then(d => { if (gen === state.gen) ex.detail[id] = d; return d; })
                .finally(() => { if (gen === state.gen) delete ex.pending[id]; });
        }
        return ex.pending[id];
    }

    function mergedTarget(id) {
        const ex = state.exploration;
        const t = ex.targets.find(x => x.id === id);
        const d = ex.detail[id];
        if (!t && !d) return null;
        return normTarget({ ...(t ? t.raw : {}), ...(d || {}) });
    }

    function rankRowsHTML(ts, { full = false } = {}) {
        if (!ts.length) return '<div class="empty-state">No exploration targets returned.</div>';
        const sel = state.exploration.selectedId;
        const contingency = state.exploration.targetsMeta && state.exploration.targetsMeta.selected_target;
        const head = full
            ? '<span>#</span><span>Target</span><span>Priority</span><span>Prosp.</span><span>Appl.</span><span>Unc.</span><span>Evid.</span><span>Strategic</span><span>Dist.</span>'
            : '<span>#</span><span>Target</span><span>Priority</span><span>Appl.</span><span>Unc.</span><span>Evid.</span>';
        return `<div class="rank-head" aria-hidden="true">${head}</div>` + ts.map(t => {
            const r = t.raw || {};
            const style = targetStyle(t);
            const gated = r.eligible_for_contingency === false;
            const lvl = maturityLevel(t.maturity);
            const strat = t.strategic ? upperHuman(typeof t.strategic === 'object' ? pick(t.strategic, 'level', 'label') : t.strategic) : 'N/A';
            const gateWhy = asArray(r.gate_blocked_reasons).join('; ');
            return `<button type="button" class="rank-row${t.id === sel ? ' selected' : ''}${gated ? ' gated' : ''}" data-target="${esc(t.id)}" role="option" aria-selected="${t.id === sel}">
                <span class="rk mono">${esc(r.priority_rank ?? '')}</span>
                <span class="rid"><span class="tmk tmk-${style}${isHigh(t.unc.level) ? ' tmk-uncertain' : ''}" aria-hidden="true">${style === 'caution' ? '!' : 'T'}</span><span class="mono">${esc(t.id)}</span>${t.id === contingency ? '<span class="star" title="Selected contingency target">★</span>' : ''}${gated ? `<span class="gate" title="${esc('Gated: ' + gateWhy)}">GATED</span>` : ''}</span>
                <span class="rbar"><span class="rbar-track"><span class="rbar-fill" style="width:${clampPct(t.priority)}%"></span></span><span class="rbar-val mono">${t.priority !== null ? esc(fmtNum(t.priority, 1)) : 'N/A'}</span></span>
                ${full ? `<span class="mono">${t.pros.score !== null ? esc(fmtNum(t.pros.score, 1)) : 'N/A'}</span>` : ''}
                <span class="lvl" data-tone="${applTone(t.appl.level)}">${esc(t.appl.level || 'N/A')}</span>
                <span class="lvl" data-tone="${uncTone(t.unc.level)}">${esc(t.unc.level || 'N/A')}</span>
                <span class="lvl" data-tone="${evTone(lvl)}">${lvl !== null ? 'L' + lvl : 'N/A'}</span>
                ${full ? `<span class="lvl" data-tone="${stratTone(strat)}">${esc(strat)}</span><span class="mono muted">${t.distanceKm !== null ? esc(fmtNum(t.distanceKm, 1)) + ' km' : ''}</span>` : ''}
            </button>`;
        }).join('');
    }

    function syncSelection() {
        const id = state.exploration.selectedId;
        $$('.rank-row').forEach(el => {
            const on = el.dataset.target === id;
            el.classList.toggle('selected', on);
            el.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        $$('.tmk-map').forEach(el => el.classList.toggle('tmk-selected', el.textContent === id));
        const row = $(`#targetList .rank-row[data-target="${CSS.escape(id || '')}"]`);
        if (row && state.currentSection === 'exploration') row.scrollIntoView({ block: 'nearest' });
    }

    async function selectTarget(id, { fly = true } = {}) {
        const ex = state.exploration;
        ex.selectedId = id;
        syncSelection();
        const t = ex.targets.find(x => x.id === id);
        const map = state.maps.exploration;
        if (t) {
            highlightRegion(t);
            if (fly && map && t.lat !== null && t.lon !== null) map.flyTo([t.lat, t.lon], Math.max(map.getZoom(), 10), { duration: 0.8 });
        }
        renderRailCard(t || null, { loading: !ex.detail[id] });
        $('#evidenceTargetId').textContent = id;
        $('#evidenceAside').innerHTML = '';
        $('#evidenceBody').innerHTML = `<div id="evMain">${loadingHTML(`Loading evidence for ${id}…`)}</div><div id="ssWrap"></div>`;
        loadSubsurface(id);
        if (state.loaded.targets) renderWhy(id);
        let d;
        try {
            d = await getDetail(id);
        } catch (e) {
            if (ex.selectedId !== id) return;
            if (t) renderRailCard(t, { error: e }); else $('#targetCard').innerHTML = errorHTML(`Target ${id} unavailable.`, e);
            const main = $('#evMain');
            if (main) main.innerHTML = errorHTML('Target evidence unavailable.', e);
            return;
        }
        if (ex.selectedId !== id) return;
        const merged = mergedTarget(id);
        if (!merged) return;
        if (!t && merged.lat !== null) {
            highlightRegion(merged);
            if (fly && map) map.flyTo([merged.lat, merged.lon], Math.max(map.getZoom(), 10), { duration: 0.8 });
        }
        renderRailCard(merged, {});
        const main = $('#evMain');
        if (main) main.innerHTML = evidenceMainHTML(merged, d);
    }

    // ═══════════════════════════════════════════════════════
    // 1 · EXPLORATION
    // ═══════════════════════════════════════════════════════
    function initExplorationMap() {
        if (state.maps.exploration || !window.L) {
            if (!window.L) $('#explorationMap').innerHTML = '<div class="error-state">Map library failed to load (external CDN). Target list and evidence remain available.</div>';
            return;
        }
        const map = L.map('explorationMap', { center: [21.6, 79.9], zoom: 8, minZoom: 4, maxZoom: 16, zoomControl: true });
        map.zoomControl.setPosition('bottomright');
        state.maps.exploration = map;

        const satTile = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { attribution: 'Esri, USGS, AeroGRID, IGN', maxZoom: 18 });
        // Keyless Esri reference labels (the previous CARTO label tiles now require an API key).
        const satLabels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', { attribution: 'Esri', maxZoom: 18 });
        const lstTile = L.tileLayer('https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_Land_Surface_Temp_Day/default/2024-05-01/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png', { attribution: 'NASA GIBS — MODIS LST 2024-05-01 (context layer, not the model feature)', maxNativeZoom: 7, maxZoom: 18, opacity: 0.9 });
        const streetTile = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: 'OpenStreetMap', maxZoom: 18 });

        state.maps.baseLayers = {
            satellite: L.layerGroup([satTile, satLabels]),
            thermal: L.layerGroup([satTile, lstTile, satLabels]),
            street: streetTile,
        };
        state.maps.currentBase = state.maps.baseLayers.satellite.addTo(map);

        // External tile failures are visual only: warn, never block or imply a model failure.
        const tileWarn = name => () => mapNotice(`${name} tiles unavailable — targets and ranks unaffected.`);
        satTile.on('tileerror', tileWarn('Satellite basemap'));
        lstTile.on('tileerror', tileWarn('MODIS LST context-layer'));
        streetTile.on('tileerror', tileWarn('Street basemap'));

        $$('.seg-btn[data-layer]').forEach(btn => btn.addEventListener('click', () => {
            const next = state.maps.baseLayers[btn.dataset.layer];
            if (!next || next === state.maps.currentBase) return;
            map.removeLayer(state.maps.currentBase);
            state.maps.currentBase = next.addTo(map);
            $$('.seg-btn[data-layer]').forEach(b => { b.classList.toggle('active', b === btn); b.setAttribute('aria-pressed', b === btn ? 'true' : 'false'); });
        }));

        map.on('mousemove', e => { $('#cursorCoordReadout').textContent = `${e.latlng.lat.toFixed(4)}°N, ${e.latlng.lng.toFixed(4)}°E`; });
        map.on('click', e => {
            $('#queryLat').value = e.latlng.lat.toFixed(4);
            $('#queryLon').value = e.latlng.lng.toFixed(4);
            queryCoordinate(e.latlng.lat, e.latlng.lng);
        });

        $('#toggleSurface').addEventListener('change', e => setSurfaceVisible(e.target.checked));
        $('#btnResetMap').addEventListener('click', fitTargets);
    }

    function mapNotice(text) {
        const el = $('#mapNotice');
        if (!el || !el.hidden) return;   // show once per load
        el.textContent = text;
        el.hidden = false;
    }

    async function loadExploration() {
        initExplorationMap();
        const ex = state.exploration;
        const list = $('#targetList');
        if (!ex.targets.length) list.innerHTML = loadingHTML('Loading targets…');
        try {
            await ensureTargets();
        } catch (e) {
            list.innerHTML = errorHTML('Exploration targets unavailable.', e, 'exploration');
            $('#targetCount').textContent = '—';
            $('#exProvenance').innerHTML = badgeHTML('UNAVAILABLE');
            $('#surfaceSourceLabel').textContent = 'Surface: unavailable';
            return;
        }
        state.loaded.exploration = true;
        const meta = ex.targetsMeta || {};
        $('#exProvenance').innerHTML = provChips(meta.provenance);
        renderTargetList();
        renderTargetMarkers();
        renderFallback();
        await loadSurface();
        const fly = ex.flyOnLoad;
        ex.flyOnLoad = false;
        const id = ex.selectedId || meta.selected_target;
        if (id) selectTarget(id, { fly: false });
        // The map may have been created while its section was hidden; size it, then frame it.
        setTimeout(() => {
            const map = state.maps.exploration;
            if (!map) return;
            map.invalidateSize();
            const t = fly && ex.targets.find(x => x.id === ex.selectedId);
            if (t && t.lat !== null) map.setView([t.lat, t.lon], 11);
            else fitTargets();
        }, 150);
    }

    // Prospectivity surface: the backend's cached ~1 km rank grid (/api/exploration/grid).
    // Intensity = prospectivity_rank / 100 (a relative rank, not a probability).
    // Nothing is interpolated or generated client-side.
    async function loadSurface() {
        let pts = null, prov = null;
        try {
            const grid = await api.get('/api/exploration/grid?stride=3', { retries: 0, timeout: 15000 });
            pts = asArray(pick(grid, 'points'));
            prov = pick(grid, 'provenance');
        } catch (_) { pts = null; }
        state.exploration.surface = pts;
        const map = state.maps.exploration;
        if (state.maps.surfaceLayer && map) { map.removeLayer(state.maps.surfaceLayer); state.maps.surfaceLayer = null; }
        if (!pts || !pts.length || !map || !L.heatLayer) {
            $('#surfaceSourceLabel').textContent = !pts ? 'Surface: unavailable' : 'Surface: heat-map plugin unavailable';
            $('#toggleSurface').disabled = true;
            return;
        }
        const heat = pts.map(p => {
            const lat = num(p.lat), lon = num(p.lon), rank = num(p.prospectivity_rank);
            return (lat !== null && lon !== null && rank !== null) ? [lat, lon, Math.max(0.02, Math.min(1, rank / 100))] : null;
        }).filter(Boolean);
        state.maps.surfaceLayer = L.heatLayer(heat, {
            radius: 14, blur: 12, maxZoom: 11, max: 1.0, minOpacity: 0.25,
            gradient: { 0.2: '#1e3a8a', 0.5: '#0891b2', 0.75: '#10b981', 0.9: '#f59e0b', 1.0: '#ef4444' },
        });
        $('#toggleSurface').disabled = false;
        $('#surfaceSourceLabel').innerHTML = `~1 km rank grid ${badgeHTML(prov ? prov.data_mode : 'CACHED')}`;
        setSurfaceVisible($('#toggleSurface').checked && state.currentSection === 'exploration');
    }

    function setSurfaceVisible(v) {
        const map = state.maps.exploration, layer = state.maps.surfaceLayer;
        if (!map || !layer) return;
        if (v && !map.hasLayer(layer)) layer.addTo(map);
        if (!v && map.hasLayer(layer)) map.removeLayer(layer);
    }

    function renderTargetList() {
        const ts = state.exploration.targets;
        $('#targetCount').textContent = `${ts.length} target${ts.length === 1 ? '' : 's'}`;
        $('#targetList').innerHTML = rankRowsHTML(ts);
    }

    function popupHTML(t) {
        const lvl = maturityLevel(t.maturity);
        return `<div class="map-popup">
            <div class="mp-title">${esc(t.id)}${t.raw && t.raw.priority_rank ? ` <span class="muted">#${esc(t.raw.priority_rank)}</span>` : ''}</div>
            <div>Prospectivity <b>${t.pros.score !== null ? esc(fmtNum(t.pros.score, 1)) : 'N/A'}</b></div>
            <div>Applicability <b>${esc(t.appl.level || 'N/A')}</b> · Uncertainty <b>${esc(t.unc.level || 'N/A')}</b></div>
            <div>Evidence <b>${lvl !== null ? 'L' + lvl : 'N/A'}</b>${t.priority !== null ? ` · Priority <b>${esc(fmtNum(t.priority, 1))}</b>` : ''}</div>
        </div>`;
    }

    function renderTargetMarkers() {
        const map = state.maps.exploration;
        if (!map) return;
        Object.values(state.maps.markers).forEach(m => map.removeLayer(m));
        state.maps.markers = {};
        state.exploration.targets.forEach(t => {
            if (t.lat === null || t.lon === null) return;
            const style = targetStyle(t);
            const icon = L.divIcon({
                className: 'tmk-wrap',
                html: `<span class="tmk tmk-${style}${isHigh(t.unc.level) ? ' tmk-uncertain' : ''} tmk-map${t.id === state.exploration.selectedId ? ' tmk-selected' : ''}">${esc(t.id)}</span>`,
                iconSize: null,
            });
            const m = L.marker([t.lat, t.lon], { icon, title: `Target ${t.id}`, alt: `Target ${t.id}`, keyboard: true, riseOnHover: true })
                .bindTooltip(popupHTML(t), { direction: 'top', offset: [0, -10], className: 'map-tip' })
                .on('click', () => selectTarget(t.id, { fly: false }));
            m.addTo(map);
            state.maps.markers[t.id] = m;
        });
    }

    function fitTargets() {
        const map = state.maps.exploration;
        const pts = state.exploration.targets.filter(t => t.lat !== null && t.lon !== null).map(t => [t.lat, t.lon]);
        // Padding keeps markers clear of the toolbar (top) and legend (bottom-left).
        if (map && pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.08), { maxZoom: 10, paddingTopLeft: [40, 60], paddingBottomRight: [40, 40] });
    }

    function highlightRegion(t) {
        const map = state.maps.exploration;
        if (!map) return;
        if (state.maps.highlight) { map.removeLayer(state.maps.highlight); state.maps.highlight = null; }
        const style = targetStyle(t);
        const color = style === 'caution' ? '#e0a43a' : '#ffffff';
        const opts = { color, weight: 2, fillColor: color, fillOpacity: 0.12, dashArray: isHigh(t.unc.level) ? '6 6' : null, className: 'target-highlight-shape' };
        if (t.geometry && typeof t.geometry === 'object') {
            try { state.maps.highlight = L.geoJSON(t.geometry, { style: () => opts }).addTo(map); } catch (_) { state.maps.highlight = null; }
        }
        if (!state.maps.highlight && t.lat !== null && t.lon !== null) {
            // Only draw an extent the backend supplied; otherwise a focus ring of fixed screen size.
            state.maps.highlight = t.radiusKm !== null
                ? L.circle([t.lat, t.lon], { ...opts, radius: t.radiusKm * 1000 }).addTo(map)
                : L.circleMarker([t.lat, t.lon], { ...opts, radius: 22, fillOpacity: 0.08 }).addTo(map);
        }
        $$('.tmk-map').forEach(el => el.classList.toggle('tmk-selected', el.textContent === t.id));
    }

    function evidenceRows(t) {
        const ev = t.evidence;
        const layers = [['surface', 'Surface'], ['geology', 'Geology'], ['subsurface', 'Subsurface']];
        const MATCH = {
            surface: n => /^(SURFACE|REMOTE|SATELLITE|SPECTRAL)/.test(n),
            geology: n => /GEOLOG/.test(n),
            subsurface: n => /SUBSURFACE|GEOPHYS|DRILL|GROUND/.test(n),
        };
        const truthy = av => {
            if (av === undefined) return true;
            if (typeof av === 'boolean') return av;
            const n = norm(av);
            return !/UNAVAIL|NONE|FALSE|MISSING|ABSENT|PENDING/.test(n) && n !== 'NO';
        };
        const status = key => {
            if (ev === undefined || ev === null) return null;
            if (Array.isArray(ev)) {
                const hit = ev.find(x => MATCH[key](norm(typeof x === 'string' ? x : pick(x, 'type', 'name', 'layer') || '')));
                if (!hit) return false;
                return typeof hit === 'string' ? true : truthy(pick(hit, 'available', 'status'));
            }
            const v = ev[key] ?? ev[`${key}_evidence`] ?? (key === 'geology' ? ev.geological : undefined);
            if (v === undefined || v === null) return null;
            if (typeof v === 'object') return truthy(pick(v, 'available', 'status'));
            return truthy(v);
        };
        return layers.map(([k, label]) => ({ key: k, label, ok: status(k) }));
    }

    function layerChipsHTML(t) {
        return `<div class="layers">${evidenceRows(t).map(e => {
            const tone = e.ok === true ? 'ok' : (e.ok === false ? 'bad' : 'neutral');
            return chip(`${e.label}${e.ok === null ? ' ?' : ''}`, tone, `${e.label} evidence: ${e.ok === true ? 'available' : (e.ok === false ? 'unavailable' : 'not reported')}`);
        }).join('')}</div>`;
    }

    function renderRailCard(t, { loading = false, error = null, query = false } = {}) {
        const card = $('#targetCard');
        if (!t) { card.innerHTML = loadingHTML('Loading target…'); return; }
        const style = targetStyle(t);
        const lvl = maturityLevel(t.maturity);
        const r = t.raw || {};
        const meta = state.exploration.targetsMeta || {};
        const isSel = !query && meta.selected_target === t.id;
        const gated = r.eligible_for_contingency === false;
        const warnings = t.warnings.filter(Boolean).map(reasonText);
        const strat = t.strategic ? upperHuman(typeof t.strategic === 'object' ? pick(t.strategic, 'level', 'label') : t.strategic) : null;
        card.dataset.style = style;
        card.innerHTML = `
            <div class="rc-head">
                <div><span class="label">${query ? 'Query point' : 'Selected target'}</span><div class="rc-id mono">${esc(t.id)}</div></div>
                <div class="rc-chips">
                    ${!query && r.priority_rank != null ? chip(`#${r.priority_rank} PRIORITY`, 'info') : ''}
                    ${isSel ? chip('CONTINGENCY TARGET', 'warn') : ''}
                    ${gated ? chip('GATED', 'warn', asArray(r.gate_blocked_reasons).join('; ')) : ''}
                    ${loading ? '<span class="badge badge-neutral">LOADING…</span>' : ''}
                    ${error ? '<span class="badge badge-unavailable">DETAIL UNAVAILABLE</span>' : ''}
                </div>
            </div>
            ${style === 'caution' ? `<div class="alert" data-tone="bad">${iconHTML('bad')}LOW APPLICABILITY — outside the model's supported domain</div>` : ''}
            ${query ? `<div class="alert" data-tone="info">${iconHTML('info')}${esc(statusText(t))}</div>` : ''}
            ${warnings.map(w => `<div class="alert" data-tone="warn">${iconHTML('warn')}${esc(w)}</div>`).join('')}
            <div class="mini-grid mini-3">
                ${miniHTML('Prospectivity', t.pros.score !== null ? fmtNum(t.pros.score, 1) : 'N/A', 'info', 'relative rank')}
                ${query ? miniHTML('Evidence', lvl !== null ? `L${lvl}` : 'N/A', evTone(lvl)) : miniHTML('Priority', t.priority !== null ? fmtNum(t.priority, 1) : 'N/A', 'info', r.priority_mode ? human(r.priority_mode).toLowerCase() : '')}
                ${query ? miniHTML('Subsurface', r.subsurface_status ? upperHuman(r.subsurface_status) : 'N/A', 'neutral') : miniHTML('Evidence', lvl !== null ? `L${lvl}` : 'N/A', evTone(lvl))}
                ${miniHTML('Applicability', t.appl.level || (t.inDomain === false ? 'OUT OF DOMAIN' : 'N/A'), applTone(t.appl.level))}
                ${miniHTML('Uncertainty', t.unc.level || 'N/A', uncTone(t.unc.level), num(r.rank_sd) !== null ? `rank SD ${fmtNum(r.rank_sd, 1)}` : '')}
                ${query ? miniHTML('In study area', r.in_study_area === true ? 'YES' : (r.in_study_area === false ? 'NO' : 'N/A'), r.in_study_area === false ? 'bad' : 'neutral') : miniHTML('Strategic', strat || 'N/A', stratTone(strat), t.distanceKm !== null ? `${fmtNum(t.distanceKm, 1)} km` : '')}
            </div>
            ${query ? '' : layerChipsHTML(t)}
            <div class="btn-row">
                ${query ? '' : `<button type="button" class="btn btn-primary btn-sm" data-why-target="${esc(t.id)}">Why This Target?</button>`}
                <button type="button" class="btn btn-ghost btn-sm" data-scroll="#evidencePanel">Inspect Evidence</button>
            </div>
            <div class="prov-row">${t.source ? badgeHTML(t.source) : ''}${t.window ? `<span class="badge badge-window">${esc(fmtWin(t.window))}</span>` : ''}</div>`;
    }

    // Backend evidence levels are L0..L4; the ladder index equals the level.
    const LADDER = ['Remote sensing', '+ Geological context', 'Ground / geochem / geophys', 'Drilling / assay', 'Resource / reserve'];

    function ladderHTML(level) {
        return `<ol class="ladder" aria-label="Evidence maturity">${LADDER.map((name, n) => {
            const cls = level === null ? '' : (n < level ? 'done' : (n === level ? 'current' : 'todo'));
            return `<li class="ladder-step ${cls}"${n === level ? ' aria-current="step"' : ''}><span class="ladder-n mono">L${n}</span><span class="ladder-name">${esc(name)}</span></li>`;
        }).join('')}</ol>`;
    }

    function evidenceMainHTML(t, d) {
        d = d || {};
        const r = t.raw || {};
        const lvl = maturityLevel(t.maturity);
        const feats = d.surface_evidence && d.surface_evidence.features;
        const geo = d.geological_evidence || d.geological_context || r.geological_context;
        const ground = asArray(r.observed_ground_evidence).length ? asArray(r.observed_ground_evidence) : asArray(d.subsurface_evidence && d.subsurface_evidence.records);
        const occ = asArray(d.documented_occurrences).length ? asArray(d.documented_occurrences) : asArray(d.documented_occurrences_within_1km);
        const ss = r.subsurface_status || (d.subsurface_evidence && d.subsurface_evidence.status);
        const why = asArray(d.why_this_target);
        const basis = asArray(d.evidence_basis);
        return `
        <div class="ev-grid">
            <div class="ev-col">
                <span class="label">Evidence Maturity</span>
                ${ladderHTML(lvl)}
            </div>
            <div class="ev-col">
                <span class="label">Evidence Layers</span>
                <ul class="layer-list">
                    <li>${chip('SURFACE', evidenceRows(t)[0].ok === true ? 'ok' : 'neutral')}<span class="muted">${esc(d.surface_evidence && d.surface_evidence.observation_window ? `Sentinel-2 / MODIS / DEM · ${fmtWin(d.surface_evidence.observation_window)}` : 'remote-sensing features')}</span></li>
                    <li>${chip('GEOLOGY', geo && pick(geo, 'unit_name') ? 'ok' : 'neutral')}<span class="muted" title="${esc(geo && geo.caveat || '')}">${esc(geo && pick(geo, 'unit_name') ? pick(geo, 'unit_name') : 'not reported')}</span></li>
                    <li>${chip('SUBSURFACE', norm(ss) === 'UNAVAILABLE' || !ss ? 'bad' : 'info')}<span class="muted">${esc(ss ? human(ss).toLowerCase() : 'not reported')}</span></li>
                </ul>
                ${ground.length ? `<span class="label">Observed Ground Evidence</span>
                    <ul class="ground-list">${ground.map(g => `<li>${badgeHTML(g.source_mode || 'REAL_GOVERNMENT')}<b>${esc(human(g.evidence_class || ''))}</b> <span class="muted">${esc(g.summary || '')}</span>${g.official_url ? ` <a href="${esc(g.official_url)}" target="_blank" rel="noopener noreferrer">source</a>` : ''}</li>`).join('')}</ul>` : ''}
            </div>
            <div class="ev-col">
                <span class="label">Documented Occurrences <span class="muted">(MRDS, reported context)</span></span>
                ${occ.length ? `<ul class="occ-list">${occ.slice(0, 6).map(o => `<li><span>${esc(o.site_name || '—')}</span><span class="muted">${esc(o.dev_stat || '')}</span><span class="mono">${esc(fmtNum(o.distance_km, 2))} km</span>${o.used_as_training_label ? '<span class="code" title="Used as a training label">LABEL</span>' : ''}</li>`).join('')}</ul>${occ.length > 6 ? `<div class="muted">+${occ.length - 6} more</div>` : ''}` : '<div class="muted">None within the footprint.</div>'}
            </div>
        </div>
        ${feats && typeof feats === 'object' ? `<div class="feat-block">
            <span class="label">Surface Indicators <span class="muted">· study-area percentile</span></span>
            <div class="feat-grid">${Object.entries(feats).filter(([, v]) => v && num(v.study_area_percentile) !== null).map(([k, v]) => `
                <div class="feat" title="${esc(human(k))}: mean ${esc(fmtNum(v.mean, 4))}">
                    <span class="feat-name">${esc(human(k))}</span>
                    <span class="feat-track"><span class="feat-fill" style="width:${clampPct(v.study_area_percentile)}%"></span></span>
                    <span class="feat-val mono">${esc(fmtNum(v.study_area_percentile, 0))}</span>
                </div>`).join('')}</div>
        </div>` : ''}
        ${why.length || basis.length ? `<details class="disclosure">
            <summary>Target Evidence Reasons</summary>
            <div class="disclosure-body">${reasonListHTML(why)}${basis.length ? `<ul class="plain-list">${basis.map(b => `<li>${esc(b)}</li>`).join('')}</ul>` : ''}</div>
        </details>` : ''}`;
    }

    // ---- Subsurface / ground evidence: observed record + next-evidence sensitivity (backend rules) ----
    async function loadSubsurface(id) {
        const ex = state.exploration;
        let r = ex.subsurface[id];
        if (!r) {
            const gen = state.gen;
            try {
                r = await api.get(`/api/exploration/targets/${enc(id)}/subsurface-scenarios?mine_id=${enc(MINE_ID)}`);
                if (gen !== state.gen) return;
                ex.subsurface[id] = r;
            } catch (e) {
                const wrap = $('#ssWrap');
                if (ex.selectedId === id && wrap) wrap.innerHTML = errorHTML('Subsurface evidence unavailable.', e);
                return;
            }
        }
        if (ex.selectedId !== id) return;
        const wrap = $('#ssWrap');
        if (!wrap) return;
        const obs = r.observed_evidence || {};
        const recs = asArray(obs.records);
        const sens = asArray(r.next_evidence_sensitivity);
        $('#evidenceAside').innerHTML = `${chip(`SUBSURFACE ${upperHuman(obs.subsurface_status || 'UNAVAILABLE')}`, recs.length ? 'info' : 'warn')}${chip(`RESERVE CONFIRMED: ${r.reserve_confirmed ? 'YES' : 'NO'}`, r.reserve_confirmed ? 'ok' : 'neutral')}`;
        const maxAbs = Math.max(1, ...sens.map(x => Math.abs(num(x.priority_change) ?? 0)));
        // Block records already listed as observed ground evidence are not repeated.
        const shown = mergedTarget(id);
        const alreadyListed = shown && asArray(shown.raw.observed_ground_evidence).length > 0;
        wrap.innerHTML = `
            ${recs.length && !alreadyListed ? `<div class="ss-records"><span class="label">Reported Block-Level Records ${badgeHTML('REAL_GOVERNMENT')}</span>
                <ul class="ground-list">${recs.map(x => `<li><b>${esc(human(x.evidence_class || ''))}</b> · ${esc(x.evidence_status || '')} <span class="muted">${esc(x.summary || '')}</span>${x.official_url ? ` <a href="${esc(x.official_url)}" target="_blank" rel="noopener noreferrer">source</a>` : ''}</li>`).join('')}</ul></div>` : ''}
            ${sens.length ? `<details class="disclosure">
                <summary>Next-Evidence Sensitivity ${badgeHTML('SIMULATED')} <span class="muted">hypothetical outcomes — not observed</span></summary>
                <div class="disclosure-body">
                    <div class="sens">${sens.map(x => {
                        const d = num(x.priority_change) ?? 0;
                        return `<div class="sens-row" title="${esc(x.recommended_next_investigation || '')}">
                            <span class="sens-name">${esc(x.outcome)}</span>
                            <span class="sens-meta mono">L${esc(x.hypothetical_evidence_level)} · ${esc(x.hypothetical_uncertainty)}</span>
                            <span class="sens-track"><span class="drv-axis"></span><span class="drv-bar ${d < 0 ? 'neg' : 'pos'}" style="${d < 0 ? 'right' : 'left'}:50%;width:${Math.abs(d) / maxAbs * 50}%"></span></span>
                            <span class="sens-val mono ${d < 0 ? 'down' : 'up'}">${fmtNum(x.exploration_priority_before, 1)} → ${fmtNum(x.exploration_priority_after, 1)}</span>
                        </div>`;
                    }).join('')}</div>
                    <div class="note">Priority before → after each hypothetical outcome. Rules are project-configured (subsurface_fusion), not an industry standard. No borehole, assay or geophysical value is generated.</div>
                </div>
            </details>` : ''}`;
    }

    async function queryCoordinate(lat, lon) {
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) { showToast('Enter a valid latitude and longitude.', 'error'); return; }
        const btn = $('#btnQueryCoord');
        setLoading(btn, true);
        const map = state.maps.exploration;
        try {
            const d = await api.post('/api/exploration/predict', { lat, lon }, { timeout: 45000 });
            state.exploration.query = d;
            const t = normTarget({ ...d, target_id: pick(d, 'target_id') || `${lat.toFixed(3)}, ${lon.toFixed(3)}`, lat: pick(d, 'lat', 'query_lat') ?? lat, lon: pick(d, 'lon', 'query_lon') ?? lon });
            const QUERY_STATUS = {
                EXPLORATION_TARGET: 'within the high-rank (target-threshold) range',
                NOT_PRIORITISED: 'below the target threshold',
                REVIEW_REQUIRED: 'high rank but low applicability — review required',
            };
            t.status = `Coordinate query (not a clustered target) · ${(QUERY_STATUS[norm(pick(d, 'status'))] || human(pick(d, 'status') || 'N/A'))}`;
            if (d.fallback_used) {
                t.warnings = t.warnings.concat([`Live query not used (${d.live_status || 'unavailable'}). Cached cell ${fmtNum(d.fallback_distance_km, 2)} km away (limit ${fmtNum(d.max_supported_fallback_distance_km, 1)} km).`]);
            } else if (d.source_mode) {
                t.warnings = t.warnings.concat([`Fixed ${d.observation_window || '2024'} reference composite, not a current image.`]);
            }
            state.exploration.selectedId = null;
            syncSelection();
            if (map) {
                if (state.maps.queryMarker) map.removeLayer(state.maps.queryMarker);
                state.maps.queryMarker = L.circleMarker([lat, lon], { radius: 7, color: '#fff', weight: 2, fillColor: '#8b5cf6', fillOpacity: 1 })
                    .bindTooltip(popupHTML(t), { direction: 'top', className: 'map-tip' }).addTo(map);
                highlightRegion(t);
            }
            renderRailCard(t, { query: true });
            $('#evidenceTargetId').textContent = 'Query point';
            $('#evidenceAside').innerHTML = provChips(d.provenance);
            $('#evidenceBody').innerHTML = `<div id="evMain">${evidenceMainHTML(t, d)}</div>`;
        } catch (e) {
            showToast(`Coordinate query failed. ${e.userMessage || ''}`, 'error');
        } finally {
            setLoading(btn, false);
        }
    }

    // DEMO_E: backend-computed live-query outage replay (satellite_fallback_demo).
    function renderFallback() {
        const panel = $('#fallbackPanel');
        const fb = state.supply && state.supply.satellite_fallback_demo;
        panel.hidden = !fb;
        if (!fb) return;
        const rows = asArray(fb.results);
        $('#fallbackBody').innerHTML = `<div class="table-wrap"><table class="table">
            <thead><tr><th>Query point</th><th>Outcome</th><th>Detail</th><th>Prospectivity</th></tr></thead>
            <tbody>${rows.map(x => {
                const ok = !x.error;
                return `<tr><td class="mono">${esc(fmtNum(x.query_lat, 4))}°N, ${esc(fmtNum(x.query_lon, 4))}°E</td>
                    <td>${ok ? chip(x.fallback_used ? 'CACHED FALLBACK' : 'LIVE', 'info') : chip(upperHuman(x.error), 'bad')}</td>
                    <td class="muted">${ok ? esc(x.fallback_used ? `cell ${fmtNum(x.fallback_distance_km, 2)} km away · limit ${fmtNum(x.max_supported_fallback_distance_km, 1)} km` : '') : esc(x.message || '')}</td>
                    <td class="mono">${ok && num(x.prospectivity_rank) !== null ? esc(fmtNum(x.prospectivity_rank, 1)) : '—'}</td></tr>`;
            }).join('')}</tbody></table></div>
            ${fb.description ? `<div class="note">${esc(fb.description)}</div>` : ''}`;
    }

    // ═══════════════════════════════════════════════════════
    // 4 · TARGET PRIORITY — WHY THIS TARGET NOW?
    // ═══════════════════════════════════════════════════════
    async function loadTargets() {
        const list = $('#rankList');
        if (!state.exploration.targets.length) list.innerHTML = loadingHTML('Loading targets…');
        try {
            await ensureTargets();
        } catch (e) {
            list.innerHTML = errorHTML('Targets unavailable.', e, 'targets');
            $('#whyPanel').innerHTML = '';
            return;
        }
        state.loaded.targets = true;
        renderTargetsScreen();
    }

    function renderTargetsScreen() {
        const ex = state.exploration, meta = ex.targetsMeta || {};
        const strat = meta.strategic_context;
        const review = norm(meta.decision_state) === 'REVIEW_REQUIRED';
        const mode = ex.targets[0] && ex.targets[0].raw.priority_mode;
        $('#tpContext').innerHTML = (review ? chip('REVIEW REQUIRED', 'bad') : (strat && strat.active ? chip('STRATEGIC CONTINGENCY ACTIVE', 'warn') : chip('NO STRATEGIC GAP', 'ok')))
            + (mode ? ` ${chip(`PRIORITY MODE: ${upperHuman(mode)}`, 'neutral')}` : '');
        $('#tpCount').textContent = `${ex.targets.length} targets`;
        $('#rankList').innerHTML = rankRowsHTML(ex.targets, { full: true });
        const pf = meta.priority_formula || {};
        $('#tpFormula').innerHTML = dlHTML([
            ['Weights', pf.weights ? Object.entries(pf.weights).map(([k, v]) => `<span class="code">${esc(human(k))} ${esc(v)}</span>`).join(' ') : ''],
            ...Object.entries(pf.components || {}).map(([k, v]) => [`Component · ${human(k)}`, esc(v)]),
            ['Gates', pf.gates ? Object.entries(pf.gates).filter(([k]) => !k.startsWith('_')).map(([k, v]) => `<span class="code">${esc(human(k))}: ${esc(v)}</span>`).join(' ') : ''],
            ['Note', esc(pf.note || '')],
            ['Target clustering', esc(meta.method || '')],
            ['Provenance', provenanceHTML(meta.provenance)],
        ]);
        const id = ex.selectedId || meta.selected_target || (ex.targets[0] && ex.targets[0].id);
        if (id && id !== ex.selectedId) selectTarget(id, { fly: false });
        else renderWhy(id);
    }

    async function renderWhy(id) {
        const el = $('#whyPanel');
        if (!id) { el.innerHTML = '<div class="empty-state">Select a target in the ranking.</div>'; return; }
        if (!state.exploration.detail[id]) el.innerHTML = loadingHTML(`Loading ${id}…`);
        let d;
        try {
            d = await getDetail(id);
        } catch (e) {
            if (state.exploration.selectedId === id) el.innerHTML = errorHTML(`Target ${id} unavailable.`, e);
            return;
        }
        if (state.exploration.selectedId !== id) return;
        el.innerHTML = whyHTML(mergedTarget(id), d);
    }

    // Target footprint drawn from the backend geometry (no basemap, no interpolation).
    function footprintSVG(geom, bbox) {
        if (!geom || !Array.isArray(bbox) || bbox.length !== 4) return '';
        const [x0, y0, x1, y1] = bbox.map(Number);
        const W = 132, H = 108, pad = 6;
        const sx = (W - 2 * pad) / Math.max(1e-9, x1 - x0), sy = (H - 2 * pad) / Math.max(1e-9, y1 - y0);
        const s = Math.min(sx, sy);
        const ox = pad + ((W - 2 * pad) - s * (x1 - x0)) / 2, oy = pad + ((H - 2 * pad) - s * (y1 - y0)) / 2;
        const P = ([lon, lat]) => `${(ox + (lon - x0) * s).toFixed(1)},${(oy + (y1 - lat) * s).toFixed(1)}`;
        const polys = geom.type === 'Polygon' ? [geom.coordinates] : (geom.type === 'MultiPolygon' ? geom.coordinates : []);
        const d = polys.map(poly => poly.map(ring => `M${ring.map(P).join('L')}Z`).join('')).join('');
        if (!d) return '';
        return `<svg class="footprint" viewBox="0 0 ${W} ${H}" role="img" aria-label="Target footprint"><rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" rx="6"></rect><path d="${d}"></path></svg>`;
    }

    function priorityCompHTML(d, priority) {
        const w = d.priority_weights_used || {}, c = d.priority_components || {};
        const keys = Object.keys(w).filter(k => num(w[k]) !== null && num(c[k]) !== null);
        const LBL = { prospectivity: 'Prospectivity', evidence_applicability: 'Evidence & applicability', strategic_relevance: 'Strategic relevance' };
        if (!keys.length) return priorityBarHTML(priority);
        const segs = keys.map(k => ({ k, w: num(w[k]), c: num(c[k]), pts: num(w[k]) * num(c[k]) * 100 }));
        const sum = segs.reduce((a, s) => a + s.pts, 0);
        // Only stack the components when they reproduce the backend priority exactly.
        const consistent = num(priority) !== null && Math.abs(sum - num(priority)) <= 0.6;
        return `<div class="pcomp">
            <div class="pcomp-head"><span class="label">Exploration Priority</span><span class="pcomp-val mono">${esc(fmtNum(priority, 1) ?? 'N/A')}<span class="muted"> / 100</span></span></div>
            ${consistent
                ? `<div class="pcomp-bar" role="img" aria-label="Priority composition">${segs.map((s, i) => `<span class="pcomp-seg seg-${i}" style="width:${s.pts.toFixed(2)}%" title="${esc(LBL[s.k] || human(s.k))}: ${esc(fmtNum(s.c, 3))} × weight ${esc(s.w)} = ${esc(fmtNum(s.pts, 1))}"></span>`).join('')}</div>`
                : `<div class="prio-track"><div class="prio-fill" style="width:${clampPct(priority)}%"></div></div>`}
            <div class="pcomp-legend">${segs.map((s, i) => `<span><i class="seg-${i}"></i>${esc(LBL[s.k] || human(s.k))} <b class="mono">${esc(fmtNum(s.c, 2))}</b><span class="muted"> × ${esc(fmtNum(s.w, 2))}</span></span>`).join('')}</div>
        </div>`;
    }

    function compareHTML(t, s0) {
        const lv = x => { const l = maturityLevel(x.maturity); return l !== null ? `L${l}` : 'N/A'; };
        const st = x => x.strategic ? upperHuman(typeof x.strategic === 'object' ? pick(x.strategic, 'level', 'label') : x.strategic) : 'N/A';
        const rows = [
            ['Priority', fmtNum(t.priority, 1), fmtNum(s0.priority, 1), 'info', 'info'],
            ['Prospectivity', fmtNum(t.pros.score, 1), fmtNum(s0.pros.score, 1), 'info', 'info'],
            ['Applicability', t.appl.level, s0.appl.level, applTone(t.appl.level), applTone(s0.appl.level)],
            ['Uncertainty', t.unc.level, s0.unc.level, uncTone(t.unc.level), uncTone(s0.unc.level)],
            ['Evidence', lv(t), lv(s0), evTone(maturityLevel(t.maturity)), evTone(maturityLevel(s0.maturity))],
            ['Strategic relevance', st(t), st(s0), stratTone(st(t)), stratTone(st(s0))],
            ['Distance to supply', t.distanceKm !== null ? `${fmtNum(t.distanceKm, 1)} km` : 'N/A', s0.distanceKm !== null ? `${fmtNum(s0.distanceKm, 1)} km` : 'N/A', 'neutral', 'neutral'],
        ];
        return `<div class="cmp">
            <div class="cmp-row cmp-head"><span></span><span class="mono">${esc(t.id)}</span><span class="mono">${esc(s0.id)} <span class="star">★</span></span></div>
            ${rows.map(([k, a, b, ta, tb]) => `<div class="cmp-row"><span class="label">${esc(k)}</span><span class="lvl" data-tone="${ta}">${esc(a ?? 'N/A')}</span><span class="lvl" data-tone="${tb}">${esc(b ?? 'N/A')}</span></div>`).join('')}
        </div>`;
    }

    function whyHTML(t, d) {
        if (!t) return '<div class="empty-state">Target not found.</div>';
        const ex = state.exploration, meta = ex.targetsMeta || {};
        const strat = meta.strategic_context;
        const sel = meta.selected_target;
        const isSel = d.is_selected_target === true || (sel && sel === t.id);
        const review = norm(meta.decision_state) === 'REVIEW_REQUIRED';
        const r = t.raw || {};
        const lvl = maturityLevel(t.maturity);
        const strat0 = t.strategic ? upperHuman(typeof t.strategic === 'object' ? pick(t.strategic, 'level', 'label') : t.strategic) : null;
        const shares = r.applicability_cell_shares && typeof r.applicability_cell_shares === 'object'
            ? 'cells ' + ['HIGH', 'MODERATE', 'LOW'].filter(k => num(r.applicability_cell_shares[k]) !== null).map(k => `${k[0]} ${fmtPct(r.applicability_cell_shares[k])}`).join(' · ') : '';
        const evLabel = String(r.evidence_level_label || '').replace(/^LEVEL\s*\d+\s*[—-]\s*/i, '');
        const status = isSel ? chip('SELECTED CONTINGENCY TARGET', 'warn')
            : review ? chip('REVIEW REQUIRED', 'bad')
            : sel ? chip(`NOT SELECTED · ${sel} IS SELECTED`, 'neutral')
            : chip('NO CONTINGENCY TARGET', 'neutral');
        const gapTile = review
            ? { label: 'Remaining Supply Gap', value: 'WITHHELD', sub: 'decision under review', tone: 'bad' }
            : strat && strat.active
                ? { label: 'Remaining Supply Gap', value: fmtT(strat.expected_residual_gap_tonnes), sub: `per period · worst ${fmtT(strat.worst_case_residual_gap_tonnes)}`, tone: 'bad', mono: true }
                : { label: 'Remaining Supply Gap', value: 'NONE', sub: 'no active strategic gap', tone: 'ok' };
        const factors = [
            { label: 'Prospectivity', value: t.pros.score !== null ? fmtNum(t.pros.score, 1) : 'N/A', sub: `relative rank${num(r.peak_prospectivity_rank) !== null ? ` · peak ${fmtNum(r.peak_prospectivity_rank, 1)}` : ''}`, tone: 'info', mono: true },
            { label: 'Evidence', value: lvl !== null ? `L${lvl}` : 'N/A', sub: evLabel, tone: evTone(lvl) },
            { label: 'Applicability', value: t.appl.level || 'N/A', sub: shares, tone: applTone(t.appl.level) },
            { label: 'Uncertainty', value: t.unc.level || 'N/A', sub: num(r.rank_sd) !== null ? `ensemble rank SD ${fmtNum(r.rank_sd, 1)}` : '', tone: uncTone(t.unc.level) },
            { label: 'Strategic Relevance', value: strat0 || 'N/A', sub: t.distanceKm !== null ? `${fmtNum(t.distanceKm, 1)} km to supply point` : '', tone: stratTone(strat0) },
            gapTile,
        ];

        // Rationale: the backend's why-now statements are shown for the backend-selected target
        // (or when no strategic gap is active). Other targets are compared with the selected one.
        let rationale;
        if (review) {
            rationale = `<div class="alert" data-tone="bad">${iconHTML('bad')}Contingency not evaluated — automated recommendation withheld.</div>
                ${reasonListHTML(state.supply && state.supply.review_reasons, 'bad')}`;
        } else if (isSel || !(strat && strat.active)) {
            rationale = reasonListHTML(t.whyNow, isSel ? 'warn' : 'neutral') || '<div class="muted">No rationale returned.</div>';
        } else {
            const s0 = ex.targets.find(x => x.id === sel);
            rationale = s0 ? `<div class="sub-label">Compared with contingency target</div>${compareHTML(t, s0)}` : '<div class="muted">Selected target not in list.</div>';
        }
        const gates = asArray(r.gate_blocked_reasons);
        const nextEv = asArray(t.nextEvidence);

        return `
            <div class="why-head">
                <div class="why-id-block">
                    <span class="label">Target</span>
                    <div class="why-id mono">${esc(t.id)}</div>
                    <div class="why-chips">${status}${r.priority_rank != null ? chip(`RANK #${r.priority_rank} OF ${ex.targets.length}`, 'info') : ''}${t.context ? chip(upperHuman(t.context), 'neutral') : ''}${r.contains_training_labels ? chip('TRAINING-LABEL OVERLAP', 'warn', 'Footprint contains records used as training labels; its rank is partly in-sample.') : ''}</div>
                    ${gates.length ? `<div class="alert" data-tone="warn">${iconHTML('warn')}Gated: ${esc(gates.join('; '))}</div>` : ''}
                </div>
                <div class="why-fp">${footprintSVG(t.geometry, r.bbox)}<span class="muted mono">${num(r.area_km2) !== null ? `${esc(fmtNum(r.area_km2, 1))} km²` : ''}${num(r.n_cells) !== null ? ` · ${esc(r.n_cells)} cells` : ''}</span></div>
            </div>

            <div class="tiles tiles-3 factor-tiles">${factors.map(tileHTML).join('')}</div>

            <div class="why-arrow" aria-hidden="true">▼</div>
            ${priorityCompHTML(d, t.priority)}

            <div class="why-cols">
                <div>
                    <h3 class="sub-title">${isSel ? 'Why This Target Now?' : 'Rationale'}</h3>
                    ${rationale}
                </div>
                <div>
                    <h3 class="sub-title">Next Required Evidence</h3>
                    ${nextEv.length ? `<ol class="steps">${nextEv.map(x => `<li>${esc(typeof x === 'object' ? reasonText(x) : x)}</li>`).join('')}</ol>` : '<div class="muted">Not returned.</div>'}
                    <div class="note">Not yet performed.</div>
                </div>
            </div>

            <div class="btn-row">
                <button type="button" class="btn btn-primary" data-open-target="${esc(t.id)}">View on Map</button>
                <button type="button" class="btn btn-ghost" data-evidence-target="${esc(t.id)}">Inspect Evidence</button>
                <button type="button" class="btn btn-ghost" data-truth="exploration">Model Truth</button>
            </div>`;
    }

    // ═══════════════════════════════════════════════════════
    // 2 · PRODUCTION FORECAST
    // ═══════════════════════════════════════════════════════
    function normHistory(h) {
        if (!h) return [];
        // Prefer 7-day period totals: same unit as the 7-day forecast.
        let rows = Array.isArray(h) ? h : pick(h, 'periods', 'history', 'series', 'data', 'records', 'rows');
        if (!rows && Array.isArray(h.dates)) {
            rows = h.dates.map((d, i) => ({ date: d, actual: h.actual?.[i] ?? h.actual_tonnes?.[i], target: h.target?.[i] ?? h.target_tonnes?.[i] }));
        }
        return asArray(rows).map(r => ({
            date: pick(r, 'date', 'period', 'day', 'timestamp', 'period_start'),
            actual: num(pick(r, 'actual_tonnes', 'actual', 'production_tonnes', 'production')),
            target: num(pick(r, 'target_tonnes', 'target')),
        })).filter(r => r.date !== undefined && r.date !== null);
    }

    function normForecast(f) {
        if (!f) return null;
        const src = pick(f, 'forecast') && typeof f.forecast === 'object' ? { ...f, ...f.forecast } : f;
        return {
            p10: num(pick(src, 'p10_tonnes', 'p10')),
            p50: num(pick(src, 'p50_tonnes', 'p50', 'forecast_tonnes')),
            p90: num(pick(src, 'p90_tonnes', 'p90')),
            target: num(pick(src, 'target_tonnes', 'target')),
            gap: num(pick(src, 'gap_p50_tonnes', 'gap_tonnes', 'supply_gap_tonnes')),
            risk: pick(src, 'risk_state', 'risk'),
            horizon: pick(src, 'forecast_horizon', 'horizon'),
            date: pick(src, 'period_start', 'forecast_date', 'period', 'period_end', 'forecast_period'),
            series: asArray(pick(src, 'forecast_series', 'series')),
            // model_contributions = {status, method, base_value_tonnes, calibration_adjustment_tonnes, items[]}
            contributions: (src.model_contributions && typeof src.model_contributions === 'object' && !Array.isArray(src.model_contributions))
                ? src.model_contributions.items
                : pick(src, 'model_contributions', 'contributions', 'drivers', 'primary_drivers'),
            contributionMeta: (src.model_contributions && !Array.isArray(src.model_contributions)) ? src.model_contributions : null,
            provenance: pick(src, 'provenance'),
            quantilesValidated: pick(src, 'quantiles_validated', 'intervals_validated'),
            raw: src,
        };
    }

    async function loadProduction() {
        const btn = $('#btnRefreshProduction');
        setLoading(btn, true);
        $('#histChart').innerHTML = loadingHTML('Loading production history…');
        $('#prContributions').innerHTML = loadingHTML('Loading forecast…');
        $('#prTiles').innerHTML = loadingHTML('Loading forecast…');
        const gen = state.gen;
        const [hist, fc] = await Promise.allSettled([
            api.get(`/api/production/history?mine_id=${enc(MINE_ID)}&days=90`),
            api.post('/api/production/forecast', { mine_id: MINE_ID }),
        ]);
        if (gen !== state.gen) return;
        loadRealQuarterly();
        state.production.history = hist.status === 'fulfilled' ? hist.value : null;
        state.production.forecast = fc.status === 'fulfilled' ? fc.value : null;
        state.loaded['production-risk'] = hist.status === 'fulfilled' || fc.status === 'fulfilled';

        const f = normForecast(state.production.forecast);
        const provs = [];
        if (state.production.history && state.production.history.provenance) provs.push(state.production.history.provenance);
        if (f && f.provenance) provs.push(f.provenance);
        $('#prProvenance').innerHTML = provs.length ? provChips(Object.assign({}, ...provs)) : badgeHTML('UNAVAILABLE');

        if (fc.status === 'fulfilled') renderForecast(f);
        else {
            $('#prTiles').innerHTML = errorHTML('Forecast unavailable.', fc.reason, 'production-risk');
            $('#prRangeChart').innerHTML = '';
            $('#prGapAside').innerHTML = '';
            $('#prContributions').innerHTML = '';
            $('#prForecastBasis').innerHTML = '';
        }
        if (hist.status === 'fulfilled') renderHistoryChart(normHistory(state.production.history), f);
        else $('#histChart').innerHTML = errorHTML('Production history unavailable.', hist.reason, 'production-risk');
        setLoading(btn, false);
    }

    function renderForecast(f) {
        const raw = f.raw;
        const days = num(raw.horizon_days);
        $('#prHorizon').textContent = `${days ? `${days}-DAY` : upperHuman(f.horizon || '')} FORECAST · ${raw.period_start || '?'} → ${raw.period_end || '?'}`;
        const iv = intervalInfo(raw);
        const appl = pick(raw, 'applicability') || {};
        const ood = appl.level === 'LOW';
        const viol = asArray(appl.range_violations).map(v => human(v.feature || v)).join(', ');
        const gapPos = f.gap !== null && f.gap > 0;
        $('#prTiles').innerHTML = [
            { label: 'Supply Status', value: f.gap === null ? 'N/A' : (gapPos ? 'SHORTFALL' : 'ON TARGET'), sub: f.risk ? `risk ${norm(f.risk)} · ${riskPolicyText(raw)}` : '', tone: f.gap === null ? 'neutral' : (gapPos ? riskTone(f.risk) : 'ok') },
            { label: 'Forecast P50', value: fmtT(f.p50), sub: f.p10 !== null && f.p90 !== null ? `P10–P90 ${fmtT(f.p10)} – ${fmtT(f.p90)}` : '', tone: 'info', mono: true },
            { label: 'Required', value: fmtT(f.target), sub: raw.target_source ? human(raw.target_source).toLowerCase() : '', tone: 'neutral', mono: true },
            { label: 'Supply Gap', value: fmtT(f.gap), sub: num(raw.gap_pct) !== null ? `${fmtNum(raw.gap_pct, 1)}% of required` : '', tone: f.gap === null ? 'neutral' : (gapPos ? 'bad' : 'ok'), mono: true },
            { label: 'Applicability', value: appl.level || 'N/A', sub: viol ? `outside training range: ${viol}` : (appl.level ? 'inputs within training range' : ''), tone: applTone(appl.level) },
            { label: 'P10–P90 Interval', value: iv.validated ? 'VALIDATED' : 'NOT VALIDATED', sub: ood ? 'not reliable: inputs outside training range' : (iv.obs !== null ? `backtest coverage ${fmtPct(iv.obs)} · nominal ${fmtPct(iv.nom)}` : ''), tone: ood ? 'warn' : (iv.validated ? 'ok' : 'warn') },
        ].map(tileHTML).join('');

        $('#prGapAside').innerHTML = gapPos ? chip(`SUPPLY GAP ${fmtT(f.gap)}`, 'bad') : (f.gap !== null ? chip('NO GAP', 'ok') : '');
        renderRangeChart($('#prRangeChart'), f);

        const cm = f.contributionMeta;
        if (cm && cm.status === 'UNAVAILABLE') {
            $('#prContributions').innerHTML = '<div class="empty-state">Model contributions unavailable for this forecast.</div>';
        } else {
            renderDriverBars($('#prContributions'), f.contributions);
            if (cm && cm.status === 'AVAILABLE') {
                $('#prContributions').insertAdjacentHTML('beforeend', `<div class="note">Base value ${esc(fmtT(cm.base_value_tonnes))} · calibration ${esc(fmtSigned(cm.calibration_adjustment_tonnes))}</div>`);
            }
        }

        $('#prForecastBasis').innerHTML = dlHTML([
            ['Forecast basis', esc(pick(raw, 'forecast_basis') || '')],
            ['Method', esc(raw.forecast_method || '')],
            ['Persistence assumption', esc(typeof raw.persistence_assumption === 'string' ? raw.persistence_assumption : (raw.persistence_assumption ? JSON.stringify(raw.persistence_assumption) : ''))],
            ['Scenario override', raw.scenario_override_applied ? badgeHTML('SIMULATED', 'INPUTS') : 'none'],
            ['Interval', esc(iv.text + (iv.window ? ` · evaluation window ${iv.window}` : ''))],
            ['Interval note', esc(raw.interval_note || '')],
            ['Applicability', appl.level ? `${esc(appl.level)}${viol ? ` — outside training range: ${esc(viol)}` : ''}${appl.method ? ` <span class="muted">(${esc(appl.method)})</span>` : ''}` : ''],
            ['Risk detail', raw.risk_detail ? esc(`gap ${fmtNum(raw.risk_detail.gap_pct, 1)}% · P10 gap ${fmtNum(raw.risk_detail.p10_gap_pct, 1)}% · ${raw.risk_detail.risk_policy_note || ''}`) : ''],
            ['Model contributions', esc(cm && cm.method ? cm.method : '')],
            ['Result status', esc(raw.status ? upperHuman(raw.status) : '')],
            ['Provenance', provenanceHTML(raw.provenance)],
        ]);
    }

    // Horizontal range: P10–P90 band, P50 marker, required marker, gap between them.
    function renderRangeChart(el, f) {
        const vals = [f.p10, f.p50, f.p90, f.target].filter(v => v !== null);
        if (vals.length < 2 || f.p50 === null) { el.innerHTML = ''; return; }
        const lo = Math.min(...vals), hi = Math.max(...vals);
        const pad = (hi - lo) * 0.12 || hi * 0.05 || 1;
        const min = Math.max(0, lo - pad), max = hi + pad;
        const x = v => ((v - min) / (max - min)) * 100;
        const marks = [];
        if (f.p10 !== null && f.p90 !== null) marks.push(`<div class="rc-band" style="left:${x(f.p10)}%;width:${x(f.p90) - x(f.p10)}%" title="P10–P90: ${esc(fmtT(f.p10))} – ${esc(fmtT(f.p90))}"><span class="rc-end rc-end-l mono">P10 ${esc(fmtT(f.p10))}</span><span class="rc-end rc-end-r mono">P90 ${esc(fmtT(f.p90))}</span></div>`);
        if (f.target !== null && f.target > f.p50) marks.push(`<div class="rc-gap" style="left:${x(f.p50)}%;width:${x(f.target) - x(f.p50)}%"><span class="mono">GAP ${esc(fmtT(f.target - f.p50))}</span></div>`);
        marks.push(`<div class="rc-mark rc-p50" style="left:${x(f.p50)}%"><span class="mono">P50 ${esc(fmtT(f.p50))}</span></div>`);
        if (f.target !== null) marks.push(`<div class="rc-mark rc-target" style="left:${x(f.target)}%"><span class="mono">REQUIRED ${esc(fmtT(f.target))}</span></div>`);
        const ticks = [0, 0.25, 0.5, 0.75, 1].map(p => min + (max - min) * p);
        el.innerHTML = `<div class="rc-track" role="img" aria-label="Forecast P10 ${esc(fmtT(f.p10))}, P50 ${esc(fmtT(f.p50))}, P90 ${esc(fmtT(f.p90))}; required ${esc(fmtT(f.target))}">${marks.join('')}</div>
            <div class="rc-axis">${ticks.map(v => `<span class="mono">${esc(fmtT(v))}</span>`).join('')}</div>
            <div class="chart-legend"><span><i class="lg-sw lg-band"></i>P10–P90 ${intervalInfo(f.raw).validated ? '(validated)' : '(not validated)'}</span><span><i class="lg-p50line"></i>P50 median</span><span><i class="lg-req"></i>Required</span>${f.target !== null && f.target > f.p50 ? '<span><i class="lg-sw lg-gap"></i>Supply gap</span>' : ''}</div>`;
    }

    const C_ACTUAL = '#7aa7d6', C_FORECAST = '#2bb3c0', C_TARGET = '#cbd5e1', C_GAP = '#e5534b';

    function renderHistoryChart(rows, f) {
        const el = $('#histChart');
        const legend = $('#histLegend');
        if (!rows.length) {
            el.innerHTML = '<div class="empty-state">No production history returned.</div>';
            legend.innerHTML = '';
            return;
        }
        // Forecast overlays the history only if it is on the same scale (a
        // per-period series). A period-total forecast on another scale is not plotted.
        const fSeries = f ? f.series.map(p => ({
            date: pick(p, 'date', 'period'),
            p10: num(pick(p, 'p10_tonnes', 'p10')), p50: num(pick(p, 'p50_tonnes', 'p50')), p90: num(pick(p, 'p90_tonnes', 'p90')), target: null,
        })).filter(p => p.date && p.p50 !== null) : [];
        const lastActual = [...rows].reverse().find(r => r.actual !== null);
        let fPoints = fSeries;
        let fNote = '';
        if (!fPoints.length && f && f.p50 !== null && lastActual) {
            const ratio = f.p50 / (lastActual.actual || 1);
            if (ratio > 0.2 && ratio < 5) {
                fPoints = [{ date: f.date || 'Forecast', p10: f.p10, p50: f.p50, p90: f.p90, target: f.target }];
            } else {
                fNote = 'Forecast is on a different scale from the history and is shown in the range chart above.';
            }
        }

        const all = rows.map(r => ({ ...r, kind: 'hist' })).concat(fPoints.map(p => ({ ...p, kind: 'fc' })));
        const vals = all.flatMap(r => [r.actual, r.target, r.p10, r.p50, r.p90]).filter(v => v !== null && v !== undefined);
        if (!vals.length) { el.innerHTML = '<div class="empty-state">History contains no numeric values.</div>'; return; }
        let yMin = Math.min(...vals), yMax = Math.max(...vals);
        const pad = (yMax - yMin) * 0.1 || yMax * 0.1 || 1;
        yMin = Math.max(0, yMin - pad); yMax += pad;

        const W = 860, H = 300, m = { l: 60, r: 96, t: 16, b: 34 };
        const n = all.length;
        const xs = i => m.l + (n === 1 ? (W - m.l - m.r) / 2 : i * (W - m.l - m.r) / (n - 1));
        const ys = v => m.t + (1 - (v - yMin) / (yMax - yMin)) * (H - m.t - m.b);
        const line = (key, filterKind) => {
            let d = '', pen = false;
            all.forEach((r, i) => {
                const v = r[key];
                if ((filterKind && r.kind !== filterKind) || v === null || v === undefined) { pen = false; return; }
                d += `${pen ? 'L' : 'M'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`;
                pen = true;
            });
            return d;
        };
        const ticks = 4;
        const grid = Array.from({ length: ticks + 1 }, (_, i) => yMin + (yMax - yMin) * i / ticks).map(v =>
            `<line class="ch-grid" x1="${m.l}" x2="${W - m.r}" y1="${ys(v)}" y2="${ys(v)}"></line><text class="ch-tick" x="${m.l - 8}" y="${ys(v) + 4}" text-anchor="end">${esc(fmtT(v))}</text>`).join('');
        const labelEvery = Math.max(1, Math.ceil(n / 7));
        const xLabels = all.map((r, i) => ((i % labelEvery === 0 && (n - 1 - i >= labelEvery / 2 || i === 0)) || i === n - 1)
            ? `<text class="ch-tick" x="${xs(i)}" y="${H - 12}" text-anchor="middle">${esc(String(r.date).slice(5, 10))}</text>` : '').join('');

        const fcStart = all.findIndex(r => r.kind === 'fc');
        let band = '', fcMarks = '';
        if (fcStart >= 0) {
            const fc = all.map((r, i) => ({ r, i })).filter(o => o.r.kind === 'fc' && o.r.p10 !== null && o.r.p90 !== null);
            if (fc.length === 1) {
                const { r, i } = fc[0];
                band = `<rect class="ch-band" x="${xs(i) - 12}" y="${ys(r.p90)}" width="24" height="${Math.max(1, ys(r.p10) - ys(r.p90))}" rx="3"></rect>`;
            } else if (fc.length > 1) {
                const top = fc.map(o => `${xs(o.i)},${ys(o.r.p90)}`).join(' ');
                const bot = fc.slice().reverse().map(o => `${xs(o.i)},${ys(o.r.p10)}`).join(' ');
                band = `<polygon class="ch-band" points="${top} ${bot}"></polygon>`;
            }
            // Required level and the P50 gap at the forecast period (backend values).
            const last = all[all.length - 1];
            const i = all.length - 1;
            if (last.kind === 'fc' && last.target !== null && last.p50 !== null) {
                fcMarks += `<line class="ch-req" x1="${xs(i) - 20}" x2="${xs(i) + 20}" y1="${ys(last.target)}" y2="${ys(last.target)}"></line>`;
                fcMarks += `<text class="ch-lbl" x="${xs(i) + 24}" y="${ys(last.target) + 4}">Required</text>`;
                if (last.target > last.p50) {
                    const gx = xs(i) + 18;
                    fcMarks += `<line class="ch-gap" x1="${gx}" x2="${gx}" y1="${ys(last.target)}" y2="${ys(last.p50)}"></line>`;
                    fcMarks += `<text class="ch-lbl ch-lbl-gap" x="${gx + 6}" y="${(ys(last.target) + ys(last.p50)) / 2 + 4}">GAP ${esc(fmtT(last.target - last.p50))}</text>`;
                }
                fcMarks += `<text class="ch-lbl ch-lbl-fc" x="${xs(i) + 24}" y="${ys(last.p50) + 16}">P50</text>`;
            }
        }
        const fcDivider = fcStart > 0 ? `<line class="ch-divider" x1="${(xs(fcStart - 1) + xs(fcStart)) / 2}" x2="${(xs(fcStart - 1) + xs(fcStart)) / 2}" y1="${m.t}" y2="${H - m.b}"></line><text class="ch-tick" x="${(xs(fcStart - 1) + xs(fcStart)) / 2 + 4}" y="${m.t + 10}">FORECAST</text>` : '';
        const fcDots = all.map((r, i) => r.kind === 'fc' && r.p50 !== null ? `<circle class="ch-fc-dot" cx="${xs(i)}" cy="${ys(r.p50)}" r="5.5"></circle>` : '').join('');
        const actualDots = all.map((r, i) => r.kind === 'hist' && r.actual !== null ? `<circle class="ch-act-dot" cx="${xs(i)}" cy="${ys(r.actual)}" r="3"></circle>` : '').join('');
        const actualPath = line('actual', 'hist');
        const targetPath = line('target', 'hist');
        const fcPath = fPoints.length > 1 ? line('p50', 'fc') : '';

        el.innerHTML = `<svg class="hist-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Production per 7-day period: actual versus required${fPoints.length ? ', with forecast and supply gap' : ''}">
                ${grid}${band}${fcDivider}
                ${targetPath ? `<path class="ch-target" d="${targetPath}" stroke="${C_TARGET}"></path>` : ''}
                ${actualPath ? `<path class="ch-actual" d="${actualPath}" stroke="${C_ACTUAL}"></path>` : ''}
                ${actualDots}
                ${fcPath ? `<path class="ch-fc" d="${fcPath}" stroke="${C_FORECAST}"></path>` : ''}
                ${fcMarks}${fcDots}
                ${xLabels}
                <line class="ch-cross" id="chCross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" visibility="hidden"></line>
            </svg>
            <div class="ch-tooltip" id="chTooltip" hidden></div>
            ${fNote ? `<div class="note">${esc(fNote)}</div>` : ''}
            <details class="disclosure"><summary>Data Table</summary><div class="table-wrap"><table class="table">
                <thead><tr><th>Period start</th><th>Actual</th><th>Required</th><th>P10</th><th>P50</th><th>P90</th></tr></thead>
                <tbody>${all.map(r => `<tr><td class="mono">${esc(r.date)}${r.kind === 'fc' ? ' (forecast)' : ''}</td><td class="mono">${esc(r.actual != null ? fmtT(r.actual) : '')}</td><td class="mono">${esc(r.target != null ? fmtT(r.target) : '')}</td><td class="mono">${esc(r.p10 != null ? fmtT(r.p10) : '')}</td><td class="mono">${esc(r.p50 != null ? fmtT(r.p50) : '')}</td><td class="mono">${esc(r.p90 != null ? fmtT(r.p90) : '')}</td></tr>`).join('')}</tbody>
            </table></div></details>`;

        legend.innerHTML = `<span><i class="lg-line" style="background:${C_ACTUAL}"></i>Actual</span>
            <span><i class="lg-req"></i>Required</span>
            ${fPoints.length ? `<span><i class="lg-dot" style="background:${C_FORECAST}"></i>Forecast P50</span>` : ''}
            ${band ? `<span><i class="lg-sw lg-band"></i>P10–P90${f && f.raw && f.raw.quantiles_validated === true ? '' : ' (not validated)'}</span>` : ''}
            ${fcMarks.includes('ch-gap') ? `<span><i class="lg-gapline" style="background:${C_GAP}"></i>Supply gap</span>` : ''}`;

        // Crosshair + tooltip
        const svg = $('.hist-svg', el), tip = $('#chTooltip', el), cross = $('#chCross', el);
        svg.addEventListener('mousemove', ev => {
            const rect = svg.getBoundingClientRect();
            const px = (ev.clientX - rect.left) / rect.width * W;
            let best = 0, bd = Infinity;
            all.forEach((_, i) => { const d = Math.abs(xs(i) - px); if (d < bd) { bd = d; best = i; } });
            const r = all[best];
            cross.setAttribute('x1', xs(best)); cross.setAttribute('x2', xs(best)); cross.setAttribute('visibility', 'visible');
            tip.hidden = false;
            tip.innerHTML = `<b>${esc(r.date)}</b>${r.kind === 'fc' ? ' <span class="muted">(forecast)</span>' : ''}<br>` +
                (r.kind === 'hist'
                    ? `Actual: ${esc(fmtT(r.actual))}<br>Required: ${esc(fmtT(r.target))}`
                    : `P10: ${esc(fmtT(r.p10))}<br>P50: ${esc(fmtT(r.p50))}<br>P90: ${esc(fmtT(r.p90))}${r.target !== null ? `<br>Required: ${esc(fmtT(r.target))}` : ''}`);
            const left = xs(best) / W * rect.width;
            tip.style.left = `${Math.min(left + 12, rect.width - 170)}px`;
        });
        svg.addEventListener('mouseleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
    }

    // ---- Real MOIL company-level quarterly production (REAL_MOIL_PUBLIC) ----
    async function loadRealQuarterly() {
        const body = $('#rqBody');
        if (!body) return;
        let r;
        try { r = await api.get('/api/production/real-quarterly?last_n=24'); }
        catch (e) { body.innerHTML = errorHTML('Real MOIL quarterly production unavailable.', e); $('#rqProvenance').innerHTML = badgeHTML('UNAVAILABLE'); return; }
        $('#rqProvenance').innerHTML = provChips(r.provenance);
        const q = asArray(r.quarters);
        const max = Math.max(...q.map(x => x.production_t || 0), 1);
        const v = r.validation || {}; const fc = r.forecast || {}; const bf = r.baseline_forecasts_t || {};
        const beats = v.model_beats_best_baseline === true;
        body.innerHTML = `
            <div class="rq-layout">
                <div>
                    <div class="rq-bars" role="img" aria-label="MOIL quarterly production, last ${q.length} quarters">
                        ${q.map(x => `<div class="rq-bar ${/derived/.test(x.basis) ? 'derived' : ''}" style="height:${(100 * (x.production_t || 0) / max).toFixed(1)}%" title="FY${esc(x.fy)} Q${x.fy_quarter}: ${fmtNum(x.production_t, 0)} t (${esc(x.basis)})"></div>`).join('')}
                    </div>
                    <div class="rq-axis"><span class="mono">${esc(q.length ? q[0].quarter_start : '')}</span><span><i class="lg-sw lg-rq"></i>stated <i class="lg-sw lg-rq-d"></i>derived</span><span class="mono">${esc(q.length ? q[q.length - 1].quarter_start : '')}</span></div>
                </div>
                <div class="tiles tiles-1">
                    ${tileHTML({ label: `Next Quarter · ${fc.quarter_start || '?'}`, value: fmtT(fc.point_t), sub: fc.empirical_error_band_t ? `indicative band ${fmtT(fc.empirical_error_band_t[0])} – ${fmtT(fc.empirical_error_band_t[1])}` : '', tone: 'info', mono: true })}
                    ${tileHTML({ label: 'Best Baseline', value: fmtT(bf.seasonal_naive_x_yoy), sub: `seasonal naive × YoY · last quarter ${fmtT(bf.last_quarter)}`, tone: 'neutral', mono: true })}
                    ${tileHTML({ label: 'Held-out MAPE · Model / Baseline', value: `${fmtNum(v.selected_test && v.selected_test.mape_pct, 1)}% / ${fmtNum(v.best_baseline_test && v.best_baseline_test.mape_pct, 1)}%`, sub: v.test_window ? `${v.test_window.join(' → ')} · real quarters` : '', tone: beats ? 'ok' : 'warn', mono: true })}
                </div>
            </div>
            <div class="chip-row">${chip(beats ? 'MODEL BEATS BASELINE' : 'MODEL DOES NOT BEAT BASELINE', beats ? 'ok' : 'warn')}${chip(`SYNTHETIC AUGMENTATION ${v.synthetic_augmentation_helps ? 'USED' : 'NOT USED'}`, 'neutral')}</div>
            <details class="disclosure"><summary>Validation Notes</summary><div class="disclosure-body">${dlHTML([
                ['Verdict', esc(v.verdict || '')],
                ['Band', esc(fc.band_note || '')],
                ['Unit', esc(r.unit || '')],
                ['Gaps', esc(r.gaps_note || '')],
                ['Claims not made', esc(r.claims_not_made || '')],
            ])}</div></details>`;
    }

    // ═══════════════════════════════════════════════════════
    // 3 · RECOVERY SCENARIOS
    // ═══════════════════════════════════════════════════════
    function currentInputs(includeConditions) {
        const scenario = ($('input[name="scenario"]:checked') || {}).value || 'normal';
        const actions = $$('input[name="action"]:checked').map(i => i.value);
        const body = { mine_id: MINE_ID, scenario, actions };
        if (includeConditions) {
            body.conditions = {
                rainfall_mm: Number($('#flipRainfall').value),
                equipment_availability: Number($('#flipEquip').value),
                blast_delay_hours: Number($('#flipBlast').value),
            };
        }
        return body;
    }

    function actionName(a) {
        if (!a) return null;
        if (typeof a === 'string') return a;
        const n = pick(a, 'label', 'name', 'portfolio', 'portfolio_name', 'id');
        if (n) return human(n);
        const acts = pick(a, 'actions');
        if (Array.isArray(acts) && acts.length) return acts.map(x => human(typeof x === 'string' ? x : pick(x, 'name', 'id'))).join(' + ');
        return null;
    }

    function normPortfolios(r) {
        const list = asArray(pick(r, 'portfolios', 'action_portfolios', 'results', 'candidates'));
        return list.map(p => {
            if (typeof p === 'string') return { name: human(p), id: p, perScenario: [], lowScenarios: [] };
            const feasible = pick(p, 'feasibility', 'status', 'feasibility_status');
            return {
                id: pick(p, 'id', 'portfolio_id', 'name', 'portfolio'),
                name: actionName(p) || '—',
                recovery: num(pick(p, 'expected_recovery_tonnes', 'recovery_tonnes')),
                residual: num(pick(p, 'expected_residual_gap_tonnes', 'residual_gap_tonnes', 'expected_gap_tonnes')),
                worst: num(pick(p, 'worst_case_residual_gap_tonnes', 'worst_tested_residual_gap_tonnes', 'worst_gap_tonnes')),
                worstScenario: p.worst_case_scenario,
                feasibility: pick(p, 'modelled_feasibility') ?? (feasible !== undefined ? feasible : (typeof p.feasible === 'boolean' ? (p.feasible ? 'FEASIBLE' : 'NOT FEASIBLE') : null)),
                selectedFlag: p.selected === true || p.is_selected === true,
                eligible: p.eligible,
                applicability: pick(p, 'selection_applicability', 'applicability'),
                lowScenarios: asArray(p.low_applicability_scenarios),
                blockedReason: p.selection_blocked_reason,
                burden: num(p.intervention_burden),
                perScenario: asArray(p.per_scenario),
                raw: p,
            };
        });
    }

    function selectedKey(r) {
        const sel = pick(r, 'selected_portfolio', 'selected', 'best_operational_action');
        if (!sel) return null;
        if (typeof sel === 'string') return sel;
        return pick(sel, 'id', 'portfolio_id', 'name', 'portfolio', 'label');
    }

    function portfolioSelected(r) {
        const selKey = r.selection_status === 'REVIEW_REQUIRED' ? null : selectedKey(r);
        return p => p.selectedFlag || (selKey !== null && (String(p.id) === String(selKey) || p.name === human(selKey)));
    }

    // Full technical table (inside "Portfolio Details").
    function renderPortfolios(r) {
        const el = $('#portfolioTable');
        const ps = normPortfolios(r);
        const isSel = portfolioSelected(r);
        if (!ps.length) { el.innerHTML = '<div class="empty-state">No action portfolios returned.</div>'; return null; }
        const applTn = a => applTone(a);
        el.innerHTML = `<div class="table-wrap"><table class="table">
            <thead><tr><th>Portfolio</th><th>Burden</th><th>Exp. recovery</th><th>Exp. residual</th><th>Worst residual</th><th>Applicability</th><th>Modelled feasibility</th><th>Automated selection</th></tr></thead>
            <tbody>${ps.map(p => `<tr class="${isSel(p) ? 'row-selected' : ''}">
                <td><span class="mono">${esc(p.id)}</span> <span class="muted">${esc(p.name)}</span></td>
                <td class="mono">${p.burden !== null ? esc(fmtNum(p.burden, 1)) : 'N/A'}</td>
                <td class="mono">${esc(fmtSigned(p.recovery))}</td>
                <td class="mono">${esc(fmtT(p.residual))}</td>
                <td class="mono">${esc(fmtT(p.worst))}</td>
                <td>${p.applicability ? chip(upperHuman(p.applicability), applTn(p.applicability), p.lowScenarios.length ? 'Outside model experience in: ' + p.lowScenarios.map(human).join(', ') : 'All tested scenarios within model experience') : 'N/A'}</td>
                <td>${p.feasibility !== null && p.feasibility !== undefined ? chip(upperHuman(p.feasibility), /NOT|INFEAS/.test(norm(p.feasibility)) ? 'bad' : (/REVIEW|UNKNOWN/.test(norm(p.feasibility)) ? 'warn' : 'ok')) : 'N/A'}</td>
                <td>${isSel(p) ? chip('SELECTED', 'ok') : p.eligible === false ? `${chip('BLOCKED', 'bad')}<div class="note">${esc(p.blockedReason || '')}</div>` : (p.eligible === true ? chip('ELIGIBLE', 'info') : 'N/A')}</td>
            </tr>`).join('')}</tbody>
        </table></div>
        ${dlHTML([
            ['Selection', esc(pick(r, 'selection_explanation') || '')],
            ['Selection rule', esc(pick(r, 'selection_rule') || '')],
            ['Applicability policy', r.applicability_policy ? Object.entries(r.applicability_policy).map(([k, v]) => `<b>${esc(human(k))}</b>: ${esc(v)}`).join('<br>') : ''],
            ['Feasibility basis', esc((ps[0] && ps[0].raw && ps[0].raw.feasibility_basis) || '')],
            ['Review note', esc(r.review_note || '')],
            ['Inputs (server-side)', r.inputs ? `<span class="mono">${Object.entries(r.inputs).map(([k, v]) => `${esc(human(k))} ${esc(fmtNum(v, 3))}`).join(' · ')}</span>` : ''],
            ['Provenance', provenanceHTML(r.provenance)],
        ])}`;
        return ps.find(isSel) || null;
    }

    // Primary recovery visual: expected → worst tested residual gap per portfolio.
    function renderPortfolioChart(r, con) {
        const el = $('#portfolioChart');
        const ps = normPortfolios(r);
        if (!ps.length) { el.innerHTML = '<div class="empty-state">No action portfolios returned.</div>'; $('#portfolioLegend').innerHTML = ''; return; }
        const isSel = portfolioSelected(r);
        const review = r.selection_status === 'REVIEW_REQUIRED';
        const mat = num(r.materiality_threshold_tonnes);
        const tol = num(con && con.strategic_requirement && con.strategic_requirement.worst_case_tolerance_tonnes);
        const max = Math.max(1, ...ps.map(p => Math.max(p.worst ?? 0, p.residual ?? 0)), mat ?? 0, tol ?? 0) * 1.05;
        const x = v => `${(Math.max(0, v) / max * 100).toFixed(2)}%`;
        const refs = `${mat !== null ? `<span class="pc-ref pc-mat" style="left:${x(mat)}"></span>` : ''}${tol !== null ? `<span class="pc-ref pc-tol" style="left:${x(tol)}"></span>` : ''}`;
        el.innerHTML = `
            ${review ? `<div class="alert" data-tone="bad">${iconHTML('bad')}<b>REVIEW REQUIRED</b>&nbsp;— no portfolio passed the modelled-feasibility and applicability gates; automated selection withheld.</div>` : ''}
            <div class="pc" role="table" aria-label="Residual supply gap by action portfolio">
                <div class="pc-row pc-head" role="row"><span>Portfolio</span><span>Status</span><span>Residual gap · expected <span class="lg-sw lg-exp"></span> → worst tested <span class="lg-sw lg-worst"></span></span><span>Expected / worst</span><span>Recovery</span></div>
                ${ps.map(p => {
                    const sel = isSel(p);
                    const blocked = p.eligible === false;
                    const status = sel ? chip('SELECTED', 'ok') : blocked ? chip('BLOCKED', 'bad', p.blockedReason || '') : (p.eligible === true ? chip('ELIGIBLE', 'info') : chip('N/A', 'neutral'));
                    return `<div class="pc-row${sel ? ' selected' : ''}${blocked ? ' blocked' : ''}" role="row" title="${esc(blocked ? (p.blockedReason || '') : (p.worstScenario ? `Worst tested: ${human(p.worstScenario)}` : ''))}">
                        <span class="pc-name"><span class="mono pc-id">${esc(p.id)}</span><span class="pc-sub">${esc(p.name)}</span></span>
                        <span>${status}</span>
                        <span class="pc-track">
                            ${p.residual !== null ? `<span class="pc-exp" style="width:${x(p.residual)}"></span>` : ''}
                            ${p.residual !== null && p.worst !== null && p.worst > p.residual ? `<span class="pc-worst" style="left:${x(p.residual)};width:${x(p.worst - p.residual)}"></span>` : ''}
                            ${p.worst !== null ? `<span class="pc-tick" style="left:${x(p.worst)}"></span>` : ''}
                            ${refs}
                        </span>
                        <span class="mono pc-vals">${esc(fmtT(p.residual))} <span class="muted">/ ${esc(fmtT(p.worst))}</span></span>
                        <span class="mono pc-rec">${esc(fmtSigned(p.recovery))}</span>
                    </div>`;
                }).join('')}
            </div>`;
        $('#portfolioLegend').innerHTML = `${mat !== null ? `<span><i class="lg-ref lg-mat"></i>Materiality ${esc(fmtT(mat))}</span>` : ''}${tol !== null ? `<span><i class="lg-ref lg-tol"></i>Worst-case tolerance ${esc(fmtT(tol))}</span>` : ''}<span>${chip('BLOCKED', 'bad')} low applicability</span>`;
    }

    const SCEN_SHORT = { NORMAL: 'Normal', HEAVY_RAIN: 'Heavy rain', EQUIPMENT_DEGRADATION: 'Equip. degr.', BLAST_DELAY: 'Blast delay', DRILL_DELAY: 'Drill delay', HAULAGE_DISRUPTION: 'Haulage', COMBINED_DISRUPTION: 'Combined' };

    // Residual gap for every portfolio × disruption scenario (backend per_scenario values).
    function renderScenarioMatrix(r) {
        const el = $('#scenarioMatrix');
        const ps = normPortfolios(r).filter(p => p.perScenario.length);
        const scen = asArray(r.disruption_scenarios).length ? asArray(r.disruption_scenarios) : [...new Set(ps.flatMap(p => p.perScenario.map(s => s.scenario)))];
        if (!ps.length || !scen.length) { el.innerHTML = '<div class="empty-state">No per-scenario results returned.</div>'; $('#matrixLegend').innerHTML = ''; return; }
        const isSel = portfolioSelected(r);
        const nominal = norm(r.nominal_scenario);
        const max = Math.max(1, ...ps.flatMap(p => p.perScenario.map(s => num(s.residual_gap_tonnes) ?? 0)));
        el.innerHTML = `<div class="table-wrap"><table class="matrix">
            <thead><tr><th>Portfolio</th>${scen.map(s => `<th class="${norm(s) === nominal ? 'nominal' : ''}" title="${esc(human(s))}${norm(s) === nominal ? ' (nominal)' : ''}">${esc(SCEN_SHORT[norm(s)] || human(s))}</th>`).join('')}</tr></thead>
            <tbody>${ps.map(p => {
                const by = Object.fromEntries(p.perScenario.map(s => [norm(s.scenario), s]));
                const worstVal = Math.max(...p.perScenario.map(s => num(s.residual_gap_tonnes) ?? -Infinity));
                return `<tr class="${isSel(p) ? 'row-selected' : ''}"><th scope="row"><span class="mono">${esc(p.id)}</span>${isSel(p) ? ' <span class="star">★</span>' : ''}</th>${scen.map(s => {
                    const c = by[norm(s)];
                    if (!c) return '<td class="mx-na">—</td>';
                    const v = num(c.residual_gap_tonnes);
                    const low = norm(c.applicability) === 'LOW';
                    const infeas = c.feasibility && /NOT|INFEAS/.test(norm(c.feasibility));
                    const ratio = v !== null ? Math.max(0, v) / max : 0;
                    const bg = v === null ? '' : (v <= 0 ? 'background:rgba(63,185,127,.22)' : `background:rgba(229,83,75,${(0.08 + 0.55 * ratio).toFixed(3)})`);
                    return `<td class="mx${low ? ' mx-low' : ''}${v !== null && v === worstVal ? ' mx-worst' : ''}${norm(s) === nominal ? ' nominal' : ''}" style="${bg}" title="${esc(`${p.id} · ${human(s)}: residual ${fmtT(v)} · P50 ${fmtT(c.p50_tonnes)} · applicability ${c.applicability || 'N/A'} · ${c.feasibility || ''}`)}">
                        <span class="mono">${v !== null ? (v / 1000).toFixed(2) : 'N/A'}</span>${low ? '<span class="mx-flag" aria-label="outside model applicability">!</span>' : ''}${infeas ? '<span class="mx-flag" aria-label="not feasible">✕</span>' : ''}
                    </td>`;
                }).join('')}</tr>`;
            }).join('')}</tbody>
        </table></div>`;
        $('#matrixLegend').innerHTML = `<span class="muted">kt</span><span><i class="lg-heat"></i>Residual gap low → high</span><span><i class="lg-flag">!</i>Outside model applicability</span><span><i class="lg-worstcell"></i>Worst tested</span>`;
    }

    function renderRecoveryHero(rec, dec, sel) {
        const r = rec || {};
        const review = r.selection_status === 'REVIEW_REQUIRED';
        const noAct = num(pick(r, 'no_action_expected_gap_tonnes')) ?? num(r.baseline && r.baseline.gap_p50_tonnes);
        $('#rcGap').textContent = fmtT(noAct);
        $('#rcGapSub').textContent = num(r.no_action_worst_case_gap_tonnes) !== null ? `worst tested ${fmtT(r.no_action_worst_case_gap_tonnes)}` : '';
        $('#rcRecovery').textContent = review ? 'WITHHELD' : fmtSigned(pick(r, 'expected_recovery_tonnes'));
        $('#rcSelected').textContent = review ? 'no eligible portfolio' : (selectedKey(r) || (sel && sel.id) || '—');

        const residual = review ? null : (num(pick(r, 'expected_residual_gap_tonnes', 'residual_gap_tonnes')) ?? (sel ? sel.residual : null));
        const el = $('#rcResidual');
        const prev = el.dataset.value ? Number(el.dataset.value) : null;
        el.textContent = review ? 'N/A' : fmtT(residual);
        el.dataset.value = residual ?? '';
        if (prev !== null && residual !== null && prev !== residual) { el.classList.remove('value-change'); void el.offsetWidth; el.classList.add('value-change'); }
        const worst = num(r.worst_case_residual_gap_tonnes);
        $('#rcResidualSub').textContent = review ? 'selection withheld' : (worst !== null ? `worst tested ${fmtT(worst)}${r.worst_case_scenario ? ` · ${human(r.worst_case_scenario).toLowerCase()}` : ''}` : '');
        $('#residualHero').dataset.tone = review ? 'bad' : (residual !== null ? (residual > 0 ? 'bad' : 'ok') : 'neutral');

        const can = pick(r, 'can_operations_close', 'operations_can_close', 'gap_closed', 'operational_sufficient');
        $('#rcCanClose').innerHTML = can === true ? chip('YES', 'ok') : can === false ? chip('NO', 'bad') : chip('NOT RETURNED', 'neutral');
        const horizon = (dec && dec.horizon) || pick(r, 'decision_horizon', 'horizon');
        $('#rcHorizon').innerHTML = horizon ? chip(upperHuman(horizon), 'neutral', /NEAR/.test(norm(horizon)) ? 'Exploration is not treated as immediate recovery.' : 'Exploration contingency may be activated.') : 'N/A';
        $('#rcSelection').innerHTML = review ? chip('REVIEW REQUIRED', 'bad') : (selectedKey(r) ? chip(`SELECTED ${selectedKey(r)}`, 'ok') : chip(upperHuman(r.selection_status || 'N/A'), 'neutral'));
    }

    function renderDecisionBar(el, dec) {
        if (!el) return;
        if (!dec) {
            el.dataset.tone = 'neutral';
            el.innerHTML = `<div class="db-main">${iconHTML('neutral')}<div><span class="label">Decision</span><div class="db-title">UNAVAILABLE</div></div></div>`;
            return;
        }
        const di = decisionInfo(dec.state);
        const key = norm(dec.state);
        el.dataset.tone = di.tone;
        let steps = '';
        if (key === 'OPERATIONAL_AND_EXPLORATION_CONTINGENCY') {
            steps = `<ol class="flow">
                <li>${chip(`RESIDUAL GAP PERSISTS${dec.residual !== null ? ` · ${fmtT(dec.residual)}` : ''}`, 'bad')}</li>
                <li>${chip('EXPLORATION CONTINGENCY ACTIVATED', 'warn')}</li>
                <li>${chip(`NEXT TARGET ${dec.nextTarget || 'NOT RETURNED'}`, 'info')}</li>
            </ol>`;
        } else if (key === 'OPERATIONAL_RESPONSE') {
            steps = `<ol class="flow"><li>${chip('OPERATIONAL RECOVERY RECOMMENDED', 'ok')}</li>${dec.residual !== null ? `<li>${chip(`RESIDUAL ${fmtT(dec.residual)}`, dec.residual > 0 ? 'warn' : 'ok')}</li>` : ''}</ol>`;
        } else if (key === 'REVIEW_REQUIRED') {
            steps = reasonListHTML(dec.reasons, 'bad') || '<div class="muted">No reasons returned.</div>';
        }
        el.innerHTML = `<div class="db-main">${iconHTML(di.tone)}<div><span class="label">Decision</span><div class="db-title">${esc(di.title)}</div></div>${dec.horizon ? `<span class="meta-chip">${esc(upperHuman(dec.horizon))} HORIZON</span>` : ''}</div>
            <div class="db-steps">${steps}</div>
            <div class="db-actions">${dec.nextTarget ? `<button type="button" class="btn btn-primary" data-why-target="${esc(dec.nextTarget)}">Why This Target?</button>` : ''}</div>`;
        el.classList.remove('decision-enter'); void el.offsetWidth; el.classList.add('decision-enter');
    }

    async function evaluateRecovery() {
        const btn = $('#btnEvaluate');
        setLoading(btn, true);
        const previous = state.decision.current;
        const body = currentInputs(false);
        const gen = state.gen;
        // Never leave a previous selection or decision on screen while a new one is computed.
        $('#portfolioChart').innerHTML = loadingHTML('Testing recovery portfolios…');
        $('#scenarioMatrix').innerHTML = '';
        $('#portfolioTable').innerHTML = '';
        $('#rcDecision').dataset.tone = 'neutral';
        $('#rcDecision').innerHTML = loadingHTML('Computing decision…');
        ['#rcGap', '#rcResidual', '#rcRecovery', '#rcSelected'].forEach(s => { $(s).textContent = '…'; });
        ['#rcGapSub', '#rcResidualSub'].forEach(s => { $(s).textContent = ''; });
        ['#rcCanClose', '#rcHorizon', '#rcSelection'].forEach(s => { $(s).innerHTML = '—'; });
        $('#residualHero').dataset.tone = 'neutral';
        let rec = null, con = null, recErr = null, conErr = null;
        try {
            rec = await api.post('/api/recovery/evaluate', body, { timeout: 30000 });
        } catch (e) { recErr = e; }
        const conBody = { ...body };
        if (rec) {
            const sk = selectedKey(rec);
            if (sk) conBody.selected_portfolio = sk;
            const res = num(pick(rec, 'expected_residual_gap_tonnes', 'residual_gap_tonnes'));
            if (res !== null) conBody.residual_gap_tonnes = res;
        }
        try {
            con = await api.post('/api/contingency/evaluate', conBody, { timeout: 30000 });
        } catch (e) { conErr = e; }
        if (gen !== state.gen) { setLoading(btn, false); return; }
        state.recovery = rec;
        state.contingency = con;
        state.loaded.recovery = true;

        let sel = null;
        if (rec) {
            sel = renderPortfolios(rec);
            renderPortfolioChart(rec, con);
            renderScenarioMatrix(rec);
            $('#rcProvenance').innerHTML = provChips(pick(rec, 'provenance'));
        } else {
            $('#portfolioChart').innerHTML = errorHTML('Recovery evaluation unavailable.', recErr, 'recovery');
            $('#scenarioMatrix').innerHTML = '';
            $('#portfolioTable').innerHTML = '';
            $('#rcProvenance').innerHTML = badgeHTML('UNAVAILABLE');
        }
        // Decision: contingency response is authoritative; recovery may carry one too.
        const dec = normDecision(con) || normDecision(rec);
        if (dec && dec.residual === null) dec.residual = num(pick(rec, 'expected_residual_gap_tonnes', 'residual_gap_tonnes'));
        renderRecoveryHero(rec || {}, dec, sel);
        renderDecisionBar($('#rcDecision'), dec);
        if (!dec && conErr) {
            $('#rcDecision').insertAdjacentHTML('beforeend', `<div class="muted">Contingency evaluation failed: ${esc(conErr.userMessage || '')}</div>`);
        }

        if (dec) {
            state.decision.previous = previous;
            state.decision.current = dec;
        }
        $('#btnLogDecision').disabled = !dec;
        if (rec) presetFlipSliders(rec);
        setLoading(btn, false);
    }

    // Sliders start at the backend's current operating state for the selected mine; for the
    // decision-flip demo state they start at the backend's documented perturbation instead.
    function presetFlipSliders(rec) {
        if (state.flipPreset) return;
        const inp = rec.inputs || {};
        const set = (id, v) => { const el = $(id); if (el && num(v) !== null) { el.value = v; el.dispatchEvent(new Event('input')); } };
        set('#flipRainfall', inp.rainfall_7d_mm);
        set('#flipEquip', inp.equipment_availability);
        set('#flipBlast', inp.blast_delay_h);
        const demoFlip = state.supply && state.supply.decision_flip;
        if (demoFlip) {
            const map = { rainfall_7d_mm: '#flipRainfall', equipment_availability: '#flipEquip', blast_delay_h: '#flipBlast' };
            asArray(demoFlip.changed_inputs).forEach(c => { if (map[c.input]) set(map[c.input], c.perturbed); });
        }
        state.flipPreset = true;
    }

    async function runFlip() {
        const btn = $('#btnFlip');
        const el = $('#flipResult');
        setLoading(btn, true);
        el.innerHTML = loadingHTML('Recomputing baseline and perturbed decisions…');
        const inputs = currentInputs(true);
        try {
            const f = await api.post('/api/decision/flip', {
                mine_id: MINE_ID,
                scenario: inputs.scenario,
                actions: inputs.actions,
                baseline_conditions: {},
                perturbed_conditions: inputs.conditions,
            }, { timeout: 60000 });
            state.decision.flipRuns++;
            state.decision.lastFlip = f;
            el.innerHTML = flipResultHTML(f);
        } catch (e) {
            el.innerHTML = errorHTML('Decision flip could not be computed.', e);
        } finally {
            setLoading(btn, false);
        }
    }

    async function loadDecisionHistory() {
        const el = $('#decisionHistory');
        try {
            const h = await api.get(`/api/decision/history?mine_id=${enc(MINE_ID)}`);
            state.decisionHistory = Array.isArray(h) ? h : asArray(pick(h, 'history', 'decisions', 'items'));
            if (!state.decisionHistory.length) { el.innerHTML = '<div class="empty-state">No decisions submitted yet.</div>'; return; }
            el.innerHTML = `<table class="table"><thead><tr><th>Time</th><th>Decision</th><th>Scenario</th><th>Next target</th><th>Status</th></tr></thead>
                <tbody>${state.decisionHistory.slice(0, 20).map(d => {
                    const di = decisionInfo(pick(d, 'decision_state', 'decision'));
                    return `<tr>
                    <td class="mono">${esc(pick(d, 'timestamp', 'created_at', 'time') || '—')}</td>
                    <td>${chip(di.title, di.tone)}</td>
                    <td>${esc(human(pick(d, 'scenario') || '—'))}</td>
                    <td class="mono">${esc(pick(d, 'next_target') || '—')}</td>
                    <td>${esc(upperHuman(pick(d, 'review_status', 'status') || '—'))}</td>
                </tr>`;
                }).join('')}</tbody></table>`;
        } catch (e) {
            el.innerHTML = errorHTML('Review log unavailable.', e);
        }
    }

    async function submitDecisionReview() {
        const btn = $('#btnLogDecision');
        const dec = state.decision.current;
        if (!dec) return;
        setLoading(btn, true);
        const inputs = currentInputs(state.decision.flipRuns > 0);
        try {
            await api.post('/api/decision/review', {
                ...inputs,
                decision_state: dec.state,
                decision_horizon: dec.horizon || null,
                next_target: dec.nextTarget || null,
                selected_portfolio: state.recovery ? selectedKey(state.recovery) : null,
            });
            showToast('Decision submitted for review.', 'success');
            loadDecisionHistory();
        } catch (e) {
            showToast(`Review submission failed. ${e.userMessage || ''}`, 'error');
        } finally {
            setLoading(btn, false);
            btn.disabled = !state.decision.current;
        }
    }

    // ═══════════════════════════════════════════════════════
    // MODEL TRUTH — status first, technical depth on demand
    // ═══════════════════════════════════════════════════════
    function metricValueHTML(v) {
        if (v === undefined || v === null || v === '') return null;
        if (typeof v === 'boolean') return v ? 'YES' : 'NO';
        if (typeof v === 'number') return esc(fmtNum(v, Math.abs(v) > 0 && Math.abs(v) < 0.01 ? 4 : 3));
        if (typeof v === 'string') return esc(v);
        if (Array.isArray(v)) return v.length ? esc(v.map(x => typeof x === 'object' ? JSON.stringify(x) : x).join(', ')) : null;
        const entries = Object.entries(v).filter(([, x]) => x !== null && x !== undefined && typeof x !== 'object');
        if (!entries.length) return null;
        return entries.map(([k, x]) => `<span class="mv-sub">${esc(human(k))}: <b>${typeof x === 'number' ? esc(fmtNum(x, 3)) : esc(x)}</b></span>`).join('');
    }

    function metricsGridHTML(src, defs) {
        return `<dl class="metric-grid">${defs.map(([label, keys, def]) => {
            const val = metricValueHTML(deepPick(src, keys));
            return `<div class="metric ${val === null ? 'metric-na' : ''}">
                <dt>${esc(label)}</dt>
                <dd>${val === null ? 'N/A <span class="muted">(not in validation report)</span>' : val}</dd>
                ${def ? `<div class="metric-def">${esc(def)}</div>` : ''}
            </div>`;
        }).join('')}</dl>`;
    }

    function trustNotes(src, keys = ['notes', 'note', 'caveats', 'message']) {
        const n = asArray(pick(src, ...keys));
        return n.length ? `<ul class="plain-list">${n.map(x => `<li>${esc(reasonText(x))}</li>`).join('')}</ul>` : '';
    }

    function provenanceRowIf(src) {
        const p = pick(src, 'provenance');
        return p ? `<div class="prov-row prov-block">${provenanceHTML(p)}</div>` : '';
    }

    function truthContextFor(section) {
        return ({ exploration: 'exploration', targets: 'exploration', 'production-risk': 'production', recovery: 'recovery' })[section] || null;
    }

    async function loadTrust() {
        const btn = $('#btnRefreshTrust');
        setLoading(btn, true);
        $('#truthTiles').innerHTML = loadingHTML('Loading model truth…');
        ['#trustExploration', '#trustProduction', '#trustReconciliation', '#trustProvenance', '#trustRecovery'].forEach(s => { $(s).innerHTML = loadingHTML(); });
        const gen = state.gen;
        const [ex, pr, pv, rc, rv] = await Promise.allSettled([
            api.get('/api/trust/exploration'),
            api.get('/api/trust/production'),
            api.get('/api/trust/provenance'),
            api.get(`/api/production/reconciliation?mine_id=${enc(MINE_ID)}`),
            api.get(`/api/trust/recovery?mine_id=${enc(MINE_ID)}`),
            ensureTargets(),
            state.supplyPromise || loadSupply(),
        ]);
        if (gen !== state.gen) { setLoading(btn, false); return; }
        state.trust.exploration = ex.status === 'fulfilled' ? ex.value : null;
        state.trust.production = pr.status === 'fulfilled' ? pr.value : null;
        state.trust.provenance = pv.status === 'fulfilled' ? pv.value : null;
        state.trust.reconciliation = rc.status === 'fulfilled' ? rc.value : null;
        state.trust.recovery = rv.status === 'fulfilled' ? rv.value : null;
        state.loaded['model-trust'] = true;

        $('#trustExploration').innerHTML = ex.status === 'fulfilled'
            ? provenanceRowIf(ex.value) + metricsGridHTML(ex.value, [
                ['Spatial validation', ['spatial_validation', 'validation_method'], 'How held-out data were formed (no random pixel splits).'],
                ['ROC-AUC (spatial CV)', ['roc_auc'], 'Ranking of held-out MRDS cells vs unlabelled background; 0.5 = no skill.'],
                ['PR-AUC (spatial CV)', ['pr_auc'], 'Precision-recall area; compare with the prevalence baseline.'],
                ['PR-AUC prevalence baseline', ['pr_auc_prevalence_baseline'], 'PR-AUC a random ranking would get.'],
                ['Top-area capture', ['top_area_capture'], 'Share of held-out occurrences inside the top 5/10/20 % of held-out area.'],
                ['Region holdout ROC-AUC', ['region_holdout'], 'Train on one half of the belt, test on the other.'],
                ['Region holdout method', ['region_holdout_method'], null],
                ['Observation window', ['observation_window'], 'Fixed reference composite, not a current image.'],
                ['Effective resolution', ['effective_resolution'], null],
                ['Subsurface evidence', ['subsurface_evidence'], null],
                ['Applicability method', ['applicability_method'], 'How out-of-experience inputs are detected.'],
                ['Uncertainty method', ['uncertainty_method'], 'How rank uncertainty is measured.'],
                ['Calibration', ['calibration'], null],
                ['Model version', ['model_version'], null],
            ]) + experimentsHTML(ex.value.experiments)
              + (asArray(ex.value.limitations).length ? `<h4 class="mini-title">Limitations</h4>${trustNotes(ex.value, ['limitations'])}` : '')
              + (asArray(ex.value.notes).length ? `<h4 class="mini-title">Notes</h4>${trustNotes(ex.value)}` : '')
            : errorHTML('Exploration validation unavailable.', ex.reason, 'model-trust');

        $('#trustProduction').innerHTML = pr.status === 'fulfilled'
            ? provenanceRowIf(pr.value) + metricsGridHTML(pr.value, [
                ['Validation method', ['validation_method'], 'Chronological rolling-origin backtest; no shuffling.'],
                ['Test window', ['test_window'], 'Untouched periods used for every metric below.'],
                ['P50 MAE (t / 7 days)', ['mae'], 'Mean absolute error of the P50 forecast on the test window.'],
                ['Best naive baseline MAE', ['baseline_mae'], 'Better of previous-period and 4-period moving average.'],
                ['Improvement vs best baseline (%)', ['mae_improvement_vs_best_baseline_pct'], 'Negative would mean the baseline wins.'],
                ['RMSE (t / 7 days)', ['rmse'], null],
                ['R²', ['r2'], null],
                ['Observed P10–P90 coverage', ['observed_coverage'], 'Share of test periods whose actual fell inside P10–P90.'],
                ['Nominal coverage', ['nominal_coverage'], 'What a P10–P90 interval is designed to contain.'],
                ['Quantiles validated', ['quantiles_validated'], 'Passes the documented coverage + hit-rate rule?'],
                ['Quantile hit rates', ['quantile_hit_rate'], 'Share of actuals at or below P10 / P50 / P90.'],
                ['Pinball loss', ['pinball_loss'], 'Quantile loss per level (lower is better).'],
                ['Shortfall classification', ['shortfall_classification'], null],
                ['Calibration window', ['protocol'], null],
                ['Forecast procedure', ['forecast_procedure'], 'Persistence assumption for unknown future conditions.'],
                ['Uncertainty method', ['uncertainty_method'], null],
                ['Applicability method', ['applicability_method'], null],
                ['Model version', ['model_version'], null],
            ]) + (asArray(pr.value.limitations).length ? `<h4 class="mini-title">Limitations</h4>${trustNotes(pr.value, ['limitations'])}` : '')
              + (asArray(pr.value.notes).length ? `<h4 class="mini-title">Notes</h4>${trustNotes(pr.value)}` : '')
            : errorHTML('Production validation unavailable.', pr.reason, 'model-trust');

        if (rv.status === 'fulfilled') renderRecoveryTrust(rv.value);
        else $('#trustRecovery').innerHTML = errorHTML('Recovery diagnostics unavailable.', rv.reason, 'model-trust');
        renderProvenanceTrust(pv);
        renderReconciliation(rc);
        renderTruth();
        updateDataModePill();
        setLoading(btn, false);
    }

    // Label-source / feature experiments: untouched western test region, 4-seed averages (backend values only).
    function experimentsHTML(e) {
        if (!e || !e.final_test_seed_averaged) return '';
        const rows = Object.entries(e.final_test_seed_averaged).map(([k, v]) => `<tr${k === e.deployed ? ' class="row-selected"' : ''}>
            <td>${esc(upperHuman(k))}${k === e.deployed ? ' <b>(deployed)</b>' : ''}</td><td class="mono">${fmtNum(v.test_roc, 3)} ± ${fmtNum(v.test_roc_sd, 3)}</td>
            <td class="mono">${fmtNum(v.test_pr, 4)}</td><td class="mono">${fmtNum(v.test_cap10 * 100, 0)} %</td></tr>`).join('');
        return `<h4 class="mini-title">Model Comparison · untouched western test region <span class="muted">(prevalence ${fmtNum(e.final_test_prevalence, 4)}; mean of 4 seeds)</span></h4>
            <div class="table-wrap"><table class="table"><thead><tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Positives in top 10 % area</th></tr></thead><tbody>${rows}</tbody></table></div>
            <div class="note">${esc(e.decision_reason || '')} ${esc(e.supplementary_note || '')}</div>
            <div class="note">Real + synthetic (model D): ${esc(e.model_d && e.model_d.reason ? e.model_d.reason : 'not run')}</div>`;
    }

    function renderRecoveryTrust(r) {
        const blocked = asArray(r.applicability_blocked_portfolios);
        $('#trustRecovery').innerHTML = `${provenanceRowIf(r)}
            ${dlHTML([
                ['Scenarios tested', esc(asArray(r.scenarios_tested).map(human).join(', '))],
                ['Portfolios evaluated', esc(r.portfolios_evaluated)],
                ['Eligible', esc(asArray(r.eligible_portfolios).join(', ') || 'none')],
                ['Applicability-blocked', blocked.length ? blocked.map(b => `<span class="mono">${esc(b.portfolio)}</span> <span class="muted">[${esc(asArray(b.scenarios).map(human).join(', '))}]</span>`).join('<br>') : 'none'],
                ['Feasibility-blocked', esc(asArray(r.feasibility_blocked_portfolios).map(x => typeof x === 'object' ? (x.portfolio || JSON.stringify(x)) : x).join(', ') || 'none')],
                ['Worst-case residual: no action → selected', `<span class="mono">${esc(fmtT(r.baseline_worst_case_residual_gap_tonnes))} → ${esc(fmtT(r.selected_worst_case_residual_gap_tonnes))}</span> ${esc(r.selected_portfolio || upperHuman(r.selection_status || ''))}${r.selected_intervention_burden != null ? ` · burden ${esc(fmtNum(r.selected_intervention_burden, 1))}` : ''}`],
                ['Constraint notes', Object.keys(r.constraint_notes || {}).length ? Object.entries(r.constraint_notes).map(([k, v]) => `${esc(k)}: ${esc(asArray(v).join(' '))}`).join('; ') : ''],
                ['Selection rule', esc(r.selection_rule || '')],
                ['Applicability policy', r.applicability_policy ? Object.entries(r.applicability_policy).map(([k, v]) => `<b>${esc(human(k))}</b>: ${esc(v)}`).join('<br>') : ''],
                ['Note', esc(r.note || '')],
            ])}`;
    }

    function renderProvenanceTrust(pv) {
        const el = $('#trustProvenance');
        if (pv.status !== 'fulfilled') { el.innerHTML = errorHTML('Provenance unavailable.', pv.reason, 'model-trust'); return; }
        const p = pv.value;
        const entries = Array.isArray(p) ? p.map((x, i) => [pick(x, 'name', 'dataset', 'component') || `Source ${i + 1}`, x])
            : Object.entries(p.sources || p.datasets || p).filter(([k]) => k !== 'notes');
        if (!entries.length) { el.innerHTML = '<div class="empty-state">No provenance entries returned.</div>'; return; }
        el.innerHTML = `<div class="table-wrap"><table class="table"><tbody>${entries.map(([k, v]) => {
            let mode = null, desc = '';
            if (typeof v === 'string') { if (modeInfo(v).known) mode = v; else desc = v; }
            else if (v && typeof v === 'object') {
                mode = pick(v, 'mode', 'data_mode', 'type', 'status');
                desc = [pick(v, 'description', 'source'), pick(v, 'provider'), fmtWindow(pick(v, 'observation_window')) ? fmtWin(fmtWindow(pick(v, 'observation_window'))) : null, pick(v, 'note')].filter(Boolean).join(' · ');
            } else if (typeof v === 'boolean') desc = v ? 'yes' : 'no';
            return `<tr><th scope="row">${esc(human(k))}</th><td>${mode ? badgeHTML(mode) : ''}</td><td class="muted">${esc(desc)}</td></tr>`;
        }).join('')}</tbody></table></div>
        ${dlHTML([
            ['Model versions', p.model_versions ? Object.entries(p.model_versions).map(([k, v]) => `<span class="code">${esc(human(k))}: ${esc(v)}</span>`).join(' ') : ''],
            ['Policies', p.policies ? Object.entries(p.policies).filter(([k]) => k !== 'note').map(([k, v]) => `<span class="code">${esc(human(k))}: ${esc(v)}</span>`).join(' ') + (p.policies.note ? `<div class="note">${esc(p.policies.note)}</div>` : '') : ''],
            ['Data modes', p.modes ? Object.entries(p.modes).map(([k, v]) => `${badgeHTML(k)} <span class="muted">${esc(v)}</span>`).join('<br>') : ''],
        ])}
        ${trustNotes(p)}`;
    }

    function renderReconciliation(rc) {
        const el = $('#trustReconciliation');
        if (rc.status !== 'fulfilled') { el.innerHTML = errorHTML('Reconciliation unavailable.', rc.reason, 'model-trust'); return; }
        const r = rc.value || {};
        const rows = recRows(r);
        const sm = r.summary || {};
        const hasActual = rows.some(x => x.actual !== null);
        if (!hasActual) {
            el.innerHTML = `${provenanceRowIf(r)}<div class="empty-state">Actual data unavailable${r.status ? ` (${esc(upperHuman(r.status))})` : ''}; accuracy cannot be reconciled.</div>`;
            return;
        }
        el.innerHTML = provenanceRowIf(r) + dlHTML([
            ['Basis', esc(r.basis || '')],
            ['Error convention', esc(r.error_convention || '')],
            ['Over / under forecasts', num(sm.over_forecast_count) !== null ? `${esc(sm.over_forecast_count)} / ${esc(sm.under_forecast_count)}` : ''],
            ['Latest error', num(sm.latest_error_tonnes) !== null ? esc(fmtSigned(sm.latest_error_tonnes)) : ''],
        ]) + `<div class="table-wrap"><table class="table"><thead><tr><th>Period</th><th>Forecast</th><th>Actual</th><th>Error</th></tr></thead>
            <tbody>${rows.map(x => `<tr><td class="mono">${esc(x.period ?? '—')}</td><td class="mono">${esc(fmtT(x.forecast))}</td><td class="mono">${x.actual !== null ? esc(fmtT(x.actual)) : 'Actual unavailable'}</td><td class="mono">${x.error !== null ? esc(fmtSigned(x.error)) : 'N/A'}</td></tr>`).join('')}</tbody></table></div>`;
    }

    function recRows(r) {
        return asArray(Array.isArray(r) ? r : pick(r, 'rows', 'records', 'reconciliation', 'history', 'periods')).map(x => ({
            period: pick(x, 'period', 'date', 'period_end'),
            start: pick(x, 'period_start', 'date'),
            forecast: num(pick(x, 'forecast_tonnes', 'forecast', 'p50_tonnes', 'p50')),
            p10: num(pick(x, 'p10_tonnes')),
            p90: num(pick(x, 'p90_tonnes')),
            actual: num(pick(x, 'actual_tonnes', 'actual')),
            error: num(pick(x, 'error_tonnes', 'error')),
        }));
    }

    function reviewTile(ctx) {
        const s = state.supply || {};
        const di = decisionInfo(s.decision_state);
        if (ctx === 'recovery' && state.recovery && state.recovery.selection_status === 'REVIEW_REQUIRED') {
            return { label: 'Review', value: 'REQUIRED', sub: 'portfolio selection withheld', tone: 'bad' };
        }
        if (norm(s.decision_state) === 'REVIEW_REQUIRED') return { label: 'Review', value: 'REQUIRED', sub: 'automated recommendation withheld', tone: 'bad' };
        if (s.human_review_required === true) return { label: 'Review', value: 'REQUIRED', sub: `human sign-off · decision ${di.short.toLowerCase()}`, tone: 'warn' };
        return { label: 'Review', value: s.decision_state ? 'NOT FLAGGED' : 'N/A', sub: '', tone: 'neutral' };
    }

    function focusTarget() {
        const ex = state.exploration, meta = ex.targetsMeta || {};
        const id = ex.selectedId || meta.selected_target || (ex.targets[0] && ex.targets[0].id);
        return id ? (mergedTarget(id) || null) : null;
    }

    function truthTiles(ctx) {
        const s = state.supply || {};
        if (ctx === 'exploration') {
            const te = state.trust.exploration || {};
            const t = focusTarget();
            const r = (t && t.raw) || {};
            const lvl = t ? maturityLevel(t.maturity) : null;
            const prov = te.provenance || {};
            return [
                { label: 'Applicability', value: t ? (t.appl.level || 'N/A') : 'N/A', sub: t ? `target ${t.id}` : 'no target selected', tone: t ? applTone(t.appl.level) : 'neutral' },
                { label: 'Uncertainty', value: t ? (t.unc.level || 'N/A') : 'N/A', sub: num(r.rank_sd) !== null ? `ensemble rank SD ${fmtNum(r.rank_sd, 1)}` : '', tone: t ? uncTone(t.unc.level) : 'neutral' },
                { label: 'Evidence', value: lvl !== null ? `L${lvl}` : 'N/A', sub: r.subsurface_status ? `subsurface ${human(r.subsurface_status).toLowerCase()}` : '', tone: evTone(lvl) },
                { label: 'Validation', value: num(te.roc_auc) !== null ? `ROC-AUC ${fmtNum(te.roc_auc, 3)}` : 'N/A', sub: te.calibration && /^NOT CALIBRATED/i.test(te.calibration) ? 'spatial CV · relative rank, not calibrated' : 'spatial CV', tone: num(te.roc_auc) !== null ? 'info' : 'neutral', mono: true },
                { label: 'Data', value: prov.data_mode ? modeInfo(prov.data_mode).short : 'N/A', sub: [prov.features_mode && `features ${modeInfo(prov.features_mode).short.toLowerCase()}`, prov.observation_window && `window ${fmtWin(prov.observation_window)}`].filter(Boolean).join(' · '), tone: prov.data_mode ? dataTone(prov.data_mode) : 'neutral' },
                reviewTile(ctx),
            ];
        }
        if (ctx === 'production') {
            const tp = state.trust.production || {};
            const f = state.production.forecast;
            const appl = (f && f.applicability) || { level: s.forecast_applicability };
            const viol = asArray(appl.range_violations).map(v => human(v.feature || v)).join(', ');
            const iv = intervalInfo(f || s);
            const prov = tp.provenance || {};
            const imp = num(tp.mae_improvement_vs_best_baseline_pct);
            return [
                { label: 'Applicability', value: appl.level || 'N/A', sub: viol ? `outside range: ${viol}` : (appl.level ? 'current inputs within training range' : ''), tone: applTone(appl.level) },
                { label: 'Uncertainty', value: iv.validated ? 'P10–P90 VALIDATED' : 'NOT VALIDATED', sub: iv.obs !== null ? `coverage ${fmtPct(iv.obs, 1)} · nominal ${fmtPct(iv.nom)}` : '', tone: appl.level === 'LOW' ? 'warn' : (iv.validated ? 'ok' : 'warn') },
                { label: 'Evidence', value: num(tp.n_test_periods) !== null ? `${tp.n_test_periods} PERIODS` : 'N/A', sub: 'rolling-origin backtest · untouched window', tone: 'info' },
                { label: 'Validation', value: num(tp.mae) !== null ? `MAE ${fmtT(tp.mae)}` : 'N/A', sub: num(tp.baseline_mae) !== null ? `best baseline ${fmtT(tp.baseline_mae)}${imp !== null ? ` (${imp > 0 ? '+' : ''}${fmtNum(imp, 1)}%)` : ''}` : '', tone: 'info', mono: true },
                { label: 'Data', value: prov.operations_mode ? modeInfo(prov.operations_mode).short : (prov.data_mode ? modeInfo(prov.data_mode).short : 'N/A'), sub: prov.weather_mode ? `weather ${human(prov.weather_mode).toLowerCase()}` : '', tone: dataTone(prov.operations_mode || prov.data_mode) },
                reviewTile(ctx),
            ];
        }
        // recovery
        const tr = state.trust.recovery || {};
        const elig = asArray(tr.eligible_portfolios), blocked = asArray(tr.applicability_blocked_portfolios);
        const prov = tr.provenance || {};
        const n = num(tr.portfolios_evaluated);
        return [
            { label: 'Applicability', value: n !== null ? `${elig.length}/${n} ELIGIBLE` : 'N/A', sub: `${blocked.length} blocked by low applicability`, tone: n === null ? 'neutral' : (elig.length ? (blocked.length ? 'warn' : 'ok') : 'bad') },
            { label: 'Uncertainty', value: `${asArray(tr.scenarios_tested).length} SCENARIOS`, sub: s.worst_case_scenario ? `worst tested: ${human(s.worst_case_scenario).toLowerCase()}` : 'robust worst-case selection', tone: 'info' },
            { label: 'Evidence', value: 'MODELLED', sub: 'simulator counterfactuals · no accuracy claimed', tone: 'warn' },
            { label: 'Feasibility', value: 'MODELLED ONLY', sub: 'input-constraint check · site confirmation required', tone: 'warn' },
            { label: 'Data', value: prov.data_mode ? modeInfo(prov.data_mode).short : 'N/A', sub: prov.operations_mode ? `operations ${modeInfo(prov.operations_mode).short.toLowerCase()}` : '', tone: dataTone(prov.data_mode) },
            reviewTile(ctx),
        ];
    }

    function svgCurve(points, { w = 280, h = 190 } = {}) {
        const m = { l: 34, r: 20, t: 10, b: 28 };
        const X = v => m.l + v * (w - m.l - m.r), Y = v => m.t + (1 - v) * (h - m.t - m.b);
        const pts = [{ a: 0, c: 0 }, ...points].map(p => `${X(p.a).toFixed(1)},${Y(p.c).toFixed(1)}`).join(' ');
        const grid = [0, 0.5, 1].map(v => `<line class="ch-grid" x1="${X(0)}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}"></line><text class="ch-tick" x="${X(0) - 6}" y="${Y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join('')
            + [0, 0.5, 1].map(v => `<text class="ch-tick" x="${X(v)}" y="${h - 10}" text-anchor="middle">${v * 100}%</text>`).join('');
        return `<svg class="curve-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="Success-rate curve: share of held-out occurrences captured versus share of area ranked">
            ${grid}
            <line class="ch-random" x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"></line>
            <polyline class="ch-curve" points="${pts}"></polyline>
            <text class="ch-lbl" x="${X(0.62)}" y="${Y(0.52)}">random</text>
        </svg>`;
    }

    function hbar(label, value, max, { cls = '', txt, mark } = {}) {
        const v = num(value);
        return `<div class="hb"><span class="hb-k">${esc(label)}</span><span class="hb-track"><span class="hb-fill ${cls}" style="width:${v !== null ? Math.min(100, v / max * 100) : 0}%"></span>${mark !== undefined && num(mark) !== null ? `<span class="hb-mark" style="left:${Math.min(100, num(mark) / max * 100)}%"></span>` : ''}</span><span class="hb-v mono">${esc(txt ?? (v !== null ? fmtNum(v, 3) : 'N/A'))}</span></div>`;
    }

    function renderTruthValidation(ctx) {
        const el = $('#truthValidation');
        if (ctx === 'exploration') {
            const te = state.trust.exploration;
            if (!te) { el.innerHTML = '<div class="empty-state">Validation unavailable.</div>'; return; }
            $('#truthValTitle').textContent = 'Validation · Spatially Held-Out';
            $('#truthValAside').innerHTML = badgeHTML(te.provenance && te.provenance.data_mode);
            const curve = asArray(te.success_rate_curve).map(p => ({ a: num(p.area_fraction), c: num(p.captured_fraction) })).filter(p => p.a !== null && p.c !== null);
            const cap = te.top_area_capture || {};
            const folds = asArray(te.per_fold).map(f => num(f.roc_auc)).filter(v => v !== null);
            const rh = te.region_holdout && num(te.region_holdout.roc_auc);
            const X = v => ((v - 0.5) / 0.5 * 100).toFixed(1);
            el.innerHTML = `<div class="val-grid">
                <div>
                    <span class="label">Success-Rate Curve${num(te.success_rate_auc) !== null ? ` <span class="muted">AUC ${esc(fmtNum(te.success_rate_auc, 3))}</span>` : ''}</span>
                    ${curve.length ? svgCurve(curve) : '<div class="muted">Not returned.</div>'}
                    <div class="axis-cap"><span>area ranked →</span><span>↑ occurrences captured</span></div>
                </div>
                <div class="val-side">
                    <span class="label">Top-Area Capture <span class="muted">model vs random</span></span>
                    ${[['top_5pct_area', 0.05, 'Top 5%'], ['top_10pct_area', 0.10, 'Top 10%'], ['top_20pct_area', 0.20, 'Top 20%']].filter(([k]) => num(cap[k]) !== null).map(([k, rnd, lbl]) => hbar(lbl, cap[k], 1, { cls: 'hb-accent', txt: `${fmtPct(cap[k])} vs ${fmtPct(rnd)}`, mark: rnd })).join('')}
                    <span class="label">ROC-AUC per Fold <span class="muted">0.5 = no skill</span></span>
                    <div class="dotstrip" role="img" aria-label="Per-fold ROC-AUC">
                        <span class="ds-axis"></span>
                        ${folds.map(v => `<span class="ds-dot" style="left:${X(v)}%" title="fold ROC-AUC ${fmtNum(v, 3)}"></span>`).join('')}
                        ${num(te.roc_auc) !== null ? `<span class="ds-mean" style="left:${X(te.roc_auc)}%" title="mean ${fmtNum(te.roc_auc, 3)}"></span>` : ''}
                        ${rh !== null ? `<span class="ds-alt" style="left:${X(rh)}%" title="region holdout ${fmtNum(rh, 3)}"></span>` : ''}
                    </div>
                    <div class="ds-scale mono"><span>0.5</span><span>0.75</span><span>1.0</span></div>
                    <div class="chart-legend chart-legend-sm"><span><i class="lg-dot" style="background:var(--text-2)"></i>fold</span><span><i class="lg-p50line"></i>mean ${esc(fmtNum(te.roc_auc, 3))}</span>${rh !== null ? `<span><i class="lg-dot" style="background:var(--warn)"></i>region holdout ${esc(fmtNum(rh, 3))}</span>` : ''}</div>
                    <div class="kv-inline"><span class="label">PR-AUC</span><span class="mono">${esc(fmtNum(te.pr_auc, 4))}</span><span class="muted">vs random ${esc(fmtNum(te.pr_auc_prevalence_baseline, 4))}</span></div>
                </div>
            </div>`;
            return;
        }
        if (ctx === 'production') {
            const tp = state.trust.production;
            if (!tp) { el.innerHTML = '<div class="empty-state">Validation unavailable.</div>'; return; }
            $('#truthValTitle').textContent = 'Validation · Rolling-Origin Backtest';
            $('#truthValAside').innerHTML = badgeHTML(tp.provenance && tp.provenance.data_mode);
            const hr = tp.quantile_hit_rate || {};
            const bl = tp.baseline || {};
            const maes = [['Model P50', tp.mae, 'hb-accent'], ['4-period mean', bl.moving_average_4 && bl.moving_average_4.mae, ''], ['Previous period', bl.previous_period && bl.previous_period.mae, '']].filter(([, v]) => num(v) !== null);
            const maxMae = Math.max(1, ...maes.map(([, v]) => num(v)));
            const rec = state.trust.reconciliation;
            el.innerHTML = `<div class="val-grid">
                <div>
                    <span class="label">Forecast vs Actual <span class="muted">out-of-sample backtest</span></span>
                    ${rec ? reconChartHTML(rec) : '<div class="muted">Reconciliation unavailable.</div>'}
                </div>
                <div class="val-side">
                    <span class="label">P10–P90 Coverage <span class="muted">tick = nominal</span></span>
                    ${hbar('Observed', tp.observed_coverage, 1, { cls: 'hb-accent', txt: fmtPct(tp.observed_coverage, 1), mark: tp.nominal_coverage })}
                    ${num(tp.p10_p90_coverage_before_recalibration) !== null ? hbar('Before recalibration', tp.p10_p90_coverage_before_recalibration, 1, { txt: fmtPct(tp.p10_p90_coverage_before_recalibration, 1), mark: tp.nominal_coverage }) : ''}
                    <span class="label">Quantile Hit Rates <span class="muted">tick = nominal</span></span>
                    ${[['P10', 'p10', 0.1], ['P50', 'p50', 0.5], ['P90', 'p90', 0.9]].filter(([, k]) => num(hr[k]) !== null).map(([l, k, nom]) => hbar(l, hr[k], 1, { cls: 'hb-accent', txt: `${fmtNum(hr[k], 3)} vs ${nom}`, mark: nom })).join('')}
                    <span class="label">MAE <span class="muted">t / 7 days · lower is better</span></span>
                    ${maes.map(([l, v, c]) => hbar(l, v, maxMae, { cls: c, txt: fmtT(v) })).join('')}
                </div>
            </div>`;
            return;
        }
        const tr = state.trust.recovery;
        if (!tr) { el.innerHTML = '<div class="empty-state">Recovery diagnostics unavailable.</div>'; return; }
        $('#truthValTitle').textContent = 'Recovery Diagnostics';
        $('#truthValAside').innerHTML = badgeHTML(tr.provenance && tr.provenance.data_mode);
        const bw = num(tr.baseline_worst_case_residual_gap_tonnes), sw = num(tr.selected_worst_case_residual_gap_tonnes);
        const maxW = Math.max(1, bw ?? 0, sw ?? 0);
        const blocked = Object.fromEntries(asArray(tr.applicability_blocked_portfolios).map(b => [b.portfolio, asArray(b.scenarios)]));
        const all = [...new Set([...asArray(tr.eligible_portfolios), ...Object.keys(blocked)])];
        el.innerHTML = `<div class="val-side">
            <span class="label">Worst-Case Residual Gap <span class="muted">no action → selected</span></span>
            ${hbar('No action', bw, maxW, { txt: fmtT(bw) })}
            ${hbar(tr.selected_portfolio ? `Selected ${tr.selected_portfolio}` : 'Selected', sw, maxW, { cls: 'hb-accent', txt: tr.selected_portfolio ? fmtT(sw) : upperHuman(tr.selection_status || 'N/A') })}
            <span class="label">Portfolio Eligibility</span>
            <div class="chip-row">${all.map(p => blocked[p] ? chip(`${p} BLOCKED`, 'bad', `Low applicability in: ${blocked[p].map(human).join(', ')}`) : chip(`${p} ELIGIBLE`, 'ok')).join('')}</div>
            <span class="label">Scenarios Tested</span>
            <div class="chip-row">${asArray(tr.scenarios_tested).map(x => `<span class="code">${esc(human(x))}</span>`).join('')}</div>
        </div>`;
    }

    function reconChartHTML(r) {
        const rows = recRows(r).filter(x => x.forecast !== null);
        if (!rows.length) return '<div class="muted">No rows.</div>';
        const vals = rows.flatMap(x => [x.forecast, x.actual, x.p10, x.p90]).filter(v => v !== null);
        const lo = Math.min(...vals) * 0.95, hi = Math.max(...vals) * 1.03;
        const W = 420, H = 190, m = { l: 48, r: 8, t: 10, b: 24 };
        const X = i => m.l + (rows.length === 1 ? 0.5 : i / (rows.length - 1)) * (W - m.l - m.r);
        const Y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
        const bandPts = rows.filter(x => x.p10 !== null && x.p90 !== null);
        const band = bandPts.length > 1 ? `<polygon class="ch-band" points="${rows.map((x, i) => x.p90 !== null ? `${X(i)},${Y(x.p90)}` : '').filter(Boolean).join(' ')} ${rows.map((x, i) => x.p10 !== null ? `${X(i)},${Y(x.p10)}` : '').filter(Boolean).reverse().join(' ')}"></polygon>` : '';
        const fc = rows.map((x, i) => `${i ? 'L' : 'M'}${X(i).toFixed(1)},${Y(x.forecast).toFixed(1)}`).join('');
        const act = rows.map((x, i) => x.actual !== null ? `<circle class="ch-act-dot" cx="${X(i)}" cy="${Y(x.actual)}" r="2.6"></circle>` : '').join('');
        const grid = [lo, (lo + hi) / 2, hi].map(v => `<line class="ch-grid" x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"></line><text class="ch-tick" x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${esc(fmtT(v))}</text>`).join('');
        const sm = r.summary || {};
        return `<svg class="curve-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Backtest forecast versus synthetic actual">${grid}${band}<path class="ch-fc" d="${fc}" stroke="${C_FORECAST}"></path>${act}
            <text class="ch-tick" x="${m.l}" y="${H - 6}">${esc(String(rows[0].start || rows[0].period || '').slice(0, 10))}</text><text class="ch-tick" x="${W - m.r}" y="${H - 6}" text-anchor="end">${esc(String(rows[rows.length - 1].start || rows[rows.length - 1].period || '').slice(0, 10))}</text></svg>
            <div class="chart-legend chart-legend-sm"><span><i class="lg-line" style="background:${C_FORECAST}"></i>Forecast P50</span><span><i class="lg-dot" style="background:${C_ACTUAL}"></i>Actual (synthetic)</span><span><i class="lg-sw lg-band"></i>P10–P90</span></div>
            <div class="kv-inline"><span class="label">MAE</span><span class="mono">${esc(fmtT(sm.mae ?? r.mae))}</span><span class="label">Bias</span><span class="mono">${esc(fmtSigned(sm.bias ?? r.bias))}</span>${num(sm.p10_p90_coverage) !== null ? `<span class="label">Coverage</span><span class="mono">${esc(fmtPct(sm.p10_p90_coverage, 1))}</span>` : ''}<span class="muted">${esc(sm.periods ?? rows.length)} periods</span></div>`;
    }

    const CTX_SOURCES = {
        exploration: ['exploration_labels', 'sentinel2', 'modis_lst', 'dem', 'geomorphology_lineaments', 'geology_macrostrat', 'exploration_blocks', 'subsurface_observed', 'exploration_grid', 'next_evidence_sensitivity', 'reserves'],
        production: ['operations', 'weather_imd', 'weather_era5', 'production_moil', 'demo_states'],
        recovery: ['operations', 'recovery_scenarios', 'weather_imd', 'demo_states'],
    };

    function sourcesHTML(ctx) {
        const p = state.trust.provenance;
        const src = p && p.sources;
        if (!src) return '<div class="empty-state">Provenance unavailable.</div>';
        const keys = (CTX_SOURCES[ctx] || Object.keys(src)).filter(k => src[k]);
        return `<ul class="src-list">${keys.map(k => {
            const v = src[k];
            const m = modeInfo(v.mode);
            const win = fmtWindow(v.observation_window);
            const title = [v.description, v.provider, v.note, v.use].filter(Boolean).join(' · ');
            return `<li title="${esc(title)}"><span class="badge badge-${m.cls}">${esc(m.short)}</span><span class="src-name">${esc(human(k))}</span><span class="src-desc">${esc(v.description || '')}${win ? ` · ${esc(fmtWin(win))}` : ''}</span></li>`;
        }).join('')}</ul>
        <button type="button" class="btn btn-ghost btn-sm" data-open-tech>All Sources &amp; Technical Details</button>`;
    }

    function renderTruth() {
        const ctx = state.trust.context;
        $$('#truthTabs .seg-btn').forEach(b => { const on = b.dataset.context === ctx; b.classList.toggle('active', on); b.setAttribute('aria-selected', on ? 'true' : 'false'); });
        $$('.tech-block').forEach(b => { b.hidden = b.dataset.context !== ctx; });
        if (!state.loaded['model-trust']) return;
        $('#truthTiles').innerHTML = truthTiles(ctx).map(tileHTML).join('');
        renderTruthValidation(ctx);
        $('#truthSources').innerHTML = sourcesHTML(ctx);
    }

    function openTruth(ctx) {
        if (ctx) state.trust.context = ctx;
        showSection('model-trust', true, { keepContext: true });
        renderTruth();
    }

    // ═══════════════════════════════════════════════════════
    // NAVIGATION
    // ═══════════════════════════════════════════════════════
    const SECTIONS = {
        'supply-command': { title: 'Overview', load: loadSupply },
        'exploration': { title: 'Exploration', load: loadExploration },
        'production-risk': { title: 'Production Forecast', load: loadProduction },
        'recovery': { title: 'Recovery Scenarios', load: () => { loadDecisionHistory(); return evaluateRecovery(); } },
        'targets': { title: 'Target Priority', load: loadTargets },
        'model-trust': { title: 'Model Truth', load: loadTrust },
    };
    const ALIASES = { overview: 'supply-command', forecast: 'production-risk', production: 'production-risk', priority: 'targets', 'model-truth': 'model-trust' };

    function showSection(id, updateHash = true, { keepContext = false } = {}) {
        id = ALIASES[id] || id;
        if (!SECTIONS[id]) id = 'supply-command';
        if (id === 'model-trust' && !keepContext) {
            const c = truthContextFor(state.currentSection);
            if (c) state.trust.context = c;
        }
        if (truthContextFor(id)) state.lastWorkflow = id;
        const changed = state.currentSection !== id;
        state.currentSection = id;
        if (updateHash && window.location.hash !== `#${id}`) history.replaceState(null, '', `#${id}`);
        $$('.nav-link').forEach(l => {
            const on = l.dataset.section === id;
            l.classList.toggle('active', on);
            if (on) l.setAttribute('aria-current', 'page'); else l.removeAttribute('aria-current');
        });
        $$('.section').forEach(s => s.classList.toggle('active', s.id === `section-${id}`));
        $('#pageTitle').textContent = SECTIONS[id].title;
        if (changed) window.scrollTo({ top: 0 });
        if (!state.loaded[id]) SECTIONS[id].load();
        else if (id === 'targets') renderTargetsScreen();
        else if (id === 'model-trust') renderTruth();
        // leaflet.heat redraws into a 0-width canvas (and throws) if the window resizes while the
        // map is hidden, so the surface is detached off-screen and re-attached once the map is sized.
        if (id !== 'exploration') setSurfaceVisible(false);
        if (id === 'exploration' && state.maps.exploration) setTimeout(() => {
            state.maps.exploration.invalidateSize();
            setSurfaceVisible($('#toggleSurface').checked);
        }, 120);
    }

    function openTarget(id, where = 'exploration', { scrollEvidence = false } = {}) {
        state.exploration.selectedId = id;
        if (where === 'exploration') state.exploration.flyOnLoad = true;
        showSection(where);
        if (where === 'exploration' && state.loaded.exploration) selectTarget(id, { fly: true });
        if (where === 'targets' && state.loaded.targets) selectTarget(id, { fly: false });
        if (scrollEvidence) setTimeout(() => $('#evidencePanel').scrollIntoView({ behavior: 'smooth', block: 'start' }), 350);
    }

    function bindEvents() {
        $$('.nav-link').forEach(link => link.addEventListener('click', e => {
            e.preventDefault();
            showSection(link.dataset.section);
            $('#sidebar').classList.remove('open');
            $('#mobileToggle').setAttribute('aria-expanded', 'false');
        }));
        $('#mobileToggle').addEventListener('click', () => {
            const open = $('#sidebar').classList.toggle('open');
            $('#mobileToggle').setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        window.addEventListener('hashchange', () => {
            const id = ALIASES[window.location.hash.slice(1)] || window.location.hash.slice(1);
            if (SECTIONS[id] && id !== state.currentSection) showSection(id, false);
        });

        $('#btnRetryBackend').addEventListener('click', retryAll);
        $('#btnRefreshSupply').addEventListener('click', loadSupply);
        $('#btnRefreshProduction').addEventListener('click', loadProduction);
        $('#btnRefreshTrust').addEventListener('click', loadTrust);

        const rankClick = e => {
            const b = e.target.closest('[data-target]');
            if (!b) return;
            selectTarget(b.dataset.target, { fly: state.currentSection === 'exploration' });
        };
        $('#targetList').addEventListener('click', rankClick);
        $('#rankList').addEventListener('click', rankClick);

        $('#coordQueryForm').addEventListener('submit', e => {
            e.preventDefault();
            const lat = parseFloat($('#queryLat').value), lon = parseFloat($('#queryLon').value);
            queryCoordinate(lat, lon);
            if (state.maps.exploration && Number.isFinite(lat) && Number.isFinite(lon)) state.maps.exploration.flyTo([lat, lon], Math.max(state.maps.exploration.getZoom(), 9), { duration: 0.8 });
        });

        $('#btnEvaluate').addEventListener('click', () => evaluateRecovery());
        $('#btnFlip').addEventListener('click', runFlip);
        $('#demoSelect').addEventListener('change', e => switchMine(e.target.value));
        $('#btnLogDecision').addEventListener('click', submitDecisionReview);
        $('#truthTabs').addEventListener('click', e => {
            const b = e.target.closest('[data-context]');
            if (!b) return;
            state.trust.context = b.dataset.context;
            renderTruth();
        });

        const bindRange = (id, out, fmt) => {
            const el = $(id);
            const upd = () => { $(out).textContent = fmt(Number(el.value)); };
            el.addEventListener('input', upd);
            upd();
        };
        bindRange('#flipRainfall', '#flipRainfallVal', v => `${v} mm`);
        bindRange('#flipEquip', '#flipEquipVal', v => `${Math.round(v * 100)}%`);
        bindRange('#flipBlast', '#flipBlastVal', v => `${v.toFixed(1)} h/day`);

        // Delegated actions rendered inside panels.
        document.addEventListener('click', e => {
            const retry = e.target.closest('[data-retry]');
            if (retry) {
                const id = retry.dataset.retry;
                state.loaded[id] = false;
                if (SECTIONS[id]) SECTIONS[id].load();
                return;
            }
            const go = e.target.closest('[data-goto]');
            if (go) { showSection(go.dataset.goto); return; }
            const why = e.target.closest('[data-why-target]');
            if (why && why.dataset.whyTarget) { openTarget(why.dataset.whyTarget, 'targets'); return; }
            const open = e.target.closest('[data-open-target]');
            if (open) { openTarget(open.dataset.openTarget, 'exploration'); return; }
            const evd = e.target.closest('[data-evidence-target]');
            if (evd) { openTarget(evd.dataset.evidenceTarget, 'exploration', { scrollEvidence: true }); return; }
            const truth = e.target.closest('[data-truth]');
            if (truth) { openTruth(truth.dataset.truth); return; }
            const sc = e.target.closest('[data-scroll]');
            if (sc) { const t = $(sc.dataset.scroll); if (t) t.scrollIntoView({ behavior: 'smooth', block: 'start' }); return; }
            if (e.target.closest('[data-open-tech]')) {
                const d = $('#truthTech');
                d.open = true;
                d.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    }

    async function retryAll() {
        const btn = $('#btnRetryBackend');
        setLoading(btn, true);
        const ok = await loadHealth();
        setLoading(btn, false);
        state.loaded = {};
        state.exploration.targetsPromise = null;
        showToast(ok || state.health ? 'Backend reachable. Reloading.' : 'Backend still unavailable.', ok || state.health ? 'success' : 'error');
        if (state.currentSection !== 'supply-command') loadSupply();
        SECTIONS[state.currentSection].load();
    }

    document.addEventListener('DOMContentLoaded', async () => {
        bindEvents();
        const fromHash = window.location.hash.slice(1);
        const first = SECTIONS[ALIASES[fromHash] || fromHash] ? (ALIASES[fromHash] || fromHash) : 'supply-command';
        await loadHealth();
        loadDemoStates();
        // The supply state feeds the sidebar workflow status on every screen.
        if (first !== 'supply-command') loadSupply();
        showSection(first);
    });
})();
