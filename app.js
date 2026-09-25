/**
 * GEO-MN // Manganese Supply Command — frontend controller.
 *
 * Rules this file follows:
 *  - Every number, decision, target and metric shown comes from the backend.
 *    Missing values render as "N/A" / "unavailable"; nothing is substituted.
 *  - Decisions and portfolio selection are never computed here.
 *  - Provenance (real / synthetic / simulated / cached) is always displayed.
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
        apiOnline: null,
        health: null,
        supply: null,
        exploration: {
            targets: [],
            targetsMeta: null,
            selectedId: null,
            detail: {},          // id -> detail response
            surface: null,
            surfaceSource: null,
            query: null,
        },
        production: { forecast: null, history: null },
        recovery: null,
        contingency: null,
        decision: { current: null, previous: null, flipRuns: 0 },
        decisionHistory: [],
        trust: { exploration: null, production: null, provenance: null, reconciliation: null },
        loaded: {},              // section -> true once fetched
        maps: { exploration: null, baseLayers: {}, currentBase: null, surfaceLayer: null, markers: {}, highlight: null, queryMarker: null },
    };

    const $ = (sel, root = document) => root.querySelector(sel);
    const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

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

    function fmtT(v) {
        v = num(v);
        if (v === null) return 'N/A';
        const a = Math.abs(v);
        if (a >= 1e6) return `${(v / 1e6).toFixed(2)} Mt`;
        if (a >= 1000) return `${(v / 1000).toFixed(1)} kt`;
        return `${Math.round(v).toLocaleString()} t`;
    }

    function fmtNum(v, dp = 3) {
        const n = num(v);
        if (n === null) return null;
        if (Number.isInteger(n)) return n.toLocaleString();
        return String(Number(n.toFixed(dp)));
    }

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
            toast.style.transform = 'translateX(30px)';
            setTimeout(() => toast.remove(), 250);
        }, 4500);
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
            <div class="error-title">${esc(title)}</div>
            ${detail ? `<div class="error-detail">${esc(detail)}</div>` : ''}
            <div class="error-detail">Try again or check the backend status.</div>
            ${retryId ? `<button type="button" class="cmd-btn cmd-btn-outline" data-retry="${esc(retryId)}">Retry</button>` : ''}
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
        const pill = $('#latencyPill');
        if (!pill) return;
        const shown = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
        pill.textContent = `Latency: ${shown}`;
        pill.title = `Last call: ${endpoint} — ${Math.round(ms)}ms round trip`;
        pill.parentElement?.classList.toggle('pill-slow', ms >= 3000);
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
    const MODE_BADGES = {
        REAL_PUBLIC: ['REAL / PUBLIC', 'real'],
        REAL: ['REAL / PUBLIC', 'real'],
        PUBLIC: ['REAL / PUBLIC', 'real'],
        SYNTHETIC: ['SYNTHETIC DEMONSTRATION DATA', 'synthetic'],
        SYNTHETIC_DEMO: ['SYNTHETIC DEMONSTRATION DATA', 'synthetic'],
        SYNTHETIC_DEMONSTRATION: ['SYNTHETIC DEMONSTRATION DATA', 'synthetic'],
        SIMULATED: ['SIMULATED SCENARIO', 'simulated'],
        SIMULATION: ['SIMULATED SCENARIO', 'simulated'],
        CACHED: ['CACHED / REPLAY', 'cached'],
        CACHE: ['CACHED / REPLAY', 'cached'],
        REPLAY: ['CACHED / REPLAY', 'cached'],
        CACHED_REPLAY: ['CACHED / REPLAY', 'cached'],
        CACHED_GRID: ['CACHED / REPLAY', 'cached'],
        LIVE: ['LIVE COORDINATE QUERY', 'live'],
        LIVE_SATELLITE: ['LIVE COORDINATE QUERY', 'live'],
        LIVE_COORDINATE_QUERY: ['LIVE COORDINATE QUERY', 'live'],
        LIVE_QUERY: ['LIVE COORDINATE QUERY', 'live'],
        UNAVAILABLE: ['UNAVAILABLE', 'unavailable'],
        NONE: ['UNAVAILABLE', 'unavailable'],
    };

    function modeInfo(value) {
        const key = norm(value);
        return MODE_BADGES[key] ? { label: MODE_BADGES[key][0], cls: MODE_BADGES[key][1], known: true }
            : { label: upperHuman(value), cls: 'neutral', known: false };
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

    // Renders a provenance object/string as a row of badges. Only values the
    // backend returned are shown; keys are used as badge prefixes.
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
        if (win) out.push(`<span class="badge badge-window"><span class="badge-k">OBSERVATION WINDOW:</span> ${esc(win)}</span>`);
        return out.length ? out.join('') : badgeHTML('UNAVAILABLE', 'PROVENANCE');
    }

    function dedupeBadges(html) {
        const tmp = document.createElement('div');
        tmp.innerHTML = html;
        const seen = new Set();
        Array.from(tmp.children).forEach(el => { if (seen.has(el.outerHTML)) el.remove(); else seen.add(el.outerHTML); });
        return tmp.innerHTML;
    }

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
                direction: pick(d, 'direction', 'effect'),
            };
        });
    }

    function driverValueText(d) {
        if (d.value === null) return '';
        // A contribution in tonnes is a tonnage, whatever the feature's own unit is.
        if (d.key && /tonnes/.test(d.key)) return `${d.value > 0 ? '+' : ''}${fmtT(d.value)}`;
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
                subsurface: t.subsurface_evidence || (t.subsurface_status ? { status: t.subsurface_status } : undefined),
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
        if (isHigh(t.pros.level) && (t.appl.level === null || !isLow(t.appl.level))) return 'priority';
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

    // ── Reusable component: decision state ──────────────────
    const DECISIONS = {
        OPERATIONAL_RESPONSE: { cls: 'op', title: 'OPERATIONAL RESPONSE' },
        OPERATIONAL_AND_EXPLORATION_CONTINGENCY: { cls: 'contingency', title: 'OPERATIONAL + EXPLORATION CONTINGENCY' },
        REVIEW_REQUIRED: { cls: 'review', title: 'REVIEW REQUIRED' },
    };

    function decisionTitle(ds) {
        if (!ds) return 'UNAVAILABLE';
        return (DECISIONS[norm(ds)] || {}).title || upperHuman(ds);
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

    function renderDecision(el, dec, { compact = false } = {}) {
        if (!el) return;
        if (!dec) {
            el.className = 'decision-block decision-none';
            el.innerHTML = `<div class="decision-label">CURRENT DECISION</div>
                <div class="decision-title">UNAVAILABLE</div>
                <div class="decision-text">No decision has been returned by the backend.</div>`;
            return;
        }
        const key = norm(dec.state);
        const def = DECISIONS[key];
        el.className = `decision-block decision-${def ? def.cls : 'unknown'} decision-enter`;
        const horizonLine = dec.horizon ? `<span class="meta-chip">DECISION HORIZON: ${esc(upperHuman(dec.horizon))}</span>` : '';
        let body = '';
        if (key === 'OPERATIONAL_RESPONSE') {
            body = `<div class="decision-text">Operational recovery actions are the recommended response for this horizon.</div>`;
        } else if (key === 'OPERATIONAL_AND_EXPLORATION_CONTINGENCY') {
            body = `<ol class="transition-steps">
                    <li><span class="step-mark">1</span>OPERATIONAL RECOVERY DOES NOT HOLD THE GAP WITHIN TOLERANCE (TESTED SCENARIOS)</li>
                    <li><span class="step-mark">2</span>RESIDUAL STRATEGIC GAP REMAINS${dec.residual !== null ? ` &middot; expected <b class="mono-val">${esc(fmtT(dec.residual))}</b>` : ''}${dec.worst !== null && dec.worst !== undefined ? ` &middot; worst tested <b class="mono-val">${esc(fmtT(dec.worst))}</b>` : ''}</li>
                    <li><span class="step-mark">3</span>EXPLORATION CONTINGENCY ACTIVATED</li>
                    <li class="step-target"><span class="step-mark">&rarr;</span>NEXT TARGET: <b class="mono-val">${dec.nextTarget ? esc(dec.nextTarget) : 'not returned'}</b></li>
                </ol>`;
        } else if (key === 'REVIEW_REQUIRED') {
            body = `<div class="decision-text">GEO-MN does not have sufficient evidence/applicability to make a responsible automated recommendation.</div>
                <div class="decision-reasons-title">Reasons:</div>
                ${dec.reasons.length
                    ? `<ul class="decision-reasons">${dec.reasons.map(r => `<li>${esc(reasonText(r))}</li>`).join('')}</ul>`
                    : '<div class="decision-text muted">No reasons were returned by the backend.</div>'}`;
        } else {
            body = `<div class="decision-text">Backend returned decision state <span class="mono-val">${esc(dec.state)}</span>.</div>`;
        }
        if (dec.summary && key !== 'REVIEW_REQUIRED') body += `<div class="decision-text muted">${esc(reasonText(dec.summary))}</div>`;
        if (key !== 'REVIEW_REQUIRED' && dec.reasons.length) {
            body += `<ul class="decision-reasons">${dec.reasons.map(r => `<li>${esc(reasonText(r))}</li>`).join('')}</ul>`;
        }
        el.innerHTML = `<div class="decision-head">
                <div>
                    <div class="decision-label">CURRENT DECISION</div>
                    <div class="decision-title"><span class="decision-icon" aria-hidden="true"></span>${esc(decisionTitle(dec.state))}</div>
                </div>
                ${horizonLine}
            </div>
            ${compact ? '' : body}`;
        // Replay the entry animation on every state change.
        el.classList.remove('decision-enter'); void el.offsetWidth; el.classList.add('decision-enter');
    }

    // ── Reusable component: why this target now ─────────────
    function renderWhyNow(el, { targetId, whyTarget, whyNow, reasonCodes, nextEvidence, showWhyTarget = true }) {
        if (!el) return;
        const items = src => asArray(src).flatMap(x => {
            if (x && typeof x === 'object' && !Array.isArray(x) && !pick(x, 'text', 'message', 'explanation', 'detail', 'description', 'code', 'reason_code')) {
                // {reason_codes:[], explanation:"", next_evidence:[]} shaped object
                return [];
            }
            return [x];
        });
        // Object-shaped why_target_now may carry its own codes/explanation/next evidence.
        if (whyNow && !Array.isArray(whyNow) && typeof whyNow === 'object') {
            reasonCodes = reasonCodes || pick(whyNow, 'reason_codes', 'codes');
            nextEvidence = nextEvidence || pick(whyNow, 'next_evidence');
            whyNow = pick(whyNow, 'explanation', 'reasons', 'text', 'items');
        }
        const codes = asArray(reasonCodes).concat(
            asArray(whyNow).filter(x => x && typeof x === 'object').map(x => pick(x, 'code', 'reason_code')).filter(Boolean)
        );
        const uniqCodes = [...new Set(codes.map(String))];
        const list = arr => `<ul class="why-list">${arr.map(x => `<li>${esc(typeof x === 'object' ? (pick(x, 'text', 'message', 'explanation', 'detail', 'description') || pick(x, 'code', 'reason_code') || '') : x)}</li>`).join('')}</ul>`;
        const whyNowItems = items(whyNow);
        const whyTargetItems = items(whyTarget);
        const nextItems = items(nextEvidence);
        el.innerHTML = `
            ${showWhyTarget ? `<div class="why-block">
                <h3 class="why-title">WHY THIS TARGET?</h3>
                ${whyTargetItems.length ? list(whyTargetItems) : '<div class="muted">Not returned by the backend.</div>'}
            </div>` : ''}
            <div class="why-block why-now">
                <h3 class="why-title">WHY THIS TARGET NOW?${targetId ? ` <span class="mono-val why-target-id">${esc(targetId)}</span>` : ''}</h3>
                ${uniqCodes.length ? `<div class="reason-codes">${uniqCodes.map(c => `<span class="reason-code">${esc(c)}</span>`).join('')}</div>` : ''}
                ${whyNowItems.length ? list(whyNowItems) : '<div class="muted">Not returned by the backend.</div>'}
            </div>
            <div class="why-block">
                <h3 class="why-title">NEXT REQUIRED EVIDENCE</h3>
                <div class="muted why-note">Methodological next steps &mdash; none of these has been performed.</div>
                ${nextItems.length ? list(nextItems) : '<div class="muted">Not returned by the backend.</div>'}
            </div>`;
    }

    // ── Driver bars ─────────────────────────────────────────
    function renderDriverBars(el, raw) {
        if (!el) return;
        const drivers = normDrivers(raw);
        if (!drivers.length) {
            el.innerHTML = '<div class="empty-state">No model contributions returned by the backend.</div>';
            return;
        }
        const max = Math.max(...drivers.map(d => Math.abs(d.value ?? 0)), 0);
        el.innerHTML = `<div class="shap-chart">${drivers.map((d, i) => {
            const width = d.value !== null && max > 0 ? Math.max(3, Math.abs(d.value) / max * 100) : 0;
            const valText = driverValueText(d);
            return `<div class="shap-row">
                <div class="shap-label" title="${esc(human(d.name))}">${esc(human(d.name))}</div>
                <div class="shap-track" role="img" aria-label="${esc(human(d.name))}${valText ? ': ' + esc(valText) : ''}">
                    ${d.value !== null ? `<div class="shap-bar ${i === 0 ? 'shap-bar-top' : ''}" style="width:${width}%"></div>` : '<div class="shap-nobar">rank only</div>'}
                </div>
                <div class="shap-value mono-val">${esc(valText)}</div>
            </div>`;
        }).join('')}</div>`;
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
        return { validated, text, window: qv && qv.evaluation_window };
    }

    function riskHTML(risk) {
        if (!risk) return '<span class="muted">N/A</span>';
        const r = norm(risk);
        const cls = ({ LOW: 'low', MEDIUM: 'medium', MODERATE: 'medium', HIGH: 'high', CRITICAL: 'critical' })[r] || 'neutral';
        return `<span class="risk-pill-badge risk-${cls}">${esc(r)}</span>`;
    }

    function riskPolicyText(src) {
        const p = pick(src, 'risk_policy', 'risk_policy_source');
        if (!p) return 'Risk policy: configured policy';
        if (typeof p === 'string') return `Risk policy: ${p}`;
        return `Risk policy: ${pick(p, 'name', 'source', 'label') || 'configured policy'}`;
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
        if (!mode) { setPill('#pillData', 'pending', 'DATA MODE: UNKNOWN'); return; }
        const m = modeInfo(mode);
        setPill('#pillData', m.cls, m.label === 'SYNTHETIC DEMONSTRATION DATA' ? 'SYNTHETIC DEMO DATA' : m.label);
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
            $('#statusDot').className = `status-indicator-dot ${state.apiOnline ? 'online' : 'warn'}`;
            $('#statusText').textContent = state.apiOnline ? 'API ONLINE' : 'API DEGRADED';
        } else {
            setPill('#pillApi', 'bad', 'API OFFLINE');
            setPill('#pillModels', 'bad', 'MODELS UNKNOWN');
            $('#statusDot').className = 'status-indicator-dot offline';
            $('#statusText').textContent = 'BACKEND UNAVAILABLE';
            ['#telApiVersion', '#telEarthEngine', '#telCache'].forEach(s => { $(s).textContent = '—'; });
        }
        updateDataModePill();
        return state.apiOnline;
    }

    // ═══════════════════════════════════════════════════════
    // SCREEN 1 — SUPPLY COMMAND
    // ═══════════════════════════════════════════════════════
    async function loadSupply() {
        const btn = $('#btnRefreshSupply');
        setLoading(btn, true);
        $('#scStatusText').textContent = 'Loading supply outlook…';
        try {
            state.supply = await api.get(`/api/supply-command?mine_id=${encodeURIComponent(MINE_ID)}`);
            renderSupply();
            renderRecoveryBaseline();
            state.loaded['supply-command'] = true;
        } catch (e) {
            state.supply = null;
            $('#scStatus').className = 'supply-status supply-status-error';
            $('#scStatusText').textContent = `Supply outlook unavailable. ${e.userMessage || ''} Try again or use the cached demo state if the backend provides one.`;
            ['#scTarget', '#scP50', '#scP90', '#scGap', '#scExpRecovery', '#scExpResidual', '#scWorstResidual', '#scBestAction', '#scNextTarget'].forEach(s => { $(s).textContent = '—'; });
            $('#scRisk').innerHTML = '<span class="muted">N/A</span>';
            $('#scDrivers').innerHTML = '<div class="empty-state">Unavailable.</div>';
            $('#scProvenance').innerHTML = badgeHTML('UNAVAILABLE', 'DATA');
            renderDecision($('#scDecision'), null);
            $('#scWhyNow').innerHTML = '';
            $('#btnViewTarget').disabled = true;
        } finally {
            setLoading(btn, false);
            updateDataModePill();
        }
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

    function renderSupply() {
        const s = state.supply || {};
        const target = pick(s, 'target_tonnes', 'target');
        const p50 = pick(s, 'p50_tonnes', 'p50');
        const p90 = pick(s, 'p90_tonnes', 'p90');
        const gap = pick(s, 'gap_p50_tonnes', 'gap_tonnes', 'supply_gap_tonnes');
        const risk = pick(s, 'risk_state', 'risk');

        $('#scHorizon').textContent = `FORECAST: NEXT 7 DAYS · DECISION HORIZON: ${s.decision_horizon ? upperHuman(s.decision_horizon) : 'N/A'}`;
        $('#scProvenance').innerHTML = provenanceHTML(s.provenance);
        $('#scTarget').textContent = fmtT(target);
        $('#scP50').textContent = fmtT(p50);
        const p10 = pick(s, 'p10_tonnes', 'p10');
        const iv = intervalInfo(s);
        $('#scP90').textContent = (num(p10) !== null && num(p90) !== null) ? `${fmtT(p10)} – ${fmtT(p90)}` : 'N/A';
        $('#scIntervalNote').textContent = iv.validated ? 'Prediction interval (validated)' : 'NOT VALIDATED — indicative only';
        $('#scIntervalNote').classList.toggle('kpi-sub-warn', !iv.validated);
        $('#scForecastBasis').innerHTML = `<b>Forecast basis:</b> ${esc(s.forecast_basis || 'Not returned by the backend.')} <span class="muted">${esc(iv.text)}${iv.window ? ` (${esc(iv.window)})` : ''}.</span>`;
        $('#scGap').textContent = fmtT(gap);
        $('#scRisk').innerHTML = riskHTML(risk);
        $('#scRiskPolicy').textContent = riskPolicyText(s);

        const statusEl = $('#scStatus');
        const g = num(gap);
        if (g === null) {
            statusEl.className = 'supply-status';
            $('#scStatusText').textContent = 'Supply gap not returned by the backend.';
        } else if (g > 0) {
            statusEl.className = `supply-status supply-status-miss risk-${norm(risk).toLowerCase()}`;
            $('#scStatusText').innerHTML = `P50 forecast is <b class="mono-val">${esc(fmtT(g))}</b> below target${risk ? ` &middot; risk ${esc(norm(risk))}` : ''}.`;
        } else {
            statusEl.className = 'supply-status supply-status-ok';
            $('#scStatusText').innerHTML = `P50 forecast meets target (gap <b class="mono-val">${esc(fmtT(g))}</b>).`;
        }

        renderDriverBars($('#scDrivers'), s.primary_drivers);

        const withheld = s.selection_status === 'REVIEW_REQUIRED';
        $('#scBestAction').textContent = withheld ? 'AUTOMATED SELECTION WITHHELD' : (actionName(s.best_operational_action) || 'Not returned');
        $('#scSelectionWhy').textContent = [s.selection_explanation, s.action_note].filter(Boolean).join(' ');
        $('#scExpRecovery').textContent = fmtT(s.expected_recovery_tonnes);
        $('#scExpResidual').textContent = fmtT(s.expected_residual_gap_tonnes);
        $('#scWorstResidual').textContent = fmtT(pick(s, 'worst_case_residual_gap_tonnes', 'worst_tested_residual_gap_tonnes'));

        const dec = normDecision(s);
        if (dec && dec.residual === null) dec.residual = num(s.expected_residual_gap_tonnes);
        renderDecision($('#scDecision'), dec);
        if (dec && !state.decision.current) state.decision.current = dec;

        renderSupplyFlip(s.decision_flip);

        const next = pick(s, 'next_target', 'next_target_id');
        $('#scNextTarget').textContent = next || 'None';
        $('#btnViewTarget').disabled = !next;
        $('#scNextTargetPanel').classList.toggle('is-empty', !next);
        if (next) {
            renderWhyNow($('#scWhyNow'), {
                targetId: next,
                whyNow: s.why_target_now,
                reasonCodes: undefined,
                nextEvidence: s.next_evidence,
                whyTarget: s.why_target,
                showWhyTarget: !!s.why_target,
            });
        } else {
            $('#scWhyNow').innerHTML = '<div class="muted">No exploration target was nominated by the backend for this decision.</div>';
        }
    }

    // ── Decision flip (backend /api/decision/flip result) ───
    function flipSideHTML(label, d) {
        if (!d) return `<div class="flip-state decision-none"><span class="kpi-label">${esc(label)}</span><div class="flip-title">NOT RETURNED</div></div>`;
        const cls = (DECISIONS[norm(d.decision_state)] || {}).cls || 'unknown';
        const port = d.selection_status === 'REVIEW_REQUIRED' ? 'withheld (review)' : (d.selected_portfolio || '—');
        return `<div class="flip-state decision-${cls}">
            <span class="kpi-label">${esc(label)}</span>
            <div class="flip-title">${esc(decisionTitle(d.decision_state))}</div>
            <div class="flip-sub">Portfolio: <b class="mono-val">${esc(port)}</b></div>
            <div class="flip-sub">Next target: <b class="mono-val">${esc(d.next_target || 'none')}</b></div>
            <div class="flip-sub">Residual (expected / worst): <b class="mono-val">${esc(fmtT(d.expected_residual_gap_tonnes))} / ${esc(fmtT(d.worst_case_residual_gap_tonnes))}</b></div>
            ${asArray(d.review_reasons).length ? `<div class="flip-sub muted">${asArray(d.review_reasons).map(r => esc(reasonText(r))).join('<br>')}</div>` : ''}
        </div>`;
    }

    function flipResultHTML(f) {
        if (!f) return '<div class="empty-state">No decision flip returned by the backend.</div>';
        const changed = asArray(f.changed_inputs);
        const flipped = f.flipped === true;
        return `
            <div class="flip-status ${flipped ? 'flip-changed' : 'flip-same'}" role="status">FLIPPED? ${flipped ? 'YES' : 'NO'} &middot; <span class="mono-val">${esc(f.transition || '')}</span></div>
            <div class="flip-states">
                ${flipSideHTML('BASELINE', f.baseline)}
                <div class="flip-arrow" aria-hidden="true">&rarr;</div>
                ${flipSideHTML('PERTURBED', f.perturbed)}
            </div>
            <div class="flip-changed-inputs"><span class="kpi-label">CHANGED INPUTS</span>
                ${changed.length ? `<table class="enterprise-table"><thead><tr><th>Input</th><th>Baseline</th><th>Perturbed</th></tr></thead><tbody>${changed.map(c => `<tr><td>${esc(human(c.input))}</td><td class="mono-val">${esc(c.baseline ?? '—')}</td><td class="mono-val">${esc(c.perturbed ?? '—')}</td></tr>`).join('')}</tbody></table>` : '<div class="muted">No inputs changed.</div>'}
            </div>
            <div class="chart-note">Both decisions were recomputed end-to-end by the backend. ${badgeHTML('SIMULATED')}</div>`;
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
        $('#telMine').textContent = id;
        // Drop every mine-dependent result so nothing stale is shown for the new state.
        state.supply = null;
        state.recovery = null;
        state.contingency = null;
        state.decision = { current: null, previous: null, flipRuns: 0 };
        state.exploration.detail = {};
        state.exploration.selectedId = null;
        state.production = { forecast: null, history: null };
        state.flipPreset = false;
        state.loaded = {};
        $('#flipResult').innerHTML = '<div class="empty-state">Adjust the conditions and run the flip to compare the baseline and perturbed decisions.</div>';
        $('#targetCard').innerHTML = '<div class="empty-state">Select an exploration target on the map or in the list.</div>';
        showToast(`Demo state ${id} selected — results recomputed by the backend (synthetic data).`, 'info');
        if (state.currentSection !== 'supply-command') loadSupply();
        SECTIONS[state.currentSection].load();
    }

    // ═══════════════════════════════════════════════════════
    // SCREEN 2 — EXPLORATION
    // ═══════════════════════════════════════════════════════
    function initExplorationMap() {
        if (state.maps.exploration || !window.L) {
            if (!window.L) $('#explorationMap').innerHTML = '<div class="error-state">Map library failed to load (external CDN). Targets, ranks and decisions below remain available.</div>';
            return;
        }
        const map = L.map('explorationMap', { center: [21.6, 79.9], zoom: 8, minZoom: 4, maxZoom: 16, zoomControl: true });
        state.maps.exploration = map;

        const satTile = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { attribution: 'Esri, USGS, AeroGRID, IGN', maxZoom: 18 });
        const satLabels = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', { attribution: 'CARTO', maxZoom: 18, subdomains: 'abcd' });
        const lstTile = L.tileLayer('https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_Land_Surface_Temp_Day/default/2024-05-01/GoogleMapsCompatible_Level7/{z}/{y}/{x}.png', { attribution: 'NASA GIBS — MODIS LST 2024-05-01 (context layer, not the model feature)', maxNativeZoom: 7, maxZoom: 18, opacity: 0.9 });
        const streetTile = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: 'OpenStreetMap', maxZoom: 18 });

        state.maps.baseLayers = {
            satellite: L.layerGroup([satTile, satLabels]),
            thermal: L.layerGroup([satTile, lstTile, satLabels]),
            street: streetTile,
        };
        state.maps.currentBase = state.maps.baseLayers.satellite.addTo(map);

        // External tile failures are visual only: warn, never block or imply a model failure.
        const tileWarn = name => () => mapNotice(`${name} tiles could not be loaded. The map context layer is degraded; targets, ranks and decisions are unaffected.`);
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

        map.on('mousemove', e => { $('#cursorCoordReadout').textContent = `Cursor: ${e.latlng.lat.toFixed(4)}°N, ${e.latlng.lng.toFixed(4)}°E`; });
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
        const list = $('#targetList');
        list.innerHTML = loadingHTML('Loading exploration targets…');
        try {
            const resp = await api.get(`/api/exploration/targets?mine_id=${encodeURIComponent(MINE_ID)}`);
            const rawList = Array.isArray(resp) ? resp : asArray(pick(resp, 'targets', 'items', 'results'));
            state.exploration.targetsMeta = Array.isArray(resp) ? null : resp;
            state.exploration.targets = rawList.map(normTarget).filter(t => t && t.id);
            state.loaded.exploration = true;
            $('#exProvenance').innerHTML = provenanceHTML(state.exploration.targetsMeta && state.exploration.targetsMeta.provenance);
            renderTargetList();
            renderTargetMarkers();
            await loadSurface();
            fitTargets();
            if (state.exploration.selectedId) selectTarget(state.exploration.selectedId, { fly: true });
        } catch (e) {
            state.exploration.targets = [];
            list.innerHTML = errorHTML('Exploration targets unavailable.', e, 'exploration');
            $('#targetCount').textContent = '—';
            $('#exProvenance').innerHTML = badgeHTML('UNAVAILABLE', 'DATA');
            $('#surfaceSourceLabel').textContent = 'Surface: unavailable';
        }
        // The map may have been created while its section was hidden; size it, then fit.
        setTimeout(() => {
            if (!state.maps.exploration) return;
            state.maps.exploration.invalidateSize();
            if (!state.exploration.selectedId) fitTargets();
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
            $('#surfaceSourceLabel').textContent = !pts ? 'Surface: unavailable from backend' : 'Surface: heat-map plugin unavailable';
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
        $('#surfaceSourceLabel').innerHTML = `Surface: cached ~1 km rank grid ${badgeHTML(prov ? prov.data_mode : 'CACHED')}`;
        setSurfaceVisible($('#toggleSurface').checked);
    }

    function setSurfaceVisible(v) {
        const map = state.maps.exploration, layer = state.maps.surfaceLayer;
        if (!map || !layer) return;
        if (v && !map.hasLayer(layer)) layer.addTo(map);
        if (!v && map.hasLayer(layer)) map.removeLayer(layer);
    }

    function levelText(l) {
        if (!l) return 'N/A';
        if (l.level && l.score !== null) return `${l.level} / ${fmtNum(l.score, 0)}`;
        if (l.level) return l.level;
        if (l.score !== null) return fmtNum(l.score, 0);
        return 'N/A';
    }

    function renderTargetList() {
        const list = $('#targetList');
        const ts = state.exploration.targets;
        $('#targetCount').textContent = `${ts.length} target${ts.length === 1 ? '' : 's'}`;
        if (!ts.length) { list.innerHTML = '<div class="empty-state">The backend returned no exploration targets.</div>'; return; }
        list.innerHTML = ts.map(t => {
            const style = targetStyle(t);
            return `<button type="button" class="target-item target-${style}${t.id === state.exploration.selectedId ? ' selected' : ''}" data-target="${esc(t.id)}" role="option" aria-selected="${t.id === state.exploration.selectedId}">
                <span class="tmk tmk-${style}${isHigh(t.unc.level) ? ' tmk-uncertain' : ''}" aria-hidden="true">${style === 'caution' ? '!' : 'T'}</span>
                <span class="ti-main">
                    <span class="ti-id mono-val">${esc(t.id)}${t.name ? ` <span class="ti-name">${esc(t.name)}</span>` : ''}</span>
                    <span class="ti-sub">Prospectivity ${esc(levelText(t.pros))} &middot; Applicability ${esc(t.appl.level || 'N/A')} &middot; Uncertainty ${esc(t.unc.level || 'N/A')}</span>
                </span>
                ${style === 'caution' ? '<span class="ti-flag">CAUTION</span>' : ''}
                ${t.priority !== null ? `<span class="ti-priority mono-val" title="Exploration priority">${esc(fmtNum(t.priority, 0))}</span>` : ''}
            </button>`;
        }).join('');
    }

    function popupHTML(t) {
        return `<div class="map-popup">
            <div class="mp-title">TARGET ${esc(t.id)}</div>
            <div>RELATIVE PROSPECTIVITY: <b>${esc(levelText(t.pros))}</b></div>
            <div>APPLICABILITY: <b>${esc(t.appl.level || 'N/A')}</b></div>
            <div>UNCERTAINTY: <b>${esc(t.unc.level || 'N/A')}</b></div>
            <div>STATUS: <b>${esc(statusText(t))}</b></div>
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
                html: `<span class="tmk tmk-${style}${isHigh(t.unc.level) ? ' tmk-uncertain' : ''} tmk-map">${esc(t.id)}</span>`,
                iconSize: null,
            });
            const m = L.marker([t.lat, t.lon], { icon, title: `Target ${t.id}`, alt: `Target ${t.id}`, keyboard: true, riseOnHover: true })
                .bindPopup(popupHTML(t))
                .on('click', () => selectTarget(t.id, { fly: false }));
            m.addTo(map);
            state.maps.markers[t.id] = m;
        });
    }

    function fitTargets() {
        const map = state.maps.exploration;
        const pts = state.exploration.targets.filter(t => t.lat !== null && t.lon !== null).map(t => [t.lat, t.lon]);
        // Padding keeps markers clear of the toolbar (top) and legend (bottom-left).
        if (map && pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.15), { maxZoom: 10, paddingTopLeft: [60, 70], paddingBottomRight: [40, 220] });
    }

    function highlightRegion(t) {
        const map = state.maps.exploration;
        if (!map) return;
        if (state.maps.highlight) { map.removeLayer(state.maps.highlight); state.maps.highlight = null; }
        const style = targetStyle(t);
        const color = style === 'caution' ? '#f59e0b' : (style === 'priority' ? '#10b981' : '#06b6d4');
        const opts = { color, weight: 2, fillColor: color, fillOpacity: 0.12, dashArray: isHigh(t.unc.level) ? '6 6' : null, className: 'target-highlight-shape' };
        if (t.geometry && typeof t.geometry === 'object') {
            try { state.maps.highlight = L.geoJSON(t.geometry, { style: () => opts }).addTo(map); } catch (_) { state.maps.highlight = null; }
        }
        if (!state.maps.highlight && t.lat !== null && t.lon !== null) {
            // Only draw an extent the backend supplied; otherwise a focus ring of fixed screen size.
            state.maps.highlight = t.radiusKm !== null
                ? L.circle([t.lat, t.lon], { ...opts, radius: t.radiusKm * 1000 }).addTo(map)
                : L.circleMarker([t.lat, t.lon], { ...opts, radius: 26, fillOpacity: 0.08 }).addTo(map);
        }
        $$('.tmk-map').forEach(el => el.classList.toggle('tmk-selected', el.textContent === t.id));
    }

    async function selectTarget(id, { fly = true } = {}) {
        state.exploration.selectedId = id;
        $$('.target-item').forEach(el => {
            const sel = el.dataset.target === id;
            el.classList.toggle('selected', sel);
            el.setAttribute('aria-selected', sel ? 'true' : 'false');
        });
        let t = state.exploration.targets.find(x => x.id === id);
        const card = $('#targetCard');
        if (t) {
            highlightRegion(t);
            const map = state.maps.exploration;
            if (fly && map && t.lat !== null && t.lon !== null) map.flyTo([t.lat, t.lon], Math.max(map.getZoom(), 10), { duration: 0.9 });
            renderTargetCard(t, { loadingDetail: !state.exploration.detail[id] });
        } else {
            card.innerHTML = loadingHTML(`Loading target ${id}…`);
        }
        if (!state.exploration.detail[id]) {
            try {
                const d = await api.get(`/api/exploration/targets/${encodeURIComponent(id)}?mine_id=${encodeURIComponent(MINE_ID)}`);
                state.exploration.detail[id] = d;
            } catch (e) {
                if (state.exploration.selectedId !== id) return;
                if (!t) { card.innerHTML = errorHTML(`Target ${id} unavailable.`, e); return; }
                renderTargetCard(t, { detailError: e });
                return;
            }
        }
        if (state.exploration.selectedId !== id) return;
        const detail = state.exploration.detail[id];
        const merged = normTarget({ ...(t ? t.raw : {}), ...(detail && detail.target ? detail.target : detail) });
        if (merged) {
            if (!t && merged.lat !== null) {
                highlightRegion(merged);
                const map = state.maps.exploration;
                if (fly && map) map.flyTo([merged.lat, merged.lon], Math.max(map.getZoom(), 10), { duration: 0.9 });
            }
            renderTargetCard(merged, {});
        }
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

    // Backend evidence levels are L0..L4; the ladder index equals the level.
    const LADDER = ['REMOTE-SENSING INDICATION', 'REMOTE + GEOLOGICAL CONTEXT', 'GROUND / GEOPHYSICAL / GEOCHEMICAL', 'DRILLING / ASSAY', 'RESOURCE / RESERVE WORK'];

    function ladderHTML(level) {
        return `<ol class="evidence-ladder" aria-label="Evidence maturity ladder">
            ${LADDER.map((name, i) => {
                const n = i;
                const cls = level === null ? '' : (n < level ? 'done' : (n === level ? 'current' : 'todo'));
                return `<li class="ladder-step ${cls}"${n === level ? ' aria-current="step"' : ''}>
                    <span class="ladder-n">L${n}</span><span class="ladder-name">${name}</span>${n === level ? '<span class="ladder-here">CURRENT</span>' : ''}
                </li>`;
            }).join('')}
        </ol>`;
    }

    function renderTargetCard(t, { loadingDetail = false, detailError = null } = {}) {
        const card = $('#targetCard');
        const style = targetStyle(t);
        const lowAppl = style === 'caution';
        const level = maturityLevel(t.maturity);
        const ev = evidenceRows(t);
        const subsurface = ev.find(e => e.key === 'subsurface');
        const src = t.source;
        const warnings = t.warnings.filter(Boolean).map(reasonText);
        const cautionBlock = lowAppl ? `<div class="caution-block" role="alert">
                <div class="caution-title">${esc(t.pros.level || 'N/A')} PROSPECTIVITY &middot; ${esc(t.appl.level || 'LOW')} MODEL APPLICABILITY</div>
                <div>Caution: prediction is outside the model's supported feature domain.${t.appl.note ? ' ' + esc(reasonText(t.appl.note)) : ''}</div>
            </div>` : '';
        const warnBlock = warnings.length ? `<div class="caution-block caution-soft">${warnings.map(w => `<div>${esc(w)}</div>`).join('')}</div>` : '';

        card.className = `panel target-card target-card-${style} focus-in`;
        card.innerHTML = `
            <div class="tc-head">
                <div>
                    <div class="kpi-label">TARGET</div>
                    <div class="tc-id mono-val">${esc(t.id)}${t.name ? ` <span class="ti-name">${esc(t.name)}</span>` : ''}</div>
                </div>
                <div class="prov-row">
                    ${src ? badgeHTML(src, 'SOURCE') : '<span class="badge badge-neutral"><span class="badge-k">SOURCE:</span> NOT REPORTED</span>'}
                    ${t.window ? `<span class="badge badge-window"><span class="badge-k">OBSERVATION WINDOW:</span> ${esc(t.window)}</span>` : ''}
                    ${t.provenance ? provenanceHTML(t.provenance) : ''}
                    ${loadingDetail ? '<span class="badge badge-neutral">LOADING DETAIL…</span>' : ''}
                    ${detailError ? `<span class="badge badge-unavailable">DETAIL UNAVAILABLE</span>` : ''}
                </div>
            </div>
            ${cautionBlock}${warnBlock}
            <div class="tc-grid">
                <div class="tc-cell ${lowAppl ? 'tc-muted-high' : ''}"><span class="kpi-label">RELATIVE PROSPECTIVITY</span><div class="tc-val">${esc(levelText(t.pros))}</div></div>
                <div class="tc-cell ${lowAppl ? 'tc-warn' : ''}"><span class="kpi-label">APPLICABILITY</span><div class="tc-val">${esc(t.appl.level || (t.inDomain === false ? 'OUT OF DOMAIN' : 'N/A'))}</div></div>
                <div class="tc-cell"><span class="kpi-label">UNCERTAINTY</span><div class="tc-val">${esc(t.unc.level || (t.unc.score !== null ? fmtNum(t.unc.score, 2) : 'N/A'))}</div></div>
                <div class="tc-cell"><span class="kpi-label">EVIDENCE MATURITY</span><div class="tc-val">${level !== null ? `LEVEL ${level}` : 'N/A'}</div></div>
                <div class="tc-cell"><span class="kpi-label">STRATEGIC RELEVANCE</span><div class="tc-val">${esc(t.strategic ? upperHuman(typeof t.strategic === 'object' ? pick(t.strategic, 'level', 'label') : t.strategic) : 'N/A')}</div></div>
                <div class="tc-cell"><span class="kpi-label">EXPLORATION PRIORITY</span><div class="tc-val mono-val">${t.priority !== null ? esc(fmtNum(t.priority, 0)) : 'N/A'}</div></div>
                <div class="tc-cell"><span class="kpi-label">GEOLOGICAL CONTEXT</span><div class="tc-val tc-val-sm">${esc(t.geology && pick(t.geology, 'unit_name') ? pick(t.geology, 'unit_name') : 'N/A')}</div></div>
                <div class="tc-cell"><span class="kpi-label">SUBSURFACE EVIDENCE</span><div class="tc-val">${esc(subsurface && subsurface.ok === true ? 'AVAILABLE' : 'UNAVAILABLE')}</div></div>
                <div class="tc-cell"><span class="kpi-label">DISTANCE TO SUPPLY POINT</span><div class="tc-val mono-val">${t.distanceKm !== null ? esc(fmtNum(t.distanceKm, 1)) + ' km' : 'N/A'}</div></div>
                <div class="tc-cell tc-cell-wide"><span class="kpi-label">STATUS</span><div class="tc-val">${esc(statusText(t))}${t.context ? ` &middot; ${esc(t.context)}` : ''}</div></div>
            </div>
            <div class="tc-split">
                <div>
                    <h3 class="why-title">EVIDENCE</h3>
                    <ul class="evidence-list">
                        ${ev.map(e => `<li class="ev-${e.ok === true ? 'yes' : (e.ok === false ? 'no' : 'unk')}">
                            <span class="ev-mark" aria-hidden="true">${e.ok === true ? '✓' : (e.ok === false ? '✕' : '?')}</span>
                            <span>${e.label}</span>
                            <span class="ev-state">${e.ok === true ? 'AVAILABLE' : (e.ok === false ? 'UNAVAILABLE' : 'NOT REPORTED')}</span>
                        </li>`).join('')}
                    </ul>
                    ${subsurface && subsurface.ok !== true ? `<div class="subsurface-note"><b>SUBSURFACE EVIDENCE UNAVAILABLE</b><br>Field mapping, geochemistry, geophysics and drilling/assay are required before any reserve or resource conclusion.</div>` : ''}
                </div>
                <div>
                    <h3 class="why-title">EVIDENCE LADDER</h3>
                    ${ladderHTML(level)}
                </div>
            </div>
            <div class="why-grid" id="tcWhy"></div>`;
        renderWhyNow($('#tcWhy'), {
            targetId: t.id,
            whyTarget: t.whyTarget,
            whyNow: t.whyNow,
            reasonCodes: t.reasonCodes,
            nextEvidence: t.nextEvidence,
        });
    }

    async function queryCoordinate(lat, lon) {
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) { showToast('Enter a valid latitude and longitude.', 'error'); return; }
        const btn = $('#btnQueryCoord');
        setLoading(btn, true);
        const map = state.maps.exploration;
        try {
            const d = await api.post('/api/exploration/predict', { lat, lon }, { timeout: 45000 });
            state.exploration.query = d;
            const t = normTarget({ ...d, target_id: pick(d, 'target_id') || `QUERY ${lat.toFixed(3)}, ${lon.toFixed(3)}`, lat: pick(d, 'lat', 'query_lat') ?? lat, lon: pick(d, 'lon', 'query_lon') ?? lon });
            const QUERY_STATUS = {
                EXPLORATION_TARGET: 'within the high-rank (target-threshold) range',
                NOT_PRIORITISED: 'below the target threshold',
                REVIEW_REQUIRED: 'high rank but low applicability — review required',
            };
            t.status = `COORDINATE QUERY (NOT A CLUSTERED TARGET) · ${(QUERY_STATUS[norm(pick(d, 'status'))] || upperHuman(pick(d, 'status') || 'N/A')).toUpperCase()}`;
            if (d.fallback_used) {
                t.warnings = t.warnings.concat([`Live query not used (${d.live_status || 'unavailable'}). Cached grid cell ${fmtNum(d.fallback_distance_km, 2)} km from the query (maximum supported ${fmtNum(d.max_supported_fallback_distance_km, 1)} km).`]);
            } else if (d.source_mode) {
                t.warnings = t.warnings.concat([`Live coordinate query from the fixed ${d.observation_window || '2024'} reference composite (not real-time).`]);
            }
            state.exploration.selectedId = null;
            $$('.target-item').forEach(el => { el.classList.remove('selected'); el.setAttribute('aria-selected', 'false'); });
            if (map) {
                if (state.maps.queryMarker) map.removeLayer(state.maps.queryMarker);
                state.maps.queryMarker = L.circleMarker([lat, lon], { radius: 7, color: '#fff', weight: 2, fillColor: '#8b5cf6', fillOpacity: 1 })
                    .bindPopup(popupHTML(t)).addTo(map).openPopup();
                highlightRegion(t);
            }
            renderTargetCard(t, {});
        } catch (e) {
            showToast(`Coordinate query failed. ${e.userMessage || ''}`, 'error');
        } finally {
            setLoading(btn, false);
        }
    }

    // ═══════════════════════════════════════════════════════
    // SCREEN 3 — PRODUCTION RISK
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
        const [hist, fc] = await Promise.allSettled([
            api.get(`/api/production/history?mine_id=${encodeURIComponent(MINE_ID)}&days=90`),
            api.post('/api/production/forecast', { mine_id: MINE_ID }),
        ]);
        state.production.history = hist.status === 'fulfilled' ? hist.value : null;
        state.production.forecast = fc.status === 'fulfilled' ? fc.value : null;
        state.loaded['production-risk'] = hist.status === 'fulfilled' || fc.status === 'fulfilled';

        const f = normForecast(state.production.forecast);
        const provs = [];
        if (state.production.history && state.production.history.provenance) provs.push(provenanceHTML(state.production.history.provenance));
        if (f && f.provenance) provs.push(provenanceHTML(f.provenance));
        $('#prProvenance').innerHTML = provs.length ? dedupeBadges(provs.join('')) : badgeHTML('UNAVAILABLE', 'DATA');

        if (fc.status === 'fulfilled') renderForecast(f);
        else {
            ['#prP10', '#prP50', '#prP90', '#prTarget', '#prGap'].forEach(s => { $(s).textContent = '—'; });
            $('#prRisk').innerHTML = '<span class="muted">N/A</span>';
            $('#prRangeChart').innerHTML = '';
            $('#prContributions').innerHTML = errorHTML('Production forecast unavailable.', fc.reason, 'production');
        }
        if (hist.status === 'fulfilled') renderHistoryChart(normHistory(state.production.history), f);
        else $('#histChart').innerHTML = errorHTML('Production history unavailable.', hist.reason, 'production');
        setLoading(btn, false);
    }

    function renderForecast(f) {
        $('#prHorizon').textContent = `FORECAST HORIZON: ${f.horizon ? upperHuman(f.horizon) : 'N/A'}`;
        $('#prP10').textContent = fmtT(f.p10);
        $('#prP50').textContent = fmtT(f.p50);
        $('#prP90').textContent = fmtT(f.p90);
        $('#prTarget').textContent = fmtT(f.target);
        $('#prGap').textContent = fmtT(f.gap);
        $('#prRisk').innerHTML = riskHTML(f.risk);
        $('#prRiskPolicy').textContent = riskPolicyText(f.raw);
        const hasBand = f.p10 !== null && f.p90 !== null;
        const iv = intervalInfo(f.raw);
        $('#prForecastSub').textContent = hasBand
            ? (iv.validated ? 'P10 / P50 / P90 — validated interval' : 'P10 / P50 / P90 — P10–P90 interval NOT validated')
            : 'Only the forecast fields returned by the backend are shown';
        $('#prForecastBasis').innerHTML = `<b>Forecast basis:</b> ${esc(pick(f.raw, 'forecast_basis') || 'Not returned by the backend.')}
            ${f.raw.scenario_override_applied ? badgeHTML('SIMULATED', 'INPUTS') : ''}
            <div class="muted">${esc(iv.text)}${iv.window ? ` · evaluation window ${esc(iv.window)}` : ''}. Period ${esc(pick(f.raw, 'period_start') || '?')} → ${esc(pick(f.raw, 'period_end') || '?')}.</div>`;
        renderRangeChart($('#prRangeChart'), f);
        const cm = f.contributionMeta;
        if (cm && cm.status === 'UNAVAILABLE') {
            $('#prContributions').innerHTML = '<div class="empty-state">Model contributions unavailable for this forecast.</div>';
        } else {
            renderDriverBars($('#prContributions'), f.contributions);
            if (cm && cm.status === 'AVAILABLE') {
                $('#prContributions').insertAdjacentHTML('beforeend', `<div class="chart-note">Relative contribution of each input to the P50 model output (tonnes), measured from a model base value of ${esc(fmtT(cm.base_value_tonnes))}; calibration adjustment ${esc(fmtT(cm.calibration_adjustment_tonnes))}. ${esc(cm.method || '')}</div>`);
            }
        }
    }

    // Horizontal range: P10–P90 band, P50 tick, target line. One axis, labelled.
    function renderRangeChart(el, f) {
        const vals = [f.p10, f.p50, f.p90, f.target].filter(v => v !== null);
        if (vals.length < 2 || f.p50 === null) { el.innerHTML = ''; return; }
        const lo = Math.min(...vals), hi = Math.max(...vals);
        const pad = (hi - lo) * 0.12 || hi * 0.05 || 1;
        const min = lo - pad, max = hi + pad;
        const x = v => ((v - min) / (max - min)) * 100;
        const marks = [];
        if (f.p10 !== null && f.p90 !== null) marks.push(`<div class="rc-band" style="left:${x(f.p10)}%;width:${x(f.p90) - x(f.p10)}%" title="P10–P90: ${esc(fmtT(f.p10))} – ${esc(fmtT(f.p90))}"></div>`);
        marks.push(`<div class="rc-p50" style="left:${x(f.p50)}%" title="P50 ${esc(fmtT(f.p50))}"><span>P50 ${esc(fmtT(f.p50))}</span></div>`);
        if (f.target !== null) marks.push(`<div class="rc-target" style="left:${x(f.target)}%" title="Target ${esc(fmtT(f.target))}"><span>TARGET ${esc(fmtT(f.target))}</span></div>`);
        el.innerHTML = `<div class="rc-track" role="img" aria-label="Forecast range P10 ${esc(fmtT(f.p10))}, P50 ${esc(fmtT(f.p50))}, P90 ${esc(fmtT(f.p90))}, target ${esc(fmtT(f.target))}">${marks.join('')}</div>
            <div class="rc-axis"><span>${esc(fmtT(min))}</span><span>${esc(fmtT(max))}</span></div>
            <div class="rc-legend">${f.p10 !== null && f.p90 !== null ? `<span><i class="lg-band"></i>${esc(intervalInfo(f.raw).text)}</span>` : ''}<span><i class="lg-p50"></i>P50 model median</span>${f.target !== null ? '<span><i class="lg-target"></i>Target (7-day plan)</span>' : ''}</div>`;
    }

    const C_ACTUAL = '#0891b2', C_FORECAST = '#d97706', C_TARGET = '#94a3b8';

    function renderHistoryChart(rows, f) {
        const el = $('#histChart');
        const legend = $('#histLegend');
        if (!rows.length) {
            el.innerHTML = '<div class="empty-state">No production history returned by the backend.</div>';
            legend.innerHTML = '';
            return;
        }
        // Forecast overlays the history only if it is on the same scale (a
        // per-period series). A period-total forecast is shown in the range
        // panel instead — plotting it on a daily axis would be misleading.
        const fSeries = f ? f.series.map(p => ({
            date: pick(p, 'date', 'period'),
            p10: num(pick(p, 'p10_tonnes', 'p10')), p50: num(pick(p, 'p50_tonnes', 'p50')), p90: num(pick(p, 'p90_tonnes', 'p90')),
        })).filter(p => p.date && p.p50 !== null) : [];
        const lastActual = [...rows].reverse().find(r => r.actual !== null);
        let fPoints = fSeries;
        let fNote = '';
        if (!fPoints.length && f && f.p50 !== null && lastActual) {
            const ratio = f.p50 / (lastActual.actual || 1);
            if (ratio > 0.2 && ratio < 5) {
                fPoints = [{ date: f.date || 'Forecast', p10: f.p10, p50: f.p50, p90: f.p90 }];
            } else {
                fNote = `Forecast is a ${f.horizon ? human(f.horizon) : 'period'} total on a different scale from the history — see the Forecast panel.`;
            }
        }

        const all = rows.map(r => ({ ...r, kind: 'hist' })).concat(fPoints.map(p => ({ ...p, kind: 'fc' })));
        const vals = all.flatMap(r => [r.actual, r.target, r.p10, r.p50, r.p90]).filter(v => v !== null && v !== undefined);
        if (!vals.length) { el.innerHTML = '<div class="empty-state">History contains no numeric values.</div>'; return; }
        let yMin = Math.min(...vals), yMax = Math.max(...vals);
        const pad = (yMax - yMin) * 0.1 || yMax * 0.1 || 1;
        yMin = Math.max(0, yMin - pad); yMax += pad;

        const W = 1000, H = 300, m = { l: 64, r: 20, t: 14, b: 36 };
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
        const labelEvery = Math.max(1, Math.ceil(n / 8));
        const xLabels = all.map((r, i) => ((i % labelEvery === 0 && (n - 1 - i >= labelEvery / 2 || i === 0)) || i === n - 1)
            ? `<text class="ch-tick" x="${xs(i)}" y="${H - 12}" text-anchor="middle">${esc(String(r.date).slice(0, 10))}</text>` : '').join('');

        const fcStart = all.findIndex(r => r.kind === 'fc');
        let band = '';
        if (fcStart >= 0) {
            const fc = all.map((r, i) => ({ r, i })).filter(o => o.r.kind === 'fc' && o.r.p10 !== null && o.r.p90 !== null);
            if (fc.length === 1) {
                const { r, i } = fc[0];
                band = `<rect class="ch-band" x="${xs(i) - 10}" y="${ys(r.p90)}" width="20" height="${Math.max(1, ys(r.p10) - ys(r.p90))}" rx="4"></rect>`;
            } else if (fc.length > 1) {
                const top = fc.map(o => `${xs(o.i)},${ys(o.r.p90)}`).join(' ');
                const bot = fc.slice().reverse().map(o => `${xs(o.i)},${ys(o.r.p10)}`).join(' ');
                band = `<polygon class="ch-band" points="${top} ${bot}"></polygon>`;
            }
        }
        const fcDivider = fcStart > 0 ? `<line class="ch-divider" x1="${(xs(fcStart - 1) + xs(fcStart)) / 2}" x2="${(xs(fcStart - 1) + xs(fcStart)) / 2}" y1="${m.t}" y2="${H - m.b}"></line><text class="ch-tick" x="${(xs(fcStart - 1) + xs(fcStart)) / 2 + 4}" y="${m.t + 10}">FORECAST →</text>` : '';
        const fcDots = all.map((r, i) => r.kind === 'fc' && r.p50 !== null ? `<circle class="ch-fc-dot" cx="${xs(i)}" cy="${ys(r.p50)}" r="5"></circle>` : '').join('');
        const actualPath = line('actual', 'hist');
        const targetPath = line('target', 'hist');
        const fcPath = fPoints.length > 1 ? line('p50', 'fc') : '';

        el.innerHTML = `<svg class="hist-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Production history: actual versus target over time${fPoints.length ? ', with forecast' : ''}">
                ${grid}${band}${fcDivider}
                ${targetPath ? `<path class="ch-target" d="${targetPath}" stroke="${C_TARGET}"></path>` : ''}
                ${actualPath ? `<path class="ch-actual" d="${actualPath}" stroke="${C_ACTUAL}"></path>` : ''}
                ${fcPath ? `<path class="ch-fc" d="${fcPath}" stroke="${C_FORECAST}"></path>` : ''}
                ${fcDots}
                ${xLabels}
                <text class="ch-axis-title" x="${W - m.r}" y="${H - 1}" text-anchor="end">Time (date) →</text>
                <line class="ch-cross" id="chCross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" visibility="hidden"></line>
            </svg>
            <div class="ch-tooltip" id="chTooltip" hidden></div>
            ${fNote ? `<div class="chart-note">${esc(fNote)}</div>` : ''}
            <details class="chart-table"><summary>View data table</summary><div class="table-responsive"><table class="enterprise-table">
                <thead><tr><th>Date</th><th>Actual</th><th>Target</th><th>P10</th><th>P50</th><th>P90</th></tr></thead>
                <tbody>${all.map(r => `<tr><td>${esc(r.date)}</td><td>${esc(r.actual != null ? fmtT(r.actual) : '')}</td><td>${esc(r.target != null ? fmtT(r.target) : '')}</td><td>${esc(r.p10 != null ? fmtT(r.p10) : '')}</td><td>${esc(r.p50 != null ? fmtT(r.p50) : '')}</td><td>${esc(r.p90 != null ? fmtT(r.p90) : '')}</td></tr>`).join('')}</tbody>
            </table></div></details>`;

        legend.innerHTML = `<span><i class="lg-line" style="background:${C_ACTUAL}"></i>Actual</span>
            <span><i class="lg-line lg-dashed" style="border-color:${C_TARGET}"></i>Target</span>
            ${fPoints.length ? `<span><i class="lg-dot" style="background:${C_FORECAST}"></i>Forecast P50 (median)</span>` : ''}
            ${band ? `<span><i class="lg-band"></i>P10–P90 forecast interval${f && f.raw && f.raw.quantiles_validated === true ? '' : ' (not validated)'}</span>` : ''}
            <span class="muted">7-day period totals · synthetic operations</span>`;

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
                    ? `Actual: ${esc(fmtT(r.actual))}<br>Target: ${esc(fmtT(r.target))}`
                    : `P10: ${esc(fmtT(r.p10))}<br>P50: ${esc(fmtT(r.p50))}<br>P90: ${esc(fmtT(r.p90))}`);
            const left = xs(best) / W * rect.width;
            tip.style.left = `${Math.min(left + 12, rect.width - 170)}px`;
        });
        svg.addEventListener('mouseleave', () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); });
    }

    // ═══════════════════════════════════════════════════════
    // SCREEN 4 — RECOVERY & CONTINGENCY
    // ═══════════════════════════════════════════════════════
    function renderRecoveryBaseline() {
        const s = state.supply;
        const set = (id, v) => { $(id).textContent = fmtT(v); };
        set('#rcTarget', pick(s, 'target_tonnes'));
        set('#rcP10', pick(s, 'p10_tonnes'));
        set('#rcP50', pick(s, 'p50_tonnes'));
        set('#rcP90', pick(s, 'p90_tonnes'));
        set('#rcGap', pick(s, 'gap_p50_tonnes', 'gap_tonnes'));
    }

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

    function normPortfolios(r) {
        const list = asArray(pick(r, 'portfolios', 'action_portfolios', 'results', 'candidates'));
        return list.map(p => {
            if (typeof p === 'string') return { name: human(p), id: p };
            const feasible = pick(p, 'feasibility', 'status', 'feasibility_status');
            return {
                id: pick(p, 'id', 'portfolio_id', 'name', 'portfolio'),
                name: actionName(p) || '—',
                recovery: num(pick(p, 'expected_recovery_tonnes', 'recovery_tonnes')),
                residual: num(pick(p, 'expected_residual_gap_tonnes', 'residual_gap_tonnes', 'expected_gap_tonnes')),
                worst: num(pick(p, 'worst_case_residual_gap_tonnes', 'worst_tested_residual_gap_tonnes', 'worst_gap_tonnes')),
                feasibility: pick(p, 'modelled_feasibility') ?? (feasible !== undefined ? feasible : (typeof p.feasible === 'boolean' ? (p.feasible ? 'FEASIBLE' : 'NOT FEASIBLE') : null)),
                selectedFlag: p.selected === true || p.is_selected === true,
                eligible: p.eligible,
                applicability: pick(p, 'selection_applicability', 'applicability'),
                lowScenarios: asArray(p.low_applicability_scenarios),
                blockedReason: p.selection_blocked_reason,
                burden: num(p.intervention_burden),
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

    function renderPortfolios(r) {
        const el = $('#portfolioTable');
        const ps = normPortfolios(r);
        const selKey = selectedKey(r);
        const isSel = p => p.selectedFlag || (selKey !== null && (String(p.id) === String(selKey) || p.name === human(selKey)));
        if (!ps.length) {
            el.innerHTML = `<div class="empty-state">No action portfolios returned by the backend.${selKey ? ` Selected: <b>${esc(human(selKey))}</b>` : ''}</div>`;
            return null;
        }
        const applCls = a => /LOW/.test(norm(a)) ? 'no' : (/MOD/.test(norm(a)) ? 'review' : 'yes');
        el.innerHTML = `<table class="enterprise-table portfolio-table">
            <thead><tr><th scope="col">ACTION PORTFOLIO</th><th scope="col">BURDEN</th><th scope="col">EXPECTED RECOVERY</th><th scope="col">EXPECTED RESIDUAL</th><th scope="col">WORST TESTED RESIDUAL</th><th scope="col">APPLICABILITY</th><th scope="col">MODELLED FEASIBILITY</th><th scope="col">AUTOMATED SELECTION</th></tr></thead>
            <tbody>${ps.map(p => `<tr class="${isSel(p) ? 'row-selected' : ''}${p.eligible === false ? ' row-blocked' : ''}">
                <td>${isSel(p) ? '<span class="sel-tag">SELECTED</span> ' : ''}${esc(p.name)}</td>
                <td class="mono-val">${p.burden !== null ? esc(fmtNum(p.burden, 1)) : 'N/A'}</td>
                <td class="mono-val">${p.recovery !== null ? (p.recovery > 0 ? '+' : '') + esc(fmtT(p.recovery)) : 'N/A'}</td>
                <td class="mono-val">${esc(fmtT(p.residual))}</td>
                <td class="mono-val">${esc(fmtT(p.worst))}</td>
                <td>${p.applicability ? `<span class="feas feas-${applCls(p.applicability)}" title="${esc(p.lowScenarios.length ? 'Outside model experience in: ' + p.lowScenarios.map(human).join(', ') : 'All tested scenarios within model experience')}">${esc(upperHuman(p.applicability))}</span>` : 'N/A'}</td>
                <td>${p.feasibility !== null && p.feasibility !== undefined ? `<span class="feas feas-${/NOT|INFEAS/.test(norm(p.feasibility)) ? 'no' : (/REVIEW|UNKNOWN/.test(norm(p.feasibility)) ? 'review' : 'yes')}">${esc(upperHuman(p.feasibility))}</span>` : 'N/A'}</td>
                <td>${p.eligible === false ? `<span class="feas feas-no">BLOCKED</span><div class="blocked-why">${esc(p.blockedReason || '')}</div><div class="muted">Diagnostic only</div>` : (p.eligible === true ? '<span class="feas feas-yes">ELIGIBLE</span>' : 'N/A')}</td>
            </tr>`).join('')}</tbody>
        </table>
        ${pick(r, 'selection_explanation') ? `<div class="chart-note"><b>Why selected:</b> ${esc(r.selection_explanation)}</div>` : ''}
        ${r.selection_status === 'REVIEW_REQUIRED' ? '<div class="caution-block" role="alert">Automated portfolio selection withheld — no portfolio passed the modelled-feasibility and applicability gates. Human review required.</div>' : ''}
        ${selKey === null && !ps.some(p => p.selectedFlag) && r.selection_status !== 'REVIEW_REQUIRED' ? '<div class="chart-note">The backend did not identify a selected portfolio.</div>' : ''}
        <div class="chart-note muted">Selection rule: ${esc(pick(r, 'selection_rule') || 'not returned')}.</div>`;
        return ps.find(isSel) || null;
    }

    function renderResidual(rec, sel, dec) {
        const residual = num(pick(rec, 'expected_residual_gap_tonnes', 'residual_gap_tonnes')) ?? (sel ? sel.residual : null);
        const el = $('#rcResidual');
        const prev = el.dataset.value ? Number(el.dataset.value) : null;
        el.textContent = fmtT(residual);
        el.dataset.value = residual ?? '';
        if (prev !== null && residual !== null && prev !== residual) { el.classList.remove('value-change'); void el.offsetWidth; el.classList.add('value-change'); }
        $('#rcResidualSub').textContent = sel ? `After backend-selected portfolio: ${sel.name}` : (rec && rec.selection_status === 'REVIEW_REQUIRED' ? 'No eligible portfolio — selection withheld' : 'After backend-selected portfolio');
        $('#residualHero').classList.toggle('residual-positive', residual !== null && residual > 0);

        const can = pick(rec, 'can_operations_close', 'operations_can_close', 'gap_closed', 'operational_sufficient');
        $('#rcCanClose').innerHTML = can === true ? '<span class="yes-no yes">YES</span>'
            : can === false ? '<span class="yes-no no">NO</span>'
            : '<span class="muted">Not returned</span>';

        const horizon = (dec && dec.horizon) || pick(rec, 'decision_horizon', 'horizon');
        const h = norm(horizon);
        $('#rcHorizon').textContent = horizon ? upperHuman(horizon) : 'N/A';
        $('#rcHorizonNote').textContent = /NEAR/.test(h) ? 'Exploration is not treated as immediate recovery.'
            : /STRATEGIC|LONG/.test(h) ? 'Exploration contingency may be activated.' : '';
    }

    async function evaluateRecovery() {
        const btn = $('#btnEvaluate');
        setLoading(btn, true);
        const previous = state.decision.current;
        const body = currentInputs(false);
        $('#portfolioTable').innerHTML = loadingHTML('Evaluating recovery portfolios…');
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
        state.recovery = rec;
        state.contingency = con;
        state.loaded.recovery = true;

        let sel = null;
        if (rec) {
            sel = renderPortfolios(rec);
            $('#rcProvenance').innerHTML = provenanceHTML(pick(rec, 'provenance'));
        } else {
            $('#portfolioTable').innerHTML = errorHTML('Recovery evaluation unavailable.', recErr, 'recovery');
            $('#rcProvenance').innerHTML = badgeHTML('UNAVAILABLE', 'DATA');
        }
        // Decision: contingency response is authoritative; recovery may carry one too.
        const dec = normDecision(con) || normDecision(rec);
        if (dec && dec.residual === null) dec.residual = num(pick(rec, 'expected_residual_gap_tonnes', 'residual_gap_tonnes'));
        renderResidual(rec || {}, sel, dec);
        renderDecision($('#rcDecision'), dec);
        if (!dec && conErr) {
            $('#rcDecision').insertAdjacentHTML('beforeend', `<div class="decision-text muted">Contingency evaluation failed: ${esc(conErr.userMessage || '')}</div>`);
        }

        const src = con || rec || {};
        const nextT = dec && dec.nextTarget;
        const whyPanel = $('#rcWhyNowPanel');
        if (nextT) {
            whyPanel.hidden = false;
            renderWhyNow($('#rcWhyNow'), {
                targetId: nextT,
                whyTarget: pick(src, 'why_target'),
                whyNow: pick(src, 'why_target_now'),
                reasonCodes: undefined,
                nextEvidence: pick(src, 'next_evidence'),
                showWhyTarget: !!pick(src, 'why_target'),
            });
            $('#rcWhyNow').insertAdjacentHTML('beforeend', `<div class="why-actions"><button type="button" class="cmd-btn cmd-btn-outline" data-open-target="${esc(nextT)}">Open ${esc(nextT)} in Exploration</button></div>`);
        } else {
            whyPanel.hidden = true;
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
    // DEMO_F scenario they start at the backend's documented perturbation instead.
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
            const h = await api.get(`/api/decision/history?mine_id=${encodeURIComponent(MINE_ID)}`);
            state.decisionHistory = Array.isArray(h) ? h : asArray(pick(h, 'history', 'decisions', 'items'));
            if (!state.decisionHistory.length) { el.innerHTML = '<div class="empty-state">No decisions recorded yet.</div>'; return; }
            el.innerHTML = `<table class="enterprise-table"><thead><tr><th>TIME</th><th>DECISION</th><th>SCENARIO</th><th>NEXT TARGET</th><th>STATUS</th></tr></thead>
                <tbody>${state.decisionHistory.slice(0, 20).map(d => `<tr>
                    <td class="mono-val">${esc(pick(d, 'timestamp', 'created_at', 'time') || '—')}</td>
                    <td>${esc(decisionTitle(pick(d, 'decision_state', 'decision')))}</td>
                    <td>${esc(human(pick(d, 'scenario') || '—'))}</td>
                    <td class="mono-val">${esc(pick(d, 'next_target') || '—')}</td>
                    <td>${esc(upperHuman(pick(d, 'review_status', 'status') || '—'))}</td>
                </tr>`).join('')}</tbody></table>`;
        } catch (e) {
            el.innerHTML = errorHTML('Decision history unavailable.', e);
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
            showToast(`Decision review failed. ${e.userMessage || ''}`, 'error');
        } finally {
            setLoading(btn, false);
            btn.disabled = !state.decision.current;
        }
    }

    // ═══════════════════════════════════════════════════════
    // SCREEN 5 — MODEL TRUST
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
                <dd>${val === null ? `N/A <span class="na-why">Not returned by the backend validation report.</span>` : val}</dd>
                ${def ? `<div class="metric-def">${esc(def)}</div>` : ''}
            </div>`;
        }).join('')}</dl>`;
    }

    function trustNotes(src) {
        const n = asArray(pick(src, 'notes', 'note', 'caveats', 'message'));
        return n.length ? `<ul class="trust-notes">${n.map(x => `<li>${esc(reasonText(x))}</li>`).join('')}</ul>` : '';
    }

    async function loadTrust() {
        const btn = $('#btnRefreshTrust');
        setLoading(btn, true);
        ['#trustExploration', '#trustProduction', '#trustReconciliation', '#trustProvenance'].forEach(s => { $(s).innerHTML = loadingHTML(); });
        const [ex, pr, pv, rc] = await Promise.allSettled([
            api.get('/api/trust/exploration'),
            api.get('/api/trust/production'),
            api.get('/api/trust/provenance'),
            api.get(`/api/production/reconciliation?mine_id=${encodeURIComponent(MINE_ID)}`),
        ]);
        state.trust.exploration = ex.status === 'fulfilled' ? ex.value : null;
        state.trust.production = pr.status === 'fulfilled' ? pr.value : null;
        state.trust.provenance = pv.status === 'fulfilled' ? pv.value : null;
        state.trust.reconciliation = rc.status === 'fulfilled' ? rc.value : null;
        state.loaded['model-trust'] = true;

        $('#trustExploration').innerHTML = ex.status === 'fulfilled'
            ? provenanceRowIf(ex.value) + metricsGridHTML(ex.value, [
                ['Spatial validation', ['spatial_validation', 'validation_method'], 'How held-out data were formed (no random pixel splits).'],
                ['ROC-AUC (spatial CV)', ['roc_auc'], 'Ranking of held-out MRDS cells vs unlabelled background; 0.5 = no skill.'],
                ['PR-AUC (spatial CV)', ['pr_auc'], 'Precision-recall area; compare with the prevalence baseline below.'],
                ['PR-AUC prevalence baseline', ['pr_auc_prevalence_baseline'], 'PR-AUC a random ranking would get.'],
                ['Top-area capture', ['top_area_capture'], 'Share of held-out occurrences inside the top 5/10/20 % of held-out area.'],
                ['Region holdout ROC-AUC', ['region_holdout'], 'Train on one half of the belt, test on the other (east/west at 79.8E).'],
                ['Region holdout method', ['region_holdout_method'], null],
                ['Observation window', ['observation_window'], 'Fixed reference composite — not real-time imagery.'],
                ['Effective resolution', ['effective_resolution'], null],
                ['Subsurface evidence', ['subsurface_evidence'], null],
                ['Applicability method', ['applicability_method'], 'How out-of-experience inputs are detected.'],
                ['Uncertainty method', ['uncertainty_method'], 'How rank uncertainty is measured.'],
                ['Calibration', ['calibration'], null],
            ]) + trustNotes(ex.value)
            : errorHTML('Exploration validation unavailable.', ex.reason, 'model-trust');

        $('#trustProduction').innerHTML = pr.status === 'fulfilled'
            ? provenanceRowIf(pr.value) + metricsGridHTML(pr.value, [
                ['Validation method', ['validation_method'], 'Chronological rolling-origin backtest; no shuffling.'],
                ['Test window', ['test_window'], 'Untouched periods used for every metric below.'],
                ['P50 MAE (t / 7 days)', ['mae'], 'Mean absolute error of the P50 forecast on the test window.'],
                ['Best naive baseline MAE', ['baseline_mae'], 'Better of previous-period and 4-period moving average, same periods.'],
                ['Improvement vs best baseline (%)', ['mae_improvement_vs_best_baseline_pct'], 'Negative would mean the baseline wins.'],
                ['RMSE (t / 7 days)', ['rmse'], 'Root-mean-square error of P50 on the test window.'],
                ['R²', ['r2'], 'Share of test-window variance explained by P50.'],
                ['Observed P10–P90 coverage', ['observed_coverage'], 'Share of test periods whose actual fell inside P10–P90.'],
                ['Nominal coverage', ['nominal_coverage'], 'What a P10–P90 interval is designed to contain.'],
                ['Quantiles validated', ['quantiles_validated'], 'Passes the documented coverage + hit-rate rule?'],
                ['Quantile hit rates', ['quantile_hit_rate'], 'Share of actuals at or below P10 / P50 / P90 (nominal 0.1 / 0.5 / 0.9).'],
                ['Pinball loss', ['pinball_loss'], 'Quantile loss per level (lower is better).'],
                ['Calibration window', ['protocol'], null],
                ['Forecast procedure', ['forecast_procedure'], 'Persistence assumption for unknown future conditions.'],
                ['Applicability method', ['applicability_method'], null],
            ]) + trustNotes(pr.value)
            : errorHTML('Production validation unavailable.', pr.reason, 'model-trust');

        renderProvenanceTrust(pv);
        renderReconciliation(rc);
        updateDataModePill();
        setLoading(btn, false);
    }

    function provenanceRowIf(src) {
        const p = pick(src, 'provenance');
        return p ? `<div class="prov-row prov-row-block">${provenanceHTML(p)}</div>` : '';
    }

    function renderProvenanceTrust(pv) {
        const el = $('#trustProvenance');
        if (pv.status !== 'fulfilled') { el.innerHTML = errorHTML('Provenance unavailable.', pv.reason, 'model-trust'); return; }
        const p = pv.value;
        const entries = Array.isArray(p) ? p.map((x, i) => [pick(x, 'name', 'dataset', 'component') || `Source ${i + 1}`, x])
            : Object.entries(p.sources || p.datasets || p).filter(([k]) => k !== 'notes');
        if (!entries.length) { el.innerHTML = '<div class="empty-state">No provenance entries returned.</div>'; return; }
        el.innerHTML = `<table class="enterprise-table prov-table"><tbody>${entries.map(([k, v]) => {
            let mode = null, desc = '';
            if (typeof v === 'string') { if (modeInfo(v).known) mode = v; else desc = v; }
            else if (v && typeof v === 'object') {
                mode = pick(v, 'mode', 'data_mode', 'type', 'status');
                desc = [pick(v, 'description', 'source', 'note'), fmtWindow(pick(v, 'observation_window'))].filter(Boolean).join(' · ');
            } else if (typeof v === 'boolean') desc = v ? 'yes' : 'no';
            return `<tr><th scope="row">${esc(upperHuman(k))}</th><td>${mode ? badgeHTML(mode) : ''} <span class="muted">${esc(desc)}</span></td></tr>`;
        }).join('')}</tbody></table>${trustNotes(p)}`;
    }

    function renderReconciliation(rc) {
        const el = $('#trustReconciliation');
        if (rc.status !== 'fulfilled') { el.innerHTML = errorHTML('Reconciliation unavailable.', rc.reason, 'model-trust'); return; }
        const r = rc.value || {};
        const rows = asArray(Array.isArray(r) ? r : pick(r, 'rows', 'records', 'reconciliation', 'history', 'periods')).map(x => ({
            period: pick(x, 'period', 'date', 'period_end'),
            forecast: num(pick(x, 'forecast_tonnes', 'forecast', 'p50_tonnes', 'p50')),
            actual: num(pick(x, 'actual_tonnes', 'actual')),
            error: num(pick(x, 'error_tonnes', 'error')),
        }));
        const mae = num(pick(r.summary || {}, 'mae') ?? pick(r, 'mae', 'rolling_mae', 'rolling_mae_tonnes'));
        const sm = r.summary || {};
        const bias = num(pick(r, 'bias', 'bias_tonnes', 'mean_error'));
        const hasActual = rows.some(x => x.actual !== null);
        const status = pick(r, 'status', 'actuals_status');
        const summary = `<div class="kpi-strip kpi-strip-4">
            <div class="kpi-tile"><span class="kpi-label">MAE (${esc(sm.periods ?? rows.length)} PERIODS)</span><div class="kpi-val mono-val">${mae !== null ? esc(fmtT(mae)) : 'N/A'}</div></div>
            <div class="kpi-tile"><span class="kpi-label">OVER / UNDER FORECASTS</span><div class="kpi-val mono-val">${num(sm.over_forecast_count) !== null ? `${esc(sm.over_forecast_count)} / ${esc(sm.under_forecast_count)}` : 'N/A'}</div></div>
            <div class="kpi-tile"><span class="kpi-label">LATEST ERROR</span><div class="kpi-val mono-val">${num(sm.latest_error_tonnes) !== null ? esc((sm.latest_error_tonnes > 0 ? '+' : '') + fmtT(sm.latest_error_tonnes)) : 'N/A'}</div></div>
            <div class="kpi-tile"><span class="kpi-label">BIAS (forecast − actual)</span><div class="kpi-val mono-val">${bias !== null ? esc((bias > 0 ? '+' : '') + fmtT(bias)) : 'N/A'}</div></div>
        </div>`;
        if (!hasActual) {
            el.innerHTML = `${provenanceRowIf(r)}<div class="empty-state empty-strong">Actual data unavailable${status ? ` (${esc(upperHuman(status))})` : ''}. Forecast accuracy cannot be reconciled until actual production is recorded.</div>${rows.length ? recTable(rows) : ''}`;
            return;
        }
        el.innerHTML = provenanceRowIf(r) + summary + (r.basis ? `<div class="chart-note">${esc(r.basis)} ${esc(r.error_convention || '')}</div>` : '') + recTable(rows);
    }

    function recTable(rows) {
        return `<div class="table-responsive"><table class="enterprise-table"><thead><tr><th>PERIOD</th><th>FORECAST</th><th>ACTUAL</th><th>ERROR</th></tr></thead>
            <tbody>${rows.map(x => `<tr><td class="mono-val">${esc(x.period ?? '—')}</td><td class="mono-val">${esc(fmtT(x.forecast))}</td><td class="mono-val">${x.actual !== null ? esc(fmtT(x.actual)) : 'Actual unavailable'}</td><td class="mono-val">${x.error !== null ? esc((x.error > 0 ? '+' : '') + fmtT(x.error)) : 'N/A'}</td></tr>`).join('')}</tbody></table></div>`;
    }

    // ═══════════════════════════════════════════════════════
    // NAVIGATION
    // ═══════════════════════════════════════════════════════
    const SECTIONS = {
        'supply-command': { title: 'SUPPLY COMMAND', load: loadSupply },
        'exploration': { title: 'EXPLORATION', load: loadExploration },
        'production-risk': { title: 'PRODUCTION RISK', load: loadProduction },
        'recovery': { title: 'RECOVERY & CONTINGENCY', load: () => { loadDecisionHistory(); return evaluateRecovery(); } },
        'model-trust': { title: 'MODEL TRUST', load: loadTrust },
    };

    function showSection(id, updateHash = true) {
        if (!SECTIONS[id]) id = 'supply-command';
        state.currentSection = id;
        if (updateHash && window.location.hash !== `#${id}`) history.replaceState(null, '', `#${id}`);
        $$('.nav-link').forEach(l => {
            const on = l.dataset.section === id;
            l.classList.toggle('active', on);
            if (on) l.setAttribute('aria-current', 'page'); else l.removeAttribute('aria-current');
        });
        $$('.section').forEach(s => s.classList.toggle('active', s.id === `section-${id}`));
        $('#pageTitle').textContent = SECTIONS[id].title;
        if (!state.loaded[id]) SECTIONS[id].load();
        if (id === 'exploration' && state.maps.exploration) setTimeout(() => state.maps.exploration.invalidateSize(), 120);
    }

    function openTarget(id) {
        state.exploration.selectedId = id;
        showSection('exploration');
        if (state.loaded.exploration) selectTarget(id, { fly: true });
        // else loadExploration selects it once targets arrive
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
            const id = window.location.hash.slice(1);
            if (SECTIONS[id] && id !== state.currentSection) showSection(id, false);
        });

        $('#btnRetryBackend').addEventListener('click', retryAll);
        $('#btnRefreshSupply').addEventListener('click', loadSupply);
        $('#btnRefreshProduction').addEventListener('click', loadProduction);
        $('#btnRefreshTrust').addEventListener('click', loadTrust);
        $('#btnViewTarget').addEventListener('click', () => {
            const id = state.supply && pick(state.supply, 'next_target', 'next_target_id');
            if (id) openTarget(String(id));
        });

        $('#targetList').addEventListener('click', e => {
            const b = e.target.closest('[data-target]');
            if (b) {
                selectTarget(b.dataset.target, { fly: true });
                const m = state.maps.markers[b.dataset.target];
                if (m) m.openPopup();
            }
        });
        $('#coordQueryForm').addEventListener('submit', e => {
            e.preventDefault();
            const lat = parseFloat($('#queryLat').value), lon = parseFloat($('#queryLon').value);
            queryCoordinate(lat, lon);
            if (state.maps.exploration && Number.isFinite(lat) && Number.isFinite(lon)) state.maps.exploration.flyTo([lat, lon], Math.max(state.maps.exploration.getZoom(), 9), { duration: 0.9 });
        });

        $('#btnEvaluate').addEventListener('click', () => evaluateRecovery());
        $('#btnFlip').addEventListener('click', runFlip);
        $('#demoSelect').addEventListener('change', e => switchMine(e.target.value));
        $('#btnLogDecision').addEventListener('click', submitDecisionReview);

        const bindRange = (id, out, fmt) => {
            const el = $(id);
            const upd = () => { $(out).textContent = fmt(Number(el.value)); };
            el.addEventListener('input', upd);
            upd();
        };
        bindRange('#flipRainfall', '#flipRainfallVal', v => `${v} mm / 7 d`);
        bindRange('#flipEquip', '#flipEquipVal', v => `${Math.round(v * 100)}%`);
        bindRange('#flipBlast', '#flipBlastVal', v => `${v.toFixed(1)} h`);

        // Delegated: retry buttons and "open target" links rendered inside panels.
        document.addEventListener('click', e => {
            const retry = e.target.closest('[data-retry]');
            if (retry) {
                const id = retry.dataset.retry;
                state.loaded[id] = false;
                if (SECTIONS[id]) SECTIONS[id].load();
                return;
            }
            const open = e.target.closest('[data-open-target]');
            if (open) openTarget(open.dataset.openTarget);
        });
    }

    async function retryAll() {
        const btn = $('#btnRetryBackend');
        setLoading(btn, true);
        const ok = await loadHealth();
        setLoading(btn, false);
        state.loaded = {};
        if (ok || state.health) {
            showToast('Backend reachable. Reloading.', 'success');
        } else {
            showToast('Backend still unavailable.', 'error');
        }
        SECTIONS[state.currentSection].load();
    }

    document.addEventListener('DOMContentLoaded', async () => {
        bindEvents();
        const fromHash = window.location.hash.slice(1);
        const first = SECTIONS[fromHash] ? fromHash : 'supply-command';
        await loadHealth();
        loadDemoStates();
        showSection(first);
        // Supply baseline feeds the Recovery screen and the data-mode pill.
        if (first !== 'supply-command') loadSupply();
    });
})();
