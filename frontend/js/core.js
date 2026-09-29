'use strict';
/* =====================================================================
   Small DOM and colour helpers shared by every script. Loaded first because
   wind-layer.js, leaflet-map.js and analysis.js read some of these (REDUCED,
   store, cv) as soon as they are parsed, not just inside functions.
   ===================================================================== */
const $ = (s, e = document) => e.querySelector(s), $$ = (s, e = document) => [...e.querySelectorAll(s)];
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const cv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ccol = id => cv('--c-' + id);
const REDUCED = typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches;
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
const alpha = (hex, a) => { hex = (hex || '').trim(); if (hex[0] !== '#' || hex.length !== 7) return hex; const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`; };
const hexMix = (a, b, t) => { const A = parseInt(a.slice(1), 16), B = parseInt(b.slice(1), 16); const c = i => Math.round(((A >> i) & 255) * (1 - t) + ((B >> i) & 255) * t); return `rgb(${c(16)},${c(8)},${c(0)})`; };
const tk = () => ({ tx: cv('--tx'), tx2: cv('--tx2'), dim: cv('--dim'), line: cv('--line'), line2: cv('--line2'), panel: cv('--panel'), raised: cv('--raised'), danger: cv('--danger'), warn: cv('--warn'), inv: cv('--inv') });
const lc = s => s.charAt(0).toLowerCase() + s.slice(1);
