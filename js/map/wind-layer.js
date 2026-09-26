'use strict';
/* =====================================================================
   WindLayer: animated wind on a Leaflet map, drawn on a canvas.
   Feed it a grid { lo0, la1, dx, dy, nx, ny, u, v } (u east, v north, m/s).
   style 'flow' animates particles, style 'arrows' draws a still field.
   ===================================================================== */
const WIND_COLORS = {
  dark: ['rgba(110,160,195,.55)', 'rgba(150,205,228,.68)', 'rgba(190,232,242,.8)', 'rgba(228,246,250,.9)', 'rgba(255,255,255,.98)'],
  light: ['rgba(70,120,165,.55)', 'rgba(45,100,150,.68)', 'rgba(28,80,135,.8)', 'rgba(15,60,110,.9)', 'rgba(8,38,80,.98)']
};
const windPalette = () => WIND_COLORS[document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark'];

const WindLayer = L.Layer.extend({
  options: { pane: 'overlayPane', count: 1400, lineWidth: 1.3, fade: .94, scale: .55, maxAge: 80, style: 'flow', opacity: 1 },
  initialize(o) { L.setOptions(this, o); this._grid = null; this._parts = []; this._raf = 0; this._last = 0; },
  setGrid(g) { this._grid = g; if (this._map && this.options.style === 'arrows') this._drawArrows(); },
  setStyle(s) { this.options.style = s; if (this._map) this._reset(); },
  setOpacity(o) { this.options.opacity = o; if (this._c) this._c.style.opacity = o; },
  onAdd(map) {
    this._map = map; this._c = L.DomUtil.create('canvas', 'agni-wind', map.getPane(this.options.pane)); this._c.style.pointerEvents = 'none';
    map.on('movestart zoomstart', this._hide, this); map.on('moveend zoomend resize', this._reset, this); this._reset();
  },
  onRemove(map) {
    cancelAnimationFrame(this._raf); this._raf = 0; map.off('movestart zoomstart', this._hide, this); map.off('moveend zoomend resize', this._reset, this);
    if (this._c && this._c.parentNode) this._c.parentNode.removeChild(this._c); this._c = null;
  },
  _hide() { cancelAnimationFrame(this._raf); this._raf = 0; if (this._c) this._c.style.opacity = 0; },
  _reset() {
    const m = this._map, c = this._c; if (!m || !c) return;
    const s = m.getSize(); c.width = s.x; c.height = s.y; L.DomUtil.setPosition(c, m.containerPointToLayerPoint([0, 0])); c.style.opacity = this.options.opacity;
    this._ctx = c.getContext('2d'); cancelAnimationFrame(this._raf); this._raf = 0; if (!this._ctx) return; this._ctx.clearRect(0, 0, s.x, s.y);
    if (!this._grid) return;
    if (this.options.style === 'arrows' || REDUCED) { this._drawArrows(); return; }
    this._parts = Array.from({ length: this.options.count }, () => this._spawn({}, s.x, s.y, true)); this._loop();
  },
  _spawn(p, W, H, anyAge) { p.x = Math.random() * W; p.y = Math.random() * H; p.age = anyAge ? Math.random() * this.options.maxAge : 0; return p; },
  _loop() {
    const step = t => { if (!this._c) return; this._raf = requestAnimationFrame(step); if (t - this._last < 32) return; this._last = t; this._frame(); };
    this._raf = requestAnimationFrame(step);
  },
  _frame() {
    const ctx = this._ctx, m = this._map, W = this._c.width, H = this._c.height, g = this._grid, k = this.options.scale, cols = windPalette();
    ctx.globalCompositeOperation = 'destination-in'; ctx.fillStyle = `rgba(0,0,0,${this.options.fade})`; ctx.fillRect(0, 0, W, H); ctx.globalCompositeOperation = 'source-over';
    ctx.lineWidth = this.options.lineWidth; ctx.lineCap = 'round';
    const bk = [[], [], [], [], []];
    for (const p of this._parts) {
      const ll = m.containerPointToLatLng([p.x, p.y]), w = gridSample(g, ll.lng, ll.lat);
      if (!w || p.age > this.options.maxAge) { this._spawn(p, W, H); continue; }
      const nx = p.x + w[0] * k, ny = p.y - w[1] * k;
      if (nx < -8 || nx > W + 8 || ny < -8 || ny > H + 8) { this._spawn(p, W, H); continue; }
      bk[Math.min(4, Math.floor(Math.hypot(w[0], w[1]) / 2.4))].push(p.x, p.y, nx, ny); p.x = nx; p.y = ny; p.age++;
    }
    bk.forEach((a, i) => { if (!a.length) return; ctx.strokeStyle = cols[i]; ctx.beginPath(); for (let j = 0; j < a.length; j += 4) { ctx.moveTo(a[j], a[j + 1]); ctx.lineTo(a[j + 2], a[j + 3]); } ctx.stroke(); });
  },
  _drawArrows() {
    const ctx = this._ctx, m = this._map, W = this._c.width, H = this._c.height, g = this._grid, cols = windPalette(); if (!ctx || !g) return;
    ctx.clearRect(0, 0, W, H); ctx.lineWidth = 1.4; ctx.lineCap = 'round';
    const gap = 44;
    for (let y = gap / 2; y < H; y += gap) for (let x = gap / 2; x < W; x += gap) {
      const ll = m.containerPointToLatLng([x, y]), w = gridSample(g, ll.lng, ll.lat); if (!w) continue;
      const sp = Math.hypot(w[0], w[1]), len = Math.min(30, 8 + sp * 3.4), a = Math.atan2(-w[1], w[0]), x1 = x + Math.cos(a) * len / 2, y1 = y + Math.sin(a) * len / 2, x0 = x - Math.cos(a) * len / 2, y0 = y - Math.sin(a) * len / 2;
      ctx.strokeStyle = cols[Math.min(4, Math.floor(sp / 2.4))]; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
      ctx.moveTo(x1, y1); ctx.lineTo(x1 - Math.cos(a - .5) * 6, y1 - Math.sin(a - .5) * 6); ctx.moveTo(x1, y1); ctx.lineTo(x1 - Math.cos(a + .5) * 6, y1 - Math.sin(a + .5) * 6); ctx.stroke();
    }
  }
});
const windGradientCss = () => 'linear-gradient(90deg,' + windPalette().join(',') + ')';
