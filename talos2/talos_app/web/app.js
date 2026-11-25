import * as THREE from 'three';
import { OrbitControls } from '/web/vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
const S = { region: null, A: null, B: null, route: null, profile: 'equilibre', mode: 'point', level: 'sat',
            picking: null, mission: false, state: null, tile: null, trackE: [], trackT: [], particles: [], scan: [],
            path: [], err: [], events: [], vox: 0, t0: null, paused: false };
let MX, MY, LON0, LAT0;
const toXY = (lon, lat) => [(lon - LON0) * MX, (lat - LAT0) * MY];
const toLL = (x, y) => [LON0 + x / MX, LAT0 + y / MY];
const fmtLL = (lon, lat) => `${lat.toFixed(5)}° N  ${lon.toFixed(5)}° E`;
const fmtT = s => { s = Math.max(0, Math.round(s)); return `${String(Math.floor(s / 3600)).padStart(2, '0')}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`; };
const km = m => m >= 1000 ? (m / 1000).toFixed(m >= 10000 ? 0 : 1) + ' km' : Math.round(m) + ' m';
const PHASE = { NAV: 'NAVIGATION', APPROACH: 'APPROCHE TRANCHÉE', DROP: 'DÉPÔT', RETREAT: 'RETRAIT', DONE: 'TERMINÉ' };

/* ============================================================ horloge, onglets */
setInterval(() => { $('utc').textContent = new Date().toISOString().slice(11, 19); }, 1000);
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  b.parentElement.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('#left .tab').forEach(t => t.classList.toggle('on', t.id === b.dataset.tab));
});
function seg(id, cb) {
  const el = $(id);
  el.querySelectorAll('button').forEach(b => b.onclick = () => {
    el.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b)); cb(b.dataset.v);
  });
}
seg('profile', v => { S.profile = v; if (S.A && S.B) computeRoute(); });
seg('mode', v => S.mode = v);
seg('level', v => S.level = v);
seg('cam', v => ctl('camera', v));
$('payload').oninput = e => $('payV').textContent = (+e.target.value).toFixed(1);

/* ============================================================ carte */
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
const map = new maplibregl.Map({
  container: 'map', attributionControl: false, center: [37.6, 48.65], zoom: 9, maxPitch: 70,
  style: {
    version: 8, glyphs: 'https://protomaps.github.io/basemaps-assets/fonts/{fontstack}/{range}.pbf',
    sources: {
      sat: { type: 'raster', tiles: [ESRI], tileSize: 256, maxzoom: 19, attribution: 'Imagerie © Esri, Maxar, Earthstar Geographics' },
      topo: { type: 'raster', tiles: ['https://a.tile.opentopomap.org/{z}/{x}/{y}.png', 'https://b.tile.opentopomap.org/{z}/{x}/{y}.png'], tileSize: 256, maxzoom: 17, attribution: '© OpenTopoMap (CC-BY-SA)' },
      dark: { type: 'raster', tiles: ['https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png', 'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'], tileSize: 256, maxzoom: 19, attribution: '© CARTO © OpenStreetMap' },
    },
    layers: [
      { id: 'bg', type: 'background', paint: { 'background-color': '#0b1016' } },
      { id: 'bm-sat', type: 'raster', source: 'sat', paint: { 'raster-saturation': -0.25, 'raster-brightness-max': 0.9 } },
      { id: 'bm-topo', type: 'raster', source: 'topo', layout: { visibility: 'none' }, paint: { 'raster-opacity': 0.9 } },
      { id: 'bm-dark', type: 'raster', source: 'dark', layout: { visibility: 'none' } },
    ]
  }
});
map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');
map.addControl(new maplibregl.AttributionControl({ compact: true, customAttribution: 'Ligne de contact : DeepStateMap · © OpenStreetMap (ODbL) · ESA WorldCover · Copernicus DEM' }), 'bottom-right');

document.querySelectorAll('input[name=bm]').forEach(r => r.onchange = () => {
  ['sat', 'topo', 'dark'].forEach(k => map.setLayoutProperty('bm-' + k, 'visibility', r.value === k ? 'visible' : 'none'));
  if (r.value === 'none') setLayer('landcover', true);
});

const fc = (features = []) => ({ type: 'FeatureCollection', features });
const pt = (c, p = {}) => ({ type: 'Feature', geometry: { type: 'Point', coordinates: c }, properties: p });
const ln = (c, p = {}) => ({ type: 'Feature', geometry: { type: 'LineString', coordinates: c }, properties: p });
const circle = (lon, lat, r, n = 40) => { const c = []; for (let i = 0; i <= n; i++) { const a = i / n * 2 * Math.PI; c.push([lon + r * Math.cos(a) / MX, lat + r * Math.sin(a) / MY]); } return { type: 'Feature', geometry: { type: 'Polygon', coordinates: [c] }, properties: {} }; };

function symbol(kind) {
  const c = document.createElement('canvas'), s = 40; c.width = c.height = s; const g = c.getContext('2d');
  if (kind === 'friend' || kind === 'depot') {
    g.fillStyle = '#9fd3ff'; g.strokeStyle = '#0a2a4a'; g.lineWidth = 2;
    g.fillRect(6, 11, 28, 18); g.strokeStyle = '#123e66'; g.lineWidth = 2.5; g.strokeRect(6, 11, 28, 18);
    g.strokeStyle = '#0a2a4a'; g.lineWidth = 2; g.beginPath();
    if (kind === 'friend') { g.moveTo(6, 11); g.lineTo(34, 29); g.moveTo(34, 11); g.lineTo(6, 29); }
    else { g.moveTo(6, 23); g.lineTo(34, 23); g.moveTo(14, 23); g.lineTo(14, 29); g.moveTo(26, 23); g.lineTo(26, 29); }
    g.stroke();
  } else if (kind === 'robot') {
    g.translate(20, 20); g.fillStyle = '#00e5ff'; g.strokeStyle = '#002a33'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(0, -15); g.lineTo(10, 11); g.lineTo(0, 6); g.lineTo(-10, 11); g.closePath(); g.fill(); g.stroke();
  } else if (kind === 'ghost') {
    g.translate(20, 20); g.strokeStyle = '#22d39a'; g.lineWidth = 2.5;
    g.beginPath(); g.moveTo(0, -13); g.lineTo(9, 10); g.lineTo(0, 5); g.lineTo(-9, 10); g.closePath(); g.stroke();
  } else if (kind === 'A' || kind === 'B') {
    g.fillStyle = kind === 'A' ? '#22d39a' : '#ffb020'; g.strokeStyle = '#000'; g.lineWidth = 2;
    g.beginPath(); g.arc(20, 20, 13, 0, 7); g.fill(); g.stroke();
    g.fillStyle = '#000'; g.font = 'bold 15px Segoe UI'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(kind, 20, 21);
  }
  return g.getImageData(0, 0, s, s);
}

const LAYERS = [
  // id, libellé, groupe, couches carte, visible, couleur
  ['front', 'Ligne de contact', 1, ['front-glow', 'front-line'], true, '#ff3b3b'],
  ['occupied', 'Zone occupée', 1, ['occ-fill', 'occ-hatch'], true, 'rgba(255,59,59,.5)'],
  ['grey', 'Zone grise (statut inconnu)', 1, ['grey-fill'], true, 'rgba(160,160,160,.6)'],
  ['danger', 'Dangerosité (proximité ligne)', 1, ['ov-danger'], false, '#ff7a00'],
  ['allied', 'Positions amies (FICTIVES)', 1, ['allied-sym', 'allied-lbl'], true, '#3fa9ff'],
  ['places', 'Localités', 1, ['places-lbl', 'places-dot'], true, '#d5e2ec'],
  ['landcover', 'Occupation du sol 10 m (WorldCover)', 2, ['ov-landcover'], false, '#006400'],
  ['hillshade', 'Relief ombré (Copernicus 30 m)', 2, ['ov-hillshade'], false, '#888'],
  ['roads', 'Routes', 2, ['roads'], true, '#f2c14e'],
  ['tracks', 'Pistes et chemins', 2, ['tracks'], false, '#c8a26a'],
  ['rows', 'Haies / lignes d\'arbres', 2, ['rows'], false, '#3cb44b'],
  ['water', 'Cours d\'eau', 2, ['water'], true, '#3a8fd8'],
  ['route', 'Itinéraire planifié', 3, ['route-casing', 'route', 'ab'], true, '#00e5ff'],
  ['tile', 'Tuile active (400 m)', 3, ['tile-line'], true, '#00e5ff'],
  ['trackE', 'Trace estimée', 3, ['track-est'], true, '#00e5ff'],
  ['trackT', 'Trace vraie (simulation)', 3, ['track-true', 'robot-true'], false, '#22d39a'],
  ['sigma', 'Incertitude de position', 3, ['sigma-fill', 'sigma-line'], true, '#00e5ff'],
  ['particles', 'Particules du filtre', 3, ['particles'], false, '#ffffff'],
  ['scan', 'Impacts LiDAR', 3, ['scan'], true, '#ff3b8a'],
  ['lpath', 'Chemin local (replanification)', 3, ['lpath'], true, '#ffffff'],
];
function setLayer(id, on) {
  const L = LAYERS.find(l => l[0] === id); if (!L) return;
  L[4] = on; L[3].forEach(l => map.getLayer(l) && map.setLayoutProperty(l, 'visibility', on ? 'visible' : 'none'));
  const cb = document.querySelector(`input[data-layer=${id}]`); if (cb) cb.checked = on;
}
function buildLayerUI() {
  LAYERS.forEach(([id, lab, grp, , vis, col]) => {
    const el = document.createElement('label');
    el.innerHTML = `<input type="checkbox" data-layer="${id}" ${vis ? 'checked' : ''}><i class="sw" style="background:${col}"></i>${lab}`;
    el.querySelector('input').onchange = e => setLayer(id, e.target.checked);
    $('layers' + grp).appendChild(el);
  });
}

async function loadRegion() {
  const R = await (await fetch('/api/region')).json(); S.region = R;
  const [lo0, la0, lo1, la1] = R.summary.bbox; LON0 = lo0; LAT0 = la0;
  const latm = (la0 + la1) / 2; MX = 111320 * Math.cos(latm * Math.PI / 180); MY = 110574;
  const corners = [[lo0, la1], [lo1, la1], [lo1, la0], [lo0, la0]];
  ['friend', 'depot', 'robot', 'ghost', 'A', 'B'].forEach(k => map.addImage(k, symbol(k), { pixelRatio: 2 }));
  // calques image (reechantillonnes en Mercator cote serveur)
  ['hillshade', 'landcover', 'danger'].forEach(k => {
    map.addSource('ov-' + k, { type: 'image', url: `/api/overlay/${k}.png`, coordinates: corners });
    map.addLayer({ id: 'ov-' + k, type: 'raster', source: 'ov-' + k, layout: { visibility: 'none' }, paint: { 'raster-opacity': k === 'landcover' ? 0.75 : 0.8, 'raster-resampling': 'nearest' } });
  });
  map.addSource('bbox', { type: 'geojson', data: { type: 'Feature', geometry: { type: 'Polygon', coordinates: [[...corners, corners[0]]] } } });
  map.addLayer({ id: 'bbox', type: 'line', source: 'bbox', paint: { 'line-color': '#00e5ff', 'line-opacity': 0.35, 'line-dasharray': [4, 3], 'line-width': 1 } });
  // vecteurs OSM
  const lineLayer = (id, src, paint, minzoom = 0, vis = true) => { map.addSource(id, { type: 'geojson', data: `/api/layer/${src}` }); map.addLayer({ id, type: 'line', source: id, minzoom, layout: { visibility: vis ? 'visible' : 'none', 'line-cap': 'round' }, paint }); };
  lineLayer('water', 'water', { 'line-color': '#3a8fd8', 'line-width': ['interpolate', ['linear'], ['zoom'], 8, ['match', ['get', 'k'], 'river', 1.2, 0.4], 14, ['match', ['get', 'k'], 'river', 3, 1.2]], 'line-opacity': 0.6 });
  lineLayer('rows', 'tree_rows', { 'line-color': '#3cb44b', 'line-width': 1.6, 'line-opacity': 0.9 }, 11, false);
  lineLayer('tracks', 'tracks', { 'line-color': '#c8a26a', 'line-width': 1, 'line-dasharray': [3, 2], 'line-opacity': 0.8 }, 12, false);
  lineLayer('roads', 'roads', { 'line-color': ['match', ['get', 'k'], 'motorway', '#ff9f43', 'trunk', '#ff9f43', 'primary', '#f2c14e', '#e0d090'], 'line-width': ['interpolate', ['linear'], ['zoom'], 8, ['match', ['get', 'k'], 'tertiary', 0.4, 1.0], 14, 3], 'line-opacity': 0.85 });
  // situation
  const F = R.front;
  map.addSource('occ', { type: 'geojson', data: { type: 'Feature', geometry: F.occupied, properties: {} } });
  map.addLayer({ id: 'occ-fill', type: 'fill', source: 'occ', paint: { 'fill-color': '#ff3b3b', 'fill-opacity': 0.2 } });
  map.addLayer({ id: 'occ-hatch', type: 'line', source: 'occ', paint: { 'line-color': '#ff3b3b', 'line-opacity': 0.35, 'line-width': 0.8 } });
  if (F.grey) { map.addSource('grey', { type: 'geojson', data: { type: 'Feature', geometry: F.grey, properties: {} } });
    map.addLayer({ id: 'grey-fill', type: 'fill', source: 'grey', paint: { 'fill-color': '#9aa0a6', 'fill-opacity': 0.35 } }); }
  map.addSource('front', { type: 'geojson', data: { type: 'Feature', geometry: F.line, properties: {} } });
  map.addLayer({ id: 'front-glow', type: 'line', source: 'front', paint: { 'line-color': '#ff3b3b', 'line-width': 9, 'line-opacity': 0.18, 'line-blur': 4 } });
  map.addLayer({ id: 'front-line', type: 'line', source: 'front', paint: { 'line-color': '#ff4040', 'line-width': 2.2 } });
  map.addSource('places', { type: 'geojson', data: fc(R.places.map(p => pt([p.lon, p.lat], { n: p.n, k: p.k, r: p.k === 'city' ? 3 : p.k === 'town' ? 2 : 1 }))) });
  map.addLayer({ id: 'places-dot', type: 'circle', source: 'places', filter: ['>=', ['get', 'r'], 2], paint: { 'circle-radius': 2.5, 'circle-color': '#d5e2ec' } });
  map.addLayer({ id: 'places-lbl', type: 'symbol', source: 'places', layout: { 'text-field': ['get', 'n'], 'text-font': ['Noto Sans Regular'], 'text-size': ['match', ['get', 'r'], 3, 14, 2, 12, 10.5], 'text-offset': [0, 0.9], 'symbol-sort-key': ['-', 0, ['get', 'r']] },
    minzoom: 0, paint: { 'text-color': '#e8f0f6', 'text-halo-color': '#000', 'text-halo-width': 1.4, 'text-opacity': ['step', ['zoom'], ['match', ['get', 'r'], 1, 0, 1], 11, 1] } });
  map.addSource('allied', { type: 'geojson', data: fc(R.allied.map(a => pt([a.lon, a.lat], a))) });
  map.addLayer({ id: 'allied-sym', type: 'symbol', source: 'allied', layout: { 'icon-image': ['match', ['get', 'kind'], 'depot', 'depot', 'friend'], 'icon-allow-overlap': true, 'icon-size': ['interpolate', ['linear'], ['zoom'], 8, 0.7, 13, 1.1] } });
  map.addLayer({ id: 'allied-lbl', type: 'symbol', source: 'allied', minzoom: 9.5, layout: { 'text-field': ['concat', ['get', 'id'], ' (FICTIF)'], 'text-font': ['Noto Sans Regular'], 'text-size': 10, 'text-offset': [0, 1.5] }, paint: { 'text-color': '#9fd3ff', 'text-halo-color': '#000', 'text-halo-width': 1.2 } });
  // mission
  const gj = (id, d = fc()) => map.addSource(id, { type: 'geojson', data: d });
  ['route', 'ab', 'tile', 'track-est', 'track-true', 'robot', 'robot-true', 'sigma', 'particles', 'scan', 'lpath'].forEach(id => gj(id));
  map.addLayer({ id: 'tile-line', type: 'line', source: 'tile', paint: { 'line-color': '#00e5ff', 'line-width': 1.2, 'line-dasharray': [2, 2], 'line-opacity': 0.8 } });
  map.addLayer({ id: 'route-casing', type: 'line', source: 'route', layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': '#002833', 'line-width': 7, 'line-opacity': 0.8 } });
  map.addLayer({ id: 'route', type: 'line', source: 'route', layout: { 'line-cap': 'round', 'line-join': 'round' }, paint: { 'line-color': '#00e5ff', 'line-width': 3, 'line-dasharray': [2, 1.2] } });
  map.addLayer({ id: 'lpath', type: 'line', source: 'lpath', paint: { 'line-color': '#ffffff', 'line-width': 1.5, 'line-dasharray': [1, 1], 'line-opacity': 0.85 } });
  map.addLayer({ id: 'track-true', type: 'line', source: 'track-true', layout: { visibility: 'none' }, paint: { 'line-color': '#22d39a', 'line-width': 2 } });
  map.addLayer({ id: 'track-est', type: 'line', source: 'track-est', paint: { 'line-color': '#00e5ff', 'line-width': 2.5 } });
  map.addLayer({ id: 'sigma-fill', type: 'fill', source: 'sigma', paint: { 'fill-color': '#00e5ff', 'fill-opacity': 0.12 } });
  map.addLayer({ id: 'sigma-line', type: 'line', source: 'sigma', paint: { 'line-color': '#00e5ff', 'line-width': 1, 'line-opacity': 0.7 } });
  map.addLayer({ id: 'particles', type: 'circle', source: 'particles', layout: { visibility: 'none' }, paint: { 'circle-radius': 1.3, 'circle-color': '#fff', 'circle-opacity': 0.6 } });
  map.addLayer({ id: 'scan', type: 'circle', source: 'scan', paint: { 'circle-radius': ['interpolate', ['linear'], ['zoom'], 14, 1, 19, 3], 'circle-color': '#ff3b8a' } });
  map.addLayer({ id: 'robot-true', type: 'symbol', source: 'robot-true', layout: { visibility: 'none', 'icon-image': 'ghost', 'icon-rotate': ['get', 'r'], 'icon-rotation-alignment': 'map', 'icon-allow-overlap': true } });
  map.addLayer({ id: 'robot', type: 'symbol', source: 'robot', layout: { 'icon-image': 'robot', 'icon-rotate': ['get', 'r'], 'icon-rotation-alignment': 'map', 'icon-allow-overlap': true, 'icon-size': 1.2 } });
  map.addLayer({ id: 'ab', type: 'symbol', source: 'ab', layout: { 'icon-image': ['get', 'k'], 'icon-allow-overlap': true } });
  buildLayerUI();
  map.fitBounds([[lo0, la0], [lo1, la1]], { padding: 30, duration: 0 });
  // listes A / B
  R.allied.forEach(a => {
    const o = new Option(`${a.id} — ${a.kind === 'depot' ? 'point logistique' : 'position avancée'} (${km(a.front_m)} de la ligne)`, a.id);
    (a.kind === 'depot' ? $('selA') : $('selB')).appendChild(o);
    if (a.kind !== 'depot') $('selA').appendChild(new Option(o.text, a.id));
  });
  const pickSel = (sel, k) => $(sel).onchange = e => { const a = R.allied.find(x => x.id === e.target.value); if (a) { setPoint(k, [a.lon, a.lat], a.label); map.flyTo({ center: [a.lon, a.lat], zoom: Math.max(map.getZoom(), 12) }); } };
  pickSel('selA', 'A'); pickSel('selB', 'B');
  dataTab(R);
  $('sb-status').textContent = `PRÊT — région ${R.summary.width_km} × ${R.summary.height_km} km, ligne de contact ${R.summary.front_length_km} km (situation ${R.summary.front_date})`;
}

function dataTab(R) {
  const s = R.summary, lc = Object.entries(s.landcover_fraction).sort((a, b) => b[1] - a[1]).map(([k, v]) => `<tr><td>${k}</td><td>${(v * 100).toFixed(1)} %</td></tr>`).join('');
  $('dataInfo').innerHTML = `<h4>RÉGION D'OPÉRATION</h4><table>
    <tr><td>Emprise</td><td>${s.width_km} × ${s.height_km} km</td></tr>
    <tr><td>Ligne de contact</td><td>${s.front_length_km} km</td></tr>
    <tr><td>Situation au</td><td>${s.front_date}</td></tr>
    <tr><td>Altitude</td><td>${s.dem_range_m[0]}–${s.dem_range_m[1]} m</td></tr>
    <tr><td>Bâtiments OSM</td><td>${s.n_buildings.toLocaleString('fr')}</td></tr>
    <tr><td>Localités</td><td>${s.n_places}</td></tr>
    <tr><td>Positions amies</td><td>${R.allied.length} <span class="warn">FICTIVES</span></td></tr></table>
    <h4>OCCUPATION DU SOL</h4><table>${lc}</table>
    <h4>JEUX DE DONNÉES</h4><table>
    <tr><td>ESA WorldCover 2021, 10 m</td><td>CC-BY 4.0</td></tr>
    <tr><td>Copernicus DEM GLO-30</td><td>Copernicus</td></tr>
    <tr><td>OpenStreetMap (Geofabrik)</td><td>ODbL</td></tr>
    <tr><td>Ligne de contact DeepStateMap</td><td>publique</td></tr>
    <tr><td>Imagerie Esri World Imagery</td><td>Esri</td></tr></table>
    <p class="warn" style="margin-top:10px">Positions amies, tranchée et unités : FICTIVES, générées pour la simulation. Les points « unités » de la source sont ignorés.</p>`;
}

/* ============================================================ points A / B, itinéraire */
document.querySelectorAll('.pick').forEach(b => b.onclick = () => {
  S.picking = S.picking === b.dataset.pick ? null : b.dataset.pick;
  document.querySelectorAll('.pick').forEach(x => x.classList.toggle('on', x.dataset.pick === S.picking));
  $('picking').classList.toggle('hidden', !S.picking); $('pickWhat').textContent = S.picking || '';
  map.getCanvas().style.cursor = S.picking ? 'crosshair' : '';
});
addEventListener('keydown', e => { if (e.key === 'Escape') { S.picking = null; document.querySelectorAll('.pick').forEach(x => x.classList.remove('on')); $('picking').classList.add('hidden'); map.getCanvas().style.cursor = ''; closeMax(); } });

async function pointInfo(lon, lat) { return (await fetch(`/api/point?lon=${lon}&lat=${lat}`)).json(); }
async function setPoint(k, ll, label) {
  const info = await pointInfo(...ll);
  if (info.status !== 'controlled') { alert(`Point refusé : zone ${info.status === 'occupied' ? 'occupée' : 'grise'}.`); return false; }
  S[k] = ll;
  $('c' + k).innerHTML = `${label ? `<b>${label}</b><br>` : ''}${fmtLL(...ll)}<br>${info.landcover} · alt. ${info.alt_m} m · ligne à ${km(info.front_m)}`;
  $('c' + k).classList.add('set');
  map.getSource('ab').setData(fc(['A', 'B'].filter(x => S[x]).map(x => pt(S[x], { k: x }))));
  if (S.A && S.B) computeRoute();
  return true;
}
map.on('click', async e => {
  const ll = [e.lngLat.lng, e.lngLat.lat];
  if (S.picking) {
    const k = S.picking; S.picking = null; document.querySelectorAll('.pick').forEach(x => x.classList.remove('on'));
    $('picking').classList.add('hidden'); map.getCanvas().style.cursor = '';
    await setPoint(k, ll); return;
  }
  const f = map.queryRenderedFeatures(e.point, { layers: ['allied-sym'] })[0];
  const info = await pointInfo(...ll);
  const html = `${f ? `<b>${f.properties.label}</b><br>` : ''}<span class="mono">${fmtLL(...ll)}</span><br>
    Statut : <b style="color:${info.status === 'controlled' ? '#22d39a' : '#ff3b3b'}">${{ controlled: 'contrôlé', occupied: 'OCCUPÉ', grey: 'ZONE GRISE' }[info.status]}</b><br>
    Ligne de contact : ${km(Math.abs(info.front_m))} · ${info.landcover} · ${info.alt_m} m<br>
    ${info.status === 'controlled' ? `<button class="btn" id="popA">Départ A</button><button class="btn" id="popB">Destination B</button>` : ''}`;
  const pop = new maplibregl.Popup({ closeButton: true, maxWidth: '300px' }).setLngLat(f ? f.geometry.coordinates : ll).setHTML(html).addTo(map);
  const at = f ? f.geometry.coordinates : ll, lab = f ? f.properties.label : undefined;
  setTimeout(() => { const a = $('popA'), b = $('popB'); if (a) a.onclick = () => { setPoint('A', at, lab); pop.remove(); }; if (b) b.onclick = () => { setPoint('B', at, lab); pop.remove(); }; }, 0);
});
let curTimer = null;
map.on('mousemove', e => {
  const { lng, lat } = e.lngLat;
  if (!MX) return;
  const [x, y] = toXY(lng, lat);
  $('cursor').textContent = `${fmtLL(lng, lat)} · X ${(x / 1000).toFixed(2)} km  Y ${(y / 1000).toFixed(2)} km · zoom ${map.getZoom().toFixed(1)}`;
});

async function computeRoute() {
  $('routeCard').classList.remove('hidden'); $('routeCard').innerHTML = 'Calcul de l\'itinéraire…';
  const r = await (await fetch('/api/route', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ a: S.A, b: S.B, profile: S.profile }) })).json();
  if (!r.ok) { $('routeCard').innerHTML = `<span class="err">✖ ${r.error}</span>`; S.route = null; $('bStart').disabled = true; map.getSource('route').setData(fc()); return; }
  S.route = r;
  map.getSource('route').setData(fc([ln(r.coords)]));
  const b = new maplibregl.LngLatBounds(); r.coords.forEach(c => b.extend(c)); map.fitBounds(b, { padding: 80, maxZoom: 15 });
  const warn = r.length_m > 3000 ? `<p class="warn">⚠ ${km(r.length_m)} à pied : ~${Math.round(r.eta_g1_min)} min pour le G1. En mode G1 physique, prévoir une longue simulation.</p>` : '';
  $('routeCard').innerHTML = `<table>
    <tr><td>Longueur</td><td>${km(r.length_m)} (vol d'oiseau ${km(r.straight_m)})</td></tr>
    <tr><td>Durée estimée (G1, 0,45 m/s)</td><td>${fmtT(r.eta_g1_min * 60)}</td></tr>
    <tr><td>Sous couvert végétal</td><td>${r.cover_pct} %</td></tr>
    <tr><td>Terrain découvert</td><td>${r.open_pct} %</td></tr>
    <tr><td>Distance minimale à la ligne</td><td>${km(r.front_min_m)}</td></tr>
    <tr><td>Dénivelé positif</td><td>${r.climb_m} m</td></tr>
    <tr><td>Calcul</td><td>${r.compute_s} s</td></tr></table>${warn}`;
  $('bStart').disabled = false;
}
$('bRoute').onclick = () => { if (S.A && S.B) computeRoute(); else alert('Définir A et B.'); };

/* ============================================================ mission */
async function ctl(action, value) { if (!S.mission) return; const r = await fetch('/api/mission/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action, value }) }); return r.json(); }
$('bStart').onclick = async () => {
  if (!S.route) return;
  const bLab = $('selB').value || '';
  const body = { xy: S.route.xy, mode: S.mode, level: S.level, delivery: $('delivery').checked, payload_kg: +$('payload').value, speed: +$('speed').value, seed: Math.floor(Math.random() * 1000), target: bLab };
  const r = await fetch('/api/mission/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) { alert((await r.json()).detail); return; }
  S.mission = true; resetMission();
  ['bPause', 'bStop', 'bReport'].forEach(id => $(id).disabled = false);
  setModeUI(S.mode);
};
function setModeUI(mode) {
  document.querySelectorAll('#mode button').forEach(x => x.classList.toggle('on', x.dataset.v === mode)); S.mode = mode;
  const phys = mode === 'g1' || mode === 'go1';
  document.body.classList.toggle('g1', phys);
  $('b-mode').textContent = { g1: 'G1 · MUJOCO', go1: 'GO1 · MUJOCO' }[mode] || 'SIM RAPIDE'; $('b-mode').className = 'badge ok';
  $('mjimg').style.display = phys ? 'block' : 'none'; $('mjoff').style.display = phys ? 'none' : 'grid';
  if (phys && !$('mjimg').src.includes('mjpg')) $('mjimg').src = '/api/mujoco.mjpg?' + Date.now();
}
$('speed').onchange = e => ctl('speed', +e.target.value);
$('bPause').onclick = async () => { const r = await ctl('pause'); $('bPause').innerHTML = r && r.paused ? '▶ Reprendre' : '❚❚ Pause'; };
$('bStop').onclick = () => ctl('stop');
$('bReport').onclick = async () => {
  const r = await (await fetch('/api/mission/report')).json();
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(r, null, 1)], { type: 'application/json' }));
  a.download = `talos_mission_${new Date().toISOString().slice(0, 19).replace(/:/g, '')}.json`; a.click();
};

function resetMission() {
  Object.assign(S, { trackE: [], trackT: [], particles: [], scan: [], path: [], err: [], events: [], tile: null, state: null, vox: 0 });
  $('log').innerHTML = ''; V3.clear(); T2.img = null;
  ['track-est', 'track-true', 'robot', 'robot-true', 'sigma', 'particles', 'scan', 'lpath', 'tile'].forEach(s => map.getSource(s) && map.getSource(s).setData(fc()));
}

/* ---------- liaison temps réel ---------- */
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { $('b-link').textContent = '● LIAISON'; $('b-link').className = 'badge ok'; };
  ws.onclose = () => { $('b-link').textContent = '○ LIAISON PERDUE'; $('b-link').className = 'badge red'; setTimeout(connect, 1500); };
  ws.onmessage = e => { const m = JSON.parse(e.data); if (m.type === 'reset') { resetMission(); return; } if (m.type === 'tick') tick(m); };
}
let lastFollow = 0;
function tick(m) {
  S.mission = true;
  const st = m.state; S.state = st;
  if (m.route) {
    if (!S.route) S.route = { coords: m.route.coords, xy: m.route.xy };
    map.getSource('route').setData(fc([ln(m.route.coords)]));
    const c = m.route.coords; map.getSource('ab').setData(fc([pt(c[0], { k: 'A' }), pt(c[c.length - 1], { k: 'B' })]));
    setModeUI(m.route.mode);
  }
  if (m.tile) { S.tile = m.tile; const [a, b, c, d] = m.tile.bounds;
    map.getSource('tile').setData(fc([ln([[a, b], [c, b], [c, d], [a, d], [a, b]])])); T2.loadImg(); T2.fit(); }
  m.track.forEach(r => { S.trackE.push([r[0], r[1]]); S.trackT.push([r[2], r[3]]); });
  m.err.forEach(r => S.err.push(r));
  S.particles = m.particles; S.scan = m.scan; S.path = m.path;
  if (m.vox.length) V3.add(m.vox);
  m.events.forEach(addLog);
  if (st.est) {
    map.getSource('track-est').setData(fc([ln(S.trackE)])); map.getSource('track-true').setData(fc([ln(S.trackT)]));
    const rot = e => 90 - e * 180 / Math.PI;
    map.getSource('robot').setData(fc([pt(st.est, { r: rot(st.eyaw) })]));
    map.getSource('robot-true').setData(fc([pt(st.true, { r: rot(st.yaw) })]));
    map.getSource('sigma').setData(fc([circle(st.est[0], st.est[1], Math.max(2 * st.sigma, 1.5))]));
    map.getSource('particles').setData(fc(S.particles.map(p => pt(toLL(...p)))));
    map.getSource('scan').setData(fc(S.scan.map(p => pt(toLL(...p)))));
    map.getSource('lpath').setData(fc(S.path.length ? [ln(S.path.map(p => toLL(...p)))] : []));
    if ($('follow3d').checked && performance.now() - lastFollow > 3000 && map.getZoom() > 13) { lastFollow = performance.now(); if (!map.getBounds().contains(st.est)) map.easeTo({ center: st.est }); }
  }
  kpis(st); chart(); T2.draw(); V3.robot(st); renderLive(false);
  $('simt').textContent = 'T+' + fmtT(st.t || 0);
  const ph = st.phase || 'NAV', tag = $('phase');
  tag.textContent = st.final ? st.status.toUpperCase() : (st.status && !String(st.status).startsWith('en cours') && !st.est ? st.status.toUpperCase() : PHASE[ph] || ph);
  tag.className = 'tag ' + (st.final ? (/accompli|arriv/.test(st.status) ? 'ok' : 'bad') : ph === 'NAV' ? 'nav' : 'dl');
  $('sb-status').textContent = st.est ? `${(PHASE[ph] || ph)} · tronçon ${st.leg}/${st.legs} · reste ${km(st.remaining_m)} · ETA ${fmtT(st.eta_s)} · ${{ g1: 'G1 MuJoCo', go1: 'Go1 MuJoCo' }[st.mode] || 'sim rapide'} · carte ${st.level} · calcul ${fmtT(st.wall_s)} (×${(st.t / Math.max(1, st.wall_s)).toFixed(1)} temps réel)${st.final ? ' · FIN : ' + st.status : ''}`
    : String(st.status || '').toUpperCase();
}
function addLog(e) {
  S.events.push(e);
  const f = $('logf').value; if (f && e.type !== f) return;
  const d = document.createElement('div'); d.className = e.type;
  d.innerHTML = `<time>${fmtT(e.t)}</time>${e.text}`;
  if (e.lon) { d.style.cursor = 'pointer'; d.onclick = () => map.flyTo({ center: [e.lon, e.lat], zoom: 17 }); }
  $('log').prepend(d);
  while ($('log').childNodes.length > 400) $('log').lastChild.remove();
}
$('logf').onchange = () => { $('log').innerHTML = ''; const ev = S.events; S.events = []; ev.forEach(addLog); };

function kpis(s) {
  if (!s.est) { $('kpis').innerHTML = ''; return; }
  const k = (v, l, cls = '') => `<div class="k ${cls}"><b>${v}</b><span>${l}</span></div>`;
  $('kpis').innerHTML =
    k(km(s.remaining_m), 'reste') + k(fmtT(s.eta_s), 'ETA') + k('±' + s.sigma.toFixed(1) + ' m', 'incertitude', s.sigma > 3 ? 'warn' : '') +
    k(s.err.toFixed(1) + ' m', 'erreur vraie (sim)', s.err > 5 ? 'bad' : s.err < 2 ? 'good' : 'warn') +
    k(s.v.toFixed(2) + ' m/s', 'vitesse') + k(s.battery_pct.toFixed(0) + ' %', 'batterie', s.battery_pct < 25 ? 'bad' : '') +
    k('+' + s.dT.toFixed(1) + ' K', `thermique · ${km(s.detect_m)}`, s.dT > 15 ? 'warn' : '') + k(s.payload.toFixed(1) + ' kg', 'charge') +
    k(km(s.front_m), 'ligne de contact', s.front_m < 2000 ? 'bad' : s.front_m < 4000 ? 'warn' : '') + k(`${s.leg}/${s.legs}`, 'tronçon') +
    k(s.contacts, 'contacts', s.contacts ? 'warn' : '') + k(s.replans, 'replanif.');
}
function fitCanvas(cv) { const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1; if (cv.width !== Math.round(r.width * d) || cv.height !== Math.round(r.height * d)) { cv.width = r.width * d; cv.height = r.height * d; } const g = cv.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0); return [g, r.width, r.height]; }
function chart() {
  const [g, w, h] = fitCanvas($('chart')); g.clearRect(0, 0, w, h);
  const E = S.err; if (E.length < 2) return;
  const t1 = E[E.length - 1][0], t0 = Math.max(0, t1 - 900), D = E.filter(r => r[0] >= t0);
  const emax = Math.max(4, ...D.map(r => Math.max(r[1], 2 * r[2]))) * 1.1;
  const X = t => 30 + (t - t0) / Math.max(1, t1 - t0) * (w - 36), Y = v => h - 12 - v / emax * (h - 22);
  g.strokeStyle = '#1d2a37'; g.fillStyle = '#566b7d'; g.font = '9px Consolas'; g.lineWidth = 1;
  [0, emax / 2, emax].forEach(v => { g.beginPath(); g.moveTo(30, Y(v)); g.lineTo(w - 6, Y(v)); g.stroke(); g.fillText(v.toFixed(0) + 'm', 2, Y(v) + 3); });
  const line = (i, col, f = v => v) => { g.strokeStyle = col; g.lineWidth = 1.4; g.beginPath(); D.forEach((r, j) => j ? g.lineTo(X(r[0]), Y(f(r[i]))) : g.moveTo(X(r[0]), Y(f(r[i])))); g.stroke(); };
  g.fillStyle = 'rgba(0,229,255,.08)'; g.beginPath(); D.forEach((r, j) => j ? g.lineTo(X(r[0]), Y(2 * r[2])) : g.moveTo(X(r[0]), Y(2 * r[2]))); g.lineTo(X(t1), Y(0)); g.lineTo(X(D[0][0]), Y(0)); g.fill();
  line(1, '#00e5ff'); line(3, 'rgba(34,211,154,.8)', v => v * emax / 1.2);
  g.font = '9.5px Segoe UI'; g.fillStyle = '#00e5ff'; g.fillText('erreur de position', 36, 10); g.fillStyle = '#7d93a6'; g.fillText('zone : ±2σ estimé', 140, 10); g.fillStyle = '#22d39a'; g.fillText('vitesse', 245, 10);
}

/* ============================================================ vue tactique 2D */
const T2 = {
  cv: $('c2d'), cx: 0, cy: 0, sc: 1.5, img: null, drag: null, follow: true,
  loadImg() { const im = new Image(); im.onload = () => { this.img = im; this.draw(); }; im.src = '/api/mission/tile.jpg?' + Date.now(); },
  fit() { if (!S.tile) return; const r = this.cv.getBoundingClientRect(); this.sc = Math.min(r.width, r.height) / 140; this.follow = true; },
  draw() {
    const [g, w, h] = fitCanvas(this.cv); g.fillStyle = '#05080b'; g.fillRect(0, 0, w, h);
    const st = S.state; if (!S.tile || !st || !st.est_xy) return;
    if (this.follow) { this.cx = st.est_xy[0]; this.cy = st.est_xy[1]; }
    const sc = this.sc, X = x => w / 2 + (x - this.cx) * sc, Y = y => h / 2 - (y - this.cy) * sc, T = S.tile;
    if (this.img && $('t2img').checked) { g.globalAlpha = 0.9; g.drawImage(this.img, X(T.x0), Y(T.y0 + T.size), T.size * sc, T.size * sc); g.globalAlpha = 1; }
    g.strokeStyle = 'rgba(0,229,255,.5)'; g.setLineDash([4, 4]); g.strokeRect(X(T.x0), Y(T.y0 + T.size), T.size * sc, T.size * sc); g.setLineDash([]);
    // grille 10 m
    g.strokeStyle = 'rgba(255,255,255,.05)'; g.lineWidth = 1; const step = sc > 6 ? 5 : sc > 2 ? 10 : 50;
    for (let x = Math.floor((this.cx - w / 2 / sc) / step) * step; x < this.cx + w / 2 / sc; x += step) { g.beginPath(); g.moveTo(X(x), 0); g.lineTo(X(x), h); g.stroke(); }
    for (let y = Math.floor((this.cy - h / 2 / sc) / step) * step; y < this.cy + h / 2 / sc; y += step) { g.beginPath(); g.moveTo(0, Y(y)); g.lineTo(w, Y(y)); g.stroke(); }
    if ($('t2truth').checked) T.obst.forEach(o => { const x = X(o[0]), y = Y(o[1]); if (x < -5 || y < -5 || x > w + 5 || y > h + 5) return; g.strokeStyle = o[3] ? '#ff5a5a' : 'rgba(34,211,154,.8)'; g.beginPath(); g.arc(x, y, Math.max(1, o[2] * sc), 0, 7); g.stroke(); });
    if (T.trench && T.trench.length) { g.strokeStyle = '#ffb020'; g.lineWidth = 3; g.beginPath(); T.trench.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))); g.stroke(); g.lineWidth = 1; }
    if (S.route) { g.strokeStyle = 'rgba(0,229,255,.5)'; g.setLineDash([6, 4]); g.lineWidth = 1.5; g.beginPath(); S.route.xy.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))); g.stroke(); g.setLineDash([]); }
    if (S.path.length) { g.strokeStyle = '#fff'; g.setLineDash([2, 3]); g.lineWidth = 1.3; g.beginPath(); S.path.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))); g.stroke(); g.setLineDash([]); }
    const trk = (arr, col) => { g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); arr.forEach((ll, i) => { const [x, y] = toXY(...ll); i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)); }); g.stroke(); };
    if ($('t2truth').checked) trk(S.trackT.slice(-3000), 'rgba(34,211,154,.9)');
    trk(S.trackE.slice(-3000), '#00e5ff');
    if ($('t2part').checked) { g.fillStyle = 'rgba(255,255,255,.55)'; S.particles.forEach(p => g.fillRect(X(p[0]) - 1, Y(p[1]) - 1, 2, 2)); }
    g.fillStyle = '#ff3b8a'; S.scan.forEach(p => g.fillRect(X(p[0]) - 1.5, Y(p[1]) - 1.5, 3, 3));
    const robot = (x, y, yaw, col, fill) => { g.save(); g.translate(X(x), Y(y)); g.rotate(-yaw); g.beginPath(); g.moveTo(11, 0); g.lineTo(-7, -6); g.lineTo(-4, 0); g.lineTo(-7, 6); g.closePath(); if (fill) { g.fillStyle = col; g.fill(); } else { g.strokeStyle = col; g.lineWidth = 2; g.stroke(); } g.restore(); };
    g.strokeStyle = 'rgba(0,229,255,.7)'; g.beginPath(); g.arc(X(st.est_xy[0]), Y(st.est_xy[1]), Math.max(4, 2 * st.sigma * sc), 0, 7); g.stroke();
    if ($('t2truth').checked) robot(st.true_xy[0], st.true_xy[1], st.yaw, '#22d39a', false);
    robot(st.est_xy[0], st.est_xy[1], st.eyaw, '#00e5ff', true);
    // barre d'echelle
    const sb = step * (sc * step < 40 ? 4 : 2); g.strokeStyle = '#d5e2ec'; g.lineWidth = 2; g.beginPath(); g.moveTo(w - 16 - sb * sc, h - 14); g.lineTo(w - 16, h - 14); g.stroke();
    g.fillStyle = '#d5e2ec'; g.font = '10px Consolas'; g.fillText(sb + ' m', w - 16 - sb * sc, h - 19);
    $('i2d').textContent = `tuile ${T.k} · ${T.imagery} · ${T.n_trees} troncs · ${T.n_walls} murs · ${T.n_craters} cratères (${T.craters_per_ha}/ha) · ligne ${km(T.front_m)}`;
  }
};
T2.cv.addEventListener('wheel', e => { e.preventDefault(); T2.sc = Math.max(0.3, Math.min(40, T2.sc * (e.deltaY > 0 ? 0.87 : 1.15))); T2.draw(); }, { passive: false });
T2.cv.addEventListener('pointerdown', e => { T2.drag = [e.clientX, e.clientY]; T2.follow = false; T2.cv.setPointerCapture(e.pointerId); });
T2.cv.addEventListener('pointermove', e => { if (!T2.drag) return; T2.cx -= (e.clientX - T2.drag[0]) / T2.sc; T2.cy += (e.clientY - T2.drag[1]) / T2.sc; T2.drag = [e.clientX, e.clientY]; T2.draw(); });
T2.cv.addEventListener('pointerup', () => T2.drag = null);
T2.cv.addEventListener('dblclick', () => { T2.follow = true; T2.draw(); });
['t2truth', 't2part', 't2img'].forEach(id => $(id).onchange = () => T2.draw());

/* ============================================================ carte 3D (three.js) */
const V3 = (() => {
  const cv = $('c3d');
  const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
  renderer.setClearColor(0x05080b);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x05080b, 120, 420);
  const cam = new THREE.PerspectiveCamera(55, 1, 0.5, 3000);
  cam.position.set(-40, 45, 40);
  const ctrl = new OrbitControls(cam, cv); ctrl.enableDamping = true; ctrl.maxPolarAngle = Math.PI * 0.49;
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const grid = new THREE.GridHelper(400, 40, 0x1d2a37, 0x111a22); scene.add(grid);
  const MAXV = 700000;
  const mk = size => { const geo = new THREE.BufferGeometry(); const pos = new Float32Array(MAXV * 3), col = new Float32Array(MAXV * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(col, 3)); geo.setDrawRange(0, 0);
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true })); scene.add(pts); return { geo, pos, col, n: 0, pts }; };
  const ground = mk(0.55), obst = mk(0.5);
  const robotM = new THREE.Mesh(new THREE.ConeGeometry(0.6, 1.8, 12), new THREE.MeshBasicMaterial({ color: 0x00e5ff }));
  robotM.rotation.x = Math.PI / 2; const robotG = new THREE.Group(); robotG.add(robotM); scene.add(robotG);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1.1, 32), new THREE.MeshBasicMaterial({ color: 0x00e5ff, side: THREE.DoubleSide, transparent: true, opacity: 0.6 }));
  ring.rotation.x = -Math.PI / 2; scene.add(ring);
  const pathGeo = new THREE.BufferGeometry(); const pathLine = new THREE.Line(pathGeo, new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 1, gapSize: 0.8 })); scene.add(pathLine);
  let O = null, zmin = null, zmax = null, n = 0, gz = 0;
  const hz = new THREE.Color();
  function add(list) {
    if (!O) { O = [list[0][0], list[0][1], list[0][2]]; }
    for (const v of list) {
      const T = v[3] ? obst : ground; if (T.n >= MAXV) continue;
      const i = T.n * 3, z = v[2] - O[2];
      T.pos[i] = v[0] - O[0]; T.pos[i + 1] = z; T.pos[i + 2] = -(v[1] - O[1]);
      if (v[3]) { const a = Math.min(1, Math.max(0, z / 10)); hz.setRGB(1, 0.55 - 0.3 * a, 0.15 + 0.2 * a); }
      else { gz = gz * 0.995 + z * 0.005; if (zmin === null) { gz = z; }
        if (zmin === null) { zmin = z - 2; zmax = z + 2; } const a = Math.min(1, Math.max(0, (z - zmin) / Math.max(1, zmax - zmin))); hz.setHSL(0.55 - 0.35 * a, 0.65, 0.32 + 0.18 * a); }
      T.col[i] = hz.r; T.col[i + 1] = hz.g; T.col[i + 2] = hz.b; T.n++;
    }
    [ground, obst].forEach(T => { T.geo.setDrawRange(0, T.n); T.geo.attributes.position.needsUpdate = true; T.geo.attributes.color.needsUpdate = true; T.geo.computeBoundingSphere(); });
    n = ground.n + obst.n;
    $('i3d').textContent = `${n.toLocaleString('fr')} voxels · obstacles ${obst.n.toLocaleString('fr')}`;
    $('i3d').title = 'glisser = tourner · clic droit = déplacer · molette = zoom';
  }
  function clear() { [ground, obst].forEach(T => { T.n = 0; T.geo.setDrawRange(0, 0); }); O = null; zmin = zmax = null; gz = 0; }
  let last = null;
  function robot(st) {
    if (!O || !st.est_xy) return;
    const x = st.est_xy[0] - O[0], z = -(st.est_xy[1] - O[1]);
    const y = gz;
    robotG.position.set(x, y + 1.2, z); robotG.rotation.set(0, st.eyaw, 0); robotM.rotation.set(0, 0, -Math.PI / 2);
    ring.position.set(x, y + 0.1, z); ring.scale.setScalar(Math.max(1, 2 * st.sigma));
    if (S.path.length) { const P = S.path.map(p => new THREE.Vector3(p[0] - O[0], y + 0.3, -(p[1] - O[1]))); pathGeo.setFromPoints(P); pathLine.computeLineDistances(); }
    grid.position.set(x - (x % 10), y - 0.5, z - (z % 10));
    if ($('follow3d').checked) { const tgt = new THREE.Vector3(x, y, z); const d = tgt.clone().sub(ctrl.target); ctrl.target.add(d); cam.position.add(d); }
    last = { x, y, z };
  }
  // altitude du robot : approx. = plus proche voxel de sol
  function loop() {
    const r = cv.getBoundingClientRect();
    if (r.width && (cv.width !== Math.round(r.width * devicePixelRatio) || cv.height !== Math.round(r.height * devicePixelRatio))) {
      renderer.setPixelRatio(devicePixelRatio); renderer.setSize(r.width, r.height, false); cam.aspect = r.width / r.height; cam.updateProjectionMatrix();
    }
    ground.pts.visible = $('ground3d').checked;
    ctrl.update(); renderer.render(scene, cam); requestAnimationFrame(loop);
  }
  loop();
  return { add, clear, robot };
})();

/* ============================================================ panneaux agrandis */
let maxed = null;
document.querySelectorAll('.mx').forEach(b => b.onclick = () => {
  const pane = $(b.dataset.mx), body = pane.querySelector('.pb');
  if (maxed) closeMax();
  maxed = { pane, body };
  $('mxTitle').textContent = pane.querySelector('.ph span').textContent;
  const head = pane.querySelector('.ph'); maxed.ctrls = [...head.children].filter(c => c.tagName !== 'SPAN' && !c.classList.contains('mx'));
  maxed.ctrls.forEach(c => $('maxi').querySelector('.mx-head').insertBefore(c, $('mxClose')));
  $('mxBody').appendChild(body); $('maxi').classList.remove('hidden');
  setTimeout(() => { T2.fit(); T2.draw(); }, 50);
});
function closeMax() {
  if (!maxed) return;
  const head = maxed.pane.querySelector('.ph');
  maxed.ctrls.forEach(c => head.insertBefore(c, head.querySelector('.mx')));
  maxed.pane.appendChild(maxed.body); $('maxi').classList.add('hidden'); maxed = null;
  setTimeout(() => { T2.fit(); T2.draw(); }, 50);
}
$('mxClose').onclick = closeMax;
addEventListener('resize', () => { T2.draw(); chart(); });

map.on('load', async () => { await loadRegion(); connect(); });

/* ============================================================ VUES (onglets) */
const SLOTS = { robotBody: $('robotBody'), body3d: $('body3d'), body2d: $('body2d') };
const HOME = {}; Object.keys(SLOTS).forEach(k => HOME[k] = SLOTS[k].parentElement);
let curView = 'v-op';
function showView(id) {
  closeMax(); curView = id;
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('on', v.id === id));
  document.querySelectorAll('#views button').forEach(b => b.classList.toggle('on', b.dataset.view === id));
  Object.keys(SLOTS).forEach(k => { const slot = document.querySelector(`#${id} .slot[data-slot=${k}]`); (slot || HOME[k]).appendChild(SLOTS[k]); });
  if (id === 'v-op') setTimeout(() => map.resize(), 30);
  if (id === 'v-ana') renderAna();
  if (id === 'v-bench') renderBench();
  if (id === 'v-rl') renderRL();
  if (id === 'v-data') renderData();
  if (id === 'v-live') renderLive(true);
  setTimeout(() => { T2.fit(); T2.draw(); }, 60);
}
document.querySelectorAll('#views button').forEach(b => b.onclick = () => showView(b.dataset.view));
seg('cam2', v => ctl('camera', v));

/* ---------- graphiques ---------- */
Chart.defaults.color = '#8fa5b8'; Chart.defaults.borderColor = '#1a2632';
Chart.defaults.font.family = 'Segoe UI, system-ui, sans-serif'; Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.labels.boxWidth = 12; Chart.defaults.animation = false;
Chart.defaults.maintainAspectRatio = false;
const PAL = ['#00e5ff', '#22d39a', '#ffb020', '#ff3b8a', '#9b7bff', '#3fa9ff', '#ff7a45', '#c9d65a'];
const CH = {};
function chartOn(id, cfg) { if (CH[id]) CH[id].destroy(); CH[id] = new Chart($(id), cfg); return CH[id]; }
const lineDS = (label, data, col, extra = {}) => ({ label, data, borderColor: col, backgroundColor: col + '22', pointRadius: 0, borderWidth: 1.6, tension: 0.15, ...extra });
const xyAxes = (xt, yt, more = {}) => ({ x: { type: 'linear', title: { display: !!xt, text: xt }, grid: { color: '#141e28' } }, y: { title: { display: !!yt, text: yt }, grid: { color: '#141e28' }, ...more } });
function wilson(k, n) { if (!n) return [0, 0]; const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [Math.max(0, c - h), Math.min(1, c + h)]; }
function successChart(id, labels, ks, ns, colors) {
  const pct = ks.map((k, i) => ns[i] ? 100 * k / ns[i] : 0), ci = ks.map((k, i) => wilson(k, ns[i]).map(v => 100 * v));
  return chartOn(id, { type: 'bar', data: { labels, datasets: [
    { label: 'réussite (%)', data: pct, backgroundColor: (colors || labels.map((_, i) => PAL[i % PAL.length])).map(c => c + 'cc'), borderRadius: 3, order: 2 },
    { label: 'IC 95 %', data: ci, type: 'bar', backgroundColor: 'rgba(232,240,246,.18)', borderColor: '#e8f0f6', borderWidth: 1, barPercentage: 0.15, order: 1 } ] },
    options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => c.datasetIndex ? `IC 95 % : ${c.raw[0].toFixed(0)}–${c.raw[1].toFixed(0)} %` : `${c.raw.toFixed(0)} % (${ks[c.dataIndex]}/${ns[c.dataIndex]})` } } },
      scales: { y: { min: 0, max: 100, title: { display: true, text: 'missions réussies (%)' }, grid: { color: '#141e28' } }, x: { stacked: true, grid: { display: false } } } } });
}

/* ---------- ROBOT EN DIRECT ---------- */
let lastLive = 0;
function renderLive(force) {
  if (curView !== 'v-live' || (!force && performance.now() - lastLive < 1000)) return; lastLive = performance.now();
  $('kpisBig').innerHTML = $('kpis').innerHTML;
  const E = S.err.slice(-900);
  chartOn('lc-err', { type: 'line', data: { datasets: [lineDS('erreur vraie', E.map(r => ({ x: r[0], y: r[1] })), PAL[0]), lineDS('±2σ estimé', E.map(r => ({ x: r[0], y: 2 * r[2] })), '#7d93a6', { fill: true, borderDash: [4, 3] })] }, options: { scales: xyAxes('t (s)', 'm', { min: 0 }) } });
  chartOn('lc-v', { type: 'line', data: { datasets: [lineDS('vitesse', E.map(r => ({ x: r[0], y: r[3] })), PAL[1], { fill: true })] }, options: { scales: xyAxes('t (s)', 'm/s', { min: 0 }) } });
}

/* ---------- ANALYSE MISSION ---------- */
$('anaRefresh').onclick = () => renderAna();
function renderAna() {
  const E = S.err, st = S.state || {};
  $('anaSub').textContent = E.length ? `${E.length} points · ${fmtT(E[E.length - 1][0])} de mission · ${S.events.length} événements` : 'aucune mission';
  if (!E.length) { $('anaCards').innerHTML = ''; return; }
  const errs = E.map(r => r[1]), mean = errs.reduce((a, b) => a + b, 0) / errs.length, p95 = [...errs].sort((a, b) => a - b)[Math.floor(0.95 * (errs.length - 1))];
  const cnt = t => S.events.filter(e => e.type === t).length, rej = S.events.filter(e => /incoherent/.test(e.text)).length;
  const card = (v, l, c = '') => `<div class="cardk ${c}"><b>${v}</b><span>${l}</span></div>`;
  $('anaCards').innerHTML = card(km(st.dist_m || 0), 'distance parcourue') + card(fmtT(E[E.length - 1][0]), 'durée') +
    card(mean.toFixed(1) + ' m', 'erreur moyenne', mean < 3 ? 'good' : mean < 8 ? 'warn' : 'bad') + card(p95.toFixed(1) + ' m', 'erreur p95', p95 < 5 ? 'good' : 'warn') +
    card((st.energy_wh || 0).toFixed(0) + ' Wh', 'énergie') + card(st.contacts || 0, 'contacts', st.contacts ? 'warn' : 'good') +
    card(cnt('replan'), 'replanifications') + card(cnt('anomaly'), 'anomalies') + card(rej, 'recalages rejetés', rej ? 'warn' : '') +
    card(st.final ? st.status : (PHASE[st.phase] || '—'), 'état', st.final ? (/accompli|arriv/.test(st.status) ? 'good' : 'bad') : '');
  const P = (i, f = v => v) => E.map(r => ({ x: r[0], y: f(r[i]) }));
  chartOn('a-err', { type: 'line', data: { datasets: [lineDS('erreur vraie (m)', P(1), PAL[0]), lineDS('±2σ estimé (m)', P(2, v => 2 * v), '#7d93a6', { fill: true, borderDash: [4, 3] })] }, options: { scales: xyAxes('t (s)', 'm', { min: 0 }) } });
  chartOn('a-v', { type: 'line', data: { datasets: [lineDS('vitesse (m/s)', P(3), PAL[1], { fill: true })] }, options: { scales: xyAxes('t (s)', 'm/s', { min: 0 }) } });
  if (E[0].length > 5) {
    chartOn('a-bat', { type: 'line', data: { datasets: [lineDS('batterie (%)', P(4), PAL[1]), lineDS('signature thermique (K)', P(5), PAL[2], { yAxisID: 'y2' })] }, options: { scales: { ...xyAxes('t (s)', 'batterie %', { min: 0, max: 100 }), y2: { position: 'right', title: { display: true, text: 'ΔT (K)' }, grid: { display: false } } } } });
    chartOn('a-front', { type: 'line', data: { datasets: [lineDS('distance à la ligne (km)', P(6, v => v / 1000), PAL[3]), lineDS('distance restante (km)', P(7, v => v / 1000), PAL[0])] }, options: { scales: xyAxes('t (s)', 'km', { min: 0 }) } });
    chartOn('a-slope', { type: 'line', data: { datasets: [lineDS('pente (%)', P(9, v => 100 * v), PAL[4], { fill: true })] }, options: { scales: xyAxes('t (s)', '%', { min: 0 }) } });
  }
  const types = {}; S.events.forEach(e => types[e.type] = (types[e.type] || 0) + 1);
  const ks = Object.keys(types);
  chartOn('a-ev', { type: 'bar', data: { labels: ks, datasets: [{ data: ks.map(k => types[k]), backgroundColor: ks.map((_, i) => PAL[i % PAL.length] + 'cc'), borderRadius: 3 }] }, options: { indexAxis: 'y', plugins: { legend: { display: false } } } });
}

/* ---------- BENCHMARKS ---------- */
let BENCH = null;
async function renderBench() {
  BENCH = await (await fetch('/api/benchmarks')).json();
  const B = BENCH, body = $('benchBody'); body.innerHTML = '';
  Object.keys(CH).filter(k => k.startsWith('bc')).forEach(k => { CH[k].destroy(); delete CH[k]; });
  const sec = (t, sub = '') => { const d = document.createElement('div'); d.className = 'sec'; d.innerHTML = t + (sub ? `<small>${sub}</small>` : ''); body.appendChild(d); };
  const grid = () => { const g = document.createElement('div'); g.className = 'grid2c'; body.appendChild(g); return g; };
  const pane = (g, title, h = '') => { const p = document.createElement('section'); p.className = 'pane'; const id = 'bc' + Math.random().toString(36).slice(2, 8);
    p.innerHTML = `<div class="ph"><span>${title}</span></div>${h || `<div class="cbox"><canvas id="${id}"></canvas></div>`}`; g.appendChild(p); return id; };
  const O = o => o || {};
  if (B.slopes) {
    sec('Robots comparés — marche sur pente', B.slopes_note || 'politiques évaluées sous MuJoCo (CPU), même protocole pour tous');
    const g = grid();
    const ids = pane(g, 'VITESSE RÉELLE SELON LA PENTE (m/s), consigne 0,5 m/s'), idf = pane(g, 'TAUX DE CHUTE SELON LA PENTE (%)');
    const rows = B.slopes.rows || B.slopes;
    const robots = [...new Set(rows.map(r => r.robot))];
    const slopes = [...new Set(rows.map(r => r.slope_deg))].sort((a, b) => a - b);
    const agg = (rb, s, f) => { const R = rows.filter(r => r.robot === rb && r.slope_deg === s); return R.length ? f(R) : null; };
    chartOn(ids, { type: 'line', data: { labels: slopes, datasets: robots.map((rb, i) => lineDS(rb, slopes.map(s => agg(rb, s, R => R.reduce((a, r) => a + (r.fell ? 0 : r.speed), 0) / R.length)), PAL[i], { pointRadius: 3 })) }, options: { scales: { x: { title: { display: true, text: 'pente (°)' } }, y: { min: 0, title: { display: true, text: 'm/s' } } } } });
    chartOn(idf, { type: 'bar', data: { labels: slopes, datasets: robots.map((rb, i) => ({ label: rb, data: slopes.map(s => agg(rb, s, R => 100 * R.filter(r => r.fell).length / R.length)), backgroundColor: PAL[i] + 'cc' })) }, options: { scales: { x: { title: { display: true, text: 'pente (°)' } }, y: { min: 0, max: 100, title: { display: true, text: '% de chutes' } } } } });
  }
  if (B.robots) {
    sec('Robots comparés — terrain accidenté', B.robots.note || '');
    const g = grid(), rows = B.robots.summary;
    successChart(pane(g, 'ESSAIS RÉUSSIS PAR ROBOT (sans chute, objectif atteint)'), rows.map(r => r.label), rows.map(r => r.success), rows.map(r => r.n));
    chartOn(pane(g, 'VITESSE MOYENNE ET CHUTES'), { type: 'bar', data: { labels: rows.map(r => r.label), datasets: [{ label: 'vitesse moyenne (m/s)', data: rows.map(r => r.speed_mean), backgroundColor: PAL[0] + 'aa' }, { label: 'chutes', data: rows.map(r => r.falls), backgroundColor: PAL[3] + 'cc', yAxisID: 'y2' }] }, options: { scales: { y: { min: 0, title: { display: true, text: 'm/s' } }, y2: { position: 'right', min: 0, grid: { display: false }, title: { display: true, text: 'chutes' } } } } });
  }
  if (B.nav_robots) {
    sec('Robots comparés — missions complètes sous MuJoCo', 'graines 100–115, recalage LiDAR 2D, dépôt à la tranchée ; relief ×1,0 = relief réel du monde, ×0,4 = atténué');
    const g = grid(), K = Object.keys(B.nav_robots), V = K.map(k => B.nav_robots[k]);
    const EXTL = { unitree_mjlab: 'G1 29 ddl (Unitree officiel)', gr00t_wbc: 'G1 29 ddl (NVIDIA GR00T)' };
    const lab = (k, v) => EXTL[v.robot] ? `${EXTL[v.robot]} · relief ×${v.relief}` : `${v.robot === 'go1' ? 'Go1' : 'G1'} ${v.policy ? '(' + (v.policy.includes('sol_plat') ? 'plat' : 'terrain UA') + ')' : '(Unitree 12 ddl, plat)'} · relief ×${v.relief}`;
    successChart(pane(g, 'MISSIONS RÉUSSIES'), K.map((k, i) => lab(k, V[i])), V.map(v => v.success), V.map(v => v.n));
    chartOn(pane(g, 'CHUTES ET DURÉE MÉDIANE'), { type: 'bar', data: { labels: K.map((k, i) => lab(k, V[i])), datasets: [{ label: 'chutes', data: V.map(v => v.falls), backgroundColor: PAL[3] + 'cc' }, { label: 'durée médiane (s)', data: V.map(v => v.time_median_s), backgroundColor: PAL[0] + '88', yAxisID: 'y2' }] }, options: { scales: { y: { min: 0, title: { display: true, text: 'chutes' } }, y2: { position: 'right', min: 0, grid: { display: false }, title: { display: true, text: 's' } } } } });
  }
  if (B.real) {
    sec('Terrain réel (Ukraine) — quelle carte embarquer ?', "2 zones × 12 départs × 5 niveaux d'information, mission complète avec dépôt");
    const g = grid(); const LV = ['trees', 'sat', 'osm', 'dem', 'none'], LL = { trees: 'arbre par arbre', sat: 'satellite', osm: 'OSM', dem: 'relief seul', none: 'aucune' };
    const sites = Object.keys(B.real.summary);
    chartOn(pane(g, 'MISSIONS RÉUSSIES (%) PAR NIVEAU DE CARTE'), { type: 'bar', data: { labels: LV.map(l => LL[l]), datasets: sites.map((s, i) => ({ label: s, data: LV.map(l => 100 * B.real.summary[s][l].success_rate), backgroundColor: PAL[i] + 'cc', borderRadius: 3 })) }, options: { scales: { y: { min: 0, max: 100 } } } });
    chartOn(pane(g, 'ERREUR DE POSITION — médiane des moyennes (m, échelle log)'), { type: 'bar', data: { labels: LV.map(l => LL[l]), datasets: sites.map((s, i) => ({ label: s, data: LV.map(l => B.real.summary[s][l].err_mean_median_m), backgroundColor: PAL[i] + 'cc', borderRadius: 3 })) }, options: { scales: { y: { type: 'logarithmic', title: { display: true, text: 'm' } } } } });
    const g2 = grid();
    sites.forEach(s => {
      const R = B.real.runs.filter(r => r.site === s), starts = [...new Set(R.map(r => r.start))].sort((a, b) => a - b);
      const cell = (st, l) => { const r = R.find(x => x.start === st && x.level === l); if (!r) return '<td>—</td>'; return `<td class="${r.success ? 'ok' : 'ko'}">${r.success ? '✔' : '✖'} ${r.err_mean_m != null ? r.err_mean_m.toFixed(1) : '—'}</td>`; };
      pane(g2, `DÉTAIL PAR DÉPART — ${s} (réussite, erreur moyenne en m)`, `<div class="pad" style="max-height:300px;overflow:auto"><table class="tb"><tr><th>départ</th>${LV.map(l => `<th>${LL[l]}</th>`).join('')}</tr>${starts.map(st => `<tr><td>${st}</td>${LV.map(l => cell(st, l)).join('')}</tr>`).join('')}</table></div>`);
    });
  }
  if (B.point) {
    sec('Simulateur ponctuel — recalage carte vs odométrie seule', 'mondes synthétiques, mission complète avec dépôt ; 400–431 = graines jamais vues pendant le développement');
    const g = grid(); const a = O(B.point.with_map_matching), b = O(B.point.dead_reckoning_only), c = O(O(B.point_holdout).with_map_matching), d = O(O(B.point_holdout).dead_reckoning_only);
    const L4 = ['100–115 recalage', '100–115 odométrie', '400–431 recalage', '400–431 odométrie'];
    successChart(pane(g, 'MISSIONS RÉUSSIES'), L4, [a.success, b.success, c.success, d.success], [a.n, b.n, c.n, d.n], [PAL[0], PAL[3], PAL[0], PAL[3]]);
    chartOn(pane(g, 'ERREUR DE POSITION (m) — médiane des moyennes et p95, échelle log'), { type: 'bar', data: { labels: L4, datasets: [{ label: 'moyenne (médiane)', data: [a, b, c, d].map(x => x.err_mean_median_m), backgroundColor: PAL[0] + 'cc' }, { label: 'p95 (médiane)', data: [a, b, c, d].map(x => x.err_p95_median_m), backgroundColor: PAL[2] + 'aa' }] }, options: { scales: { y: { type: 'logarithmic' } } } });
  }
  if (B.synth_tree && B.synth_osm) {
    const g = grid(); const a = B.synth_tree.with_map_matching, b = B.synth_osm.with_map_matching, c = B.synth_osm.dead_reckoning_only;
    successChart(pane(g, 'CARTE ARBRE PAR ARBRE vs CARTE OSM (mondes synthétiques avec chemins)'), ['arbre par arbre', 'OSM', 'odométrie seule'], [a.success, b.success, c.success], [a.n, b.n, c.n]);
    if (B.mujoco2d) { const m2 = B.mujoco2d.with_map_matching, m3 = O(O(B.mujoco3d).with_map_matching), md = O(B.mujoco2d.dead_reckoning_only);
      successChart(pane(g, 'G1 SOUS MUJOCO — LiDAR 2D / LiDAR 3D / odométrie'), [`LiDAR 2D (${m2.falls} chutes)`, `LiDAR 3D (${m3.falls} chutes)`, `odométrie (${md.falls} chutes)`], [m2.success, m3.success, md.success], [m2.n, m3.n, md.n]); }
  }
  if (B.calib) {
    sec('Politique de marche Unitree pré-entraînée (sol plat) — domaine de consigne mesuré');
    const g = grid(); const inc = [...new Set(B.calib.map(r => r.incline_mm))].sort((a, b) => a - b);
    chartOn(pane(g, 'VITESSE RÉELLE SELON LA CONSIGNE, PAR PENTE'), { type: 'line', data: { datasets: inc.map((i, k) => lineDS(`pente ${(Math.atan(i) * 180 / Math.PI).toFixed(0)}°`, B.calib.filter(r => r.incline_mm === i && r.wz_cmd === 0 && r.vx_cmd > 0).sort((p, q) => p.vx_cmd - q.vx_cmd).map(r => ({ x: r.vx_cmd, y: r.fell ? 0 : r.speed })), PAL[k % 8], { pointRadius: 3 })) }, options: { scales: xyAxes('consigne vx (m/s)', 'vitesse réelle (m/s)', { min: 0 }) } });
  }
  if (!body.children.length) body.innerHTML = '<p class="dim">Aucun benchmark dans results/.</p>';
}

/* ---------- ENTRAÎNEMENT ---------- */
let RLsel = null;
async function renderRL() {
  const D = await (await fetch('/api/training')).json(), body = $('rlBody');
  Object.keys(CH).filter(k => k.startsWith('rl')).forEach(k => { CH[k].destroy(); delete CH[k]; });
  if (!D.runs.length) { body.innerHTML = '<p class="dim">Aucun entraînement dans results/rl/ (lancer rl/train.py dans WSL).</p>'; return; }
  RLsel = RLsel && D.runs.find(r => r.id === RLsel) ? RLsel : D.runs[D.runs.length - 1].id;
  const last = r => r.progress.length ? r.progress[r.progress.length - 1] : null;
  body.innerHTML = `<div class="runlist">${D.runs.map(r => `<button class="btn ${r.id === RLsel ? 'pick on' : ''}" data-run="${r.id}">${r.id} · ${last(r) ? (last(r).step / 1e6).toFixed(0) + ' M pas · R ' + last(r).reward.toFixed(1) : '—'}</button>`).join('')}</div>
    <div class="sec">Comparaison des courbes d'apprentissage<small>récompense d'évaluation selon le nombre de pas de simulation</small></div>
    <div class="grid2c"><section class="pane"><div class="ph"><span>RÉCOMPENSE MOYENNE (évaluation)</span></div><div class="cbox tall"><canvas id="rl-all"></canvas></div></section>
    <section class="pane"><div class="ph"><span>DÉBIT D'ENTRAÎNEMENT (pas de simulation / s)</span></div><div class="cbox tall"><canvas id="rl-sps"></canvas></div></section></div>
    <div class="sec">Détail : ${RLsel}</div><div class="grid2c" id="rlDet"></div>`;
  body.querySelectorAll('[data-run]').forEach(b => b.onclick = () => { RLsel = b.dataset.run; renderRL(); });
  chartOn('rl-all', { type: 'line', data: { datasets: D.runs.map((r, i) => lineDS(r.id, r.progress.map(p => ({ x: p.step / 1e6, y: p.reward })), PAL[i % 8], { pointRadius: 2 })) }, options: { scales: xyAxes('pas (millions)', 'récompense') } });
  chartOn('rl-sps', { type: 'line', data: { datasets: D.runs.map((r, i) => lineDS(r.id, r.progress.filter(p => p.sps).map(p => ({ x: p.step / 1e6, y: p.sps })), PAL[i % 8], { pointRadius: 2 })) }, options: { scales: xyAxes('pas (millions)', 'pas / s', { min: 0 }) } });
  const R = D.runs.find(r => r.id === RLsel), det = $('rlDet');
  const keys = Object.keys(last(R) || {}).filter(k => !['step', 'reward', 'reward_std', 'sps', 'wall_s'].includes(k)).slice(0, 10);
  const mk = (title) => { const s = document.createElement('section'); s.className = 'pane'; const id = 'rl' + Math.random().toString(36).slice(2, 7); s.innerHTML = `<div class="ph"><span>${title}</span></div><div class="cbox"><canvas id="${id}"></canvas></div>`; det.appendChild(s); return id; };
  const idr = mk('RÉCOMPENSE ± ÉCART-TYPE');
  chartOn(idr, { type: 'line', data: { datasets: [lineDS('récompense', R.progress.map(p => ({ x: p.step / 1e6, y: p.reward })), PAL[0], { pointRadius: 2 }), lineDS('+σ', R.progress.map(p => ({ x: p.step / 1e6, y: p.reward + (p.reward_std || 0) })), '#7d93a6', { borderDash: [3, 3] }), lineDS('−σ', R.progress.map(p => ({ x: p.step / 1e6, y: p.reward - (p.reward_std || 0) })), '#7d93a6', { borderDash: [3, 3] })] }, options: { scales: xyAxes('pas (millions)', '') } });
  if (keys.length) { const idk = mk('MÉTRIQUES D’ÉVALUATION'); chartOn(idk, { type: 'line', data: { datasets: keys.map((k, i) => lineDS(k, R.progress.filter(p => p[k] != null).map(p => ({ x: p.step / 1e6, y: p[k] })), PAL[(i + 1) % 8])) }, options: { scales: xyAxes('pas (millions)', '') } }); }
  const cfg = R.config || {};
  const s = document.createElement('section'); s.className = 'pane';
  s.innerHTML = `<div class="ph"><span>CONFIGURATION</span></div><div class="pad" style="max-height:290px;overflow:auto"><table class="tb">${Object.entries(cfg).map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === 'object' ? JSON.stringify(v) : v}</td></tr>`).join('')}</table></div>`; det.appendChild(s);
  if (R.eval && R.eval.rows) { const s2 = document.createElement('section'); s2.className = 'pane'; const rows = R.eval.rows;
    s2.innerHTML = `<div class="ph"><span>ÉVALUATION SOUS MUJOCO (CPU)</span></div><div class="pad" style="max-height:290px;overflow:auto"><table class="tb"><tr>${Object.keys(rows[0] || {}).map(k => `<th>${k}</th>`).join('')}</tr>${rows.map(r => `<tr>${Object.values(r).map(v => `<td>${typeof v === 'number' ? +v.toFixed(3) : v}</td>`).join('')}</tr>`).join('')}</table></div>`; det.appendChild(s2); }
  if (R.media.length) { const s3 = document.createElement('section'); s3.className = 'pane'; s3.style.gridColumn = '1 / -1';
    s3.innerHTML = `<div class="ph"><span>MÉDIAS (terrain d'entraînement, rendus de la politique)</span></div><div class="gal">${R.media.map(f => f.endsWith('.mp4') ? `<video src="/api/results/rl/${R.id}/${f}" controls loop muted></video>` : `<img src="/api/results/rl/${R.id}/${f}" title="${f}">`).join('')}</div>`; det.appendChild(s3); }
}

/* ---------- DONNÉES ---------- */
function renderData() {
  const R = S.region; if (!R) return;
  $('dataBig').innerHTML = $('dataInfo').innerHTML;
  const lc = Object.entries(R.summary.landcover_fraction).sort((a, b) => b[1] - a[1]);
  const LCC = { foret: '#1b7a1b', arbustes: '#ffbb22', prairie: '#e6e64c', cultures: '#f096ff', bati: '#fa3c3c', 'sol nu': '#b4b4b4', eau: '#0064c8', 'zone humide': '#0096a0' };
  chartOn('d-lc', { type: 'doughnut', data: { labels: lc.map(x => x[0]), datasets: [{ data: lc.map(x => +(100 * x[1]).toFixed(1)), backgroundColor: lc.map(x => LCC[x[0]] || '#888'), borderColor: '#0e151d' }] }, options: { plugins: { legend: { position: 'right' } } } });
  const al = R.allied;
  chartOn('d-al', { type: 'bar', data: { labels: al.map(a => a.id), datasets: [{ label: 'distance à la ligne (km)', data: al.map(a => a.front_m / 1000), backgroundColor: al.map(a => a.kind === 'depot' ? PAL[5] + 'cc' : PAL[0] + 'cc') }] }, options: { plugins: { legend: { display: false } } } });
  const pk = {}; R.places.forEach(p => pk[p.k] = (pk[p.k] || 0) + 1);
  chartOn('d-pl', { type: 'bar', data: { labels: Object.keys(pk), datasets: [{ data: Object.values(pk), backgroundColor: PAL.map(c => c + 'cc') }] }, options: { plugins: { legend: { display: false } } } });
}

window.TALOS = { S, map, setPoint, computeRoute, showView };
