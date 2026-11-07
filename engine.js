/* Boat race engine: features, tree-ensemble prediction, expected value. Runs in the browser and in node (for tests). */
(function (root) {
  const BASE = ['nat_win', 'nat_2', 'motor_2', 'boat_2', 'ex', 'age'];

  function mean(a) { const v = a.filter(x => x != null && !Number.isNaN(x)); return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null; }
  // pandas-style rank: average for ties, null stays null. ascending=false -> highest value gets rank 1
  function rank(vals, ascending) {
    const out = vals.map(() => null);
    const idx = vals.map((v, i) => [v, i]).filter(p => p[0] != null && !Number.isNaN(p[0]));
    idx.sort((a, b) => ascending ? a[0] - b[0] : b[0] - a[0]);
    let i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const r = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) out[idx[k][1]] = r;
      i = j + 1;
    }
    return out;
  }

  // boats: array of raw objects {boat, cls, age, nat_win, nat_2, motor_2, boat_2, ex}; returns array of feature maps
  function featuresForRace(boats) {
    const feats = boats.map(b => Object.assign({}, b));
    for (const c of BASE) {
      const vals = boats.map(b => b[c]);
      const m = mean(vals);
      const rk = rank(vals, c === 'ex');
      feats.forEach((f, i) => { f[c + '_d'] = (vals[i] == null || m == null) ? null : Math.round((vals[i] - m) * 1e6) / 1e6; f[c + '_r'] = rk[i]; });
    }
    return feats;
  }

  function predictRaw(model, f) {
    const x = model.cols.map(c => { const v = f[c]; return (v == null || Number.isNaN(v)) ? NaN : v; });
    let s = model.base;
    for (const t of model.trees) {
      let n = 0;
      for (;;) {
        const ft = t.f[n];
        if (ft < 0) { s += t.v[n]; break; }
        const v = x[ft];
        if (Number.isNaN(v)) n = t.m[n] ? t.l[n] : t.r[n];
        else n = v <= t.t[n] ? t.l[n] : t.r[n];
      }
    }
    return s;
  }
  const sigmoid = z => 1 / (1 + Math.exp(-z));

  // Returns {p: win probabilities (sum 1), estOdds: estimated market win odds, mode: 'full'|'pre'}
  function predictRace(models, boats) {
    const hasEx = boats.length === 6 && boats.every(b => b.ex != null && b.ex > 0);
    const mode = hasEx ? 'full' : 'pre';
    const raw = hasEx ? boats : boats.map(b => Object.assign({}, b, { ex: null }));
    const feats = featuresForRace(raw);
    const pr = feats.map(f => sigmoid(predictRaw(models[mode], f)));
    const tot = pr.reduce((a, b) => a + b, 0);
    const p = pr.map(x => x / tot);
    const estOdds = feats.map(f => Math.exp(predictRaw(models['mkt_' + mode], f)));
    return { p, estOdds, mode };
  }

  // Map one race object of the BoatraceOpenAPI to raw model inputs. Returns null when the race cannot be scored.
  function rawFromApiRace(race) {
    const rc = race.racers || {};
    const pv = (race.preview && race.preview.racers) || {};
    const keys = Object.keys(rc).sort((a, b) => a - b);
    if (keys.length !== 6) return null;
    const boats = keys.map(k => {
      const r = rc[k], p = pv[k] || {};
      return { boat: Number(k), cls: r.rank_number, age: r.age, nat_win: r.national_win_rate, nat_2: r.national_top_2_percent,
               motor_2: r.motor_top_2_percent, boat_2: r.boat_top_2_percent, ex: p.exhibition_time != null ? p.exhibition_time : null };
    });
    if (boats.some(b => b.cls == null || b.nat_win == null)) return null;
    return boats;
  }

  const api = { predictRace, rawFromApiRace, featuresForRace, predictRaw, rank };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Engine = api;
})(typeof window !== 'undefined' ? window : globalThis);
