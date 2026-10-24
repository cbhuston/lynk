// Lynk Lab: an AI racing analyst that runs entirely in the browser.
//
// Data: DRF result-chart data for the 2025 Kentucky Derby, loaded into DuckDB-WASM (real SQL, in the browser).
// AI:   a small open model run on the visitor's GPU with WebLLM (no server, no API keys).
//
// How a question is answered (cheapest path first):
//   Tier 0  a keyword router maps common questions to tested SQL tools        -> 0 LLM tokens to pick the query
//   Tier 1  the LLM chooses a tool from a one-line-per-tool catalog          -> small prompt, JSON out
//   Tier 2  the LLM writes SQL against a compact schema, validated + retried -> only when no tool fits
//   Answer  the LLM reads ONLY the rows the query returned (capped) and writes 2-5 sentences
// Without WebGPU the app still answers every Tier 0 question with tables and a templated summary.

import * as duckdb from 'https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.32.0/+esm';   // 1.29.0 had a filter bug (see lynk-lab page)

const TABLES = ['field', 'past_performances', 'starts', 'derby_history', 'race', 'exotic_payoffs'];
const MODEL_PREFS = {
  fast: ['Qwen2.5-1.5B-Instruct-q4f16_1-MLC', 'Qwen2-1.5B-Instruct-q4f16_1-MLC', 'Llama-3.2-1B-Instruct-q4f16_1-MLC'],
  smart: ['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Llama-3.2-3B-Instruct-q4f16_1-MLC', 'Phi-3.5-mini-instruct-q4f16_1-MLC'],
};
const state = { conn: null, engine: null, field: [], horses: [], jockeys: [], trainers: [], tokens: 0, reveal: false };
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/'/g, "''");
const html = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const fmt = (v) => (v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))) ? '' :
  (typeof v === 'number' ? (Number.isInteger(v) ? (Math.abs(v) >= 10000 ? v.toLocaleString() : String(v)) : (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : v.toFixed(2))) : String(v));

// ------------------------------------------------------------------ data layer (DuckDB-WASM)
async function initDb() {
  const bundle = await duckdb.selectBundle(duckdb.getJsDelivrBundles());
  const workerUrl = URL.createObjectURL(new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' }));
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), new Worker(workerUrl));
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  state.conn = await db.connect();
  for (const t of TABLES) {
    await db.registerFileURL(`${t}.parquet`, new URL(`data/${t}.parquet`, location.href).href, duckdb.DuckDBDataProtocol.HTTP, false);
    await state.conn.query(`CREATE TABLE ${t} AS SELECT * FROM '${t}.parquet'`);
  }
}

function toPlain(row) {
  const out = {};
  for (const [k, v] of Object.entries(row)) {
    if (typeof v === 'bigint') out[k] = Number(v);
    else if (v instanceof Date) out[k] = v.toISOString().slice(0, 10);
    else if (typeof v === 'number' && /date/.test(k) && v > 1e11) out[k] = new Date(v).toISOString().slice(0, 10);
    else out[k] = v;
  }
  return out;
}

async function q(sql) {
  const res = await state.conn.query(sql);
  const dateCols = new Set(res.schema.fields.filter((f) => /Date|Timestamp/.test(String(f.type))).map((f) => f.name));
  return res.toArray().map((r) => {
    const o = toPlain(r.toJSON());
    for (const k of Object.keys(o)) {
      const v = o[k];
      if (dateCols.has(k) && typeof v === 'number') o[k] = new Date(v).toISOString().slice(0, 10);
      else if (v && typeof v === 'object' && ArrayBuffer.isView(v)) o[k] = Array.from(v).reduce((acc, w, i) => acc + Number(w) * 2 ** (32 * i), 0);
    }
    return o;
  });
}

// ------------------------------------------------------------------ entity resolution
function findAll(text, list) {
  // Full-name matches win; a last-name match only counts when it is unambiguous and no full match shares it.
  const clean = (x) => x.toLowerCase().replace(/\s*\(jpn\)/, '').replace(/[.,']/g, ' ').replace(/\s+/g, ' ').trim();
  const t = clean(text);
  const tokens = (n) => clean(n).split(' ').filter((p) => p.length > 1 && !['jr', 'iii', 'ii'].includes(p));
  const has = (w) => new RegExp(`\b${w}\b`).test(t);
  const scored = [];
  for (const name of list) {
    const parts = tokens(name), first = parts[0], last = parts[parts.length - 1];
    if (t.includes(clean(name))) scored.push({ name, score: 3, last });
    else if (parts.length > 1 && has(first) && has(last)) scored.push({ name, score: 2, last });
    else if (last && last.length > 3 && has(last)) scored.push({ name, score: 1, last });
  }
  const strong = scored.filter((x) => x.score >= 2);
  const strongLasts = new Set(strong.map((x) => x.last));
  const counts = {};
  scored.filter((x) => x.score === 1).forEach((x) => { counts[x.last] = (counts[x.last] || 0) + 1; });
  const weak = scored.filter((x) => x.score === 1 && counts[x.last] === 1 && !strongLasts.has(x.last));
  return [...new Set([...strong, ...weak].map((x) => x.name))];
}

// ------------------------------------------------------------------ tools (tested SQL, ported from the original Lynk Lab queries)
const ROI = (n = 'win_payoff') => `round(100.0 * (sum(${n}) - 2 * count(*)) / (2 * count(*)), 1)`;
const slices = (col, name) => `
  SELECT 'Every start since 2012' AS slice, count(*) AS starts, sum((finish = 1)::INT) AS wins, round(100 * avg((finish = 1)::INT), 1) AS win_pct,
         round(100 * avg((finish <= 3)::INT), 1) AS in_the_money_pct, ${ROI()} AS roi_pct FROM starts WHERE ${col} = '${esc(name)}'
  UNION ALL SELECT 'Last 365 days', count(*), sum((finish = 1)::INT), round(100 * avg((finish = 1)::INT), 1), round(100 * avg((finish <= 3)::INT), 1), ${ROI()}
         FROM starts WHERE ${col} = '${esc(name)}' AND race_date >= DATE '2024-05-03'
  UNION ALL SELECT 'At Churchill Downs', count(*), sum((finish = 1)::INT), round(100 * avg((finish = 1)::INT), 1), round(100 * avg((finish <= 3)::INT), 1), ${ROI()}
         FROM starts WHERE ${col} = '${esc(name)}' AND track_code = 'CD'
  UNION ALL SELECT 'Graded stakes', count(*), sum((finish = 1)::INT), round(100 * avg((finish = 1)::INT), 1), round(100 * avg((finish <= 3)::INT), 1), ${ROI()}
         FROM starts WHERE ${col} = '${esc(name)}' AND grade IS NOT NULL
  UNION ALL SELECT 'Grade 1 races', count(*), sum((finish = 1)::INT), round(100 * avg((finish = 1)::INT), 1), round(100 * avg((finish <= 3)::INT), 1), ${ROI()}
         FROM starts WHERE ${col} = '${esc(name)}' AND grade = '1'
  UNION ALL SELECT 'As the betting favorite', count(*), sum((finish = 1)::INT), round(100 * avg((finish = 1)::INT), 1), round(100 * avg((finish <= 3)::INT), 1), ${ROI()}
         FROM starts WHERE ${col} = '${esc(name)}' AND is_favorite
  UNION ALL SELECT 'Dirt routes (8f+), 2023 on', count(*), sum((finish = 1)::INT), round(100 * avg((finish = 1)::INT), 1), round(100 * avg((finish <= 3)::INT), 1), ${ROI()}
         FROM starts WHERE ${col} = '${esc(name)}' AND surface = 'Dirt' AND distance_furlongs >= 8 AND race_date >= DATE '2023-01-01'`;

const SUMMARY_PP = `SELECT horse, count(*) AS starts, sum((finish = 1)::INT) AS wins, sum((finish = 2)::INT) AS seconds, sum((finish = 3)::INT) AS thirds,
  max(speed_figure) AS best_figure, max(CASE WHEN rn = 1 THEN speed_figure END) AS last_figure,
  round(avg(CASE WHEN rn <= 3 THEN speed_figure END), 1) AS last3_avg_figure,
  sum((grade IS NOT NULL AND finish = 1)::INT) AS graded_stakes_wins, round(sum(earnings)) AS earnings
  FROM (SELECT *, row_number() OVER (PARTITION BY horse ORDER BY race_date DESC) AS rn FROM past_performances) pp`;

const TOOLS = {
  field_overview: {
    desc: 'the full field with connections, morning line and AI win probability',
    run: async () => ({ title: 'The 2025 Kentucky Derby field', sql: `SELECT post_position, horse, jockey, trainer, morning_line,
        round(100 * ai_win_probability, 1) AS ai_win_pct FROM field WHERE NOT scratched ORDER BY ai_win_probability DESC` }),
    summarize: (r) => `${r.length} horses go to post. The AI's top three are ${r.slice(0, 3).map((x) => `${x.horse} (${x.ai_win_pct}%)`).join(', ')}.`,
  },
  ai_predictions: {
    desc: 'AI win probabilities compared with the morning line (value)',
    run: async () => ({ title: 'AI win probability vs the morning line', sql: `SELECT horse, round(100 * ai_win_probability, 1) AS ai_win_pct, morning_line,
        round(100 / (try_cast(split_part(morning_line, '-', 1) AS DOUBLE) / try_cast(split_part(morning_line, '-', 2) AS DOUBLE) + 1), 1) AS morning_line_pct,
        round(100 * ai_win_probability - 100 / (try_cast(split_part(morning_line, '-', 1) AS DOUBLE) / try_cast(split_part(morning_line, '-', 2) AS DOUBLE) + 1), 1) AS ai_minus_line
        FROM field WHERE NOT scratched ORDER BY ai_win_probability DESC` }),
    summarize: (r) => {
      const v = [...r].filter((x) => x.ai_minus_line !== null).sort((a, b) => b.ai_minus_line - a.ai_minus_line)[0];
      return `The model's top pick is ${r[0].horse} at ${r[0].ai_win_pct}%. Relative to the morning line, the biggest overlay is ${v.horse}: ${v.ai_win_pct}% from the model against ${v.morning_line_pct}% implied by ${v.morning_line}.`;
    },
  },
  horse_profile: {
    desc: 'career summary and last five starts for one horse', args: ['horse'],
    run: async ({ horse }) => ({ title: `${horse}: career and recent form`, sql: `SELECT race_date, track_code, race_name, grade, distance_furlongs AS furlongs, surface,
        track_condition, finish, field_size, beaten_lengths, speed_figure, final_odds, jockey, winner FROM past_performances
        WHERE horse = '${esc(horse)}' ORDER BY race_date DESC LIMIT 5`, extra: `${SUMMARY_PP} WHERE horse = '${esc(horse)}' GROUP BY horse` }),
    summarize: (r, x, a) => !(x && x[0]) ? `${a.horse} raced outside North America before the Derby, so its past performances are not in the chart data. The AI still rates it from what is known about its connections and pedigree.` : `${x[0].horse} has ${x[0].starts} starts with ${x[0].wins} wins, ${x[0].seconds} seconds and ${x[0].thirds} thirds, earning $${fmt(x[0].earnings)}. Best speed figure ${x[0].best_figure}, last ${x[0].last_figure}. Most recent start: ${r[0].race_name || 'a ' + r[0].track_code + ' race'}, finished ${r[0].finish} of ${r[0].field_size}.`,
  },
  past_performances: {
    desc: 'every earlier start of one horse', args: ['horse'],
    run: async ({ horse }) => ({ title: `${horse}: past performances`, sql: `SELECT race_date, track_code, race_name, grade, distance_furlongs AS furlongs, surface,
        track_condition, post, finish, field_size, beaten_lengths, speed_figure, final_odds, jockey, trip_comment, winner FROM past_performances
        WHERE horse = '${esc(horse)}' ORDER BY race_date DESC` }),
    summarize: (r, x, a) => !r.length ? `${a.horse} raced outside North America before the Derby, so its past performances are not in the chart data.` : `${r.length} earlier starts, from ${r[r.length - 1]?.race_date} to ${r[0]?.race_date}. Speed figures ranged from ${Math.min(...r.map((x) => x.speed_figure ?? 999))} to ${Math.max(...r.map((x) => x.speed_figure ?? 0))}.`,
  },
  compare_horses: {
    desc: 'side-by-side comparison of two or more horses', args: ['horses'],
    run: async ({ horses }) => ({ title: `Comparing ${horses.join(' and ')}`, sql: `SELECT s.*, round(100 * f.ai_win_probability, 1) AS ai_win_pct, f.morning_line
        FROM (${SUMMARY_PP} WHERE horse IN (${horses.map((h) => `'${esc(h)}'`).join(', ')}) GROUP BY horse) s JOIN field f USING (horse)
        ORDER BY f.ai_win_probability DESC` }),
    summarize: (r) => r.map((x) => `${x.horse}: ${x.wins} wins from ${x.starts}, best figure ${x.best_figure}, last-3 average ${x.last3_avg_figure}, AI ${x.ai_win_pct}%.`).join(' '),
  },
  top_figures: {
    desc: 'the field ranked by speed figures',
    run: async () => ({ title: 'Speed figures across the field', sql: `SELECT s.horse, s.best_figure, s.last_figure, s.last3_avg_figure, s.starts, s.wins,
        round(100 * f.ai_win_probability, 1) AS ai_win_pct FROM (${SUMMARY_PP} GROUP BY horse) s JOIN field f USING (horse)
        WHERE NOT f.scratched ORDER BY s.last3_avg_figure DESC NULLS LAST` }),
    summarize: (r) => `${r[0].horse} has the best recent figures (last-3 average ${r[0].last3_avg_figure}), followed by ${r[1].horse} (${r[1].last3_avg_figure}) and ${r[2].horse} (${r[2].last3_avg_figure}).`,
  },
  pace_map: {
    desc: 'early speed: who is likely to be on or near the lead',
    run: async () => ({ title: 'Pace map from recent running lines', sql: `SELECT horse, round(avg((position_first_call - 1.0) / nullif(field_size - 1, 0)), 2) AS early_position,
        count(position_first_call) AS lines_used,
        CASE WHEN avg((position_first_call - 1.0) / nullif(field_size - 1, 0)) < 0.25 THEN 'Early speed'
             WHEN avg((position_first_call - 1.0) / nullif(field_size - 1, 0)) < 0.5 THEN 'Presser'
             ELSE 'Closer' END AS style
        FROM (SELECT *, row_number() OVER (PARTITION BY horse ORDER BY race_date DESC) AS rn FROM past_performances WHERE position_first_call IS NOT NULL) p
        WHERE rn <= 4 AND horse IN (SELECT horse FROM field WHERE NOT scratched) GROUP BY horse ORDER BY early_position` }),
    summarize: (r) => `Early speed: ${r.filter((x) => x.style === 'Early speed').map((x) => x.horse).join(', ') || 'none clear'}. Closers: ${r.filter((x) => x.style === 'Closer').map((x) => x.horse).slice(0, 5).join(', ')}.`,
  },
  jockey_stats: {
    desc: 'a jockey\'s record by slice (win %, in the money %, ROI on $2 win bets)', args: ['jockey'],
    run: async ({ jockey }) => ({ title: `${jockey}: riding record`, sql: slices('jockey', jockey) }),
    summarize: (r, x, a, question) => { const i = sliceIndex(question); return `${a.jockey}, ${r[i].slice.toLowerCase()}: ${r[i].wins} wins from ${r[i].starts} rides (${r[i].win_pct}%), in the money ${r[i].in_the_money_pct}%, and a $2 win bet on every ride returned ${r[i].roi_pct}%. Since 2012 overall: ${r[0].win_pct}% winners from ${r[0].starts.toLocaleString()} rides.`; },
  },
  trainer_stats: {
    desc: 'a trainer\'s record by slice, plus ROI on the last 30 starts', args: ['trainer'],
    run: async ({ trainer }) => ({ title: `${trainer}: training record`, sql: `${slices('trainer', trainer)}
        UNION ALL SELECT 'Last 30 starts', count(*), sum((finish = 1)::INT), round(100 * avg((finish = 1)::INT), 1), round(100 * avg((finish <= 3)::INT), 1), ${ROI()}
        FROM (SELECT * FROM starts WHERE trainer = '${esc(trainer)}' ORDER BY race_date DESC LIMIT 30)` }),
    summarize: (r, x, a, question) => { const i = sliceIndex(question, true); return `${a.trainer}, ${r[i].slice.toLowerCase()}: ${r[i].wins} wins from ${r[i].starts} starters (${r[i].win_pct}%), in the money ${r[i].in_the_money_pct}%, $2 win ROI ${r[i].roi_pct}%. Since 2012 overall: ${r[0].wins.toLocaleString()} wins from ${r[0].starts.toLocaleString()} starters, ${r[3].wins} in graded stakes.`; },
  },
  jockey_trainer: {
    desc: 'how a jockey and trainer do together', args: ['jockey', 'trainer'],
    run: async ({ jockey, trainer }) => ({ title: `${jockey} riding for ${trainer}`, sql: `SELECT count(*) AS starts, sum((finish = 1)::INT) AS wins,
        round(100 * avg((finish = 1)::INT), 1) AS win_pct, round(100 * avg((finish <= 3)::INT), 1) AS in_the_money_pct, ${ROI()} AS roi_pct,
        CAST(CAST(max(race_date) AS DATE) AS VARCHAR) AS last_ride FROM starts WHERE jockey = '${esc(jockey)}' AND trainer = '${esc(trainer)}'` }),
    summarize: (r, x, a) => `${a.jockey} has ridden ${r[0].starts} times for ${a.trainer}, winning ${r[0].wins} (${r[0].win_pct ?? 0}%).`,
  },
  derby_record: {
    desc: 'Kentucky Derby history (2012-2024) for a jockey, trainer, or all winners', args: ['name?'],
    run: async ({ name }) => name ? ({ title: `${name} in the Kentucky Derby since 2012`, sql: `SELECT year, horse, jockey, trainer, post, final_odds, finish, speed_figure
        FROM derby_history WHERE jockey = '${esc(name)}' OR trainer = '${esc(name)}' ORDER BY year DESC` })
      : ({ title: 'Kentucky Derby winners since 2012', sql: `SELECT year, horse, jockey, trainer, post, final_odds, speed_figure, round(win_payoff, 2) AS win_payoff
        FROM derby_history WHERE finish = 1 ORDER BY year DESC` }),
    summarize: (r, x, a) => a.name ? `${a.name} has ${r.length} Derby starters since 2012 with ${r.filter((z) => z.finish === 1).length} wins.` : `${r.length} Derbies on file. Favorites won ${r.filter((z) => z.final_odds < 4).length} of them.`,
  },
  jockey_changes: {
    desc: 'horses with a new rider for the Derby',
    run: async () => ({ title: 'Rider changes for the Derby', sql: `SELECT f.horse, f.jockey AS derby_jockey,
        (SELECT count(*) FROM past_performances p WHERE p.horse = f.horse AND p.jockey = f.jockey) AS previous_rides_together,
        (SELECT p.jockey FROM past_performances p WHERE p.horse = f.horse ORDER BY race_date DESC LIMIT 1) AS last_race_jockey
        FROM field f WHERE NOT f.scratched ORDER BY previous_rides_together, f.horse` }),
    summarize: (r) => `${r.filter((x) => x.previous_rides_together === 0).length} horses have a jockey who has never ridden them before: ${r.filter((x) => x.previous_rides_together === 0).map((x) => x.horse).join(', ')}.`,
  },
  condition_records: {
    desc: 'records on wet dirt tracks (muddy, sloppy, wet fast) or another track condition, for the Derby trainers, jockeys or horses', args: ['role', 'condition', 'name?'],
    run: async ({ role = 'trainer', condition = 'wet', name = null }) => {
      const conds = CONDITIONS[condition] || CONDITIONS.wet;
      const inList = conds.map((c) => `'${c}'`).join(', ');
      const label = CONDITION_LABEL[condition] || CONDITION_LABEL.wet;
      if (role === 'horse') {
        return { title: name ? `${name} on ${label}` : `Derby horses on ${label}`, sql: `SELECT horse, count(*) FILTER (WHERE track_condition IN (${inList})) AS starts_on_condition,
            CAST(count(*) FILTER (WHERE track_condition IN (${inList}) AND finish = 1) AS INTEGER) AS wins_on_condition,
            max(speed_figure) FILTER (WHERE track_condition IN (${inList})) AS best_figure_on_condition,
            max(speed_figure) FILTER (WHERE track_condition = 'Fast') AS best_figure_fast_track, count(*) AS career_starts
            FROM past_performances WHERE ${name ? `horse = '${esc(name)}'` : `horse IN (SELECT horse FROM field WHERE NOT scratched)`}
            GROUP BY horse ORDER BY starts_on_condition DESC, best_figure_on_condition DESC NULLS LAST` };
      }
      const col = role === 'jockey' ? 'jockey' : 'trainer';
      const on = `track_condition IN (${inList}) AND surface = 'Dirt'`;
      return { title: name ? `${name} on ${label} since 2012` : `Derby ${col}s on ${label} since 2012`, sql: `SELECT ${col},
          count(*) FILTER (WHERE ${on}) AS starts_on_condition,
          CAST(count(*) FILTER (WHERE ${on} AND finish = 1) AS INTEGER) AS wins,
          round(100 * avg((finish = 1)::INT) FILTER (WHERE ${on}), 1) AS win_pct,
          round(100 * avg((finish <= 3)::INT) FILTER (WHERE ${on}), 1) AS in_the_money_pct,
          round(100 * (sum(win_payoff) FILTER (WHERE ${on}) - 2 * count(*) FILTER (WHERE ${on})) / nullif(2 * count(*) FILTER (WHERE ${on}), 0), 1) AS roi_pct,
          round(100 * avg((finish = 1)::INT) FILTER (WHERE track_condition = 'Fast' AND surface = 'Dirt'), 1) AS fast_dirt_win_pct
          FROM starts WHERE ${name ? `${col} = '${esc(name)}'` : `is_derby_${col}`}
          GROUP BY ${col} HAVING count(*) FILTER (WHERE ${on}) >= ${name ? 1 : 20}
          ORDER BY win_pct DESC` };
    },
    summarize: (r, x, a) => {
      const label = CONDITION_LABEL[a.condition] || CONDITION_LABEL.wet;
      if (!r.length) return a.name ? `${a.name} has no starts on ${label} in the data, so I can't rate them on it.` : 'There are not enough starts on that track condition to say.';
      if (a.name && (a.role || 'trainer') === 'horse') {
        const h = r[0];
        return h.starts_on_condition ? `${h.horse} has run ${h.starts_on_condition === 1 ? 'once' : `${h.starts_on_condition} times`} on ${label}, winning ${h.wins_on_condition}, with a best figure of ${h.best_figure_on_condition} there against ${h.best_figure_fast_track ?? 'n/a'} on a fast track.`
          : `${h.horse} has never run on ${label} (${h.career_starts} career starts), so there is no record to judge. Its best fast-track figure is ${h.best_figure_fast_track ?? 'n/a'}.`;
      }
      if (a.name) {
        const who = a.role === 'jockey' ? 'jockey' : 'trainer', z = r[0];
        return `${z[who]} on ${label} since 2012: ${z.wins} wins from ${z.starts_on_condition} starts (${z.win_pct}%), in the money ${z.in_the_money_pct}%, $2 win ROI ${z.roi_pct}%. On fast dirt the win rate is ${z.fast_dirt_win_pct}%.`;
      }
      if ((a.role || 'trainer') === 'horse') return `${r.filter((z) => z.starts_on_condition > 0).length} of the Derby horses have run on ${label}. ${r[0].horse} has the most experience (${r[0].starts_on_condition} starts, ${r[0].wins_on_condition} wins, best figure ${r[0].best_figure_on_condition ?? 'n/a'}).`;
      const who = a.role === 'jockey' ? 'jockey' : 'trainer', top = r[0];
      return `Among the Derby ${who}s with at least 20 such starts, ${top[who]} does best on ${label}: ${top.wins} wins from ${top.starts_on_condition} starts (${top.win_pct}%), against ${top.fast_dirt_win_pct}% on fast dirt, with a $2 win ROI of ${top.roi_pct}%.${r[1] ? ` Next is ${r[1][who]} at ${r[1].win_pct}%.` : ''}`;
    },
  },
  record: {
    desc: 'win %, in-the-money %, ROI for a Derby jockey or trainer (or a leaderboard of them) under any mix of filters: track, track condition, surface, sprint/route, year, graded stakes, favorite', args: ['role', 'name?', 'filters'],
    run: async ({ role = 'trainer', name = null, filters = {} }) => {
      const col = role === 'jockey' ? 'jockey' : 'trainer';
      const where = [name ? `${col} = '${esc(name)}'` : `is_derby_${col}`, ...filterSql(filters)].join(' AND ');
      const label = filterLabel(filters);
      return { title: name ? `${name} ${label}` : `Derby ${col}s ${label}`, sql: `SELECT ${col}, count(*) AS starts, CAST(sum((finish = 1)::INT) AS INTEGER) AS wins,
          round(100 * avg((finish = 1)::INT), 1) AS win_pct, round(100 * avg((finish <= 3)::INT), 1) AS in_the_money_pct, ${ROI()} AS roi_pct,
          CAST(CAST(max(race_date) AS DATE) AS VARCHAR) AS most_recent
          FROM starts WHERE ${where} GROUP BY ${col} ${name ? '' : 'HAVING count(*) >= 10'} ORDER BY win_pct DESC` };
    },
    summarize: (r, x, a) => {
      const col = a.role === 'jockey' ? 'jockey' : 'trainer', label = filterLabel(a.filters || {});
      if (!r.length) return a.name ? `${a.name} has no starts ${label} in the data, so there is no record to report.` : `No Derby ${col} has at least 10 starts ${label}.`;
      const z = r[0], plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
      const line = `${plural(z.wins, 'win')} from ${plural(z.starts, 'start')} (${z.win_pct}%), in the money ${z.in_the_money_pct}% of the time, and a $2 win bet on every start returned ${z.roi_pct}%`;
      if (a.name) return `${z[col]} ${label}: ${line}.${z.starts < 20 ? ' That is a small sample, so treat it with caution.' : ''}`;
      return `Among the Derby ${col}s with at least 10 starts ${label}, ${z[col]} leads with ${line}.${r[1] ? ` Next is ${r[1][col]} at ${r[1].win_pct}% from ${r[1].starts} starts.` : ''}`;
    },
  },
  horse_wins: {
    desc: 'which 2025 Derby horses have won races, one row per horse with the races listed, under any filter: graded stakes, Grade 1, track, surface, track condition, sprint/route, year', args: ['filters'],
    run: async ({ filters = {} }) => ({ title: `Derby horses that have won ${filterLabel(filters, false).replace(/^in /, '') || 'a race'}`,
      sql: `SELECT horse, count(*) AS wins, string_agg(coalesce(nullif(race_name, ''), 'a ' || race_type) || ' (' || track_code || ', ' || strftime(race_date, '%b %Y') || ')', '; ' ORDER BY race_date DESC) AS races
        FROM starts WHERE ${['is_derby_horse', 'finish = 1', ...filterSql(filters)].join(' AND ')} GROUP BY horse ORDER BY wins DESC, horse` }),
    summarize: (r, x, a) => {
      const label = filterLabel(a.filters || {}, false), what = label ? `won ${label.replace(/^in /, '')}` : 'won a race';
      if (!r.length) return `None of the 2025 Derby horses have ${what} in the data.`;
      const names = r.map((z) => z.wins > 1 ? `${z.horse} (${z.wins})` : z.horse);
      const list = names.length > 1 ? `${names.slice(0, -1).join(', ')} and ${names.at(-1)}` : names[0];
      return `${r.length} of the Derby horses have ${what}: ${list}. ${r[0].horse}'s wins: ${r[0].races}.`;
    },
  },
  results: {
    desc: 'the official result, final odds and payoffs',
    run: async () => ({ title: 'Official result', sql: `SELECT finish, horse, jockey, trainer, final_odds, beaten_lengths, speed_figure, win_payoff, place_payoff,
        show_payoff, trip_comment FROM field WHERE NOT scratched ORDER BY finish` }),
    summarize: (r) => `${r[0].horse} won, paying $${r[0].win_payoff} to win, from ${r[1].horse} and ${r[2].horse}. The AI had the winner at ${state.field.find((f) => f.horse === r[0].horse)?.ai_pct}%.`,
  },
};

function sliceIndex(question = '', trainer = false) {
  const t = question.toLowerCase();
  if (/churchill/.test(t)) return 2;
  if (/grade 1|\bg1\b/.test(t)) return 4;
  if (/graded|stakes/.test(t)) return 3;
  if (/favou?rite/.test(t)) return 5;
  if (/route|dirt|distance/.test(t)) return 6;
  if (trainer && /last 30|recent/.test(t)) return 7;
  if (/last year|365|recent|this year/.test(t)) return 1;
  return 0;
}

const CONDITIONS = { wet: ['Muddy', 'Sloppy', 'Wet Fast'], muddy: ['Muddy'], sloppy: ['Sloppy'], good: ['Good'], fast: ['Fast'], firm: ['Firm'], soft: ['Soft', 'Yielding'] };
const CONDITION_LABEL = { wet: 'wet dirt tracks (muddy, sloppy or wet fast)', muddy: 'muddy tracks', sloppy: 'sloppy tracks', good: 'good tracks', fast: 'fast tracks', firm: 'firm turf', soft: 'soft or yielding turf' };
function conditionOf(t) {
  if (/\bslop(py)?\b/.test(t) && !/mud/.test(t)) return 'sloppy';
  if (/\bmud(dy)?\b|\bwet\b|\boff[- ]track|\boff going|\brain/.test(t)) return 'wet';
  if (/\bfirm\b/.test(t)) return 'firm';
  if (/\b(soft|yielding)\b/.test(t)) return 'soft';
  return null;
}

const TRACKS = [];   // [{code, name, keys: ['churchill downs', 'churchill']}] loaded from the data

async function loadTracks() {
  const rows = await q(`SELECT track_code, any_value(track_name) AS track_name, count(*) AS n FROM starts GROUP BY track_code ORDER BY n DESC`);
  for (const r of rows) {
    const name = String(r.track_name || '').toLowerCase().replace(/[^a-z ]/g, ' ').replace(/\s+/g, ' ').trim();
    if (!name || name === r.track_code.toLowerCase()) continue;
    const keys = [name];
    const first = name.split(' ')[0];
    if (first.length >= 5 && !['downs', 'park', 'race', 'racing', 'track'].includes(first)) keys.push(first);
    TRACKS.push({ code: r.track_code, name: r.track_name, keys });
  }
  TRACKS.sort((a, b) => Math.max(...b.keys.map((k) => k.length)) - Math.max(...a.keys.map((k) => k.length)));
}

function filtersOf(t) {
  const f = {};
  for (const tr of TRACKS) {
    if (tr.keys.some((k) => new RegExp(`\\b${k}\\b`).test(t))) { f.track = tr; break; }
  }
  const cond = conditionOf(t);
  if (cond) f.condition = cond;
  if (/\bturf\b|\bgrass\b/.test(t)) f.surface = 'Turf';
  else if (/\bsynthetic\b|\bpoly/.test(t)) f.surface = 'Synthetic';
  else if (/\bdirt\b/.test(t) || cond) f.surface = cond === 'firm' || cond === 'soft' ? 'Turf' : 'Dirt';
  if (/\broutes?\b|two turns|1 1\/[48]|mile and/.test(t)) f.distance = 'route';
  else if (/\bsprints?\b/.test(t)) f.distance = 'sprint';
  const since = t.match(/\bsince (20\d\d)\b/), inYear = t.match(/\bin (20\d\d)\b/);
  if (since) f.since = Number(since[1]);
  if (inYear) f.year = Number(inYear[1]);
  if (/grade 1|\bg1\b/.test(t)) f.grade = '1';
  else if (/graded|stakes/.test(t)) f.graded = true;
  if (/favou?rite/.test(t)) f.favorite = true;
  return f;
}

function filterSql(f) {
  const w = [];
  if (f.track) w.push(`track_code = '${esc(f.track.code)}'`);
  if (f.condition) w.push(`track_condition IN (${(CONDITIONS[f.condition] || []).map((c) => `'${c}'`).join(', ')})`);
  if (f.surface) w.push(`surface = '${f.surface}'`);
  if (f.distance === 'route') w.push('distance_furlongs >= 8');
  if (f.distance === 'sprint') w.push('distance_furlongs < 8');
  if (f.since) w.push(`race_date >= DATE '${f.since}-01-01'`);
  if (f.year) w.push(`year(race_date) = ${f.year}`);
  if (f.grade) w.push(`grade = '${f.grade}'`);
  if (f.graded) w.push('grade IS NOT NULL');
  if (f.favorite) w.push('is_favorite');
  return w;
}

function filterLabel(f, since = true) {
  const parts = [];
  if (f.condition) parts.push(`on ${CONDITION_LABEL[f.condition]}`);
  else if (f.surface) parts.push(`on ${f.surface.toLowerCase()}`);
  if (f.distance) parts.push(f.distance === 'route' ? 'in routes (8 furlongs or more)' : 'in sprints (under 8 furlongs)');
  if (f.grade) parts.push('in Grade 1 races'); else if (f.graded) parts.push('in graded stakes');
  if (f.favorite) parts.push('as the betting favorite');
  if (f.track) parts.push(`at ${f.track.name}`);
  if (f.year) parts.push(`in ${f.year}`); else if (since || f.since) parts.push(`since ${f.since || 2012}`);
  return parts.join(' ');
}

// ------------------------------------------------------------------ Tier 0: keyword router (no tokens)
function route(text) {
  const t = text.toLowerCase();
  const horses = findAll(text, state.horses), jockeys = findAll(text, state.jockeys), trainers = findAll(text, state.trainers);
  if (/\b(suggest|recommend|who should i|best bet|pick a|give me a bet|bet on)\b/.test(t)) return { tool: 'suggest_bet' };
  const cond = conditionOf(t);
  const filters = filtersOf(t);
  const personRole = trainers.length && !jockeys.length ? 'trainer' : jockeys.length && !trainers.length ? 'jockey'
    : (!jockeys.length && !trainers.length && /\btrainers?\b/.test(t)) ? 'trainer' : (!jockeys.length && !trainers.length && /\b(jockeys?|riders?)\b/.test(t)) ? 'jockey' : null;
  const hasFilter = Object.keys(filters).some((k) => k !== 'surface' || /\b(turf|grass|dirt|synthetic)\b/.test(t));
  const specialSlice = /churchill/.test(t) && Object.keys(filters).length === 1;   // the built-in slices already cover Churchill alone
  if (personRole && !horses.length && hasFilter && !(specialSlice && (trainers.length || jockeys.length))) {
    return { tool: 'record', args: { role: personRole, name: personRole === 'trainer' ? (trainers[0] || null) : (jockeys[0] || null), filters } };
  }
  if (cond && !jockeys.length && !trainers.length) {
    return { tool: 'condition_records', args: { role: 'horse', condition: cond, name: horses[0] || null } };
  }
  if (!horses.length && !jockeys.length && !trainers.length && /\b(horses?|field|entrants?|runners?|starters?)\b/.test(t)
      && /\b(won|wins?|winners?)\b/.test(t) && !/won the (kentucky )?derby|derby winners?/.test(t))
    return { tool: 'horse_wins', args: { filters } };
  if (/derb(y|ies)/.test(t) && /\b(history|before|past|previous|prior|since|last \w+ years|years|winners)\b/.test(t))
    return { tool: 'derby_record', args: { name: jockeys[0] || trainers[0] || null } };
  if (/\b(who won|result|payoff|paid|order of finish)\b/.test(t)) return { tool: 'results', args: {} };
  if (horses.length >= 2 || (horses.length && /\b(compare|vs|versus)\b/.test(t))) return { tool: 'compare_horses', args: { horses } };
  if (horses.length === 1) return { tool: /\b(all|every|past perf|history|starts)\b/.test(t) ? 'past_performances' : 'horse_profile', args: { horse: horses[0] } };
  if (jockeys.length && trainers.length) return { tool: 'jockey_trainer', args: { jockey: jockeys[0], trainer: trainers[0] } };
  if (jockeys.length) return { tool: 'jockey_stats', args: { jockey: jockeys[0] } };
  if (trainers.length) return { tool: 'trainer_stats', args: { trainer: trainers[0] } };
  if (/\b(pace|lead|early speed|front|closer)\b/.test(t)) return { tool: 'pace_map', args: {} };
  if (/\b(speed figure|figures|fastest|beyer)\b/.test(t)) return { tool: 'top_figures', args: {} };
  if (/\b(new jockey|jockey change|first time|rider change)\b/.test(t)) return { tool: 'jockey_changes', args: {} };
  if (/\b(ai|model|probabilit|chance|value|overlay)\b/.test(t)) return { tool: 'ai_predictions', args: {} };
  if (/\b(field|line ?up|entries|who is running|post positions?)\b/.test(t)) return { tool: 'field_overview', args: {} };
  return null;
}

// ------------------------------------------------------------------ LLM (WebLLM, optional)
// Verbose data dictionary (the old Lynk Lab sent descriptions like these on every call). To stay token-efficient, only the
// tables a question needs are sent, plus the real values of the categorical columns read from the data at load time.
const DICTIONARY = {
  field: `TABLE field -- the 2025 Kentucky Derby (Churchill Downs, May 3 2025, 1 1/4 miles on dirt, sloppy track, Grade 1, $5,000,000). One row per entry, 21 rows (19 starters + 2 scratched).
  post_position INTEGER: post position as drawn (also the program number); NULL for scratched horses
  starting_gate INTEGER: gate the horse actually broke from after the scratches
  horse VARCHAR: horse name, exactly as listed below; sex VARCHAR: Colt or Gelding; color VARCHAR
  sire, dam, damsire VARCHAR: pedigree (father, mother, mother's father); owner VARCHAR; breeder VARCHAR
  jockey VARCHAR, trainer VARCHAR: names exactly as listed below; weight INTEGER: pounds carried (126 for all)
  medication VARCHAR ('L' Lasix, 'B' Bute, NULL none); equipment VARCHAR ('B' blinkers, 'F' front bandages, NULL none)
  scratched BOOLEAN: TRUE if the horse did not run (always filter with NOT scratched for starters)
  morning_line VARCHAR: track handicapper's odds like '5-1' or '20-1'; NULL for the two Japan-based horses
  final_odds DOUBLE: odds to $1 at the close (7.98 means 7.98-1); finish INTEGER: official finish position (1 = won)
  beaten_lengths DOUBLE: lengths behind the winner at the finish; speed_figure INTEGER: DRF speed figure for this race
  pos_start, pos_quarter, pos_half, pos_three_quarters, pos_stretch INTEGER: running position at each call (1 = leading)
  lengths_behind_quarter, lengths_behind_half, lengths_behind_three_quarters, lengths_behind_stretch DOUBLE
  trip_comment VARCHAR: chart comment, e.g. 'bid 5w, led, gave in'; win_payoff, place_payoff, show_payoff DOUBLE: $ per $2 ticket (0 if not paid)
  earnings DOUBLE: purse money won in this race
  ai_win_probability DOUBLE 0-1: Lynk fundamental model win probability made BEFORE the race (no odds used); ai_rank INTEGER (1 = model's top pick)`,
  past_performances: `TABLE past_performances -- every earlier North American start of the 2025 Derby horses (114 rows, Aug 2024 to Apr 2025). Japan-based horses have no rows.
  race_id VARCHAR; horse VARCHAR: Derby horse name (join to field.horse); race_date DATE; track_code VARCHAR (e.g. 'CD' Churchill Downs, 'GP' Gulfstream Park, 'KEE' Keeneland, 'SA' Santa Anita, 'OP' Oaklawn Park, 'FG' Fair Grounds, 'AQU' Aqueduct, 'SAR' Saratoga)
  track_name VARCHAR; race_number INTEGER; race_name VARCHAR: stakes name such as 'Santa Anita Derby' (empty for maiden or allowance races)
  class_codes VARCHAR: DRF class code; race_type VARCHAR: 'MSW' maiden special weight, 'MCL' maiden claiming, 'ALW' allowance, 'AOC' allowance optional claiming, 'STK' stakes
  grade VARCHAR: '1', '2', '3' for graded stakes, NULL otherwise; purse DOUBLE in $; distance_furlongs DOUBLE (8 = one mile, 10 = 1 1/4 miles)
  surface VARCHAR: 'Dirt', 'Turf' or 'Synthetic'; track_condition VARCHAR (see values below); field_size INTEGER: number of starters
  post INTEGER: post position; program_number VARCHAR; jockey VARCHAR; trainer VARCHAR; owner VARCHAR; weight INTEGER; medication VARCHAR; equipment VARCHAR
  final_odds DOUBLE: odds to $1; is_favorite BOOLEAN: lowest final odds in that race; finish INTEGER (1 = won); beaten_lengths DOUBLE
  speed_figure INTEGER: DRF speed figure; win_payoff, place_payoff, show_payoff DOUBLE: $ per $2 ticket (0 if not paid); earnings DOUBLE: $ won in that race
  trip_comment VARCHAR; claimed BOOLEAN; final_time_seconds DOUBLE: winner's time; winner VARCHAR: name of the horse that won that race
  position_start, position_first_call, position_stretch INTEGER: running positions (1 = leading); lengths_behind_first_call, lengths_behind_stretch DOUBLE`,
  starts: `TABLE starts -- every start from 2012-01-01 to 2025-05-02 by any horse, jockey or trainer in the 2025 Derby field (273,455 rows). Rows include other horses ridden by the Derby jockeys and trained by the Derby trainers.
  race_id VARCHAR; race_date DATE; track_code VARCHAR; track_name VARCHAR; race_number INTEGER; race_name VARCHAR (stakes name, often empty); class_codes VARCHAR
  race_type VARCHAR (same codes as past_performances); grade VARCHAR ('1','2','3' or NULL); purse DOUBLE; distance_furlongs DOUBLE; surface VARCHAR; track_condition VARCHAR
  field_size INTEGER; horse VARCHAR; post INTEGER; program_number VARCHAR; jockey VARCHAR; trainer VARCHAR; owner VARCHAR; weight INTEGER; medication VARCHAR; equipment VARCHAR
  final_odds DOUBLE; is_favorite BOOLEAN; finish INTEGER (1 = won); beaten_lengths DOUBLE; speed_figure INTEGER
  win_payoff, place_payoff, show_payoff DOUBLE: $ per $2 ticket, 0 when the bet lost; earnings DOUBLE; trip_comment VARCHAR; claimed BOOLEAN; final_time_seconds DOUBLE
  is_derby_horse, is_derby_jockey, is_derby_trainer BOOLEAN: whether the row's horse / jockey / trainer is in the 2025 Derby field
  IMPORTANT: "Derby horses" means WHERE is_derby_horse (the 2025 field). is_derby_trainer and is_derby_jockey are only for questions about trainers or jockeys; they include many other horses.
  To limit to the Derby trainers use WHERE is_derby_trainer (likewise is_derby_jockey, is_derby_horse). For one person use WHERE trainer = 'exact name'.`,
  derby_history: `TABLE derby_history -- every Kentucky Derby runner from 2012 to 2024 (247 rows, one per horse per year; the 2020 Derby was run in September).
  year INTEGER; race_date DATE; horse VARCHAR; post INTEGER; jockey VARCHAR; trainer VARCHAR; owner VARCHAR; final_odds DOUBLE; is_favorite BOOLEAN
  finish INTEGER (1 = won, as published in the race-day chart; Medina Spirit 2021 was later disqualified); beaten_lengths DOUBLE; speed_figure INTEGER
  win_payoff, place_payoff, show_payoff DOUBLE ($ per $2); track_condition VARCHAR; surface VARCHAR; trip_comment VARCHAR
  Also the same race columns as starts (race_id, race_name, race_number, track_code, track_name, grade, race_type, purse, distance_furlongs, field_size, program_number, weight, medication, equipment, earnings, claimed, final_time_seconds).`,
  race: `TABLE race -- one row describing the 2025 Kentucky Derby: race_name, track_name, race_date, race_number, grade, purse, distance_furlongs, surface, track_condition, weather, temperature,
  post_time_str, final_time_seconds (winner's time in seconds), fraction_quarter, fraction_half, fraction_mile (fractional times in hundredths of a second), track_variant`,
  exotic_payoffs: `TABLE exotic_payoffs -- official exotic payoffs for the 2025 Kentucky Derby (exacta, trifecta, superfecta, daily doubles and pick-N bets ending in the Derby):
  race_id VARCHAR; wager_type VARCHAR; winning VARCHAR (winning combination); minimum DOUBLE (base bet); payoff DOUBLE; pool_total DOUBLE`,
};
const NO_DATA_SCOPE = `The data does NOT include: workouts, veterinary records, weather or track conditions of other days except as recorded per race, results after May 3 2025, races outside North America, sire or dam statistics beyond the Derby horses' own races, betting pool sizes over time, or any race other than those described.`;
const VALUE_HINTS = {};   // filled from the data at load time

function tablesFor(question) {
  const t = question.toLowerCase();
  const pick = new Set(['field']);
  if (/jockey|rider|trainer|roi|return|since|career|record|win (rate|percent|%)|track|churchill|keeneland|saratoga|favorite|graded|stakes|condition|mud|slop|wet|turf|dirt/.test(t)) pick.add('starts');
  if (state.horses.some((h) => t.includes(h.toLowerCase())) || /last race|previous|prep|figure|form|past|beat|won the|finished|pace|speed|early/.test(t)) pick.add('past_performances');
  if (/derb(y|ies)|year|history|previous derb|last \w+ years/.test(t)) pick.add('derby_history');
  if (/fraction|time|weather|temperature|variant|post time/.test(t)) pick.add('race');
  if (/exacta|trifecta|superfecta|double|pick|payoff|paid/.test(t)) pick.add('exotic_payoffs');
  if (pick.size === 1) { pick.add('past_performances'); pick.add('starts'); }
  return [...pick];
}

function schemaFor(question) {
  const tables = tablesFor(question);
  const hints = Object.entries(VALUE_HINTS).map(([k, v]) => `${k}: ${v}`).join('\n');
  return `${tables.map((t) => DICTIONARY[t]).join('\n\n')}\n\nActual values in the data:\n${hints}\n\nFormulas: win% = 100*avg((finish=1)::INT); in-the-money% = 100*avg((finish<=3)::INT); ROI% on $2 win bets = 100*(sum(win_payoff)-2*count(*))/(2*count(*)).\n${NO_DATA_SCOPE}`;
}

async function loadValueHints() {
  const vals = async (sql) => (await q(sql)).map((r) => Object.values(r)[0]).filter((v) => v !== null && v !== '').join(', ');
  VALUE_HINTS.track_condition = await vals(`SELECT DISTINCT track_condition FROM starts ORDER BY 1`);
  VALUE_HINTS.surface = await vals(`SELECT DISTINCT surface FROM starts ORDER BY 1`);
  VALUE_HINTS.race_type = await vals(`SELECT DISTINCT race_type FROM starts ORDER BY 1`);
  VALUE_HINTS.grade = await vals(`SELECT DISTINCT grade FROM starts WHERE grade IS NOT NULL ORDER BY 1`);
  VALUE_HINTS['track_code (most used, code = name)'] = await vals(`SELECT track_code || ' = ' || any_value(track_name) FROM starts GROUP BY track_code ORDER BY count(*) DESC LIMIT 30`);
  VALUE_HINTS['exotic wager_type'] = await vals(`SELECT DISTINCT wager_type FROM exotic_payoffs ORDER BY 1`);
}

const SQL_EXAMPLES = [
  ['How fast did Journalism work out before the Derby?', 'NO_DATA: the data has race results only, no workouts.'],
  ['Which Derby trainers have won the most graded stakes since 2023?', `SELECT trainer, count(*) AS graded_wins FROM starts WHERE is_derby_trainer AND grade IS NOT NULL AND finish = 1 AND race_date >= DATE '2023-01-01' GROUP BY trainer ORDER BY graded_wins DESC LIMIT 10`],
  ['Which Derby horses have won a Grade 1?', `SELECT horse, count(*) AS grade1_wins, string_agg(race_name, '; ') AS races FROM starts WHERE is_derby_horse AND grade = '1' AND finish = 1 GROUP BY horse ORDER BY grade1_wins DESC`],
  ['How do Derby horses that raced at Keeneland do in their last start?', `SELECT horse, race_date, race_name, finish, speed_figure FROM past_performances WHERE track_code = 'KEE' ORDER BY race_date DESC`],
  ['Which Derby trainer wins most often on sloppy tracks?', `SELECT trainer, count(*) AS starts, round(100*avg((finish=1)::INT),1) AS win_pct FROM starts WHERE is_derby_trainer AND track_condition = 'Sloppy' AND surface = 'Dirt' GROUP BY trainer HAVING count(*) >= 20 ORDER BY win_pct DESC`],
  ['What is Flavien Prat\'s ROI on Churchill Downs favorites?', `SELECT count(*) AS starts, round(100*avg((finish=1)::INT),1) AS win_pct, round(100*(sum(win_payoff)-2*count(*))/(2*count(*)),1) AS roi_pct FROM starts WHERE jockey = 'Flavien Prat' AND track_code = 'CD' AND is_favorite`],
];

function namesBlock() {
  return `Horses: ${state.horses.join('; ')}\nJockeys: ${state.jockeys.join('; ')}\nTrainers: ${state.trainers.join('; ')}`;
}

async function llm(messages, maxTokens = 220, temperature = 0.3) {
  const r = await state.engine.chat.completions.create({ messages, max_tokens: maxTokens, temperature });
  const used = (r.usage?.prompt_tokens || 0) + (r.usage?.completion_tokens || 0);
  state.tokens += used;
  $('meter').textContent = `Tokens used this session: ${state.tokens.toLocaleString()} (all on your device)`;
  return { text: r.choices[0].message.content.trim(), usage: r.usage || {} };
}

async function chooseTool(question) {   // Tier 1
  const catalog = Object.entries(TOOLS).map(([k, v]) => `${k}(${(v.args || []).join(', ')}): ${v.desc}`).join('\n');
  const { text, usage } = await llm([
    { role: 'system', content: `Route questions about the 2025 Kentucky Derby to one tool. Reply with JSON only, like {"tool":"jockey_stats","args":{"jockey":"Flavien Prat"}}. Only choose a tool if it answers the question exactly as asked. If the question needs a filter or grouping the tool does not mention (a distance, surface, track, year, race type, sire or owner), reply {"tool":"sql"}.\nTools:\n${catalog}\n${namesBlock()}` },
    { role: 'user', content: question }], 80, 0);
  try { return { ...JSON.parse(text.match(/\{[\s\S]*\}/)[0]), usage }; } catch { return { tool: 'sql', usage }; }
}

function safeSql(sql) {
  let s = sql.replace(/```sql|```/g, '').trim().replace(/;+\s*$/, '');
  if (!/^(select|with)\b/i.test(s) || /;/.test(s)) throw new Error('Only a single SELECT is allowed.');
  if (/\b(insert|update|delete|create|drop|alter|attach|copy|install|load|pragma|export|set)\b/i.test(s.replace(/'[^']*'/g, ''))) throw new Error('Read-only queries only.');
  return `SELECT * FROM (${s}) LIMIT 100`;
}

async function writeSql(question, error = null, previous = null) {   // Tier 2
  const shots = SQL_EXAMPLES.flatMap(([u, a]) => [{ role: 'user', content: u }, { role: 'assistant', content: a }]);
  const msgs = [{ role: 'system', content: `You write one DuckDB SQL query that answers a question about the 2025 Kentucky Derby data. Reply with the SQL only, no explanation. Use only the tables and columns described below, with names spelled exactly as listed. If the question cannot be answered from these tables, reply exactly: NO_DATA: <one-sentence reason>.\n\n${schemaFor(question)}\n\n${namesBlock()}` }, ...shots,
    { role: 'user', content: question }];
  if (error) msgs.push({ role: 'assistant', content: previous }, { role: 'user', content: `That failed with: ${error}. Fix the query using only the columns described. SQL only, or NO_DATA: <reason>.` });
  return llm(msgs, 220, 0);
}

function compactRows(rows, maxRows = 15) {
  if (!rows.length) return '(no rows)';
  if (rows.length <= 5) {   // few rows: label every value so a small model cannot swap columns
    return rows.map((r, i) => `row ${i + 1}: ` + Object.entries(r).slice(0, 14).map(([k, v]) => `${k.replace(/_/g, ' ')} = ${fmt(v).replace(/,/g, '')}`).join('; ')).join('\n');
  }
  const cols = Object.keys(rows[0]).slice(0, 12);
  const line = (r) => cols.map((c) => fmt(r[c]).replace(/,/g, '')).join(',');
  return [cols.join(','), ...rows.slice(0, maxRows).map(line)].join('\n') + (rows.length > maxRows ? `\n(${rows.length - maxRows} more rows not shown)` : '');
}

async function narrate(question, rows, title) {   // Answer step: the model sees only these rows, and is told what they are
  return llm([
    { role: 'system', content: 'You are a concise horse racing analyst for the 2025 Kentucky Derby. Answer in 2-5 sentences using ONLY the data rows given. Use plain English, not column names. Never invent numbers, and never claim a filter (track condition, distance, surface, year, track) that is not stated in the data description. If the rows do not answer the question, say so in one sentence and describe what they do show.' },
    { role: 'user', content: `Question: ${question}\nThe data below is: ${title}\nData:\n${compactRows(rows)}` }], 200, 0.2);
}

// ------------------------------------------------------------------ bet suggestions (vary every time)
async function suggestBet() {
  const field = state.field.filter((f) => !f.scratched);
  const weights = field.map((f) => Math.pow(f.ai_win_probability, 0.8));   // tempered: longshots get a real chance to be picked
  let r = Math.random() * weights.reduce((a, b) => a + b, 0), chosen = field[0];
  for (let i = 0; i < field.length; i++) { r -= weights[i]; if (r <= 0) { chosen = field[i]; break; } }
  const share = 100 * weights[field.indexOf(chosen)] / weights.reduce((a, b) => a + b, 0);
  const [prof] = await q(`${SUMMARY_PP} WHERE horse = '${esc(chosen.horse)}' GROUP BY horse`);
  const [jk] = await q(`SELECT round(100*avg((finish=1)::INT),1) AS win_pct, count(*) AS n FROM starts WHERE jockey = '${esc(chosen.jockey)}' AND race_date >= DATE '2024-05-03'`);
  const [tr] = await q(`SELECT round(100*avg((finish=1)::INT),1) AS win_pct, sum((grade IS NOT NULL AND finish=1)::INT) AS graded FROM starts WHERE trainer = '${esc(chosen.trainer)}' AND race_date >= DATE '2024-05-03'`);
  const odds = chosen.morning_line ? chosen.morning_line.split('-').map(Number) : null;
  const mlOdds = odds ? odds[0] / odds[1] : null;
  const mlPct = mlOdds !== null ? 100 / (mlOdds + 1) : null;
  const category = mlOdds === null ? 'no morning line' : mlOdds <= 10 ? 'favorite' : mlOdds < 30 ? 'middle of the pack' : 'longshot';
  const bet = { favorite: 'a Win bet, and use them as a key on top in exactas', 'middle of the pack': 'a Win/Place bet, and include them in exacta and trifecta boxes',
    longshot: 'a Place/Show bet, and use them underneath in trifectas and superfectas', 'no morning line': 'a small Win/Place bet' }[category];
  const angles = [];
  if (mlPct !== null && 100 * chosen.ai_win_probability > mlPct) angles.push(`the model makes them ${(100 * chosen.ai_win_probability).toFixed(1)}% to win against ${mlPct.toFixed(1)}% implied by the ${chosen.morning_line} morning line`);
  if (prof?.last3_avg_figure) angles.push(`their last three speed figures average ${prof.last3_avg_figure} with a best of ${prof.best_figure}`);
  if (prof?.graded_stakes_wins) angles.push(`they have ${prof.graded_stakes_wins} graded stakes win${prof.graded_stakes_wins > 1 ? 's' : ''}`);
  if (jk?.win_pct && jk.n > 0) angles.push(`${chosen.jockey} has won ${jk.win_pct}% of ${jk.n} rides over the past year`);
  if (tr?.win_pct && tr.graded !== null) angles.push(`${chosen.trainer} has won with ${tr.win_pct}% of starters over the past year, ${tr.graded} in graded stakes`);
  if (!prof) angles.push(`${chosen.horse} prepped outside North America, so the model rates it from its connections and pedigree (${chosen.sire} out of ${chosen.dam})`);
  if (angles.length < 2) angles.push(`the model gives them a ${(100 * chosen.ai_win_probability).toFixed(1)}% chance to win, ranked ${chosen.ai_rank} of ${field.length}`);
  const facts = angles.sort(() => Math.random() - 0.5).slice(0, 3);
  let text = `${chosen.horse} (${chosen.morning_line ? `${chosen.morning_line}, ${category}` : 'no morning line'}): ${facts.join('; ')}. Based on that, consider ${bet}.`;
  let usage = null;
  if (state.engine) {
    const out = await llm([{ role: 'system', content: 'You are an upbeat but honest horse racing analyst. In 3-4 sentences, make the case for betting this horse in the 2025 Kentucky Derby using ONLY the facts given, then recommend the bet given.' },
      { role: 'user', content: `Horse: ${chosen.horse}\nOdds category: ${category} (${chosen.morning_line || 'n/a'})\nFacts: ${facts.join('; ')}\nRecommended bet: ${bet}` }], 180, 0.8);
    text = out.text; usage = out.usage;
  }
  return { text, note: `Picked at random with a ${share.toFixed(1)}% chance, in proportion to the AI's win probabilities, so each click can surface a different horse.`, usage };
}

// ------------------------------------------------------------------ chat flow
function tableHtml(rows, max = 25) {
  if (!rows?.length) return '<p class="sub">No rows.</p>';
  const cols = Object.keys(rows[0]);
  return `<div class="table-wrap"><table><tr>${cols.map((c) => `<th>${html(c.replace(/_/g, ' '))}</th>`).join('')}</tr>${rows.slice(0, max)
    .map((r) => `<tr>${cols.map((c) => `<td>${html(fmt(r[c]))}</td>`).join('')}</tr>`).join('')}</table></div>${rows.length > max ? `<p class="sub">${rows.length - max} more rows</p>` : ''}`;
}

function addMsg(role, content) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = content;
  $('messages').appendChild(div);
  $('messages').scrollTop = $('messages').scrollHeight;
  return div;
}

const prose = (s) => html(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>');

async function answer(question) {
  addMsg('user', html(question));
  const box = addMsg('bot', '<span class="sub">Thinking…</span>');
  const trace = [];
  try {
    let plan = route(question);
    trace.push(plan ? `Tier 0 router → ${plan.tool} (0 tokens)` : 'Tier 0 router: no match');
    if (plan?.tool === 'suggest_bet') {
      const s = await suggestBet();
      box.innerHTML = `${prose(s.text)}<div class="trace">${html(s.note)}${s.usage ? ` · ${s.usage.prompt_tokens} prompt + ${s.usage.completion_tokens} generated tokens` : ''}</div>`;
      return refreshChips();
    }
    if (!plan && state.engine) {
      const choice = await chooseTool(question);
      trace.push(`Tier 1 tool choice → ${choice.tool} (${choice.usage.prompt_tokens || '?'} + ${choice.usage.completion_tokens || '?'} tokens)`);
      plan = choice.tool && choice.tool !== 'sql' && TOOLS[choice.tool] ? { tool: choice.tool, args: choice.args || {} } : { tool: 'sql' };
    }
    if (!plan) {
      box.innerHTML = `I can answer that once the AI model is loaded (it writes SQL for questions my built-in tools do not cover). Meanwhile, try one of the suggestions below.`;
      return refreshChips();
    }
    let rows, sql, title, extra;
    if (plan.tool === 'sql') {
      let gen = await writeSql(question);
      trace.push(`Tier 2 text-to-SQL with the data dictionary for ${tablesFor(question).join(', ')} (${gen.usage.prompt_tokens} + ${gen.usage.completion_tokens} tokens)`);
      if (/^\s*NO_DATA/i.test(gen.text)) {
        const reason = gen.text.replace(/^\s*NO_DATA:?\s*/i, '');
        box.innerHTML = `I can't answer that from the data I have. ${html(reason)}<div class="trace">${html(trace.join(' → '))}</div>`;
        return refreshChips();
      }
      const aboutHorses = /\bhorses?\b|\bfield\b/i.test(question) && !/\b(trainers?|jockeys?|riders?)\b/i.test(question);
      try {
        if (aboutHorses && /is_derby_(trainer|jockey)/.test(gen.text) && !/is_derby_horse/.test(gen.text))
          throw new Error('the question is about Derby horses, so filter with is_derby_horse, not is_derby_trainer or is_derby_jockey');
        sql = safeSql(gen.text); rows = await q(sql);
      }
      catch (e) {
        gen = await writeSql(question, e.message, gen.text);
        trace.push(`retry after error (${gen.usage.prompt_tokens} + ${gen.usage.completion_tokens} tokens)`);
        sql = safeSql(gen.text); rows = await q(sql);
      }
      title = `results of a custom SQL query: ${sql.replace(/\s+/g, ' ').slice(0, 400)}`;
    } else {
      const spec = await TOOLS[plan.tool].run(plan.args || {});
      sql = spec.sql; title = spec.title; rows = await q(sql);
      if (spec.extra) extra = await q(spec.extra);
    }
    trace.push(`DuckDB returned ${rows.length} rows`);
    const empty = !rows.length || rows.every((r) => Object.values(r).every((v) => v === null || v === 0 || v === ''));
    if (empty && !(extra && extra.length) && !['horse_profile', 'past_performances'].includes(plan.tool)) {
      box.innerHTML = `<b>${html(title.startsWith('results of a custom') ? 'Custom query' : title)}</b><br>I can't answer that from the data I have: nothing in the 2025 Derby dataset matches the question.
        <details><summary>SQL and steps</summary><pre>${html(sql)}</pre><div class="trace">${html(trace.join(' → '))}</div></details>`;
      return refreshChips();
    }
    let text;
    if (state.engine && plan.tool === 'sql') {
      const n = await narrate(question, extra ? [...extra, ...rows] : rows, title);
      text = n.text; trace.push(`answer (${n.usage.prompt_tokens} + ${n.usage.completion_tokens} tokens)`);
    } else {
      text = TOOLS[plan.tool]?.summarize ? TOOLS[plan.tool].summarize(rows, extra, plan.args || {}, question) : '';
    }
    box.innerHTML = `<b>${html(title.startsWith('results of a custom') ? 'Custom query' : title)}</b><br>${prose(text)}${tableHtml(extra ? extra : rows)}${extra ? tableHtml(rows) : ''}
      <details><summary>SQL and steps</summary><pre>${html(sql)}</pre><div class="trace">${html(trace.join(' → '))}</div></details>`;
  } catch (e) {
    box.innerHTML = `Sorry, I could not answer that one (${html(e.message)}). Try rephrasing, or pick a suggestion below.`;
  }
  refreshChips();
}

// ------------------------------------------------------------------ suggestions (different every time)
const CHIP_TEMPLATES = [
  'Show me {horse}\'s past performances', 'Compare {horse} and {horse2}', 'How has {jockey} done in Grade 1 races?',
  'What is {trainer}\'s record at Churchill Downs?', 'Who has the best speed figures?', 'Which horses have early speed?',
  'Which horses have a new jockey for the Derby?', 'How does {jockey} do riding for {trainer}?', 'Where does the AI see value?',
  '{trainer} in past Kentucky Derbies', 'Who won the Derby the last ten years?', 'What is {jockey}\'s ROI as the favorite?',
  'Which Derby horses have won a Grade 1?', 'Which trainer does best on a muddy track?', '{trainer} on wet tracks at Churchill Downs', 'Which jockey wins most in graded stakes at Keeneland?', 'How do the Derby jockeys do on sloppy tracks?', 'Who won and what did it pay?', 'Tell me about {horse}',
];
function refreshChips() {
  const f = state.field.filter((x) => !x.scratched);
  const fill = (t) => {
    const a = pick(f), b = pick(f.filter((x) => x !== a));
    return t.replace('{horse}', a.horse).replace('{horse2}', b.horse).replace('{jockey}', pick(f).jockey).replace('{trainer}', pick(f).trainer);
  };
  const chosen = [...CHIP_TEMPLATES].sort(() => Math.random() - 0.5).slice(0, 4).map(fill);
  $('chips').innerHTML = `<button class="chip special" data-q="Suggest a bet">Suggest a bet</button>` + chosen.map((c) => `<button class="chip" data-q="${html(c)}">${html(c)}</button>`).join('');
  document.querySelectorAll('.chip').forEach((b) => b.onclick = () => answer(b.dataset.q));
}

// ------------------------------------------------------------------ field table and horse drawer
function renderField() {
  const showResult = $('reveal').checked;
  const head = `<tr><th>PP</th><th>Horse</th><th>Jockey / Trainer</th><th>ML</th><th>AI win %</th>${showResult ? '<th>Odds</th><th>Finish</th>' : ''}</tr>`;
  const maxP = Math.max(...state.field.map((f) => f.ai_win_probability || 0));
  const rows = [...state.field].sort((a, b) => showResult ? ((a.finish ?? 99) - (b.finish ?? 99)) : ((a.post_position ?? 99) - (b.post_position ?? 99)))
    .map((f) => `<tr class="${f.scratched ? 'scratched' : 'clickable'}" data-horse="${html(f.horse)}">
      <td>${f.post_position ?? 'SCR'}</td><td><div class="horse">${html(f.horse)}</div><div class="sub">${html(f.sire)} – ${html(f.dam)}</div></td>
      <td>${html(f.jockey)}<div class="sub">${html(f.trainer)}</div></td><td>${html(f.morning_line || '–')}</td>
      <td>${f.ai_win_probability ? `${(100 * f.ai_win_probability).toFixed(1)}%<div class="bar" style="width:${(100 * f.ai_win_probability / maxP).toFixed(0)}%"></div>` : '–'}</td>
      ${showResult ? `<td>${f.final_odds ? f.final_odds.toFixed(2) : '–'}</td><td class="result">${f.finish ?? '–'}</td>` : ''}</tr>`).join('');
  $('field').innerHTML = head + rows;
  document.querySelectorAll('#field tr.clickable').forEach((tr) => tr.onclick = () => openHorse(tr.dataset.horse));
}

async function openHorse(horse) {
  const f = state.field.find((x) => x.horse === horse);
  const [s] = await q(`${SUMMARY_PP} WHERE horse = '${esc(horse)}' GROUP BY horse`);
  const pp = await q(`SELECT race_date, track_code, race_name, grade, distance_furlongs AS furlongs, surface, track_condition AS track, finish, field_size,
    beaten_lengths, speed_figure, final_odds, jockey, trip_comment, winner FROM past_performances WHERE horse = '${esc(horse)}' ORDER BY race_date DESC`);
  $('drawer-body').innerHTML = `<div class="eyebrow">Post ${f.post_position} · ${html(f.sex)} · ${html(f.color || '')}</div><h2 style="font-size:22px;margin:4px 0">${html(horse)}</h2>
    <div class="sub">${html(f.sire)} – ${html(f.dam)}, by ${html(f.damsire)} · Bred by ${html(f.breeder || '–')} · Owner ${html(f.owner)}</div>
    <div class="sub">Jockey ${html(f.jockey)} · Trainer ${html(f.trainer)} · Morning line ${html(f.morning_line || '–')}</div>
    <div class="stat-row">
      <div class="stat"><b>${(100 * f.ai_win_probability).toFixed(1)}%</b><span>AI win probability</span></div>
      <div class="stat"><b>${s?.starts ?? 0}-${s?.wins ?? 0}-${s?.seconds ?? 0}-${s?.thirds ?? 0}</b><span>Starts-wins-2nds-3rds</span></div>
      <div class="stat"><b>${s?.best_figure ?? '–'}</b><span>Best speed figure</span></div>
      <div class="stat"><b>${s?.last3_avg_figure ?? '–'}</b><span>Last 3 average</span></div>
      <div class="stat"><b>$${fmt(s?.earnings ?? 0)}</b><span>Earnings</span></div>
    </div>
    <div class="chips"><button class="chip" data-q="Tell me about ${html(horse)}">Ask the analyst</button><button class="chip" data-q="How has ${html(f.jockey)} done in Grade 1 races?">Jockey record</button><button class="chip" data-q="What is ${html(f.trainer)}'s record at Churchill Downs?">Trainer record</button></div>
    <h3 style="font-size:14px">Past performances</h3>${tableHtml(pp, 50)}`;
  $('drawer').classList.remove('hidden');
  document.querySelectorAll('#drawer .chip').forEach((b) => b.onclick = () => { $('drawer').classList.add('hidden'); answer(b.dataset.q); });
}

// ------------------------------------------------------------------ model loading
async function loadModel() {
  if (!navigator.gpu) {
    addMsg('bot', 'Your browser does not expose WebGPU, so the AI model cannot run here. Every suggestion and built-in question still works with tables and summaries. Chrome or Edge on a desktop with a recent GPU will run the model.');
    return;
  }
  $('load-llm').disabled = true; $('load-bar').classList.remove('hidden');
  const webllm = await import('https://esm.run/@mlc-ai/web-llm');
  const available = new Set(webllm.prebuiltAppConfig.model_list.map((m) => m.model_id));
  const modelId = MODEL_PREFS[$('model-select').value].find((m) => available.has(m)) || MODEL_PREFS.fast.find((m) => available.has(m));
  $('llm-status').textContent = `Loading ${modelId}…`;
  state.engine = await webllm.CreateMLCEngine(modelId, {
    initProgressCallback: (p) => { $('load-fill').style.width = `${Math.round((p.progress || 0) * 100)}%`; $('load-text').textContent = p.text; },
  }, { context_window_size: 8192 });
  $('load-bar').classList.add('hidden');
  $('llm-status').textContent = `AI ready: ${modelId.split('-q4')[0]} (on your GPU)`; $('llm-status').className = 'pill ok';
  $('load-llm').textContent = 'AI loaded';
  addMsg('bot', 'The AI model is loaded and running locally. It can now answer open-ended questions by choosing tools or writing SQL against the data.');
}

// ------------------------------------------------------------------ start
async function main() {
  try {
    await initDb();
    await loadValueHints();
    await loadTracks();
    state.field = await q(`SELECT *, round(100 * ai_win_probability, 1) AS ai_pct FROM field`);
    state.horses = state.field.filter((f) => !f.scratched).map((f) => f.horse);
    state.jockeys = [...new Set(state.field.filter((f) => !f.scratched).map((f) => f.jockey))];
    state.trainers = [...new Set(state.field.filter((f) => !f.scratched).map((f) => f.trainer))];
    const [r] = await q(`SELECT * FROM race`);
    const [n] = await q(`SELECT count(*) AS n FROM starts`);
    $('facts').textContent = `Churchill Downs · Saturday, May 3, 2025 · 1 1/4 miles, dirt (${r.track_condition.toLowerCase()}) · $${fmt(r.purse)} · Grade 1 · ${state.horses.length} starters`;
    $('db-status').textContent = `SQL engine ready · ${n.n.toLocaleString()} historical starts`; $('db-status').className = 'pill ok';
    renderField(); refreshChips();
    addMsg('bot', 'Welcome to Lynk Lab. Ask about any horse, jockey or trainer in the 2025 Kentucky Derby, or tap a suggestion. Built-in questions work right away; load the AI model to ask anything and get written answers.');
  } catch (e) {
    $('db-status').textContent = 'Data failed to load'; addMsg('bot', `Could not load the data: ${html(e.message)}`);
  }
}

$('reveal').onchange = renderField;
$('load-llm').onclick = () => loadModel().catch((e) => { $('llm-status').textContent = 'Model failed to load'; addMsg('bot', `The model could not load: ${html(e.message)}`); $('load-llm').disabled = false; });
$('ask').onsubmit = (e) => { e.preventDefault(); const v = $('question').value.trim(); if (v) { $('question').value = ''; answer(v); } };
$('close-drawer').onclick = () => $('drawer').classList.add('hidden');
$('drawer').onclick = (e) => { if (e.target.id === 'drawer') $('drawer').classList.add('hidden'); };
main();
