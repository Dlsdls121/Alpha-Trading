/* Dashboard renderer.
 *
 * The organising principle: a signal is only as useful as its reasoning, so the
 * factor breakdown is a first-class part of every card rather than a debug view.
 * Conviction is shown as a bar and never as a percentage probability, because it
 * is not one.
 */

const $ = (sel, root = document) => root.querySelector(sel);

const fmt = (n, dp = 2) =>
  n === null || n === undefined || Number.isNaN(n)
    ? "-"
    : Number(n).toLocaleString("en-IN", { minimumFractionDigits: dp, maximumFractionDigits: dp });

const fmt0 = (n) => fmt(n, 0);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const CATEGORY_LABELS = {
  trend: "Trend",
  momentum: "Momentum",
  positioning: "Options positioning",
  volatility: "Volatility & premium cost",
  cost: "Cost & timing",
  liquidity: "Liquidity",
  relative: "Relative strength",
  structure: "Price structure",
};

const DIRECTION_WORDS = { long: "Buy", short: "Buy puts", no_trade: "Stand aside" };

/* ---------------------------------------------------------------- render */

function renderBanners(data) {
  const out = [];

  if (!data.live_data) {
    const notes = (data.degraded || []).slice(0, 4).map((d) => `<li>${esc(d)}</li>`).join("");
    out.push(`<div class="banner warn">
      <strong>Simulated data.</strong> These numbers are generated, not live market data,
      so the signals below are illustrations of the reasoning - not tradeable calls.
      Set <code>ALPHA_DATA_MODE=live</code> to use real NSE data.
      ${notes ? `<ul>${notes}</ul>` : ""}
    </div>`);
  }

  const expiryToday = (data.expiries || []).filter((e) => e.is_expiry_day);
  if (expiryToday.length) {
    out.push(`<div class="banner info"><strong>Expiry today:</strong>
      ${expiryToday.map((e) => esc(e.symbol)).join(", ")}. Premium buying is blocked on
      expiry day - time value collapses within hours.</div>`);
  }

  const warnings = (data.expiries || []).map((e) => e.warning).filter(Boolean);
  if (warnings.length) {
    out.push(`<div class="banner info">${esc(warnings[0])}</div>`);
  }

  $("#banners").innerHTML = out.join("");
}

function convictionBlock(sig) {
  return `<div class="conv">
    <div class="conv-row"><span>Conviction</span><span>${sig.conviction}/100</span></div>
    <div class="conv-bar"><i class="conv-fill ${sig.direction}"
       style="width:${Math.max(sig.conviction, 2)}%"></i></div>
  </div>`;
}

function vetoBlock(sig) {
  const v = sig.scorecard?.vetoes || [];
  if (!v.length) return "";
  return `<div class="vetoes">${v
    .map(
      (x) => `<div class="veto ${esc(x.severity)}">
        <div><span class="vt">${x.severity === "block" ? "Blocked" : "Caution"}: ${esc(x.label)}</span>
        <span class="vd">${esc(x.detail)}</span></div>
      </div>`
    )
    .join("")}</div>`;
}

function factorRow(f) {
  // Contribution bar is scaled so the widest factor in view fills half the track.
  const pct = Math.min(Math.abs(f.contribution) * 55, 50);
  const bar =
    f.weight > 0 && Math.abs(f.contribution) > 0.001
      ? `<div class="contrib"><i class="${f.contribution >= 0 ? "pos" : "neg"}"
           style="width:${pct}%"></i></div>`
      : "";

  // weight 0 = shown as evidence but deliberately not voting on direction
  const chip =
    f.weight === 0
      ? `<span class="chip ctx" title="Context only - does not vote on direction">context</span>`
      : `<span class="chip ${esc(f.verdict)}">${esc(f.verdict)}</span>`;

  return `<div class="factor">
    <div class="f-head">
      ${chip}
      <span class="f-label">${esc(f.label)}</span>
      <span class="f-value">${esc(f.value || "")}</span>
    </div>
    <div class="f-detail">${esc(f.detail)}</div>
    ${bar}
  </div>`;
}

function whyBlock(sig) {
  const factors = sig.scorecard?.factors || [];
  if (!factors.length) return "";

  const groups = {};
  factors.forEach((f) => (groups[f.category] ||= []).push(f));

  const order = ["trend", "momentum", "relative", "positioning", "structure",
                 "volatility", "cost", "liquidity"];
  const cats = Object.keys(groups).sort(
    (a, b) => (order.indexOf(a) + 99) % 99 - ((order.indexOf(b) + 99) % 99)
  );

  const body = cats
    .map(
      (c) => `<div class="cat">
        <div class="cat-name">${esc(CATEGORY_LABELS[c] || c)}</div>
        ${groups[c].map(factorRow).join("")}
      </div>`
    )
    .join("");

  const agree = Math.round((sig.scorecard.agreement || 0) * 100);
  const score = sig.scorecard.raw_score;

  return `<details class="why">
    <summary>Why this call - ${factors.length} factors examined</summary>
    <div class="why-body">
      <p class="muted">Weighted score ${score >= 0 ? "+" : ""}${fmt(score, 2)}
      (-1 to +1), with ${agree}% of the directional weight on one side.
      Factors marked <span class="chip ctx">context</span> are shown as evidence but
      carry no vote on direction - expensive premium is a cost, not a market view.</p>
      ${body}
    </div>
  </details>`;
}

function invalidationBlock(sig) {
  if (!sig.invalidated_by?.length) return "";
  return `<details class="why">
    <summary>What would make this wrong</summary>
    <div class="why-body"><ul class="muted" style="margin:.4rem 0 0;padding-left:1.1rem">
      ${sig.invalidated_by.map((x) => `<li>${esc(x)}</li>`).join("")}
    </ul></div>
  </details>`;
}

function optionCard(sig) {
  const leg = sig.leg;
  const kv = [];
  if (sig.spot !== null) kv.push(["Spot", fmt0(sig.spot)]);
  if (sig.invalidation) kv.push(["Invalid below/above", fmt0(sig.invalidation)]);
  if (sig.targets?.length) kv.push(["Targets", sig.targets.map(fmt0).join(" / ")]);
  if (sig.horizon && sig.horizon !== "-") kv.push(["Horizon", sig.horizon]);

  const legHtml = leg
    ? `<div class="leg">
         <div class="leg-name">${esc(leg.tradingsymbol)} &nbsp;<span class="muted">at ~${fmt(leg.ltp)}</span></div>
         <div class="leg-why">${esc(leg.rationale)}</div>
       </div>`
    : "";

  return `<article class="card ${esc(sig.direction)}">
    <div class="card-top">
      <div class="card-title">
        <span class="sym">${esc(sig.symbol)}</span>
        <span class="badge ${esc(sig.direction)}">${esc(DIRECTION_WORDS[sig.direction] || sig.direction)}</span>
        <span class="spot">${sig.spot ? fmt0(sig.spot) : ""}</span>
      </div>
      ${convictionBlock(sig)}
      <p class="summary">${esc(sig.summary)}</p>
      ${kv.length ? `<dl class="kv">${kv.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>` : ""}
      ${legHtml}
      ${vetoBlock(sig)}
    </div>
    ${whyBlock(sig)}
    ${invalidationBlock(sig)}
  </article>`;
}

function equityCard(sig) {
  const kv = [
    ["Price", fmt(sig.spot)],
    ["Stop", fmt(sig.invalidation)],
  ];
  if (sig.targets?.length) kv.push(["Targets", sig.targets.map((t) => fmt(t)).join(" / ")]);
  if (sig.spot && sig.invalidation && sig.targets?.length) {
    const rr = (sig.targets[0] - sig.spot) / (sig.spot - sig.invalidation);
    kv.push(["Reward:risk", `${fmt(rr, 1)} : 1`]);
  }
  kv.push(["Horizon", sig.horizon || "-"]);

  return `<article class="card ${esc(sig.direction)}">
    <div class="card-top">
      <div class="card-title">
        <span class="sym">${esc(sig.symbol)}</span>
        <span class="badge ${esc(sig.direction)}">${sig.direction === "long" ? "Accumulate" : "Avoid"}</span>
      </div>
      ${convictionBlock(sig)}
      <p class="summary">${esc(sig.summary)}</p>
      <dl class="kv">${kv.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
      ${vetoBlock(sig)}
    </div>
    ${whyBlock(sig)}
    ${invalidationBlock(sig)}
  </article>`;
}

function sectorTable(rows) {
  if (!rows?.length) return `<div class="empty">No sector data.</div>`;
  return `<div class="scroll-x"><table>
    <thead><tr><th>#</th><th>Sector</th><th class="num">3m RS vs NIFTY</th>
      <th class="num">Names</th><th>Leaders</th></tr></thead>
    <tbody>${rows
      .map(
        (r) => `<tr>
        <td>${r.rank}</td><td>${esc(r.sector)}</td>
        <td class="num ${r.mean_rs_3m >= 0 ? "pos" : "neg"}">${r.mean_rs_3m >= 0 ? "+" : ""}${fmt(r.mean_rs_3m, 1)}</td>
        <td class="num">${r.constituents}</td>
        <td class="muted">${esc((r.leaders || []).join(", "))}</td>
      </tr>`
      )
      .join("")}</tbody></table></div>`;
}

function expiryTable(rows) {
  if (!rows?.length) return "";
  return `<div class="scroll-x"><table>
    <thead><tr><th>Symbol</th><th>Cycle</th><th>Next expiry</th>
      <th class="num">Days</th><th class="num">Sessions</th></tr></thead>
    <tbody>${rows
      .map(
        (e) => `<tr>
        <td><strong>${esc(e.symbol)}</strong></td>
        <td>${e.has_weekly ? "Weekly" : "Monthly only"}</td>
        <td>${esc(e.next_expiry)}${e.is_expiry_day ? " <span class='chip bearish'>today</span>" : ""}</td>
        <td class="num">${e.days_to_expiry}</td>
        <td class="num">${e.sessions_to_expiry}</td>
      </tr>`
      )
      .join("")}</tbody></table></div>`;
}

function render(data) {
  renderBanners(data);
  $("#asof").textContent = `As of ${data.as_of} - generated ${new Date(
    data.generated_at
  ).toLocaleTimeString()}`;
  $("#disclaimer").textContent = data.disclaimer || "";

  const opts = data.options || [];
  const eqs = data.equities || [];

  const errs = (data.errors || []).length
    ? `<div class="banner warn"><strong>Some analysis failed:</strong>
       ${data.errors.map((e) => `${esc(e.symbol)}: ${esc(e.error)}`).join("; ")}</div>`
    : "";

  $("#content").innerHTML = `
    ${errs}
    <section>
      <div class="sec-head"><h2>Index options</h2>
        <span class="hint">NIFTY &amp; BANKNIFTY - option buying</span></div>
      <div class="cards two">${
        opts.length ? opts.map(optionCard).join("") : `<div class="empty">No index signals.</div>`
      }</div>
    </section>

    <section>
      <div class="sec-head"><h2>Positional equity</h2>
        <span class="hint">multi-day holds, ranked by conviction</span></div>
      <div class="cards">${
        eqs.length
          ? eqs.map(equityCard).join("")
          : `<div class="empty">Nothing in the universe currently clears the bar for a positional long.
             That is a result, not an error - the filters are doing their job.</div>`
      }</div>
    </section>

    <section>
      <div class="sec-head"><h2>Sector rotation</h2>
        <span class="hint">mean 3-month relative strength of scanned constituents</span></div>
      ${sectorTable(data.sectors)}
    </section>

    <section>
      <div class="sec-head"><h2>Expiry calendar</h2>
        <span class="hint">BANKNIFTY has been monthly-only since Nov 2024</span></div>
      ${expiryTable(data.expiries)}
    </section>
  `;
}

/* ------------------------------------------------------------------ load */

async function load() {
  const btn = $("#refresh");
  btn.disabled = true;
  btn.textContent = "Loading...";
  try {
    const res = await fetch("/api/brief", { cache: "no-store" });
    if (!res.ok) throw new Error(`server returned ${res.status}`);
    render(await res.json());
  } catch (err) {
    $("#content").innerHTML = `<div class="banner warn">
      <strong>Could not load analysis.</strong> ${esc(err.message)}
      <br>Check the server is running, then tap Refresh.</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh";
  }
}

$("#refresh").addEventListener("click", load);
load();
