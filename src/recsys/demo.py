DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GameMatch — Hybrid Recommender Demo</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #f6f7fb;
      --muted: #a6adbd;
      --panel: rgba(21, 25, 38, .82);
      --line: rgba(255, 255, 255, .09);
      --accent: #7c5cff;
      --accent-2: #25d0ab;
      --danger: #ff7387;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 0%, rgba(124, 92, 255, .26), transparent 34rem),
        radial-gradient(circle at 92% 18%, rgba(37, 208, 171, .16), transparent 28rem),
        #090b12;
      font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input { font: inherit; }
    .shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0 64px; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 20px; }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: -.02em; }
    .logo {
      width: 38px; height: 38px; display: grid; place-items: center; border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), #ad79ff); box-shadow: 0 10px 35px rgba(124, 92, 255, .35);
    }
    .status { color: var(--muted); font-size: 13px; display: flex; gap: 8px; align-items: center; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent-2); box-shadow: 0 0 16px var(--accent-2); }
    .hero { padding: 78px 0 46px; max-width: 860px; }
    .eyebrow { color: #b9aaff; font-weight: 700; text-transform: uppercase; letter-spacing: .13em; font-size: 12px; }
    h1 { margin: 13px 0 17px; font-size: clamp(42px, 7vw, 78px); line-height: .98; letter-spacing: -.055em; }
    .hero p { margin: 0; max-width: 700px; color: var(--muted); font-size: 18px; }
    .metrics { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 28px; }
    .metric { padding: 9px 12px; border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,.035); color: #d9ddea; }
    .metric strong { color: white; }
    .workspace { border: 1px solid var(--line); background: var(--panel); backdrop-filter: blur(20px); border-radius: 24px; overflow: hidden; box-shadow: 0 30px 90px rgba(0,0,0,.35); }
    .controls { padding: 22px; display: grid; grid-template-columns: auto 1fr auto; gap: 14px; border-bottom: 1px solid var(--line); }
    .tabs { display: flex; padding: 4px; gap: 4px; background: rgba(255,255,255,.045); border-radius: 12px; }
    .tab { border: 0; color: var(--muted); background: transparent; padding: 9px 13px; border-radius: 9px; cursor: pointer; }
    .tab.active { color: white; background: #34304d; }
    .search { width: 100%; border: 1px solid var(--line); color: white; background: rgba(4,6,12,.55); border-radius: 12px; padding: 0 15px; outline: none; }
    .search:focus { border-color: rgba(124,92,255,.8); box-shadow: 0 0 0 3px rgba(124,92,255,.14); }
    .run { border: 0; color: white; font-weight: 750; padding: 0 19px; border-radius: 12px; cursor: pointer; background: linear-gradient(135deg, var(--accent), #9c6cff); }
    .run:disabled { opacity: .6; cursor: wait; }
    .meta { min-height: 53px; padding: 14px 22px; color: var(--muted); display: flex; align-items: center; justify-content: space-between; gap: 14px; }
    .strategy { color: #aef1df; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .results { padding: 0 22px 24px; display: grid; grid-template-columns: repeat(auto-fill, minmax(205px, 1fr)); gap: 14px; }
    .card { min-width: 0; overflow: hidden; border: 1px solid var(--line); background: rgba(255,255,255,.035); border-radius: 16px; }
    .art { aspect-ratio: 16/10; display: grid; place-items: center; background: linear-gradient(135deg, #24273a, #11131d); color: #676e82; font-size: 34px; overflow: hidden; }
    .art img { width: 100%; height: 100%; object-fit: contain; background: #fff; }
    .copy { padding: 14px; }
    .rank { color: #a795ff; font-size: 12px; font-weight: 800; }
    .title { margin: 6px 0 4px; font-size: 15px; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .detail { color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .empty { grid-column: 1/-1; padding: 54px 20px; text-align: center; color: var(--muted); }
    .error { color: var(--danger); }
    footer { color: #72798c; padding-top: 22px; font-size: 12px; text-align: center; }
    @media (max-width: 720px) {
      .shell { width: min(100% - 20px, 1180px); padding-top: 18px; }
      .hero { padding: 52px 4px 34px; }
      .controls { grid-template-columns: 1fr; }
      .search, .run { min-height: 46px; }
      .meta { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand"><span class="logo">G</span> GameMatch</div>
      <div class="status"><span class="dot"></span><span id="health">Checking API</span></div>
    </header>

    <section class="hero">
      <div class="eyebrow">Portfolio recommender system</div>
      <h1>Find the next game worth playing.</h1>
      <p>A production-minded Top-K recommender blending collaborative preference
        signals with product metadata and explicit cold-start fallbacks.</p>
      <div class="metrics">
        <span class="metric"><strong>0.6428</strong> Hit Rate@10</span>
        <span class="metric"><strong>0.4288</strong> NDCG@10</span>
        <span class="metric"><strong>79.99%</strong> Catalog coverage@20</span>
        <span class="metric"><strong>10k</strong> Test users</span>
      </div>
    </section>

    <main class="workspace">
      <div class="controls">
        <div class="tabs" aria-label="Recommendation mode">
          <button class="tab active" data-mode="user">For a user</button>
          <button class="tab" data-mode="item">Similar items</button>
        </div>
        <input id="query" class="search" value="unknown-demo-user"
          aria-label="User or item ID" placeholder="Enter a user ID">
        <button id="run" class="run">Recommend</button>
      </div>
      <div class="meta">
        <span id="summary">Try the default ID to see the cold-start fallback.</span>
        <span id="strategy" class="strategy">ready</span>
      </div>
      <section id="results" class="results">
        <div class="empty">Recommendations will appear here.</div>
      </section>
    </main>
    <footer>Temporal holdout evaluation · BPR matrix factorization · TF-IDF metadata retrieval</footer>
  </div>

  <script>
    const state = { mode: "user" };
    const input = document.querySelector("#query");
    const run = document.querySelector("#run");
    const results = document.querySelector("#results");
    const summary = document.querySelector("#summary");
    const strategy = document.querySelector("#strategy");

    function safe(value, fallback = "Unavailable") {
      const text = value === null || value === undefined || value === "" ? fallback : String(value);
      return text.replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char]));
    }
    function imageUrl(value) {
      try {
        const url = new URL(value);
        return ["https:", "http:"].includes(url.protocol) ? url.href : null;
      } catch { return null; }
    }
    function card(item) {
      const source = imageUrl(item.image_url);
      const image = source
        ? `<img src="${source}" alt="">`
        : "🎮";
      const details = [item.store || item.main_category, item.average_rating ? `★ ${item.average_rating}` : null, item.price ? `$${item.price}` : null]
        .filter(Boolean).join(" · ") || `ASIN ${safe(item.item_id)}`;
      return `<article class="card">
        <div class="art">${image}</div>
        <div class="copy">
          <div class="rank">#${item.rank}</div>
          <h2 class="title" title="${safe(item.title, item.item_id)}">${safe(item.title, item.item_id)}</h2>
          <div class="detail">${safe(details)}</div>
        </div>
      </article>`;
    }
    async function recommend() {
      const value = input.value.trim();
      if (!value) { input.focus(); return; }
      run.disabled = true;
      run.textContent = "Ranking…";
      results.innerHTML = '<div class="empty">Scoring candidates…</div>';
      const endpoint = state.mode === "user" ? `/recommend/${encodeURIComponent(value)}?k=8` : `/similar/${encodeURIComponent(value)}?k=8`;
      try {
        const response = await fetch(endpoint);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Request failed");
        strategy.textContent = data.strategy;
        summary.textContent = state.mode === "user"
          ? `Top ${data.recommendations.length} results for ${data.user_id}`
          : `Items related to ${safe(data.seed_item?.title, data.seed_item?.item_id)}`;
        results.innerHTML = data.recommendations.map(card).join("");
      } catch (error) {
        strategy.textContent = "unavailable";
        summary.textContent = error.message;
        summary.classList.add("error");
        results.innerHTML = '<div class="empty">This mode requires a metadata-aware hybrid artifact.</div>';
      } finally {
        run.disabled = false;
        run.textContent = "Recommend";
      }
    }
    document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(item => item.classList.toggle("active", item === tab));
      state.mode = tab.dataset.mode;
      input.placeholder = state.mode === "user" ? "Enter a user ID" : "Enter a parent ASIN";
      input.value = state.mode === "user" ? "unknown-demo-user" : "";
      summary.classList.remove("error");
      summary.textContent = state.mode === "user"
        ? "Try the default ID to see the cold-start fallback."
        : "Enter a product parent ASIN to explore metadata similarity.";
      strategy.textContent = "ready";
      results.innerHTML = '<div class="empty">Recommendations will appear here.</div>';
    }));
    run.addEventListener("click", recommend);
    input.addEventListener("keydown", event => { if (event.key === "Enter") recommend(); });
    fetch("/health").then(response => {
      document.querySelector("#health").textContent = response.ok ? "API online" : "API unavailable";
    }).catch(() => document.querySelector("#health").textContent = "API unavailable");
  </script>
</body>
</html>
"""
