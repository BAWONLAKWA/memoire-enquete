// api/get-stats.js  (Vercel serverless function)
//
//   GET /api/get-stats
//     -> public, aggregate-only: just ob/la/ab/rp/ai scores per respondent.
//        Used by the survey's own end-screen comparison.
//
//   GET /api/get-stats?full=true&key=XXXX
//     -> full records (demographics + scores + raw item answers), gated by
//        DASHBOARD_KEY. Used by the "Espace chercheur" dashboard.
//
// Required environment variables (Vercel: Project Settings > Environment Variables):
//   UPSTASH_REDIS_REST_URL    - from your Upstash database dashboard
//   UPSTASH_REDIS_REST_TOKEN  - from your Upstash database dashboard
//   DASHBOARD_KEY             - researcher code, e.g. "2026" (same as the in-app PIN)

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Content-Type", "application/json");

  const URL_ = process.env.UPSTASH_REDIS_REST_URL;
  const TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN;
  const DASHBOARD_KEY = process.env.DASHBOARD_KEY || "2026";

  if (!URL_ || !TOKEN) {
    res.status(500).json({ error: "Missing UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN." });
    return;
  }

  const wantsFull = req.query.full === "true";
  if (wantsFull && req.query.key !== DASHBOARD_KEY) {
    res.status(401).json({ error: "Invalid key" });
    return;
  }

  try {
    const r = await fetch(URL_, {
      method: "POST",
      headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify(["LRANGE", "responses", "0", "-1"]),
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);

    const records = (data.result || [])
      .map((s) => {
        try {
          return JSON.parse(s);
        } catch (e) {
          return null;
        }
      })
      .filter(Boolean);

    const payload = wantsFull
      ? records
      : records.map((rec) => ({ ob: rec.ob, la: rec.la, ab: rec.ab, rp: rec.rp, ai: rec.ai }));

    res.status(200).json(payload);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
