// api/submit.js  (Vercel serverless function)
//
// Receives one survey response (JSON) and appends it to a Redis list via
// Upstash's REST API. This replaces the earlier Netlify Forms approach,
// which only worked while the site itself was hosted on Netlify.
//
// Required environment variables (Vercel: Project Settings > Environment Variables):
//   UPSTASH_REDIS_REST_URL    - from your Upstash database dashboard
//   UPSTASH_REDIS_REST_TOKEN  - from your Upstash database dashboard

module.exports = async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Content-Type", "application/json");

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const URL_ = process.env.UPSTASH_REDIS_REST_URL;
  const TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!URL_ || !TOKEN) {
    res.status(500).json({ error: "Missing UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN." });
    return;
  }

  let record = req.body;
  if (typeof record === "string") {
    try {
      record = JSON.parse(record);
    } catch (e) {
      res.status(400).json({ error: "Invalid JSON body." });
      return;
    }
  }
  if (!record || typeof record !== "object") {
    res.status(400).json({ error: "Empty or invalid body." });
    return;
  }

  try {
    const r = await fetch(URL_, {
      method: "POST",
      headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
      body: JSON.stringify(["RPUSH", "responses", JSON.stringify(record)]),
    });
    const data = await r.json();
    if (data.error) throw new Error(data.error);
    res.status(200).json({ ok: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
