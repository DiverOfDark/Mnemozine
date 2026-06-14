// Drives the served Mnemozine SPA with the bundled Chromium and full-page
// screenshots each of the 8 screens. Used by the live verification pass.
import { chromium } from "playwright";
import fs from "node:fs";

const CHROME =
  "/var/home/diverofdark/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome";
const BASE = process.env.BASE ?? "http://127.0.0.1:8765";
const OUT = "/var/home/diverofdark/Projects/Mnemozine/docs/webui-screenshots";
fs.mkdirSync(OUT, { recursive: true });

// screen -> { path, waitFor (text that proves real data rendered) }
const screens = [
  { name: "dashboard", path: "/", waitFor: "Memories" },
  { name: "memories", path: "/memories", waitFor: "thiserror" },
  {
    name: "memory-detail",
    path: "/memories/mem-pref-old",
    waitFor: "anyhow",
  },
  { name: "graph", path: "/graph?entity=memory-layer", waitFor: null },
  { name: "recall", path: "/recall", waitFor: null },
  { name: "logs", path: "/logs", waitFor: "Ingested" },
  { name: "maintenance", path: "/maintenance", waitFor: "consolidate" },
  { name: "eval", path: "/eval", waitFor: null },
];

const consoleErrors = {};

const browser = await chromium.launch({
  executablePath: CHROME,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});
const ctx = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 1,
});
const page = await ctx.newPage();

const results = [];

for (const s of screens) {
  const errs = [];
  page.removeAllListeners("console");
  page.on("console", (m) => {
    if (m.type() === "error") errs.push(m.text());
  });
  page.on("pageerror", (e) => errs.push("pageerror: " + e.message));

  const url = BASE + s.path;
  await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
  // Give react-query a beat to fetch + paint.
  await page.waitForTimeout(1500);
  if (s.waitFor) {
    try {
      await page.getByText(s.waitFor, { exact: false }).first().waitFor({
        timeout: 8000,
      });
    } catch {
      // fall through; we still screenshot and report
    }
  }
  await page.waitForTimeout(800);

  // Detect a crashed/empty shell: an error boundary or totally empty root.
  const bodyText = (await page.locator("body").innerText()).trim();
  const file = `${OUT}/${s.name}.png`;
  await page.screenshot({ path: file, fullPage: true });

  const hasWait = s.waitFor ? bodyText.includes(s.waitFor) : true;
  consoleErrors[s.name] = errs;
  results.push({
    name: s.name,
    url,
    bytes: fs.statSync(file).size,
    textLen: bodyText.length,
    proofText: s.waitFor,
    proofFound: hasWait,
    consoleErrors: errs.length,
    firstError: errs[0] ?? null,
    snippet: bodyText.slice(0, 120).replace(/\s+/g, " "),
  });
  console.log(
    `[${s.name}] ${file} bytes=${fs.statSync(file).size} textLen=${bodyText.length} proof(${s.waitFor})=${hasWait} consoleErrors=${errs.length}`
  );
}

console.log("\nJSON_RESULTS=" + JSON.stringify(results));
await browser.close();
