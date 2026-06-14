// Drives the Recall playground: types a query, runs recall, and screenshots the
// ranked results + SessionStart index preview once populated.
import { chromium } from "playwright";
import fs from "node:fs";

const CHROME =
  "/var/home/diverofdark/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome";
const BASE = process.env.BASE ?? "http://127.0.0.1:8765";
const OUT = "/var/home/diverofdark/Projects/Mnemozine/docs/webui-screenshots";

const browser = await chromium.launch({
  executablePath: CHROME,
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
});
const errs = [];
page.on("console", (m) => m.type() === "error" && errs.push(m.text()));
page.on("pageerror", (e) => errs.push("pageerror: " + e.message));

await page.goto(BASE + "/recall", { waitUntil: "networkidle" });
await page.waitForTimeout(800);

// Type a query into the QUERY input and run. Placeholder uses a unicode
// ellipsis (…); target via attribute substring so it never grabs the topbar.
const queryInput = page.locator('input[placeholder^="what would the agent"]');
await queryInput.click();
await queryInput.fill("rust error handling preferences");
// Scope defaults from useScope (empty=all). Press Enter to run recall.
await queryInput.press("Enter");

// Wait for results to render (a "why it surfaced" note / score / index text).
try {
  await page.getByText("thiserror", { exact: false }).first().waitFor({
    timeout: 12000,
  });
} catch {
  /* still screenshot */
}
await page.waitForTimeout(1500);

const body = (await page.locator("body").innerText()).trim();
await page.screenshot({ path: `${OUT}/recall.png`, fullPage: true });

const proof = {
  hasThiserror: body.includes("thiserror"),
  hasIndexPreview:
    body.toLowerCase().includes("index") || body.includes("token"),
  hasBudget: body.includes("500") || body.toLowerCase().includes("budget"),
  consoleErrors: errs.length,
  firstError: errs[0] ?? null,
};
console.log("RECALL_PROOF=" + JSON.stringify(proof));
console.log("BODY_SNIPPET=" + body.slice(0, 400).replace(/\s+/g, " "));
await browser.close();
