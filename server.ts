import express from "express";
import path from "node:path";
import {
  loadErrorContractJson,
  loadOmniExamples,
  runAcceptanceSuite,
} from "./src/omni/acceptance.ts";
import {
  executeOmniSurfaceAotFirst,
  explainOmniErrorCode,
  extractOmniRoutes,
  formatOmniHumanDiagnostic,
  handleOmniHttpRequest,
  listAllErrorExplanations,
  listOmniRepairStrategies,
  OMNI_ERROR_CONTRACT_VERSION,
  OMNI_HARNESS_CONTRACT_VERSION,
  OMNI_REPAIR_STRATEGIES_VERSION,
  omniHarnessSpecObj,
  parseCliOmniValue,
  parseOmniStructuredError,
  runOmniAgentDriver,
  runOmniHarness,
  runPhaseDPureOmniSelfhostBootstrap,
  wrapOmniSurfaceSnippet,
} from "./src/omni/lang.ts";

function parseArgsList(rawArgs: unknown): string[] {
  if (Array.isArray(rawArgs)) {
    return rawArgs.map((a) => String(a).trim()).filter(Boolean);
  }
  if (typeof rawArgs === "string") {
    const trimmed = rawArgs.trim();
    if (!trimmed) return [];
    return trimmed
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return [];
}

function formatHexTokens(u16Stream: number[]): string {
  const lines: string[] = [];
  for (let i = 0; i < u16Stream.length; i += 8) {
    const slice = u16Stream.slice(i, i + 8).map((w) => `0x${w.toString(16).toUpperCase().padStart(4, "0")}`);
    lines.push(`${i.toString().padStart(4, "0")}: ${slice.join(" ")}`);
  }
  return lines.join("\n");
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "2mb" }));

  const routeKvStore = new Map<bigint, bigint>();

  app.get("/api/health", (_req, res) => {
    res.json({
      status: "ok",
      harnessVersion: OMNI_HARNESS_CONTRACT_VERSION,
      errorContractVersion: OMNI_ERROR_CONTRACT_VERSION,
      repairStrategiesVersion: OMNI_REPAIR_STRATEGIES_VERSION,
    });
  });

  app.get("/api/examples", (_req, res) => {
    try {
      const examples = loadOmniExamples();
      res.json({ ok: true, examples });
    } catch (e) {
      res.status(500).json({ ok: false, error: e instanceof Error ? e.message : String(e) });
    }
  });

  app.get("/api/spec", (_req, res) => {
    try {
      res.json({
        ok: true,
        harness: omniHarnessSpecObj(),
        errorContract: loadErrorContractJson(),
        repairStrategies: listOmniRepairStrategies(),
        explanations: listAllErrorExplanations(),
        bootstrap: runPhaseDPureOmniSelfhostBootstrap(),
      });
    } catch (e) {
      res.status(500).json({ ok: false, error: e instanceof Error ? e.message : String(e) });
    }
  });

  app.get("/api/bootstrap", (_req, res) => {
    try {
      res.json({ ok: true, bootstrap: runPhaseDPureOmniSelfhostBootstrap() });
    } catch (e) {
      res.status(500).json({ ok: false, error: e instanceof Error ? e.message : String(e) });
    }
  });

  app.get("/api/explain/:code", (req, res) => {
    const info = explainOmniErrorCode(req.params.code);
    if (!info) {
      res.status(404).json({ ok: false, error: `Unknown error code: ${req.params.code}` });
      return;
    }
    res.json({ ok: true, explanation: info });
  });

  app.post("/api/run", (req, res) => {
    const src = typeof req.body?.src === "string" ? req.body.src : "";
    const strict = Boolean(req.body?.strict);
    const argStrings = parseArgsList(req.body?.args);
    const cliArgs = argStrings.map(parseCliOmniValue);

    try {
      const detail = executeOmniSurfaceAotFirst(src, cliArgs, strict);
      res.json({
        ok: true,
        result: detail.formattedResult,
        backend: detail.backend,
        tokens: detail.u16Stream.length,
        u16Stream: detail.u16Stream,
        hexStream: formatHexTokens(detail.u16Stream),
        asmText: detail.asmText,
        wasmSize: detail.wasmBytes ? detail.wasmBytes.length : null,
        wasmValid: detail.wasmValid,
        elfSize: detail.elfBytes ? detail.elfBytes.length : null,
        elfHash: detail.elfFnv1a64,
        inspectText: detail.inspectText,
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const errObj = parseOmniStructuredError(msg);
      const diagSrc = strict ? src : wrapOmniSurfaceSnippet(src);
      res.json({
        ok: false,
        error: errObj,
        humanDiagnostic: formatOmniHumanDiagnostic(diagSrc, errObj),
      });
    }
  });

  app.post("/api/harness", (req, res) => {
    const src = typeof req.body?.src === "string" ? req.body.src : "";
    const strict = Boolean(req.body?.strict);
    const includeRetryHint = req.body?.includeRetryHint !== false;
    const cliArgs = parseArgsList(req.body?.args).map(parseCliOmniValue);

    const harness = runOmniHarness(src, cliArgs, strict, includeRetryHint);
    let humanDiagnostic: string | null = null;
    if (!harness.ok && harness.code) {
      const diagSrc = strict ? src : wrapOmniSurfaceSnippet(src);
      humanDiagnostic = formatOmniHumanDiagnostic(diagSrc, {
        code: harness.code,
        line: harness.line ?? 1,
        col: harness.col ?? 1,
        message: harness.message ?? "",
        retryHint: harness.retry_hint ?? "",
      });
    }
    res.json({ ok: true, harness, humanDiagnostic });
  });

  app.post("/api/agent-repair", (req, res) => {
    const src = typeof req.body?.src === "string" ? req.body.src : "";
    const strict = Boolean(req.body?.strict);
    const maxAttemptsRaw = Number(req.body?.maxAttempts ?? 6);
    const maxAttempts = Number.isFinite(maxAttemptsRaw) ? Math.min(12, Math.max(1, maxAttemptsRaw)) : 6;
    const cliArgs = parseArgsList(req.body?.args).map(parseCliOmniValue);

    try {
      const report = runOmniAgentDriver(src, cliArgs, strict, maxAttempts);
      res.json({ ok: true, report });
    } catch (e) {
      res.status(400).json({ ok: false, error: e instanceof Error ? e.message : String(e) });
    }
  });

  app.post("/api/route-exec", (req, res) => {
    const src = typeof req.body?.src === "string" ? req.body.src : "";
    const method = typeof req.body?.method === "string" ? req.body.method : "GET";
    const reqPath = typeof req.body?.path === "string" ? req.body.path : "/";
    const resetKv = Boolean(req.body?.resetKv);

    if (resetKv) {
      routeKvStore.clear();
    }
    if (req.body?.kv && typeof req.body.kv === "object") {
      routeKvStore.clear();
      for (const [k, v] of Object.entries(req.body.kv as Record<string, string | number>)) {
        try {
          routeKvStore.set(BigInt(k), BigInt(v));
        } catch {
          // ignore invalid entry
        }
      }
    }

    try {
      const routes = extractOmniRoutes(src);
      const response = handleOmniHttpRequest(src, method, reqPath, routeKvStore);
      res.json({ ok: true, routes, response });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      const errObj = parseOmniStructuredError(msg);
      res.json({
        ok: false,
        error: errObj,
        humanDiagnostic: formatOmniHumanDiagnostic(src, errObj),
      });
    }
  });

  app.post("/api/acceptance", (_req, res) => {
    try {
      const report = runAcceptanceSuite();
      res.json({ ok: true, report });
    } catch (e) {
      res.status(500).json({ ok: false, error: e instanceof Error ? e.message : String(e) });
    }
  });

  if (process.env.NODE_ENV !== "production") {
    const { createServer } = await import("vite");
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), "dist");
    app.use(express.static(distDir));
    app.use((_req, res) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  }

  const port = Number(process.env.PORT || 3000);
  app.listen(port, "0.0.0.0", () => {
    console.log(`Omni Language Workbench listening on http://0.0.0.0:${port}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start Omni Language Workbench:", err);
  process.exit(1);
});
