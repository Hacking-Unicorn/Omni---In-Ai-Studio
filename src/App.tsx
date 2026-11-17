import React, { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  CheckCircle2,
  Code2,
  Cpu,
  Globe,
  Layers,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
  Terminal,
  Wrench,
  XCircle,
} from "lucide-react";

type WorkspaceTab = "playground" | "repair" | "routes" | "acceptance";
type InspectSubTab = "inspect" | "asm" | "hex" | "harness";
type SpecSubTab = "suite" | "bootstrap" | "errors" | "strategies";

interface OmniExample {
  id: string;
  category: "pure_program" | "positive_golden" | "negative_golden" | "pure_core" | "stdlib";
  path: string;
  name: string;
  source: string;
  args: string[];
  strict: boolean;
  expectedResult?: string;
  expectedBackend?: string;
  expectedErrorCode?: string;
  expectedLine?: number;
  expectedCol?: number;
  lockedTokens?: number | null;
}

interface StructuredError {
  code: string;
  line: number | null;
  col: number | null;
  message: string;
  retryHint: string;
}

interface RunResult {
  ok: boolean;
  result?: string;
  backend?: "aot" | "vm";
  tokens?: number;
  hexStream?: string;
  asmText?: string;
  wasmSize?: number | null;
  wasmValid?: boolean;
  elfSize?: number | null;
  elfHash?: string | null;
  inspectText?: string;
  error?: StructuredError;
  humanDiagnostic?: string;
}

interface BootstrapSummary {
  fixedPointVerified: boolean;
  gen1Words: number;
  gen2Words: number;
  bcFnv1a64: string;
  elfBytes: number;
  elfFnv1a64: string;
  stage1Words: number;
  stage1ElfBytes: number;
  stage1ElfFnv1a64: string;
  suiteWords: number;
  totalPureModules: number;
}

interface HarnessResult {
  ok: boolean;
  version: string;
  result?: string;
  tokens: number | null;
  backend?: "aot" | "vm";
  code?: string;
  line?: number | null;
  col?: number | null;
  message?: string;
  retry_hint?: string;
}

interface AgentHistoryEntry {
  attempt: number;
  ok: boolean;
  code?: string;
  line?: number;
  col?: number;
  message?: string;
  retry_hint?: string;
  strategy_id?: string;
  strategy_version?: string;
  source: string;
}

interface AgentDriverReport {
  ok: boolean;
  version: string;
  strategy_version: string;
  attempts: number;
  history: AgentHistoryEntry[];
  final_source: string;
  harness: HarnessResult;
}

interface RouteSpec {
  fnIdx: number;
  method: string;
  pathPattern: string;
}

interface RouteExecResponse {
  ok: boolean;
  routes?: RouteSpec[];
  response?: {
    status: number;
    contentType: string;
    body: string;
    kvSnapshot?: Record<string, string>;
  };
  error?: StructuredError;
  humanDiagnostic?: string;
}

interface AcceptanceCase {
  id: string;
  kind: "positive_golden" | "negative_golden" | "pure_program" | "error_contract";
  passed: boolean;
  detail: string;
  actual?: string;
  expected?: string;
  tokens?: number | null;
  backend?: string;
}

interface AcceptanceReport {
  passed: boolean;
  manifestVersion: string;
  harnessVersion: string;
  errorContractVersion: string;
  total: number;
  passedCount: number;
  failedCount: number;
  durationMs: number;
  cases: AcceptanceCase[];
}

interface ErrorExplanation {
  code: string;
  contract: string;
  stage: string;
  summary: string;
  hint: string;
  bad: string;
  repaired: string;
}

interface RepairStrategySpec {
  id: string;
  version: string;
  error_code: string;
  context_pattern: string;
  description: string;
}

const REPAIR_PRESETS: Array<{ label: string; args: string; strict: boolean; src: string }> = [
  {
    label: "Missing #[cap(AI)] on egate()",
    args: "12",
    strict: false,
    src: `fn main(x: i64) -> trit {\n  let g: trit = egate(x);\n  return g;\n}`,
  },
  {
    label: "@untrusted Taint Leak on Return",
    args: "42",
    strict: false,
    src: `fn main(x: i64 @untrusted) -> i64 {\n  return x;\n}`,
  },
  {
    label: "Linear Arena Leak (missing drop)",
    args: "9",
    strict: false,
    src: `fn main(x: i64) -> i64 {\n  let a: arena = arena(1024);\n  return x + 1;\n}`,
  },
  {
    label: "Borrow Conflict (drop order)",
    args: "7",
    strict: false,
    src: `fn main(x: i64) -> i64 {\n  let a: arena = arena(1024);\n  let r: i64 = borrow_r(a);\n  drop(a);\n  drop(r);\n  return x;\n}`,
  },
  {
    label: "#[strict] Ambiguous Grammar + Missing Type",
    args: "5",
    strict: true,
    src: `#[strict]\nfn main(x: int) -> trit {\n  let y = x;\n  return TRIT_POS;\n}`,
  },
  {
    label: "Corrupt @PACK5 Sentinel Byte (250)",
    args: "",
    strict: true,
    src: `#[strict]\nfn main() -> trit_vec {\n  let b: bytes = anew(0);\n  b = iset(b, 250);\n  let v: trit_vec = unpack5(b, 5);\n  return v;\n}`,
  },
];

const CATEGORY_LABELS: Record<OmniExample["category"], string> = {
  pure_program: "Pure Omni Programs",
  positive_golden: "Positive Goldens (24)",
  negative_golden: "Negative Goldens (15)",
  pure_core: "Pure Self-Host Core",
  stdlib: "Standard Library",
};

export function App() {
  const [tab, setTab] = useState<WorkspaceTab>("playground");
  const [examples, setExamples] = useState<OmniExample[]>([]);
  const [selectedExampleId, setSelectedExampleId] = useState<string>("");
  const [exampleFilter, setExampleFilter] = useState<string>("");
  const [categoryFilter, setCategoryFilter] = useState<string>("all");

  // Playground state
  const [source, setSource] = useState<string>("");
  const [argsInput, setArgsInput] = useState<string>("15, 40");
  const [strictMode, setStrictMode] = useState<boolean>(true);
  const [running, setRunning] = useState<boolean>(false);
  const [runResult, setRunResult] = useState<RunResult | null>(null);
  const [harnessResult, setHarnessResult] = useState<HarnessResult | null>(null);
  const [inspectTab, setInspectTab] = useState<InspectSubTab>("inspect");

  // Agent repair state
  const [repairSrc, setRepairSrc] = useState<string>(REPAIR_PRESETS[0].src);
  const [repairArgs, setRepairArgs] = useState<string>(REPAIR_PRESETS[0].args);
  const [repairStrict, setRepairStrict] = useState<boolean>(REPAIR_PRESETS[0].strict);
  const [repairMaxAttempts, setRepairMaxAttempts] = useState<number>(6);
  const [repairLoading, setRepairLoading] = useState<boolean>(false);
  const [repairReport, setRepairReport] = useState<AgentDriverReport | null>(null);

  // Route host state
  const [routeSrc, setRouteSrc] = useState<string>("");
  const [routeMethod, setRouteMethod] = useState<string>("POST");
  const [routePath, setRoutePath] = useState<string>("/ledger/42");
  const [routeResult, setRouteResult] = useState<RouteExecResponse | null>(null);
  const [routeLoading, setRouteLoading] = useState<boolean>(false);

  // Acceptance & spec state
  const [acceptanceReport, setAcceptanceReport] = useState<AcceptanceReport | null>(null);
  const [acceptanceLoading, setAcceptanceLoading] = useState<boolean>(false);
  const [acceptanceFilter, setAcceptanceFilter] = useState<string>("all");
  const [specTab, setSpecTab] = useState<SpecSubTab>("suite");
  const [explanations, setExplanations] = useState<ErrorExplanation[]>([]);
  const [strategies, setStrategies] = useState<RepairStrategySpec[]>([]);
  const [bootstrapSummary, setBootstrapSummary] = useState<BootstrapSummary | null>(null);
  const [contractVersions, setContractVersions] = useState<{
    harness: string;
    error: string;
    isa: string;
  }>({
    harness: "omni-harness-v1.0.0",
    error: "omni-error-contract-v1.0.0",
    isa: "0x7F00",
  });

  useEffect(() => {
    void (async () => {
      try {
        const [exRes, specRes] = await Promise.all([
          fetch("/api/examples").then((r) => r.json()),
          fetch("/api/spec").then((r) => r.json()),
        ]);
        if (exRes.ok && Array.isArray(exRes.examples)) {
          setExamples(exRes.examples);
          const initial =
            exRes.examples.find((e: OmniExample) => e.id === "pure_program:ternary_neural_router") ??
            exRes.examples[0];
          if (initial) {
            setSelectedExampleId(initial.id);
            setSource(initial.source);
            setArgsInput(initial.args.join(", "));
            setStrictMode(initial.strict);
          }
          const kvSvc = exRes.examples.find((e: OmniExample) => e.id === "pure_program:kv_ledger_service");
          if (kvSvc) {
            setRouteSrc(kvSvc.source);
          }
        }
        if (specRes.ok) {
          setExplanations(specRes.explanations ?? []);
          setStrategies(specRes.repairStrategies ?? []);
          if (specRes.bootstrap) {
            setBootstrapSummary(specRes.bootstrap);
          }
          setContractVersions({
            harness: specRes.harness?.version ?? "omni-harness-v1.0.0",
            error: specRes.errorContract?.contract_version ?? "omni-error-contract-v1.0.0",
            isa: specRes.harness?.isa_version ?? "0x7F00",
          });
        }
      } catch {
        // ignore initial fetch race
      }
    })();
  }, []);

  const filteredExamples = useMemo(() => {
    const q = exampleFilter.trim().toLowerCase();
    return examples.filter((ex) => {
      if (categoryFilter !== "all" && ex.category !== categoryFilter) return false;
      if (!q) return true;
      return (
        ex.name.toLowerCase().includes(q) ||
        ex.path.toLowerCase().includes(q) ||
        (ex.expectedErrorCode ?? "").toLowerCase().includes(q)
      );
    });
  }, [examples, exampleFilter, categoryFilter]);

  const routeExamples = useMemo(
    () => examples.filter((ex) => ex.source.includes("route ")),
    [examples],
  );

  const selectExample = (ex: OmniExample) => {
    setSelectedExampleId(ex.id);
    setSource(ex.source);
    setArgsInput(ex.args.join(", "));
    setStrictMode(ex.strict);
    setRunResult(null);
    setHarnessResult(null);
  };

  const executePlayground = async (overrideSrc?: string, overrideArgs?: string, overrideStrict?: boolean) => {
    setRunning(true);
    const s = overrideSrc ?? source;
    const a = overrideArgs ?? argsInput;
    const st = overrideStrict ?? strictMode;
    try {
      const [runRes, harRes] = await Promise.all([
        fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ src: s, args: a, strict: st }),
        }).then((r) => r.json()),
        fetch("/api/harness", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ src: s, args: a, strict: st, includeRetryHint: true }),
        }).then((r) => r.json()),
      ]);
      setRunResult(runRes);
      if (harRes.ok) {
        setHarnessResult(harRes.harness);
      }
    } finally {
      setRunning(false);
    }
  };

  const runAgentRepair = async (customSrc?: string, customArgs?: string, customStrict?: boolean) => {
    setRepairLoading(true);
    try {
      const res = await fetch("/api/agent-repair", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          src: customSrc ?? repairSrc,
          args: customArgs ?? repairArgs,
          strict: customStrict ?? repairStrict,
          maxAttempts: repairMaxAttempts,
        }),
      }).then((r) => r.json());
      if (res.ok) {
        setRepairReport(res.report);
      }
    } finally {
      setRepairLoading(false);
    }
  };

  const sendPlaygroundErrorToRepair = () => {
    setRepairSrc(source);
    setRepairArgs(argsInput);
    setRepairStrict(strictMode);
    setTab("repair");
    void runAgentRepair(source, argsInput, strictMode);
  };

  const dispatchRoute = async (resetKv = false, methodOverride?: string, pathOverride?: string) => {
    setRouteLoading(true);
    try {
      const res = await fetch("/api/route-exec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          src: routeSrc,
          method: methodOverride ?? routeMethod,
          path: pathOverride ?? routePath,
          resetKv,
        }),
      }).then((r) => r.json());
      setRouteResult(res);
    } finally {
      setRouteLoading(false);
    }
  };

  const runAcceptance = async () => {
    setAcceptanceLoading(true);
    try {
      const res = await fetch("/api/acceptance", { method: "POST" }).then((r) => r.json());
      if (res.ok) {
        setAcceptanceReport(res.report);
      }
    } finally {
      setAcceptanceLoading(false);
    }
  };

  const filteredAcceptanceCases = useMemo(() => {
    if (!acceptanceReport) return [];
    if (acceptanceFilter === "all") return acceptanceReport.cases;
    if (acceptanceFilter === "failed") return acceptanceReport.cases.filter((c) => !c.passed);
    return acceptanceReport.cases.filter((c) => c.kind === acceptanceFilter);
  }, [acceptanceReport, acceptanceFilter]);

  return (
    <div className="min-h-screen bg-[#0b0e14] text-slate-100 flex flex-col">
      {/* Top Workbench Header */}
      <header className="border-b border-slate-800/90 bg-[#0f141d] px-4 py-3">
        <div className="max-w-[1600px] mx-auto flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Cpu className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-tight text-slate-100">
                Omni Language Workbench
              </h1>
              <p className="text-xs text-slate-400">
                16-bit Token ISA ({contractVersions.isa}..0x7F7F) &middot; {contractVersions.harness} &middot;{" "}
                {contractVersions.error}
              </p>
            </div>
          </div>

          <nav className="flex flex-wrap items-center gap-1.5 bg-[#0b0e14] p-1 rounded-lg border border-slate-800">
            <button
              type="button"
              onClick={() => setTab("playground")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                tab === "playground"
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                  : "text-slate-400 hover:text-slate-200 border border-transparent"
              }`}
            >
              <Code2 className="h-3.5 w-3.5" />
              Compiler &amp; VM / WASM
            </button>
            <button
              type="button"
              onClick={() => setTab("repair")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                tab === "repair"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                  : "text-slate-400 hover:text-slate-200 border border-transparent"
              }`}
            >
              <Wrench className="h-3.5 w-3.5" />
              Harness &amp; Agent Repair
            </button>
            <button
              type="button"
              onClick={() => setTab("routes")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                tab === "routes"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                  : "text-slate-400 hover:text-slate-200 border border-transparent"
              }`}
            >
              <Globe className="h-3.5 w-3.5" />
              HTTP Route Host
            </button>
            <button
              type="button"
              onClick={() => {
                setTab("acceptance");
                if (!acceptanceReport && !acceptanceLoading) {
                  void runAcceptance();
                }
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                tab === "acceptance"
                  ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                  : "text-slate-400 hover:text-slate-200 border border-transparent"
              }`}
            >
              <CheckCircle2 className="h-3.5 w-3.5" />
              Acceptance &amp; Spec
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4">
        {/* TAB 1: COMPILER & VM / WASM PLAYGROUND */}
        {tab === "playground" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-full">
            {/* Left Example Sidebar */}
            <aside className="lg:col-span-3 bg-[#111622] border border-slate-800 rounded-xl flex flex-col overflow-hidden max-h-[82vh]">
              <div className="p-3 border-b border-slate-800 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                    <Layers className="h-3.5 w-3.5 text-cyan-400" />
                    Repository Programs
                  </span>
                  <span className="text-[11px] text-slate-400">{filteredExamples.length}</span>
                </div>
                <div className="relative">
                  <Search className="h-3.5 w-3.5 text-slate-500 absolute left-2.5 top-2.5" />
                  <input
                    type="text"
                    value={exampleFilter}
                    onChange={(e) => setExampleFilter(e.target.value)}
                    placeholder="Filter programs or codes..."
                    className="w-full bg-[#0b0e14] border border-slate-700/80 rounded-lg pl-8 pr-2.5 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <select
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                  className="w-full bg-[#0b0e14] border border-slate-700/80 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                >
                  <option value="all">All Categories ({examples.length})</option>
                  {(Object.keys(CATEGORY_LABELS) as Array<OmniExample["category"]>).map((cat) => (
                    <option key={cat} value={cat}>
                      {CATEGORY_LABELS[cat]}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60">
                {filteredExamples.map((ex) => {
                  const active = ex.id === selectedExampleId;
                  return (
                    <button
                      key={ex.id}
                      type="button"
                      onClick={() => selectExample(ex)}
                      className={`w-full text-left px-3 py-2.5 transition-colors cursor-pointer ${
                        active ? "bg-cyan-500/15 border-l-2 border-l-cyan-400" : "hover:bg-slate-800/40"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-mono font-medium text-slate-200 truncate">
                          {ex.name}
                        </span>
                        {ex.category === "negative_golden" && ex.expectedErrorCode ? (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30 shrink-0">
                            {ex.expectedErrorCode}
                          </span>
                        ) : ex.expectedBackend ? (
                          <span
                            className={`text-[10px] font-mono px-1.5 py-0.5 rounded border shrink-0 ${
                              ex.expectedBackend === "aot"
                                ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                                : "bg-cyan-500/15 text-cyan-300 border-cyan-500/30"
                            }`}
                          >
                            {ex.expectedBackend.toUpperCase()}
                          </span>
                        ) : (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 shrink-0">
                            {ex.category === "pure_program" ? "PURE" : ex.category === "stdlib" ? "STD" : "CORE"}
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-500 truncate mt-0.5">{ex.path}</div>
                    </button>
                  );
                })}
              </div>
            </aside>

            {/* Center Editor Pane */}
            <section className="lg:col-span-5 bg-[#111622] border border-slate-800 rounded-xl flex flex-col overflow-hidden">
              <div className="p-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-slate-200">Surface Source</span>
                  <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer ml-2">
                    <input
                      type="checkbox"
                      checked={strictMode}
                      onChange={(e) => setStrictMode(e.target.checked)}
                      className="rounded border-slate-600 bg-slate-900 text-cyan-500 focus:ring-cyan-500"
                    />
                    <span className="font-mono">#[strict]</span>
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1.5 bg-[#0b0e14] border border-slate-700/80 rounded-lg px-2 py-1">
                    <span className="text-[11px] text-slate-400 font-mono">args:</span>
                    <input
                      type="text"
                      value={argsInput}
                      onChange={(e) => setArgsInput(e.target.value)}
                      placeholder="15, 40"
                      className="w-32 bg-transparent text-xs font-mono text-slate-100 focus:outline-none"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => void executePlayground()}
                    disabled={running}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-xs font-semibold text-white transition-colors cursor-pointer"
                  >
                    <Play className="h-3.5 w-3.5" />
                    {running ? "Running..." : "Compile & Run"}
                  </button>
                </div>
              </div>
              <textarea
                value={source}
                onChange={(e) => setSource(e.target.value)}
                spellCheck={false}
                className="flex-1 min-h-[460px] w-full bg-[#0b0e14] p-3.5 text-xs font-mono leading-relaxed text-slate-100 focus:outline-none resize-none"
                placeholder="Write Omni surface code or select a program from the left..."
              />
            </section>

            {/* Right Output & Inspector Pane */}
            <section className="lg:col-span-4 bg-[#111622] border border-slate-800 rounded-xl flex flex-col overflow-hidden max-h-[82vh]">
              <div className="p-3 border-b border-slate-800 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                  <Terminal className="h-3.5 w-3.5 text-cyan-400" />
                  Verification &amp; Execution Output
                </span>
                {runResult?.ok && (
                  <span
                    className={`text-[11px] font-mono px-2 py-0.5 rounded border ${
                      runResult.backend === "aot"
                        ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
                        : "bg-cyan-500/15 text-cyan-300 border-cyan-500/40"
                    }`}
                  >
                    {runResult.backend === "aot" ? "WASM AOT + Native Validate" : "OMNI DETERMINISTIC VM"}
                  </span>
                )}
              </div>

              <div className="p-3 space-y-3 border-b border-slate-800 bg-[#0d111a]">
                {!runResult ? (
                  <p className="text-xs text-slate-400">
                    Click <strong className="text-slate-200">Compile &amp; Run</strong> to compile surface syntax to
                    the 16-bit token ISA, run the Pure-Omni verifier kernel &amp; static checks, and execute via WASM
                    AOT or deterministic VM.
                  </p>
                ) : runResult.ok ? (
                  <div className="space-y-2">
                    <div className="grid grid-cols-3 gap-2">
                      <div className="bg-[#111622] border border-slate-800 rounded-lg p-2.5">
                        <div className="text-[11px] text-slate-400">Result</div>
                        <div className="text-base font-mono font-bold text-emerald-300 mt-0.5 truncate">
                          {runResult.result}
                        </div>
                      </div>
                      <div className="bg-[#111622] border border-slate-800 rounded-lg p-2.5">
                        <div className="text-[11px] text-slate-400">ISA Tokens</div>
                        <div className="text-base font-mono font-bold text-cyan-300 mt-0.5">
                          {runResult.tokens} <span className="text-xs font-normal text-slate-400">u16</span>
                        </div>
                      </div>
                      <div className="bg-[#111622] border border-slate-800 rounded-lg p-2.5">
                        <div className="text-[11px] text-slate-400">
                          {runResult.backend === "aot" ? "WASM Binary" : "VM Engine"}
                        </div>
                        <div className="text-base font-mono font-bold text-slate-200 mt-0.5">
                          {runResult.backend === "aot" ? `${runResult.wasmSize ?? 0} B` : "Bytecode"}
                        </div>
                      </div>
                    </div>
                    {runResult.elfSize && (
                      <div className="flex items-center justify-between text-[11px] font-mono bg-[#111622] border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-300">
                        <span>x86-64 Linux ELF64 AOT: {runResult.elfSize} B</span>
                        {runResult.elfHash && <span className="text-cyan-300">{runResult.elfHash}</span>}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="bg-rose-950/30 border border-rose-500/40 rounded-lg p-3 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-mono font-bold text-rose-300 flex items-center gap-1.5">
                        <ShieldAlert className="h-4 w-4 text-rose-400" />
                        error[{runResult.error?.code}] &middot; line {runResult.error?.line ?? 1}:
                        {runResult.error?.col ?? 1}
                      </span>
                      <button
                        type="button"
                        onClick={sendPlaygroundErrorToRepair}
                        className="flex items-center gap-1 px-2 py-1 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 text-[11px] font-medium cursor-pointer"
                      >
                        <Wrench className="h-3 w-3" />
                        Auto-Repair
                      </button>
                    </div>
                    <pre className="text-[11px] font-mono text-rose-100/90 whitespace-pre-wrap overflow-x-auto bg-[#0b0e14]/80 p-2 rounded border border-rose-500/20">
                      {runResult.humanDiagnostic}
                    </pre>
                  </div>
                )}
              </div>

              {/* Inspector Sub-tabs */}
              <div className="flex items-center gap-1 px-3 pt-2 border-b border-slate-800 bg-[#111622]">
                {(
                  [
                    ["inspect", "Inspect Lens"],
                    ["asm", ".omni_asm"],
                    ["hex", "u16 Stream"],
                    ["harness", "Harness JSON"],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setInspectTab(key)}
                    className={`px-2.5 py-1.5 text-xs font-medium border-b-2 transition-colors cursor-pointer ${
                      inspectTab === key
                        ? "border-cyan-400 text-cyan-300"
                        : "border-transparent text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="flex-1 overflow-auto p-3 bg-[#0b0e14]">
                {inspectTab === "inspect" && (
                  <pre className="text-[11px] font-mono text-slate-300 whitespace-pre-wrap">
                    {runResult?.inspectText ?? "Run a valid program to view its decoded ISA & verifier lens."}
                  </pre>
                )}
                {inspectTab === "asm" && (
                  <pre className="text-[11px] font-mono text-cyan-200/90 whitespace-pre-wrap">
                    {runResult?.asmText ?? "Run a valid program to view lowered .omni_asm SSA instructions."}
                  </pre>
                )}
                {inspectTab === "hex" && (
                  <pre className="text-[11px] font-mono text-emerald-300/90 whitespace-pre-wrap">
                    {runResult?.hexStream ?? "Run a valid program to inspect raw 16-bit token words."}
                  </pre>
                )}
                {inspectTab === "harness" && (
                  <pre className="text-[11px] font-mono text-amber-200/90 whitespace-pre-wrap">
                    {harnessResult
                      ? JSON.stringify(harnessResult, null, 2)
                      : "Run a program to inspect the version-locked omni-harness-v1.0.0 contract output."}
                  </pre>
                )}
              </div>
            </section>
          </div>
        )}

        {/* TAB 2: HARNESS & AUTONOMOUS AGENT REPAIR */}
        {tab === "repair" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <section className="lg:col-span-5 bg-[#111622] border border-slate-800 rounded-xl flex flex-col overflow-hidden">
              <div className="p-3 border-b border-slate-800 space-y-2.5">
                <div className="flex items-center justify-between">
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-amber-300 flex items-center gap-1.5">
                    <Wrench className="h-3.5 w-3.5" />
                    Broken Program Input (`omni-agent-driver`)
                  </h2>
                  <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={repairStrict}
                      onChange={(e) => setRepairStrict(e.target.checked)}
                      className="rounded border-slate-600 bg-slate-900 text-amber-500"
                    />
                    <span className="font-mono">#[strict]</span>
                  </label>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {REPAIR_PRESETS.map((preset) => (
                    <button
                      key={preset.label}
                      type="button"
                      onClick={() => {
                        setRepairSrc(preset.src);
                        setRepairArgs(preset.args);
                        setRepairStrict(preset.strict);
                        setRepairReport(null);
                      }}
                      className="text-[11px] px-2 py-1 rounded bg-[#0b0e14] hover:bg-slate-800 border border-slate-700 text-slate-300 cursor-pointer"
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                  <div className="flex items-center gap-2">
                    <div className="flex items-center gap-1.5 bg-[#0b0e14] border border-slate-700 rounded-lg px-2 py-1">
                      <span className="text-[11px] text-slate-400 font-mono">args:</span>
                      <input
                        type="text"
                        value={repairArgs}
                        onChange={(e) => setRepairArgs(e.target.value)}
                        className="w-24 bg-transparent text-xs font-mono text-slate-100 focus:outline-none"
                      />
                    </div>
                    <div className="flex items-center gap-1.5 bg-[#0b0e14] border border-slate-700 rounded-lg px-2 py-1">
                      <span className="text-[11px] text-slate-400 font-mono">max:</span>
                      <input
                        type="number"
                        min={1}
                        max={12}
                        value={repairMaxAttempts}
                        onChange={(e) => setRepairMaxAttempts(Number(e.target.value) || 6)}
                        className="w-12 bg-transparent text-xs font-mono text-slate-100 focus:outline-none"
                      />
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => void runAgentRepair()}
                    disabled={repairLoading}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-xs font-semibold text-white cursor-pointer"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${repairLoading ? "animate-spin" : ""}`} />
                    {repairLoading ? "Repairing..." : "Run Autonomous Repair"}
                  </button>
                </div>
              </div>

              <textarea
                value={repairSrc}
                onChange={(e) => setRepairSrc(e.target.value)}
                spellCheck={false}
                className="min-h-[380px] w-full bg-[#0b0e14] p-3.5 text-xs font-mono leading-relaxed text-slate-100 focus:outline-none resize-none"
              />
            </section>

            <section className="lg:col-span-7 bg-[#111622] border border-slate-800 rounded-xl flex flex-col overflow-hidden">
              <div className="p-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-xs font-semibold text-slate-200">
                    Autonomous Repair Trajectory ({repairReport?.strategy_version ?? "omni-repair-strategies-v1.1.0"})
                  </h2>
                </div>
                {repairReport && (
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs font-mono px-2.5 py-0.5 rounded border ${
                        repairReport.ok
                          ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
                          : "bg-rose-500/15 text-rose-300 border-rose-500/40"
                      }`}
                    >
                      {repairReport.ok
                        ? `CONVERGED (${repairReport.attempts} attempts -> ${repairReport.harness.result})`
                        : `FAILED AFTER ${repairReport.attempts} ATTEMPTS`}
                    </span>
                    {repairReport.ok && (
                      <button
                        type="button"
                        onClick={() => {
                          setSource(repairReport.final_source);
                          setArgsInput(repairArgs);
                          setStrictMode(repairStrict);
                          setTab("playground");
                          void executePlayground(repairReport.final_source, repairArgs, repairStrict);
                        }}
                        className="flex items-center gap-1 px-2.5 py-1 rounded bg-cyan-600 hover:bg-cyan-500 text-xs font-medium text-white cursor-pointer"
                      >
                        Load Fixed Code
                        <ArrowRight className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                )}
              </div>

              <div className="p-4 space-y-3 overflow-y-auto max-h-[72vh]">
                {!repairReport ? (
                  <p className="text-xs text-slate-400">
                    Select a broken preset or paste invalid Omni code on the left, then click{" "}
                    <strong className="text-slate-200">Run Autonomous Repair</strong>. The driver executes the
                    version-locked harness loop, matches structured `error[CODE]` diagnostics to the 16 canonical
                    repair strategies, patches the source, and re-verifies via AOT/VM.
                  </p>
                ) : (
                  <>
                    {repairReport.history.map((step) => (
                      <div
                        key={step.attempt}
                        className={`rounded-lg border p-3 space-y-2 ${
                          step.ok
                            ? "bg-emerald-950/20 border-emerald-500/35"
                            : "bg-[#0b0e14] border-slate-800"
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono font-semibold text-slate-200">
                              Attempt #{step.attempt}
                            </span>
                            {step.ok ? (
                              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                                VERIFIED OK
                              </span>
                            ) : (
                              <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30">
                                error[{step.code}] @ {step.line}:{step.col}
                              </span>
                            )}
                          </div>
                          {step.strategy_id && (
                            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">
                              strategy: {step.strategy_id}
                            </span>
                          )}
                        </div>
                        {step.message && (
                          <div className="text-[11px] font-mono text-slate-300">{step.message}</div>
                        )}
                        {step.retry_hint && (
                          <div className="text-[11px] text-amber-200/80">Hint: {step.retry_hint}</div>
                        )}
                        <pre className="text-[11px] font-mono text-slate-200 bg-[#0d111a] p-2.5 rounded border border-slate-800/80 overflow-x-auto">
                          {step.source}
                        </pre>
                      </div>
                    ))}
                  </>
                )}
              </div>
            </section>
          </div>
        )}

        {/* TAB 3: HTTP ROUTE HOST (`omni serve`) */}
        {tab === "routes" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            <section className="lg:col-span-6 bg-[#111622] border border-slate-800 rounded-xl flex flex-col overflow-hidden">
              <div className="p-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-emerald-300 flex items-center gap-1.5">
                  <Globe className="h-3.5 w-3.5" />
                  Route Service Source (`omni serve`)
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {routeExamples.map((ex) => (
                    <button
                      key={ex.id}
                      type="button"
                      onClick={() => {
                        setRouteSrc(ex.source);
                        setRouteResult(null);
                        if (ex.name.includes("kv_ledger")) {
                          setRouteMethod("POST");
                          setRoutePath("/ledger/42");
                        } else if (ex.name.includes("route_kv")) {
                          setRouteMethod("GET");
                          setRoutePath("/users/42");
                        } else {
                          setRouteMethod("GET");
                          setRoutePath("/users/42");
                        }
                      }}
                      className="text-[11px] font-mono px-2 py-1 rounded bg-[#0b0e14] hover:bg-slate-800 border border-slate-700 text-slate-300 cursor-pointer"
                    >
                      {ex.name}
                    </button>
                  ))}
                </div>
              </div>
              <textarea
                value={routeSrc}
                onChange={(e) => setRouteSrc(e.target.value)}
                spellCheck={false}
                className="min-h-[460px] w-full bg-[#0b0e14] p-3.5 text-xs font-mono leading-relaxed text-slate-100 focus:outline-none resize-none"
              />
            </section>

            <section className="lg:col-span-6 bg-[#111622] border border-slate-800 rounded-xl flex flex-col overflow-hidden">
              <div className="p-3 border-b border-slate-800 space-y-2.5">
                <div className="flex flex-wrap items-center gap-2">
                  <select
                    value={routeMethod}
                    onChange={(e) => setRouteMethod(e.target.value)}
                    className="bg-[#0b0e14] border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs font-mono font-semibold text-emerald-300 focus:outline-none"
                  >
                    <option value="GET">GET</option>
                    <option value="POST">POST</option>
                    <option value="PUT">PUT</option>
                    <option value="DELETE">DELETE</option>
                  </select>
                  <input
                    type="text"
                    value={routePath}
                    onChange={(e) => setRoutePath(e.target.value)}
                    placeholder="/ledger/42"
                    className="flex-1 bg-[#0b0e14] border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-100 focus:outline-none focus:border-emerald-500"
                  />
                  <button
                    type="button"
                    onClick={() => void dispatchRoute(false)}
                    disabled={routeLoading}
                    className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-xs font-semibold text-white cursor-pointer"
                  >
                    {routeLoading ? "Sending..." : "Dispatch"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void dispatchRoute(true)}
                    disabled={routeLoading}
                    className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 cursor-pointer"
                  >
                    Reset KV
                  </button>
                </div>

                {routeResult?.routes && routeResult.routes.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[11px] text-slate-400">Extracted routes:</span>
                    {routeResult.routes.map((rt) => (
                      <button
                        key={`${rt.method}-${rt.pathPattern}`}
                        type="button"
                        onClick={() => {
                          const samplePath = rt.pathPattern.replace(/:([a-zA-Z0-9_]+)/g, "42");
                          setRouteMethod(rt.method);
                          setRoutePath(samplePath);
                          void dispatchRoute(false, rt.method, samplePath);
                        }}
                        className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 cursor-pointer"
                      >
                        {rt.method} {rt.pathPattern} (fn #{rt.fnIdx})
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="p-4 space-y-4 overflow-y-auto flex-1">
                {!routeResult ? (
                  <p className="text-xs text-slate-400">
                    Dispatch an HTTP request against the capability-gated Omni route compiler (`kv_put`, `db_get`,
                    `db_query`, `auth_verify`, `req.param`). State persists across requests in the host KV store until
                    you click <strong className="text-slate-200">Reset KV</strong>.
                  </p>
                ) : routeResult.ok && routeResult.response ? (
                  <>
                    <div className="bg-[#0b0e14] border border-slate-800 rounded-lg p-3 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-300">HTTP Response</span>
                        <span
                          className={`text-xs font-mono px-2 py-0.5 rounded border ${
                            routeResult.response.status < 400
                              ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                              : "bg-rose-500/15 text-rose-300 border-rose-500/30"
                          }`}
                        >
                          HTTP {routeResult.response.status} &middot; {routeResult.response.contentType}
                        </span>
                      </div>
                      <pre className="text-xs font-mono text-emerald-200 bg-[#111622] p-2.5 rounded border border-slate-800 overflow-x-auto">
                        {routeResult.response.body}
                      </pre>
                    </div>

                    <div className="bg-[#0b0e14] border border-slate-800 rounded-lg p-3 space-y-2">
                      <span className="text-xs font-semibold text-slate-300">Host KV Store Snapshot</span>
                      {Object.keys(routeResult.response.kvSnapshot ?? {}).length === 0 ? (
                        <div className="text-xs text-slate-500 font-mono">(empty)</div>
                      ) : (
                        <table className="w-full text-xs font-mono">
                          <thead>
                            <tr className="text-left text-slate-400 border-b border-slate-800">
                              <th className="py-1">Key (i64)</th>
                              <th className="py-1">Value (i64)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(routeResult.response.kvSnapshot ?? {}).map(([k, v]) => (
                              <tr key={k} className="border-b border-slate-800/50 text-slate-200">
                                <td className="py-1 text-cyan-300">{k}</td>
                                <td className="py-1 text-emerald-300">{v}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="bg-rose-950/30 border border-rose-500/40 rounded-lg p-3">
                    <div className="text-xs font-mono font-bold text-rose-300 mb-2">
                      error[{routeResult.error?.code}]
                    </div>
                    <pre className="text-[11px] font-mono text-rose-100 whitespace-pre-wrap">
                      {routeResult.humanDiagnostic}
                    </pre>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}

        {/* TAB 4: ACCEPTANCE SUITE & SPEC REFERENCE */}
        {tab === "acceptance" && (
          <div className="space-y-4">
            <div className="bg-[#111622] border border-slate-800 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setSpecTab("suite")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer ${
                    specTab === "suite"
                      ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Acceptance Suite ({acceptanceReport ? `${acceptanceReport.passedCount}/${acceptanceReport.total}` : "Run"})
                </button>
                <button
                  type="button"
                  onClick={() => setSpecTab("bootstrap")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer ${
                    specTab === "bootstrap"
                      ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Self-Host Bootstrap (Gen-1 == Gen-2)
                </button>
                <button
                  type="button"
                  onClick={() => setSpecTab("errors")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer flex items-center gap-1.5 ${
                    specTab === "errors"
                      ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <BookOpen className="h-3.5 w-3.5" />
                  Error Codes (--explain, {explanations.length})
                </button>
                <button
                  type="button"
                  onClick={() => setSpecTab("strategies")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium cursor-pointer ${
                    specTab === "strategies"
                      ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Repair Strategies (v1.1.0, {strategies.length})
                </button>
              </div>

              <button
                type="button"
                onClick={() => void runAcceptance()}
                disabled={acceptanceLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white cursor-pointer"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${acceptanceLoading ? "animate-spin" : ""}`} />
                {acceptanceLoading ? "Running Suite..." : "Re-run Full Acceptance Suite"}
              </button>
            </div>

            {specTab === "suite" && (
              <div className="bg-[#111622] border border-slate-800 rounded-xl overflow-hidden">
                {acceptanceReport && (
                  <div className="p-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2 bg-[#0d111a]">
                    <div className="flex items-center gap-3">
                      <span
                        className={`text-xs font-mono font-bold px-2.5 py-1 rounded border ${
                          acceptanceReport.passed
                            ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
                            : "bg-rose-500/15 text-rose-300 border-rose-500/40"
                        }`}
                      >
                        {acceptanceReport.passedCount}/{acceptanceReport.total} PASSED
                      </span>
                      <span className="text-xs text-slate-400 font-mono">
                        {acceptanceReport.manifestVersion} &middot; {acceptanceReport.durationMs}ms
                      </span>
                    </div>
                    <select
                      value={acceptanceFilter}
                      onChange={(e) => setAcceptanceFilter(e.target.value)}
                      className="bg-[#0b0e14] border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200"
                    >
                      <option value="all">All Checks ({acceptanceReport.total})</option>
                      <option value="positive_golden">Positive Goldens (24)</option>
                      <option value="negative_golden">Negative Goldens (15)</option>
                      <option value="pure_program">Pure Programs (6)</option>
                      <option value="error_contract">Error Contract Probes (16)</option>
                      <option value="failed">Failures ({acceptanceReport.failedCount})</option>
                    </select>
                  </div>
                )}

                <div className="overflow-x-auto max-h-[68vh]">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-[#0d111a] text-slate-400 border-b border-slate-800 sticky top-0">
                      <tr>
                        <th className="py-2.5 px-3">Status</th>
                        <th className="py-2.5 px-3">Category</th>
                        <th className="py-2.5 px-3">Case ID</th>
                        <th className="py-2.5 px-3">Backend</th>
                        <th className="py-2.5 px-3">Tokens</th>
                        <th className="py-2.5 px-3">Verification Detail</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {filteredAcceptanceCases.map((c) => (
                        <tr key={`${c.kind}-${c.id}`} className="hover:bg-slate-800/30">
                          <td className="py-2 px-3 font-mono">
                            {c.passed ? (
                              <span className="inline-flex items-center gap-1 text-emerald-300">
                                <CheckCircle2 className="h-3.5 w-3.5" /> PASS
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-rose-300">
                                <XCircle className="h-3.5 w-3.5" /> FAIL
                              </span>
                            )}
                          </td>
                          <td className="py-2 px-3 font-mono text-slate-400">{c.kind}</td>
                          <td className="py-2 px-3 font-mono font-medium text-slate-200">{c.id}</td>
                          <td className="py-2 px-3 font-mono text-cyan-300">
                            {c.backend ? c.backend.toUpperCase() : "—"}
                          </td>
                          <td className="py-2 px-3 font-mono text-slate-300">{c.tokens ?? "—"}</td>
                          <td className="py-2 px-3 font-mono text-slate-300">{c.detail}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {specTab === "bootstrap" && bootstrapSummary && (
              <div className="bg-[#111622] border border-slate-800 rounded-xl p-4 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100">
                      Phase A–E Pure-Omni Self-Hosting Fixed-Point Verification
                    </h3>
                    <p className="text-xs text-slate-400">
                      Bit-identical Stage-0 -&gt; Gen-1 -&gt; Gen-2 `.omni_bc` &amp; `x86-64` Linux ELF64 self-compilation +
                      19,731-word Stage-1 compiler stream.
                    </p>
                  </div>
                  <span className="text-xs font-mono font-bold px-2.5 py-1 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/40">
                    {bootstrapSummary.fixedPointVerified ? "GEN-1 == GEN-2 VERIFIED" : "UNVERIFIED"}
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
                  <div className="bg-[#0b0e14] border border-slate-800 rounded-lg p-3 space-y-1">
                    <div className="text-slate-400">Gen-1 / Gen-2 `.omni_bc` Stream</div>
                    <div className="text-sm font-bold text-cyan-300">
                      {bootstrapSummary.gen1Words} == {bootstrapSummary.gen2Words} u16 words
                    </div>
                    <div className="text-[11px] text-slate-400">{bootstrapSummary.bcFnv1a64}</div>
                  </div>
                  <div className="bg-[#0b0e14] border border-slate-800 rounded-lg p-3 space-y-1">
                    <div className="text-slate-400">Gen-1 / Gen-2 `x86-64` Linux ELF64</div>
                    <div className="text-sm font-bold text-emerald-300">{bootstrapSummary.elfBytes} bytes</div>
                    <div className="text-[11px] text-slate-400">{bootstrapSummary.elfFnv1a64}</div>
                  </div>
                  <div className="bg-[#0b0e14] border border-slate-800 rounded-lg p-3 space-y-1">
                    <div className="text-slate-400">Stage-1 `.omni_bc` Self-Compiler</div>
                    <div className="text-sm font-bold text-amber-300">
                      {bootstrapSummary.stage1Words} u16 words ({bootstrapSummary.stage1ElfBytes} B ELF64)
                    </div>
                    <div className="text-[11px] text-slate-400">{bootstrapSummary.stage1ElfFnv1a64}</div>
                  </div>
                </div>
                <div className="text-xs font-mono text-slate-300 bg-[#0b0e14] border border-slate-800 rounded-lg p-3">
                  Pure-Omni Core Suite: {bootstrapSummary.totalPureModules} modules &middot; {bootstrapSummary.suiteWords} verified u16 words
                </div>
              </div>
            )}

            {specTab === "errors" && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {explanations.map((ex) => (
                  <div key={ex.code} className="bg-[#111622] border border-slate-800 rounded-xl p-3.5 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-bold text-rose-300">error[{ex.code}]</span>
                      <span className="text-[11px] font-mono text-slate-400">{ex.stage}</span>
                    </div>
                    <p className="text-xs text-slate-300">{ex.summary}</p>
                    <p className="text-[11px] text-amber-300/90">Hint: {ex.hint}</p>
                    <div className="grid grid-cols-1 gap-1.5 pt-1 text-[11px] font-mono">
                      <div className="bg-rose-950/25 border border-rose-500/20 rounded p-2 text-rose-200">
                        Bad: {ex.bad}
                      </div>
                      <div className="bg-emerald-950/25 border border-emerald-500/20 rounded p-2 text-emerald-200">
                        Repaired: {ex.repaired}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {specTab === "strategies" && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {strategies.map((st) => (
                  <div key={st.id} className="bg-[#111622] border border-slate-800 rounded-xl p-3.5 space-y-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-mono font-bold text-amber-300">{st.id}</span>
                      <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30">
                        {st.error_code}
                      </span>
                    </div>
                    <div className="text-[11px] font-mono text-slate-400">pattern: {st.context_pattern}</div>
                    <p className="text-xs text-slate-300">{st.description}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
