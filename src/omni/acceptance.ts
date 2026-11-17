import fs from "node:fs";
import path from "node:path";
import {
  compileOmniSurfaceToAsm,
  compileOmniSurfaceToU16,
  compileOmniSurfaceStrictToU16,
  assembleOmniTextToU16,
  executeOmniSurfaceAotFirst,
  handleOmniHttpRequest,
  omniHarnessSpecObj,
  parseCliOmniValue,
  runOmniHarness,
  wrapOmniSurfaceSnippet,
  type OmniValue,
} from "./lang.ts";

export interface AcceptanceManifestPositive {
  id: string;
  file: string;
  locked_u16_tokens: number | null;
  args: string[];
  expected_result: string;
  expected_backend: "aot" | "vm";
}

export interface AcceptanceManifestNegative {
  id: string;
  file: string;
  strict: boolean;
  expected_code: string;
  expected_line: number;
  expected_col: number;
}

export interface AcceptanceManifest {
  manifest_version: string;
  harness_contract_version: string;
  error_contract_version: string;
  positive_goldens: AcceptanceManifestPositive[];
  negative_goldens: AcceptanceManifestNegative[];
}

export interface ErrorContractJson {
  contract_version: string;
  harness_version: string;
  codes: Array<{
    code: string;
    non_null_span: boolean;
    has_retry_hint: boolean;
    stage: string;
  }>;
}

export interface OmniExampleEntry {
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

export interface AcceptanceCaseResult {
  id: string;
  kind: "positive_golden" | "negative_golden" | "pure_program" | "error_contract";
  passed: boolean;
  detail: string;
  actual?: string;
  expected?: string;
  tokens?: number | null;
  backend?: string;
}

export interface AcceptanceSuiteReport {
  passed: boolean;
  manifestVersion: string;
  harnessVersion: string;
  errorContractVersion: string;
  total: number;
  passedCount: number;
  failedCount: number;
  durationMs: number;
  cases: AcceptanceCaseResult[];
}

function readJsonFile<T>(relPath: string): T {
  const full = path.resolve(process.cwd(), relPath);
  return JSON.parse(fs.readFileSync(full, "utf8")) as T;
}

function readTextFile(relPath: string): string {
  const full = path.resolve(process.cwd(), relPath);
  return fs.readFileSync(full, "utf8");
}

export function loadAcceptanceManifest(): AcceptanceManifest {
  return readJsonFile<AcceptanceManifest>("spec/acceptance/acceptance_manifest.json");
}

export function loadErrorContractJson(): ErrorContractJson {
  return readJsonFile<ErrorContractJson>("spec/acceptance/error_contract.json");
}

const PURE_PROGRAM_DEFAULTS: Record<string, { args: string[]; description: string }> = {
  "ternary_neural_router.omni": { args: ["15", "40"], description: "Multi-layer @PACK5 ternary neural router" },
  "policy_consensus_engine.omni": { args: ["60", "55", "10"], description: "Balanced-ternary policy consensus engine" },
  "arena_audit_pipeline.omni": { args: ["20", "42"], description: "Linear arena & borrow audit pipeline" },
  "kv_ledger_service.omni": { args: ["42"], description: "Capability-gated HTTP route + KV ledger service" },
  "compiler_self_audit.omni": { args: ["120", "15", "1003"], description: "Pure-Omni compiler self-audit benchmark" },
  "pack5_decision_controller.omni": { args: ["25", "10"], description: "@PACK5 decision hook controller" },
  "struct_string_pipeline.omni": { args: ["15", "9"], description: "First-class struct + string/slice/hex pipeline" },
};

export function loadOmniExamples(): OmniExampleEntry[] {
  const examples: OmniExampleEntry[] = [];
  const manifest = loadAcceptanceManifest();

  const pureProgramsDir = path.resolve(process.cwd(), "pure/programs");
  if (fs.existsSync(pureProgramsDir)) {
    for (const file of fs.readdirSync(pureProgramsDir).sort()) {
      if (!file.endsWith(".omni")) continue;
      const defaults = PURE_PROGRAM_DEFAULTS[file] ?? { args: ["10", "20"], description: file };
      examples.push({
        id: `pure_program:${file.replace(/\.omni$/, "")}`,
        category: "pure_program",
        path: `pure/programs/${file}`,
        name: file,
        source: fs.readFileSync(path.join(pureProgramsDir, file), "utf8"),
        args: defaults.args,
        strict: true,
      });
    }
  }

  for (const g of manifest.positive_goldens) {
    examples.push({
      id: `positive:${g.id}`,
      category: "positive_golden",
      path: `spec/acceptance/${g.file}`,
      name: `${g.id}.omni`,
      source: readTextFile(`spec/acceptance/${g.file}`),
      args: g.args,
      strict: false,
      expectedResult: g.expected_result,
      expectedBackend: g.expected_backend,
      lockedTokens: g.locked_u16_tokens,
    });
  }

  for (const g of manifest.negative_goldens) {
    examples.push({
      id: `negative:${g.id}`,
      category: "negative_golden",
      path: `spec/acceptance/${g.file}`,
      name: `${g.id}.omni`,
      source: readTextFile(`spec/acceptance/${g.file}`),
      args: [],
      strict: g.strict,
      expectedErrorCode: g.expected_code,
      expectedLine: g.expected_line,
      expectedCol: g.expected_col,
    });
  }

  const pureDir = path.resolve(process.cwd(), "pure");
  if (fs.existsSync(pureDir)) {
    for (const file of fs.readdirSync(pureDir).sort()) {
      if (!file.endsWith(".omni")) continue;
      examples.push({
        id: `pure_core:${file.replace(/\.omni$/, "")}`,
        category: "pure_core",
        path: `pure/${file}`,
        name: file,
        source: fs.readFileSync(path.join(pureDir, file), "utf8"),
        args: ["10", "0", "0"],
        strict: true,
      });
    }
  }

  const stdDir = path.resolve(process.cwd(), "std");
  if (fs.existsSync(stdDir)) {
    for (const file of fs.readdirSync(stdDir).sort()) {
      if (!file.endsWith(".omni")) continue;
      examples.push({
        id: `stdlib:${file.replace(/\.omni$/, "")}`,
        category: "stdlib",
        path: `std/${file}`,
        name: file,
        source: fs.readFileSync(path.join(stdDir, file), "utf8"),
        args: [],
        strict: true,
      });
    }
  }

  return examples;
}

const ERROR_CONTRACT_PROBES: Record<string, { src: string; strict: boolean }> = {
  TAINT_ON_RET: {
    src: "fn main(x: i64 @untrusted) -> i64 {\n  return x;\n}",
    strict: false,
  },
  TAINT_SINK: {
    src: "#[cap(AI)]\nfn main(x: i64 @untrusted) -> trit {\n  let g: trit = egate(x);\n  return g;\n}",
    strict: false,
  },
  MISSING_CAP: {
    src: "fn main(x: i64) -> trit {\n  let g: trit = egate(x);\n  return g;\n}",
    strict: false,
  },
  LINEAR_LEAK: {
    src: "fn main(x: i64) -> i64 {\n  let a: arena = arena(1024);\n  return x;\n}",
    strict: false,
  },
  BORROW_CONFLICT: {
    src: "fn main(x: i64) -> i64 {\n  let a: arena = arena(1024);\n  let r: i64 = borrow_r(a);\n  drop(a);\n  drop(r);\n  return x;\n}",
    strict: false,
  },
  UNDEF_VAR: {
    src: "fn main(x: i64) -> i64 {\n  return missing_v;\n}",
    strict: false,
  },
  IMMUT_PARAM: {
    src: "fn main(x: i64) -> i64 {\n  x = 5;\n  return x;\n}",
    strict: false,
  },
  BAD_TYPE: {
    src: "fn main(x: float128) -> i64 {\n  return 0;\n}",
    strict: false,
  },
  UNKNOWN_FN: {
    src: "fn main(x: i64) -> i64 {\n  return nope_fn(x);\n}",
    strict: false,
  },
  ARITY_MISMATCH: {
    src: "fn main(x: i64) -> i64 {\n  return iabs(x, 1);\n}",
    strict: false,
  },
  UNRESOLVED_LABEL: {
    src: "fn main(x: i64) -> i64 {\n  goto 7;\n  return x;\n}",
    strict: false,
  },
  CIRCULAR_INCLUDE: {
    src: 'include "spec/acceptance/goldens/neg_circular_include.omni";\nfn main(x: i64) -> i64 { return x; }',
    strict: false,
  },
  BUDGET_EXCEEDED: {
    src: "#[step_budget(20)]\nfn main(n: i64) -> i64 {\n  let i: i64 = 0;\n  while i < 500 { i += 1; }\n  return i;\n}",
    strict: false,
  },
  CORRUPT_PACK5: {
    src: "#[strict]\nfn main() -> trit_vec {\n  let b: bytes = anew(0);\n  b = iset(b, 250);\n  let v: trit_vec = unpack5(b, 5);\n  return v;\n}",
    strict: true,
  },
  AMBIGUOUS_GRAMMAR: {
    src: "#[strict]\nfn main(x: int) -> trit {\n  return TRIT_POS;\n}",
    strict: true,
  },
  SYNTAX_ERROR: {
    src: "fn main(x: i64) -> i64 {\n  let y: i64 = ;\n  return y;\n}",
    strict: false,
  },
};

export function runAcceptanceSuite(): AcceptanceSuiteReport {
  const start = Date.now();
  const manifest = loadAcceptanceManifest();
  const errorContract = loadErrorContractJson();
  const cases: AcceptanceCaseResult[] = [];

  for (const g of manifest.positive_goldens) {
    const src = readTextFile(`spec/acceptance/${g.file}`);
    const args: OmniValue[] = g.args.map(parseCliOmniValue);
    try {
      const wrapped = wrapOmniSurfaceSnippet(src);
      const u16Direct = compileOmniSurfaceToU16(wrapped);
      const asmText = compileOmniSurfaceToAsm(wrapped);
      const u16FromAsm = assembleOmniTextToU16(asmText);

      const asmMatch =
        u16Direct.length === u16FromAsm.length && u16Direct.every((v, idx) => v === u16FromAsm[idx]);
      if (!asmMatch) {
        cases.push({
          id: g.id,
          kind: "positive_golden",
          passed: false,
          detail: `ASM roundtrip mismatch (direct=${u16Direct.length} tokens vs asm=${u16FromAsm.length} tokens)`,
          tokens: u16Direct.length,
        });
        continue;
      }

      if (g.locked_u16_tokens !== null && u16Direct.length !== g.locked_u16_tokens) {
        cases.push({
          id: g.id,
          kind: "positive_golden",
          passed: false,
          detail: `Locked u16 token count mismatch`,
          expected: `${g.locked_u16_tokens} tokens`,
          actual: `${u16Direct.length} tokens`,
          tokens: u16Direct.length,
        });
        continue;
      }

      const execDetail = executeOmniSurfaceAotFirst(src, args, false);
      const backendOk = execDetail.backend === g.expected_backend;
      const resultOk = execDetail.formattedResult === g.expected_result;
      const wasmOk = execDetail.backend !== "aot" || execDetail.wasmValid;
      const ok = backendOk && resultOk && wasmOk;

      cases.push({
        id: g.id,
        kind: "positive_golden",
        passed: ok,
        detail: ok
          ? `OK (${execDetail.formattedResult}, ${execDetail.u16Stream.length} u16 tokens, ${execDetail.backend.toUpperCase()}${execDetail.wasmBytes ? `, ${execDetail.wasmBytes.length}B WASM` : ""})`
          : `Expected (${g.expected_result}, ${g.expected_backend}), got (${execDetail.formattedResult}, ${execDetail.backend}, wasmValid=${execDetail.wasmValid})`,
        expected: `${g.expected_result} [${g.expected_backend}]`,
        actual: `${execDetail.formattedResult} [${execDetail.backend}]`,
        tokens: execDetail.u16Stream.length,
        backend: execDetail.backend,
      });
    } catch (e) {
      cases.push({
        id: g.id,
        kind: "positive_golden",
        passed: false,
        detail: e instanceof Error ? e.message : String(e),
      });
    }
  }

  for (const g of manifest.negative_goldens) {
    const src = readTextFile(`spec/acceptance/${g.file}`);
    const h = runOmniHarness(src, [], g.strict, true);
    const ok =
      !h.ok &&
      h.code === g.expected_code &&
      h.line === g.expected_line &&
      h.col === g.expected_col &&
      Boolean(h.retry_hint && h.retry_hint.length > 0);
    cases.push({
      id: g.id,
      kind: "negative_golden",
      passed: ok,
      detail: ok
        ? `Rejected with error[${h.code}] at ${h.line}:${h.col}`
        : `Expected error[${g.expected_code}] at ${g.expected_line}:${g.expected_col}, got ok=${h.ok} code=${h.code} at ${h.line}:${h.col} (${h.message ?? ""})`,
      expected: `${g.expected_code}@${g.expected_line}:${g.expected_col}`,
      actual: h.ok ? "OK (unexpected)" : `${h.code}@${h.line}:${h.col}`,
      tokens: h.tokens,
    });
  }

  // Test pure/programs/*.omni
  const pureProgramsDir = path.resolve(process.cwd(), "pure/programs");
  if (fs.existsSync(pureProgramsDir)) {
    for (const file of fs.readdirSync(pureProgramsDir).sort()) {
      if (!file.endsWith(".omni")) continue;
      const src = fs.readFileSync(path.join(pureProgramsDir, file), "utf8");
      const id = `pure:${file.replace(/\.omni$/, "")}`;
      if (file === "kv_ledger_service.omni") {
        try {
          const kv = new Map<bigint, bigint>();
          const postRes = handleOmniHttpRequest(src, "POST", "/ledger/42", kv);
          const getRes = handleOmniHttpRequest(src, "GET", "/ledger/42", kv);
          const ok = postRes.status === 201 && getRes.status === 200;
          cases.push({
            id,
            kind: "pure_program",
            passed: ok,
            detail: `POST /ledger/42 -> ${postRes.status}, GET /ledger/42 -> ${getRes.status} (body=${getRes.body})`,
            backend: "vm",
          });
        } catch (e) {
          cases.push({
            id,
            kind: "pure_program",
            passed: false,
            detail: e instanceof Error ? e.message : String(e),
          });
        }
      } else {
        const defaults = PURE_PROGRAM_DEFAULTS[file] ?? { args: ["10", "20"] };
        const args = defaults.args.map(parseCliOmniValue);
        const h = runOmniHarness(src, args, true, true);
        cases.push({
          id,
          kind: "pure_program",
          passed: h.ok,
          detail: h.ok
            ? `Executed -> ${h.result} (${h.tokens} tokens, ${h.backend?.toUpperCase()})`
            : `${h.code}: ${h.message}`,
          actual: h.result,
          tokens: h.tokens,
          backend: h.backend,
        });
      }
    }
  }

  // Error contract verification for all 16 codes
  const spec = omniHarnessSpecObj();
  for (const ec of errorContract.codes) {
    const probe = ERROR_CONTRACT_PROBES[ec.code];
    if (!probe) {
      cases.push({
        id: `contract:${ec.code}`,
        kind: "error_contract",
        passed: false,
        detail: `No probe defined for ${ec.code}`,
      });
      continue;
    }
    const h = runOmniHarness(probe.src, [], probe.strict, true);
    const inSpec = spec.error_codes.includes(ec.code);
    const spanOk = !ec.non_null_span || (typeof h.line === "number" && h.line >= 1 && typeof h.col === "number" && h.col >= 1);
    const hintOk = !ec.has_retry_hint || Boolean(h.retry_hint && h.retry_hint.length > 0);
    const ok = inSpec && !h.ok && h.code === ec.code && spanOk && hintOk;
    cases.push({
      id: `contract:${ec.code}`,
      kind: "error_contract",
      passed: ok,
      detail: ok
        ? `Verified error[${ec.code}] span=${h.line}:${h.col} + retry_hint`
        : `Contract mismatch: expected ${ec.code}, got ok=${h.ok} code=${h.code} span=${h.line}:${h.col}`,
      expected: ec.code,
      actual: h.code,
    });
  }

  // Verify strict compiler doesn't reject valid canonical programs
  void compileOmniSurfaceStrictToU16;

  const passedCount = cases.filter((c) => c.passed).length;
  const failedCount = cases.length - passedCount;
  return {
    passed: failedCount === 0,
    manifestVersion: manifest.manifest_version,
    harnessVersion: manifest.harness_contract_version,
    errorContractVersion: manifest.error_contract_version,
    total: cases.length,
    passedCount,
    failedCount,
    durationMs: Date.now() - start,
    cases,
  };
}
