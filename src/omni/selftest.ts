import assert from "node:assert/strict";
import fs from "node:fs";
import { runAcceptanceSuite } from "./acceptance.ts";
import {
  buildStep7Stage1CompilerStream,
  compileOmniSurfaceViaStage1ToElf,
  compilePureU16ToWasm,
  compilePureU16ToX86_64ElfWithArgs,
  executeWasmModuleI64WithBudgets,
} from "./aot.ts";
import {
  SimulatedNarrowHostAdapter,
  compileOmniSurfaceToU16,
  executeOmniSurfaceAotFirst,
  executeOmniSurfaceMultiFnWasm,
  explainOmniErrorCode,
  handleOmniHttpRequest,
  listAllErrorCodes,
  listOmniRepairStrategies,
  parseCliOmniValue,
  runOmniAgentDriver,
  runPhaseDPureOmniSelfhostBootstrap,
  serveOmniHttpViaNarrowHost,
} from "./lang.ts";
const I64_MIN = -(1n << 63n);

function main(): void {
  const report = runAcceptanceSuite();
  for (const c of report.cases) {
    const tag = c.passed ? "PASS" : "FAIL";
    console.log(`[${tag}] (${c.kind}) ${c.id}: ${c.detail}`);
  }
  console.log(
    `\nAcceptance suite (${report.manifestVersion}): ${report.passedCount}/${report.total} passed in ${report.durationMs}ms`,
  );
  if (!report.passed) {
    process.exitCode = 1;
    return;
  }

  // 1. Additional compositional invariants & boundary WASM AOT arithmetic checks
  const wrapMod = compileOmniSurfaceToU16(
    "fn main(x: i64, y: i64) -> i64 { let a: i64 = iabs(x); let b: i64 = x / y; let c: i64 = shl(1, 65); return a + b + c; }",
  );
  const wasm = compilePureU16ToWasm(wrapMod);
  assert.equal(WebAssembly.validate(wasm as BufferSource), true, "WASM validate must succeed");
  const resMin = executeWasmModuleI64WithBudgets(wasm, [I64_MIN, -1n]);
  assert.equal(resMin, 2n, "Wrapping iabs(I64_MIN) + idiv(I64_MIN, -1) + shl(1, 65) should equal 2");

  // 2. Phase D Self-Hosting Fixed-Point Bootstrap & Stage-1 ELF64 Parity
  const bootstrap = runPhaseDPureOmniSelfhostBootstrap();
  assert.equal(bootstrap.bcEqual, true, "Gen-1 == Gen-2 .omni_bc must match");
  assert.equal(bootstrap.gen1Words, 159, "gen1_compiler.omni_bc must be 159 u16 words");
  assert.equal(bootstrap.bcFnv1a64, "0x23B8D17257CCA9C4", "gen1_compiler.omni_bc FNV-1a64 must match");
  assert.equal(bootstrap.elfEqual, true, "Gen-1 == Gen-2 ELF64 must match");
  assert.equal(bootstrap.elfBytes, 577, "gen1_compiler.elf must be 577 bytes");
  assert.equal(bootstrap.elfFnv1a64, "0xDD6CAF3064840CBC", "gen1_compiler.elf FNV-1a64 must match");
  assert.equal(bootstrap.stage1Words, 17604, "Stage-1 self-compiler must be 17,604 u16 words");
  assert.equal(bootstrap.stage1BcFnv1a64, "0x11B4C6484B39E009", "Stage-1 .omni_bc FNV-1a64 must match");
  assert.equal(bootstrap.stage1ElfEqual, true, "Stage-1 == Stage-2 ELF64 must match");
  assert.equal(bootstrap.stage1ElfBytes, 36821, "Stage-1 ELF64 must be 36,821 bytes");
  assert.equal(bootstrap.stage1ElfFnv1a64, "0x4A6F0FA646FC2980", "Stage-1 ELF64 FNV-1a64 must match");
  assert.equal(bootstrap.suiteEqual, true, "20-module Pure-Core Suite Gen-1 == Gen-2 must match");
  assert.equal(bootstrap.suiteWords, 14935, "20-module Pure-Core Suite must be 14,935 u16 words");
  assert.equal(bootstrap.suiteFnv1a64, "0x9139294D92349612", "20-module Pure-Core Suite FNV-1a64 must match");
  assert.equal(bootstrap.totalPureModules, 20, "Must have 20 pure modules (8 stdlib + 5 core + 7 programs)");
  console.log(
    `[PASS] (selfhost_bootstrap) Gen-1==Gen-2 fixed-point verified: bc=${bootstrap.gen1Words}w (${bootstrap.bcFnv1a64}), elf=${bootstrap.elfBytes}B (${bootstrap.elfFnv1a64}), stage1=${bootstrap.stage1Words}w/${bootstrap.stage1ElfBytes}B (${bootstrap.stage1ElfFnv1a64}), suite=${bootstrap.suiteWords}w (${bootstrap.suiteFnv1a64})`,
  );

  // Also check Stage-1 ELF64 parity on control-flow goldens
  for (const goldenPath of [
    "spec/acceptance/goldens/branch_loop_golden.omni",
    "spec/acceptance/goldens/match_trit_golden.omni",
    "spec/acceptance/goldens/while_sum_even_golden.omni",
  ]) {
    const gSrc = fs.readFileSync(goldenPath, "utf8");
    const gStream = compileOmniSurfaceToU16(gSrc);
    const elfDirect = compilePureU16ToX86_64ElfWithArgs(gStream, 7n, 3n, 0n);
    const elfStage1 = compileOmniSurfaceViaStage1ToElf(gStream, 7n, 3n, 0n);
    assert.deepEqual(elfDirect, elfStage1, `Stage-1 ELF64 parity mismatch on ${goldenPath}`);
  }
  assert.equal(buildStep7Stage1CompilerStream().length, 17604);

  // 3. Multi-Function WebAssembly AOT Execution (Beyond single-function Rust hot-path)
  const multiFnSrc = fs.readFileSync("spec/acceptance/goldens/multi_fn_weighted_term.omni", "utf8");
  const multiFnWasmRes = executeOmniSurfaceMultiFnWasm(multiFnSrc, [7n, 3n]);
  assert.equal(multiFnWasmRes.value, 37n, "Multi-function Wasm AOT on multi_fn_weighted_term must equal 37");
  assert.equal(multiFnWasmRes.functionCount, 2);

  const stdlibPipelineSrc = fs.readFileSync("spec/acceptance/goldens/stdlib_macro_pipeline_golden.omni", "utf8");
  const stdlibWasmRes = executeOmniSurfaceMultiFnWasm(stdlibPipelineSrc, [45n, 10n, 40n]);
  assert.equal(stdlibWasmRes.value, 100n, "Multi-function Wasm AOT on stdlib_macro_pipeline_golden must equal 100");

  const routerSrc = fs.readFileSync("pure/programs/ternary_neural_router.omni", "utf8");
  const routerWasmRes = executeOmniSurfaceMultiFnWasm(routerSrc, [15n, 40n]);
  assert.equal(routerWasmRes.value, 566n, "Multi-function Wasm AOT on ternary_neural_router.omni must equal 566");
  console.log(
    `[PASS] (multi_fn_wasm_aot) Multi-function WebAssembly AOT verified across ${routerWasmRes.functionCount}-fn ternary_neural_router (${routerWasmRes.wasmBytes.length}B Wasm -> 566)`,
  );

  // 4. First-Class Surface Structs & String/Byte/Hex Intrinsics (loaded from pure/programs/struct_string_pipeline.omni)
  const structAndStrSrc = fs.readFileSync("pure/programs/struct_string_pipeline.omni", "utf8");
  const structRes = executeOmniSurfaceAotFirst(
    structAndStrSrc,
    [parseCliOmniValue("15"), parseCliOmniValue("9")],
    true,
  );
  // 77*100 + 2*10 + (15+9) + 6 + 8 = 7700 + 20 + 24 + 6 + 8 = 7758
  assert.equal(structRes.formattedResult, "7758");
  console.log("[PASS] (surface_structs_and_strings) pure/programs/struct_string_pipeline.omni -> 7758");

  // 5. Autonomous agent repair checks across multiple error classes
  const brokenCases: Array<{ name: string; src: string; args: string[]; strict: boolean }> = [
    {
      name: "missing_cap_single",
      src: "fn main(x: i64) -> trit {\n  let g: trit = egate(x);\n  return g;\n}",
      args: ["12"],
      strict: false,
    },
    {
      name: "taint_on_ret",
      src: "fn main(x: i64 @untrusted) -> i64 {\n  return x;\n}",
      args: ["42"],
      strict: false,
    },
    {
      name: "linear_leak",
      src: "fn main(x: i64) -> i64 {\n  let a: arena = arena(1024);\n  return x + 1;\n}",
      args: ["9"],
      strict: false,
    },
    {
      name: "borrow_conflict",
      src: "fn main(x: i64) -> i64 {\n  let a: arena = arena(1024);\n  let r: i64 = borrow_r(a);\n  drop(a);\n  drop(r);\n  return x;\n}",
      args: ["7"],
      strict: false,
    },
    {
      name: "strict_bad_type_and_ambiguous",
      src: "#[strict]\nfn main(x: int) -> trit {\n  let y = x;\n  return TRIT_POS;\n}",
      args: ["5"],
      strict: true,
    },
    {
      name: "corrupt_pack5",
      src: "#[strict]\nfn main() -> trit_vec {\n  let b: bytes = anew(0);\n  b = iset(b, 250);\n  let v: trit_vec = unpack5(b, 5);\n  return v;\n}",
      args: [],
      strict: true,
    },
  ];

  for (const bc of brokenCases) {
    const rep = runOmniAgentDriver(
      bc.src,
      bc.args.map(parseCliOmniValue),
      bc.strict,
      6,
    );
    assert.equal(rep.ok, true, `Agent repair failed for ${bc.name}: ${JSON.stringify(rep.history)}`);
    console.log(`[PASS] (agent_repair) ${bc.name}: repaired in ${rep.attempts} attempts -> ${rep.harness.result}`);
  }

  // 6. Repair strategies & explain inventory check
  assert.equal(listOmniRepairStrategies().length, 16, "Expected 16 repair strategies");
  assert.equal(listAllErrorCodes().length, 16, "Expected 16 error codes");
  for (const code of listAllErrorCodes()) {
    assert.ok(explainOmniErrorCode(code), `Missing explain for ${code}`);
  }

  // 7. HTTP route stateful KV roundtrip & Narrow Host Adapter
  const routeSrc = `#[cap(NET, DB_R, DB_W, AUTH)]
route POST /users/:id (req: i64) {
  let id: i64 = auth_verify(req.param("id"));
  let ok: i64 = kv_put(id, id * 10);
  respond(201, ok);
}
#[cap(NET, DB_R, AUTH)]
route GET /users/:id (req: i64) {
  let id: i64 = auth_verify(req.param("id"));
  let v: i64 = db_get(id);
  respond(200, v);
}`;
  const kv = new Map<bigint, bigint>();
  const rPost = handleOmniHttpRequest(routeSrc, "POST", "/users/21", kv);
  assert.equal(rPost.status, 201);
  const rGet = handleOmniHttpRequest(routeSrc, "GET", "/users/21", kv);
  assert.equal(rGet.status, 200);
  assert.equal(JSON.parse(rGet.body).data, 210);

  const simHost = new SimulatedNarrowHostAdapter();
  simHost.enqueueHttpRequest("POST /users/9 HTTP/1.1\r\nHost: localhost\r\n\r\n");
  simHost.enqueueHttpRequest("GET /users/9 HTTP/1.1\r\nHost: localhost\r\n\r\n");
  const narrowRes = serveOmniHttpViaNarrowHost(simHost, routeSrc, "127.0.0.1:0", "kv.db", 2);
  assert.equal(narrowRes.responses.length, 2);
  assert.equal(JSON.parse(narrowRes.responses[1].body).data, 90);
  console.log("[PASS] (http_route & narrow_host) Stateful POST/GET /users/:id roundtrip -> 210 & Narrow Host -> 90");
  console.log("\nALL SELFTESTS PASSED.");
}

main();
