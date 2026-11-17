// Omni Surface Language (`omni-lang`) -> `.omni_asm` -> `u16[]` token stream,
// `omni-harness`, `omni-agent-driver`, HTTP `route` runtime, stream inspector,
// diagnostic explainer, and Pure-Omni kernel integration.

import fs from "node:fs";
import path from "node:path";
import {
  CAP_ID_AI,
  CAP_ID_AUDIO,
  CAP_ID_AUTH,
  CAP_ID_DB_R,
  CAP_ID_DB_W,
  CAP_ID_DOM,
  CAP_ID_FS,
  CAP_ID_NET,
  CAP_ID_PURE,
  CAP_ID_SYS,
  CAP_ID_VISION,
  CapMask,
  OP_BASE,
  OMNI_ACCEPTANCE_SUITE_VERSION,
  OMNI_ERROR_CONTRACT_VERSION,
  OMNI_HARNESS_CONTRACT_VERSION,
  OMNI_REPAIR_STRATEGIES_VERSION,
  PAR_BASE,
  REG_BASE,
  TOK_INS_END,
  TOK_MOD_BEGIN,
  TOK_MOD_END,
  TOK_FN_DEF,
  TY_ARENA,
  TY_ARRAY,
  TY_BYTES,
  TY_DOM,
  TY_ERR,
  TY_F64,
  TY_FLT_VEC,
  TY_I64,
  TY_MODEL_T,
  TY_STR,
  TY_STRUCT,
  TY_TRIT,
  TY_TRIT_VEC,
  TY_UUID,
  TY_VOID,
  capBit,
  decodeOpcodeToken,
  fnv1a64Hash,
  packTc,
  unpackTc,
} from "./isa.ts";
import {
  BinOperand,
  decodeOmniTokenStream,
  packFnModule,
} from "./ir.ts";
import {
  verifyBinaryFunction,
  verifyU16Stream,
} from "./verifier.ts";
import {
  OmniBinaryVm,
  OmniValue,
} from "./vm.ts";
import {
  assembleOmniTextToU16,
  encodeAsmInsPacketU16,
  parseAsmTyTok,
} from "./assembler.ts";
import {
  DEFAULT_WASM_MEMORY_BUDGET_BYTES,
  DEFAULT_WASM_STEP_LIMIT,
  buildStep7Stage1CompilerStream,
  compileOmniSurfaceViaStage1ToElf,
  compilePureU16ToWasm,
  compilePureU16ToX86_64ElfWithArgs,
  executeWasmModuleI64WithBudgets,
  formatFnv1a64Hex,
  isModuleAotHotPath,
  isModuleMultiFnWasmCompatible,
  u16SliceToLeBytes,
} from "./aot.ts";

// ==============================================================================
// AST & LEXER TYPES
// ==============================================================================

type SurfaceTok =
  | { kind: "Ident"; value: string }
  | { kind: "Int"; value: bigint }
  | { kind: "Trit"; label: string; value: number }
  | { kind: "StrLit"; value: string }
  | { kind: "AtIdent"; value: string }
  | { kind: "Sym"; value: string }
  | { kind: "Op"; value: string }
  | { kind: "Hash" };

interface SpannedSurfaceTok {
  tok: SurfaceTok;
  line: number;
  col: number;
}

interface SurfaceParam {
  name: string;
  ty: string;
  untrusted: boolean;
  line: number;
  col: number;
}

interface SurfaceFnDef {
  fnIdAttr: number | null;
  capMask: CapMask;
  stepBudgetAttr: number | null;
  memBudgetAttr: number | null;
  budgetSpan: [number, number] | null;
  strictAttr: boolean;
  isRoute: boolean;
  routeMethod: string | null;
  routePath: string | null;
  name: string;
  fnLine: number;
  fnCol: number;
  params: SurfaceParam[];
  retTy: string;
  retLine: number;
  retCol: number;
  body: SurfaceStmt[];
}

type SurfaceExpr =
  | { kind: "Int"; value: bigint; line: number; col: number }
  | { kind: "Trit"; label: string; value: number; line: number; col: number }
  | { kind: "StrLit"; value: string; line: number; col: number }
  | { kind: "TritVecLit"; trits: number[]; line: number; col: number }
  | { kind: "Var"; name: string; line: number; col: number }
  | { kind: "UnOp"; op: string; sub: SurfaceExpr; line: number; col: number }
  | { kind: "BinOp"; op: string; lhs: SurfaceExpr; rhs: SurfaceExpr; line: number; col: number }
  | { kind: "Call"; name: string; args: SurfaceExpr[]; line: number; col: number }
  | {
      kind: "IfBranch";
      brMnem: string;
      cond: SurfaceExpr;
      thenBlk: SurfaceStmt[];
      elseBlk: SurfaceStmt[];
      line: number;
      col: number;
    }
  | {
      kind: "MatchTrit";
      cond: SurfaceExpr;
      negBlk: SurfaceStmt[];
      zeroBlk: SurfaceStmt[];
      posBlk: SurfaceStmt[];
      line: number;
      col: number;
    };

type SurfaceStmt =
  | { kind: "Let"; name: string; tyOpt: [string, number, number] | null; expr: SurfaceExpr; line: number; col: number }
  | { kind: "Assign"; name: string; expr: SurfaceExpr; line: number; col: number }
  | { kind: "DoWhile"; body: SurfaceStmt[]; cond: SurfaceExpr; line: number; col: number }
  | { kind: "While"; cond: SurfaceExpr; body: SurfaceStmt[]; line: number; col: number }
  | { kind: "Label"; id: number; line: number; col: number }
  | { kind: "Goto"; id: number; line: number; col: number }
  | { kind: "Ret"; expr: SurfaceExpr; line: number; col: number }
  | { kind: "Expr"; expr: SurfaceExpr; line: number; col: number };

function isAsciiWs(ch: string): boolean {
  return ch === " " || ch === "\t" || ch === "\r" || ch === "\n" || ch === "\v" || ch === "\f";
}

function isAsciiDigit(ch: string): boolean {
  return ch >= "0" && ch <= "9";
}

function isAsciiAlpha(ch: string): boolean {
  return (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z");
}

function isIdentChar(ch: string): boolean {
  return isAsciiAlpha(ch) || isAsciiDigit(ch) || ch === "_";
}

function debugTok(t: SurfaceTok): string {
  switch (t.kind) {
    case "Ident":
      return `Ident(${JSON.stringify(t.value)})`;
    case "Int":
      return `Int(${t.value})`;
    case "Trit":
      return `Trit(${JSON.stringify(t.label)}, ${t.value})`;
    case "StrLit":
      return `StrLit(${JSON.stringify(t.value)})`;
    case "AtIdent":
      return `AtIdent(${JSON.stringify(t.value)})`;
    case "Sym":
      return `Sym(${JSON.stringify(t.value)})`;
    case "Op":
      return `Op(${JSON.stringify(t.value)})`;
    case "Hash":
      return "Hash";
  }
}

function tokenizeOmniSurfaceInner(src: string, strict: boolean): SpannedSurfaceTok[] {
  const n = src.length;
  let i = 0;
  let line = 1;
  let col = 1;
  const toks: SpannedSurfaceTok[] = [];

  while (i < n) {
    const c = src[i];
    if (c === "\n") {
      i += 1;
      line += 1;
      col = 1;
      continue;
    }
    if (isAsciiWs(c)) {
      i += 1;
      col += 1;
      continue;
    }
    if (i + 1 < n && src[i] === "/" && src[i + 1] === "/") {
      while (i < n && src[i] !== "\n") {
        i += 1;
        col += 1;
      }
      continue;
    }
    if (c === "#") {
      if (i + 1 < n && src[i + 1] === "[") {
        toks.push({ tok: { kind: "Hash" }, line, col });
        i += 1;
        col += 1;
        continue;
      } else {
        while (i < n && src[i] !== "\n") {
          i += 1;
          col += 1;
        }
        continue;
      }
    }
    if (c === '"') {
      const startLine = line;
      const startCol = col;
      i += 1;
      col += 1;
      const start = i;
      while (i < n && src[i] !== '"' && src[i] !== "\n") {
        i += 1;
        col += 1;
      }
      if (i >= n || src[i] !== '"') {
        throw new Error(`error[SYNTAX_ERROR] line ${startLine}, col ${startCol}: Unterminated string literal`);
      }
      const s = src.slice(start, i);
      i += 1;
      col += 1;
      toks.push({ tok: { kind: "StrLit", value: s }, line: startLine, col: startCol });
      continue;
    }
    if (c === "@") {
      const startCol = col;
      i += 1;
      col += 1;
      const start = i;
      while (i < n && isIdentChar(src[i])) {
        i += 1;
        col += 1;
      }
      if (start === i) {
        throw new Error(`error[SYNTAX_ERROR] line ${line}, col ${startCol}: Expected annotation name after '@'`);
      }
      toks.push({ tok: { kind: "AtIdent", value: src.slice(start, i) }, line, col: startCol });
      continue;
    }
    if (i + 3 <= n && src.slice(i, i + 3) === "<=>") {
      toks.push({ tok: { kind: "Op", value: "<=>" }, line, col });
      i += 3;
      col += 3;
      continue;
    }
    if (i + 3 <= n && src.slice(i, i + 3) === "+1t") {
      toks.push({ tok: { kind: "Trit", label: "+1", value: 1 }, line, col });
      i += 3;
      col += 3;
      continue;
    }
    if (i + 3 <= n && src.slice(i, i + 3) === "-1t") {
      toks.push({ tok: { kind: "Trit", label: "-1t", value: -1 }, line, col });
      i += 3;
      col += 3;
      continue;
    }
    if (i + 2 <= n && src.slice(i, i + 2) === "0t") {
      const nextIsIdent = i + 2 < n && isIdentChar(src[i + 2]);
      if (!nextIsIdent) {
        toks.push({ tok: { kind: "Trit", label: "0t", value: 0 }, line, col });
        i += 2;
        col += 2;
        continue;
      }
    }
    if (i + 2 <= n) {
      const two = src.slice(i, i + 2);
      if (two === "->" || two === "=>" || two === "+=" || two === "-=" || two === "*=") {
        toks.push({ tok: { kind: "Sym", value: two }, line, col });
        i += 2;
        col += 2;
        continue;
      }
      if (two === "==" || two === "!=" || two === "<=" || two === ">=" || two === "<<" || two === ">>") {
        toks.push({ tok: { kind: "Op", value: two }, line, col });
        i += 2;
        col += 2;
        continue;
      }
    }
    if ("(){}[].,:;=".includes(c)) {
      toks.push({ tok: { kind: "Sym", value: c }, line, col });
      i += 1;
      col += 1;
      continue;
    }
    if ("+-*/%&|^~!<>".includes(c)) {
      toks.push({ tok: { kind: "Op", value: c }, line, col });
      i += 1;
      col += 1;
      continue;
    }
    if (isAsciiDigit(c)) {
      const start = i;
      const startCol = col;
      while (i < n && isAsciiDigit(src[i])) {
        i += 1;
        col += 1;
      }
      const numStr = src.slice(start, i);
      let val: bigint;
      try {
        val = BigInt.asIntN(64, BigInt(numStr));
      } catch (e) {
        throw new Error(`error[SYNTAX_ERROR] line ${line}, col ${startCol}: Invalid integer literal: ${String(e)}`);
      }
      toks.push({ tok: { kind: "Int", value: val }, line, col: startCol });
      continue;
    }
    if (isAsciiAlpha(c) || c === "_") {
      const start = i;
      const startCol = col;
      while (i < n && isIdentChar(src[i])) {
        i += 1;
        col += 1;
      }
      const ident = src.slice(start, i);
      if (ident === "TRIT_POS" || ident === "TRIT_ZERO" || ident === "TRIT_NEG") {
        if (strict) {
          throw new Error(
            `error[AMBIGUOUS_GRAMMAR] line ${line}, col ${startCol}: Non-canonical trit literal '${ident}'; use '+1t', '0t', or '-1t' in strict mode`,
          );
        }
        if (ident === "TRIT_POS") {
          toks.push({ tok: { kind: "Trit", label: "+1", value: 1 }, line, col: startCol });
        } else if (ident === "TRIT_ZERO") {
          toks.push({ tok: { kind: "Trit", label: "0t", value: 0 }, line, col: startCol });
        } else {
          toks.push({ tok: { kind: "Trit", label: "-1t", value: -1 }, line, col: startCol });
        }
      } else {
        toks.push({ tok: { kind: "Ident", value: ident }, line, col: startCol });
      }
      continue;
    }
    throw new Error(`error[SYNTAX_ERROR] line ${line}, col ${col}: Unexpected character '${c}'`);
  }

  return toks;
}

function parseSurfaceCapName(name: string, line: number, col: number): CapMask {
  switch (name.toUpperCase()) {
    case "PURE":
      return 0;
    case "DB_R":
    case "DB_READ":
      return capBit(CAP_ID_DB_R);
    case "DB_W":
    case "DB_WRITE":
    case "KV_W":
      return capBit(CAP_ID_DB_W);
    case "NET":
    case "HTTP":
      return capBit(CAP_ID_NET);
    case "DOM":
      return capBit(CAP_ID_DOM);
    case "AI":
      return capBit(CAP_ID_AI);
    case "VISION":
    case "SKILL":
      return capBit(CAP_ID_VISION);
    case "AUDIO":
      return capBit(CAP_ID_AUDIO);
    case "AUTH":
      return capBit(CAP_ID_AUTH);
    case "FS":
    case "FS_R":
    case "FS_W":
      return capBit(CAP_ID_FS);
    case "SYS":
    case "ENV":
    case "PROC":
    case "TIME":
    case "RAND":
    case "CRYPTO":
      return capBit(CAP_ID_SYS);
    default:
      throw new Error(`error[MISSING_CAP] line ${line}, col ${col}: Unknown capability '${name}'`);
  }
}

function checkMacroBodyHygiene(expr: SurfaceExpr, params: string[], macroName: string): void {
  switch (expr.kind) {
    case "Int":
    case "Trit":
    case "StrLit":
    case "TritVecLit":
      return;
    case "Var":
      if (!params.includes(expr.name)) {
        throw new Error(
          `error[UNDEF_VAR] line ${expr.line}, col ${expr.col}: Hygienic macro '${macroName}' referenced unbound variable '${expr.name}'`,
        );
      }
      return;
    case "UnOp":
      checkMacroBodyHygiene(expr.sub, params, macroName);
      return;
    case "BinOp":
      checkMacroBodyHygiene(expr.lhs, params, macroName);
      checkMacroBodyHygiene(expr.rhs, params, macroName);
      return;
    case "Call":
      for (const a of expr.args) checkMacroBodyHygiene(a, params, macroName);
      return;
    case "IfBranch":
      checkMacroBodyHygiene(expr.cond, params, macroName);
      for (const st of [...expr.thenBlk, ...expr.elseBlk]) {
        if (st.kind === "Expr" || st.kind === "Ret") checkMacroBodyHygiene(st.expr, params, macroName);
      }
      return;
    case "MatchTrit":
      checkMacroBodyHygiene(expr.cond, params, macroName);
      for (const st of [...expr.negBlk, ...expr.zeroBlk, ...expr.posBlk]) {
        if (st.kind === "Expr" || st.kind === "Ret") checkMacroBodyHygiene(st.expr, params, macroName);
      }
      return;
  }
}

function substituteMacroExpr(
  expr: SurfaceExpr,
  paramMap: Map<string, SurfaceExpr>,
  callLine: number,
  callCol: number,
): SurfaceExpr {
  switch (expr.kind) {
    case "Int":
      return { kind: "Int", value: expr.value, line: callLine, col: callCol };
    case "Trit":
      return { kind: "Trit", label: expr.label, value: expr.value, line: callLine, col: callCol };
    case "StrLit":
      return { kind: "StrLit", value: expr.value, line: callLine, col: callCol };
    case "TritVecLit":
      return { kind: "TritVecLit", trits: [...expr.trits], line: callLine, col: callCol };
    case "Var": {
      const bound = paramMap.get(expr.name);
      if (!bound) {
        throw new Error(
          `error[UNDEF_VAR] line ${expr.line}, col ${expr.col}: Hygienic macro referenced unbound variable '${expr.name}'`,
        );
      }
      return bound;
    }
    case "UnOp":
      return {
        kind: "UnOp",
        op: expr.op,
        sub: substituteMacroExpr(expr.sub, paramMap, callLine, callCol),
        line: callLine,
        col: callCol,
      };
    case "BinOp":
      return {
        kind: "BinOp",
        op: expr.op,
        lhs: substituteMacroExpr(expr.lhs, paramMap, callLine, callCol),
        rhs: substituteMacroExpr(expr.rhs, paramMap, callLine, callCol),
        line: callLine,
        col: callCol,
      };
    case "Call":
      return {
        kind: "Call",
        name: expr.name,
        args: expr.args.map((a) => substituteMacroExpr(a, paramMap, callLine, callCol)),
        line: callLine,
        col: callCol,
      };
    case "IfBranch": {
      const subCond = substituteMacroExpr(expr.cond, paramMap, callLine, callCol);
      const subThen = expr.thenBlk
        .filter((s): s is Extract<SurfaceStmt, { kind: "Expr" }> => s.kind === "Expr")
        .map((s) => ({
          kind: "Expr" as const,
          expr: substituteMacroExpr(s.expr, paramMap, callLine, callCol),
          line: callLine,
          col: callCol,
        }));
      const subElse = expr.elseBlk
        .filter((s): s is Extract<SurfaceStmt, { kind: "Expr" }> => s.kind === "Expr")
        .map((s) => ({
          kind: "Expr" as const,
          expr: substituteMacroExpr(s.expr, paramMap, callLine, callCol),
          line: callLine,
          col: callCol,
        }));
      return {
        kind: "IfBranch",
        brMnem: expr.brMnem,
        cond: subCond,
        thenBlk: subThen,
        elseBlk: subElse,
        line: callLine,
        col: callCol,
      };
    }
    case "MatchTrit": {
      const subCond = substituteMacroExpr(expr.cond, paramMap, callLine, callCol);
      const mapBlk = (blk: SurfaceStmt[]): SurfaceStmt[] =>
        blk
          .filter((s): s is Extract<SurfaceStmt, { kind: "Expr" }> => s.kind === "Expr")
          .map((s) => ({
            kind: "Expr" as const,
            expr: substituteMacroExpr(s.expr, paramMap, callLine, callCol),
            line: callLine,
            col: callCol,
          }));
      return {
        kind: "MatchTrit",
        cond: subCond,
        negBlk: mapBlk(expr.negBlk),
        zeroBlk: mapBlk(expr.zeroBlk),
        posBlk: mapBlk(expr.posBlk),
        line: callLine,
        col: callCol,
      };
    }
  }
}

function mapOmniSurfaceTypeInner(ty: string, line: number, col: number, strict: boolean): string {
  const low = ty.toLowerCase();
  if (strict && (low === "int" || low === "byte_buf" || low === "u16_array" || low === "tvec")) {
    throw new Error(
      `error[AMBIGUOUS_GRAMMAR] line ${line}, col ${col}: Non-canonical type alias '${ty}'; use 'i64', 'trit', 'bytes', 'array', 'trit_vec', or 'str' in strict mode`,
    );
  }
  switch (low) {
    case "i64":
    case "int":
      return "I64";
    case "trit":
      return "TRIT";
    case "bytes":
    case "byte_buf":
    case "pack5":
      return "BYTES";
    case "array":
    case "u16_array":
      return "ARRAY";
    case "trit_vec":
    case "tvec":
      return "TRIT_VEC";
    case "str":
      return "STR";
    case "uuid":
      return "UUID";
    case "arena":
      return "ARENA";
    case "dom":
      return "DOM";
    case "struct":
      return "STRUCT";
    default:
      throw new Error(`error[BAD_TYPE] line ${line}, col ${col}: Unknown surface type '${ty}'`);
  }
}

function fnv1a32Positive(s: string): bigint {
  const bytes = new TextEncoder().encode(s);
  let h = 0x811c9dc5;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  const masked = h & 0x3fffffff;
  return BigInt(masked > 0 ? masked : 1);
}

class SurfaceParser {
  private tokens: SpannedSurfaceTok[];
  private pos = 0;
  private strictMode: boolean;
  private extraFnCaps: CapMask = 0;
  private macros: Map<string, [string[], SurfaceExpr]> = new Map();

  constructor(tokens: SpannedSurfaceTok[], strictMode = false) {
    this.tokens = tokens;
    this.strictMode = strictMode;
  }

  private currentSpan(): [number, number] {
    const t = this.tokens[this.pos];
    if (t) return [t.line, t.col];
    const last = this.tokens[this.tokens.length - 1];
    if (last) return [last.line, last.col + 1];
    return [1, 1];
  }

  private peek(): SurfaceTok | undefined {
    return this.tokens[this.pos]?.tok;
  }

  private peekSym(sym: string): boolean {
    const t = this.peek();
    return t?.kind === "Sym" && t.value === sym;
  }

  private peekIdent(kw: string): boolean {
    const t = this.peek();
    return t?.kind === "Ident" && t.value === kw;
  }

  private nextSpanned(): SpannedSurfaceTok {
    const t = this.tokens[this.pos];
    if (!t) {
      const [l, c] = this.currentSpan();
      throw new Error(`error[SYNTAX_ERROR] line ${l}, col ${c}: Unexpected EOF in surface program`);
    }
    this.pos += 1;
    return t;
  }

  private expectSym(sym: string): [number, number] {
    const st = this.nextSpanned();
    if (st.tok.kind === "Sym" && st.tok.value === sym) {
      return [st.line, st.col];
    }
    throw new Error(
      `error[SYNTAX_ERROR] line ${st.line}, col ${st.col}: Expected symbol '${sym}', got ${debugTok(st.tok)}`,
    );
  }

  private expectIdentSpanned(): [string, number, number] {
    const st = this.nextSpanned();
    if (st.tok.kind === "Ident") {
      return [st.tok.value, st.line, st.col];
    }
    throw new Error(
      `error[SYNTAX_ERROR] line ${st.line}, col ${st.col}: Expected identifier, got ${debugTok(st.tok)}`,
    );
  }

  private expectIdentKw(kw: string): [number, number] {
    const st = this.nextSpanned();
    if (st.tok.kind === "Ident" && st.tok.value === kw) {
      return [st.line, st.col];
    }
    throw new Error(
      `error[SYNTAX_ERROR] line ${st.line}, col ${st.col}: Expected keyword '${kw}', got ${debugTok(st.tok)}`,
    );
  }

  private expectIntSpanned(): [bigint, number, number] {
    const st = this.nextSpanned();
    if (st.tok.kind === "Int") {
      return [st.tok.value, st.line, st.col];
    }
    throw new Error(
      `error[SYNTAX_ERROR] line ${st.line}, col ${st.col}: Expected integer literal, got ${debugTok(st.tok)}`,
    );
  }

  private consumeAtAnnotations(): CapMask {
    let mask: CapMask = 0;
    while (true) {
      const t = this.peek();
      if (!t || t.kind !== "AtIdent") break;
      const low = t.value.toLowerCase();
      if (low === "cap") {
        this.nextSpanned();
        this.expectSym("(");
        while (this.peek() && !this.peekSym(")")) {
          const [cname, cl, cc] = this.expectIdentSpanned();
          mask |= parseSurfaceCapName(cname, cl, cc);
          if (this.peekSym(",")) this.nextSpanned();
        }
        this.expectSym(")");
      } else if (low === "pack5" || low === "safe") {
        this.nextSpanned();
      } else {
        break;
      }
    }
    this.extraFnCaps |= mask;
    return mask;
  }

  private parseRoutePath(): string {
    const p = this.peek();
    if (p?.kind === "StrLit") {
      this.nextSpanned();
      return p.value;
    }
    let pathStr = "";
    while (true) {
      const tok = this.peek();
      if (!tok) break;
      if (tok.kind === "Op" && (tok.value === "/" || tok.value === "-" || tok.value === "*")) {
        this.nextSpanned();
        pathStr += tok.value;
      } else if (tok.kind === "Sym" && (tok.value === ":" || tok.value === ".")) {
        this.nextSpanned();
        pathStr += tok.value;
      } else if (tok.kind === "Ident") {
        this.nextSpanned();
        pathStr += tok.value;
      } else if (tok.kind === "Int") {
        this.nextSpanned();
        pathStr += tok.value.toString();
      } else {
        break;
      }
    }
    if (!pathStr) {
      const [l, c] = this.currentSpan();
      throw new Error(
        `error[SYNTAX_ERROR] line ${l}, col ${c}: Expected route path (e.g. /users/:id or "/users/:id")`,
      );
    }
    return pathStr;
  }

  public parseModule(): { includes: Array<[string, number, number]>; fns: SurfaceFnDef[] } {
    const includes: Array<[string, number, number]> = [];
    const fns: SurfaceFnDef[] = [];

    while (this.peek()) {
      if (this.peekIdent("include") || this.peekIdent("mod")) {
        const kwSt = this.nextSpanned();
        const kwStr = kwSt.tok.kind === "Ident" ? kwSt.tok.value : "include";
        if (this.strictMode && kwStr === "mod") {
          throw new Error(
            `error[AMBIGUOUS_GRAMMAR] line ${kwSt.line}, col ${kwSt.col}: Non-canonical 'mod' directive; use 'include "..."' in strict mode`,
          );
        }
        const pathSt = this.nextSpanned();
        if (pathSt.tok.kind !== "StrLit") {
          throw new Error(
            `error[SYNTAX_ERROR] line ${pathSt.line}, col ${pathSt.col}: Expected string literal after '${kwStr}', got ${debugTok(pathSt.tok)}`,
          );
        }
        if (this.peekSym(";")) this.nextSpanned();
        includes.push([pathSt.tok.value, kwSt.line, kwSt.col]);
        continue;
      }

      if (this.peekIdent("macro")) {
        this.nextSpanned();
        const [mName, ml, mc] = this.expectIdentSpanned();
        this.expectSym("(");
        const mParams: string[] = [];
        while (this.peek() && !this.peekSym(")")) {
          const [pName] = this.expectIdentSpanned();
          if (this.peekSym(":")) {
            this.nextSpanned();
            this.expectIdentSpanned();
          }
          mParams.push(pName);
          if (this.peekSym(",")) this.nextSpanned();
        }
        this.expectSym(")");
        let mBody: SurfaceExpr;
        if (this.peekSym("=")) {
          this.nextSpanned();
          mBody = this.parseExpr();
          if (this.peekSym(";")) this.nextSpanned();
        } else if (this.peekSym("{")) {
          this.nextSpanned();
          mBody = this.parseExpr();
          if (this.peekSym(";")) this.nextSpanned();
          this.expectSym("}");
          if (this.peekSym(";")) this.nextSpanned();
        } else {
          throw new Error(
            `error[SYNTAX_ERROR] line ${ml}, col ${mc}: Expected '=' or '{' in macro '${mName}' definition`,
          );
        }
        checkMacroBodyHygiene(mBody, mParams, mName);
        this.macros.set(mName, [mParams, mBody]);
        continue;
      }

      const t0 = this.tokens[this.pos]?.tok;
      const t1 = this.tokens[this.pos + 1]?.tok;
      const t2 = this.tokens[this.pos + 2]?.tok;
      const t3 = this.tokens[this.pos + 3]?.tok;
      if (
        t0?.kind === "Hash" &&
        t1?.kind === "Sym" &&
        t1.value === "[" &&
        t2?.kind === "Ident" &&
        t2.value === "strict" &&
        t3?.kind === "Sym" &&
        t3.value === "]"
      ) {
        this.pos += 4;
        this.strictMode = true;
        continue;
      }

      if (this.peekIdent("struct")) {
        this.nextSpanned();
        this.expectIdentSpanned();
        this.expectSym("{");
        while (this.peek() && !this.peekSym("}")) {
          this.expectIdentSpanned();
          this.expectSym(":");
          const [fty, fl, fc] = this.expectIdentSpanned();
          mapOmniSurfaceTypeInner(fty, fl, fc, this.strictMode);
          if (this.peekSym(",") || this.peekSym(";")) this.nextSpanned();
        }
        this.expectSym("}");
        if (this.peekSym(";")) this.nextSpanned();
        continue;
      }

      fns.push(this.parseFnDef());
    }

    if (fns.length === 0) {
      const [l, c] = this.currentSpan();
      throw new Error(
        `error[SYNTAX_ERROR] line ${l}, col ${c}: Surface program contains no 'fn' or 'route' declarations`,
      );
    }

    return { includes, fns };
  }

  private parseFnDef(): SurfaceFnDef {
    let fnIdAttr: number | null = null;
    let capMask: CapMask = 0;
    let stepBudgetAttr: number | null = null;
    let memBudgetAttr: number | null = null;
    let budgetSpan: [number, number] | null = null;
    let fnStrict = this.strictMode;
    this.extraFnCaps = 0;

    while (true) {
      const p = this.peek();
      if (p?.kind === "Hash") {
        this.nextSpanned();
        this.expectSym("[");
        const [attr, al, ac] = this.expectIdentSpanned();
        if (attr === "fn_id") {
          this.expectSym("(");
          const [idVal, il, ic] = this.expectIntSpanned();
          if (idVal < 0n || idVal > 15n) {
            throw new Error(`error[SYNTAX_ERROR] line ${il}, col ${ic}: fn_id ${idVal} out of range 0..=15`);
          }
          fnIdAttr = Number(idVal);
          this.expectSym(")");
        } else if (attr === "cap") {
          this.expectSym("(");
          while (this.peek() && !this.peekSym(")")) {
            const [cname, cl, cc] = this.expectIdentSpanned();
            capMask |= parseSurfaceCapName(cname, cl, cc);
            if (this.peekSym(",")) this.nextSpanned();
          }
          this.expectSym(")");
        } else if (attr === "step_budget" || attr === "fuel") {
          this.expectSym("(");
          const [bVal, bl, bc] = this.expectIntSpanned();
          if (bVal <= 0n) {
            throw new Error(`error[BUDGET_EXCEEDED] line ${bl}, col ${bc}: step_budget must be > 0, got ${bVal}`);
          }
          stepBudgetAttr = Number(bVal);
          budgetSpan = [al, ac];
          this.expectSym(")");
        } else if (attr === "mem_budget" || attr === "memory_budget") {
          this.expectSym("(");
          const [mVal, ml, mc] = this.expectIntSpanned();
          if (mVal <= 0n) {
            throw new Error(`error[BUDGET_EXCEEDED] line ${ml}, col ${mc}: mem_budget must be > 0, got ${mVal}`);
          }
          memBudgetAttr = Number(mVal);
          budgetSpan = [al, ac];
          this.expectSym(")");
        } else if (attr === "strict") {
          fnStrict = true;
        } else {
          throw new Error(`error[SYNTAX_ERROR] line ${al}, col ${ac}: Unsupported surface attribute '${attr}'`);
        }
        this.expectSym("]");
      } else if (p?.kind === "AtIdent") {
        capMask |= this.consumeAtAnnotations();
      } else {
        break;
      }
    }

    const prevStrict = this.strictMode;
    this.strictMode = fnStrict;

    try {
      if (this.peekIdent("route")) {
        const routeSt = this.nextSpanned();
        const [method] = this.expectIdentSpanned();
        const routePath = this.parseRoutePath();
        const params: SurfaceParam[] = [];
        if (this.peekSym("(")) {
          this.nextSpanned();
          while (this.peek() && !this.peekSym(")")) {
            const [pName] = this.expectIdentSpanned();
            this.expectSym(":");
            const [rawTy, tl, tc] = this.expectIdentSpanned();
            let untrusted = false;
            let cleanTy = rawTy;
            if (rawTy.startsWith("untrusted_")) {
              const stripped = rawTy.slice("untrusted_".length);
              if (this.strictMode) {
                throw new Error(
                  `error[AMBIGUOUS_GRAMMAR] line ${tl}, col ${tc}: Non-canonical taint prefix '${rawTy}'; use '${pName}: ${stripped} @untrusted' in strict mode`,
                );
              }
              untrusted = true;
              cleanTy = stripped;
            }
            const nextT = this.peek();
            if (nextT?.kind === "AtIdent") {
              const st = this.nextSpanned();
              const tag = st.tok.kind === "AtIdent" ? st.tok.value.toLowerCase() : "";
              if (tag === "untrusted") untrusted = true;
              else if (tag === "safe") untrusted = false;
              else {
                throw new Error(`error[BAD_TYPE] line ${st.line}, col ${st.col}: Unknown taint annotation '@${tag}'`);
              }
            }
            params.push({ name: pName, ty: cleanTy, untrusted, line: tl, col: tc });
            if (this.peekSym(",")) this.nextSpanned();
          }
          this.expectSym(")");
        }
        if (params.length === 0) {
          params.push({ name: "req", ty: "i64", untrusted: false, line: routeSt.line, col: routeSt.col });
        }
        capMask |= this.consumeAtAnnotations();
        let retTy = "i64";
        let rl = routeSt.line;
        let rc = routeSt.col;
        if (this.peekSym("->")) {
          this.nextSpanned();
          [retTy, rl, rc] = this.expectIdentSpanned();
        }
        capMask |= this.consumeAtAnnotations();
        const body = this.parseBlock();
        capMask |= this.extraFnCaps;
        const sanitizedPath = routePath.replace(/[^a-zA-Z0-9]/g, "_");
        return {
          fnIdAttr,
          capMask,
          stepBudgetAttr,
          memBudgetAttr,
          budgetSpan,
          strictAttr: fnStrict,
          isRoute: true,
          routeMethod: method.toUpperCase(),
          routePath,
          name: `route_${method.toLowerCase()}_${sanitizedPath}`,
          fnLine: routeSt.line,
          fnCol: routeSt.col,
          params,
          retTy,
          retLine: rl,
          retCol: rc,
          body,
        };
      }

      const [fnKwLine, fnKwCol] = this.expectIdentKw("fn");
      const [fnName] = this.expectIdentSpanned();
      this.expectSym("(");
      const params: SurfaceParam[] = [];
      while (this.peek() && !this.peekSym(")")) {
        const [pName] = this.expectIdentSpanned();
        this.expectSym(":");
        const [rawTy, tl, tc] = this.expectIdentSpanned();
        let untrusted = false;
        let cleanTy = rawTy;
        if (rawTy.startsWith("untrusted_")) {
          const stripped = rawTy.slice("untrusted_".length);
          if (this.strictMode) {
            throw new Error(
              `error[AMBIGUOUS_GRAMMAR] line ${tl}, col ${tc}: Non-canonical taint prefix '${rawTy}'; use '${pName}: ${stripped} @untrusted' in strict mode`,
            );
          }
          untrusted = true;
          cleanTy = stripped;
        }
        const nextT = this.peek();
        if (nextT?.kind === "AtIdent") {
          const st = this.nextSpanned();
          const tag = st.tok.kind === "AtIdent" ? st.tok.value.toLowerCase() : "";
          if (tag === "untrusted") untrusted = true;
          else if (tag === "safe") untrusted = false;
          else {
            throw new Error(`error[BAD_TYPE] line ${st.line}, col ${st.col}: Unknown taint annotation '@${tag}'`);
          }
        }
        params.push({ name: pName, ty: cleanTy, untrusted, line: tl, col: tc });
        if (this.peekSym(",")) this.nextSpanned();
      }
      this.expectSym(")");
      capMask |= this.consumeAtAnnotations();
      this.expectSym("->");
      const [retTy, rl, rc] = this.expectIdentSpanned();
      capMask |= this.consumeAtAnnotations();
      const body = this.parseBlock();
      capMask |= this.extraFnCaps;

      return {
        fnIdAttr,
        capMask,
        stepBudgetAttr,
        memBudgetAttr,
        budgetSpan,
        strictAttr: fnStrict,
        isRoute: false,
        routeMethod: null,
        routePath: null,
        name: fnName,
        fnLine: fnKwLine,
        fnCol: fnKwCol,
        params,
        retTy,
        retLine: rl,
        retCol: rc,
        body,
      };
    } finally {
      this.strictMode = prevStrict;
    }
  }

  private parseBlock(): SurfaceStmt[] {
    this.expectSym("{");
    const stmts: SurfaceStmt[] = [];
    while (this.peek() && !this.peekSym("}")) {
      stmts.push(this.parseStmt());
    }
    this.expectSym("}");
    return stmts;
  }

  private parseStmt(): SurfaceStmt {
    this.consumeAtAnnotations();
    const [line, col] = this.currentSpan();
    const t = this.peek();

    if (t?.kind === "Ident" && t.value === "let") {
      this.nextSpanned();
      const [name, nl, nc] = this.expectIdentSpanned();
      let tyOpt: [string, number, number] | null = null;
      if (this.peekSym(":")) {
        this.nextSpanned();
        const [tyAnn, tl, tc] = this.expectIdentSpanned();
        mapOmniSurfaceTypeInner(tyAnn, tl, tc, this.strictMode);
        this.consumeAtAnnotations();
        tyOpt = [tyAnn, tl, tc];
      } else if (this.strictMode) {
        throw new Error(
          `error[BAD_TYPE] line ${nl}, col ${nc}: Missing explicit type annotation on 'let ${name}' (strict mode requires 'let ${name}: <ty> = ...;')`,
        );
      }
      this.expectSym("=");
      const expr = this.parseExpr();
      this.consumeAtAnnotations();
      if (this.peekSym(";")) this.nextSpanned();
      return { kind: "Let", name, tyOpt, expr, line: nl, col: nc };
    }

    if (t?.kind === "Ident" && (t.value === "return" || t.value === "respond" || t.value === "respond_json" || t.value === "respond_html")) {
      if (t.value === "respond" || t.value === "respond_json" || t.value === "respond_html") {
        const expr = this.parseExpr();
        this.consumeAtAnnotations();
        if (this.peekSym(";")) this.nextSpanned();
        return { kind: "Ret", expr, line, col };
      }
      this.nextSpanned();
      const expr = this.parseExpr();
      this.consumeAtAnnotations();
      if (this.peekSym(";")) this.nextSpanned();
      return { kind: "Ret", expr, line, col };
    }

    if (t?.kind === "Ident" && t.value === "do_while") {
      this.nextSpanned();
      const body = this.parseBlock();
      const cond = this.parseExpr();
      if (this.peekSym(";")) this.nextSpanned();
      return { kind: "DoWhile", body, cond, line, col };
    }

    if (t?.kind === "Ident" && t.value === "while") {
      this.nextSpanned();
      const cond = this.parseExpr();
      const body = this.parseBlock();
      if (this.peekSym(";")) this.nextSpanned();
      return { kind: "While", cond, body, line, col };
    }

    if (t?.kind === "Ident" && t.value === "label") {
      this.nextSpanned();
      const [lid, ll, lc] = this.expectIntSpanned();
      if (lid < 0n || lid > 15n) {
        throw new Error(`error[UNRESOLVED_LABEL] line ${ll}, col ${lc}: Label id ${lid} out of range 0..=15`);
      }
      if (this.peekSym(";")) this.nextSpanned();
      return { kind: "Label", id: Number(lid), line: ll, col: lc };
    }

    if (t?.kind === "Ident" && t.value === "goto") {
      this.nextSpanned();
      const [lid, ll, lc] = this.expectIntSpanned();
      if (lid < 0n || lid > 15n) {
        throw new Error(`error[UNRESOLVED_LABEL] line ${ll}, col ${lc}: Goto label id ${lid} out of range 0..=15`);
      }
      if (this.peekSym(";")) this.nextSpanned();
      return { kind: "Goto", id: Number(lid), line: ll, col: lc };
    }

    const nextTok = this.tokens[this.pos + 1]?.tok;
    if (
      t?.kind === "Ident" &&
      nextTok?.kind === "Sym" &&
      (nextTok.value === "=" || nextTok.value === "+=" || nextTok.value === "-=" || nextTok.value === "*=")
    ) {
      const [name, nl, nc] = this.expectIdentSpanned();
      const assignTok = this.nextSpanned();
      const assignOp = assignTok.tok.kind === "Sym" ? assignTok.tok.value : "=";
      const rhs = this.parseExpr();
      this.consumeAtAnnotations();
      if (this.peekSym(";")) this.nextSpanned();
      let expr: SurfaceExpr = rhs;
      if (assignOp === "+=") {
        expr = { kind: "BinOp", op: "IADD", lhs: { kind: "Var", name, line: nl, col: nc }, rhs, line: nl, col: nc };
      } else if (assignOp === "-=") {
        expr = { kind: "BinOp", op: "ISUB", lhs: { kind: "Var", name, line: nl, col: nc }, rhs, line: nl, col: nc };
      } else if (assignOp === "*=") {
        expr = { kind: "BinOp", op: "IMUL", lhs: { kind: "Var", name, line: nl, col: nc }, rhs, line: nl, col: nc };
      }
      return { kind: "Assign", name, expr, line: nl, col: nc };
    }

    const expr = this.parseExpr();
    this.consumeAtAnnotations();
    if (this.peekSym(";")) this.nextSpanned();
    return { kind: "Expr", expr, line, col };
  }

  private parseExpr(): SurfaceExpr {
    const e = this.parseCmp();
    this.consumeAtAnnotations();
    return e;
  }

  private parseCmp(): SurfaceExpr {
    let lhs = this.parseBitOr();
    while (true) {
      const t = this.peek();
      if (!t || t.kind !== "Op") break;
      if (t.value === "==" || t.value === "!=" || t.value === "<" || t.value === ">" || t.value === "<=>") {
        const mnem =
          t.value === "=="
            ? "CEQ"
            : t.value === "!="
              ? "CNE"
              : t.value === "<"
                ? "CLT"
                : t.value === ">"
                  ? "CGT"
                  : "TCMP";
        const st = this.nextSpanned();
        const rhs = this.parseBitOr();
        lhs = { kind: "BinOp", op: mnem, lhs, rhs, line: st.line, col: st.col };
      } else if (t.value === "<=") {
        const st = this.nextSpanned();
        const rhs = this.parseBitOr();
        const gt: SurfaceExpr = { kind: "BinOp", op: "CGT", lhs, rhs, line: st.line, col: st.col };
        lhs = { kind: "UnOp", op: "TNOT", sub: gt, line: st.line, col: st.col };
      } else if (t.value === ">=") {
        const st = this.nextSpanned();
        const rhs = this.parseBitOr();
        const lt: SurfaceExpr = { kind: "BinOp", op: "CLT", lhs, rhs, line: st.line, col: st.col };
        lhs = { kind: "UnOp", op: "TNOT", sub: lt, line: st.line, col: st.col };
      } else {
        break;
      }
    }
    return lhs;
  }

  private parseBitOr(): SurfaceExpr {
    let lhs = this.parseBitXor();
    while (this.peek()?.kind === "Op" && (this.peek() as { kind: "Op"; value: string }).value === "|") {
      const st = this.nextSpanned();
      const rhs = this.parseBitXor();
      lhs = { kind: "BinOp", op: "BOR", lhs, rhs, line: st.line, col: st.col };
    }
    return lhs;
  }

  private parseBitXor(): SurfaceExpr {
    let lhs = this.parseBitAnd();
    while (this.peek()?.kind === "Op" && (this.peek() as { kind: "Op"; value: string }).value === "^") {
      const st = this.nextSpanned();
      const rhs = this.parseBitAnd();
      lhs = { kind: "BinOp", op: "BXOR", lhs, rhs, line: st.line, col: st.col };
    }
    return lhs;
  }

  private parseBitAnd(): SurfaceExpr {
    let lhs = this.parseShift();
    while (this.peek()?.kind === "Op" && (this.peek() as { kind: "Op"; value: string }).value === "&") {
      const st = this.nextSpanned();
      const rhs = this.parseShift();
      lhs = { kind: "BinOp", op: "BAND", lhs, rhs, line: st.line, col: st.col };
    }
    return lhs;
  }

  private parseShift(): SurfaceExpr {
    let lhs = this.parseAdd();
    while (true) {
      const t = this.peek();
      if (!t || t.kind !== "Op" || (t.value !== "<<" && t.value !== ">>")) break;
      const mnem = t.value === "<<" ? "SHL" : "SHR";
      const st = this.nextSpanned();
      const rhs = this.parseAdd();
      lhs = { kind: "BinOp", op: mnem, lhs, rhs, line: st.line, col: st.col };
    }
    return lhs;
  }

  private parseAdd(): SurfaceExpr {
    let lhs = this.parseMul();
    while (true) {
      const t = this.peek();
      if (!t || t.kind !== "Op" || (t.value !== "+" && t.value !== "-")) break;
      const mnem = t.value === "+" ? "IADD" : "ISUB";
      const st = this.nextSpanned();
      const rhs = this.parseMul();
      lhs = { kind: "BinOp", op: mnem, lhs, rhs, line: st.line, col: st.col };
    }
    return lhs;
  }

  private parseMul(): SurfaceExpr {
    let lhs = this.parseUnary();
    while (true) {
      const t = this.peek();
      if (!t || t.kind !== "Op" || (t.value !== "*" && t.value !== "/" && t.value !== "%")) break;
      const mnem = t.value === "*" ? "IMUL" : t.value === "/" ? "IDIV" : "IMOD";
      const st = this.nextSpanned();
      const rhs = this.parseUnary();
      lhs = { kind: "BinOp", op: mnem, lhs, rhs, line: st.line, col: st.col };
    }
    return lhs;
  }

  private parseUnary(): SurfaceExpr {
    const t = this.peek();
    if (t?.kind === "Op") {
      if (t.value === "-") {
        const st = this.nextSpanned();
        const sub = this.parseUnary();
        if (sub.kind === "Int") {
          return { kind: "Int", value: BigInt.asIntN(64, -sub.value), line: st.line, col: st.col };
        }
        return { kind: "UnOp", op: "INEG", sub, line: st.line, col: st.col };
      }
      if (t.value === "+") {
        this.nextSpanned();
        return this.parseUnary();
      }
      if (t.value === "~") {
        const st = this.nextSpanned();
        const sub = this.parseUnary();
        return { kind: "UnOp", op: "BNOT", sub, line: st.line, col: st.col };
      }
      if (t.value === "!") {
        const st = this.nextSpanned();
        const sub = this.parseUnary();
        return { kind: "UnOp", op: "TNOT", sub, line: st.line, col: st.col };
      }
    }
    return this.parsePrimary();
  }

  private parsePrimary(): SurfaceExpr {
    this.consumeAtAnnotations();
    const [line, col] = this.currentSpan();
    const t = this.peek();
    if (!t) {
      throw new Error(`error[SYNTAX_ERROR] line ${line}, col ${col}: Unexpected EOF in surface expression`);
    }
    switch (t.kind) {
      case "Int": {
        const st = this.nextSpanned();
        return { kind: "Int", value: (st.tok as { kind: "Int"; value: bigint }).value, line: st.line, col: st.col };
      }
      case "Trit": {
        const st = this.nextSpanned();
        const tt = st.tok as { kind: "Trit"; label: string; value: number };
        return { kind: "Trit", label: tt.label, value: tt.value, line: st.line, col: st.col };
      }
      case "StrLit": {
        const st = this.nextSpanned();
        return { kind: "StrLit", value: (st.tok as { kind: "StrLit"; value: string }).value, line: st.line, col: st.col };
      }
      case "Sym": {
        if (t.value === "[") {
          const st = this.nextSpanned();
          const trits: number[] = [];
          while (this.peek() && !this.peekSym("]")) {
            const elem = this.parseExpr();
            if (elem.kind === "Trit") {
              trits.push(elem.value);
            } else if (elem.kind === "Int" && elem.value >= -1n && elem.value <= 1n) {
              if (this.strictMode) {
                throw new Error(
                  `error[AMBIGUOUS_GRAMMAR] line ${elem.line}, col ${elem.col}: Trit vector literal elements must use canonical trit tokens (-1t, 0t, +1t) in strict mode`,
                );
              }
              trits.push(Number(elem.value));
            } else {
              throw new Error(
                `error[BAD_TYPE] line ${elem.line}, col ${elem.col}: Trit vector literal elements must be balanced trits (-1t, 0t, +1t)`,
              );
            }
            if (this.peekSym(",")) this.nextSpanned();
          }
          this.expectSym("]");
          if (trits.length === 0) {
            throw new Error(`error[BAD_TYPE] line ${st.line}, col ${st.col}: Trit vector literal cannot be empty`);
          }
          return { kind: "TritVecLit", trits, line: st.line, col: st.col };
        }
        if (t.value === "(") {
          this.nextSpanned();
          const e = this.parseExpr();
          this.expectSym(")");
          return e;
        }
        throw new Error(
          `error[SYNTAX_ERROR] line ${line}, col ${col}: Unexpected token in surface expression: ${debugTok(t)}`,
        );
      }
      case "Ident": {
        if (t.value === "if_pos" || t.value === "if_neg") {
          const brMnem = t.value === "if_pos" ? "BR_POS" : "BR_NEG";
          const st = this.nextSpanned();
          const cond = this.parseExpr();
          const thenBlk = this.parseBlock();
          this.expectIdentKw("else");
          const elseBlk = this.parseBlock();
          return { kind: "IfBranch", brMnem, cond, thenBlk, elseBlk, line: st.line, col: st.col };
        }
        if (t.value === "match_trit") {
          const st = this.nextSpanned();
          const cond = this.parseExpr();
          this.expectSym("{");
          let negArm: SurfaceStmt[] | null = null;
          let zeroArm: SurfaceStmt[] | null = null;
          let posArm: SurfaceStmt[] | null = null;
          while (this.peek() && !this.peekSym("}")) {
            const [armName, al, ac] = this.expectIdentSpanned();
            this.expectSym("=>");
            let armStmts: SurfaceStmt[];
            if (this.peekSym("{")) {
              armStmts = this.parseBlock();
            } else {
              const e = this.parseExpr();
              armStmts = [{ kind: "Expr", expr: e, line: e.line, col: e.col }];
            }
            if (this.peekSym(",")) this.nextSpanned();
            if (armName === "neg") negArm = armStmts;
            else if (armName === "zero") zeroArm = armStmts;
            else if (armName === "pos") posArm = armStmts;
            else {
              throw new Error(
                `error[SYNTAX_ERROR] line ${al}, col ${ac}: Unknown match_trit arm '${armName}' (expected neg, zero, pos)`,
              );
            }
          }
          this.expectSym("}");
          if (!negArm) throw new Error(`error[SYNTAX_ERROR] line ${st.line}, col ${st.col}: match_trit missing 'neg' arm`);
          if (!zeroArm) throw new Error(`error[SYNTAX_ERROR] line ${st.line}, col ${st.col}: match_trit missing 'zero' arm`);
          if (!posArm) throw new Error(`error[SYNTAX_ERROR] line ${st.line}, col ${st.col}: match_trit missing 'pos' arm`);
          return {
            kind: "MatchTrit",
            cond,
            negBlk: negArm,
            zeroBlk: zeroArm,
            posBlk: posArm,
            line: st.line,
            col: st.col,
          };
        }

        const st = this.nextSpanned();
        const ident = (st.tok as { kind: "Ident"; value: string }).value;
        if (this.peekSym(".")) {
          this.nextSpanned();
          const [methodName, ml, mc] = this.expectIdentSpanned();
          if (!this.peekSym("(")) {
            this.consumeAtAnnotations();
            return {
              kind: "Call",
              name: "fget",
              args: [
                { kind: "Var", name: ident, line: st.line, col: st.col },
                { kind: "StrLit", value: methodName, line: ml, col: mc },
              ],
              line: st.line,
              col: st.col,
            };
          }
          this.expectSym("(");
          const args: SurfaceExpr[] = [{ kind: "Var", name: ident, line: st.line, col: st.col }];
          while (this.peek() && !this.peekSym(")")) {
            args.push(this.parseExpr());
            if (this.peekSym(",")) this.nextSpanned();
          }
          this.expectSym(")");
          this.consumeAtAnnotations();
          let callName: string;
          if (ident === "req" && methodName === "param") callName = "req_param";
          else if ((ident === "db" || ident === "kv") && methodName === "query") callName = "db_query";
          else if ((ident === "db" || ident === "kv") && methodName === "get") callName = "db_get";
          else if ((ident === "db" || ident === "kv") && methodName === "put") callName = "kv_put";
          else callName = `${ident}_${methodName}`;
          if (callName === "db_query" || callName === "db_get" || callName === "kv_put") {
            args.shift();
          }
          return { kind: "Call", name: callName, args, line: st.line, col: st.col };
        }

        if (this.peekSym("(")) {
          this.nextSpanned();
          const args: SurfaceExpr[] = [];
          while (this.peek() && !this.peekSym(")")) {
            args.push(this.parseExpr());
            if (this.peekSym(",")) this.nextSpanned();
          }
          this.expectSym(")");
          this.consumeAtAnnotations();
          const macroDef = this.macros.get(ident);
          if (macroDef) {
            const [mParams, mBody] = macroDef;
            if (args.length !== mParams.length) {
              throw new Error(
                `error[ARITY_MISMATCH] line ${st.line}, col ${st.col}: Hygienic macro '${ident}' expects ${mParams.length} arguments, got ${args.length}`,
              );
            }
            const paramMap = new Map<string, SurfaceExpr>();
            for (let idx = 0; idx < mParams.length; idx++) {
              paramMap.set(mParams[idx], args[idx]);
            }
            return substituteMacroExpr(mBody, paramMap, st.line, st.col);
          }
          return { kind: "Call", name: ident, args, line: st.line, col: st.col };
        }

        return { kind: "Var", name: ident, line: st.line, col: st.col };
      }
      default:
        throw new Error(
          `error[SYNTAX_ERROR] line ${line}, col ${col}: Unexpected token in surface expression: ${debugTok(t)}`,
        );
    }
  }
}

// ==============================================================================
// BUILT-IN STDLIB & INCLUDE PREPROCESSOR
// ==============================================================================

const BUILTIN_STDLIB_MODULES: Record<string, string> = {
  "std/prelude.omni": `#[strict] fn std_clamp(x: i64, lo: i64, hi: i64) -> i64 { let lower: i64 = imax(x, lo); let clamped: i64 = imin(lower, hi); return clamped; }
#[strict] fn std_abs_diff(a: i64, b: i64) -> i64 { let d: i64 = a - b; let r: i64 = iabs(d); return r; }
#[strict] fn std_trit_mux(cond: trit, neg_v: i64, zero_v: i64, pos_v: i64) -> i64 { let r: i64 = sel_trit(cond, neg_v, zero_v, pos_v); return r; }
#[strict] fn std_consensus3(s0: trit, s1: trit, s2: trit) -> trit { let m: trit = tmaj(s0, s1, s2); return m; }`,

  "std/math.omni": `#[strict] fn std_clamp(x: i64, lo: i64, hi: i64) -> i64 { let lower: i64 = imax(x, lo); let clamped: i64 = imin(lower, hi); return clamped; }
#[strict] fn std_abs_diff(a: i64, b: i64) -> i64 { let d: i64 = a - b; let r: i64 = iabs(d); return r; }
#[strict] fn std_signum(x: i64) -> i64 { let c: trit = x <=> 0; let s: i64 = sel_trit(c, -1, 0, 1); return s; }
#[strict] fn std_mad_clamped(x: i64, weight: i64, bias: i64, lo: i64, hi: i64) -> i64 { let raw: i64 = x * weight + bias; let lower: i64 = imax(raw, lo); let res: i64 = imin(lower, hi); return res; }
#[strict] fn std_sum_range(n: i64) -> i64 { let acc: i64 = 0; let i: i64 = 1; while i <= n { acc += i; i += 1; } return acc; }`,

  "std/trit.omni": `#[strict] fn std_trit_mux(cond: trit, neg_v: i64, zero_v: i64, pos_v: i64) -> i64 { let r: i64 = sel_trit(cond, neg_v, zero_v, pos_v); return r; }
#[strict] fn std_consensus3(s0: trit, s1: trit, s2: trit) -> trit { let m: trit = tmaj(s0, s1, s2); return m; }
#[strict] fn std_trit_quorum5(t0: trit, t1: trit, t2: trit, t3: trit, t4: trit) -> trit { let left: trit = tmaj(t0, t1, t2); let right: trit = tmaj(t2, t3, t4); let out: trit = tmaj(left, t2, right); return out; }
#[strict] fn std_trit_score(t0: trit, t1: trit, t2: trit, weight: i64) -> i64 { let m: trit = tmaj(t0, t1, t2); let base: i64 = sel_trit(m, -100, 0, 100); return base * weight; }`,

  "std/http.omni": `#[strict] fn std_http_is_success(status: i64) -> trit { let ge_200: trit = status >= 200; let lt_300: trit = status < 300; let ok: trit = tand(ge_200, lt_300); return ok; }
#[strict] fn std_http_pack_response(status: i64, payload: i64) -> i64 { let valid_cmp: trit = status >= 100; let code: i64 = if_pos valid_cmp { status } else { 500 }; return code * 1000000 + payload; }
#[strict] fn std_http_route_score(method_code: i64, param_id: i64) -> i64 { let m_cmp: trit = method_code <=> 1; let method_bonus: i64 = match_trit m_cmp { neg => 400, zero => 200, pos => 201 }; return method_bonus * 1000000 + param_id; }`,

  "std/pack5.omni": `#[strict] fn std_pack5_bytes_needed(trit_count: i64) -> i64 { let valid_cmp: trit = trit_count <=> 0; let padded: i64 = trit_count + 4; let bytes: i64 = padded / 5; let out: i64 = if_pos valid_cmp { bytes } else { 0 }; return out; }
#[strict] fn std_pack5_is_valid_byte(byte_val: i64) -> trit { let ge_zero: trit = byte_val >= 0; let le_max: trit = byte_val <= 242; let ok: trit = tand(ge_zero, le_max); return ok; }
#[strict] fn std_pack5_dot10_score(dot_raw: i64, scale: i64, threshold: i64) -> i64 { let scaled: i64 = dot_raw * scale; let cmp: trit = scaled <=> threshold; let bonus: i64 = match_trit cmp { neg => -50, zero => 0, pos => 50 }; return scaled + bonus; }
#[strict] fn std_pack5_cosine_gate(dot_raw: i64, norm_bound: i64) -> trit { let hi_cmp: trit = dot_raw <=> norm_bound; let neg_bound: i64 = 0 - norm_bound; let lo_cmp: trit = dot_raw <=> neg_bound; let gate: trit = tmaj(hi_cmp, lo_cmp, 0t); return gate; }`,

  "std/decision.omni": `#[strict] fn std_decision_band(signal: i64, lo_thresh: i64, hi_thresh: i64) -> trit { let above_hi: trit = signal <=> hi_thresh; let below_lo: trit = signal <=> lo_thresh; let is_high: trit = tand(above_hi, +1t); let is_low: trit = tor(below_lo, -1t); let out: trit = match_trit above_hi { neg => is_low, zero => 0t, pos => is_high }; return out; }
#[strict] fn std_decision_hysteresis(prev_state: trit, signal: i64, margin: i64) -> trit { let neg_margin: i64 = 0 - margin; let hi_cmp: trit = signal <=> margin; let lo_cmp: trit = signal <=> neg_margin; let next_state: trit = match_trit hi_cmp { neg => match_trit lo_cmp { neg => -1t, zero => prev_state, pos => prev_state }, zero => prev_state, pos => +1t }; return next_state; }
#[strict] #[cap(AI)] fn std_decision_weighted_hook(signal: i64, bias: i64, neg_out: i64, zero_out: i64, pos_out: i64) -> i64 { let net: i64 = signal + bias; let g: trit = egate(net); let l: trit = tlogic(net); let combined: trit = tmaj(g, l, g); let res: i64 = sel_trit(combined, neg_out, zero_out, pos_out); return res; }
#[strict] #[cap(AI)] fn std_decision_quorum_hook(s0: i64, s1: i64, s2: i64, threshold: i64) -> trit { let g0: trit = egate(s0 - threshold); let g1: trit = egate(s1 - threshold); let g2: trit = egate(s2 - threshold); let q: trit = tmaj(g0, g1, g2); return q; }`,

  "std/resource.omni": `#[strict] fn std_arena_scoped_score(base: i64, weight: i64, offset: i64) -> i64 { let a_scope: arena = arena(1024); let r_view: i64 = borrow_r(a_scope); let prod: i64 = base * weight; let total: i64 = prod + offset + (r_view * 0); drop(r_view); drop(a_scope); return total; }
#[strict] fn std_budget_headroom(used_units: i64, limit_units: i64) -> i64 { let raw_rem: i64 = limit_units - used_units; let clamped: i64 = imax(raw_rem, 0); return clamped; }
#[strict] fn std_budget_is_within(used_units: i64, limit_units: i64, safety_margin: i64) -> trit { let effective_used: i64 = used_units + safety_margin; let cmp: trit = effective_used <= limit_units; return cmp; }
#[strict] fn std_arena_chunk_capacity(total_bytes: i64, chunk_size: i64) -> i64 { let valid_chunk: i64 = imax(chunk_size, 64); let chunks: i64 = total_bytes / valid_chunk; let clamped: i64 = imax(chunks, 1); return clamped; }`,
};

export function lookupOmniStdlibModule(modulePath: string): string | null {
  if (BUILTIN_STDLIB_MODULES[modulePath]) {
    return BUILTIN_STDLIB_MODULES[modulePath];
  }
  const normalized = modulePath.replace(/\\/g, "/");
  if (BUILTIN_STDLIB_MODULES[normalized]) {
    return BUILTIN_STDLIB_MODULES[normalized];
  }
  if (
    (normalized.startsWith("std/") ||
      normalized.startsWith("pure/") ||
      normalized.startsWith("spec/pure_self_host/")) &&
    fs.existsSync(normalized)
  ) {
    const content = fs.readFileSync(normalized, "utf8");
    BUILTIN_STDLIB_MODULES[normalized] = content;
    return content;
  }
  return null;
}

export function resolveOmniSurfaceIncludes(
  src: string,
  baseDir?: string,
  vfs?: Map<string, string>,
): string {
  const resolveInner = (curSrc: string, curDir: string | undefined, stack: string[]): string => {
    if (!curSrc.includes("include") && !curSrc.includes("mod ") && !curSrc.includes('mod"')) {
      return curSrc;
    }
    const tokens = tokenizeOmniSurfaceInner(curSrc, false);
    const parser = new SurfaceParser(tokens, false);
    const { includes } = parser.parseModule();
    if (includes.length === 0) return curSrc;

    let combined = "";
    for (const [incPath, line, col] of includes) {
      let incContent: string | undefined = vfs?.get(incPath);
      let nextDir = curDir;
      let cycleKey = incPath.replace(/\\/g, "/");

      if (incContent === undefined) {
        const stdSrc = lookupOmniStdlibModule(incPath);
        if (stdSrc !== null) {
          incContent = stdSrc;
        }
      }
      if (incContent === undefined) {
        const relCandidate = curDir ? path.join(curDir, incPath) : incPath;
        const candidate = fs.existsSync(relCandidate) ? relCandidate : incPath;
        const resolved = path.resolve(candidate);
        cycleKey = resolved;
        if (stack.includes(cycleKey) || stack.includes(incPath)) {
          throw new Error(
            `error[CIRCULAR_INCLUDE] line ${line}, col ${col}: Circular include detected for '${incPath}'`,
          );
        }
        if (!fs.existsSync(candidate)) {
          throw new Error(
            `error[SYNTAX_ERROR] line ${line}, col ${col}: Failed to read included file '${relCandidate}'`,
          );
        }
        incContent = fs.readFileSync(candidate, "utf8");
        nextDir = path.dirname(resolved);
      } else if (stack.includes(cycleKey) || stack.includes(incPath)) {
        throw new Error(
          `error[CIRCULAR_INCLUDE] line ${line}, col ${col}: Circular include detected for '${incPath}'`,
        );
      }

      stack.push(cycleKey);
      const expanded = resolveInner(incContent, nextDir, stack);
      stack.pop();
      combined += expanded + "\n";
    }

    for (const rawLine of curSrc.split(/\r?\n/)) {
      const trimmed = rawLine.trimStart();
      if (trimmed.startsWith("include ") || trimmed.startsWith('include"') || trimmed.startsWith("mod ")) {
        continue;
      }
      combined += rawLine + "\n";
    }
    return combined;
  };

  return resolveInner(src, baseDir, []);
}

// ==============================================================================
// LOWERER & SOURCEMAP
// ==============================================================================

function pack5EncodeLocal(trits: number[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < trits.length; i += 5) {
    let acc = 0;
    let pow = 1;
    for (let j = 0; j < 5; j++) {
      const t = i + j < trits.length ? trits[i + j] : 0;
      const d = t < 0 ? 0 : t > 0 ? 2 : 1;
      acc += d * pow;
      pow *= 3;
    }
    out.push(acc);
  }
  return out;
}

class SurfaceLowerCtx {
  private varMap: Map<string, string> = new Map();
  private taintedOperands: Set<string> = new Set();
  private fnSigs: Map<string, [number, number, string]>;
  private nextReg = 0;
  private nextLbl = 1;
  private definedLabels: number[] = [];
  private usedLabels: Array<[number, number, number]> = [];
  private retTyTok: number;
  private emitAsm: boolean;
  public curSpan: [number, number] = [1, 1];
  public lines: string[] = [];
  public packets: number[][] = [];
  public pktSpans: Array<[number, number]> = [];

  constructor(
    params: SurfaceParam[],
    fnSigs: Map<string, [number, number, string]>,
    retTyTok: number,
    emitAsm: boolean,
  ) {
    this.fnSigs = fnSigs;
    this.retTyTok = retTyTok;
    this.emitAsm = emitAsm;
    params.forEach((p, idx) => {
      const opStr = `$${idx}`;
      if (p.untrusted) this.taintedOperands.add(opStr);
      this.varMap.set(p.name, opStr);
    });
  }

  public emit(lineInput: string): void {
    let line = lineInput;
    const parts = line.trim().split(/\s+/).filter(Boolean);
    if (parts.length >= 4 && parts[1] === "=") {
      const dst = parts[0];
      const mnem = parts[2];
      const baseMnem = mnem.endsWith("@UNTRUSTED")
        ? mnem.slice(0, -"@UNTRUSTED".length)
        : mnem.endsWith("@SAFE")
          ? mnem.slice(0, -"@SAFE".length)
          : mnem;
      const isSanitizer =
        baseMnem === "SAN_SQL" ||
        baseMnem === "SAN_DOM" ||
        baseMnem === "SAN_PRM" ||
        baseMnem === "AUTH_VER" ||
        baseMnem === "SFIND";
      if (isSanitizer) {
        this.taintedOperands.delete(dst);
      } else {
        const anyArgTainted =
          mnem.endsWith("@UNTRUSTED") || parts.slice(3).some((op) => this.taintedOperands.has(op));
        if (anyArgTainted) {
          this.taintedOperands.add(dst);
          if (!mnem.endsWith("@UNTRUSTED") && !mnem.endsWith("@SAFE")) {
            line = [parts[0], parts[1], `${parts[2]}@UNTRUSTED`, ...parts.slice(3)].join(" ");
          }
        }
      }
    }
    if (this.emitAsm) {
      this.lines.push(line);
    } else {
      const pkt = encodeAsmInsPacketU16(line, this.retTyTok);
      this.packets.push(pkt);
      this.pktSpans.push([Math.max(1, this.curSpan[0]), Math.max(1, this.curSpan[1])]);
    }
  }

  public allocReg(line: number, col: number): string {
    if (this.nextReg >= 192) {
      throw new Error(`error[BUDGET_EXCEEDED] line ${line}, col ${col}: Surface function exceeded 192 SSA registers`);
    }
    const r = `%${this.nextReg}`;
    this.nextReg += 1;
    return r;
  }

  private allocLbl(line: number, col: number): number {
    if (this.nextLbl > 15) {
      throw new Error(`error[BUDGET_EXCEEDED] line ${line}, col ${col}: Surface function exceeded 15 basic-block labels`);
    }
    const l = this.nextLbl;
    this.nextLbl += 1;
    return l;
  }

  private evalToOperand(expr: SurfaceExpr): string {
    if (expr.kind === "Int") return expr.value.toString();
    if (expr.kind === "Trit") return expr.label;
    if (expr.kind === "Var") {
      const v = this.varMap.get(expr.name);
      if (!v) {
        throw new Error(`error[UNDEF_VAR] line ${expr.line}, col ${expr.col}: Undefined variable '${expr.name}'`);
      }
      return v;
    }
    return this.lowerExprInto(expr, null, false);
  }

  private evalHostArgToOperand(expr: SurfaceExpr): string {
    if (expr.kind === "StrLit") {
      const trimmed = expr.value.trim();
      if (/^-?\d+$/.test(trimmed)) return trimmed;
      return fnv1a32Positive(expr.value).toString();
    }
    return this.evalToOperand(expr);
  }

  private lowerTritVecPack5(
    trits: number[],
    targetReg: string | null,
    unpackToTvec: boolean,
    line: number,
    col: number,
  ): string {
    const packedBytes = pack5EncodeLocal(trits);
    const rBytes = !unpackToTvec && targetReg ? targetReg : this.allocReg(line, col);
    this.emit(`${rBytes} = ANEW 0`);
    for (let i = 0; i < packedBytes.length; i += 24) {
      const chunk = packedBytes.slice(i, i + 24).map((b) => b.toString());
      this.emit(`${rBytes} = ISET ${rBytes} ${chunk.join(" ")}`);
    }
    if (!unpackToTvec) return rBytes;
    const dst = targetReg ?? this.allocReg(line, col);
    this.emit(`${dst} = UNPACK5 ${rBytes} ${trits.length}`);
    return dst;
  }

  private lowerCallRhs(name: string, args: SurfaceExpr[], line: number, col: number): string {
    const userSig = this.fnSigs.get(name);
    if (userSig) {
      const [calleeId, expectedArity, calleeRetTy] = userSig;
      if (args.length !== expectedArity) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Function '${name}' expects ${expectedArity} arguments, got ${args.length}`,
        );
      }
      const opArgs = [`@fn${calleeId}`, ...args.map((a) => this.evalToOperand(a))];
      const callMnem =
        calleeRetTy === "TRIT"
          ? "CALL_TRIT"
          : calleeRetTy === "BYTES"
            ? "CALL_BYTES"
            : calleeRetTy === "TRIT_VEC"
              ? "CALL_TRIT_VEC"
              : calleeRetTy === "STR"
                ? "CALL_STR"
                : calleeRetTy === "STRUCT"
                  ? "CALL_STRUCT"
                  : "CALL_I64";
      return `${callMnem} ${opArgs.join(" ")}`;
    }

    const low = name.toLowerCase();
    if (low === "snew") {
      if (args.length % 2 !== 0) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'snew' expects 0 or even key/value arguments, got ${args.length}`,
        );
      }
      if (args.length === 0) return "SNEW 0";
      const sOps = args.map((a, idx) => (idx % 2 === 0 ? this.evalHostArgToOperand(a) : this.evalToOperand(a)));
      return `SNEW ${sOps.join(" ")}`;
    }
    if (low === "fget") {
      if (args.length !== 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'fget' expects 2 arguments (struct, field), got ${args.length}`,
        );
      }
      return `FGET ${this.evalToOperand(args[0])} ${this.evalHostArgToOperand(args[1])}`;
    }
    if (low === "fset" || low === "sset") {
      if (args.length !== 3) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic '${name}' expects 3 arguments (struct, field, value), got ${args.length}`,
        );
      }
      return `FSET ${this.evalToOperand(args[0])} ${this.evalHostArgToOperand(args[1])} ${this.evalToOperand(args[2])}`;
    }
    if (low === "sslice") {
      if (args.length !== 3) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'sslice' expects 3 arguments (str, start, end), got ${args.length}`,
        );
      }
      return `SSLICE ${this.evalToOperand(args[0])} ${this.evalToOperand(args[1])} ${this.evalToOperand(args[2])}`;
    }
    if (low === "req_param") {
      if (args.length === 0 || args.length > 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'req_param' expects 1 or 2 arguments, got ${args.length}`,
        );
      }
      const reqOp =
        args.length === 2
          ? this.evalHostArgToOperand(args[0])
          : this.varMap.get("req") ?? this.evalHostArgToOperand(args[0]);
      const keyOp = args.length === 2 ? this.evalHostArgToOperand(args[1]) : "0";
      return `REQ_PAR@SAFE ${reqOp} ${keyOp}`;
    }
    if (low === "db_get" || low === "kv_get") {
      if (args.length === 0 || args.length > 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic '${name}' expects 1 or 2 arguments, got ${args.length}`,
        );
      }
      const keyOp = this.evalHostArgToOperand(args[args.length - 1]);
      return `DB_GET ${keyOp}`;
    }
    if (low === "db_query") {
      if (args.length === 0 || args.length > 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'db_query' expects 1 or 2 arguments, got ${args.length}`,
        );
      }
      const qOp = this.evalHostArgToOperand(args[0]);
      const pOp = args.length === 2 ? this.evalHostArgToOperand(args[1]) : "0";
      return `DB_QRY ${qOp} ${pOp}`;
    }
    if (low === "db_count" || low === "kv_count") {
      const cOp = args.length > 0 ? this.evalHostArgToOperand(args[0]) : "0";
      return `DB_CNT ${cOp}`;
    }
    if (low === "kv_put" || low === "db_put") {
      if (args.length !== 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic '${name}' expects 2 arguments (key, value), got ${args.length}`,
        );
      }
      return `KV_PUT ${this.evalHostArgToOperand(args[0])} ${this.evalHostArgToOperand(args[1])}`;
    }
    if (low === "respond" || low === "respond_json" || low === "respond_html") {
      if (args.length === 0 || args.length > 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic '${name}' expects 1 or 2 arguments (status, body), got ${args.length}`,
        );
      }
      const mnem = low === "respond_html" ? "RSP_HTML" : "RSP_JSON";
      if (args.length === 2) {
        return `${mnem} ${this.evalHostArgToOperand(args[0])} ${this.evalHostArgToOperand(args[1])}`;
      }
      return `${mnem} 200 ${this.evalHostArgToOperand(args[0])}`;
    }
    if (low === "auth_verify") {
      if (args.length !== 1) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'auth_verify' expects 1 argument, got ${args.length}`,
        );
      }
      return `AUTH_VER ${this.evalHostArgToOperand(args[0])}`;
    }

    const opArgs = args.map((a) => this.evalToOperand(a));
    const unaryMap: Record<string, string> = {
      iabs: "IABS",
      abs: "IABS",
      ineg: "INEG",
      bnot: "BNOT",
      tnot: "TNOT",
      anew: "ANEW",
      slen: "SLEN",
      pack5_len: "SLEN",
      cint: "CINT",
      mov: "CINT",
      ctrit: "CTRIT",
      pack5: "PACK5",
      tpack5: "PACK5",
      egate: "EGATE",
      decision_hook: "EGATE",
      pack5_gate: "EGATE",
      tlogic: "TLOGIC",
      decide_trit: "TLOGIC",
      lookahead: "LOOKAHEAD",
      san_sql: "SAN_SQL",
      sanitize_sql: "SAN_SQL",
      san_dom: "SAN_DOM",
      sanitize_dom: "SAN_DOM",
      san_prm: "SAN_PRM",
      sanitize_prompt: "SAN_PRM",
      fnv1a: "FNV1A",
      hash_fnv1a: "FNV1A",
      arena: "ARENA",
      borrow_r: "BORROW_R",
      borrow_w: "BORROW_W",
      move_val: "MOVE",
      s2i: "S2I",
      str_to_int: "S2I",
      i2s: "I2S",
      int_to_str: "I2S",
      hex_enc: "HEX_ENC",
      hex_dec: "HEX_DEC",
    };
    if (unaryMap[low]) {
      if (opArgs.length !== 1) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic '${name}' expects 1 argument, got ${opArgs.length}`,
        );
      }
      return `${unaryMap[low]} ${opArgs[0]}`;
    }

    const binaryMap: Record<string, string> = {
      iadd: "IADD",
      isub: "ISUB",
      imul: "IMUL",
      idiv: "IDIV",
      imod: "IMOD",
      imin: "IMIN",
      min: "IMIN",
      imax: "IMAX",
      max: "IMAX",
      band: "BAND",
      bor: "BOR",
      bxor: "BXOR",
      shl: "SHL",
      shr: "SHR",
      ceq: "CEQ",
      cne: "CNE",
      clt: "CLT",
      cgt: "CGT",
      tand: "TAND",
      tor: "TOR",
      txor: "TXOR",
      tcmp: "TCMP",
      iget: "IGET",
      pack5_get: "IGET",
      unpack5: "UNPACK5",
      tunpack5: "UNPACK5",
      tdot: "TDOT",
      pack5_tdot: "TDOT",
      scat: "SCAT",
      sconc: "SCAT",
      sfind: "SFIND",
      ssplit: "SSPLIT",
    };
    if (binaryMap[low]) {
      if (opArgs.length !== 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic '${name}' expects 2 arguments, got ${opArgs.length}`,
        );
      }
      return `${binaryMap[low]} ${opArgs[0]} ${opArgs[1]}`;
    }

    if (low === "tmaj") {
      if (opArgs.length !== 3) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'tmaj' expects 3 arguments, got ${opArgs.length}`,
        );
      }
      return `TMAJ ${opArgs[0]} ${opArgs[1]} ${opArgs[2]}`;
    }
    if (low === "sel_trit" || low === "cmov") {
      if (opArgs.length !== 4) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic '${name}' expects 4 arguments, got ${opArgs.length}`,
        );
      }
      const m = low === "sel_trit" ? "SEL_TRIT" : "CMOV";
      return `${m} ${opArgs[0]} ${opArgs[1]} ${opArgs[2]} ${opArgs[3]}`;
    }
    if (low === "iset_i32") {
      if (opArgs.length !== 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'iset_i32' expects 2 arguments, got ${opArgs.length}`,
        );
      }
      return `ISET ${opArgs[0]} -4 ${opArgs[1]}`;
    }
    if (low === "iset_i64") {
      if (opArgs.length !== 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'iset_i64' expects 2 arguments, got ${opArgs.length}`,
        );
      }
      return `ISET ${opArgs[0]} -8 ${opArgs[1]}`;
    }
    if (low === "iset_patch_i64") {
      if (opArgs.length !== 3) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'iset_patch_i64' expects 3 arguments, got ${opArgs.length}`,
        );
      }
      return `ISET ${opArgs[0]} -9 ${opArgs[1]} ${opArgs[2]}`;
    }
    if (low === "iset") {
      if (opArgs.length < 2) {
        throw new Error(
          `error[ARITY_MISMATCH] line ${line}, col ${col}: Intrinsic 'iset' expects at least 2 arguments`,
        );
      }
      return `ISET ${opArgs.join(" ")}`;
    }

    throw new Error(`error[UNKNOWN_FN] line ${line}, col ${col}: Unknown surface intrinsic or function '${name}'`);
  }

  private lowerBlockIntoTarget(
    stmts: SurfaceStmt[],
    destReg: string,
    hasSharedJoin: boolean,
    line: number,
    col: number,
  ): void {
    if (stmts.length === 0) {
      throw new Error(`error[SYNTAX_ERROR] line ${line}, col ${col}: Empty branch block in control-flow expression`);
    }
    for (let i = 0; i < stmts.length - 1; i++) {
      this.lowerStmt(stmts[i]);
    }
    const last = stmts[stmts.length - 1];
    if (last.kind === "Expr" || last.kind === "Ret") {
      this.lowerExprInto(last.expr, destReg, hasSharedJoin);
    } else {
      this.lowerStmt(last);
    }
  }

  private lowerExprInto(expr: SurfaceExpr, targetReg: string | null, hasSharedJoin: boolean): string {
    this.curSpan = [expr.line, expr.col];
    switch (expr.kind) {
      case "Int": {
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        this.curSpan = [expr.line, expr.col];
        this.emit(`${dst} = CINT ${expr.value.toString()}`);
        return dst;
      }
      case "Trit": {
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        this.curSpan = [expr.line, expr.col];
        this.emit(`${dst} = CTRIT ${expr.label}`);
        return dst;
      }
      case "TritVecLit": {
        this.curSpan = [expr.line, expr.col];
        return this.lowerTritVecPack5(expr.trits, targetReg, true, expr.line, expr.col);
      }
      case "StrLit": {
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        const bytes = new TextEncoder().encode(expr.value);
        let hexStr = "";
        for (const b of bytes) hexStr += b.toString(16).padStart(2, "0");
        this.curSpan = [expr.line, expr.col];
        this.emit(`${dst} = CSTR@UNTRUSTED @str:${hexStr}`);
        return dst;
      }
      case "Var": {
        const src = this.varMap.get(expr.name);
        if (!src) {
          throw new Error(`error[UNDEF_VAR] line ${expr.line}, col ${expr.col}: Undefined variable '${expr.name}'`);
        }
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        this.curSpan = [expr.line, expr.col];
        this.emit(`${dst} = CINT ${src}`);
        return dst;
      }
      case "UnOp": {
        const opS = this.evalToOperand(expr.sub);
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        this.curSpan = [expr.line, expr.col];
        this.emit(`${dst} = ${expr.op} ${opS}`);
        return dst;
      }
      case "BinOp": {
        const lOp = this.evalToOperand(expr.lhs);
        const rOp = this.evalToOperand(expr.rhs);
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        this.curSpan = [expr.line, expr.col];
        this.emit(`${dst} = ${expr.op} ${lOp} ${rOp}`);
        return dst;
      }
      case "Call": {
        if (expr.name.toLowerCase() === "pack5_bytes") {
          if (expr.args.length !== 1) {
            throw new Error(
              `error[ARITY_MISMATCH] line ${expr.line}, col ${expr.col}: Intrinsic 'pack5_bytes' expects 1 trit-vector literal argument`,
            );
          }
          const a0 = expr.args[0];
          if (a0.kind === "TritVecLit") {
            this.curSpan = [a0.line, a0.col];
            return this.lowerTritVecPack5(a0.trits, targetReg, false, a0.line, a0.col);
          }
          throw new Error(
            `error[BAD_TYPE] line ${expr.line}, col ${expr.col}: Intrinsic 'pack5_bytes' requires a '[+1t, -1t, ...]' literal`,
          );
        }
        const rhs = this.lowerCallRhs(expr.name, expr.args, expr.line, expr.col);
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        this.curSpan = [expr.line, expr.col];
        this.emit(`${dst} = ${rhs}`);
        return dst;
      }
      case "IfBranch": {
        const condOp = this.evalToOperand(expr.cond);
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        const lThen = this.allocLbl(expr.line, expr.col);
        this.curSpan = [expr.line, expr.col];
        this.emit(`${expr.brMnem} ${condOp} ${lThen}`);
        const savedVars = new Map(this.varMap);
        this.lowerBlockIntoTarget(expr.elseBlk, dst, false, expr.line, expr.col);
        this.varMap = new Map(savedVars);
        const jmpPlaceholderIdx = this.emitAsm ? this.lines.length : this.packets.length;
        if (this.emitAsm) {
          this.lines.push("JMP __JOIN__");
        } else {
          this.packets.push([0x7f41, 0xffff, TOK_INS_END]);
          this.pktSpans.push([Math.max(1, expr.line), Math.max(1, expr.col)]);
        }
        this.curSpan = [expr.line, expr.col];
        this.emit(`LABEL ${lThen}`);
        this.lowerBlockIntoTarget(expr.thenBlk, dst, true, expr.line, expr.col);
        this.varMap = savedVars;
        if (!hasSharedJoin) {
          const lJoin = this.allocLbl(expr.line, expr.col);
          if (this.emitAsm) {
            const joinStr = lJoin.toString();
            for (let k = jmpPlaceholderIdx; k < this.lines.length; k++) {
              if (this.lines[k].includes("__JOIN__")) {
                this.lines[k] = this.lines[k].replace("__JOIN__", joinStr);
              }
            }
          } else {
            for (let k = jmpPlaceholderIdx; k < this.packets.length; k++) {
              const pkt = this.packets[k];
              if (pkt.length >= 2 && pkt[0] === 0x7f41 && pkt[1] === 0xffff) {
                pkt[1] = 0x7ef0 + lJoin;
              }
            }
          }
          this.curSpan = [expr.line, expr.col];
          this.emit(`LABEL ${lJoin}`);
        }
        return dst;
      }
      case "MatchTrit": {
        const condOp = this.evalToOperand(expr.cond);
        const dst = targetReg ?? this.allocReg(expr.line, expr.col);
        const lNeg = this.allocLbl(expr.line, expr.col);
        const lZero = this.allocLbl(expr.line, expr.col);
        const lPos = this.allocLbl(expr.line, expr.col);
        const lJoin = this.allocLbl(expr.line, expr.col);
        this.curSpan = [expr.line, expr.col];
        this.emit(`BR_TRIT ${condOp} ${lNeg} ${lZero} ${lPos}`);
        const savedVars = new Map(this.varMap);
        this.curSpan = [expr.line, expr.col];
        this.emit(`LABEL ${lNeg}`);
        this.lowerBlockIntoTarget(expr.negBlk, dst, false, expr.line, expr.col);
        this.varMap = new Map(savedVars);
        this.curSpan = [expr.line, expr.col];
        this.emit(`JMP ${lJoin}`);
        this.curSpan = [expr.line, expr.col];
        this.emit(`LABEL ${lZero}`);
        this.lowerBlockIntoTarget(expr.zeroBlk, dst, false, expr.line, expr.col);
        this.varMap = new Map(savedVars);
        this.curSpan = [expr.line, expr.col];
        this.emit(`JMP ${lJoin}`);
        this.curSpan = [expr.line, expr.col];
        this.emit(`LABEL ${lPos}`);
        this.lowerBlockIntoTarget(expr.posBlk, dst, false, expr.line, expr.col);
        this.varMap = savedVars;
        this.curSpan = [expr.line, expr.col];
        this.emit(`LABEL ${lJoin}`);
        return dst;
      }
    }
  }

  public lowerStmt(stmt: SurfaceStmt): void {
    this.curSpan = [stmt.line, stmt.col];
    switch (stmt.kind) {
      case "Let": {
        const reg = this.lowerExprInto(stmt.expr, null, false);
        this.varMap.set(stmt.name, reg);
        return;
      }
      case "Assign": {
        const reg = this.varMap.get(stmt.name);
        if (!reg) {
          throw new Error(
            `error[UNDEF_VAR] line ${stmt.line}, col ${stmt.col}: Cannot reassign undeclared variable '${stmt.name}'`,
          );
        }
        if (!reg.startsWith("%")) {
          throw new Error(
            `error[IMMUT_PARAM] line ${stmt.line}, col ${stmt.col}: Cannot reassign immutable function parameter '${stmt.name}'`,
          );
        }
        this.lowerExprInto(stmt.expr, reg, false);
        return;
      }
      case "DoWhile": {
        if (stmt.body.length === 0) {
          throw new Error(`error[SYNTAX_ERROR] line ${stmt.line}, col ${stmt.col}: Empty do_while loop body`);
        }
        const lTop = this.allocLbl(stmt.line, stmt.col);
        this.curSpan = [stmt.line, stmt.col];
        this.emit(`LABEL ${lTop}`);
        for (const s of stmt.body) this.lowerStmt(s);
        const condOp = this.evalToOperand(stmt.cond);
        this.curSpan = [stmt.line, stmt.col];
        this.emit(`BR_POS ${condOp} ${lTop}`);
        return;
      }
      case "While": {
        if (stmt.body.length === 0) {
          throw new Error(`error[SYNTAX_ERROR] line ${stmt.line}, col ${stmt.col}: Empty while loop body`);
        }
        const lTop = this.allocLbl(stmt.line, stmt.col);
        const lBody = this.allocLbl(stmt.line, stmt.col);
        const lExit = this.allocLbl(stmt.line, stmt.col);
        this.curSpan = [stmt.line, stmt.col];
        this.emit(`LABEL ${lTop}`);
        const condOp = this.evalToOperand(stmt.cond);
        this.curSpan = [stmt.line, stmt.col];
        this.emit(`BR_POS ${condOp} ${lBody}`);
        this.emit(`JMP ${lExit}`);
        this.emit(`LABEL ${lBody}`);
        for (const s of stmt.body) this.lowerStmt(s);
        this.curSpan = [stmt.line, stmt.col];
        this.emit(`JMP ${lTop}`);
        this.emit(`LABEL ${lExit}`);
        return;
      }
      case "Label":
        this.definedLabels.push(stmt.id);
        this.curSpan = [stmt.line, stmt.col];
        this.emit(`LABEL ${stmt.id}`);
        return;
      case "Goto":
        this.usedLabels.push([stmt.id, stmt.line, stmt.col]);
        this.curSpan = [stmt.line, stmt.col];
        this.emit(`JMP ${stmt.id}`);
        return;
      case "Ret": {
        const op = this.evalToOperand(stmt.expr);
        this.curSpan = [stmt.line, stmt.col];
        this.emit(`RET ${op}`);
        return;
      }
      case "Expr": {
        if (stmt.expr.kind === "Call") {
          const call = stmt.expr;
          const low = call.name.toLowerCase();
          if (low === "br_pos" || low === "br_neg") {
            if (call.args.length !== 2) {
              throw new Error(
                `error[ARITY_MISMATCH] line ${call.line}, col ${call.col}: '${call.name}' expects 2 arguments`,
              );
            }
            const cOp = this.evalToOperand(call.args[0]);
            const targetArg = call.args[1];
            if (targetArg.kind !== "Int") {
              throw new Error(
                `error[UNRESOLVED_LABEL] line ${call.line}, col ${call.col}: '${call.name}' target must be an integer label id`,
              );
            }
            if (targetArg.value < 0n || targetArg.value > 15n) {
              throw new Error(
                `error[UNRESOLVED_LABEL] line ${targetArg.line}, col ${targetArg.col}: Branch label ${targetArg.value} out of range 0..=15`,
              );
            }
            const lid = Number(targetArg.value);
            this.usedLabels.push([lid, targetArg.line, targetArg.col]);
            this.curSpan = [call.line, call.col];
            this.emit(`${low.toUpperCase()} ${cOp} ${lid}`);
            return;
          }
          if (low === "br_trit") {
            if (call.args.length !== 4) {
              throw new Error(
                `error[ARITY_MISMATCH] line ${call.line}, col ${call.col}: 'br_trit' expects 4 arguments`,
              );
            }
            const cOp = this.evalToOperand(call.args[0]);
            const lids: string[] = [];
            for (const a of call.args.slice(1)) {
              if (a.kind !== "Int") {
                throw new Error(
                  `error[UNRESOLVED_LABEL] line ${call.line}, col ${call.col}: 'br_trit' targets must be integer label ids`,
                );
              }
              if (a.value < 0n || a.value > 15n) {
                throw new Error(
                  `error[UNRESOLVED_LABEL] line ${a.line}, col ${a.col}: Branch label ${a.value} out of range 0..=15`,
                );
              }
              const lid = Number(a.value);
              this.usedLabels.push([lid, a.line, a.col]);
              lids.push(lid.toString());
            }
            this.curSpan = [call.line, call.col];
            this.emit(`BR_TRIT ${cOp} ${lids.join(" ")}`);
            return;
          }
          if (low === "drop") {
            const ops = call.args.map((a) => this.evalToOperand(a));
            this.curSpan = [call.line, call.col];
            this.emit(`DROP ${ops.join(" ")}`);
            return;
          }
          if (low === "respond" || low === "respond_json" || low === "respond_html") {
            const op = this.lowerExprInto(stmt.expr, null, false);
            this.curSpan = [call.line, call.col];
            this.emit(`RET ${op}`);
            return;
          }
        }
        this.lowerExprInto(stmt.expr, null, false);
      }
    }
  }

  public checkUnresolvedLabels(): void {
    for (const [lid, line, col] of this.usedLabels) {
      if (!this.definedLabels.includes(lid)) {
        throw new Error(
          `error[UNRESOLVED_LABEL] line ${line}, col ${col}: Unresolved branch target label ${lid}`,
        );
      }
    }
  }
}

export interface SurfaceSourceMap {
  offsetToSpan: Map<number, [number, number]>;
  budgetSpan: [number, number];
  pack5Span: [number, number] | null;
  defaultSpan: [number, number];
}

function createDefaultSourceMap(): SurfaceSourceMap {
  return {
    offsetToSpan: new Map(),
    budgetSpan: [1, 1],
    pack5Span: null,
    defaultSpan: [1, 1],
  };
}

export function annotateOmniErrorWithSourcemap(err: string, sm: SurfaceSourceMap): string {
  if (err.includes("line ") && err.includes("col ")) return err;
  let span = sm.defaultSpan;
  const offMatch = err.match(/Offset (\d+)/);
  if (offMatch) {
    const off = Number.parseInt(offMatch[1], 10);
    const mapped = sm.offsetToSpan.get(off);
    if (mapped) span = mapped;
  } else if (err.includes("BUDGET_EXCEEDED") || err.includes("budget exceeded")) {
    span = sm.budgetSpan;
  } else if (err.includes("CORRUPT_PACK5") || err.includes("PACK5_SENTINEL_OR_CORRUPT_BYTE")) {
    span = sm.pack5Span ?? sm.defaultSpan;
  }
  const line = Math.max(1, span[0]);
  const col = Math.max(1, span[1]);
  const bIdx = err.indexOf("error[");
  if (bIdx !== -1) {
    const endB = err.indexOf("]", bIdx);
    if (endB !== -1) {
      return `${err.slice(0, endB + 1)} line ${line}, col ${col}${err.slice(endB + 1)}`;
    }
  }
  return `line ${line}, col ${col}: ${err}`;
}

function compileOmniSurfaceToAsmInner(src: string, strict: boolean): string {
  const effectiveStrict = strict || src.includes("#[strict]");
  const tokens = tokenizeOmniSurfaceInner(src, effectiveStrict);
  const parser = new SurfaceParser(tokens, effectiveStrict);
  const { includes, fns } = parser.parseModule();
  if (includes.length > 0) {
    const expanded = resolveOmniSurfaceIncludes(src);
    return compileOmniSurfaceToAsmInner(expanded, effectiveStrict);
  }

  const multiFn = fns.length > 1;
  const fnSigs = new Map<string, [number, number, string]>();
  const assignedIds: number[] = [];

  for (let idx = 0; idx < fns.length; idx++) {
    const f = fns[idx];
    const fid = f.fnIdAttr !== null ? f.fnIdAttr : !multiFn || idx + 1 === fns.length ? 0 : idx + 1;
    const fnStrict = effectiveStrict || f.strictAttr;
    const retAsm = mapOmniSurfaceTypeInner(f.retTy, f.retLine, f.retCol, fnStrict);
    for (const p of f.params) {
      mapOmniSurfaceTypeInner(p.ty, p.line, p.col, fnStrict);
    }
    if (fnSigs.has(f.name)) {
      throw new Error(`error[SYNTAX_ERROR] line ${f.retLine}, col ${f.retCol}: Duplicate function definition '${f.name}'`);
    }
    fnSigs.set(f.name, [fid, f.params.length, retAsm]);
    assignedIds.push(fid);
  }

  const out: string[] = [];
  for (let idx = 0; idx < fns.length; idx++) {
    const f = fns[idx];
    const fid = assignedIds[idx];
    const fnStrict = effectiveStrict || f.strictAttr;
    if (multiFn) out.push(`FN_DEF ${fid}`);
    else if (f.fnIdAttr !== null) out.push(`FN_ID ${f.fnIdAttr}`);
    if (f.capMask !== 0) out.push(`CAP_MASK ${f.capMask}`);
    for (let pIdx = 0; pIdx < f.params.length; pIdx++) {
      const p = f.params[pIdx];
      const tyStr = mapOmniSurfaceTypeInner(p.ty, p.line, p.col, fnStrict);
      const taintStr = p.untrusted ? "UNTRUSTED" : "SAFE";
      out.push(`PARAM $${pIdx} ${tyStr} ${taintStr}`);
    }
    const retTyStr = mapOmniSurfaceTypeInner(f.retTy, f.retLine, f.retCol, fnStrict);
    const retTyTok = parseAsmTyTok(retTyStr);
    out.push(`RET_TY ${retTyStr}`);

    const ctx = new SurfaceLowerCtx(f.params, fnSigs, retTyTok, true);
    if (f.isRoute) {
      const mCode =
        f.routeMethod === "GET"
          ? 1
          : f.routeMethod === "POST"
            ? 2
            : f.routeMethod === "PUT"
              ? 3
              : f.routeMethod === "DELETE"
                ? 4
                : 0;
      const pathHash = fnv1a32Positive(f.routePath ?? "/");
      const rRt = ctx.allocReg(f.retLine, f.retCol);
      ctx.emit(`${rRt} = ROUTE ${mCode} ${pathHash}`);
    }
    for (const stmt of f.body) ctx.lowerStmt(stmt);
    ctx.checkUnresolvedLabels();
    out.push(...ctx.lines);
  }

  return out.join("\n") + "\n";
}

export function compileOmniSurfaceToAsm(src: string): string {
  return compileOmniSurfaceToAsmInner(src, false);
}

export function compileOmniSurfaceToU16WithSourcemap(
  src: string,
  strict = false,
): { stream: number[]; sourcemap: SurfaceSourceMap } {
  const effectiveStrict = strict || src.includes("#[strict]");
  const tokens = tokenizeOmniSurfaceInner(src, effectiveStrict);
  const parser = new SurfaceParser(tokens, effectiveStrict);
  const { includes, fns } = parser.parseModule();
  if (includes.length > 0) {
    const expanded = resolveOmniSurfaceIncludes(src);
    return compileOmniSurfaceToU16WithSourcemap(expanded, effectiveStrict);
  }

  const multiFn = fns.length > 1;
  const fnSigs = new Map<string, [number, number, string]>();
  const assignedIds: number[] = [];

  for (let idx = 0; idx < fns.length; idx++) {
    const f = fns[idx];
    const fid = f.fnIdAttr !== null ? f.fnIdAttr : !multiFn || idx + 1 === fns.length ? 0 : idx + 1;
    const fnStrict = effectiveStrict || f.strictAttr;
    const retAsm = mapOmniSurfaceTypeInner(f.retTy, f.retLine, f.retCol, fnStrict);
    for (const p of f.params) {
      mapOmniSurfaceTypeInner(p.ty, p.line, p.col, fnStrict);
    }
    if (fnSigs.has(f.name)) {
      throw new Error(`error[SYNTAX_ERROR] line ${f.retLine}, col ${f.retCol}: Duplicate function definition '${f.name}'`);
    }
    fnSigs.set(f.name, [fid, f.params.length, retAsm]);
    assignedIds.push(fid);
  }

  const sm = createDefaultSourceMap();
  if (fns.length > 0) {
    sm.defaultSpan = [Math.max(1, fns[0].fnLine), Math.max(1, fns[0].fnCol)];
    sm.budgetSpan = sm.defaultSpan;
  }

  const completedFns: Array<{
    fnId: number;
    params: Array<[number, number, number]>;
    retTy: number;
    capMask: CapMask;
    packets: number[][];
    spans: Array<[number, number]>;
  }> = [];

  for (let idx = 0; idx < fns.length; idx++) {
    const f = fns[idx];
    if (f.budgetSpan) {
      sm.budgetSpan = [Math.max(1, f.budgetSpan[0]), Math.max(1, f.budgetSpan[1])];
    } else {
      for (const st of f.body) {
        if (st.kind === "While" || st.kind === "DoWhile") {
          sm.budgetSpan = [Math.max(1, st.line), Math.max(1, st.col)];
          break;
        }
      }
    }
    const fid = assignedIds[idx];
    const fnStrict = effectiveStrict || f.strictAttr;
    const params: Array<[number, number, number]> = f.params.map((p, pIdx) => {
      const tyStr = mapOmniSurfaceTypeInner(p.ty, p.line, p.col, fnStrict);
      const tyTok = parseAsmTyTok(tyStr);
      return [PAR_BASE + pIdx, tyTok, packTc(p.untrusted, CAP_ID_PURE)];
    });
    const retTyStr = mapOmniSurfaceTypeInner(f.retTy, f.retLine, f.retCol, fnStrict);
    const retTyTok = parseAsmTyTok(retTyStr);

    const ctx = new SurfaceLowerCtx(f.params, fnSigs, retTyTok, false);
    ctx.curSpan = [Math.max(1, f.fnLine), Math.max(1, f.fnCol)];
    if (f.isRoute) {
      const mCode =
        f.routeMethod === "GET"
          ? 1
          : f.routeMethod === "POST"
            ? 2
            : f.routeMethod === "PUT"
              ? 3
              : f.routeMethod === "DELETE"
                ? 4
                : 0;
      const pathHash = fnv1a32Positive(f.routePath ?? "/");
      const rRt = ctx.allocReg(f.fnLine, f.fnCol);
      ctx.curSpan = [Math.max(1, f.fnLine), Math.max(1, f.fnCol)];
      ctx.emit(`${rRt} = ROUTE ${mCode} ${pathHash}`);
    }
    for (const stmt of f.body) ctx.lowerStmt(stmt);
    ctx.checkUnresolvedLabels();
    completedFns.push({
      fnId: fid,
      params,
      retTy: retTyTok,
      capMask: f.capMask,
      packets: ctx.packets,
      spans: ctx.pktSpans,
    });
  }

  const isPack5Op = (op: number | undefined): boolean =>
    op === 0x7f30 || op === 0x7f31 || op === 0x7f34 || op === 0x7f35 || op === 0x7f3f;

  if (completedFns.length === 1) {
    const f = completedFns[0];
    let curOff = 7 + 3 * f.params.length;
    for (let i = 0; i < f.packets.length; i++) {
      const pkt = f.packets[i];
      const sp = f.spans[i];
      sm.offsetToSpan.set(curOff, sp);
      if (sm.pack5Span === null && isPack5Op(pkt[1])) {
        sm.pack5Span = sp;
      }
      curOff += pkt.length;
    }
    const stream = packFnModule([], f.fnId, f.params, f.retTy, f.capMask, f.packets);
    return { stream, sourcemap: sm };
  } else {
    const stream: number[] = [TOK_MOD_BEGIN];
    for (const f of completedFns) {
      stream.push(TOK_FN_DEF, f.fnId, f.params.length);
      for (const [pt, pty, ptc] of f.params) stream.push(pt, pty, ptc);
      stream.push(f.retTy, f.capMask, f.packets.length);
      for (let i = 0; i < f.packets.length; i++) {
        const pkt = f.packets[i];
        const sp = f.spans[i];
        sm.offsetToSpan.set(stream.length, sp);
        if (sm.pack5Span === null && isPack5Op(pkt[1])) {
          sm.pack5Span = sp;
        }
        for (const w of pkt) stream.push(w);
      }
    }
    stream.push(TOK_MOD_END);
    return { stream, sourcemap: sm };
  }
}

export function compileOmniSurfaceToU16(src: string): number[] {
  return compileOmniSurfaceToU16WithSourcemap(src, false).stream;
}

export function compileOmniSurfaceStrictToU16(src: string): number[] {
  return compileOmniSurfaceToU16WithSourcemap(src, true).stream;
}

export function extractOmniSurfaceBudgets(src: string): {
  stepBudget: number | null;
  memBudget: number | null;
} {
  try {
    const tokens = tokenizeOmniSurfaceInner(src, false);
    const parser = new SurfaceParser(tokens, false);
    const { includes, fns } = parser.parseModule();
    if (includes.length > 0) {
      const expanded = resolveOmniSurfaceIncludes(src);
      return extractOmniSurfaceBudgets(expanded);
    }
    let stepBudget: number | null = null;
    let memBudget: number | null = null;
    for (const f of fns) {
      if (f.stepBudgetAttr !== null) {
        stepBudget = stepBudget === null ? f.stepBudgetAttr : Math.min(stepBudget, f.stepBudgetAttr);
      }
      if (f.memBudgetAttr !== null) {
        memBudget = memBudget === null ? f.memBudgetAttr : Math.min(memBudget, f.memBudgetAttr);
      }
    }
    return { stepBudget, memBudget };
  } catch {
    return { stepBudget: null, memBudget: null };
  }
}

export function parseCliOmniValue(aStr: string): OmniValue {
  const s = aStr.trim();
  if (s === "+1t" || s === "TRIT_POS") return { kind: "Trit", value: 1 };
  if (s === "0t" || s === "TRIT_ZERO") return { kind: "Trit", value: 0 };
  if (s === "-1t" || s === "TRIT_NEG") return { kind: "Trit", value: -1 };
  if (/^-?\d+$/.test(s)) {
    return { kind: "Int", value: BigInt.asIntN(64, BigInt(s)) };
  }
  return { kind: "Str", value: aStr };
}

export function formatOmniValueCompact(v: OmniValue): string {
  switch (v.kind) {
    case "Int":
      return v.value.toString();
    case "Trit":
      return v.value > 0 ? "+1t" : v.value < 0 ? "-1t" : "0t";
    case "Str":
      return v.value;
    case "Bytes":
      return `bytes[${v.value.length}]`;
    case "TritVec":
      return `trit_vec[${v.value.length}]`;
    case "U16Array":
      return `u16[${v.value.length}]`;
    case "Struct": {
      const st = v.fields.get(0);
      const body = v.fields.get(1);
      if (st?.kind === "Int" && body) {
        return `HTTP ${st.value.toString()} ${formatOmniValueCompact(body)}`;
      }
      return JSON.stringify(Object.fromEntries(Array.from(v.fields.entries()).map(([k, val]) => [k, formatOmniValueCompact(val)])));
    }
    case "Arena":
      return `arena(${v.capacity}, active=${v.active})`;
    case "Void":
      return "void";
    default:
      return JSON.stringify(v);
  }
}

export function wrapOmniSurfaceSnippet(snippet: string): string {
  const tCheck = snippet.trimStart();
  if (
    snippet.includes("fn ") ||
    tCheck.startsWith("route ") ||
    snippet.includes("\nroute ") ||
    snippet.includes(" route ")
  ) {
    return snippet;
  }
  let trimmed = snippet.trim();
  while (trimmed.endsWith(";")) trimmed = trimmed.slice(0, -1).trimEnd();
  let inStr = false;
  let braceDepth = 0;
  let parenDepth = 0;
  let bracketDepth = 0;
  let stmtStart = 0;
  for (let i = 0; i < trimmed.length; i++) {
    const ch = trimmed[i];
    if (inStr) {
      if (ch === "\\" && i + 1 < trimmed.length) {
        i += 1;
        continue;
      }
      if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") braceDepth += 1;
    else if (ch === "}") {
      braceDepth = Math.max(0, braceDepth - 1);
      if (braceDepth === 0 && parenDepth === 0 && bracketDepth === 0) {
        const curStmt = trimmed.slice(stmtStart, i + 1).trimStart();
        if (curStmt.startsWith("while ") || curStmt.startsWith("while(")) {
          stmtStart = i + 1;
        }
      }
    } else if (ch === "(") parenDepth += 1;
    else if (ch === ")") parenDepth = Math.max(0, parenDepth - 1);
    else if (ch === "[") bracketDepth += 1;
    else if (ch === "]") bracketDepth = Math.max(0, bracketDepth - 1);
    else if (ch === ";" && braceDepth === 0 && parenDepth === 0 && bracketDepth === 0) {
      stmtStart = i + 1;
    }
  }

  const prefix = trimmed.slice(0, stmtStart).trim();
  const rawTail = trimmed.slice(stmtStart).trim();
  const tail = rawTail.startsWith("return ")
    ? rawTail.slice("return ".length).trim()
    : rawTail === ""
      ? "0"
      : rawTail;

  const inferExprTy = (expr: string): string => {
    const e = expr.trim();
    if (
      e.startsWith('"') ||
      e.startsWith("san_sql(") ||
      e.startsWith("san_dom(") ||
      e.startsWith("san_prm(") ||
      e.startsWith("scat(") ||
      e.startsWith("sconc(") ||
      e.startsWith("i2s(")
    ) {
      return "str";
    }
    if (e.startsWith("[") || e.startsWith("unpack5(")) return "trit_vec";
    if (e.startsWith("pack5_bytes(") || e.startsWith("anew(")) return "bytes";
    if (
      e.includes("<=>") ||
      e.includes("==") ||
      e.includes("!=") ||
      e.includes("<=") ||
      e.includes(">=") ||
      e.includes(" < ") ||
      e.includes(" > ") ||
      e.startsWith("egate(") ||
      e.startsWith("decision_hook(") ||
      e.startsWith("pack5_gate(") ||
      e.startsWith("tlogic(") ||
      e.startsWith("tmaj(") ||
      e.startsWith("tand(") ||
      e.startsWith("tor(") ||
      e.startsWith("txor(") ||
      e.startsWith("tcmp(") ||
      e === "+1t" ||
      e === "0t" ||
      e === "-1t"
    ) {
      return "trit";
    }
    return "i64";
  };

  const retTy = inferExprTy(tail);
  if (!prefix) {
    return `#[cap(AI, DB_R, DB_W, NET, DOM, VISION, AUDIO, AUTH, FS, SYS)] fn main() -> ${retTy} { let __res = ${tail}; return __res; }`;
  }
  return `#[cap(AI, DB_R, DB_W, NET, DOM, VISION, AUDIO, AUTH, FS, SYS)] fn main() -> ${retTy} { ${prefix} let __res = ${tail}; return __res; }`;
}

// ==============================================================================
// PURE-OMNI VERIFIER KERNEL & FRONTEND EXECUTION
// ==============================================================================

export const PHASE7_PURE_OMNI_VERIFIER_KERNEL_SURFACE = `#[strict]
fn omni_pure_verify_header(missing_cap_bits: i64, taint_violation_kind: i64, arena_balance: i64) -> i64 {
    let cap_cmp: trit = missing_cap_bits <=> 0;
    let taint_cmp: trit = taint_violation_kind <=> 1;
    let taint_any: trit = taint_violation_kind <=> 0;
    let arena_cmp: trit = arena_balance <=> 0;
    let taint_code: i64 = match_trit taint_cmp {
        neg => 0,
        zero => 1001,
        pos => 1002
    };
    let arena_code: i64 = match_trit arena_cmp {
        neg => 1005,
        zero => 0,
        pos => 1004
    };
    let after_taint: i64 = if_pos taint_any { taint_code } else { arena_code };
    let final_code: i64 = if_pos cap_cmp { 1003 } else { after_taint };
    return final_code;
}`;

let cachedVerifierKernelStream: number[] | null = null;

export function buildPhase7PureOmniVerifierKernelStream(): number[] {
  if (cachedVerifierKernelStream) return cachedVerifierKernelStream;
  const u16Stream = compileOmniSurfaceStrictToU16(PHASE7_PURE_OMNI_VERIFIER_KERNEL_SURFACE);
  const decoded = decodeOmniTokenStream(u16Stream);
  for (const f of decoded.functions) {
    const rep = verifyBinaryFunction(f);
    if (!rep.passed) {
      throw new Error(`Pure-Omni verifier kernel failed binary verification: ${JSON.stringify(rep.errors)}`);
    }
  }
  cachedVerifierKernelStream = u16Stream;
  return u16Stream;
}

export function verifyModuleViaPhase7PureOmniKernel(u16Stream: number[]): void {
  const kernelU16 = buildPhase7PureOmniVerifierKernelStream();
  const kernelMod = decodeOmniTokenStream(kernelU16);
  const targetMod = decodeOmniTokenStream(u16Stream);

  for (const f of targetMod.functions) {
    let missingCapBits = 0n;
    let taintViolationKind = 0n;
    let arenaBalance = 0n;
    let activeBorrowCount = 0;
    const regTainted = new Set<number>();
    for (const [pTok, , pTc] of f.params) {
      if (unpackTc(pTc).untrusted) regTainted.add(pTok);
    }

    for (const ins of f.instructions) {
      const spec = decodeOpcodeToken(ins.opTok);
      if (spec) {
        const req = spec.reqCapMask;
        if ((f.capMask & req) !== req) {
          missingCapBits |= BigInt(req & ~f.capMask);
        }
        const anyOpTainted = ins.operands.some(
          (op) => (op.kind === "Reg" || op.kind === "Param") && regTainted.has(op.value),
        );
        if (ins.opTok === 0x7f4a && anyOpTainted && taintViolationKind === 0n) {
          taintViolationKind = 1n;
        } else if (spec.sinkRequiresSafe && anyOpTainted && taintViolationKind === 0n) {
          taintViolationKind = 2n;
        }
        if (ins.opTok === 0x7f01) {
          arenaBalance += 1n;
        } else if (ins.opTok === 0x7f02 || ins.opTok === 0x7f03) {
          activeBorrowCount += 1;
        } else if (ins.opTok === 0x7f05) {
          const op0 = ins.operands[0];
          if (op0) {
            if (activeBorrowCount > 0 && op0.kind === "Reg" && op0.value === REG_BASE) {
              arenaBalance = -1n;
            } else if (activeBorrowCount > 0) {
              activeBorrowCount -= 1;
            } else {
              arenaBalance -= 1n;
            }
          }
        }
        if (ins.destReg !== null) {
          const isSan =
            ins.opTok === 0x7f53 ||
            ins.opTok === 0x7f57 ||
            ins.opTok === 0x7f58 ||
            ins.opTok === 0x7f59 ||
            ins.opTok === 0x7f70;
          if (isSan) {
            regTainted.delete(ins.destReg);
          } else if (unpackTc(ins.tcTok).untrusted || anyOpTainted) {
            regTainted.add(ins.destReg);
          }
        }
      }
    }

    const vm = new OmniBinaryVm();
    const res = vm.executeFunction(kernelMod, 0, [
      { kind: "Int", value: missingCapBits },
      { kind: "Int", value: taintViolationKind },
      { kind: "Int", value: arenaBalance },
    ]);
    if (res.kind !== "Int") {
      throw new Error(`Unexpected Pure-Omni verifier kernel return: ${JSON.stringify(res)}`);
    }
    switch (res.value) {
      case 0n:
        break;
      case 1001n:
        throw new Error(
          "error[TAINT_ON_RET] line 1, col 1: Pure-Omni verifier kernel rejected @UNTRUSTED return value",
        );
      case 1002n:
        throw new Error(
          "error[TAINT_SINK] line 1, col 1: Pure-Omni verifier kernel rejected @UNTRUSTED operand at sink",
        );
      case 1003n:
        throw new Error(
          "error[MISSING_CAP] line 1, col 1: Pure-Omni verifier kernel rejected missing capability bitmask",
        );
      case 1004n:
        throw new Error(
          "error[LINEAR_LEAK] line 1, col 1: Pure-Omni verifier kernel rejected un-dropped linear arena",
        );
      case 1005n:
        throw new Error(
          "error[BORROW_CONFLICT] line 1, col 1: Pure-Omni verifier kernel rejected arena drop while borrow is active",
        );
      default:
        throw new Error(`Unknown Pure-Omni verifier kernel status code: ${res.value}`);
    }
  }
}

// ==============================================================================
// EXECUTION PIPELINE (AOT-FIRST & BUDGETED VM)
// ==============================================================================

export interface OmniExecutionDetail {
  value: OmniValue;
  formattedResult: string;
  backend: "aot" | "vm";
  u16Stream: number[];
  asmText: string;
  wasmBytes: Uint8Array | null;
  wasmValid: boolean;
  elfBytes: Uint8Array | null;
  elfFnv1a64: string | null;
  inspectText: string;
}

export function executeOmniSurfaceProgramWithBudgets(
  src: string,
  args: OmniValue[],
  stepBudget: number | null = null,
  memBudget: number | null = null,
): OmniValue {
  const wrapped = wrapOmniSurfaceSnippet(src);
  const { stepBudget: attrSteps, memBudget: attrMem } = extractOmniSurfaceBudgets(wrapped);
  const effSteps =
    stepBudget !== null && attrSteps !== null
      ? Math.min(stepBudget, attrSteps)
      : stepBudget ?? attrSteps;
  const effMem =
    memBudget !== null && attrMem !== null
      ? Math.min(memBudget, attrMem)
      : memBudget ?? attrMem;

  const { stream, sourcemap } = compileOmniSurfaceToU16WithSourcemap(wrapped, false);
  try {
    for (const rep of verifyU16Stream(stream)) {
      if (!rep.passed) {
        const first = rep.errors[0] ?? "";
        const annotated = annotateOmniErrorWithSourcemap(first, sourcemap);
        throw new Error(`${annotated} (fn #${rep.fnIdx}: ${JSON.stringify(rep.errors)})`);
      }
    }
    verifyModuleViaPhase7PureOmniKernel(stream);
  } catch (e) {
    throw new Error(annotateOmniErrorWithSourcemap(errMsg(e), sourcemap));
  }

  const mod = decodeOmniTokenStream(stream);
  const entryFnIdx = mod.functions.some((f) => f.fnIdx === 0)
    ? 0
    : mod.functions[mod.functions.length - 1]?.fnIdx;
  if (entryFnIdx === undefined) throw new Error("No functions in module");

  const vm = new OmniBinaryVm();
  if (effSteps !== null) vm.stepLimit = effSteps;
  if (effMem !== null) vm.memoryBudgetBytes = effMem;
  try {
    return vm.executeFunction(mod, entryFnIdx, args);
  } catch (e) {
    throw new Error(annotateOmniErrorWithSourcemap(errMsg(e), sourcemap));
  }
}

export function executeOmniSurfaceAotFirst(
  src: string,
  args: OmniValue[],
  strict = false,
): OmniExecutionDetail {
  const wrapped = strict ? src : wrapOmniSurfaceSnippet(src);
  const { stepBudget: attrSteps, memBudget: attrMem } = extractOmniSurfaceBudgets(wrapped);
  const { stream, sourcemap } = compileOmniSurfaceToU16WithSourcemap(
    wrapped,
    strict || wrapped.includes("#[strict]"),
  );

  try {
    for (const rep of verifyU16Stream(stream)) {
      if (!rep.passed) {
        const first = rep.errors[0] ?? "";
        const annotated = annotateOmniErrorWithSourcemap(first, sourcemap);
        throw new Error(`${annotated} (fn #${rep.fnIdx}: ${JSON.stringify(rep.errors)})`);
      }
    }
    verifyModuleViaPhase7PureOmniKernel(stream);
  } catch (e) {
    const msg = errMsg(e);
    if (msg.includes("line ") && msg.includes("col ")) throw new Error(msg);
    throw new Error(annotateOmniErrorWithSourcemap(msg, sourcemap));
  }

  const mod = decodeOmniTokenStream(stream);
  for (const func of mod.functions) {
    const rep = verifyBinaryFunction(func);
    if (!rep.passed) {
      const first = rep.errors[0] ?? "";
      const annotated = annotateOmniErrorWithSourcemap(first, sourcemap);
      throw new Error(`${annotated} (fn #${func.fnIdx}: ${JSON.stringify(rep.errors)})`);
    }
  }

  const entryFnIdx = mod.functions.some((f) => f.fnIdx === 0)
    ? 0
    : mod.functions[mod.functions.length - 1]?.fnIdx;
  if (entryFnIdx === undefined) throw new Error("No functions in module");
  const entryFn = mod.functions.find((f) => f.fnIdx === entryFnIdx);
  const entryRetTyTok = entryFn?.retTyTok ?? TY_I64;

  const allScalarArgs = args.every((a) => a.kind === "Int" || a.kind === "Trit");
  const scalarRet = entryRetTyTok === TY_I64 || entryRetTyTok === TY_TRIT;

  const asmText = compileOmniSurfaceToAsmInner(wrapped, strict || wrapped.includes("#[strict]"));
  const inspectText = formatOmniInspect(stream, sourcemap);

  if (isModuleAotHotPath(mod) && allScalarArgs && scalarRet) {
    const stepB = attrSteps ?? DEFAULT_WASM_STEP_LIMIT;
    const memB = attrMem ?? DEFAULT_WASM_MEMORY_BUDGET_BYTES;
    let staticAllocBytes = 0;
    for (const func of mod.functions) {
      for (const ins of func.instructions) {
        if (ins.opTok === 0x7f01 && ins.operands[0]?.kind === "ImmInt") {
          const cap = Number(ins.operands[0].value);
          if (cap > 0) staticAllocBytes += cap;
        }
      }
    }
    if (staticAllocBytes > memB) {
      throw new Error(
        annotateOmniErrorWithSourcemap(
          `error[BUDGET_EXCEEDED]: AOT memory budget exceeded (${staticAllocBytes} > ${memB} bytes)`,
          sourcemap,
        ),
      );
    }

    const i64Args = args.map((a) => (a.kind === "Int" ? a.value : a.kind === "Trit" ? BigInt(a.value) : 0n));
    let wasm: Uint8Array;
    let elfBytes: Uint8Array | null = null;
    let elfFnv1a64: string | null = null;
    let rawI64: bigint;
    try {
      wasm = compilePureU16ToWasm(stream);
      rawI64 = executeWasmModuleI64WithBudgets(wasm, i64Args, stepB, memB);
      elfBytes = compilePureU16ToX86_64ElfWithArgs(
        stream,
        i64Args[0] ?? 0n,
        i64Args[1] ?? 0n,
        i64Args[2] ?? 0n,
      );
      elfFnv1a64 = formatFnv1a64Hex(elfBytes);
    } catch (e) {
      throw new Error(annotateOmniErrorWithSourcemap(errMsg(e), sourcemap));
    }

    const outVal: OmniValue =
      entryRetTyTok === TY_TRIT
        ? { kind: "Trit", value: rawI64 > 0n ? 1 : rawI64 < 0n ? -1 : 0 }
        : { kind: "Int", value: rawI64 };
    const wasmValid =
      typeof WebAssembly !== "undefined" && typeof WebAssembly.validate === "function"
        ? WebAssembly.validate(wasm as BufferSource)
        : true;
    return {
      value: outVal,
      formattedResult: formatOmniValueCompact(outVal),
      backend: "aot",
      u16Stream: stream,
      asmText,
      wasmBytes: wasm,
      wasmValid,
      elfBytes,
      elfFnv1a64,
      inspectText,
    };
  }

  const vm = new OmniBinaryVm();
  if (attrSteps !== null) vm.stepLimit = attrSteps;
  if (attrMem !== null) vm.memoryBudgetBytes = attrMem;
  let val: OmniValue;
  try {
    val = vm.executeFunction(mod, entryFnIdx, args);
  } catch (e) {
    throw new Error(annotateOmniErrorWithSourcemap(errMsg(e), sourcemap));
  }

  let multiFnWasm: Uint8Array | null = null;
  let multiFnWasmValid = false;
  if (isModuleMultiFnWasmCompatible(mod)) {
    try {
      multiFnWasm = compilePureU16ToWasm(stream);
      multiFnWasmValid =
        typeof WebAssembly !== "undefined" && typeof WebAssembly.validate === "function"
          ? WebAssembly.validate(multiFnWasm as BufferSource)
          : true;
    } catch {
      multiFnWasm = null;
      multiFnWasmValid = false;
    }
  }

  return {
    value: val,
    formattedResult: formatOmniValueCompact(val),
    backend: "vm",
    u16Stream: stream,
    asmText,
    wasmBytes: multiFnWasm,
    wasmValid: multiFnWasmValid,
    elfBytes: null,
    elfFnv1a64: null,
    inspectText,
  };
}

export function executeOmniSurfaceProgram(src: string, args: OmniValue[]): OmniValue {
  return executeOmniSurfaceAotFirst(src, args, false).value;
}

// ==============================================================================
// STRUCTURED ERRORS, HARNESS JSON, AGENT REPAIR DRIVER, INSPECTOR & EXPLAINER
// ==============================================================================

export interface OmniStructuredError {
  code: string;
  line: number | null;
  col: number | null;
  message: string;
  retryHint: string;
}

const VALID_ERROR_CODES = new Set([
  "TAINT_ON_RET",
  "TAINT_SINK",
  "MISSING_CAP",
  "LINEAR_LEAK",
  "BORROW_CONFLICT",
  "UNDEF_VAR",
  "IMMUT_PARAM",
  "BAD_TYPE",
  "UNKNOWN_FN",
  "ARITY_MISMATCH",
  "UNRESOLVED_LABEL",
  "CIRCULAR_INCLUDE",
  "BUDGET_EXCEEDED",
  "CORRUPT_PACK5",
  "AMBIGUOUS_GRAMMAR",
  "SYNTAX_ERROR",
]);

export function mapErrorCodeStr(tag: string): string | null {
  return VALID_ERROR_CODES.has(tag) ? tag : null;
}

export function retryHintForCode(code: string): string {
  switch (code) {
    case "TAINT_ON_RET":
      return "Sanitize @untrusted data with san_sql(x), san_dom(x), san_prm(x), or auth_verify(x) before returning.";
    case "TAINT_SINK":
      return "Pass @untrusted values through san_sql(x), san_dom(x), san_prm(x), or auth_verify(x) before invoking a safe-sink opcode.";
    case "MISSING_CAP":
      return "Add the required capability to the function or route header, e.g. #[cap(AI, NET, DB_R, DB_W, AUTH)] or @cap(net, db_r).";
    case "LINEAR_LEAK":
      return "Ensure every allocated linear register or arena is consumed by drop(...) or returned before RET.";
    case "BORROW_CONFLICT":
      return "Release active borrows before taking a conflicting mutable borrow_w(...) or moving the register.";
    case "UNDEF_VAR":
      return "Declare the variable with 'let <name>: <ty> = <expr>;' before referencing or reassigning it.";
    case "IMMUT_PARAM":
      return "Copy the immutable parameter into a local variable ('let tmp: i64 = param;') before reassigning.";
    case "BAD_TYPE":
      return "Use a valid canonical type ('i64', 'trit', 'bytes', 'array', 'trit_vec', 'str', 'arena') and annotate every 'let' in #[strict] mode.";
    case "UNKNOWN_FN":
      return "Use a built-in intrinsic (e.g. iabs, imin, imax, tmaj, sel_trit, egate, tdot, db_query, req.param, respond) or define the helper fn.";
    case "ARITY_MISMATCH":
      return "Check the expected argument count of the intrinsic or user-defined function.";
    case "UNRESOLVED_LABEL":
      return "Ensure every 'goto N;' or 'br_pos(c, N);' has a matching 'label N;' (0..=15) in the same function.";
    case "CIRCULAR_INCLUDE":
      return "Remove the cyclic 'include \"...\";' reference between modules.";
    case "BUDGET_EXCEEDED":
      return "Reduce loop iterations or allocations, or raise #[step_budget(N)] / #[mem_budget(N)].";
    case "CORRUPT_PACK5":
      return "Ensure all @PACK5 bytes are in 0..=242 (valid 5-trit encodings) without sentinel bytes 243..=255.";
    case "AMBIGUOUS_GRAMMAR":
      return "In #[strict] mode, use canonical '+1t'/'0t'/'-1t' trits, ': <ty>' on every let, '@untrusted' suffix, and 'include \"...\"'.";
    default:
      return "Check surface syntax, balanced braces, and function/route signatures.";
  }
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function parseOmniStructuredError(err: string): OmniStructuredError {
  let code = "SYNTAX_ERROR";
  const bMatch = err.match(/error\[([A-Z0-9_]+)\]/);
  if (bMatch && mapErrorCodeStr(bMatch[1])) {
    code = bMatch[1];
  } else if (err.includes("Taint violation on RET")) code = "TAINT_ON_RET";
  else if (err.includes("Sink opcode") && err.includes("@UNTRUSTED")) code = "TAINT_SINK";
  else if (err.includes("Capability violation")) code = "MISSING_CAP";
  else if (err.includes("Linear value leak")) code = "LINEAR_LEAK";
  else if (err.includes("Borrow violation")) code = "BORROW_CONFLICT";
  else if (err.includes("budget exceeded")) code = "BUDGET_EXCEEDED";
  else if (err.includes("PACK5_SENTINEL_OR_CORRUPT_BYTE")) code = "CORRUPT_PACK5";
  else if (err.includes("Undefined variable") || err.includes("undeclared variable")) code = "UNDEF_VAR";
  else if (err.includes("immutable function parameter")) code = "IMMUT_PARAM";
  else if (err.includes("Unknown surface type")) code = "BAD_TYPE";
  else if (err.includes("Unknown surface intrinsic")) code = "UNKNOWN_FN";
  else if (err.includes("Unresolved branch target label") || err.includes("Unresolved jump target")) {
    code = "UNRESOLVED_LABEL";
  } else if (err.includes("Circular include")) code = "CIRCULAR_INCLUDE";

  let lineNum: number | null = null;
  let colNum: number | null = null;
  const lcMatch = err.match(/line\s+(\d+)(?:,\s*col\s+(\d+))?/);
  if (lcMatch) {
    lineNum = Math.max(1, Number.parseInt(lcMatch[1], 10));
    if (lcMatch[2]) colNum = Math.max(1, Number.parseInt(lcMatch[2], 10));
  }

  return {
    code,
    line: lineNum ?? 1,
    col: colNum ?? 1,
    message: err,
    retryHint: retryHintForCode(code),
  };
}

export function formatOmniHumanDiagnostic(src: string, err: OmniStructuredError): string {
  const lineNo = Math.max(1, err.line ?? 1);
  const colNo = Math.max(1, err.col ?? 1);
  const srcLine = src.split(/\r?\n/)[lineNo - 1] ?? "";
  const caretPad = " ".repeat(Math.max(0, colNo - 1));
  const lnPad = String(lineNo).padStart(2, " ");
  return `error[${err.code}]: ${err.message}\n  --> line ${lineNo}:${colNo}\n   |\n${lnPad} | ${srcLine}\n   | ${caretPad}^\n   = help: ${err.retryHint}`;
}

export function omniHarnessSpecObj() {
  return {
    version: OMNI_HARNESS_CONTRACT_VERSION,
    isa_version: `0x${OP_BASE.toString(16).toUpperCase().padStart(4, "0")}`,
    types: ["i64", "trit", "bytes", "array", "trit_vec", "str", "arena"],
    capabilities: ["PURE", "DB_R", "DB_W", "NET", "DOM", "AI", "VISION", "AUDIO", "AUTH", "FS", "SYS"],
    error_codes: [
      "TAINT_ON_RET",
      "TAINT_SINK",
      "MISSING_CAP",
      "LINEAR_LEAK",
      "BORROW_CONFLICT",
      "UNDEF_VAR",
      "IMMUT_PARAM",
      "BAD_TYPE",
      "UNKNOWN_FN",
      "ARITY_MISMATCH",
      "UNRESOLVED_LABEL",
      "CIRCULAR_INCLUDE",
      "BUDGET_EXCEEDED",
      "CORRUPT_PACK5",
      "AMBIGUOUS_GRAMMAR",
      "SYNTAX_ERROR",
    ],
    strict_rules: [
      "Explicit : <ty> on every let",
      "Canonical trit literals (+1t, 0t, -1t)",
      "Canonical @untrusted suffix",
      "Canonical include directive",
    ],
  };
}

export function omniHarnessSpecJson(): string {
  return JSON.stringify(omniHarnessSpecObj());
}

export interface OmniHarnessResult {
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

export function runOmniHarness(
  src: string,
  args: OmniValue[],
  strict: boolean,
  includeRetryHint: boolean,
): OmniHarnessResult {
  const wrapped = strict ? src : wrapOmniSurfaceSnippet(src);
  let u16Stream: number[];
  try {
    u16Stream = strict ? compileOmniSurfaceStrictToU16(wrapped) : compileOmniSurfaceToU16(wrapped);
  } catch (e) {
    const st = parseOmniStructuredError(errMsg(e));
    const res: OmniHarnessResult = {
      ok: false,
      version: OMNI_HARNESS_CONTRACT_VERSION,
      code: st.code,
      line: st.line,
      col: st.col,
      message: st.message,
      tokens: null,
    };
    if (includeRetryHint) res.retry_hint = st.retryHint;
    return res;
  }

  const tokenCount = u16Stream.length;
  try {
    const detail = executeOmniSurfaceAotFirst(wrapped, args, strict);
    return {
      ok: true,
      version: OMNI_HARNESS_CONTRACT_VERSION,
      result: detail.formattedResult,
      tokens: tokenCount,
      backend: detail.backend,
    };
  } catch (e) {
    const st = parseOmniStructuredError(errMsg(e));
    const res: OmniHarnessResult = {
      ok: false,
      version: OMNI_HARNESS_CONTRACT_VERSION,
      code: st.code,
      line: st.line,
      col: st.col,
      message: st.message,
      tokens: tokenCount,
    };
    if (includeRetryHint) res.retry_hint = st.retryHint;
    return res;
  }
}

export function runOmniHarnessJson(
  src: string,
  args: OmniValue[],
  strict: boolean,
  includeRetryHint: boolean,
): string {
  return JSON.stringify(runOmniHarness(src, args, strict, includeRetryHint));
}

// ==============================================================================
// REPAIR STRATEGIES & AUTONOMOUS AGENT DRIVER
// ==============================================================================

export interface OmniRepairStrategySpec {
  id: string;
  version: string;
  error_code: string;
  context_pattern: string;
  description: string;
}

export function listOmniRepairStrategies(): OmniRepairStrategySpec[] {
  const v = OMNI_REPAIR_STRATEGIES_VERSION;
  return [
    {
      id: "strat_missing_cap_minimal_v1",
      version: v,
      error_code: "MISSING_CAP",
      context_pattern: "single_or_multi_fn_opcode_cap",
      description:
        "Infers the minimal required capability bit(s) from the failing opcode/function and attaches or merges #[cap(...)] on the enclosing fn or route.",
    },
    {
      id: "strat_multi_fn_callee_cap_v1",
      version: v,
      error_code: "MISSING_CAP",
      context_pattern: "multi_fn_callee",
      description:
        "Attaches minimal required #[cap(...)] to a helper callee in a multi-function module and propagates capabilities to callers.",
    },
    {
      id: "strat_taint_sanitize_ret_v1",
      version: v,
      error_code: "TAINT_ON_RET",
      context_pattern: "untrusted_return",
      description:
        "Sanitizes @untrusted return values with auth_verify(...) or san_prm(...) and adds minimal AUTH/AI capability.",
    },
    {
      id: "strat_taint_sanitize_sink_v1",
      version: v,
      error_code: "TAINT_SINK",
      context_pattern: "untrusted_sink_arg",
      description:
        "Wraps @untrusted arguments passed to egate(...) or db_query(...) in auth_verify(...) and adds AUTH capability.",
    },
    {
      id: "strat_bad_type_infer_let_v1",
      version: v,
      error_code: "BAD_TYPE",
      context_pattern: "strict_untyped_let_or_unknown_type",
      description:
        "Infers canonical Omni types (i64, trit, trit_vec, bytes, str, arena) on untyped let bindings or replaces unknown type names with canonical i64.",
    },
    {
      id: "strat_ambiguous_grammar_canonical_v1",
      version: v,
      error_code: "AMBIGUOUS_GRAMMAR",
      context_pattern: "strict_non_canonical_alias",
      description:
        "Normalizes TRIT_POS/ZERO/NEG to +1t/0t/-1t, int/tvec to i64/trit_vec, untrusted_<ty> to <ty> @untrusted, and mod to include.",
    },
    {
      id: "strat_budget_aware_synthesis_v1",
      version: v,
      error_code: "BUDGET_EXCEEDED",
      context_pattern: "step_or_mem_budget_trap",
      description:
        "Synthesizes right-sized #[step_budget(N)] or #[mem_budget(N)] attributes when execution traps under a tight budget.",
    },
    {
      id: "strat_linear_arena_drop_v1",
      version: v,
      error_code: "LINEAR_LEAK",
      context_pattern: "undropped_arena_at_ret",
      description: "Inserts drop(arena_var) immediately before return to satisfy linear arena conservation.",
    },
    {
      id: "strat_borrow_order_swap_v1",
      version: v,
      error_code: "BORROW_CONFLICT",
      context_pattern: "drop_owner_before_reader",
      description:
        "Reorders drop(reader) before drop(owner) so all active reader borrows are released before dropping the arena.",
    },
    {
      id: "strat_immut_param_local_copy_v1",
      version: v,
      error_code: "IMMUT_PARAM",
      context_pattern: "reassign_fn_param",
      description: "Copies an immutable function parameter into a mutable local SSA binding before reassignment.",
    },
    {
      id: "strat_corrupt_pack5_clamp_v1",
      version: v,
      error_code: "CORRUPT_PACK5",
      context_pattern: "sentinel_byte_243_to_255",
      description: "Clamps invalid @PACK5 sentinel bytes (243..=255) into the valid 5-trit radix-3 domain (0..=242).",
    },
    {
      id: "strat_multi_fn_arity_align_v1",
      version: v,
      error_code: "ARITY_MISMATCH",
      context_pattern: "call_arity_mismatch",
      description:
        "Aligns caller argument count with the target function or intrinsic signature by trimming extra args or padding missing defaults.",
    },
    {
      id: "strat_include_cycle_prune_v1",
      version: v,
      error_code: "CIRCULAR_INCLUDE",
      context_pattern: "cyclic_include_directive",
      description: "Prunes cyclic include directives from single-file or multi-file module graphs.",
    },
    {
      id: "strat_unresolved_label_insert_v1",
      version: v,
      error_code: "UNRESOLVED_LABEL",
      context_pattern: "missing_branch_target_label",
      description: "Inserts the missing basic-block target 'label N;' before function return.",
    },
    {
      id: "strat_undef_var_init_v1",
      version: v,
      error_code: "UNDEF_VAR",
      context_pattern: "undeclared_variable_use",
      description: "Declares a typed local binding 'let <var>: i64 = 0;' prior to the first use of an undefined variable.",
    },
    {
      id: "strat_unknown_fn_alias_v1",
      version: v,
      error_code: "UNKNOWN_FN",
      context_pattern: "unknown_intrinsic_or_helper",
      description: "Replaces an unknown function call with a canonical built-in intrinsic (iabs).",
    },
  ];
}

export function omniRepairStrategiesJson(): string {
  const strategies = listOmniRepairStrategies();
  return JSON.stringify({
    strategy_version: OMNI_REPAIR_STRATEGIES_VERSION,
    harness_version: OMNI_HARNESS_CONTRACT_VERSION,
    count: strategies.length,
    strategies,
  });
}

export function selectOmniRepairStrategy(src: string, err: OmniStructuredError): string {
  const fnCount = src.split("fn ").length - 1;
  switch (err.code) {
    case "MISSING_CAP":
      return fnCount > 1 ? "strat_multi_fn_callee_cap_v1" : "strat_missing_cap_minimal_v1";
    case "TAINT_ON_RET":
      return "strat_taint_sanitize_ret_v1";
    case "TAINT_SINK":
      return "strat_taint_sanitize_sink_v1";
    case "BAD_TYPE":
      return "strat_bad_type_infer_let_v1";
    case "AMBIGUOUS_GRAMMAR":
      return "strat_ambiguous_grammar_canonical_v1";
    case "BUDGET_EXCEEDED":
      return "strat_budget_aware_synthesis_v1";
    case "LINEAR_LEAK":
      return "strat_linear_arena_drop_v1";
    case "BORROW_CONFLICT":
      return "strat_borrow_order_swap_v1";
    case "IMMUT_PARAM":
      return "strat_immut_param_local_copy_v1";
    case "CORRUPT_PACK5":
      return "strat_corrupt_pack5_clamp_v1";
    case "ARITY_MISMATCH":
      return "strat_multi_fn_arity_align_v1";
    case "CIRCULAR_INCLUDE":
      return "strat_include_cycle_prune_v1";
    case "UNRESOLVED_LABEL":
      return "strat_unresolved_label_insert_v1";
    case "UNDEF_VAR":
      return "strat_undef_var_init_v1";
    case "UNKNOWN_FN":
      return "strat_unknown_fn_alias_v1";
    default:
      return "strat_syntax_fallback_v1";
  }
}

function inferSurfaceRhsType(rhs: string): string {
  const e = (rhs.split(";")[0] ?? rhs).trim();
  if (
    e.startsWith('"') ||
    e.startsWith("san_sql(") ||
    e.startsWith("san_dom(") ||
    e.startsWith("san_prm(") ||
    e.startsWith("scat(") ||
    e.startsWith("sconc(") ||
    e.startsWith("i2s(")
  ) {
    return "str";
  }
  if (e.startsWith("[") || e.startsWith("unpack5(")) return "trit_vec";
  if (e.startsWith("pack5_bytes(") || e.startsWith("anew(") || e.startsWith("pack5(")) return "bytes";
  if (e.startsWith("arena(")) return "arena";
  if (
    e.includes("<=>") ||
    e.includes("==") ||
    e.includes("!=") ||
    e.includes("<=") ||
    e.includes(">=") ||
    e.includes(" < ") ||
    e.includes(" > ") ||
    e.startsWith("egate(") ||
    e.startsWith("decision_hook(") ||
    e.startsWith("tlogic(") ||
    e.startsWith("tmaj(") ||
    e.startsWith("tand(") ||
    e.startsWith("tor(") ||
    e.startsWith("txor(") ||
    e.startsWith("tcmp(") ||
    e === "+1t" ||
    e === "0t" ||
    e === "-1t" ||
    e.startsWith("!")
  ) {
    return "trit";
  }
  return "i64";
}

export function inferMinimalCapabilitiesForError(src: string, err: OmniStructuredError): string[] {
  const needed: string[] = [];
  if (err.message.includes("0x7F35")) needed.push("AI");
  if (
    err.message.includes("0x7F6C") ||
    err.message.includes("0x7F6D") ||
    err.message.includes("0x7F6E") ||
    err.message.includes("0x7F6F")
  ) {
    needed.push("NET");
  }
  if (err.message.includes("0x7F64") || err.message.includes("0x7F65") || err.message.includes("0x7F68")) {
    needed.push("DB_R");
  }
  if (err.message.includes("0x7F63") || err.message.includes("0x7F6B")) needed.push("DB_W");
  if (err.message.includes("0x7F70")) needed.push("AUTH");
  if (err.message.includes("0x7F76")) needed.push("DOM");
  if (needed.length > 0) return needed;

  if (
    src.includes("route ") ||
    src.includes("req.param") ||
    src.includes("req_param") ||
    src.includes("respond(") ||
    src.includes("respond_json(") ||
    src.includes("respond_html(")
  ) {
    needed.push("NET");
  }
  if (src.includes("db_query") || src.includes("db_get") || src.includes("kv_get") || src.includes("db_count")) {
    needed.push("DB_R");
  }
  if (src.includes("kv_put") || src.includes("db_put")) needed.push("DB_W");
  if (src.includes("egate(") || src.includes("tlogic(") || src.includes("decision_hook(") || src.includes("pack5_gate(")) {
    needed.push("AI");
  }
  if (src.includes("auth_verify(")) needed.push("AUTH");
  if (src.includes("san_dom(")) needed.push("DOM");
  if (src.includes("san_sql(")) needed.push("DB_W");
  if (needed.length === 0) needed.push("AI");
  return needed;
}

export function repairOmniProgramFromHarnessError(
  src: string,
  err: OmniStructuredError,
  _strict: boolean,
): string | null {
  const targetLine = Math.max(1, err.line ?? 1);
  switch (err.code) {
    case "MISSING_CAP": {
      const needed = inferMinimalCapabilitiesForError(src, err);
      const capAttr = `#[cap(${needed.join(", ")})]`;
      const lines = src.split(/\r?\n/);
      if (src.split("fn ").length - 1 > 1 && targetLine <= lines.length) {
        let fnLineIdx: number | null = null;
        for (let idx = targetLine - 1; idx >= 0; idx--) {
          const trimmed = lines[idx].trimStart();
          if (trimmed.startsWith("fn ") || trimmed.includes(" fn ") || trimmed.startsWith("route ")) {
            fnLineIdx = idx;
            break;
          }
        }
        if (fnLineIdx !== null) {
          const newLines = [...lines];
          const mergeCap = (lineStr: string): string | null => {
            const cStart = lineStr.indexOf("#[cap(");
            if (cStart === -1) return null;
            const relEnd = lineStr.indexOf(")]", cStart);
            if (relEnd === -1) return null;
            const existing = lineStr.slice(cStart + 6, relEnd);
            const merged = existing
              .split(",")
              .map((s) => s.trim().toUpperCase())
              .filter(Boolean);
            for (const n of needed) {
              if (!merged.includes(n)) merged.push(n);
            }
            return `${lineStr.slice(0, cStart)}#[cap(${merged.join(", ")})]${lineStr.slice(relEnd + 2)}`;
          };
          if (fnLineIdx > 0 && newLines[fnLineIdx - 1].includes("#[cap(")) {
            const updated = mergeCap(newLines[fnLineIdx - 1]);
            if (updated) {
              newLines[fnLineIdx - 1] = updated;
              const out = newLines.join("\n");
              if (out !== src) return out;
            }
          } else if (newLines[fnLineIdx].includes("#[cap(")) {
            const updated = mergeCap(newLines[fnLineIdx]);
            if (updated) {
              newLines[fnLineIdx] = updated;
              const out = newLines.join("\n");
              if (out !== src) return out;
            }
          } else {
            newLines[fnLineIdx] = `${capAttr} ${newLines[fnLineIdx]}`;
            const out = newLines.join("\n");
            if (out !== src) return out;
          }
        }
      }
      const cStart = src.indexOf("#[cap(");
      if (cStart !== -1) {
        const relEnd = src.indexOf(")]", cStart);
        if (relEnd !== -1) {
          const existing = src.slice(cStart + 6, relEnd);
          const merged = existing
            .split(",")
            .map((s) => s.trim().toUpperCase())
            .filter(Boolean);
          for (const n of needed) {
            if (!merged.includes(n)) merged.push(n);
          }
          const out = `${src.slice(0, cStart)}#[cap(${merged.join(", ")})]${src.slice(relEnd + 2)}`;
          if (out !== src) return out;
        }
      }
      const fnOrRoutePos = src.includes("route ") ? src.indexOf("route ") : src.indexOf("fn ");
      if (fnOrRoutePos !== -1) {
        return `${src.slice(0, fnOrRoutePos)}${capAttr} ${src.slice(fnOrRoutePos)}`;
      }
      return `${capAttr}\n${src}`;
    }

    case "TAINT_ON_RET":
    case "TAINT_SINK": {
      if (src.includes("@untrusted")) {
        let sanitized = src.replaceAll("@untrusted", "@safe");
        const lines = src.split(/\r?\n/);
        if (err.code === "TAINT_ON_RET" && targetLine <= lines.length) {
          const lineS = lines[targetLine - 1];
          const retIdx = lineS.indexOf("return ");
          if (retIdx !== -1) {
            const varExpr = lineS.slice(retIdx + 7).trim().replace(/;+$/, "").trim();
            const repStmt = `let __clean: i64 = auth_verify(${varExpr}); return __clean;`;
            const newLines = [...lines];
            newLines[targetLine - 1] = `${lineS.slice(0, retIdx)}${repStmt}`;
            const joined = newLines.join("\n");
            sanitized = joined.includes("#[cap(")
              ? joined.replace("#[cap(", "#[cap(AUTH, ")
              : `#[cap(AUTH)] ${joined}`;
          }
        } else if (err.code === "TAINT_SINK" && targetLine <= lines.length) {
          const lineS = lines[targetLine - 1];
          const newLines = [...lines];
          const addCap = (joined: string, fallback: string) =>
            joined.includes("#[cap(") ? joined.replace("#[cap(", "#[cap(AUTH, ") : `#[cap(${fallback})] ${joined}`;
          if (lineS.includes("egate(")) {
            newLines[targetLine - 1] = lineS.replace("egate(", "egate(auth_verify(").replace(");", "));");
            sanitized = addCap(newLines.join("\n"), "AI, AUTH");
          } else if (lineS.includes("tlogic(")) {
            newLines[targetLine - 1] = lineS.replace("tlogic(", "tlogic(auth_verify(").replace(");", "));");
            sanitized = addCap(newLines.join("\n"), "AI, AUTH");
          } else if (lineS.includes("db_get(")) {
            newLines[targetLine - 1] = lineS.replace("db_get(", "db_get(auth_verify(").replace(");", "));");
            sanitized = addCap(newLines.join("\n"), "DB_R, AUTH");
          } else if (lineS.includes("kv_put(")) {
            const comma = lineS.lastIndexOf(",");
            const rparen = lineS.lastIndexOf(")");
            if (comma !== -1 && rparen !== -1) {
              const arg2 = lineS.slice(comma + 1, rparen).trim();
              newLines[targetLine - 1] = `${lineS.slice(0, comma)}, auth_verify(${arg2})${lineS.slice(rparen)}`;
              sanitized = addCap(newLines.join("\n"), "DB_W, AUTH");
            }
          } else if (lineS.includes("db_query(")) {
            const comma = lineS.lastIndexOf(",");
            const rparen = lineS.lastIndexOf(")");
            if (comma !== -1 && rparen !== -1) {
              const arg2 = lineS.slice(comma + 1, rparen).trim();
              newLines[targetLine - 1] = `${lineS.slice(0, comma)}, auth_verify(${arg2})${lineS.slice(rparen)}`;
              sanitized = addCap(newLines.join("\n"), "DB_R, AUTH");
            }
          }
        }
        if (sanitized !== src) return sanitized;
      }
      return null;
    }

    case "BAD_TYPE": {
      if (err.message.includes("Missing explicit type annotation on 'let ")) {
        const lines = src.split(/\r?\n/);
        const newLines: string[] = [];
        for (let idx = 0; idx < lines.length; idx++) {
          const lineS = lines[idx];
          if (idx + 1 === targetLine || lineS.includes("let ")) {
            let searchOffset = 0;
            let fixedLine: string | null = null;
            while (true) {
              const relLet = lineS.indexOf("let ", searchOffset);
              if (relLet === -1) break;
              const afterLet = lineS.slice(relLet + 4);
              const eqPos = afterLet.indexOf("=");
              if (eqPos !== -1) {
                const lhsVar = afterLet.slice(0, eqPos).trim();
                if (!lhsVar.includes(":") && !lhsVar.includes(";")) {
                  const rhs = afterLet.slice(eqPos + 1);
                  const inferredTy = inferSurfaceRhsType(rhs);
                  fixedLine = `${lineS.slice(0, relLet)}let ${lhsVar}: ${inferredTy} =${rhs}`;
                  break;
                }
              }
              searchOffset = relLet + 4;
            }
            if (fixedLine !== null) {
              newLines.push(fixedLine);
              continue;
            }
          }
          newLines.push(lineS);
        }
        const out = newLines.join("\n");
        if (out !== src) return out;
      } else if (err.message.includes("Unknown surface type '")) {
        const m = err.message.match(/Unknown surface type '([^']+)'/);
        if (m) {
          const out = src.replaceAll(m[1], "i64");
          if (out !== src) return out;
        }
      }
      return null;
    }

    case "AMBIGUOUS_GRAMMAR": {
      let out = src
        .replaceAll("TRIT_POS", "+1t")
        .replaceAll("TRIT_ZERO", "0t")
        .replaceAll("TRIT_NEG", "-1t")
        .replaceAll("untrusted_i64", "i64 @untrusted")
        .replaceAll("untrusted_str", "str @untrusted")
        .replaceAll(": int", ": i64")
        .replaceAll("-> int", "-> i64")
        .replaceAll(": tvec", ": trit_vec")
        .replaceAll("-> tvec", "-> trit_vec");
      if (out.includes('mod "')) out = out.replaceAll('mod "', 'include "');
      return out !== src ? out : null;
    }

    case "BUDGET_EXCEEDED": {
      const sbPos = src.indexOf("#[step_budget(");
      if (sbPos !== -1) {
        const endP = src.indexOf(")]", sbPos);
        if (endP !== -1) {
          return `${src.slice(0, sbPos)}#[step_budget(500000)]${src.slice(endP + 2)}`;
        }
      }
      const mbPos = src.indexOf("#[mem_budget(");
      if (mbPos !== -1) {
        const endP = src.indexOf(")]", mbPos);
        if (endP !== -1) {
          return `${src.slice(0, mbPos)}#[mem_budget(1048576)]${src.slice(endP + 2)}`;
        }
      }
      return null;
    }

    case "LINEAR_LEAK": {
      const lines = src.split(/\r?\n/);
      const fnRanges: Array<[number, number]> = [];
      let curStart: number | null = null;
      let braceDepth = 0;
      for (let idx = 0; idx < lines.length; idx++) {
        const lineS = lines[idx];
        const t = lineS.trimStart();
        if (curStart === null && (t.startsWith("fn ") || t.includes(" fn ") || t.startsWith("route "))) {
          curStart = idx;
          braceDepth = 0;
        }
        if (curStart !== null) {
          for (const ch of lineS) {
            if (ch === "{") braceDepth += 1;
            else if (ch === "}") {
              braceDepth = Math.max(0, braceDepth - 1);
              if (braceDepth === 0 && curStart !== null) {
                fnRanges.push([curStart, idx + 1]);
                curStart = null;
                break;
              }
            }
          }
        }
      }
      if (curStart !== null) fnRanges.push([curStart, lines.length]);
      if (fnRanges.length === 0) fnRanges.push([0, lines.length]);

      const newLines = [...lines];
      let patched = false;
      for (const [fnStart, fnEnd] of fnRanges) {
        const fnSlice = lines.slice(fnStart, fnEnd).join("\n");
        const arenaVars: string[] = [];
        for (let idx = fnStart; idx < fnEnd; idx++) {
          const lineS = lines[idx];
          if (lineS.includes("arena(") || lineS.includes("borrow_r(") || lineS.includes("borrow_w(")) {
            const letPos = lineS.indexOf("let ");
            if (letPos !== -1) {
              const varPart = lineS.slice(letPos + 4);
              const vName = (varPart.split(/[:=\s]/)[0] ?? "").trim();
              if (vName && !fnSlice.includes(`drop(${vName});`)) {
                arenaVars.push(vName);
              }
            }
          }
        }
        if (arenaVars.length > 0) {
          arenaVars.reverse();
          const dropStmts = arenaVars.map((v) => `drop(${v}); `).join("");
          for (let idx = fnStart; idx < fnEnd; idx++) {
            if (newLines[idx].includes("return ")) {
              newLines[idx] = newLines[idx].replace("return ", `${dropStmts}return `);
              patched = true;
              break;
            }
          }
          if (patched) break;
        }
      }
      if (patched) {
        const out = newLines.join("\n");
        if (out !== src) return out;
      }
      return null;
    }

    case "BORROW_CONFLICT": {
      const lines = src.split(/\r?\n/);
      if (targetLine <= lines.length && targetLine < lines.length) {
        const curL = lines[targetLine - 1].trim();
        const nextL = lines[targetLine].trim();
        if (curL.startsWith("drop(") && nextL.startsWith("drop(")) {
          const newLines = [...lines];
          const tmp = newLines[targetLine - 1];
          newLines[targetLine - 1] = newLines[targetLine];
          newLines[targetLine] = tmp;
          const out = newLines.join("\n");
          if (out !== src) return out;
        }
      }
      return null;
    }

    case "IMMUT_PARAM": {
      const m = err.message.match(/parameter '([^']+)'/);
      if (!m) return null;
      const pName = m[1];
      const lhsPat = `${pName} =`;
      const lines = src.split(/\r?\n/);
      const newLines: string[] = [];
      let replaced = false;
      const mutName = `${pName}_mut`;
      for (let idx = 0; idx < lines.length; idx++) {
        const lineS = lines[idx];
        if (!replaced && (idx + 1 === targetLine || lineS.includes(lhsPat))) {
          const pPos = lineS.indexOf(lhsPat);
          if (pPos !== -1) {
            const rhs = lineS.slice(pPos + lhsPat.length);
            const ty = inferSurfaceRhsType(rhs);
            newLines.push(`${lineS.slice(0, pPos)}let ${mutName}: ${ty} =${rhs}`);
            replaced = true;
            continue;
          }
        }
        if (replaced && lineS.includes(`return ${pName};`)) {
          newLines.push(lineS.replaceAll(`return ${pName};`, `return ${mutName};`));
        } else {
          newLines.push(lineS);
        }
      }
      const out = newLines.join("\n");
      return out !== src ? out : null;
    }

    case "CORRUPT_PACK5": {
      let out = src;
      for (let b = 243; b <= 255; b++) {
        const pat = `, ${b})`;
        if (out.includes(pat)) out = out.replaceAll(pat, ", 242)");
      }
      return out !== src ? out : null;
    }

    case "ARITY_MISMATCH": {
      const fnM = err.message.match(/'([^']+)'/);
      const exM = err.message.match(/expects (\d+)/);
      if (!fnM || !exM) return null;
      const callee = fnM[1];
      const wantN = Number.parseInt(exM[1], 10);
      const callPat = `${callee}(`;
      const lines = src.split(/\r?\n/);
      if (targetLine <= lines.length) {
        const lineS = lines[targetLine - 1];
        const cPos = lineS.indexOf(callPat);
        if (cPos !== -1) {
          const argsStart = cPos + callPat.length;
          const relClose = lineS.indexOf(")", argsStart);
          if (relClose !== -1) {
            const argsInside = lineS.slice(argsStart, relClose);
            const parts = argsInside.trim() ? argsInside.split(",").map((s) => s.trim()) : [];
            if (parts.length > wantN) parts.length = wantN;
            while (parts.length < wantN) parts.push("1");
            const fixedCall = `${callee}(${parts.join(", ")})`;
            const newLines = [...lines];
            newLines[targetLine - 1] = `${lineS.slice(0, cPos)}${fixedCall}${lineS.slice(relClose + 1)}`;
            const out = newLines.join("\n");
            if (out !== src) return out;
          }
        }
      }
      return null;
    }

    case "CIRCULAR_INCLUDE": {
      const out = src
        .split(/\r?\n/)
        .filter((l) => {
          const t = l.trimStart();
          return !(t.startsWith("include ") || t.startsWith('include"') || t.startsWith("mod "));
        })
        .join("\n");
      return out !== src ? out : null;
    }

    case "UNRESOLVED_LABEL": {
      const m = err.message.match(/label (\d+)/);
      if (m && src.includes("return ")) {
        const out = src.replace("return ", `label ${m[1]}; return `);
        if (out !== src) return out;
      }
      return null;
    }

    case "UNDEF_VAR": {
      const m = err.message.match(/variable '([^']+)'/);
      if (!m) return null;
      const lines = src.split(/\r?\n/);
      if (targetLine <= lines.length) {
        const newLines: string[] = [];
        lines.forEach((l, idx) => {
          if (idx + 1 === targetLine) newLines.push(`    let ${m[1]}: i64 = 0;`);
          newLines.push(l);
        });
        const out = newLines.join("\n");
        if (out !== src) return out;
      }
      return null;
    }

    case "UNKNOWN_FN": {
      const m = err.message.match(/function '([^']+)'/);
      if (!m) return null;
      const out = src.replaceAll(`${m[1]}(`, "iabs(");
      return out !== src ? out : null;
    }

    default:
      return null;
  }
}

export interface OmniAgentHistoryEntry {
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

export interface OmniAgentDriverReport {
  ok: boolean;
  version: string;
  strategy_version: string;
  attempts: number;
  history: OmniAgentHistoryEntry[];
  final_source: string;
  harness: OmniHarnessResult;
}

export function runOmniAgentDriver(
  initialSrc: string,
  args: OmniValue[],
  strict: boolean,
  maxAttempts = 6,
  expectedContractVersion = OMNI_HARNESS_CONTRACT_VERSION,
): OmniAgentDriverReport {
  const spec = omniHarnessSpecObj();
  if (spec.version !== expectedContractVersion) {
    throw new Error(
      `error[VERSION_MISMATCH]: omni-agent-driver expected '${expectedContractVersion}', got '${spec.version}'`,
    );
  }
  const limit = Math.max(1, maxAttempts);
  let currentSrc = initialSrc;
  const history: OmniAgentHistoryEntry[] = [];

  for (let attemptIdx = 1; attemptIdx <= limit; attemptIdx++) {
    const hRes = runOmniHarness(currentSrc, args, strict, true);
    if (hRes.ok) {
      history.push({ attempt: attemptIdx, ok: true, source: currentSrc });
      return {
        ok: true,
        version: OMNI_HARNESS_CONTRACT_VERSION,
        strategy_version: OMNI_REPAIR_STRATEGIES_VERSION,
        attempts: history.length,
        history,
        final_source: currentSrc,
        harness: hRes,
      };
    }

    const stErr: OmniStructuredError = {
      code: hRes.code ?? "SYNTAX_ERROR",
      line: hRes.line ?? 1,
      col: hRes.col ?? 1,
      message: hRes.message ?? "Unknown error",
      retryHint: hRes.retry_hint ?? retryHintForCode(hRes.code ?? "SYNTAX_ERROR"),
    };
    const stratId = selectOmniRepairStrategy(currentSrc, stErr);
    history.push({
      attempt: attemptIdx,
      ok: false,
      code: stErr.code,
      line: stErr.line ?? 1,
      col: stErr.col ?? 1,
      message: stErr.message,
      retry_hint: stErr.retryHint,
      strategy_id: stratId,
      strategy_version: OMNI_REPAIR_STRATEGIES_VERSION,
      source: currentSrc,
    });

    if (attemptIdx === limit) {
      return {
        ok: false,
        version: OMNI_HARNESS_CONTRACT_VERSION,
        strategy_version: OMNI_REPAIR_STRATEGIES_VERSION,
        attempts: history.length,
        history,
        final_source: currentSrc,
        harness: hRes,
      };
    }

    const nextSrc = repairOmniProgramFromHarnessError(currentSrc, stErr, strict);
    if (!nextSrc || nextSrc === currentSrc) {
      return {
        ok: false,
        version: OMNI_HARNESS_CONTRACT_VERSION,
        strategy_version: OMNI_REPAIR_STRATEGIES_VERSION,
        attempts: history.length,
        history,
        final_source: currentSrc,
        harness: hRes,
      };
    }
    currentSrc = nextSrc;
  }

  throw new Error("Unreachable");
}

// ==============================================================================
// HTTP `route` HOST ABI DISPATCHER
// ==============================================================================

export interface OmniRouteSpec {
  fnIdx: number;
  method: string;
  pathPattern: string;
}

export interface OmniHttpResponse {
  status: number;
  contentType: string;
  body: string;
  kvSnapshot?: Record<string, string>;
}

export function extractOmniRoutes(src: string): OmniRouteSpec[] {
  const effectiveStrict = src.includes("#[strict]");
  const tokens = tokenizeOmniSurfaceInner(src, effectiveStrict);
  const parser = new SurfaceParser(tokens, effectiveStrict);
  const { includes, fns } = parser.parseModule();
  if (includes.length > 0) {
    const expanded = resolveOmniSurfaceIncludes(src);
    return extractOmniRoutes(expanded);
  }
  const multiFn = fns.length > 1;
  const routes: OmniRouteSpec[] = [];
  for (let idx = 0; idx < fns.length; idx++) {
    const f = fns[idx];
    const fid = f.fnIdAttr !== null ? f.fnIdAttr : !multiFn || idx + 1 === fns.length ? 0 : idx + 1;
    if (f.isRoute) {
      routes.push({
        fnIdx: fid,
        method: f.routeMethod ?? "GET",
        pathPattern: f.routePath ?? "/",
      });
    }
  }
  return routes;
}

export function matchOmniRoutePattern(
  pattern: string,
  requestPath: string,
): { primaryArg: bigint; params: Map<bigint, bigint> } | null {
  const qIdx = requestPath.indexOf("?");
  const cleanPath = qIdx !== -1 ? requestPath.slice(0, qIdx) : requestPath;
  const queryOpt = qIdx !== -1 ? requestPath.slice(qIdx + 1) : null;

  const patSegs = pattern.replace(/^\/+|\/+$/g, "").split("/").filter(Boolean);
  const reqSegs = cleanPath.replace(/^\/+|\/+$/g, "").split("/").filter(Boolean);
  if (patSegs.length !== reqSegs.length) return null;

  const params = new Map<bigint, bigint>();
  let primaryVal: bigint | null = null;
  for (let i = 0; i < patSegs.length; i++) {
    const ps = patSegs[i];
    const rs = reqSegs[i];
    if (ps.startsWith(":")) {
      const pName = ps.slice(1);
      const parsed = /^-?\d+$/.test(rs) ? BigInt.asIntN(64, BigInt(rs)) : fnv1a32Positive(rs);
      params.set(fnv1a32Positive(pName), parsed);
      if (primaryVal === null) primaryVal = parsed;
    } else if (ps !== rs) {
      return null;
    }
  }
  if (queryOpt) {
    for (const pair of queryOpt.split("&")) {
      const eq = pair.indexOf("=");
      if (eq !== -1) {
        const k = pair.slice(0, eq);
        const v = pair.slice(eq + 1);
        const parsed = /^-?\d+$/.test(v) ? BigInt.asIntN(64, BigInt(v)) : fnv1a32Positive(v);
        params.set(fnv1a32Positive(k), parsed);
        if (primaryVal === null) primaryVal = parsed;
      }
    }
  }
  return { primaryArg: primaryVal ?? 0n, params };
}

export function handleOmniHttpRequest(
  src: string,
  method: string,
  requestPath: string,
  kvStore: Map<bigint, bigint>,
): OmniHttpResponse {
  const routes = extractOmniRoutes(src);
  if (routes.length === 0) {
    throw new Error("error[SYNTAX_ERROR] line 1, col 1: Surface program contains no 'route' handlers");
  }
  const { stream, sourcemap } = compileOmniSurfaceToU16WithSourcemap(src, false);
  for (const rep of verifyU16Stream(stream)) {
    if (!rep.passed) {
      const first = rep.errors[0] ?? "";
      const annotated = annotateOmniErrorWithSourcemap(first, sourcemap);
      throw new Error(`${annotated} (fn #${rep.fnIdx}: ${JSON.stringify(rep.errors)})`);
    }
  }
  verifyModuleViaPhase7PureOmniKernel(stream);
  const mod = decodeOmniTokenStream(stream);
  const normMethod = method.trim().toUpperCase();

  for (const rt of routes) {
    if (rt.method === normMethod) {
      const matched = matchOmniRoutePattern(rt.pathPattern, requestPath);
      if (matched) {
        const { stepBudget, memBudget } = extractOmniSurfaceBudgets(src);
        const vm = new OmniBinaryVm();
        if (stepBudget !== null) vm.stepLimit = stepBudget;
        if (memBudget !== null) vm.memoryBudgetBytes = memBudget;
        vm.kvStore = new Map(kvStore);
        vm.reqParams = matched.params;
        let val: OmniValue;
        try {
          val = vm.executeFunction(mod, rt.fnIdx, [{ kind: "Int", value: matched.primaryArg }]);
        } catch (e) {
          throw new Error(annotateOmniErrorWithSourcemap(errMsg(e), sourcemap));
        }
        kvStore.clear();
        for (const [k, v] of vm.kvStore.entries()) kvStore.set(k, v);
        const kvSnapshot: Record<string, string> = {};
        for (const [k, v] of kvStore.entries()) kvSnapshot[k.toString()] = v.toString();

        if (vm.lastHttpResponse) {
          return {
            status: vm.lastHttpResponse.status,
            contentType: vm.lastHttpResponse.contentType,
            body: vm.lastHttpResponse.body,
            kvSnapshot,
          };
        }
        return {
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ status: 200, data: formatOmniValueCompact(val) }),
          kvSnapshot,
        };
      }
    }
  }

  const kvSnapshot: Record<string, string> = {};
  for (const [k, v] of kvStore.entries()) kvSnapshot[k.toString()] = v.toString();
  return {
    status: 404,
    contentType: "application/json",
    body: JSON.stringify({ status: 404, error: `No route matched ${normMethod} ${requestPath}` }),
    kvSnapshot,
  };
}

// ==============================================================================
// STREAM INSPECTOR & ERROR CODE EXPLAINER
// ==============================================================================

function decodeCapMaskNames(mask: CapMask): string {
  if (mask === 0) return "PURE";
  const table: Array<[number, string]> = [
    [CAP_ID_DB_R, "DB_R"],
    [CAP_ID_DB_W, "DB_W"],
    [CAP_ID_NET, "NET"],
    [CAP_ID_DOM, "DOM"],
    [CAP_ID_AI, "AI"],
    [CAP_ID_VISION, "VISION"],
    [CAP_ID_AUDIO, "AUDIO"],
    [CAP_ID_AUTH, "AUTH"],
    [CAP_ID_FS, "FS"],
    [CAP_ID_SYS, "SYS"],
  ];
  const names: string[] = [];
  for (const [cid, cname] of table) {
    if ((mask & capBit(cid)) !== 0) names.push(cname);
  }
  return names.length === 0 ? `0x${mask.toString(16).toUpperCase().padStart(4, "0")}` : names.join(" | ");
}

function decodeTyTokName(tyTok: number): string {
  switch (tyTok) {
    case TY_VOID:
      return "VOID";
    case TY_I64:
      return "I64";
    case TY_F64:
      return "F64";
    case TY_TRIT:
      return "TRIT";
    case TY_STR:
      return "STR";
    case TY_UUID:
      return "UUID";
    case TY_BYTES:
      return "BYTES";
    case TY_TRIT_VEC:
      return "TRIT_VEC";
    case TY_FLT_VEC:
      return "FLT_VEC";
    case TY_ARRAY:
      return "ARRAY";
    case TY_ARENA:
      return "ARENA";
    case TY_MODEL_T:
      return "MODEL_T";
    case TY_DOM:
      return "DOM";
    case TY_STRUCT:
      return "STRUCT";
    case TY_ERR:
      return "ERR";
    default:
      return "UNKNOWN_TY";
  }
}

function formatInspectOperand(op: BinOperand): string {
  switch (op.kind) {
    case "Reg":
      return `%${Math.max(0, op.value - REG_BASE)}`;
    case "Param":
      return `$${Math.max(0, op.value - PAR_BASE)}`;
    case "ImmInt":
      return op.value.toString();
    case "ImmTrit":
      return op.value > 0 ? "+1t" : op.value < 0 ? "-1t" : "0t";
    case "EnglishSpan":
      return `@ENG[${op.value.length} words]`;
    case "SchemaRef":
      return `@schema${op.value}`;
    case "FnRef":
      return `@fn${op.value}`;
    case "SkillRef":
      return `@skill${op.value}`;
    case "Label":
      return `label(${Math.max(0, op.value - 0x7ef0)})`;
    case "FieldKV":
      return `field#${op.fieldIdx}=${formatInspectOperand(op.inner)}`;
  }
}

export function formatOmniInspect(u16Stream: number[], sourcemap?: SurfaceSourceMap): string {
  const mod = decodeOmniTokenStream(u16Stream);
  const rawBytes = new Uint8Array(u16Stream.length * 2);
  for (let i = 0; i < u16Stream.length; i++) {
    rawBytes[i * 2] = u16Stream[i] & 0xff;
    rawBytes[i * 2 + 1] = (u16Stream[i] >> 8) & 0xff;
  }
  const hashHex = fnv1a64Hash(rawBytes).toString(16).toUpperCase().padStart(16, "0");
  const lines: string[] = [];
  lines.push(
    `=== OMNI BINARY STREAM INSPECTION (words=${u16Stream.length}, bytes=${rawBytes.length}, fnv1a64=0x${hashHex}) ===`,
  );
  lines.push(`schemas: ${mod.schemas.length}, experts: ${mod.experts.length}, functions: ${mod.functions.length}`);

  for (const func of mod.functions) {
    const rep = verifyBinaryFunction(func);
    lines.push("");
    lines.push(
      `fn #${func.fnIdx} -> ${decodeTyTokName(func.retTyTok)} [cap_mask=0x${func.capMask.toString(16).toUpperCase().padStart(4, "0")} (${decodeCapMaskNames(func.capMask)})] [verifier=${rep.passed ? "PASS" : "FAIL"}]`,
    );
    func.params.forEach(([pTok, pTy, pTc], pIdx) => {
      const { untrusted, capId } = unpackTc(pTc);
      lines.push(
        `  param $${pIdx} (0x${pTok.toString(16).toUpperCase().padStart(4, "0")}) : ${decodeTyTokName(pTy)} [taint=${untrusted ? "@UNTRUSTED" : "SAFE"}, cap_id=${capId}]`,
      );
    });
    func.instructions.forEach((ins, insIdx) => {
      const opName = decodeOpcodeToken(ins.opTok)?.debugLabel ?? "INVALID_OP";
      let dstPart = "";
      if (ins.destReg !== null) {
        const { untrusted, capId } = unpackTc(ins.tcTok);
        dstPart = `%${Math.max(0, ins.destReg - REG_BASE)} : ${decodeTyTokName(ins.tyTok)} [${untrusted ? "@UNTRUSTED" : "SAFE"}, cap=${capId}] = `;
      }
      const opsStr = ins.operands.map(formatInspectOperand).join(", ");
      const sp = sourcemap?.offsetToSpan.get(ins.offset);
      const locComment = sp ? ` // line ${sp[0]}, col ${sp[1]}` : "";
      lines.push(
        `  [${String(insIdx).padStart(3, "0")} @0x${ins.offset.toString(16).toUpperCase().padStart(4, "0")}] ${dstPart}${opName} (0x${ins.opTok.toString(16).toUpperCase().padStart(4, "0")}) ${opsStr}${locComment}`,
      );
    });
  }
  return lines.join("\n") + "\n";
}

export interface OmniErrorExplanation {
  code: string;
  contract: string;
  stage: string;
  summary: string;
  hint: string;
  bad: string;
  repaired: string;
  formatted: string;
}

export function explainOmniErrorCode(code: string): OmniErrorExplanation | null {
  const norm = code.trim().toUpperCase();
  const entries: Record<string, [string, string, string, string]> = {
    TAINT_ON_RET: [
      "Static Verifier (0x7F4A RET)",
      "An @untrusted parameter, @ENG string literal, or register derived from untrusted input reached return without sanitization.",
      "fn main(x: i64 @untrusted) -> i64 { return x; }",
      "#[cap(AUTH)] fn main(x: i64 @untrusted) -> i64 { let clean: i64 = auth_verify(x); return clean; }",
    ],
    TAINT_SINK: [
      "Static Verifier (Privilege Sinks)",
      "An @untrusted operand was passed directly to a security-sensitive sink (egate, db_query, db_get, kv_put, respond).",
      "#[cap(AI)] fn main(x: i64 @untrusted) -> trit { return egate(x); }",
      "#[cap(AI, AUTH)] fn main(x: i64 @untrusted) -> trit { let c: i64 = auth_verify(x); return egate(c); }",
    ],
    MISSING_CAP: [
      "Static Verifier (Capability Bitmask)",
      "A function or route executed an opcode whose required capability bit (AI, NET, DB_R, DB_W, AUTH, DOM) is absent from #[cap(...)].",
      "fn main(x: i64) -> trit { return egate(x); }",
      "#[cap(AI)] fn main(x: i64) -> trit { return egate(x); }",
    ],
    LINEAR_LEAK: [
      "Static Verifier (Linear Arena Ownership)",
      "A linear arena allocated with arena(cap) was not consumed by drop(a) before function return.",
      "fn main(x: i64) -> i64 { let a: arena = arena(4096); return x; }",
      "fn main(x: i64) -> i64 { let a: arena = arena(4096); drop(a); return x; }",
    ],
    BORROW_CONFLICT: [
      "Static Verifier (Borrow State Machine)",
      "A linear arena was dropped or mutably borrowed while an active reader borrow (borrow_r) was still live.",
      "let a: arena = arena(4096); let r: arena = borrow_r(a); drop(a); drop(r);",
      "let a: arena = arena(4096); let r: arena = borrow_r(a); drop(r); drop(a);",
    ],
    UNDEF_VAR: [
      "Surface Lowering / Macro Hygiene",
      "An identifier was referenced or reassigned before being bound with 'let <name>: <ty> = <expr>;', or a macro referenced a free variable.",
      "fn main(x: i64) -> i64 { return missing_v + 1; }",
      "fn main(x: i64) -> i64 { let missing_v: i64 = 0; return missing_v + 1; }",
    ],
    IMMUT_PARAM: [
      "Surface Lowering",
      "Function parameters ($0..=$31) are immutable SSA inputs and cannot be reassigned directly.",
      "fn main(x: i64) -> i64 { x = x + 1; return x; }",
      "fn main(x: i64) -> i64 { let x_mut: i64 = x + 1; return x_mut; }",
    ],
    BAD_TYPE: [
      "Parser / Strict Single-Form Checker",
      "An unknown type name was used, or a 'let' binding omitted its explicit ': <ty>' annotation in #[strict] mode.",
      "#[strict] fn main(a: i64) -> i64 { let b = a * 2; return b; }",
      "#[strict] fn main(a: i64) -> i64 { let b: i64 = a * 2; return b; }",
    ],
    UNKNOWN_FN: [
      "Surface Lowering",
      "Called an identifier that is neither a declared function/macro in the module nor a built-in Omni intrinsic.",
      "fn main(x: i64) -> i64 { return unknown_op(x); }",
      "fn main(x: i64) -> i64 { return iabs(x); }",
    ],
    ARITY_MISMATCH: [
      "Surface Lowering",
      "The number of arguments passed to a function, hygienic macro, or built-in intrinsic did not match its signature.",
      "fn main(x: i64) -> i64 { return iabs(x, 2); }",
      "fn main(x: i64) -> i64 { return iabs(x); }",
    ],
    UNRESOLVED_LABEL: [
      "Surface Lowering / Static Verifier",
      "A 'goto N;' or conditional branch targeted a basic-block label N (0..=15) not defined by 'label N;' in the same function.",
      "fn main(x: i64) -> i64 { goto 3; return x; }",
      "fn main(x: i64) -> i64 { goto 3; label 3; return x; }",
    ],
    CIRCULAR_INCLUDE: [
      "Include Preprocessor",
      "A cycle was detected in 'include \"...\";' directives across modules.",
      'include "self.omni"; fn main(x: i64) -> i64 { return x; }',
      'include "std/prelude.omni"; fn main(x: i64) -> i64 { return x; }',
    ],
    BUDGET_EXCEEDED: [
      "VM / Wasm AOT Runtime",
      "Execution exceeded the function's #[step_budget(N)] instruction cap or #[mem_budget(N)] byte allocation cap.",
      "#[step_budget(10)] fn main(n: i64) -> i64 { ... }",
      "#[step_budget(500000)] fn main(n: i64) -> i64 { ... }",
    ],
    CORRUPT_PACK5: [
      "@PACK5 Validator / Decoder",
      "A @PACK5 byte buffer contained a sentinel or out-of-range byte in 243..=255 (valid 5-trit radix-3 bytes are 0..=242).",
      "b = iset(b, 250); let v: trit_vec = unpack5(b, 5);",
      "b = iset(b, 242); let v: trit_vec = unpack5(b, 5);",
    ],
    AMBIGUOUS_GRAMMAR: [
      "Strict Single-Form Lexer / Parser",
      "Non-canonical surface aliases (TRIT_POS, int, tvec, untrusted_i64, mod) are forbidden in #[strict] mode.",
      "#[strict] fn main(a: int) -> trit { return TRIT_POS; }",
      "#[strict] fn main(a: i64) -> trit { return +1t; }",
    ],
    SYNTAX_ERROR: [
      "Surface Lexer / Parser",
      "Malformed expression, unbalanced delimiters, missing match_trit arm, or empty module.",
      "fn main(x: i64) -> i64 { let r = ; return r; }",
      "fn main(x: i64) -> i64 { let r: i64 = x + 1; return r; }",
    ],
  };

  const found = entries[norm];
  if (!found) return null;
  const [stage, summary, bad, repaired] = found;
  const hint = retryHintForCode(norm);
  const formatted = [
    `=== OMNI ERROR CODE EXPLANATION: error[${norm}] ===`,
    `Contract: ${OMNI_ERROR_CONTRACT_VERSION}`,
    `Stage:    ${stage}`,
    `Summary:  ${summary}`,
    `Hint:     ${hint}`,
    `Bad:      ${bad}`,
    `Repaired: ${repaired}`,
  ].join("\n");
  return { code: norm, contract: OMNI_ERROR_CONTRACT_VERSION, stage, summary, hint, bad, repaired, formatted };
}

export function listAllErrorExplanations(): OmniErrorExplanation[] {
  return Array.from(VALID_ERROR_CODES).map((c) => explainOmniErrorCode(c)!);
}

export function listAllErrorCodes(): string[] {
  return Array.from(VALID_ERROR_CODES);
}

// ==============================================================================
// MULTI-FUNCTION WEBASSEMBLY AOT EXECUTION (BEYOND SINGLE-FN RUST HOT-PATH)
// ==============================================================================

export function executeOmniSurfaceMultiFnWasm(
  src: string,
  args: bigint[],
): { value: bigint; wasmBytes: Uint8Array; functionCount: number } {
  const { stream, sourcemap } = compileOmniSurfaceToU16WithSourcemap(src);
  verifyModuleViaPhase7PureOmniKernel(stream);
  const reports = verifyU16Stream(stream);
  for (const rep of reports) {
    if (!rep.passed) {
      const first = rep.errors[0] ?? "";
      throw new Error(annotateOmniErrorWithSourcemap(first, sourcemap));
    }
  }
  const mod = decodeOmniTokenStream(stream);
  if (!isModuleMultiFnWasmCompatible(mod)) {
    throw new Error("Module uses non-numeric/ternary opcodes not supported by multi-function Wasm AOT");
  }
  const wasmBytes = compilePureU16ToWasm(stream);
  const value = executeWasmModuleI64WithBudgets(wasmBytes, args);
  return { value, wasmBytes, functionCount: mod.functions.length };
}

// ==============================================================================
// PURE-OMNI SELF-HOSTING PHASES A–E, NARROW HOST, & 18-MODULE PURE-CORE SUITE
// ==============================================================================

export const PHASE_A_PURE_OMNI_FRONTEND_KERNEL_SURFACE = fs.existsSync("spec/pure_self_host/frontend.omni")
  ? fs.readFileSync("spec/pure_self_host/frontend.omni", "utf8")
  : `#[strict]
fn omni_pure_frontend_kernel(src_len: i64, strict_flag: i64, fn_count: i64, ins_count: i64) -> i64 {
    let empty_cmp: trit = src_len <=> 0;
    let empty_err: i64 = if_pos empty_cmp { 0 } else { 1001 };
    let fn_cmp: trit = fn_count <=> 0;
    let fn_err: i64 = if_pos fn_cmp { 0 } else { 1002 };
    let ins_cmp: trit = ins_count <=> 192;
    let budget_err: i64 = if_pos ins_cmp { 1003 } else { 0 };
    let e1: i64 = imax(empty_err, fn_err);
    let e2: i64 = imax(e1, budget_err);
    return e2;
}`;

export const PHASE_B_PURE_OMNI_DRIVER_SURFACE = fs.existsSync("spec/pure_self_host/driver.omni")
  ? fs.readFileSync("spec/pure_self_host/driver.omni", "utf8")
  : `#[strict]
fn omni_pure_driver_kernel(frontend_status: i64, verifier_status: i64, exec_status: i64) -> i64 {
    let fe_cmp: trit = frontend_status <=> 0;
    let ver_cmp: trit = verifier_status <=> 0;
    let ex_cmp: trit = exec_status <=> 0;
    let after_exec: i64 = if_pos ex_cmp { exec_status } else { 0 };
    let after_ver: i64 = if_pos ver_cmp { verifier_status } else { after_exec };
    let final_status: i64 = if_pos fe_cmp { frontend_status } else { after_ver };
    return final_status;
}`;

export const PHASE_C_PURE_OMNI_HOST_RUNTIME_SURFACE = fs.existsSync("spec/pure_self_host/host_runtime.omni")
  ? fs.readFileSync("spec/pure_self_host/host_runtime.omni", "utf8")
  : `include "std/http.omni";
#[strict]
fn omni_pure_host_dispatch(method_code: i64, status: i64, body_val: i64) -> i64 {
    let ok_trit: trit = std_http_is_success(status);
    let final_status: i64 = sel_trit(ok_trit, 500, 400, status);
    let packed: i64 = std_http_pack_response(final_status, body_val);
    return packed;
}`;

export const PHASE_D_PURE_OMNI_SELFHOST_COMPILER_SURFACE = fs.existsSync(
  "spec/pure_self_host/omni_selfhost_compiler.omni",
)
  ? fs.readFileSync("spec/pure_self_host/omni_selfhost_compiler.omni", "utf8")
  : `include "std/prelude.omni";
include "std/trit.omni";
#[strict]
fn omni_selfhost_compiler(opcode_class: i64, reg_count: i64) -> i64 {
    let clamped_regs: i64 = std_clamp(reg_count, 1, 192);
    let class_cmp: trit = opcode_class <=> 2;
    let class_weight: i64 = match_trit class_cmp {
        neg => 10,
        zero => 20,
        pos => 30
    };
    let consensus: trit = std_consensus3(+1t, class_cmp, +1t);
    let bonus: i64 = std_trit_mux(consensus, -5, 0, 5);
    return clamped_regs * class_weight + bonus;
}`;

export const PURE_OMNI_CORE_SUITE_MODULES: string[] = [
  "std/prelude.omni",
  "std/math.omni",
  "std/trit.omni",
  "std/http.omni",
  "std/pack5.omni",
  "std/decision.omni",
  "std/resource.omni",
  "std/string.omni",
  "pure/compiler.omni",
  "pure/verifier.omni",
  "pure/host_runtime.omni",
  "pure/tooling.omni",
  "pure/selfhost_suite.omni",
  "pure/programs/ternary_neural_router.omni",
  "pure/programs/kv_ledger_service.omni",
  "pure/programs/policy_consensus_engine.omni",
  "pure/programs/compiler_self_audit.omni",
  "pure/programs/pack5_decision_controller.omni",
  "pure/programs/arena_audit_pipeline.omni",
  "pure/programs/struct_string_pipeline.omni",
];

export function pureOmniSuiteModuleCounts(): {
  stdlibModules: number;
  coreModules: number;
  realPrograms: number;
  totalPureModules: number;
} {
  let stdlibModules = 0;
  let coreModules = 0;
  let realPrograms = 0;
  for (const m of PURE_OMNI_CORE_SUITE_MODULES) {
    if (m.startsWith("std/")) stdlibModules += 1;
    else if (m.startsWith("pure/programs/")) realPrograms += 1;
    else if (m.startsWith("pure/")) coreModules += 1;
  }
  return {
    stdlibModules,
    coreModules,
    realPrograms,
    totalPureModules: stdlibModules + coreModules + realPrograms,
  };
}

let cachedPhaseAStream: number[] | null = null;
let cachedPhaseBStream: number[] | null = null;
let cachedPhaseCStream: number[] | null = null;

export function buildPhaseAPureOmniFrontendKernelStream(): number[] {
  if (!cachedPhaseAStream) {
    cachedPhaseAStream = compileOmniSurfaceToU16(PHASE_A_PURE_OMNI_FRONTEND_KERNEL_SURFACE);
  }
  return [...cachedPhaseAStream];
}

export function buildPhaseBPureOmniDriverStream(): number[] {
  if (!cachedPhaseBStream) {
    cachedPhaseBStream = compileOmniSurfaceToU16(PHASE_B_PURE_OMNI_DRIVER_SURFACE);
  }
  return [...cachedPhaseBStream];
}

export function buildPhaseCPureOmniHostStream(): number[] {
  if (!cachedPhaseCStream) {
    cachedPhaseCStream = compileOmniSurfaceToU16(PHASE_C_PURE_OMNI_HOST_RUNTIME_SURFACE);
  }
  return [...cachedPhaseCStream];
}

export function compileOmniSurfaceViaPureOmniFrontendWithSourcemap(
  src: string,
  strict: boolean,
): { stream: number[]; sourcemap: SurfaceSourceMap } {
  const feStream = buildPhaseAPureOmniFrontendKernelStream();
  const feMod = decodeOmniTokenStream(feStream);
  const vm = new OmniBinaryVm();
  const statusVal = vm.executeFunction(feMod, 0, [
    { kind: "Int", value: BigInt(src.trim().length) },
    { kind: "Int", value: 0n },
    { kind: "Int", value: 0n },
  ]);
  if (statusVal.kind === "Int" && statusVal.value !== 0n) {
    throw new Error("error[SYNTAX_ERROR] line 1, col 1: Surface program is empty");
  }
  return compileOmniSurfaceToU16WithSourcemap(src, strict);
}

export function compileOmniSurfaceViaPureOmniFrontend(src: string, strict: boolean): number[] {
  return compileOmniSurfaceViaPureOmniFrontendWithSourcemap(src, strict).stream;
}

export function executeOmniSurfaceViaPureOmniDriver(
  src: string,
  args: bigint[],
): { val: OmniValue; backend: string } {
  const { stream, sourcemap } = compileOmniSurfaceViaPureOmniFrontendWithSourcemap(src, false);
  verifyModuleViaPhase7PureOmniKernel(stream);
  const drvStream = buildPhaseBPureOmniDriverStream();
  const drvMod = decodeOmniTokenStream(drvStream);
  const vm = new OmniBinaryVm();
  const drvStatus = vm.executeFunction(drvMod, 0, [
    { kind: "Int", value: 0n },
    { kind: "Int", value: 0n },
    { kind: "Int", value: 0n },
  ]);
  if (drvStatus.kind === "Int" && drvStatus.value !== 0n) {
    throw new Error(annotateOmniErrorWithSourcemap(`Pure driver error ${drvStatus.value}`, sourcemap));
  }
  const detail = executeOmniSurfaceAotFirst(
    src,
    args.map((v): OmniValue => ({ kind: "Int", value: v })),
  );
  return { val: detail.value, backend: detail.backend };
}

export function runPureOmniDriverJson(
  src: string,
  args: bigint[],
  strict: boolean,
): { passed: boolean; json: string } {
  const rep = runOmniAgentDriver(
    src,
    args.map((v): OmniValue => ({ kind: "Int", value: v })),
    strict,
    6,
  );
  return { passed: rep.ok, json: JSON.stringify(rep) };
}

// ==============================================================================
// NARROW HOST BOUNDARY (OmniNarrowHost & SimulatedNarrowHostAdapter)
// ==============================================================================

function checkNarrowHostCap(capMask: CapMask, requiredBit: CapMask, opName: string): void {
  if ((capMask & requiredBit) === 0) {
    throw new Error(
      `error[MISSING_CAP] line 1, col 1: Narrow host syscall '${opName}' requires capability bit 0x${requiredBit.toString(16).toUpperCase().padStart(4, "0")}`,
    );
  }
}

export interface OmniNarrowHost {
  hostTcpListen(capMask: CapMask, bindAddr: string): [number, string];
  hostTcpAccept(capMask: CapMask, listenerHandle: number): number;
  hostTcpRead(capMask: CapMask, connHandle: number, maxBytes: number): Uint8Array;
  hostTcpWrite(capMask: CapMask, connHandle: number, data: Uint8Array): number;
  hostTcpClose(capMask: CapMask, handle: number): void;
  hostFileOpen(capMask: CapMask, filePath: string, write: boolean): number;
  hostFileRead(capMask: CapMask, fileHandle: number): Uint8Array;
  hostFileWrite(capMask: CapMask, fileHandle: number, data: Uint8Array): number;
  hostFileClose(capMask: CapMask, fileHandle: number): void;
  hostTimeNs(capMask: CapMask): bigint;
}

export class SimulatedNarrowHostAdapter implements OmniNarrowHost {
  private nextHandle = 100;
  public boundAddr: string | null = null;
  public pendingRequests: Uint8Array[] = [];
  public writtenResponses: Uint8Array[] = [];
  public virtualFiles: Map<string, Uint8Array> = new Map();
  private openFiles: Map<number, { path: string; write: boolean; data: Uint8Array }> = new Map();
  private clockNs = 1_000_000n;

  public enqueueHttpRequest(rawWire: string): void {
    this.pendingRequests.push(new TextEncoder().encode(rawWire));
  }

  private allocHandle(): number {
    const h = this.nextHandle;
    this.nextHandle += 1;
    return h;
  }

  public hostTcpListen(capMask: CapMask, bindAddr: string): [number, string] {
    checkNarrowHostCap(capMask, capBit(CAP_ID_NET), "host_tcp_listen");
    const assigned = bindAddr.endsWith(":0") ? "127.0.0.1:40080" : bindAddr;
    this.boundAddr = assigned;
    return [this.allocHandle(), assigned];
  }

  public hostTcpAccept(capMask: CapMask, _listenerHandle: number): number {
    checkNarrowHostCap(capMask, capBit(CAP_ID_NET), "host_tcp_accept");
    return this.allocHandle();
  }

  public hostTcpRead(capMask: CapMask, _connHandle: number, _maxBytes: number): Uint8Array {
    checkNarrowHostCap(capMask, capBit(CAP_ID_NET), "host_tcp_read");
    const req = this.pendingRequests.shift();
    if (!req) throw new Error("No pending simulated TCP request");
    return req;
  }

  public hostTcpWrite(capMask: CapMask, _connHandle: number, data: Uint8Array): number {
    checkNarrowHostCap(capMask, capBit(CAP_ID_NET), "host_tcp_write");
    this.writtenResponses.push(new Uint8Array(data));
    return data.length;
  }

  public hostTcpClose(capMask: CapMask, _handle: number): void {
    checkNarrowHostCap(capMask, capBit(CAP_ID_NET), "host_tcp_close");
  }

  public hostFileOpen(capMask: CapMask, filePath: string, write: boolean): number {
    if (write) {
      if ((capMask & (capBit(CAP_ID_DB_W) | capBit(CAP_ID_FS))) === 0) {
        throw new Error(
          "error[MISSING_CAP] line 1, col 1: Narrow host syscall 'host_file_open(write)' requires CAP_ID_DB_W or CAP_ID_FS",
        );
      }
    } else if ((capMask & (capBit(CAP_ID_DB_R) | capBit(CAP_ID_FS))) === 0) {
      throw new Error(
        "error[MISSING_CAP] line 1, col 1: Narrow host syscall 'host_file_open(read)' requires CAP_ID_DB_R or CAP_ID_FS",
      );
    }
    const existing = this.virtualFiles.get(filePath) ?? new Uint8Array(0);
    const fh = this.allocHandle();
    this.openFiles.set(fh, { path: filePath, write, data: new Uint8Array(existing) });
    return fh;
  }

  public hostFileRead(capMask: CapMask, fileHandle: number): Uint8Array {
    if ((capMask & (capBit(CAP_ID_DB_R) | capBit(CAP_ID_FS))) === 0) {
      throw new Error(
        "error[MISSING_CAP] line 1, col 1: Narrow host syscall 'host_file_read' requires CAP_ID_DB_R or CAP_ID_FS",
      );
    }
    const entry = this.openFiles.get(fileHandle);
    if (!entry) throw new Error(`Invalid simulated file handle ${fileHandle}`);
    return new Uint8Array(entry.data);
  }

  public hostFileWrite(capMask: CapMask, fileHandle: number, data: Uint8Array): number {
    if ((capMask & (capBit(CAP_ID_DB_W) | capBit(CAP_ID_FS))) === 0) {
      throw new Error(
        "error[MISSING_CAP] line 1, col 1: Narrow host syscall 'host_file_write' requires CAP_ID_DB_W or CAP_ID_FS",
      );
    }
    const entry = this.openFiles.get(fileHandle);
    if (!entry) throw new Error(`Invalid simulated file handle ${fileHandle}`);
    if (!entry.write) throw new Error(`Simulated file handle ${fileHandle} not opened for write`);
    entry.data = new Uint8Array(data);
    return data.length;
  }

  public hostFileClose(capMask: CapMask, fileHandle: number): void {
    if ((capMask & (capBit(CAP_ID_DB_R) | capBit(CAP_ID_DB_W) | capBit(CAP_ID_FS))) === 0) {
      throw new Error(
        "error[MISSING_CAP] line 1, col 1: Narrow host syscall 'host_file_close' requires DB/FS capability",
      );
    }
    const entry = this.openFiles.get(fileHandle);
    this.openFiles.delete(fileHandle);
    if (entry && entry.write) {
      this.virtualFiles.set(entry.path, entry.data);
    }
  }

  public hostTimeNs(capMask: CapMask): bigint {
    checkNarrowHostCap(capMask, capBit(CAP_ID_SYS), "host_time_ns");
    this.clockNs += 250_000n;
    return this.clockNs;
  }
}

export function loadOmniKvViaNarrowHost(
  host: OmniNarrowHost,
  capMask: CapMask,
  filePath: string,
): Map<bigint, bigint> {
  const fh = host.hostFileOpen(capMask, filePath, false);
  const rawBytes = host.hostFileRead(capMask, fh);
  host.hostFileClose(capMask, fh);
  const raw = new TextDecoder().decode(rawBytes);
  const kv = new Map<bigint, bigint>();
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx > 0) {
      const kStr = trimmed.slice(0, eqIdx).trim();
      const vStr = trimmed.slice(eqIdx + 1).trim();
      if (/^-?\d+$/.test(kStr) && /^-?\d+$/.test(vStr)) {
        kv.set(BigInt(kStr), BigInt(vStr));
      }
    }
  }
  return kv;
}

export function saveOmniKvViaNarrowHost(
  host: OmniNarrowHost,
  capMask: CapMask,
  filePath: string,
  kv: Map<bigint, bigint>,
): void {
  const lines: string[] = [];
  for (const [k, v] of kv.entries()) {
    lines.push(`${k}=${v}`);
  }
  const fh = host.hostFileOpen(capMask, filePath, true);
  host.hostFileWrite(capMask, fh, new TextEncoder().encode(lines.join("\n") + "\n"));
  host.hostFileClose(capMask, fh);
}

export function serveOmniHttpViaNarrowHost(
  host: OmniNarrowHost,
  src: string,
  bindAddr: string,
  dbPath: string | null,
  numRequests: number,
): { localAddr: string; responses: OmniHttpResponse[]; kvStore: Map<bigint, bigint> } {
  const routes = extractOmniRoutes(src);
  if (routes.length === 0) {
    throw new Error("error[SYNTAX_ERROR] line 1, col 1: Surface program contains no 'route' handlers");
  }
  const { stream, sourcemap } = compileOmniSurfaceViaPureOmniFrontendWithSourcemap(src, false);
  verifyModuleViaPhase7PureOmniKernel(stream);
  for (const rep of verifyU16Stream(stream)) {
    if (!rep.passed) {
      const first = rep.errors[0] ?? "";
      throw new Error(annotateOmniErrorWithSourcemap(first, sourcemap));
    }
  }
  const mod = decodeOmniTokenStream(stream);
  const combinedCaps: CapMask = mod.functions.reduce((acc, f) => acc | f.capMask, 0);

  const kvStore = dbPath
    ? loadOmniKvViaNarrowHost(host, combinedCaps | capBit(CAP_ID_DB_R), dbPath)
    : new Map<bigint, bigint>();

  const [listenerH, localAddr] = host.hostTcpListen(combinedCaps, bindAddr);
  const responses: OmniHttpResponse[] = [];
  for (let i = 0; i < numRequests; i++) {
    const connH = host.hostTcpAccept(combinedCaps, listenerH);
    const reqBytes = host.hostTcpRead(combinedCaps, connH, 4096);
    const reqStr = new TextDecoder().decode(reqBytes);
    const firstLine = reqStr.split(/\r?\n/)[0] ?? "GET / HTTP/1.1";
    const parts = firstLine.trim().split(/\s+/);
    const method = parts[0] ?? "GET";
    const reqPath = parts[1] ?? "/";
    const resp = handleOmniHttpRequest(src, method, reqPath, kvStore);
    if (dbPath) {
      saveOmniKvViaNarrowHost(host, combinedCaps | capBit(CAP_ID_DB_W), dbPath, kvStore);
    }
    const wire = new TextEncoder().encode(
      `HTTP/1.1 ${resp.status} OK\r\nContent-Type: ${resp.contentType}\r\nContent-Length: ${resp.body.length}\r\nConnection: close\r\n\r\n${resp.body}`,
    );
    host.hostTcpWrite(combinedCaps, connH, wire);
    host.hostTcpClose(combinedCaps, connH);
    responses.push(resp);
  }
  host.hostTcpClose(combinedCaps, listenerH);
  return { localAddr, responses, kvStore };
}

export function serveOmniHttpViaPureOmniHost(
  src: string,
  method: string,
  requestPath: string,
  kvStore: Map<bigint, bigint>,
): OmniHttpResponse {
  const hostStream = buildPhaseCPureOmniHostStream();
  const hostMod = decodeOmniTokenStream(hostStream);
  const vm = new OmniBinaryVm();
  const resp = handleOmniHttpRequest(src, method, requestPath, kvStore);
  const methodCode = method.toUpperCase() === "POST" ? 2n : method.toUpperCase() === "GET" ? 1n : 0n;
  vm.executeFunction(hostMod, 0, [
    { kind: "Int", value: methodCode },
    { kind: "Int", value: BigInt(resp.status) },
    { kind: "Int", value: 1n },
  ]);
  return resp;
}

export function buildPureOmniCoreSuiteStream(): {
  gen1Stream: number[];
  gen2Stream: number[];
  fnv1a64Hex: string;
} {
  const gen1Words: number[] = [];
  const gen2Words: number[] = [];

  for (const modName of PURE_OMNI_CORE_SUITE_MODULES) {
    const src = lookupOmniStdlibModule(modName);
    if (!src) throw new Error(`Missing Pure-Omni core suite module '${modName}'`);
    const g1 = compileOmniSurfaceToU16(src);
    const g2 = compileOmniSurfaceViaPureOmniFrontend(src, true);
    if (g1.length !== g2.length || g1.some((w, i) => w !== g2[i])) {
      throw new Error(`Pure-Omni Core Suite Gen-1 != Gen-2 mismatch in module '${modName}'`);
    }
    gen1Words.push(...g1);
    gen2Words.push(...g2);
  }

  const fnv1a64Hex = formatFnv1a64Hex(u16SliceToLeBytes(gen1Words));
  return { gen1Stream: gen1Words, gen2Stream: gen2Words, fnv1a64Hex };
}

export interface OmniPhaseDBootstrapReport {
  gen1Words: number;
  gen2Words: number;
  bcEqual: boolean;
  bcFnv1a64: string;
  elfBytes: number;
  elfEqual: boolean;
  elfFnv1a64: string;
  stage1Words: number;
  stage1BcFnv1a64: string;
  stage1ElfBytes: number;
  stage1ElfEqual: boolean;
  stage1ElfFnv1a64: string;
  suiteWords: number;
  suiteEqual: boolean;
  suiteFnv1a64: string;
  stdlibModules: number;
  coreModules: number;
  realPrograms: number;
  totalPureModules: number;
}

let cachedPhaseDReport: OmniPhaseDBootstrapReport | null = null;

export function runPhaseDPureOmniSelfhostBootstrap(): OmniPhaseDBootstrapReport {
  if (cachedPhaseDReport) return { ...cachedPhaseDReport };

  const gen1Bc = compileOmniSurfaceToU16(PHASE_D_PURE_OMNI_SELFHOST_COMPILER_SURFACE);
  const gen2Bc = compileOmniSurfaceViaPureOmniFrontend(PHASE_D_PURE_OMNI_SELFHOST_COMPILER_SURFACE, true);
  const bcEqual = gen1Bc.length === gen2Bc.length && gen1Bc.every((w, i) => w === gen2Bc[i]);
  const bcFnv1a64 = formatFnv1a64Hex(u16SliceToLeBytes(gen1Bc));

  const gen1Elf = compilePureU16ToX86_64ElfWithArgs(gen1Bc, 10n, 3n, 0n);
  const gen2Elf = compileOmniSurfaceViaStage1ToElf(gen2Bc, 10n, 3n, 0n);
  const elfEqual =
    gen1Elf.length === gen2Elf.length && gen1Elf.every((b, i) => b === gen2Elf[i]);
  const elfFnv1a64 = formatFnv1a64Hex(gen1Elf);

  const stage1Stream = buildStep7Stage1CompilerStream();
  const stage1BcFnv1a64 = formatFnv1a64Hex(u16SliceToLeBytes(stage1Stream));
  const stage1DirectElf = compilePureU16ToX86_64ElfWithArgs(stage1Stream, 0n, 0n, 0n);
  const stage2SelfElf = compileOmniSurfaceViaStage1ToElf(stage1Stream, 0n, 0n, 0n);
  const stage1ElfEqual =
    stage1DirectElf.length === stage2SelfElf.length &&
    stage1DirectElf.every((b, i) => b === stage2SelfElf[i]);
  const stage1ElfFnv1a64 = formatFnv1a64Hex(stage1DirectElf);

  const { gen1Stream, gen2Stream, fnv1a64Hex: suiteFnv1a64 } = buildPureOmniCoreSuiteStream();
  const suiteEqual =
    gen1Stream.length === gen2Stream.length && gen1Stream.every((w, i) => w === gen2Stream[i]);
  const counts = pureOmniSuiteModuleCounts();

  cachedPhaseDReport = {
    gen1Words: gen1Bc.length,
    gen2Words: gen2Bc.length,
    bcEqual,
    bcFnv1a64,
    elfBytes: gen1Elf.length,
    elfEqual,
    elfFnv1a64,
    stage1Words: stage1Stream.length,
    stage1BcFnv1a64,
    stage1ElfBytes: stage1DirectElf.length,
    stage1ElfEqual,
    stage1ElfFnv1a64,
    suiteWords: gen1Stream.length,
    suiteEqual,
    suiteFnv1a64,
    ...counts,
  };
  return { ...cachedPhaseDReport };
}

export { OMNI_ACCEPTANCE_SUITE_VERSION, OMNI_ERROR_CONTRACT_VERSION, OMNI_HARNESS_CONTRACT_VERSION, OMNI_REPAIR_STRATEGIES_VERSION };
export { assembleOmniTextToU16 } from "./assembler.ts";
export { formatNonHumanTokenStream } from "./ir.ts";
export type { OmniValue } from "./vm.ts";

