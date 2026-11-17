// Omni VM — deterministic executing virtual machine with full control flow
// (LABEL, JMP, BR_TRIT, BR_POS, BR_NEG), balanced ternary, wrapping 64-bit integer ALU,
// byte/u16 buffers, ternary decision hooks (EGATE/TLOGIC), inter-function CALL,
// relational tables, and configurable step_limit and memory_budget_bytes.

import {
  clampTrit,
  fnv1a64Hash,
  pack5ByteLen,
  pack5Decode,
  pack5Encode,
  pack5GetTrit,
  pack5SumDirect,
  pack5TdotDirect,
  tand,
  tmaj,
  tnot,
  tor,
  txor,
} from "./isa.ts";
import { BinModule, BinOperand, extractLabelId } from "./ir.ts";
import { tokenToSlot, verifyBinaryFunction } from "./verifier.ts";

export type OmniValue =
  | { kind: "Void" }
  | { kind: "Int"; value: bigint }
  | { kind: "Trit"; value: number }
  | { kind: "Str"; value: string }
  | { kind: "Bytes"; value: number[] }
  | { kind: "U16Array"; value: number[] }
  | { kind: "TritVec"; value: number[] }
  | { kind: "Struct"; fields: Map<number, OmniValue> }
  | { kind: "Arena"; capacity: number; active: boolean }
  | { kind: "SkillModel"; skillId: number; active: boolean }
  | { kind: "DomNode"; tagId: number; children: OmniValue[] };

export const DEFAULT_VM_STEP_LIMIT = 50_000_000;
export const DEFAULT_VM_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024;
export const MAX_VM_CALL_DEPTH = 64;

const I64_MIN = -(1n << 63n);

function i64(n: bigint): bigint {
  return BigInt.asIntN(64, n);
}

function u64(n: bigint): bigint {
  return BigInt.asUintN(64, n);
}

export class OmniBinaryVm {
  public tables: Map<number, Array<Map<number, OmniValue>>> = new Map();
  public kvStore: Map<bigint, bigint> = new Map();
  public reqParams: Map<bigint, bigint> = new Map();
  public lastHttpResponse: { status: number; contentType: string; body: string } | null = null;
  public stepLimit: number;
  public stepsExecuted = 0;
  public memoryBudgetBytes: number;
  public allocatedBytes = 0;
  public callDepth = 0;

  constructor(
    stepLimit = DEFAULT_VM_STEP_LIMIT,
    memoryBudgetBytes = DEFAULT_VM_MEMORY_BUDGET_BYTES,
  ) {
    this.stepLimit = stepLimit;
    this.memoryBudgetBytes = memoryBudgetBytes;
  }

  public clearTables(): void {
    this.tables.clear();
    this.kvStore.clear();
    this.reqParams.clear();
    this.lastHttpResponse = null;
    this.stepsExecuted = 0;
    this.allocatedBytes = 0;
    this.callDepth = 0;
  }

  private chargeMemory(bytes: number): void {
    const nextAlloc = this.allocatedBytes + bytes;
    if (nextAlloc > this.memoryBudgetBytes) {
      throw new Error(
        `error[BUDGET_EXCEEDED]: VM memory budget exceeded (${nextAlloc} > ${this.memoryBudgetBytes} bytes)`,
      );
    }
    this.allocatedBytes = nextAlloc;
  }

  private evalOp(op: BinOperand, regs: OmniValue[]): OmniValue {
    switch (op.kind) {
      case "Reg":
      case "Param": {
        const s = tokenToSlot(op.value);
        return s !== null ? regs[s] : { kind: "Void" };
      }
      case "ImmInt":
        return { kind: "Int", value: i64(op.value) };
      case "ImmTrit":
        return { kind: "Trit", value: op.value };
      case "EnglishSpan": {
        const bytes = Uint8Array.from(op.value.map((u) => u & 0xff));
        return { kind: "Str", value: new TextDecoder().decode(bytes) };
      }
      default:
        return { kind: "Void" };
    }
  }

  private evalAsI64(op: BinOperand, regs: OmniValue[]): bigint | null {
    const v = this.evalOp(op, regs);
    if (v.kind === "Int") return v.value;
    if (v.kind === "Trit") return BigInt(v.value);
    return null;
  }

  private evalAsTrit(op: BinOperand, regs: OmniValue[]): number | null {
    const v = this.evalOp(op, regs);
    if (v.kind === "Trit") return clampTrit(v.value);
    if (v.kind === "Int") {
      if (v.value < -1n) return -1;
      if (v.value > 1n) return 1;
      return Number(v.value);
    }
    return null;
  }

  public executeFunction(module: BinModule, fnIdx: number, args: OmniValue[]): OmniValue {
    const func = module.functions.find((f) => f.fnIdx === fnIdx);
    if (!func) {
      throw new Error(`Function index ${fnIdx} not found`);
    }
    const rep = verifyBinaryFunction(func);
    if (!rep.passed) {
      throw new Error(`Binary verification failed: ${JSON.stringify(rep.errors)}`);
    }

    const regs: OmniValue[] = Array.from({ length: 224 }, () => ({ kind: "Void" }));
    for (let i = 0; i < func.params.length; i++) {
      const [parTok] = func.params[i];
      const slot = tokenToSlot(parTok);
      if (slot !== null && i < args.length) {
        regs[slot] = args[i];
      }
    }

    const labelPcs = new Int32Array(256).fill(-1);
    for (let idx = 0; idx < func.instructions.length; idx++) {
      const cand = func.instructions[idx];
      if (cand.opTok === 0x7f40 && cand.operands.length > 0) {
        const lid = extractLabelId(cand.operands[0]);
        if (lid !== null && lid < 256) {
          labelPcs[lid] = idx + 1;
        }
      }
    }

    const findLabelPc = (targetLbl: number): number | null => {
      if (targetLbl < 256 && labelPcs[targetLbl] !== -1) {
        return labelPcs[targetLbl];
      }
      const pos = func.instructions.findIndex(
        (cand) => cand.opTok === 0x7f40 && cand.operands.length > 0 && extractLabelId(cand.operands[0]) === targetLbl,
      );
      return pos >= 0 ? pos + 1 : null;
    };

    let pc = 0;
    while (pc < func.instructions.length) {
      this.stepsExecuted += 1;
      if (this.stepsExecuted > this.stepLimit) {
        throw new Error(
          `error[BUDGET_EXCEEDED]: VM fuel/step budget exceeded (${this.stepsExecuted} > ${this.stepLimit})`,
        );
      }

      const ins = func.instructions[pc];

      if (ins.opTok === 0x7f00 || ins.opTok === 0x7f40 || ins.opTok === 0x7f46) {
        pc += 1;
        continue;
      }

      if (ins.opTok === 0x7f41) {
        const targetLbl = ins.operands[0] ? extractLabelId(ins.operands[0]) : null;
        if (targetLbl !== null) {
          const nextPc = findLabelPc(targetLbl);
          if (nextPc !== null) {
            pc = nextPc;
            continue;
          }
        }
        pc += 1;
        continue;
      }

      if (ins.opTok === 0x7f42) {
        const c = ins.operands[0] ? this.evalAsTrit(ins.operands[0], regs) : null;
        const lNeg = ins.operands[1] ? extractLabelId(ins.operands[1]) : null;
        const lZero = ins.operands[2] ? extractLabelId(ins.operands[2]) : null;
        const lPos = ins.operands[3] ? extractLabelId(ins.operands[3]) : null;
        if (c !== null && lNeg !== null && lZero !== null && lPos !== null) {
          const chosen = c < 0 ? lNeg : c === 0 ? lZero : lPos;
          const nextPc = findLabelPc(chosen);
          if (nextPc !== null) {
            pc = nextPc;
            continue;
          }
        }
        pc += 1;
        continue;
      }

      if (ins.opTok === 0x7f43 || ins.opTok === 0x7f44) {
        const want = ins.opTok === 0x7f43 ? 1 : -1;
        const c = ins.operands[0] ? this.evalAsTrit(ins.operands[0], regs) : null;
        const targetLbl = ins.operands[1] ? extractLabelId(ins.operands[1]) : null;
        if (c !== null && targetLbl !== null && c === want) {
          const nextPc = findLabelPc(targetLbl);
          if (nextPc !== null) {
            pc = nextPc;
            continue;
          }
        }
        pc += 1;
        continue;
      }

      if (ins.opTok === 0x7f05) {
        const op0 = ins.operands[0];
        if (op0 && op0.kind === "Reg") {
          const slot = tokenToSlot(op0.value);
          if (slot !== null) {
            const cur = regs[slot];
            if (cur.kind === "Arena" && cur.active) {
              this.allocatedBytes = Math.max(0, this.allocatedBytes - cur.capacity);
            }
            regs[slot] = { kind: "Void" };
          }
        }
        pc += 1;
        continue;
      }

      if (ins.opTok === 0x7f4a) {
        return ins.operands[0] ? this.evalOp(ins.operands[0], regs) : { kind: "Void" };
      }

      let outVal: OmniValue = { kind: "Void" };
      switch (ins.opTok) {
        case 0x7f01: {
          const v = this.evalOp(ins.operands[0], regs);
          if (v.kind === "Int") {
            const req = v.value > 0n ? Number(v.value) : 0;
            this.chargeMemory(req);
            outVal = { kind: "Arena", capacity: req, active: true };
          }
          break;
        }
        case 0x7f02:
        case 0x7f03: {
          const v = this.evalOp(ins.operands[0], regs);
          outVal = v.kind === "Arena" ? { kind: "Arena", capacity: 0, active: true } : v;
          break;
        }
        case 0x7f04:
        case 0x7f06:
        case 0x7f09:
          outVal = this.evalOp(ins.operands[0], regs);
          break;
        case 0x7f08: {
          const v = this.evalOp(ins.operands[0], regs);
          if (v.kind === "Str") {
            this.chargeMemory(Math.max(1, v.value.length));
          }
          outVal = v;
          break;
        }
        case 0x7f0a: {
          // SNEW: allocate a new Struct (with optional alternating fieldKey, fieldVal pairs)
          this.chargeMemory(64);
          const fields = new Map<number, OmniValue>();
          for (let j = 0; j + 1 < ins.operands.length; j += 2) {
            const k = this.evalAsI64(ins.operands[j], regs);
            const v = this.evalOp(ins.operands[j + 1], regs);
            if (k !== null) {
              fields.set(Number(BigInt.asIntN(32, k)), v);
            }
          }
          outVal = { kind: "Struct", fields };
          break;
        }
        case 0x7f0b: {
          // FGET <struct_reg>, <field_key>
          const sVal = this.evalOp(ins.operands[0], regs);
          const kVal = this.evalAsI64(ins.operands[1], regs);
          if (sVal.kind === "Struct" && kVal !== null) {
            const found = sVal.fields.get(Number(BigInt.asIntN(32, kVal)));
            outVal = found ?? { kind: "Int", value: 0n };
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f0c: {
          // FSET <struct_reg>, <field_key>, <val>
          const sVal = this.evalOp(ins.operands[0], regs);
          const kVal = this.evalAsI64(ins.operands[1], regs);
          const vVal = this.evalOp(ins.operands[2], regs);
          const nextFields = sVal.kind === "Struct" ? new Map(sVal.fields) : new Map<number, OmniValue>();
          if (kVal !== null) {
            this.chargeMemory(16);
            nextFields.set(Number(BigInt.asIntN(32, kVal)), vVal);
          }
          outVal = { kind: "Struct", fields: nextFields };
          break;
        }
        case 0x7f0d:
          this.chargeMemory(64);
          outVal = { kind: "Bytes", value: [] };
          break;
        case 0x7f0e: {
          const arrVal = this.evalOp(ins.operands[0], regs);
          const idxVal = this.evalOp(ins.operands[1], regs);
          if (idxVal.kind === "Int" && idxVal.value >= 0n) {
            const i = Number(idxVal.value);
            if (arrVal.kind === "U16Array" && i < arrVal.value.length) {
              outVal = { kind: "Int", value: BigInt(arrVal.value[i]) };
            } else if (arrVal.kind === "Bytes" && i < arrVal.value.length) {
              outVal = { kind: "Int", value: BigInt(arrVal.value[i]) };
            } else if (arrVal.kind === "TritVec" && i < arrVal.value.length) {
              const packed = pack5Encode(arrVal.value);
              outVal = { kind: "Int", value: BigInt(pack5GetTrit(packed, i)) };
            } else {
              outVal = { kind: "Int", value: 0n };
            }
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f0f: {
          const base = this.evalOp(ins.operands[0], regs);
          const inPlace =
            ins.destReg !== null &&
            ins.operands[0]?.kind === "Reg" &&
            ins.destReg === ins.operands[0].value &&
            base.kind === "Bytes";
          const buf = base.kind === "Bytes" ? (inPlace ? base.value : [...base.value]) : [];
          if (
            ins.operands.length === 3 &&
            ins.operands[1].kind === "ImmInt" &&
            ins.operands[1].value === -4n
          ) {
            this.chargeMemory(4);
            const v = this.evalAsI64(ins.operands[2], regs);
            if (v !== null) {
              const n = Number(BigInt.asIntN(32, v));
              buf.push(n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff, (n >> 24) & 0xff);
            }
          } else if (
            ins.operands.length === 3 &&
            ins.operands[1].kind === "ImmInt" &&
            ins.operands[1].value === -8n
          ) {
            this.chargeMemory(8);
            const v = this.evalAsI64(ins.operands[2], regs);
            if (v !== null) {
              const u = u64(v);
              for (let k = 0; k < 8; k++) {
                buf.push(Number((u >> BigInt(k * 8)) & 0xffn));
              }
            }
          } else if (
            ins.operands.length === 4 &&
            ins.operands[1].kind === "ImmInt" &&
            ins.operands[1].value === -9n
          ) {
            const off = this.evalAsI64(ins.operands[2], regs);
            const v = this.evalAsI64(ins.operands[3], regs);
            if (off !== null && v !== null && off >= 0n) {
              const uOff = Number(off);
              if (uOff < buf.length) {
                buf[uOff] = Number(v & 0xffffffffn);
              }
            }
          } else {
            this.chargeMemory(Math.max(0, ins.operands.length - 1));
            for (let j = 1; j < ins.operands.length; j++) {
              const bVal = this.evalAsI64(ins.operands[j], regs) ?? 0n;
              buf.push(Number(bVal & 0xffn));
            }
          }
          outVal = { kind: "Bytes", value: buf };
          break;
        }
        case 0x7f10: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? i64(a + b) : 0n };
          break;
        }
        case 0x7f11: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? i64(a - b) : 0n };
          break;
        }
        case 0x7f12: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? i64(a * b) : 0n };
          break;
        }
        case 0x7f13: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          let r = 0n;
          if (a !== null && b !== null && b !== 0n) {
            r = a === I64_MIN && b === -1n ? I64_MIN : i64(a / b);
          }
          outVal = { kind: "Int", value: r };
          break;
        }
        case 0x7f14: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          let r = 0n;
          if (a !== null && b !== null && b !== 0n) {
            r = a === I64_MIN && b === -1n ? 0n : i64(a % b);
          }
          outVal = { kind: "Int", value: r };
          break;
        }
        case 0x7f15: {
          const a = this.evalAsI64(ins.operands[0], regs);
          outVal = { kind: "Int", value: a !== null ? i64(-a) : 0n };
          break;
        }
        case 0x7f16: {
          const a = this.evalAsI64(ins.operands[0], regs);
          outVal = { kind: "Int", value: a !== null ? (a < 0n ? i64(-a) : a) : 0n };
          break;
        }
        case 0x7f17: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? (a < b ? a : b) : 0n };
          break;
        }
        case 0x7f18: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? (a > b ? a : b) : 0n };
          break;
        }
        case 0x7f1e: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? i64(a & b) : 0n };
          break;
        }
        case 0x7f1f: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? i64(a | b) : 0n };
          break;
        }
        case 0x7f20: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? i64(a ^ b) : 0n };
          break;
        }
        case 0x7f21: {
          const a = this.evalAsI64(ins.operands[0], regs);
          outVal = { kind: "Int", value: a !== null ? i64(~a) : 0n };
          break;
        }
        case 0x7f22: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? i64(a << (b & 63n)) : 0n };
          break;
        }
        case 0x7f23: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Int", value: a !== null && b !== null ? i64(a >> (b & 63n)) : 0n };
          break;
        }
        case 0x7f50: {
          const v = this.evalOp(ins.operands[0], regs);
          if (v.kind === "Bytes" || v.kind === "U16Array" || v.kind === "TritVec") {
            outVal = { kind: "Int", value: BigInt(v.value.length) };
          } else if (v.kind === "Str") {
            outVal = { kind: "Int", value: BigInt(new TextEncoder().encode(v.value).length) };
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f51: {
          const a = this.evalOp(ins.operands[0], regs);
          const b = this.evalOp(ins.operands[1], regs);
          if (a.kind === "Bytes" && b.kind === "Bytes") {
            const inPlace =
              ins.destReg !== null &&
              ins.operands[0]?.kind === "Reg" &&
              ins.destReg === ins.operands[0].value &&
              ins.operands[1]?.kind === "Reg" &&
              ins.destReg !== ins.operands[1].value;
            if (inPlace) {
              this.chargeMemory(Math.max(1, b.value.length));
              for (let k = 0; k < b.value.length; k++) {
                a.value.push(b.value[k]);
              }
              outVal = a;
            } else {
              this.chargeMemory(Math.max(1, a.value.length + b.value.length));
              outVal = { kind: "Bytes", value: [...a.value, ...b.value] };
            }
          } else if (a.kind === "Str" && b.kind === "Str") {
            this.chargeMemory(Math.max(1, a.value.length + b.value.length));
            outVal = { kind: "Str", value: `${a.value}${b.value}` };
          } else {
            outVal = { kind: "Str", value: "" };
          }
          break;
        }
        case 0x7f52: {
          // SSLICE <str_or_bytes>, <start>, <end>
          const s = this.evalOp(ins.operands[0], regs);
          const start = Number(this.evalAsI64(ins.operands[1], regs) ?? 0n);
          const end = Number(this.evalAsI64(ins.operands[2], regs) ?? 0n);
          if (s.kind === "Str") {
            const lo = Math.max(0, Math.min(s.value.length, start));
            const hi = Math.max(lo, Math.min(s.value.length, end));
            const sliced = s.value.slice(lo, hi);
            this.chargeMemory(Math.max(1, sliced.length));
            outVal = { kind: "Str", value: sliced };
          } else if (s.kind === "Bytes") {
            const lo = Math.max(0, Math.min(s.value.length, start));
            const hi = Math.max(lo, Math.min(s.value.length, end));
            const sliced = s.value.slice(lo, hi);
            this.chargeMemory(Math.max(1, sliced.length));
            outVal = { kind: "Bytes", value: sliced };
          } else {
            outVal = { kind: "Str", value: "" };
          }
          break;
        }
        case 0x7f53: {
          // SFIND <haystack>, <needle> -> index or -1
          const h = this.evalOp(ins.operands[0], regs);
          const n = this.evalOp(ins.operands[1], regs);
          if (h.kind === "Str" && n.kind === "Str") {
            outVal = { kind: "Int", value: BigInt(h.value.indexOf(n.value)) };
          } else {
            outVal = { kind: "Int", value: -1n };
          }
          break;
        }
        case 0x7f54: {
          // SSPLIT <str>, <sep> -> Bytes of segment lengths or count
          const h = this.evalOp(ins.operands[0], regs);
          const sep = this.evalOp(ins.operands[1], regs);
          if (h.kind === "Str" && sep.kind === "Str") {
            const parts = h.value.split(sep.value);
            outVal = { kind: "Bytes", value: parts.map((p) => p.length & 0xff) };
          } else {
            outVal = { kind: "Bytes", value: [] };
          }
          break;
        }
        case 0x7f55: {
          const s = this.evalOp(ins.operands[0], regs);
          if (s.kind === "Str") {
            let parsed = 0n;
            try {
              const t = s.value.trim();
              if (/^-?\d+$/.test(t)) parsed = i64(BigInt(t));
            } catch {
              parsed = 0n;
            }
            outVal = { kind: "Int", value: parsed };
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f56: {
          const v = this.evalAsI64(ins.operands[0], regs);
          if (v !== null) {
            const s = v.toString();
            this.chargeMemory(Math.max(1, s.length));
            outVal = { kind: "Str", value: s };
          } else {
            outVal = { kind: "Str", value: "" };
          }
          break;
        }
        case 0x7f5e: {
          // HEX_ENC <bytes_or_str> -> hex string
          const v = this.evalOp(ins.operands[0], regs);
          const bytes =
            v.kind === "Bytes"
              ? v.value
              : v.kind === "Str"
                ? Array.from(new TextEncoder().encode(v.value))
                : [];
          const hex = bytes.map((b) => (b & 0xff).toString(16).padStart(2, "0")).join("");
          this.chargeMemory(Math.max(1, hex.length));
          outVal = { kind: "Str", value: hex };
          break;
        }
        case 0x7f5f: {
          // HEX_DEC <hex_str> -> Bytes
          const v = this.evalOp(ins.operands[0], regs);
          const out: number[] = [];
          if (v.kind === "Str") {
            const clean = v.value.trim().replace(/^0x/i, "");
            for (let i = 0; i + 1 < clean.length; i += 2) {
              const b = Number.parseInt(clean.slice(i, i + 2), 16);
              if (Number.isFinite(b)) out.push(b & 0xff);
            }
          }
          this.chargeMemory(Math.max(1, out.length));
          outVal = { kind: "Bytes", value: out };
          break;
        }
        case 0x7f24: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null && a === b ? 1 : -1 };
          break;
        }
        case 0x7f25: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null && a !== b ? 1 : -1 };
          break;
        }
        case 0x7f26: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null && a < b ? 1 : -1 };
          break;
        }
        case 0x7f27: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null && a > b ? 1 : -1 };
          break;
        }
        case 0x7f28: {
          const a = this.evalAsTrit(ins.operands[0], regs);
          outVal = { kind: "Trit", value: a !== null ? tnot(a) : 0 };
          break;
        }
        case 0x7f29: {
          const a = this.evalAsTrit(ins.operands[0], regs);
          const b = this.evalAsTrit(ins.operands[1], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null ? tand(a, b) : 0 };
          break;
        }
        case 0x7f2a: {
          const a = this.evalAsTrit(ins.operands[0], regs);
          const b = this.evalAsTrit(ins.operands[1], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null ? tor(a, b) : 0 };
          break;
        }
        case 0x7f2b: {
          const a = this.evalAsTrit(ins.operands[0], regs);
          const b = this.evalAsTrit(ins.operands[1], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null ? txor(a, b) : 0 };
          break;
        }
        case 0x7f2c: {
          const a = this.evalAsTrit(ins.operands[0], regs);
          const b = this.evalAsTrit(ins.operands[1], regs);
          const c = this.evalAsTrit(ins.operands[2], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null && c !== null ? tmaj(a, b, c) : 0 };
          break;
        }
        case 0x7f2d: {
          const a = this.evalAsI64(ins.operands[0], regs);
          const b = this.evalAsI64(ins.operands[1], regs);
          outVal = { kind: "Trit", value: a !== null && b !== null ? (a < b ? -1 : a > b ? 1 : 0) : 0 };
          break;
        }
        case 0x7f30: {
          const v = this.evalOp(ins.operands[0], regs);
          if (v.kind === "TritVec") {
            const packed = pack5Encode(v.value);
            this.chargeMemory(packed.length);
            outVal = { kind: "Bytes", value: packed };
          } else {
            outVal = { kind: "Bytes", value: [] };
          }
          break;
        }
        case 0x7f31: {
          const b = this.evalOp(ins.operands[0], regs);
          const len = this.evalAsI64(ins.operands[1], regs);
          if (b.kind === "Bytes" && len !== null) {
            const n = len > 0n ? Number(len) : 0;
            this.chargeMemory(pack5ByteLen(n));
            outVal = { kind: "TritVec", value: pack5Decode(b.value, n) };
          } else {
            outVal = { kind: "TritVec", value: [] };
          }
          break;
        }
        case 0x7f34: {
          const va = this.evalOp(ins.operands[0], regs);
          const vb = this.evalOp(ins.operands[1], regs);
          if (va.kind === "Bytes" && vb.kind === "Bytes") {
            const explicitN = ins.operands[2] ? this.evalAsI64(ins.operands[2], regs) : null;
            const nTrits = explicitN !== null ? Math.max(0, Number(explicitN)) : Math.min(va.value.length, vb.value.length) * 5;
            outVal = { kind: "Int", value: BigInt(pack5TdotDirect(va.value, vb.value, nTrits)) };
          } else if (va.kind === "TritVec" && vb.kind === "TritVec") {
            const pa = pack5Encode(va.value);
            const pb = pack5Encode(vb.value);
            const nTrits = Math.min(va.value.length, vb.value.length);
            outVal = { kind: "Int", value: BigInt(pack5TdotDirect(pa, pb, nTrits)) };
          } else if (va.kind === "Bytes" && vb.kind === "TritVec") {
            const pb = pack5Encode(vb.value);
            const nTrits = Math.min(va.value.length * 5, vb.value.length);
            outVal = { kind: "Int", value: BigInt(pack5TdotDirect(va.value, pb, nTrits)) };
          } else if (va.kind === "TritVec" && vb.kind === "Bytes") {
            const pa = pack5Encode(va.value);
            const nTrits = Math.min(va.value.length, vb.value.length * 5);
            outVal = { kind: "Int", value: BigInt(pack5TdotDirect(pa, vb.value, nTrits)) };
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f35:
        case 0x7f3f: {
          const v = this.evalOp(ins.operands[0], regs);
          let sum = 0n;
          if (v.kind === "TritVec") {
            const packed = pack5Encode(v.value);
            sum = BigInt(pack5SumDirect(packed, v.value.length));
          } else if (v.kind === "Bytes") {
            const explicitN = ins.operands[1] ? this.evalAsI64(ins.operands[1], regs) : null;
            const nTrits = explicitN !== null ? Math.max(0, Number(explicitN)) : v.value.length * 5;
            sum = BigInt(pack5SumDirect(v.value, nTrits));
          } else if (v.kind === "Int") {
            sum = v.value;
          } else if (v.kind === "Trit") {
            sum = BigInt(v.value);
          }
          outVal = { kind: "Trit", value: sum > 0n ? 1 : sum < 0n ? -1 : 0 };
          break;
        }
        case 0x7f36:
          outVal = { kind: "Trit", value: -1 };
          break;
        case 0x7f37:
          outVal = { kind: "Struct", fields: new Map() };
          break;
        case 0x7f39: {
          const m = new Map<number, OmniValue>();
          const t = ins.operands[2] ? this.evalAsTrit(ins.operands[2], regs) : null;
          if (t !== null) m.set(0, { kind: "Int", value: BigInt(t) });
          m.set(1, { kind: "Int", value: 64n });
          outVal = { kind: "Struct", fields: m };
          break;
        }
        case 0x7f45: {
          const c = this.evalAsTrit(ins.operands[0], regs);
          if (c !== null) {
            if (c < 0) outVal = this.evalOp(ins.operands[1], regs);
            else if (c === 0) outVal = this.evalOp(ins.operands[2], regs);
            else outVal = this.evalOp(ins.operands[3], regs);
          }
          break;
        }
        case 0x7f48: {
          const op0 = ins.operands[0];
          if (op0 && op0.kind === "FnRef") {
            if (this.callDepth >= MAX_VM_CALL_DEPTH) {
              throw new Error(
                `error[BUDGET_EXCEEDED]: VM call depth budget exceeded (${this.callDepth} >= ${MAX_VM_CALL_DEPTH})`,
              );
            }
            const callArgs = ins.operands.slice(1).map((op) => this.evalOp(op, regs));
            this.callDepth += 1;
            try {
              outVal = this.executeFunction(module, op0.value, callArgs);
            } finally {
              this.callDepth = Math.max(0, this.callDepth - 1);
            }
          }
          break;
        }
        case 0x7f57: {
          const s = this.evalOp(ins.operands[0], regs);
          outVal = {
            kind: "Str",
            value: s.kind === "Str" ? s.value.replace(/'/g, "''").replace(/;/g, "") : "",
          };
          break;
        }
        case 0x7f58: {
          const s = this.evalOp(ins.operands[0], regs);
          outVal = {
            kind: "Str",
            value:
              s.kind === "Str"
                ? s.value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
                : "",
          };
          break;
        }
        case 0x7f59:
          outVal = this.evalOp(ins.operands[0], regs);
          break;
        case 0x7f5d: {
          const s = this.evalOp(ins.operands[0], regs);
          if (s.kind === "Str") {
            const bytes = new TextEncoder().encode(s.value);
            outVal = { kind: "Int", value: i64(fnv1a64Hash(bytes)) };
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f63: {
          const op0 = ins.operands[0];
          if (op0 && op0.kind === "SchemaRef") {
            this.chargeMemory(64 * ins.operands.length);
            const row = new Map<number, OmniValue>();
            for (const op of ins.operands.slice(1)) {
              if (op.kind === "FieldKV") {
                row.set(op.fieldIdx, this.evalOp(op.inner, regs));
              }
            }
            const list = this.tables.get(op0.value) ?? [];
            list.push(row);
            this.tables.set(op0.value, list);
            outVal = { kind: "Struct", fields: new Map(row) };
          }
          break;
        }
        case 0x7f64:
        case 0x7f65: {
          if (ins.operands.length === 1) {
            const key = this.evalAsI64(ins.operands[0], regs) ?? 0n;
            const val = this.kvStore.get(key) ?? i64(key * 10n);
            outVal = { kind: "Int", value: val };
          } else if (ins.operands.length >= 2) {
            const tbl = this.evalAsI64(ins.operands[0], regs) ?? 0n;
            const key = this.evalAsI64(ins.operands[1], regs) ?? 0n;
            const composite = i64(tbl * 1_000_000n + key);
            const val = this.kvStore.get(composite) ?? this.kvStore.get(key) ?? i64(key * 10n);
            outVal = { kind: "Int", value: val };
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f68: {
          let totalRows = 0;
          for (const rows of this.tables.values()) totalRows += rows.length;
          outVal = { kind: "Int", value: BigInt(this.kvStore.size + totalRows) };
          break;
        }
        case 0x7f6b: {
          this.chargeMemory(16);
          if (ins.operands.length >= 2) {
            const key = this.evalAsI64(ins.operands[0], regs) ?? 0n;
            const val = this.evalAsI64(ins.operands[1], regs) ?? 0n;
            this.kvStore.set(key, val);
            outVal = { kind: "Int", value: val };
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f6c: {
          const methodId = ins.operands[0] ? (this.evalAsI64(ins.operands[0], regs) ?? 1n) : 1n;
          const routeId = ins.operands[1] ? (this.evalAsI64(ins.operands[1], regs) ?? 0n) : 0n;
          outVal = { kind: "Int", value: i64(methodId * 1000n + (routeId & 0x3ffn)) };
          break;
        }
        case 0x7f6d: {
          const evalParamIdx = (op0: BinOperand): OmniValue => {
            if (op0.kind === "ImmInt" && op0.value >= 0n && op0.value < 32n) {
              const slot = 192 + Number(op0.value);
              return regs[slot].kind !== "Void" ? regs[slot] : { kind: "Int", value: op0.value };
            }
            return this.evalOp(op0, regs);
          };
          if (ins.operands[1]) {
            const keyHash = this.evalAsI64(ins.operands[1], regs);
            if (keyHash !== null && this.reqParams.has(keyHash)) {
              outVal = { kind: "Int", value: this.reqParams.get(keyHash)! };
            } else if (ins.operands[0]) {
              outVal = evalParamIdx(ins.operands[0]);
            } else {
              outVal = { kind: "Int", value: 0n };
            }
          } else if (ins.operands[0]) {
            const keyHash = this.evalAsI64(ins.operands[0], regs);
            if (keyHash !== null && this.reqParams.has(keyHash)) {
              outVal = { kind: "Int", value: this.reqParams.get(keyHash)! };
            } else {
              outVal = evalParamIdx(ins.operands[0]);
            }
          } else {
            outVal = { kind: "Int", value: 0n };
          }
          break;
        }
        case 0x7f6e: {
          const status = ins.operands[0] ? (this.evalAsI64(ins.operands[0], regs) ?? 200n) : 200n;
          const body = ins.operands[1] ? (this.evalAsI64(ins.operands[1], regs) ?? 0n) : 0n;
          const clamped = Math.min(599, Math.max(100, Number(status)));
          this.lastHttpResponse = {
            status: clamped,
            contentType: "application/json",
            body: `{"status":${status},"data":${body}}`,
          };
          outVal = { kind: "Int", value: i64(status * 1_000_000n + body) };
          break;
        }
        case 0x7f6f: {
          const status = ins.operands[0] ? (this.evalAsI64(ins.operands[0], regs) ?? 200n) : 200n;
          const body = ins.operands[1] ? this.evalOp(ins.operands[1], regs) : { kind: "Void" as const };
          const outS = `<http status="${status}">${JSON.stringify(body)}</http>`;
          this.chargeMemory(outS.length);
          this.lastHttpResponse = {
            status: Math.min(599, Math.max(100, Number(status))),
            contentType: "text/html; charset=utf-8",
            body: outS,
          };
          outVal = { kind: "Str", value: outS };
          break;
        }
        case 0x7f70:
          outVal = ins.operands[0] ? this.evalOp(ins.operands[0], regs) : { kind: "Int", value: 0n };
          break;
        case 0x7f76: {
          const op0 = ins.operands[0];
          if (op0 && op0.kind === "ImmInt") {
            this.chargeMemory(32 * ins.operands.length);
            outVal = {
              kind: "DomNode",
              tagId: Number(op0.value & 0xffffn),
              children: ins.operands.slice(1).map((op) => this.evalOp(op, regs)),
            };
          }
          break;
        }
        default:
          outVal = { kind: "Void" };
      }

      if (ins.destReg !== null) {
        const slot = tokenToSlot(ins.destReg);
        if (slot !== null) {
          regs[slot] = outVal;
        }
      }
      pc += 1;
    }

    return { kind: "Void" };
  }
}
