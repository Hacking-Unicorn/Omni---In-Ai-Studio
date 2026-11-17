// Omni AOT — Two-Pass x86-64 Linux ELF64 Compiler, Multi-Function WebAssembly v1 Compiler,
// Budgeted 64-bit Wasm Evaluator, and Stage-1 Pure-u16 Self-Hosting Compiler Stream.

import {
  CAP_ID_PURE,
  LBL_BASE,
  PACK5_MAX_VALID,
  PAR_BASE,
  PAR_MAX,
  REG_BASE,
  REG_MAX,
  TOK_FN_DEF,
  TOK_IMM_I16,
  TOK_IMM_I32,
  TOK_INS_END,
  TOK_MOD_END,
  TOK_TRIT_NEG,
  TOK_TRIT_POS,
  TOK_TRIT_ZERO,
  TY_ARENA,
  TY_BYTES,
  TY_I64,
  TY_TRIT,
  decodeOpcodeToken,
  isWasmAotSupportedOpcode,
  isX86_64AotSupportedOpcode,
  packTc,
} from "./isa.ts";
import {
  BinFunction,
  BinInstruction,
  BinModule,
  BinOperand,
  decodeOmniTokenStream,
  extractLabelId,
  packFnModule,
} from "./ir.ts";
import { verifyBinaryFunction } from "./verifier.ts";
import { OmniBinaryVm } from "./vm.ts";

export const DEFAULT_WASM_STEP_LIMIT = 50_000_000;
export const DEFAULT_WASM_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024;

const I64_MIN = -(1n << 63n);

function i64(n: bigint): bigint {
  return BigInt.asIntN(64, n);
}

export function fnv1a64Hash(bytes: Uint8Array): bigint {
  let hash = 0xcbf29ce484222325n;
  for (let i = 0; i < bytes.length; i++) {
    hash ^= BigInt(bytes[i]);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash;
}

export function formatFnv1a64Hex(bytes: Uint8Array): string {
  return `0x${fnv1a64Hash(bytes).toString(16).toUpperCase().padStart(16, "0")}`;
}

export function u16SliceToLeBytes(words: number[]): Uint8Array {
  const out = new Uint8Array(words.length * 2);
  for (let i = 0; i < words.length; i++) {
    const w = words[i] & 0xffff;
    out[i * 2] = w & 0xff;
    out[i * 2 + 1] = (w >>> 8) & 0xff;
  }
  return out;
}

function pushU16Le(buf: number[], v: number): void {
  buf.push(v & 0xff, (v >>> 8) & 0xff);
}

function pushU32Le(buf: number[], v: number): void {
  buf.push(v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff);
}

function pushU64Le(buf: number[], v: bigint): void {
  const u = BigInt.asUintN(64, v);
  for (let i = 0; i < 8; i++) {
    buf.push(Number((u >> BigInt(i * 8)) & 0xffn));
  }
}

function encodeUleb128(val: number): number[] {
  let v = val >>> 0;
  const out: number[] = [];
  do {
    let byte = v & 0x7f;
    v >>>= 7;
    if (v !== 0) byte |= 0x80;
    out.push(byte);
  } while (v !== 0);
  return out;
}

function encodeSleb128(val: bigint): number[] {
  let v = val;
  const out: number[] = [];
  let more = true;
  while (more) {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    const signBit = (byte & 0x40) !== 0;
    if ((v === 0n && !signBit) || (v === -1n && signBit)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    out.push(byte);
  }
  return out;
}

function decodeUleb128(bytes: Uint8Array, state: { pos: number }): number {
  let result = 0;
  let shift = 0;
  while (true) {
    if (state.pos >= bytes.length) throw new Error("Unexpected EOF in Wasm ULEB128");
    const byte = bytes[state.pos++];
    result |= (byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) break;
    shift += 7;
    if (shift >= 35) throw new Error("Wasm ULEB128 overflow");
  }
  return result >>> 0;
}

function decodeSleb128(bytes: Uint8Array, state: { pos: number }): bigint {
  let result = 0n;
  let shift = 0n;
  let byte = 0;
  while (true) {
    if (state.pos >= bytes.length) throw new Error("Unexpected EOF in Wasm SLEB128");
    byte = bytes[state.pos++];
    result |= BigInt(byte & 0x7f) << shift;
    shift += 7n;
    if ((byte & 0x80) === 0) break;
    if (shift >= 64n) break;
  }
  if (shift < 64n && (byte & 0x40) !== 0) {
    result |= (~0n) << shift;
  }
  return i64(result);
}

interface Pack5RegEntry {
  bytes: number[];
  nTrits: number;
  isTv: boolean;
}

function collectFuncPack5Map(func: BinFunction): Map<number, Pack5RegEntry> {
  const map = new Map<number, Pack5RegEntry>();
  const dynamicByteRegs = new Set<number>();
  for (const ins of func.instructions) {
    if (ins.opTok === 0x7f0f) {
      const isTypedWrite =
        (ins.operands.length === 3 &&
          ins.operands[1].kind === "ImmInt" &&
          (ins.operands[1].value === -4n || ins.operands[1].value === -8n)) ||
        (ins.operands.length === 4 && ins.operands[1].kind === "ImmInt" && ins.operands[1].value === -9n);
      if (isTypedWrite && ins.destReg !== null) {
        dynamicByteRegs.add(ins.destReg);
      }
    }
  }
  for (const ins of func.instructions) {
    if (ins.destReg !== null && dynamicByteRegs.has(ins.destReg)) {
      continue;
    }
    if (ins.opTok === 0x7f0d) {
      if (ins.destReg !== null) map.set(ins.destReg, { bytes: [], nTrits: 0, isTv: false });
    } else if (ins.opTok === 0x7f0f) {
      const isTypedWrite =
        (ins.operands.length === 3 &&
          ins.operands[1].kind === "ImmInt" &&
          (ins.operands[1].value === -4n || ins.operands[1].value === -8n)) ||
        (ins.operands.length === 4 && ins.operands[1].kind === "ImmInt" && ins.operands[1].value === -9n);
      if (!isTypedWrite) {
        const first = ins.operands[0];
        const prev = first && first.kind === "Reg" ? map.get(first.value) : undefined;
        const bytes = prev ? [...prev.bytes] : [];
        for (const opB of ins.operands.slice(1)) {
          if (opB.kind === "ImmInt") {
            bytes.push(Number(opB.value & 0xffn));
          }
        }
        if (ins.destReg !== null) {
          map.set(ins.destReg, { bytes, nTrits: bytes.length * 5, isTv: false });
        }
      }
    } else if (ins.opTok === 0x7f31) {
      const first = ins.operands[0];
      const second = ins.operands[1];
      if (ins.destReg !== null && first && first.kind === "Reg") {
        const bytes = map.get(first.value)?.bytes ? [...map.get(first.value)!.bytes] : [];
        const nTrits =
          second && second.kind === "ImmInt" ? Math.max(0, Number(second.value)) : bytes.length * 5;
        map.set(ins.destReg, { bytes, nTrits, isTv: true });
      }
    } else if (ins.opTok === 0x7f30) {
      const first = ins.operands[0];
      if (ins.destReg !== null && first && first.kind === "Reg") {
        const entry = map.get(first.value);
        if (entry) {
          map.set(ins.destReg, { bytes: [...entry.bytes], nTrits: entry.nTrits, isTv: false });
        }
      }
    }
  }
  return map;
}

function pack5WordFromBytes(bytes: number[], nTrits: number): bigint {
  let word = BigInt(nTrits & 0xff);
  for (let i = 0; i < Math.min(bytes.length, 7); i++) {
    word |= BigInt(bytes[i] & 0xff) << BigInt((i + 1) * 8);
  }
  return i64(word);
}

function validateFuncPack5Ops(func: BinFunction, regPack5: Map<number, Pack5RegEntry>): void {
  const checkReg = (r: number) => {
    const entry = regPack5.get(r);
    if (entry && entry.bytes.some((b) => b > PACK5_MAX_VALID)) {
      throw new Error("error[CORRUPT_PACK5]: PACK5_SENTINEL_OR_CORRUPT_BYTE");
    }
  };
  for (const ins of func.instructions) {
    if (
      ins.opTok === 0x7f30 ||
      ins.opTok === 0x7f31 ||
      ins.opTok === 0x7f34 ||
      ins.opTok === 0x7f35 ||
      ins.opTok === 0x7f3f
    ) {
      for (const op of ins.operands) {
        if (op.kind === "Reg") checkReg(op.value);
      }
    }
  }
}

export function isModuleAotHotPath(module: BinModule): boolean {
  if (module.functions.length !== 1) return false;
  const func = module.functions[0];
  if (func.retTyTok !== TY_I64 && func.retTyTok !== TY_TRIT) return false;
  if (!func.params.every(([, ty]) => ty === TY_I64 || ty === TY_TRIT || ty === TY_ARENA)) return false;
  return func.instructions.every(
    (ins) => isWasmAotSupportedOpcode(ins.opTok) && isX86_64AotSupportedOpcode(ins.opTok),
  );
}

export function isModuleMultiFnWasmCompatible(module: BinModule): boolean {
  if (module.functions.length === 0) return false;
  for (const func of module.functions) {
    if (func.retTyTok !== TY_I64 && func.retTyTok !== TY_TRIT) return false;
    if (!func.params.every(([, ty]) => ty === TY_I64 || ty === TY_TRIT || ty === TY_ARENA)) return false;
    for (const ins of func.instructions) {
      if (!isWasmAotSupportedOpcode(ins.opTok) && ins.opTok !== 0x7f48) {
        return false;
      }
    }
  }
  return true;
}

// ==============================================================================
// TWO-PASS x86-64 LINUX ELF64 NATIVE COMPILER
// ==============================================================================

export function compilePureU16ToX86_64Elf(stream: number[], arg0: bigint, arg1: bigint): Uint8Array {
  return compilePureU16ToX86_64ElfWithArgs(stream, arg0, arg1, 0n);
}

export function compilePureU16ToX86_64ElfWithArgs(
  stream: number[],
  arg0: bigint,
  arg1: bigint,
  arg2: bigint,
): Uint8Array {
  const mod = decodeOmniTokenStream(stream);
  const func = mod.functions[0];
  if (!func) throw new Error("No function in module");
  const rep = verifyBinaryFunction(func);
  if (!rep.passed) {
    throw new Error(`Cannot AOT-compile unverified binary stream: ${JSON.stringify(rep.errors)}`);
  }

  for (const ins of func.instructions) {
    if (!isX86_64AotSupportedOpcode(ins.opTok)) {
      const lbl = decodeOpcodeToken(ins.opTok)?.debugLabel ?? "OP_UNKNOWN";
      throw new Error(
        `Opcode 0x${ins.opTok.toString(16).toUpperCase().padStart(4, "0")} (${lbl}) requires high-level VM runtime state and is not lowered by standalone x86-64 ELF64 AOT`,
      );
    }
  }

  const regPack5 = collectFuncPack5Map(func);
  validateFuncPack5Ops(func, regPack5);

  const labelOffsets = new Map<number, number>();
  for (const ins of func.instructions) {
    if (ins.opTok === 0x7f40 && ins.operands[0]) {
      const lbl = extractLabelId(ins.operands[0]);
      if (lbl !== null) labelOffsets.set(lbl, 0);
    }
  }

  let code: number[] = [];
  for (let pass = 0; pass < 2; pass++) {
    code = [];
    // Prologue: push rbp; mov rbp, rsp; sub rsp, 2048; zero virtual frame; load %p0..%p2
    code.push(0x55, 0x48, 0x89, 0xe5, 0x48, 0x81, 0xec, 0x00, 0x08, 0x00, 0x00);
    code.push(
      0x48, 0x8d, 0xbd, 0x00, 0xf8, 0xff, 0xff,
      0xb9, 0x00, 0x01, 0x00, 0x00,
      0x31, 0xc0,
      0xf3, 0x48, 0xab,
    );
    code.push(0x48, 0xb8);
    pushU64Le(code, arg0);
    code.push(0x48, 0x89, 0x45, 0xf8);
    code.push(0x48, 0xb8);
    pushU64Le(code, arg1);
    code.push(0x48, 0x89, 0x45, 0xf0);
    code.push(0x48, 0xb8);
    pushU64Le(code, arg2);
    code.push(0x48, 0x89, 0x45, 0xe8);

    const slotDisp = (tok: number): number => {
      if (tok >= PAR_BASE && tok <= PAR_MAX) {
        return -8 * (tok - PAR_BASE + 1);
      }
      return -24 - 8 * (tok - REG_BASE + 1);
    };

    const emitLoadRax = (c: number[], op: BinOperand) => {
      if (op.kind === "Param" || op.kind === "Reg") {
        c.push(0x48, 0x8b, 0x85);
        pushU32Le(c, slotDisp(op.value));
      } else if (op.kind === "ImmInt") {
        c.push(0x48, 0xb8);
        pushU64Le(c, op.value);
      } else if (op.kind === "ImmTrit") {
        c.push(0x48, 0xb8);
        pushU64Le(c, BigInt(op.value));
      } else {
        c.push(0x48, 0x31, 0xc0);
      }
    };

    const emitLoadRcx = (c: number[], op: BinOperand) => {
      if (op.kind === "Param" || op.kind === "Reg") {
        c.push(0x48, 0x8b, 0x8d);
        pushU32Le(c, slotDisp(op.value));
      } else if (op.kind === "ImmInt") {
        c.push(0x48, 0xb9);
        pushU64Le(c, op.value);
      } else if (op.kind === "ImmTrit") {
        c.push(0x48, 0xb9);
        pushU64Le(c, BigInt(op.value));
      } else {
        c.push(0x48, 0x31, 0xc9);
      }
    };

    const emitStoreRax = (c: number[], dest: number | null) => {
      if (dest !== null) {
        c.push(0x48, 0x89, 0x85);
        pushU32Le(c, slotDisp(dest));
      }
    };

    const emitRel32 = (c: number[], lblTok: number, curPass: number) => {
      const target = labelOffsets.get(lblTok);
      if (target === undefined) {
        throw new Error(`Unresolved binary label token 0x${lblTok.toString(16).toUpperCase()}`);
      }
      const nextRip = c.length + 4;
      const rel = curPass === 0 ? 0 : target - nextRip;
      pushU32Le(c, rel);
    };

    const emitExtractPack5TritToRax = (c: number[], op: BinOperand, tritIdx: number) => {
      let directByte = false;
      if (op.kind === "Reg") {
        const entry = regPack5.get(op.value);
        if (entry) {
          const bVal = BigInt(entry.bytes[Math.floor(tritIdx / 5)] ?? 121);
          c.push(0x48, 0xb8);
          pushU64Le(c, bVal);
          directByte = true;
        }
      }
      if (!directByte) {
        emitLoadRax(c, op);
        const shift = ((Math.floor(tritIdx / 5) + 1) * 8) & 0xff;
        c.push(0x48, 0xc1, 0xe8, shift, 0x48, 0x25, 0xff, 0x00, 0x00, 0x00);
      }
      const pow3 = [1n, 3n, 9n, 27n, 81n][tritIdx % 5];
      if (pow3 > 1n) {
        c.push(0x48, 0x31, 0xd2, 0x48, 0xb9);
        pushU64Le(c, pow3);
        c.push(0x48, 0xf7, 0xf1);
      }
      c.push(
        0x48, 0x31, 0xd2,
        0x48, 0xb9, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x48, 0xf7, 0xf1,
        0x48, 0x89, 0xd0,
        0x48, 0x83, 0xe8, 0x01,
      );
    };

    for (const ins of func.instructions) {
      switch (ins.opTok) {
        case 0x7f00:
        case 0x7f05:
          break;
        case 0x7f40: {
          const lbl = ins.operands[0] ? extractLabelId(ins.operands[0]) : null;
          if (lbl === null) throw new Error("Missing label operand");
          labelOffsets.set(lbl, code.length);
          break;
        }
        case 0x7f41: {
          const lbl = ins.operands[0] ? extractLabelId(ins.operands[0]) : null;
          if (lbl === null) throw new Error("Missing JMP label");
          code.push(0xe9);
          emitRel32(code, lbl, pass);
          break;
        }
        case 0x7f43: {
          const lbl = ins.operands[1] ? extractLabelId(ins.operands[1]) : null;
          if (lbl === null) throw new Error("Missing BR_POS label");
          emitLoadRax(code, ins.operands[0]);
          code.push(0x48, 0x83, 0xf8, 0x01, 0x0f, 0x84);
          emitRel32(code, lbl, pass);
          break;
        }
        case 0x7f44: {
          const lbl = ins.operands[1] ? extractLabelId(ins.operands[1]) : null;
          if (lbl === null) throw new Error("Missing BR_NEG label");
          emitLoadRax(code, ins.operands[0]);
          code.push(0x48, 0x83, 0xf8, 0xff, 0x0f, 0x84);
          emitRel32(code, lbl, pass);
          break;
        }
        case 0x7f42: {
          const lNeg = ins.operands[1] ? extractLabelId(ins.operands[1]) : null;
          const lZero = ins.operands[2] ? extractLabelId(ins.operands[2]) : null;
          const lPos = ins.operands[3] ? extractLabelId(ins.operands[3]) : null;
          if (lNeg === null || lZero === null || lPos === null) throw new Error("Missing BR_TRIT labels");
          emitLoadRax(code, ins.operands[0]);
          code.push(0x48, 0x85, 0xc0, 0x0f, 0x8c);
          emitRel32(code, lNeg, pass);
          code.push(0x0f, 0x84);
          emitRel32(code, lZero, pass);
          code.push(0xe9);
          emitRel32(code, lPos, pass);
          break;
        }
        case 0x7f01:
        case 0x7f02:
        case 0x7f03:
        case 0x7f04:
        case 0x7f06:
        case 0x7f09:
        case 0x7f57:
        case 0x7f58:
        case 0x7f59:
          emitLoadRax(code, ins.operands[0]);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f30:
        case 0x7f31: {
          let word = 0n;
          if (ins.destReg !== null) {
            const entry = regPack5.get(ins.destReg);
            if (entry) word = pack5WordFromBytes(entry.bytes, entry.nTrits);
          }
          code.push(0x48, 0xb8);
          pushU64Le(code, word);
          emitStoreRax(code, ins.destReg);
          break;
        }
        case 0x7f34: {
          let nTrits = 0;
          if (ins.operands[0]?.kind === "Reg" && ins.operands[1]?.kind === "Reg") {
            const na = regPack5.get(ins.operands[0].value)?.nTrits ?? 0;
            const nb = regPack5.get(ins.operands[1].value)?.nTrits ?? 0;
            nTrits = Math.min(na, nb);
          }
          code.push(0x4d, 0x31, 0xc0);
          for (let idx = 0; idx < nTrits; idx++) {
            emitExtractPack5TritToRax(code, ins.operands[0], idx);
            code.push(0x49, 0x89, 0xc1);
            emitExtractPack5TritToRax(code, ins.operands[1], idx);
            code.push(0x4c, 0x0f, 0xaf, 0xc8, 0x4d, 0x01, 0xc8);
          }
          code.push(0x4c, 0x89, 0xc0);
          emitStoreRax(code, ins.destReg);
          break;
        }
        case 0x7f35:
        case 0x7f3f: {
          const pack5Len = ins.operands[0]?.kind === "Reg" ? regPack5.get(ins.operands[0].value)?.nTrits : undefined;
          if (pack5Len !== undefined) {
            code.push(0x4d, 0x31, 0xc0);
            for (let idx = 0; idx < pack5Len; idx++) {
              emitExtractPack5TritToRax(code, ins.operands[0], idx);
              code.push(0x49, 0x01, 0xc0);
            }
            code.push(0x4c, 0x89, 0xc0);
          } else {
            emitLoadRax(code, ins.operands[0]);
          }
          code.push(
            0x48, 0x85, 0xc0,
            0x0f, 0x9f, 0xc2,
            0x0f, 0x9c, 0xc0,
            0x0f, 0xb6, 0xd2,
            0x0f, 0xb6, 0xc0,
            0x48, 0x29, 0xc2,
            0x48, 0x89, 0xd0,
          );
          emitStoreRax(code, ins.destReg);
          break;
        }
        case 0x7f0d:
        case 0x7f0f:
        case 0x7f50:
        case 0x7f51:
          break;
        case 0x7f0e: {
          const arrOp = ins.operands[0];
          const idxOp = ins.operands[1];
          if (arrOp?.kind === "Reg" && idxOp) {
            const entry = regPack5.get(arrOp.value);
            if (entry && entry.nTrits > 0) {
              if (idxOp.kind === "ImmInt" && idxOp.value >= 0n) {
                const idxNum = Number(idxOp.value);
                if (entry.isTv) {
                  emitExtractPack5TritToRax(code, arrOp, idxNum);
                } else {
                  const bVal = BigInt(entry.bytes[idxNum] ?? 0);
                  code.push(0x48, 0xb8);
                  pushU64Le(code, bVal);
                }
                emitStoreRax(code, ins.destReg);
              } else {
                emitLoadRax(code, idxOp);
                code.push(0x49, 0x89, 0xc2, 0x4d, 0x31, 0xc0);
                for (let k = 0; k < entry.nTrits; k++) {
                  if (entry.isTv) {
                    emitExtractPack5TritToRax(code, arrOp, k);
                  } else {
                    const bVal = BigInt(entry.bytes[k] ?? 0);
                    code.push(0x48, 0xb8);
                    pushU64Le(code, bVal);
                  }
                  code.push(0x49, 0x81, 0xfa);
                  pushU32Le(code, k);
                  code.push(0x4c, 0x0f, 0x44, 0xc0);
                }
                code.push(0x4c, 0x89, 0xc0);
                emitStoreRax(code, ins.destReg);
              }
            }
          }
          break;
        }
        case 0x7f10:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x01, 0xc8);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f11:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x29, 0xc8);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f12:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x0f, 0xaf, 0xc1);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f13:
        case 0x7f14:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(
            0x48, 0x85, 0xc9,
            0x75, 0x05,
            0x48, 0x31, 0xc0,
            0xeb, 0x1e,
            0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
            0x48, 0x39, 0xd0,
            0x75, 0x0c,
            0x48, 0x83, 0xf9, 0xff,
            0x75, 0x06,
            0x48, 0x31, 0xd2,
            0xeb, 0x05,
            0x48, 0x99,
            0x48, 0xf7, 0xf9,
          );
          if (ins.opTok === 0x7f14) {
            code.push(0x48, 0x89, 0xd0);
          }
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f15:
        case 0x7f28:
          emitLoadRax(code, ins.operands[0]);
          code.push(0x48, 0xf7, 0xd8);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f16:
          emitLoadRax(code, ins.operands[0]);
          code.push(0x48, 0x89, 0xc2, 0x48, 0xf7, 0xda, 0x48, 0x0f, 0x48, 0xc2);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f17:
        case 0x7f29:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x39, 0xc8, 0x48, 0x0f, 0x4f, 0xc1);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f18:
        case 0x7f2a:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x39, 0xc8, 0x48, 0x0f, 0x4c, 0xc1);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f1e:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x21, 0xc8);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f1f:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x09, 0xc8);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f20:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x31, 0xc8);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f21:
          emitLoadRax(code, ins.operands[0]);
          code.push(0x48, 0xf7, 0xd0);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f22:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x80, 0xe1, 0x3f, 0x48, 0xd3, 0xe0);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f23:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x80, 0xe1, 0x3f, 0x48, 0xd3, 0xf8);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f24:
        case 0x7f25:
        case 0x7f26:
        case 0x7f27: {
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x39, 0xc8);
          const setcc = ins.opTok === 0x7f24 ? 0x94 : ins.opTok === 0x7f25 ? 0x95 : ins.opTok === 0x7f26 ? 0x9c : 0x9f;
          code.push(0x0f, setcc, 0xc0, 0x0f, 0xb6, 0xc0, 0x48, 0x8d, 0x44, 0x00, 0xff);
          emitStoreRax(code, ins.destReg);
          break;
        }
        case 0x7f2b:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x0f, 0xaf, 0xc1, 0x48, 0xf7, 0xd8);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f2c:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x01, 0xc8);
          emitLoadRcx(code, ins.operands[2]);
          code.push(
            0x48, 0x01, 0xc8,
            0x48, 0x85, 0xc0,
            0x0f, 0x9f, 0xc2,
            0x0f, 0x9c, 0xc0,
            0x0f, 0xb6, 0xd2,
            0x0f, 0xb6, 0xc0,
            0x48, 0x29, 0xc2,
            0x48, 0x89, 0xd0,
          );
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f2d:
          emitLoadRax(code, ins.operands[0]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(
            0x48, 0x39, 0xc8,
            0x0f, 0x9f, 0xc2,
            0x0f, 0x9c, 0xc0,
            0x0f, 0xb6, 0xd2,
            0x0f, 0xb6, 0xc0,
            0x48, 0x29, 0xc2,
            0x48, 0x89, 0xd0,
          );
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f45:
          emitLoadRcx(code, ins.operands[0]);
          emitLoadRax(code, ins.operands[2]);
          emitLoadRcx(code, ins.operands[1]);
          code.push(0x48, 0x89, 0xca);
          emitLoadRcx(code, ins.operands[0]);
          code.push(0x48, 0x85, 0xc9, 0x48, 0x0f, 0x4c, 0xc2);
          emitLoadRcx(code, ins.operands[3]);
          code.push(0x48, 0x89, 0xca);
          emitLoadRcx(code, ins.operands[0]);
          code.push(0x48, 0x85, 0xc9, 0x48, 0x0f, 0x4f, 0xc2);
          emitStoreRax(code, ins.destReg);
          break;
        case 0x7f4a:
          emitLoadRax(code, ins.operands[0]);
          code.push(0x48, 0x89, 0xc7, 0xb8, 0x3c, 0x00, 0x00, 0x00, 0x0f, 0x05);
          break;
        default:
          break;
      }
    }
    // Default exit syscall
    code.push(0x31, 0xff, 0xb8, 0x3c, 0x00, 0x00, 0x00, 0x0f, 0x05);
  }

  // Wrap in 64-bit Linux ELF64 header (120 bytes) + machine code
  const elf: number[] = [];
  const loadVaddr = 0x400000n;
  const hdrSize = 64 + 56;
  const entryVaddr = loadVaddr + BigInt(hdrSize);
  const totalSize = BigInt(hdrSize + code.length);

  elf.push(0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  pushU16Le(elf, 2);
  pushU16Le(elf, 0x3e);
  pushU32Le(elf, 1);
  pushU64Le(elf, entryVaddr);
  pushU64Le(elf, 64n);
  pushU64Le(elf, 0n);
  pushU32Le(elf, 0);
  pushU16Le(elf, 64);
  pushU16Le(elf, 56);
  pushU16Le(elf, 1);
  pushU16Le(elf, 0);
  pushU16Le(elf, 0);
  pushU16Le(elf, 0);

  pushU32Le(elf, 1);
  pushU32Le(elf, 5);
  pushU64Le(elf, 0n);
  pushU64Le(elf, loadVaddr);
  pushU64Le(elf, loadVaddr);
  pushU64Le(elf, totalSize);
  pushU64Le(elf, totalSize);
  pushU64Le(elf, 0x1000n);

  elf.push(...code);
  return Uint8Array.from(elf);
}

// ==============================================================================
// STAGE-1 PURE-u16 SELF-HOSTING COMPILER STREAM (.omni_bc -> ELF64)
// ==============================================================================

const TC_SAFE = packTc(false, CAP_ID_PURE);

function imm(n: number): number[] {
  return [TOK_IMM_I16, n & 0xffff];
}

function setReg(ins: number[][], reg: number, op: number, ty: number, ops: number[]): void {
  ins.push([reg, op, ty, TC_SAFE, ...ops, TOK_INS_END]);
}

function emitLabel(ins: number[][], lblId: number): void {
  ins.push([0x7f40, ...imm(lblId), TOK_INS_END]);
}

function emitJmp(ins: number[][], lblId: number): void {
  ins.push([0x7f41, ...imm(lblId), TOK_INS_END]);
}

function emitBrPos(ins: number[][], condReg: number, lblId: number): void {
  ins.push([0x7f43, condReg, ...imm(lblId), TOK_INS_END]);
}

function emitBrTrit(ins: number[][], condReg: number, lNeg: number, lZero: number, lPos: number): void {
  ins.push([0x7f42, condReg, ...imm(lNeg), ...imm(lZero), ...imm(lPos), TOK_INS_END]);
}

let cachedStage1Stream: number[] | null = null;

export function buildStep7Stage1CompilerStream(): number[] {
  if (cachedStage1Stream) return [...cachedStage1Stream];

  const pBc = PAR_BASE;
  const pArg0 = PAR_BASE + 1;
  const pArg1 = PAR_BASE + 2;
  const pArg2 = PAR_BASE + 3;

  const ins: number[][] = [];
  let nextReg = REG_BASE;
  const allocReg = (op: number, ty: number, ops: number[]): number => {
    const r = nextReg++;
    setReg(ins, r, op, ty, ops);
    return r;
  };

  // Canonical ascending SSA register pool (33 registers total, < REG_MAX)
  const rBcLen = allocReg(0x7f50, TY_I64, [pBc]);
  const rLblTbl = allocReg(0x7f0d, TY_BYTES, []);
  const rPass = allocReg(0x7f06, TY_I64, imm(0));
  const rCodeOut = allocReg(0x7f0d, TY_BYTES, []);
  const rPos = allocReg(0x7f06, TY_I64, imm(0));
  const rFirstWord = allocReg(0x7f06, TY_I64, imm(0));
  const rDest = allocReg(0x7f06, TY_I64, imm(0));
  const rOp = allocReg(0x7f06, TY_I64, imm(0));
  const rOpCur = allocReg(0x7f06, TY_I64, imm(0));
  const rOp0Kind = allocReg(0x7f06, TY_I64, imm(0));
  const rOp0Val = allocReg(0x7f06, TY_I64, imm(0));
  const rOp1Kind = allocReg(0x7f06, TY_I64, imm(0));
  const rOp1Val = allocReg(0x7f06, TY_I64, imm(0));
  const rOp2Kind = allocReg(0x7f06, TY_I64, imm(0));
  const rOp2Val = allocReg(0x7f06, TY_I64, imm(0));
  const rOp3Kind = allocReg(0x7f06, TY_I64, imm(0));
  const rOp3Val = allocReg(0x7f06, TY_I64, imm(0));

  const s0 = allocReg(0x7f06, TY_I64, imm(0));
  const s1 = allocReg(0x7f06, TY_I64, imm(0));
  const s2 = allocReg(0x7f06, TY_I64, imm(0));
  const s3 = allocReg(0x7f06, TY_I64, imm(0));
  const s4 = allocReg(0x7f06, TY_I64, imm(0));
  const s5 = allocReg(0x7f06, TY_I64, imm(0));
  const s6 = allocReg(0x7f06, TY_I64, imm(0));
  const s7 = allocReg(0x7f06, TY_I64, imm(0));

  const t0 = allocReg(0x7f09, TY_TRIT, [TOK_TRIT_ZERO]);
  const t1 = allocReg(0x7f09, TY_TRIT, [TOK_TRIT_ZERO]);
  const t2 = allocReg(0x7f09, TY_TRIT, [TOK_TRIT_ZERO]);
  const t3 = allocReg(0x7f09, TY_TRIT, [TOK_TRIT_ZERO]);
  const t4 = allocReg(0x7f09, TY_TRIT, [TOK_TRIT_ZERO]);
  const t5 = allocReg(0x7f09, TY_TRIT, [TOK_TRIT_ZERO]);

  const b0 = allocReg(0x7f0d, TY_BYTES, []);
  const b1 = allocReg(0x7f0d, TY_BYTES, []);

  // Pre-size rLblTbl with 64 zero slots
  for (let i = 0; i < 64; i++) {
    setReg(ins, rLblTbl, 0x7f0f, TY_BYTES, [rLblTbl, ...imm(0)]);
  }

  const emitPushU8Imm = (bufReg: number, byteVal: number) => {
    setReg(ins, bufReg, 0x7f0f, TY_BYTES, [bufReg, ...imm(byteVal & 0xff)]);
  };
  const emitPushU32Reg = (bufReg: number, valReg: number) => {
    setReg(ins, bufReg, 0x7f0f, TY_BYTES, [bufReg, ...imm(-4), valReg]);
  };
  const emitPushU64Reg = (bufReg: number, valReg: number) => {
    setReg(ins, bufReg, 0x7f0f, TY_BYTES, [bufReg, ...imm(-8), valReg]);
  };

  const emitSlotDisp = (outReg: number, tokReg: number) => {
    setReg(ins, t0, 0x7f2d, TY_TRIT, [tokReg, ...imm(PAR_BASE)]);
    setReg(ins, s0, 0x7f11, TY_I64, [tokReg, ...imm(PAR_BASE - 1)]);
    setReg(ins, s1, 0x7f12, TY_I64, [s0, ...imm(-8)]);
    setReg(ins, s2, 0x7f11, TY_I64, [tokReg, ...imm(REG_BASE - 1)]);
    setReg(ins, s3, 0x7f12, TY_I64, [s2, ...imm(-8)]);
    setReg(ins, s3, 0x7f11, TY_I64, [s3, ...imm(24)]);
    setReg(ins, outReg, 0x7f45, TY_I64, [t0, s3, s1, s1]);
  };

  const emitDecodeOperandAtCur = (outKindReg: number, outValReg: number) => {
    setReg(ins, s0, 0x7f0e, TY_I64, [pBc, rOpCur]);
    setReg(ins, t0, 0x7f24, TY_TRIT, [s0, ...imm(TOK_INS_END)]);

    setReg(ins, t1, 0x7f26, TY_TRIT, [s0, ...imm(REG_BASE)]);
    setReg(ins, t2, 0x7f27, TY_TRIT, [s0, ...imm(PAR_MAX)]);
    setReg(ins, s1, 0x7f45, TY_I64, [t1, ...imm(1), ...imm(0), ...imm(0)]);
    setReg(ins, s1, 0x7f45, TY_I64, [t2, s1, ...imm(0), ...imm(0)]);
    setReg(ins, outKindReg, 0x7f45, TY_I64, [t0, s1, s1, ...imm(0)]);

    setReg(ins, s3, 0x7f10, TY_I64, [rOpCur, ...imm(1)]);
    setReg(ins, s2, 0x7f0e, TY_I64, [pBc, s3]);
    setReg(ins, t3, 0x7f27, TY_TRIT, [s2, ...imm(32767)]);
    setReg(ins, s4, 0x7f11, TY_I64, [s2, ...imm(32767)]);
    setReg(ins, s4, 0x7f11, TY_I64, [s4, ...imm(32767)]);
    setReg(ins, s4, 0x7f11, TY_I64, [s4, ...imm(2)]);
    setReg(ins, s4, 0x7f45, TY_I64, [t3, s2, s2, s4]);

    setReg(ins, t1, 0x7f24, TY_TRIT, [s0, ...imm(TOK_IMM_I16)]);
    setReg(ins, t2, 0x7f24, TY_TRIT, [s0, ...imm(TOK_TRIT_NEG)]);
    setReg(ins, t3, 0x7f24, TY_TRIT, [s0, ...imm(TOK_TRIT_ZERO)]);
    setReg(ins, t4, 0x7f24, TY_TRIT, [s0, ...imm(TOK_TRIT_POS)]);

    setReg(ins, s5, 0x7f06, TY_I64, [s0]);
    setReg(ins, s5, 0x7f45, TY_I64, [t2, s5, s5, ...imm(-1)]);
    setReg(ins, s5, 0x7f45, TY_I64, [t3, s5, s5, ...imm(0)]);
    setReg(ins, s5, 0x7f45, TY_I64, [t4, s5, s5, ...imm(1)]);
    setReg(ins, s5, 0x7f45, TY_I64, [t1, s5, s5, s4]);
    setReg(ins, outValReg, 0x7f45, TY_I64, [t0, s5, s5, ...imm(0)]);

    setReg(ins, s6, 0x7f10, TY_I64, [rOpCur, ...imm(1)]);
    setReg(ins, s7, 0x7f10, TY_I64, [rOpCur, ...imm(2)]);
    setReg(ins, s6, 0x7f45, TY_I64, [t1, s6, s6, s7]);
    setReg(ins, rOpCur, 0x7f45, TY_I64, [t0, s6, s6, rOpCur]);
  };

  const emitLoadReg64 = (opKindReg: number, opValReg: number, isRcx: boolean) => {
    setReg(ins, t5, 0x7f24, TY_TRIT, [opKindReg, ...imm(1)]);
    emitSlotDisp(s4, opValReg);

    setReg(ins, b0, 0x7f0d, TY_BYTES, []);
    emitPushU8Imm(b0, 0x48);
    emitPushU8Imm(b0, 0x8b);
    emitPushU8Imm(b0, isRcx ? 0x8d : 0x85);
    emitPushU32Reg(b0, s4);

    setReg(ins, b1, 0x7f0d, TY_BYTES, []);
    emitPushU8Imm(b1, 0x48);
    emitPushU8Imm(b1, isRcx ? 0xb9 : 0xb8);
    emitPushU64Reg(b1, opValReg);

    setReg(ins, b0, 0x7f45, TY_BYTES, [t5, b1, b1, b0]);
    setReg(ins, rCodeOut, 0x7f51, TY_BYTES, [rCodeOut, b0]);
  };

  const emitStoreRax = (destTokReg: number) => {
    emitSlotDisp(s4, destTokReg);
    emitPushU8Imm(rCodeOut, 0x48);
    emitPushU8Imm(rCodeOut, 0x89);
    emitPushU8Imm(rCodeOut, 0x85);
    emitPushU32Reg(rCodeOut, s4);
  };

  const emitNormLblIdx = (outIdxReg: number, rawLblValReg: number) => {
    setReg(ins, t0, 0x7f26, TY_TRIT, [rawLblValReg, ...imm(LBL_BASE)]);
    setReg(ins, s0, 0x7f11, TY_I64, [rawLblValReg, ...imm(LBL_BASE)]);
    setReg(ins, outIdxReg, 0x7f45, TY_I64, [t0, s0, s0, rawLblValReg]);
  };

  const emitRel32Lookup = (lblValReg: number) => {
    emitNormLblIdx(s5, lblValReg);
    setReg(ins, s6, 0x7f0e, TY_I64, [rLblTbl, s5]);
    setReg(ins, s7, 0x7f50, TY_I64, [rCodeOut]);
    setReg(ins, s7, 0x7f10, TY_I64, [s7, ...imm(4)]);
    setReg(ins, s6, 0x7f11, TY_I64, [s6, s7]);
    setReg(ins, t0, 0x7f24, TY_TRIT, [rPass, ...imm(0)]);
    setReg(ins, s6, 0x7f45, TY_I64, [t0, s6, s6, ...imm(0)]);
    emitPushU32Reg(rCodeOut, s6);
  };

  const lblPassLoop = 1;
  const lblPassDone = 2;
  const lblInsLoop = 3;
  const lblInsBody = 4;
  const lblInsEndScan = 5;
  const lblScanWord = 6;
  const lblInsNext = 7;
  const lblPassNext = 8;

  emitLabel(ins, lblPassLoop);
  setReg(ins, t0, 0x7f27, TY_TRIT, [rPass, ...imm(1)]);
  emitBrPos(ins, t0, lblPassDone);

  setReg(ins, rCodeOut, 0x7f0d, TY_BYTES, []);
  for (const b of [
    0x55, 0x48, 0x89, 0xe5, 0x48, 0x81, 0xec, 0x00, 0x08, 0x00, 0x00,
    0x48, 0x8d, 0xbd, 0x00, 0xf8, 0xff, 0xff,
    0xb9, 0x00, 0x01, 0x00, 0x00,
    0x31, 0xc0,
    0xf3, 0x48, 0xab,
    0x48, 0xb8,
  ]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitPushU64Reg(rCodeOut, pArg0);
  for (const b of [0x48, 0x89, 0x45, 0xf8, 0x48, 0xb8]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitPushU64Reg(rCodeOut, pArg1);
  for (const b of [0x48, 0x89, 0x45, 0xf0, 0x48, 0xb8]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitPushU64Reg(rCodeOut, pArg2);
  for (const b of [0x48, 0x89, 0x45, 0xe8]) {
    emitPushU8Imm(rCodeOut, b);
  }

  setReg(ins, s0, 0x7f0e, TY_I64, [pBc, ...imm(3)]);
  setReg(ins, s0, 0x7f12, TY_I64, [s0, ...imm(3)]);
  setReg(ins, rPos, 0x7f10, TY_I64, [s0, ...imm(7)]);

  emitLabel(ins, lblInsLoop);
  setReg(ins, t0, 0x7f26, TY_TRIT, [rPos, rBcLen]);
  emitBrTrit(ins, t0, lblPassNext, lblPassNext, lblInsBody);

  emitLabel(ins, lblInsBody);
  setReg(ins, rFirstWord, 0x7f0e, TY_I64, [pBc, rPos]);
  setReg(ins, t0, 0x7f24, TY_TRIT, [rFirstWord, ...imm(TOK_MOD_END)]);
  emitBrPos(ins, t0, lblPassNext);
  setReg(ins, t0, 0x7f24, TY_TRIT, [rFirstWord, ...imm(TOK_FN_DEF)]);
  emitBrPos(ins, t0, lblPassNext);

  setReg(ins, t0, 0x7f27, TY_TRIT, [rFirstWord, ...imm(REG_MAX)]);
  setReg(ins, s1, 0x7f10, TY_I64, [rPos, ...imm(1)]);
  setReg(ins, s0, 0x7f0e, TY_I64, [pBc, s1]);
  setReg(ins, s2, 0x7f10, TY_I64, [rPos, ...imm(4)]);

  setReg(ins, rDest, 0x7f45, TY_I64, [t0, rFirstWord, rFirstWord, ...imm(0)]);
  setReg(ins, rOp, 0x7f45, TY_I64, [t0, s0, s0, rFirstWord]);
  setReg(ins, rOpCur, 0x7f45, TY_I64, [t0, s2, s2, s1]);

  emitDecodeOperandAtCur(rOp0Kind, rOp0Val);
  emitDecodeOperandAtCur(rOp1Kind, rOp1Val);
  emitDecodeOperandAtCur(rOp2Kind, rOp2Val);
  emitDecodeOperandAtCur(rOp3Kind, rOp3Val);

  let nextDispLbl = 14;
  const allocLbl = (): number => nextDispLbl++;

  const l7f40 = allocLbl();
  const l7f41 = allocLbl();
  const l7f43 = allocLbl();
  const l7f44 = allocLbl();
  const l7f42 = allocLbl();
  const lCopy = allocLbl();
  const l7f10 = allocLbl();
  const l7f11 = allocLbl();
  const l7f12 = allocLbl();
  const l7f13 = allocLbl();
  const l7f14 = allocLbl();
  const l7f15 = allocLbl();
  const l7f16 = allocLbl();
  const lMax = allocLbl();
  const lMin = allocLbl();
  const l7f1e = allocLbl();
  const l7f1f = allocLbl();
  const l7f20 = allocLbl();
  const l7f21 = allocLbl();
  const l7f22 = allocLbl();
  const l7f23 = allocLbl();
  const l7f24 = allocLbl();
  const l7f25 = allocLbl();
  const l7f26 = allocLbl();
  const l7f27 = allocLbl();
  const l7f2b = allocLbl();
  const l7f2c = allocLbl();
  const l7f2d = allocLbl();
  const lSign = allocLbl();
  const l7f45 = allocLbl();
  const l7f4a = allocLbl();

  const emitOpBranch = (opCode: number, lblMatch: number) => {
    setReg(ins, t0, 0x7f24, TY_TRIT, [rOp, ...imm(opCode)]);
    emitBrPos(ins, t0, lblMatch);
  };

  emitOpBranch(0x7f40, l7f40);
  emitOpBranch(0x7f41, l7f41);
  emitOpBranch(0x7f43, l7f43);
  emitOpBranch(0x7f44, l7f44);
  emitOpBranch(0x7f42, l7f42);
  for (const opC of [0x7f01, 0x7f02, 0x7f03, 0x7f04, 0x7f06, 0x7f09, 0x7f57, 0x7f58, 0x7f59]) {
    emitOpBranch(opC, lCopy);
  }
  emitOpBranch(0x7f10, l7f10);
  emitOpBranch(0x7f11, l7f11);
  emitOpBranch(0x7f12, l7f12);
  emitOpBranch(0x7f13, l7f13);
  emitOpBranch(0x7f14, l7f14);
  emitOpBranch(0x7f15, l7f15);
  emitOpBranch(0x7f28, l7f15);
  emitOpBranch(0x7f16, l7f16);
  emitOpBranch(0x7f17, lMax);
  emitOpBranch(0x7f29, lMax);
  emitOpBranch(0x7f18, lMin);
  emitOpBranch(0x7f2a, lMin);
  emitOpBranch(0x7f1e, l7f1e);
  emitOpBranch(0x7f1f, l7f1f);
  emitOpBranch(0x7f20, l7f20);
  emitOpBranch(0x7f21, l7f21);
  emitOpBranch(0x7f22, l7f22);
  emitOpBranch(0x7f23, l7f23);
  emitOpBranch(0x7f24, l7f24);
  emitOpBranch(0x7f25, l7f25);
  emitOpBranch(0x7f26, l7f26);
  emitOpBranch(0x7f27, l7f27);
  emitOpBranch(0x7f2b, l7f2b);
  emitOpBranch(0x7f2c, l7f2c);
  emitOpBranch(0x7f2d, l7f2d);
  emitOpBranch(0x7f35, lSign);
  emitOpBranch(0x7f3f, lSign);
  emitOpBranch(0x7f45, l7f45);
  emitOpBranch(0x7f4a, l7f4a);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, l7f40);
  emitNormLblIdx(s5, rOp0Val);
  setReg(ins, s6, 0x7f50, TY_I64, [rCodeOut]);
  setReg(ins, rLblTbl, 0x7f0f, TY_BYTES, [rLblTbl, ...imm(-9), s5, s6]);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, l7f41);
  emitPushU8Imm(rCodeOut, 0xe9);
  emitRel32Lookup(rOp0Val);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, l7f43);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  for (const b of [0x48, 0x83, 0xf8, 0x01, 0x0f, 0x84]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitRel32Lookup(rOp1Val);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, l7f44);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  for (const b of [0x48, 0x83, 0xf8, 0xff, 0x0f, 0x84]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitRel32Lookup(rOp1Val);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, l7f42);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  for (const b of [0x48, 0x85, 0xc0, 0x0f, 0x8c]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitRel32Lookup(rOp1Val);
  for (const b of [0x0f, 0x84]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitRel32Lookup(rOp2Val);
  emitPushU8Imm(rCodeOut, 0xe9);
  emitRel32Lookup(rOp3Val);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, lCopy);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  emitStoreRax(rDest);
  emitJmp(ins, lblInsEndScan);

  const emitBinOpBlock = (lbl: number, x86Bytes: number[]) => {
    emitLabel(ins, lbl);
    emitLoadReg64(rOp0Kind, rOp0Val, false);
    emitLoadReg64(rOp1Kind, rOp1Val, true);
    for (const b of x86Bytes) {
      emitPushU8Imm(rCodeOut, b);
    }
    emitStoreRax(rDest);
    emitJmp(ins, lblInsEndScan);
  };

  emitBinOpBlock(l7f10, [0x48, 0x01, 0xc8]);
  emitBinOpBlock(l7f11, [0x48, 0x29, 0xc8]);
  emitBinOpBlock(l7f12, [0x48, 0x0f, 0xaf, 0xc1]);

  const divSeq = [
    0x48, 0x85, 0xc9, 0x75, 0x05, 0x48, 0x31, 0xc0, 0xeb, 0x1e,
    0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
    0x48, 0x39, 0xd0, 0x75, 0x0c, 0x48, 0x83, 0xf9, 0xff, 0x75, 0x06,
    0x48, 0x31, 0xd2, 0xeb, 0x05, 0x48, 0x99, 0x48, 0xf7, 0xf9,
  ];
  emitBinOpBlock(l7f13, divSeq);
  emitBinOpBlock(l7f14, [...divSeq, 0x48, 0x89, 0xd0]);

  emitLabel(ins, l7f15);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  for (const b of [0x48, 0xf7, 0xd8]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitStoreRax(rDest);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, l7f16);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  for (const b of [0x48, 0x89, 0xc2, 0x48, 0xf7, 0xda, 0x48, 0x0f, 0x48, 0xc2]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitStoreRax(rDest);
  emitJmp(ins, lblInsEndScan);

  emitBinOpBlock(lMax, [0x48, 0x39, 0xc8, 0x48, 0x0f, 0x4f, 0xc1]);
  emitBinOpBlock(lMin, [0x48, 0x39, 0xc8, 0x48, 0x0f, 0x4c, 0xc1]);
  emitBinOpBlock(l7f1e, [0x48, 0x21, 0xc8]);
  emitBinOpBlock(l7f1f, [0x48, 0x09, 0xc8]);
  emitBinOpBlock(l7f20, [0x48, 0x31, 0xc8]);

  emitLabel(ins, l7f21);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  for (const b of [0x48, 0xf7, 0xd0]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitStoreRax(rDest);
  emitJmp(ins, lblInsEndScan);

  emitBinOpBlock(l7f22, [0x80, 0xe1, 0x3f, 0x48, 0xd3, 0xe0]);
  emitBinOpBlock(l7f23, [0x80, 0xe1, 0x3f, 0x48, 0xd3, 0xf8]);

  for (const [lblCmp, setcc] of [
    [l7f24, 0x94],
    [l7f25, 0x95],
    [l7f26, 0x9c],
    [l7f27, 0x9f],
  ] as const) {
    emitBinOpBlock(lblCmp, [0x48, 0x39, 0xc8, 0x0f, setcc, 0xc0, 0x0f, 0xb6, 0xc0, 0x48, 0x8d, 0x44, 0x00, 0xff]);
  }

  emitBinOpBlock(l7f2b, [0x48, 0x0f, 0xaf, 0xc1, 0x48, 0xf7, 0xd8]);

  emitLabel(ins, l7f2c);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  emitLoadReg64(rOp1Kind, rOp1Val, true);
  for (const b of [0x48, 0x01, 0xc8]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitLoadReg64(rOp2Kind, rOp2Val, true);
  for (const b of [
    0x48, 0x01, 0xc8, 0x48, 0x85, 0xc0, 0x0f, 0x9f, 0xc2, 0x0f, 0x9c, 0xc0,
    0x0f, 0xb6, 0xd2, 0x0f, 0xb6, 0xc0, 0x48, 0x29, 0xc2, 0x48, 0x89, 0xd0,
  ]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitStoreRax(rDest);
  emitJmp(ins, lblInsEndScan);

  emitBinOpBlock(l7f2d, [
    0x48, 0x39, 0xc8, 0x0f, 0x9f, 0xc2, 0x0f, 0x9c, 0xc0,
    0x0f, 0xb6, 0xd2, 0x0f, 0xb6, 0xc0, 0x48, 0x29, 0xc2, 0x48, 0x89, 0xd0,
  ]);

  emitLabel(ins, lSign);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  for (const b of [
    0x48, 0x85, 0xc0, 0x0f, 0x9f, 0xc2, 0x0f, 0x9c, 0xc0,
    0x0f, 0xb6, 0xd2, 0x0f, 0xb6, 0xc0, 0x48, 0x29, 0xc2, 0x48, 0x89, 0xd0,
  ]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitStoreRax(rDest);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, l7f45);
  emitLoadReg64(rOp0Kind, rOp0Val, true);
  emitLoadReg64(rOp2Kind, rOp2Val, false);
  emitLoadReg64(rOp1Kind, rOp1Val, true);
  for (const b of [0x48, 0x89, 0xca]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitLoadReg64(rOp0Kind, rOp0Val, true);
  for (const b of [0x48, 0x85, 0xc9, 0x48, 0x0f, 0x4c, 0xc2]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitLoadReg64(rOp3Kind, rOp3Val, true);
  for (const b of [0x48, 0x89, 0xca]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitLoadReg64(rOp0Kind, rOp0Val, true);
  for (const b of [0x48, 0x85, 0xc9, 0x48, 0x0f, 0x4f, 0xc2]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitStoreRax(rDest);
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, l7f4a);
  emitLoadReg64(rOp0Kind, rOp0Val, false);
  for (const b of [0x48, 0x89, 0xc7, 0xb8, 0x3c, 0x00, 0x00, 0x00, 0x0f, 0x05]) {
    emitPushU8Imm(rCodeOut, b);
  }
  emitJmp(ins, lblInsEndScan);

  emitLabel(ins, lblInsEndScan);
  setReg(ins, rPos, 0x7f06, TY_I64, [rOpCur]);
  emitLabel(ins, lblScanWord);
  setReg(ins, s0, 0x7f0e, TY_I64, [pBc, rPos]);
  setReg(ins, t0, 0x7f24, TY_TRIT, [s0, ...imm(TOK_INS_END)]);
  setReg(ins, t1, 0x7f24, TY_TRIT, [s0, ...imm(TOK_IMM_I16)]);
  setReg(ins, t2, 0x7f24, TY_TRIT, [s0, ...imm(TOK_IMM_I32)]);
  setReg(ins, s1, 0x7f10, TY_I64, [rPos, ...imm(1)]);
  setReg(ins, s2, 0x7f10, TY_I64, [rPos, ...imm(2)]);
  setReg(ins, s3, 0x7f10, TY_I64, [rPos, ...imm(3)]);
  setReg(ins, s1, 0x7f45, TY_I64, [t1, s1, s1, s2]);
  setReg(ins, rPos, 0x7f45, TY_I64, [t2, s1, s1, s3]);
  emitBrPos(ins, t0, lblInsNext);
  emitJmp(ins, lblScanWord);

  emitLabel(ins, lblInsNext);
  emitJmp(ins, lblInsLoop);

  emitLabel(ins, lblPassNext);
  for (const b of [0x31, 0xff, 0xb8, 0x3c, 0x00, 0x00, 0x00, 0x0f, 0x05]) {
    emitPushU8Imm(rCodeOut, b);
  }
  setReg(ins, rPass, 0x7f10, TY_I64, [rPass, ...imm(1)]);
  emitJmp(ins, lblPassLoop);

  emitLabel(ins, lblPassDone);
  setReg(ins, s0, 0x7f50, TY_I64, [rCodeOut]);
  setReg(ins, s0, 0x7f10, TY_I64, [s0, ...imm(120)]);

  setReg(ins, b0, 0x7f0d, TY_BYTES, []);
  const elfPrefix: number[] = [
    0x7f, 0x45, 0x4c, 0x46, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0x3e, 0, 1, 0, 0, 0,
    0x78, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
    64, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 64, 0, 56, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 5, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00,
  ];
  for (const b of elfPrefix) {
    emitPushU8Imm(b0, b);
  }
  emitPushU64Reg(b0, s0);
  emitPushU64Reg(b0, s0);
  for (const b of [0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]) {
    emitPushU8Imm(b0, b);
  }
  setReg(ins, b0, 0x7f51, TY_BYTES, [b0, rCodeOut]);
  ins.push([0x7f4a, b0, TOK_INS_END]);

  const out = packFnModule(
    [],
    0,
    [
      [pBc, TY_BYTES, TC_SAFE],
      [pArg0, TY_I64, TC_SAFE],
      [pArg1, TY_I64, TC_SAFE],
      [pArg2, TY_I64, TC_SAFE],
    ],
    TY_BYTES,
    0,
    ins,
  );
  cachedStage1Stream = out;
  return [...out];
}

export function compileOmniSurfaceViaStage1ToElf(
  u16Stream: number[],
  arg0: bigint,
  arg1: bigint,
  arg2: bigint,
): Uint8Array {
  const stage1Stream = buildStep7Stage1CompilerStream();
  const stage1Mod = decodeOmniTokenStream(stage1Stream);
  const stage1Fn = stage1Mod.functions[0];
  if (!stage1Fn) throw new Error("Stage-1 compiler module missing entry function");
  const rep = verifyBinaryFunction(stage1Fn);
  if (!rep.passed) {
    throw new Error(`Stage-1 compiler failed static verification: ${JSON.stringify(rep.errors)}`);
  }

  const vm = new OmniBinaryVm(50_000_000, 256 * 1024 * 1024);
  const res = vm.executeFunction(stage1Mod, 0, [
    { kind: "U16Array", value: u16Stream.map((w) => w & 0xffff) },
    { kind: "Int", value: arg0 },
    { kind: "Int", value: arg1 },
    { kind: "Int", value: arg2 },
  ]);
  if (res.kind !== "Bytes") {
    throw new Error(`Stage-1 compiler returned unexpected value kind: ${res.kind}`);
  }
  return Uint8Array.from(res.value.map((b) => b & 0xff));
}

// ==============================================================================
// SINGLE- AND MULTI-FUNCTION WEBASSEMBLY v1 COMPILER & EVALUATOR
// ==============================================================================

function compileSingleBinFunctionToWasmBody(
  func: BinFunction,
  fnIdToWasmIdx: Map<number, number>,
): number[] {
  const rep = verifyBinaryFunction(func);
  if (!rep.passed) {
    throw new Error(`Cannot Wasm-compile unverified binary stream: ${JSON.stringify(rep.errors)}`);
  }

  for (const ins of func.instructions) {
    if (!isWasmAotSupportedOpcode(ins.opTok) && ins.opTok !== 0x7f48) {
      const lbl = decodeOpcodeToken(ins.opTok)?.debugLabel ?? "OP_UNKNOWN";
      throw new Error(
        `Opcode 0x${ins.opTok.toString(16).toUpperCase().padStart(4, "0")} (${lbl}) requires high-level VM runtime state and is not lowered by standalone WebAssembly AOT`,
      );
    }
  }

  const numParams = func.params.length;
  let maxRegUsed = 0;
  for (const ins of func.instructions) {
    if (ins.destReg !== null) {
      maxRegUsed = Math.max(maxRegUsed, ins.destReg - REG_BASE + 1);
    }
  }
  const numLocals = maxRegUsed + 4;
  const scratch0 = numParams + maxRegUsed;
  const scratch1 = numParams + maxRegUsed + 1;
  const bufLenLocal = numParams + maxRegUsed + 2;
  const pcLocal = numParams + maxRegUsed + 3;

  const hasControlFlow = func.instructions.some((ins) => ins.opTok >= 0x7f40 && ins.opTok <= 0x7f44);
  const labelPcs = new Map<number, number>();
  func.instructions.forEach((ins, idx) => {
    if (ins.opTok === 0x7f40 && ins.operands[0]) {
      const lbl = extractLabelId(ins.operands[0]);
      if (lbl !== null) labelPcs.set(lbl, idx);
    }
  });

  const lookupLblPc = (lbl: number): number => {
    const pc = labelPcs.get(lbl);
    if (pc === undefined) throw new Error(`Unresolved Wasm label 0x${lbl.toString(16).toUpperCase()}`);
    return pc;
  };

  const body: number[] = [];
  body.push(...encodeUleb128(1), ...encodeUleb128(numLocals), 0x7e);

  const emitWasmOp = (b: number[], op: BinOperand) => {
    if (op.kind === "Param") {
      b.push(0x20, ...encodeUleb128(op.value - PAR_BASE));
    } else if (op.kind === "Reg") {
      b.push(0x20, ...encodeUleb128(numParams + (op.value - REG_BASE)));
    } else if (op.kind === "ImmInt") {
      b.push(0x42, ...encodeSleb128(op.value));
    } else if (op.kind === "ImmTrit") {
      b.push(0x42, ...encodeSleb128(BigInt(op.value)));
    } else {
      b.push(0x42, 0x00);
    }
  };

  const emitWasmSetDest = (b: number[], dest: number | null) => {
    if (dest !== null) {
      b.push(0x21, ...encodeUleb128(numParams + (dest - REG_BASE)));
    }
  };

  const regPack5 = collectFuncPack5Map(func);
  validateFuncPack5Ops(func, regPack5);

  const emitExtractPack5TritWasm = (b: number[], op: BinOperand, tritIdx: number) => {
    let directByte = false;
    if (op.kind === "Reg") {
      const entry = regPack5.get(op.value);
      if (entry) {
        const bVal = BigInt(entry.bytes[Math.floor(tritIdx / 5)] ?? 121);
        b.push(0x42, ...encodeSleb128(bVal));
        directByte = true;
      }
    }
    if (!directByte) {
      emitWasmOp(b, op);
      const shift = BigInt((Math.floor(tritIdx / 5) + 1) * 8);
      b.push(0x42, ...encodeSleb128(shift), 0x87, 0x42, ...encodeSleb128(255n), 0x83);
    }
    const pow3 = [1n, 3n, 9n, 27n, 81n][tritIdx % 5];
    if (pow3 > 1n) {
      b.push(0x42, ...encodeSleb128(pow3), 0x7f);
    }
    b.push(0x42, 0x03, 0x81, 0x42, 0x01, 0x7d);
  };

  const emitStraightIns = (b: number[], ins: BinInstruction) => {
    switch (ins.opTok) {
      case 0x7f00:
      case 0x7f05:
        break;
      case 0x7f01:
      case 0x7f02:
      case 0x7f03:
      case 0x7f04:
      case 0x7f06:
      case 0x7f09:
      case 0x7f57:
      case 0x7f58:
      case 0x7f59:
        emitWasmOp(b, ins.operands[0]);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f30:
      case 0x7f31: {
        let word = 0n;
        if (ins.destReg !== null) {
          const entry = regPack5.get(ins.destReg);
          if (entry) word = pack5WordFromBytes(entry.bytes, entry.nTrits);
        }
        b.push(0x42, ...encodeSleb128(word));
        emitWasmSetDest(b, ins.destReg);
        break;
      }
      case 0x7f34: {
        let nTrits = 0;
        if (ins.operands[0]?.kind === "Reg" && ins.operands[1]?.kind === "Reg") {
          const na = regPack5.get(ins.operands[0].value)?.nTrits ?? 0;
          const nb = regPack5.get(ins.operands[1].value)?.nTrits ?? 0;
          nTrits = Math.min(na, nb);
        }
        b.push(0x42, 0x00);
        for (let idx = 0; idx < nTrits; idx++) {
          emitExtractPack5TritWasm(b, ins.operands[0], idx);
          emitExtractPack5TritWasm(b, ins.operands[1], idx);
          b.push(0x7e, 0x7c);
        }
        emitWasmSetDest(b, ins.destReg);
        break;
      }
      case 0x7f35:
      case 0x7f3f: {
        const pack5Len = ins.operands[0]?.kind === "Reg" ? regPack5.get(ins.operands[0].value)?.nTrits : undefined;
        if (pack5Len !== undefined) {
          b.push(0x42, 0x00);
          for (let idx = 0; idx < pack5Len; idx++) {
            emitExtractPack5TritWasm(b, ins.operands[0], idx);
            b.push(0x7c);
          }
        } else {
          emitWasmOp(b, ins.operands[0]);
        }
        b.push(0x22, ...encodeUleb128(scratch0));
        b.push(0x42, 0x00, 0x55, 0xad);
        b.push(0x20, ...encodeUleb128(scratch0));
        b.push(0x42, 0x00, 0x53, 0xad, 0x7d);
        emitWasmSetDest(b, ins.destReg);
        break;
      }
      case 0x7f0d:
        b.push(0x42, 0x00, 0x21, ...encodeUleb128(bufLenLocal));
        if (ins.destReg !== null) {
          b.push(0x42, 0x00);
          emitWasmSetDest(b, ins.destReg);
        }
        break;
      case 0x7f0e: {
        let handled = false;
        const arrOp = ins.operands[0];
        const idxOp = ins.operands[1];
        if (arrOp?.kind === "Reg" && idxOp) {
          const entry = regPack5.get(arrOp.value);
          if (entry) {
            if (idxOp.kind === "ImmInt" && idxOp.value >= 0n) {
              const idxNum = Number(idxOp.value);
              if (entry.isTv) {
                emitExtractPack5TritWasm(b, arrOp, idxNum);
              } else {
                const bVal = BigInt(entry.bytes[idxNum] ?? 0);
                b.push(0x42, ...encodeSleb128(bVal));
              }
              emitWasmSetDest(b, ins.destReg);
              handled = true;
            } else {
              b.push(0x42, 0x00);
              for (let k = 0; k < entry.nTrits; k++) {
                if (entry.isTv) {
                  emitExtractPack5TritWasm(b, arrOp, k);
                } else {
                  const bVal = BigInt(entry.bytes[k] ?? 0);
                  b.push(0x42, ...encodeSleb128(bVal));
                }
                emitWasmOp(b, idxOp);
                b.push(0x42, ...encodeSleb128(BigInt(k)), 0x51, 0xad, 0x7e, 0x7c);
              }
              emitWasmSetDest(b, ins.destReg);
              handled = true;
            }
          }
        }
        if (!handled) {
          b.push(0x42, 0x00);
          emitWasmSetDest(b, ins.destReg);
        }
        break;
      }
      case 0x7f0f: {
        let delta = 0n;
        if (
          ins.operands.length === 3 &&
          ins.operands[1].kind === "ImmInt" &&
          ins.operands[1].value === -4n
        ) {
          delta = 4n;
        } else if (
          ins.operands.length === 3 &&
          ins.operands[1].kind === "ImmInt" &&
          ins.operands[1].value === -8n
        ) {
          delta = 8n;
        } else if (
          ins.operands.length === 4 &&
          ins.operands[1].kind === "ImmInt" &&
          ins.operands[1].value === -9n
        ) {
          delta = 0n;
        } else {
          delta = BigInt(Math.max(0, ins.operands.length - 1));
        }
        if (delta !== 0n) {
          b.push(0x20, ...encodeUleb128(bufLenLocal));
          b.push(0x42, ...encodeSleb128(delta), 0x7c);
          b.push(0x21, ...encodeUleb128(bufLenLocal));
        }
        if (ins.destReg !== null) {
          b.push(0x20, ...encodeUleb128(bufLenLocal));
          emitWasmSetDest(b, ins.destReg);
        }
        break;
      }
      case 0x7f50:
        if (ins.destReg !== null) {
          b.push(0x20, ...encodeUleb128(bufLenLocal));
          emitWasmSetDest(b, ins.destReg);
        }
        break;
      case 0x7f10:
      case 0x7f11:
      case 0x7f12:
      case 0x7f1e:
      case 0x7f1f:
      case 0x7f20:
      case 0x7f22:
      case 0x7f23: {
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[1]);
        const wasmOp =
          ins.opTok === 0x7f10
            ? 0x7c
            : ins.opTok === 0x7f11
              ? 0x7d
              : ins.opTok === 0x7f12
                ? 0x7e
                : ins.opTok === 0x7f1e
                  ? 0x83
                  : ins.opTok === 0x7f1f
                    ? 0x84
                    : ins.opTok === 0x7f20
                      ? 0x85
                      : ins.opTok === 0x7f22
                        ? 0x86
                        : 0x87;
        b.push(wasmOp);
        emitWasmSetDest(b, ins.destReg);
        break;
      }
      case 0x7f13:
      case 0x7f14: {
        emitWasmOp(b, ins.operands[0]);
        b.push(0x21, ...encodeUleb128(scratch0));
        emitWasmOp(b, ins.operands[1]);
        b.push(0x21, ...encodeUleb128(scratch1));
        if (ins.opTok === 0x7f13) b.push(0x42, 0x00);
        b.push(0x20, ...encodeUleb128(scratch0));
        b.push(0x42, 0x01);
        b.push(0x20, ...encodeUleb128(scratch1));
        b.push(0x20, ...encodeUleb128(scratch1), 0x50);
        b.push(0x20, ...encodeUleb128(scratch0), 0x42, ...encodeSleb128(I64_MIN), 0x51);
        b.push(0x20, ...encodeUleb128(scratch1), 0x42, 0x7f, 0x51);
        b.push(0x71, 0x72, 0x1b);
        if (ins.opTok === 0x7f13) {
          b.push(0x7f, 0x20, ...encodeUleb128(scratch1), 0x50, 0x1b);
        } else {
          b.push(0x81);
        }
        emitWasmSetDest(b, ins.destReg);
        break;
      }
      case 0x7f15:
      case 0x7f28:
        b.push(0x42, 0x00);
        emitWasmOp(b, ins.operands[0]);
        b.push(0x7d);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f16:
        b.push(0x42, 0x00);
        emitWasmOp(b, ins.operands[0]);
        b.push(0x7d);
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[0]);
        b.push(0x42, 0x00, 0x53, 0x1b);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f21:
        emitWasmOp(b, ins.operands[0]);
        b.push(0x42, 0x7f, 0x85);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f17:
      case 0x7f29:
      case 0x7f18:
      case 0x7f2a:
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[1]);
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[1]);
        b.push(ins.opTok === 0x7f17 || ins.opTok === 0x7f29 ? 0x53 : 0x55, 0x1b);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f2b:
        b.push(0x42, 0x00);
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[1]);
        b.push(0x7e, 0x7d);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f24:
      case 0x7f25:
      case 0x7f26:
      case 0x7f27: {
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[1]);
        const cmpOp = ins.opTok === 0x7f24 ? 0x51 : ins.opTok === 0x7f25 ? 0x52 : ins.opTok === 0x7f26 ? 0x53 : 0x55;
        b.push(cmpOp, 0xad, 0x42, 0x02, 0x7e, 0x42, 0x01, 0x7d);
        emitWasmSetDest(b, ins.destReg);
        break;
      }
      case 0x7f2c:
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[1]);
        b.push(0x7c);
        emitWasmOp(b, ins.operands[2]);
        b.push(0x7c, 0x22, ...encodeUleb128(scratch0));
        b.push(0x42, 0x00, 0x55, 0xad);
        b.push(0x20, ...encodeUleb128(scratch0), 0x42, 0x00, 0x53, 0xad, 0x7d);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f2d:
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[1]);
        b.push(0x55, 0xad);
        emitWasmOp(b, ins.operands[0]);
        emitWasmOp(b, ins.operands[1]);
        b.push(0x53, 0xad, 0x7d);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f45:
        emitWasmOp(b, ins.operands[1]);
        emitWasmOp(b, ins.operands[2]);
        emitWasmOp(b, ins.operands[0]);
        b.push(0x42, 0x00, 0x53, 0x1b, 0x21, ...encodeUleb128(scratch1));
        emitWasmOp(b, ins.operands[3]);
        b.push(0x20, ...encodeUleb128(scratch1));
        emitWasmOp(b, ins.operands[0]);
        b.push(0x42, 0x00, 0x55, 0x1b);
        emitWasmSetDest(b, ins.destReg);
        break;
      case 0x7f48: {
        // CALL <callee_fn_id> <arg0> ...
        const calleeOp = ins.operands[0];
        const calleeId =
          calleeOp && (calleeOp.kind === "FnRef" || calleeOp.kind === "ImmInt")
            ? Number(calleeOp.value)
            : 0;
        const wasmFnIdx = fnIdToWasmIdx.get(calleeId);
        if (wasmFnIdx === undefined) {
          throw new Error(`Unresolved Wasm CALL target fn #${calleeId}`);
        }
        for (const argOp of ins.operands.slice(1)) {
          emitWasmOp(b, argOp);
        }
        b.push(0x10, ...encodeUleb128(wasmFnIdx));
        emitWasmSetDest(b, ins.destReg);
        break;
      }
      case 0x7f4a:
        emitWasmOp(b, ins.operands[0]);
        b.push(0x0f);
        break;
      default:
        break;
    }
  };

  if (!hasControlFlow) {
    for (const ins of func.instructions) {
      emitStraightIns(body, ins);
    }
  } else {
    body.push(0x03, 0x40);
    func.instructions.forEach((ins, insIdx) => {
      body.push(0x20, ...encodeUleb128(pcLocal));
      body.push(0x42, ...encodeSleb128(BigInt(insIdx)), 0x51, 0x04, 0x40);

      if (ins.opTok === 0x7f40) {
        body.push(0x42, ...encodeSleb128(BigInt(insIdx + 1)), 0x21, ...encodeUleb128(pcLocal), 0x0c, 0x01);
      } else if (ins.opTok === 0x7f41) {
        const lbl = ins.operands[0] ? extractLabelId(ins.operands[0]) : null;
        if (lbl === null) throw new Error("Missing JMP label");
        body.push(0x42, ...encodeSleb128(BigInt(lookupLblPc(lbl))), 0x21, ...encodeUleb128(pcLocal), 0x0c, 0x01);
      } else if (ins.opTok === 0x7f43 || ins.opTok === 0x7f44) {
        const lbl = ins.operands[1] ? extractLabelId(ins.operands[1]) : null;
        if (lbl === null) throw new Error("Missing BR label");
        body.push(0x42, ...encodeSleb128(BigInt(lookupLblPc(lbl))));
        body.push(0x42, ...encodeSleb128(BigInt(insIdx + 1)));
        emitWasmOp(body, ins.operands[0]);
        body.push(0x42, ...encodeSleb128(ins.opTok === 0x7f43 ? 1n : -1n), 0x51, 0x1b);
        body.push(0x21, ...encodeUleb128(pcLocal), 0x0c, 0x01);
      } else if (ins.opTok === 0x7f42) {
        const lNeg = ins.operands[1] ? extractLabelId(ins.operands[1]) : null;
        const lZero = ins.operands[2] ? extractLabelId(ins.operands[2]) : null;
        const lPos = ins.operands[3] ? extractLabelId(ins.operands[3]) : null;
        if (lNeg === null || lZero === null || lPos === null) throw new Error("Missing BR_TRIT labels");
        body.push(0x42, ...encodeSleb128(BigInt(lookupLblPc(lNeg))));
        body.push(0x42, ...encodeSleb128(BigInt(lookupLblPc(lZero))));
        emitWasmOp(body, ins.operands[0]);
        body.push(0x42, 0x00, 0x53, 0x1b, 0x21, ...encodeUleb128(scratch1));
        body.push(0x42, ...encodeSleb128(BigInt(lookupLblPc(lPos))));
        body.push(0x20, ...encodeUleb128(scratch1));
        emitWasmOp(body, ins.operands[0]);
        body.push(0x42, 0x00, 0x55, 0x1b);
        body.push(0x21, ...encodeUleb128(pcLocal), 0x0c, 0x01);
      } else if (ins.opTok === 0x7f4a) {
        emitWasmOp(body, ins.operands[0]);
        body.push(0x0f);
      } else {
        emitStraightIns(body, ins);
        body.push(0x42, ...encodeSleb128(BigInt(insIdx + 1)), 0x21, ...encodeUleb128(pcLocal), 0x0c, 0x01);
      }

      body.push(0x0b);
    });
    body.push(0x0b, 0x42, 0x00);
  }
  body.push(0x0b);
  return body;
}

export function compilePureU16ToWasm(stream: number[]): Uint8Array {
  const mod = decodeOmniTokenStream(stream);
  if (mod.functions.length === 0) throw new Error("No function in module");

  const fnIdToWasmIdx = new Map<number, number>();
  mod.functions.forEach((f, idx) => {
    fnIdToWasmIdx.set(f.fnIdx, idx);
  });

  const fnCnt = mod.functions.length;
  const typeSec: number[] = [...encodeUleb128(fnCnt)];
  const funcSec: number[] = [...encodeUleb128(fnCnt)];
  const codeSec: number[] = [...encodeUleb128(fnCnt)];

  for (let idx = 0; idx < fnCnt; idx++) {
    const func = mod.functions[idx];
    const numParams = func.params.length;
    typeSec.push(0x60, ...encodeUleb128(numParams), ...Array(numParams).fill(0x7e), 0x01, 0x7e);
    funcSec.push(...encodeUleb128(idx));
    const body = compileSingleBinFunctionToWasmBody(func, fnIdToWasmIdx);
    codeSec.push(...encodeUleb128(body.length), ...body);
  }

  const entryWasmIdx = fnIdToWasmIdx.get(0) ?? fnCnt - 1;
  const expSec = [0x01, 0x07, 0x6f, 0x6d, 0x6e, 0x69, 0x5f, 0x66, 0x6e, 0x00, ...encodeUleb128(entryWasmIdx)];

  const wasm: number[] = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
  const sections: Array<[number, number[]]> = [
    [1, typeSec],
    [3, funcSec],
    [7, expSec],
    [10, codeSec],
  ];
  for (const [secId, payload] of sections) {
    wasm.push(secId, ...encodeUleb128(payload.length), ...payload);
  }
  return Uint8Array.from(wasm);
}

function skipWasmToMatchingEnd(body: Uint8Array, pos: number): number {
  let depth = 1;
  while (pos < body.length) {
    const op = body[pos++];
    if (op === 0x02 || op === 0x03 || op === 0x04) {
      if (pos >= body.length) throw new Error("Truncated Wasm block header during skip");
      pos += 1;
      depth += 1;
    } else if (op === 0x0b) {
      depth -= 1;
      if (depth === 0) return pos;
    } else if (op === 0x0c || op === 0x10 || op === 0x20 || op === 0x21 || op === 0x22) {
      const st = { pos };
      decodeUleb128(body, st);
      pos = st.pos;
    } else if (op === 0x42) {
      const st = { pos };
      decodeSleb128(body, st);
      pos = st.pos;
    }
  }
  throw new Error("Unclosed Wasm block while skipping to end");
}

export function executeWasmModuleI64WithBudgets(
  wasm: Uint8Array,
  args: bigint[],
  stepLimit = DEFAULT_WASM_STEP_LIMIT,
  memoryBudgetBytes = DEFAULT_WASM_MEMORY_BUDGET_BYTES,
): bigint {
  if (
    wasm.length < 8 ||
    wasm[0] !== 0x00 ||
    wasm[1] !== 0x61 ||
    wasm[2] !== 0x73 ||
    wasm[3] !== 0x6d ||
    wasm[4] !== 0x01
  ) {
    throw new Error("Invalid WebAssembly '\\0asm' v1 header");
  }

  if (typeof WebAssembly !== "undefined" && typeof WebAssembly.validate === "function") {
    const copy = new Uint8Array(wasm.length);
    copy.set(wasm);
    if (!WebAssembly.validate(copy)) {
      throw new Error("Generated WebAssembly module failed native WebAssembly.validate()");
    }
  }

  const st = { pos: 8 };
  const paramCounts: number[] = [];
  const codeBodies: Uint8Array[] = [];
  let entryFnIdx = 0;

  while (st.pos < wasm.length) {
    const secId = wasm[st.pos++];
    const secLen = decodeUleb128(wasm, st);
    const secEnd = st.pos + secLen;
    if (secEnd > wasm.length) throw new Error("Truncated Wasm section");

    if (secId === 1) {
      const p = { pos: st.pos };
      const typeCnt = decodeUleb128(wasm, p);
      if (typeCnt === 0) throw new Error("Invalid Wasm Type section");
      for (let t = 0; t < typeCnt; t++) {
        if (p.pos >= secEnd || wasm[p.pos] !== 0x60) throw new Error("Invalid Wasm func type tag");
        p.pos += 1;
        const np = decodeUleb128(wasm, p);
        paramCounts.push(np);
        p.pos += np;
        const nres = decodeUleb128(wasm, p);
        p.pos += nres;
      }
    } else if (secId === 7) {
      const p = { pos: st.pos };
      const expCnt = decodeUleb128(wasm, p);
      for (let e = 0; e < expCnt; e++) {
        const nameLen = decodeUleb128(wasm, p);
        p.pos += nameLen;
        const kind = wasm[p.pos++];
        const idx = decodeUleb128(wasm, p);
        if (kind === 0x00) entryFnIdx = idx;
      }
    } else if (secId === 10) {
      const p = { pos: st.pos };
      const fnCnt = decodeUleb128(wasm, p);
      if (fnCnt === 0) throw new Error("Empty Wasm Code section");
      for (let f = 0; f < fnCnt; f++) {
        const bodyLen = decodeUleb128(wasm, p);
        if (p.pos + bodyLen > secEnd) throw new Error("Truncated Wasm function body");
        codeBodies.push(wasm.subarray(p.pos, p.pos + bodyLen));
        p.pos += bodyLen;
      }
    }
    st.pos = secEnd;
  }

  if (codeBodies.length === 0) throw new Error("Missing Wasm Code section");

  const budgetState = { stepsExecuted: 0 };

  const evalFn = (fnIdx: number, callArgs: bigint[], callDepth: number): bigint => {
    if (callDepth > 256) throw new Error("Wasm call stack overflow");
    const codeBody = codeBodies[fnIdx];
    if (!codeBody) throw new Error(`Invalid Wasm function index ${fnIdx}`);
    const numParams = paramCounts[fnIdx] ?? 0;

    const bp = { pos: 0 };
    const declCnt = decodeUleb128(codeBody, bp);
    let totalLocals = numParams;
    for (let i = 0; i < declCnt; i++) {
      const cnt = decodeUleb128(codeBody, bp);
      if (bp.pos >= codeBody.length) throw new Error("Truncated Wasm local decl");
      bp.pos += 1;
      totalLocals += cnt;
    }

    const frameBytes = wasm.length + Math.max(totalLocals, callArgs.length) * 8 * (callDepth + 1);
    if (frameBytes > memoryBudgetBytes) {
      throw new Error(
        `error[BUDGET_EXCEEDED]: Wasm AOT memory budget exceeded (${frameBytes} > ${memoryBudgetBytes} bytes)`,
      );
    }

    const locals: bigint[] = Array(Math.max(totalLocals, callArgs.length)).fill(0n);
    for (let i = 0; i < callArgs.length && i < locals.length; i++) {
      locals[i] = i64(callArgs[i]);
    }
    const stack: bigint[] = [];
    const ctrlStack: Array<[number, number]> = [];

    while (bp.pos < codeBody.length) {
      budgetState.stepsExecuted += 1;
      if (budgetState.stepsExecuted > stepLimit) {
        throw new Error(
          `error[BUDGET_EXCEEDED]: Wasm AOT step budget exceeded (${budgetState.stepsExecuted} > ${stepLimit})`,
        );
      }
      const runtimeBytes = frameBytes + (stack.length + ctrlStack.length) * 8;
      if (runtimeBytes > memoryBudgetBytes) {
        throw new Error(
          `error[BUDGET_EXCEEDED]: Wasm AOT memory budget exceeded (${runtimeBytes} > ${memoryBudgetBytes} bytes)`,
        );
      }

      const op = codeBody[bp.pos++];
      if (op === 0x02 || op === 0x03) {
        if (bp.pos >= codeBody.length) throw new Error("Truncated Wasm block/loop header");
        bp.pos += 1;
        ctrlStack.push([op, bp.pos]);
      } else if (op === 0x04) {
        if (bp.pos >= codeBody.length) throw new Error("Truncated Wasm if header");
        bp.pos += 1;
        const cond = stack.pop();
        if (cond === undefined) throw new Error("Wasm stack underflow on if");
        if (cond !== 0n) {
          ctrlStack.push([0x04, bp.pos]);
        } else {
          bp.pos = skipWasmToMatchingEnd(codeBody, bp.pos);
        }
      } else if (op === 0x0c) {
        const depth = decodeUleb128(codeBody, bp);
        if (depth >= ctrlStack.length) throw new Error("Wasm br depth out of bounds");
        for (let d = 0; d < depth; d++) ctrlStack.pop();
        const [targetOp, targetPos] = ctrlStack[ctrlStack.length - 1];
        if (targetOp === 0x03) {
          bp.pos = targetPos;
        } else {
          ctrlStack.pop();
          bp.pos = skipWasmToMatchingEnd(codeBody, targetPos);
        }
      } else if (op === 0x10) {
        const calleeIdx = decodeUleb128(codeBody, bp);
        const calleeArity = paramCounts[calleeIdx] ?? 0;
        const calleeArgs: bigint[] = Array(calleeArity).fill(0n);
        for (let a = calleeArity - 1; a >= 0; a--) {
          const v = stack.pop();
          if (v === undefined) throw new Error("Wasm stack underflow on call");
          calleeArgs[a] = v;
        }
        const retVal = evalFn(calleeIdx, calleeArgs, callDepth + 1);
        stack.push(retVal);
      } else if (op === 0x20) {
        const idx = decodeUleb128(codeBody, bp);
        if (idx >= locals.length) throw new Error("Wasm local.get out of bounds");
        stack.push(locals[idx]);
      } else if (op === 0x21) {
        const idx = decodeUleb128(codeBody, bp);
        const v = stack.pop();
        if (v === undefined) throw new Error("Wasm stack underflow on local.set");
        if (idx >= locals.length) throw new Error("Wasm local.set out of bounds");
        locals[idx] = v;
      } else if (op === 0x22) {
        const idx = decodeUleb128(codeBody, bp);
        const v = stack[stack.length - 1];
        if (v === undefined) throw new Error("Wasm stack underflow on local.tee");
        if (idx >= locals.length) throw new Error("Wasm local.tee out of bounds");
        locals[idx] = v;
      } else if (op === 0x42) {
        stack.push(decodeSleb128(codeBody, bp));
      } else if (op === 0x50) {
        const a = stack.pop();
        if (a === undefined) throw new Error("Wasm stack underflow on i64.eqz");
        stack.push(a === 0n ? 1n : 0n);
      } else if (op === 0x51 || op === 0x52 || op === 0x53 || op === 0x55) {
        const b = stack.pop();
        const a = stack.pop();
        if (a === undefined || b === undefined) throw new Error("Wasm stack underflow on cmp");
        const r = op === 0x51 ? a === b : op === 0x52 ? a !== b : op === 0x53 ? a < b : a > b;
        stack.push(r ? 1n : 0n);
      } else if (op === 0x71) {
        const b = stack.pop();
        const a = stack.pop();
        if (a === undefined || b === undefined) throw new Error("Wasm stack underflow on i32.and");
        stack.push((a & b) & 1n);
      } else if (op === 0x72) {
        const b = stack.pop();
        const a = stack.pop();
        if (a === undefined || b === undefined) throw new Error("Wasm stack underflow on i32.or");
        stack.push((a | b) & 1n);
      } else if (
        op === 0x7c ||
        op === 0x7d ||
        op === 0x7e ||
        op === 0x83 ||
        op === 0x84 ||
        op === 0x85 ||
        op === 0x86 ||
        op === 0x87
      ) {
        const b = stack.pop();
        const a = stack.pop();
        if (a === undefined || b === undefined) throw new Error("Wasm stack underflow on binop");
        let r = 0n;
        if (op === 0x7c) r = i64(a + b);
        else if (op === 0x7d) r = i64(a - b);
        else if (op === 0x7e) r = i64(a * b);
        else if (op === 0x83) r = i64(a & b);
        else if (op === 0x84) r = i64(a | b);
        else if (op === 0x85) r = i64(a ^ b);
        else if (op === 0x86) r = i64(a << (b & 63n));
        else r = i64(a >> (b & 63n));
        stack.push(r);
      } else if (op === 0x7f) {
        const b = stack.pop();
        const a = stack.pop();
        if (a === undefined || b === undefined) throw new Error("Wasm stack underflow on i64.div_s");
        if (b === 0n || (a === I64_MIN && b === -1n)) throw new Error("Wasm trap on unguarded i64.div_s");
        stack.push(i64(a / b));
      } else if (op === 0x81) {
        const b = stack.pop();
        const a = stack.pop();
        if (a === undefined || b === undefined) throw new Error("Wasm stack underflow on i64.rem_s");
        if (b === 0n) throw new Error("Wasm trap on unguarded i64.rem_s");
        stack.push(a === I64_MIN && b === -1n ? 0n : i64(a % b));
      } else if (op === 0xad) {
        const a = stack.pop();
        if (a === undefined) throw new Error("Wasm stack underflow on i64.extend_i32_u");
        stack.push(BigInt.asUintN(32, a));
      } else if (op === 0x1b) {
        const c = stack.pop();
        const v2 = stack.pop();
        const v1 = stack.pop();
        if (c === undefined || v2 === undefined || v1 === undefined) {
          throw new Error("Wasm stack underflow on select");
        }
        stack.push(c !== 0n ? v1 : v2);
      } else if (op === 0x0f) {
        const ret = stack.pop();
        if (ret === undefined) throw new Error("Wasm stack empty on return");
        return ret;
      } else if (op === 0x0b) {
        if (ctrlStack.length === 0) break;
        ctrlStack.pop();
      } else {
        throw new Error(`Unsupported Wasm opcode 0x${op.toString(16).toUpperCase().padStart(2, "0")}`);
      }
    }

    const finalVal = stack.pop();
    if (finalVal === undefined) throw new Error("Wasm execution finished with empty stack");
    return finalVal;
  };

  return evalFn(entryFnIdx, args, 0);
}
