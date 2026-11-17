// Omni IR — binary AST structures, bounds-checked pure-u16 decoder, and module framing.

import {
  CAP_ID_PURE,
  CapMask,
  ENGLISH_MAX_TOKEN,
  LBL_BASE,
  LBL_MAX,
  PAR_BASE,
  PAR_MAX,
  REG_BASE,
  REG_MAX,
  TOK_ENG_SPAN,
  TOK_EXPERT_DEF,
  TOK_FIELD_KV,
  TOK_FN_DEF,
  TOK_IMM_I16,
  TOK_IMM_I32,
  TOK_INS_END,
  TOK_MOD_BEGIN,
  TOK_MOD_END,
  TOK_SCHEMA_DEF,
  TOK_SYM_FN,
  TOK_SYM_SCHEMA,
  TOK_SYM_SKILL,
  TOK_TRIT_NEG,
  TOK_TRIT_POS,
  TOK_TRIT_ZERO,
  TY_VOID,
  packTc,
} from "./isa.ts";

export type BinOperand =
  | { kind: "Reg"; value: number }
  | { kind: "Param"; value: number }
  | { kind: "Label"; value: number }
  | { kind: "ImmInt"; value: bigint }
  | { kind: "ImmTrit"; value: number }
  | { kind: "EnglishSpan"; value: number[] }
  | { kind: "SchemaRef"; value: number }
  | { kind: "FnRef"; value: number }
  | { kind: "SkillRef"; value: number }
  | { kind: "FieldKV"; fieldIdx: number; inner: BinOperand };

export function extractLabelId(op: BinOperand): number | null {
  if (op.kind === "Label") return op.value;
  if (op.kind === "ImmInt") return Number(op.value & 0xffffn);
  return null;
}

export interface BinInstruction {
  offset: number;
  destReg: number | null;
  opTok: number;
  tyTok: number;
  tcTok: number;
  operands: BinOperand[];
}

export interface BinFunction {
  fnIdx: number;
  params: Array<[number, number, number]>; // [parTok, tyTok, tcTok]
  retTyTok: number;
  capMask: CapMask;
  instructions: BinInstruction[];
}

export interface BinModule {
  schemas: Array<[number, Array<[number, number]>]>;
  experts: Array<[number, number, number, number]>;
  functions: BinFunction[];
}

function parseOneOperand(stream: number[], state: { pos: number }): BinOperand {
  if (state.pos >= stream.length) {
    throw new Error("Unexpected EOF while decoding binary operand");
  }
  const tok = stream[state.pos];
  state.pos += 1;

  if (tok >= REG_BASE && tok <= REG_MAX) return { kind: "Reg", value: tok };
  if (tok >= PAR_BASE && tok <= PAR_MAX) return { kind: "Param", value: tok };
  if (tok >= LBL_BASE && tok <= LBL_MAX) return { kind: "Label", value: tok };
  if (tok === TOK_TRIT_NEG) return { kind: "ImmTrit", value: -1 };
  if (tok === TOK_TRIT_ZERO) return { kind: "ImmTrit", value: 0 };
  if (tok === TOK_TRIT_POS) return { kind: "ImmTrit", value: 1 };

  if (tok === TOK_IMM_I16) {
    if (state.pos >= stream.length) throw new Error("EOF in TOK_IMM_I16");
    const raw = stream[state.pos];
    state.pos += 1;
    const signed = (raw << 16) >> 16;
    return { kind: "ImmInt", value: BigInt(signed) };
  }

  if (tok === TOK_IMM_I32) {
    if (state.pos + 1 >= stream.length) throw new Error("EOF in TOK_IMM_I32");
    const hi = stream[state.pos];
    const lo = stream[state.pos + 1];
    state.pos += 2;
    const combined = ((hi << 16) | lo) | 0;
    return { kind: "ImmInt", value: BigInt(combined) };
  }

  if (tok === TOK_ENG_SPAN) {
    if (state.pos >= stream.length) throw new Error("EOF in TOK_ENG_SPAN len");
    const len = stream[state.pos];
    state.pos += 1;
    if (state.pos + len > stream.length) {
      throw new Error("Truncated English token span");
    }
    const span: number[] = [];
    for (let i = 0; i < len; i++) {
      span.push(stream[state.pos + i] & ENGLISH_MAX_TOKEN);
    }
    state.pos += len;
    return { kind: "EnglishSpan", value: span };
  }

  if (tok === TOK_SYM_SCHEMA) {
    if (state.pos >= stream.length) throw new Error("EOF in TOK_SYM_SCHEMA");
    const idx = stream[state.pos];
    state.pos += 1;
    return { kind: "SchemaRef", value: idx };
  }

  if (tok === TOK_SYM_FN) {
    if (state.pos >= stream.length) throw new Error("EOF in TOK_SYM_FN");
    const idx = stream[state.pos];
    state.pos += 1;
    return { kind: "FnRef", value: idx };
  }

  if (tok === TOK_SYM_SKILL) {
    if (state.pos >= stream.length) throw new Error("EOF in TOK_SYM_SKILL");
    const idx = stream[state.pos];
    state.pos += 1;
    return { kind: "SkillRef", value: idx };
  }

  if (tok === TOK_FIELD_KV) {
    if (state.pos >= stream.length) throw new Error("EOF in TOK_FIELD_KV");
    const fIdx = stream[state.pos];
    state.pos += 1;
    const inner = parseOneOperand(stream, state);
    return { kind: "FieldKV", fieldIdx: fIdx, inner };
  }

  throw new Error(`Invalid operand token 0x${tok.toString(16).toUpperCase().padStart(4, "0")} at offset ${state.pos - 1}`);
}

export function decodeOmniTokenStream(stream: number[]): BinModule {
  if (stream.length === 0 || stream[0] !== TOK_MOD_BEGIN) {
    throw new Error("Missing 0x7EE0 (TOK_MOD_BEGIN) header token");
  }
  const mod: BinModule = { schemas: [], experts: [], functions: [] };
  const state = { pos: 1 };

  while (state.pos < stream.length) {
    const tag = stream[state.pos];
    state.pos += 1;
    if (tag === TOK_MOD_END) {
      break;
    } else if (tag === TOK_SCHEMA_DEF) {
      if (state.pos + 2 > stream.length) throw new Error("Truncated TOK_SCHEMA_DEF header");
      const sId = stream[state.pos];
      const fCnt = stream[state.pos + 1];
      state.pos += 2;
      if (state.pos + fCnt * 2 > stream.length) throw new Error("Truncated TOK_SCHEMA_DEF fields");
      const fields: Array<[number, number]> = [];
      for (let i = 0; i < fCnt; i++) {
        fields.push([stream[state.pos], stream[state.pos + 1]]);
        state.pos += 2;
      }
      mod.schemas.push([sId, fields]);
    } else if (tag === TOK_EXPERT_DEF) {
      if (state.pos + 4 > stream.length) throw new Error("Truncated TOK_EXPERT_DEF header");
      mod.experts.push([
        stream[state.pos],
        stream[state.pos + 1],
        stream[state.pos + 2],
        stream[state.pos + 3],
      ]);
      state.pos += 4;
    } else if (tag === TOK_FN_DEF) {
      if (state.pos + 2 > stream.length) throw new Error("Truncated TOK_FN_DEF header");
      const fnIdx = stream[state.pos];
      const pCnt = stream[state.pos + 1];
      state.pos += 2;
      if (state.pos + pCnt * 3 + 3 > stream.length) throw new Error("Truncated TOK_FN_DEF signature");
      const params: Array<[number, number, number]> = [];
      for (let i = 0; i < pCnt; i++) {
        params.push([stream[state.pos], stream[state.pos + 1], stream[state.pos + 2]]);
        state.pos += 3;
      }
      const retTyTok = stream[state.pos];
      const capMask = stream[state.pos + 1];
      const insCnt = stream[state.pos + 2];
      state.pos += 3;

      const instructions: BinInstruction[] = [];
      for (let i = 0; i < insCnt; i++) {
        if (state.pos >= stream.length) {
          throw new Error("Unexpected EOF inside function instruction body");
        }
        const insOffset = state.pos;
        const firstTok = stream[state.pos];
        let destReg: number | null = null;
        let opTok: number;
        let tyTok: number;
        let tcTok: number;

        if (firstTok >= REG_BASE && firstTok <= REG_MAX) {
          if (state.pos + 4 > stream.length) throw new Error("Truncated SSA instruction header");
          destReg = firstTok;
          opTok = stream[state.pos + 1];
          tyTok = stream[state.pos + 2];
          tcTok = stream[state.pos + 3];
          state.pos += 4;
        } else {
          opTok = firstTok;
          tyTok = TY_VOID;
          tcTok = packTc(false, CAP_ID_PURE);
          state.pos += 1;
        }

        const operands: BinOperand[] = [];
        while (state.pos < stream.length && stream[state.pos] !== TOK_INS_END) {
          operands.push(parseOneOperand(stream, state));
        }
        if (state.pos >= stream.length || stream[state.pos] !== TOK_INS_END) {
          throw new Error(`Missing 0x7EE4 (TOK_INS_END) for instruction at ${insOffset}`);
        }
        state.pos += 1;

        instructions.push({ offset: insOffset, destReg, opTok, tyTok, tcTok, operands });
      }

      mod.functions.push({ fnIdx, params, retTyTok, capMask, instructions });
    } else {
      throw new Error(`Unexpected structural token 0x${tag.toString(16).toUpperCase().padStart(4, "0")} at offset ${state.pos - 1}`);
    }
  }
  return mod;
}

export function encodeU16StreamToBytes(stream: number[]): Uint8Array {
  const bytes = new Uint8Array(stream.length * 2);
  for (let i = 0; i < stream.length; i++) {
    bytes[i * 2] = stream[i] & 0xff;
    bytes[i * 2 + 1] = (stream[i] >> 8) & 0xff;
  }
  return bytes;
}

export function decodeBytesToU16Stream(bytes: Uint8Array): number[] {
  const out: number[] = [];
  for (let i = 0; i + 1 < bytes.length; i += 2) {
    out.push(bytes[i] | (bytes[i + 1] << 8));
  }
  return out;
}

export function formatNonHumanTokenStream(stream: number[]): string {
  const parts: string[] = [];
  for (let i = 0; i < stream.length; i++) {
    const hex = stream[i].toString(16).toUpperCase().padStart(4, "0");
    if (i > 0) {
      parts.push(i % 12 === 0 ? "\n" : " ");
    }
    parts.push(hex);
  }
  return parts.join("");
}

export function packFnModule(
  schemas: Array<[number, Array<[number, number]>]>,
  fnId: number,
  params: Array<[number, number, number]>,
  retTy: number,
  capMask: CapMask,
  insPackets: number[][],
): number[] {
  const stream: number[] = [TOK_MOD_BEGIN];
  for (const [sId, fields] of schemas) {
    stream.push(TOK_SCHEMA_DEF, sId, fields.length);
    for (const [fTy, fTc] of fields) {
      stream.push(fTy, fTc);
    }
  }
  stream.push(TOK_FN_DEF, fnId, params.length);
  for (const [pTok, pTy, pTc] of params) {
    stream.push(pTok, pTy, pTc);
  }
  stream.push(retTy, capMask, insPackets.length);
  for (const pkt of insPackets) {
    for (const w of pkt) stream.push(w);
  }
  stream.push(TOK_MOD_END);
  return stream;
}
