// Omni Assembler — mnemonic text-to-pure-u16 bytecode assembler supporting
// multi-function FN_DEF, CAP_MASK, numeric, bitwise, balanced ternary, SEL_TRIT,
// PACK5/UNPACK5/TDOT/EGATE/TLOGIC, inter-function CALL_*, byte buffers,
// and control-flow (LABEL, JMP, BR_POS, BR_NEG, BR_TRIT).

import {
  CAP_ID_PURE,
  CapMask,
  ENGLISH_MAX_TOKEN,
  LBL_BASE,
  PAR_BASE,
  REG_BASE,
  TOK_ENG_SPAN,
  TOK_FN_DEF,
  TOK_IMM_I16,
  TOK_IMM_I32,
  TOK_INS_END,
  TOK_MOD_BEGIN,
  TOK_MOD_END,
  TOK_SYM_FN,
  TOK_TRIT_NEG,
  TOK_TRIT_POS,
  TOK_TRIT_ZERO,
  TY_ARENA,
  TY_ARRAY,
  TY_BYTES,
  TY_DOM,
  TY_I64,
  TY_STR,
  TY_STRUCT,
  TY_TRIT,
  TY_TRIT_VEC,
  TY_UUID,
  packTc,
} from "./isa.ts";
import { packFnModule } from "./ir.ts";

export function parseAsmTyTok(s: string): number {
  switch (s) {
    case "I64":
      return TY_I64;
    case "TRIT":
      return TY_TRIT;
    case "STR":
      return TY_STR;
    case "UUID":
      return TY_UUID;
    case "BYTES":
      return TY_BYTES;
    case "ARRAY":
      return TY_ARRAY;
    case "TRIT_VEC":
      return TY_TRIT_VEC;
    case "ARENA":
      return TY_ARENA;
    case "DOM":
      return TY_DOM;
    case "STRUCT":
      return TY_STRUCT;
    default:
      return TY_I64;
  }
}

function parseOpMnemonic(m: string, currentRetTy: number): [number, number] {
  switch (m) {
    case "CINT":
    case "MOV":
      return [0x7f06, TY_I64];
    case "CSTR":
      return [0x7f08, TY_STR];
    case "CTRIT":
      return [0x7f09, TY_TRIT];
    case "ARENA":
      return [0x7f01, TY_ARENA];
    case "SNEW":
      return [0x7f0a, TY_STRUCT];
    case "FGET":
      return [0x7f0b, TY_I64];
    case "FSET":
      return [0x7f0c, TY_STRUCT];
    case "ANEW":
      return [0x7f0d, TY_BYTES];
    case "IGET":
      return [0x7f0e, TY_I64];
    case "ISET":
      return [0x7f0f, TY_BYTES];
    case "SLEN":
      return [0x7f50, TY_I64];
    case "SCAT":
    case "SCONC":
      return [0x7f51, TY_STR];
    case "SSLICE":
      return [0x7f52, TY_STR];
    case "SFIND":
      return [0x7f53, TY_I64];
    case "SSPLIT":
      return [0x7f54, TY_BYTES];
    case "S2I":
      return [0x7f55, TY_I64];
    case "I2S":
      return [0x7f56, TY_STR];
    case "HEX_ENC":
      return [0x7f5e, TY_STR];
    case "HEX_DEC":
      return [0x7f5f, TY_BYTES];
    case "IADD":
      return [0x7f10, TY_I64];
    case "ISUB":
      return [0x7f11, TY_I64];
    case "IMUL":
      return [0x7f12, TY_I64];
    case "IDIV":
      return [0x7f13, TY_I64];
    case "IMOD":
      return [0x7f14, TY_I64];
    case "INEG":
      return [0x7f15, TY_I64];
    case "IABS":
      return [0x7f16, TY_I64];
    case "IMIN":
      return [0x7f17, TY_I64];
    case "IMAX":
      return [0x7f18, TY_I64];
    case "BAND":
      return [0x7f1e, TY_I64];
    case "BOR":
      return [0x7f1f, TY_I64];
    case "BXOR":
      return [0x7f20, TY_I64];
    case "BNOT":
      return [0x7f21, TY_I64];
    case "SHL":
      return [0x7f22, TY_I64];
    case "SHR":
      return [0x7f23, TY_I64];
    case "CEQ":
      return [0x7f24, TY_TRIT];
    case "CNE":
      return [0x7f25, TY_TRIT];
    case "CLT":
      return [0x7f26, TY_TRIT];
    case "CGT":
      return [0x7f27, TY_TRIT];
    case "TNOT":
      return [0x7f28, TY_TRIT];
    case "TAND":
      return [0x7f29, TY_TRIT];
    case "TOR":
      return [0x7f2a, TY_TRIT];
    case "TXOR":
      return [0x7f2b, TY_TRIT];
    case "TMAJ":
      return [0x7f2c, TY_TRIT];
    case "TCMP":
      return [0x7f2d, TY_TRIT];
    case "PACK5":
    case "TPACK5":
      return [0x7f30, TY_BYTES];
    case "UNPACK5":
    case "TUNPACK5":
      return [0x7f31, TY_TRIT_VEC];
    case "TDOT":
      return [0x7f34, TY_I64];
    case "EGATE":
      return [0x7f35, TY_TRIT];
    case "LOOKAHEAD":
      return [0x7f36, TY_TRIT];
    case "TLOGIC":
      return [0x7f3f, TY_TRIT];
    case "SEL_TRIT":
    case "CMOV":
      return [0x7f45, currentRetTy];
    case "CALL":
    case "CALL_I64":
      return [0x7f48, TY_I64];
    case "CALL_TRIT":
      return [0x7f48, TY_TRIT];
    case "CALL_BYTES":
      return [0x7f48, TY_BYTES];
    case "CALL_TRIT_VEC":
      return [0x7f48, TY_TRIT_VEC];
    case "CALL_STR":
      return [0x7f48, TY_STR];
    case "CALL_STRUCT":
      return [0x7f48, TY_STRUCT];
    case "BORROW_R":
      return [0x7f02, TY_I64];
    case "BORROW_W":
      return [0x7f03, TY_I64];
    case "MOVE":
      return [0x7f04, TY_I64];
    case "SAN_SQL":
      return [0x7f57, TY_STR];
    case "SAN_DOM":
      return [0x7f58, TY_STR];
    case "SAN_PRM":
      return [0x7f59, TY_STR];
    case "FNV1A":
      return [0x7f5d, TY_I64];
    case "DB_GET":
    case "KV_GET":
      return [0x7f64, TY_I64];
    case "DB_QRY":
      return [0x7f65, TY_I64];
    case "DB_CNT":
      return [0x7f68, TY_I64];
    case "KV_PUT":
    case "DB_PUT":
      return [0x7f6b, TY_I64];
    case "ROUTE":
    case "HTTP_SRV":
      return [0x7f6c, TY_I64];
    case "REQ_PAR":
      return [0x7f6d, TY_I64];
    case "RSP_JSON":
      return [0x7f6e, TY_I64];
    case "RSP_HTML":
      return [0x7f6f, TY_STR];
    case "AUTH_VER":
      return [0x7f70, TY_I64];
    default:
      throw new Error(`Unknown assembler mnemonic '${m}'`);
  }
}

function pushOperandTok(pkt: number[], tokStr: string): void {
  if (tokStr.startsWith("%")) {
    const n = Number.parseInt(tokStr.slice(1), 10);
    if (!Number.isFinite(n) || n < 0) throw new Error(`Invalid register '${tokStr}'`);
    pkt.push(REG_BASE + n);
  } else if (tokStr.startsWith("$")) {
    const n = Number.parseInt(tokStr.slice(1), 10);
    if (!Number.isFinite(n) || n < 0) throw new Error(`Invalid param '${tokStr}'`);
    pkt.push(PAR_BASE + n);
  } else if (tokStr.startsWith("@fn")) {
    const n = Number.parseInt(tokStr.slice(3), 10);
    if (!Number.isFinite(n) || n < 0) throw new Error(`Invalid fn ref '${tokStr}'`);
    pkt.push(TOK_SYM_FN, n);
  } else if (tokStr.startsWith("@str:")) {
    const hexS = tokStr.slice(5);
    if (hexS.length % 2 !== 0) throw new Error("Invalid @str hex length");
    const byteLen = hexS.length / 2;
    pkt.push(TOK_ENG_SPAN, byteLen);
    for (let i = 0; i < byteLen; i++) {
      const b = Number.parseInt(hexS.slice(i * 2, i * 2 + 2), 16);
      pkt.push(b & ENGLISH_MAX_TOKEN);
    }
  } else if (tokStr === "+1" || tokStr === "+1t" || tokStr === "TRIT_POS") {
    pkt.push(TOK_TRIT_POS);
  } else if (tokStr === "0t" || tokStr === "TRIT_ZERO") {
    pkt.push(TOK_TRIT_ZERO);
  } else if (tokStr === "-1t" || tokStr === "TRIT_NEG") {
    pkt.push(TOK_TRIT_NEG);
  } else {
    if (!/^-?\d+$/.test(tokStr)) {
      throw new Error(`Invalid integer operand '${tokStr}'`);
    }
    const big = BigInt(tokStr);
    if (big >= -32768n && big <= 32767n) {
      pkt.push(TOK_IMM_I16, Number(big) & 0xffff);
    } else {
      const u = Number(BigInt.asUintN(32, big));
      pkt.push(TOK_IMM_I32, (u >>> 16) & 0xffff, u & 0xffff);
    }
  }
}

export function encodeAsmInsPacketU16(line: string, retTy: number): number[] {
  const tcSafe = packTc(false, CAP_ID_PURE);
  const parts = line.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) throw new Error("Empty instruction packet");

  if (parts[0] === "NOP") {
    return [0x7f00, TOK_INS_END];
  }
  if (parts[0] === "DROP" && parts.length >= 2) {
    const pkt = [0x7f05];
    pushOperandTok(pkt, parts[1]);
    pkt.push(TOK_INS_END);
    return pkt;
  }
  if (parts[0] === "RET" && parts.length >= 2) {
    const pkt = [0x7f4a];
    pushOperandTok(pkt, parts[1]);
    pkt.push(TOK_INS_END);
    return pkt;
  }
  if (parts[0] === "LABEL" && parts.length >= 2) {
    const lid = Number.parseInt(parts[1], 10);
    if (!Number.isFinite(lid) || lid < 0 || lid > 15) throw new Error("Label id > 15");
    return [0x7f40, LBL_BASE + lid, TOK_INS_END];
  }
  if (parts[0] === "JMP" && parts.length >= 2) {
    const lid = Number.parseInt(parts[1], 10);
    if (!Number.isFinite(lid) || lid < 0 || lid > 15) throw new Error("Label id > 15");
    return [0x7f41, LBL_BASE + lid, TOK_INS_END];
  }
  if ((parts[0] === "BR_POS" || parts[0] === "BR_NEG") && parts.length >= 3) {
    const pkt = [parts[0] === "BR_POS" ? 0x7f43 : 0x7f44];
    pushOperandTok(pkt, parts[1]);
    const lid = Number.parseInt(parts[2], 10);
    if (!Number.isFinite(lid) || lid < 0 || lid > 15) throw new Error("Label id > 15");
    pkt.push(LBL_BASE + lid, TOK_INS_END);
    return pkt;
  }
  if (parts[0] === "BR_TRIT" && parts.length >= 5) {
    const pkt = [0x7f42];
    pushOperandTok(pkt, parts[1]);
    for (let j = 2; j <= 4; j++) {
      const lid = Number.parseInt(parts[j], 10);
      if (!Number.isFinite(lid) || lid < 0 || lid > 15) throw new Error("Label id > 15");
      pkt.push(LBL_BASE + lid);
    }
    pkt.push(TOK_INS_END);
    return pkt;
  }
  if (parts.length >= 4 && parts[1] === "=") {
    if (!parts[0].startsWith("%")) throw new Error("Expected %N destination");
    const dIdx = Number.parseInt(parts[0].slice(1), 10);
    if (!Number.isFinite(dIdx) || dIdx < 0) throw new Error(`Invalid dest register '${parts[0]}'`);

    let opMnem = parts[2];
    let untrSuffix = false;
    if (opMnem.endsWith("@UNTRUSTED")) {
      opMnem = opMnem.slice(0, -"@UNTRUSTED".length);
      untrSuffix = true;
    } else if (opMnem.endsWith("@SAFE")) {
      opMnem = opMnem.slice(0, -"@SAFE".length);
    }

    const [opTok, tyTok] = parseOpMnemonic(opMnem, retTy);
    const insTc = untrSuffix ? packTc(true, CAP_ID_PURE) : tcSafe;
    const pkt = [REG_BASE + dIdx, opTok, tyTok, insTc];
    for (const opS of parts.slice(3)) {
      pushOperandTok(pkt, opS);
    }
    pkt.push(TOK_INS_END);
    return pkt;
  }

  throw new Error(`Cannot parse assembler line: '${line}'`);
}

export function assembleOmniTextToU16(src: string): number[] {
  let fnId = 0;
  let capMask: CapMask = 0;
  let params: Array<[number, number, number]> = [];
  let retTy = TY_I64;
  let insPackets: number[][] = [];
  const completedFns: Array<[number, Array<[number, number, number]>, number, CapMask, number[][]]> = [];
  let startedFnDef = false;

  for (const rawLine of src.split(/\r?\n/)) {
    const line = (rawLine.split("#")[0] ?? "").trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === "FN_DEF" && parts.length >= 2) {
      if (startedFnDef || insPackets.length > 0 || params.length > 0) {
        completedFns.push([fnId, params, retTy, capMask, insPackets]);
        params = [];
        insPackets = [];
        retTy = TY_I64;
        capMask = 0;
      }
      fnId = Number.parseInt(parts[1], 10);
      startedFnDef = true;
    } else if (parts[0] === "FN_ID" && parts.length >= 2) {
      fnId = Number.parseInt(parts[1], 10);
    } else if (parts[0] === "CAP_MASK" && parts.length >= 2) {
      capMask = Number.parseInt(parts[1], 10);
    } else if (parts[0] === "PARAM" && parts.length >= 3) {
      if (!parts[1].startsWith("$")) throw new Error("Expected $N in PARAM");
      const pIdx = Number.parseInt(parts[1].slice(1), 10);
      const ty = parseAsmTyTok(parts[2]);
      const untr = parts[3] === "UNTRUSTED";
      params.push([PAR_BASE + pIdx, ty, packTc(untr, CAP_ID_PURE)]);
    } else if (parts[0] === "RET_TY" && parts.length >= 2) {
      retTy = parseAsmTyTok(parts[1]);
    } else {
      insPackets.push(encodeAsmInsPacketU16(line, retTy));
    }
  }

  if (completedFns.length === 0) {
    return packFnModule([], fnId, params, retTy, capMask, insPackets);
  }
  completedFns.push([fnId, params, retTy, capMask, insPackets]);
  const stream: number[] = [TOK_MOD_BEGIN];
  for (const [fId, fParams, fRet, fCap, fIns] of completedFns) {
    stream.push(TOK_FN_DEF, fId, fParams.length);
    for (const [pt, pty, ptc] of fParams) {
      stream.push(pt, pty, ptc);
    }
    stream.push(fRet, fCap, fIns.length);
    for (const pkt of fIns) {
      for (const w of pkt) stream.push(w);
    }
  }
  stream.push(TOK_MOD_END);
  return stream;
}
