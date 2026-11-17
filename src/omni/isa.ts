// Omni ISA — complete 16-bit token partition, 128-opcode specification table,
// balanced ternary algebra, and Pack5 radix-3 codec.

export const VOCAB_SIZE = 32_768;
export const ENGLISH_MAX_TOKEN = 0x7dff;

export const OMNI_HARNESS_CONTRACT_VERSION = "omni-harness-v1.0.0";
export const OMNI_REPAIR_STRATEGIES_VERSION = "omni-repair-strategies-v1.1.0";
export const OMNI_NARROW_HOST_VERSION = "omni-narrow-host-v1.0.0";
export const OMNI_ACCEPTANCE_SUITE_VERSION = "omni-acceptance-suite-v1.0.0";
export const OMNI_ERROR_CONTRACT_VERSION = "omni-error-contract-v1.0.0";
export const OMNI_PHASE7_PURE_KERNEL_VERSION = "omni-pure-kernel-v1.0.0";
export const OMNI_PURE_FRONTEND_VERSION = "omni-pure-frontend-v1.0.0";
export const OMNI_PURE_DRIVER_VERSION = "omni-pure-driver-v1.0.0";
export const OMNI_PURE_HOST_VERSION = "omni-pure-host-v1.0.0";
export const OMNI_PURE_SELFHOST_VERSION = "omni-pure-selfhost-v1.0.0";

export const REG_BASE = 0x7e00; // 0x7E00 .. 0x7EBF (192 SSA registers)
export const REG_MAX = 0x7ebf;
export const PAR_BASE = 0x7ec0; // 0x7EC0 .. 0x7EDF (32 function parameters)
export const PAR_MAX = 0x7edf;

export const TOK_MOD_BEGIN = 0x7ee0;
export const TOK_SCHEMA_DEF = 0x7ee1;
export const TOK_EXPERT_DEF = 0x7ee2;
export const TOK_FN_DEF = 0x7ee3;
export const TOK_INS_END = 0x7ee4;
export const TOK_IMM_I16 = 0x7ee5;
export const TOK_IMM_I32 = 0x7ee6;
export const TOK_TRIT_NEG = 0x7ee7;
export const TOK_TRIT_ZERO = 0x7ee8;
export const TOK_TRIT_POS = 0x7ee9;
export const TOK_ENG_SPAN = 0x7eea;
export const TOK_SYM_SCHEMA = 0x7eeb;
export const TOK_SYM_FN = 0x7eec;
export const TOK_SYM_SKILL = 0x7eed;
export const TOK_FIELD_KV = 0x7eee;
export const TOK_MOD_END = 0x7eef;

export const LBL_BASE = 0x7ef0;
export const LBL_MAX = 0x7eff;

export const OP_BASE = 0x7f00;
export const OP_MAX = 0x7f7f;

export const TY_VOID = 0x7f80;
export const TY_I64 = 0x7f81;
export const TY_F64 = 0x7f82;
export const TY_TRIT = 0x7f83;
export const TY_STR = 0x7f84;
export const TY_UUID = 0x7f85;
export const TY_BYTES = 0x7f86;
export const TY_TRIT_VEC = 0x7f87;
export const TY_FLT_VEC = 0x7f88;
export const TY_ARRAY = 0x7f89;
export const TY_ARENA = 0x7f8a;
export const TY_MODEL_T = 0x7f8b;
export const TY_DOM = 0x7f8c;
export const TY_STRUCT = 0x7f8d;
export const TY_ERR = 0x7f8e;

export const TC_BASE = 0x7fc0;
export const TC_UNTRUSTED_BIT = 0x0020;

export type CapMask = number;
export const CAP_ID_PURE = 0;
export const CAP_ID_DB_R = 1;
export const CAP_ID_DB_W = 2;
export const CAP_ID_NET = 3;
export const CAP_ID_DOM = 4;
export const CAP_ID_AI = 5;
export const CAP_ID_VISION = 6;
export const CAP_ID_AUDIO = 7;
export const CAP_ID_AUTH = 8;
export const CAP_ID_FS = 9;
export const CAP_ID_SYS = 10;

export function capBit(capId: number): CapMask {
  return capId === 0 ? 0 : 1 << (capId - 1);
}

export function packTc(untrusted: boolean, capId: number): number {
  return (TC_BASE | (untrusted ? TC_UNTRUSTED_BIT : 0) | (capId & 0x1f)) & 0xffff;
}

export function unpackTc(tcTok: number): { untrusted: boolean; capId: number; capMask: CapMask } {
  const raw = Math.max(0, tcTok - TC_BASE);
  const untrusted = (raw & TC_UNTRUSTED_BIT) !== 0;
  const capId = raw & 0x1f;
  return { untrusted, capId, capMask: capBit(capId) };
}

export interface OpcodeSpec {
  tokenId: number;
  reqCapMask: CapMask;
  sinkRequiresSafe: boolean;
  isSanitizer: boolean;
  hasDestReg: boolean;
  debugLabel: string;
}

export const OMNI_ISA_TABLE: OpcodeSpec[] = [
  { tokenId: 0x7f00, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F00_NOP" },
  { tokenId: 0x7f01, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F01_ARENA" },
  { tokenId: 0x7f02, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F02_BORROW_R" },
  { tokenId: 0x7f03, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F03_BORROW_W" },
  { tokenId: 0x7f04, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F04_MOVE" },
  { tokenId: 0x7f05, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F05_DROP" },
  { tokenId: 0x7f06, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F06_CINT" },
  { tokenId: 0x7f07, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F07_CFLT" },
  { tokenId: 0x7f08, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F08_CSTR" },
  { tokenId: 0x7f09, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F09_CTRIT" },
  { tokenId: 0x7f0a, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F0A_SNEW" },
  { tokenId: 0x7f0b, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F0B_FGET" },
  { tokenId: 0x7f0c, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F0C_FSET" },
  { tokenId: 0x7f0d, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F0D_ANEW" },
  { tokenId: 0x7f0e, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F0E_IGET" },
  { tokenId: 0x7f0f, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F0F_ISET" },
  { tokenId: 0x7f10, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F10_IADD" },
  { tokenId: 0x7f11, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F11_ISUB" },
  { tokenId: 0x7f12, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F12_IMUL" },
  { tokenId: 0x7f13, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F13_IDIV" },
  { tokenId: 0x7f14, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F14_IMOD" },
  { tokenId: 0x7f15, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F15_INEG" },
  { tokenId: 0x7f16, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F16_IABS" },
  { tokenId: 0x7f17, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F17_IMIN" },
  { tokenId: 0x7f18, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F18_IMAX" },
  { tokenId: 0x7f19, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F19_FADD" },
  { tokenId: 0x7f1a, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F1A_FSUB" },
  { tokenId: 0x7f1b, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F1B_FMUL" },
  { tokenId: 0x7f1c, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F1C_FDIV" },
  { tokenId: 0x7f1d, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F1D_FSQRT" },
  { tokenId: 0x7f1e, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F1E_BAND" },
  { tokenId: 0x7f1f, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F1F_BOR" },
  { tokenId: 0x7f20, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F20_BXOR" },
  { tokenId: 0x7f21, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F21_BNOT" },
  { tokenId: 0x7f22, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F22_SHL" },
  { tokenId: 0x7f23, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F23_SHR" },
  { tokenId: 0x7f24, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F24_CEQ" },
  { tokenId: 0x7f25, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F25_CNE" },
  { tokenId: 0x7f26, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F26_CLT" },
  { tokenId: 0x7f27, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F27_CGT" },
  { tokenId: 0x7f28, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F28_TNOT" },
  { tokenId: 0x7f29, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F29_TAND" },
  { tokenId: 0x7f2a, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F2A_TOR" },
  { tokenId: 0x7f2b, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F2B_TXOR" },
  { tokenId: 0x7f2c, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F2C_TMAJ" },
  { tokenId: 0x7f2d, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F2D_TCMP" },
  { tokenId: 0x7f2e, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F2E_TADD" },
  { tokenId: 0x7f2f, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F2F_TMUL" },
  { tokenId: 0x7f30, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F30_TPACK5" },
  { tokenId: 0x7f31, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F31_TUNPACK5" },
  { tokenId: 0x7f32, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F32_TQUANT" },
  { tokenId: 0x7f33, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F33_TMATMUL" },
  { tokenId: 0x7f34, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F34_TDOT" },
  { tokenId: 0x7f35, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F35_EGATE" },
  { tokenId: 0x7f36, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F36_LOOKAHEAD" },
  { tokenId: 0x7f37, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F37_PREFETCH" },
  { tokenId: 0x7f38, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F38_HMATMUL" },
  { tokenId: 0x7f39, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F39_HOLO_EXEC" },
  { tokenId: 0x7f3a, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F3A_TALL" },
  { tokenId: 0x7f3b, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F3B_TANY" },
  { tokenId: 0x7f3c, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F3C_TVNEW" },
  { tokenId: 0x7f3d, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F3D_TVGET" },
  { tokenId: 0x7f3e, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F3E_TVSET" },
  { tokenId: 0x7f3f, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F3F_TLOGIC" },
  { tokenId: 0x7f40, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F40_LABEL" },
  { tokenId: 0x7f41, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F41_JMP" },
  { tokenId: 0x7f42, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F42_BR_TRIT" },
  { tokenId: 0x7f43, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F43_BR_POS" },
  { tokenId: 0x7f44, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F44_BR_NEG" },
  { tokenId: 0x7f45, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F45_SEL_TRIT" },
  { tokenId: 0x7f46, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F46_LBOUND" },
  { tokenId: 0x7f47, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F47_LNEXT" },
  { tokenId: 0x7f48, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F48_CALL" },
  { tokenId: 0x7f49, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F49_TCALL" },
  { tokenId: 0x7f4a, reqCapMask: 0, sinkRequiresSafe: true, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F4A_RET" },
  { tokenId: 0x7f4b, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F4B_TRY" },
  { tokenId: 0x7f4c, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F4C_ERR" },
  { tokenId: 0x7f4d, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F4D_IS_OK" },
  { tokenId: 0x7f4e, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F4E_UNWRAP" },
  { tokenId: 0x7f4f, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: false, debugLabel: "OP_7F4F_ASSERT" },
  { tokenId: 0x7f50, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F50_SLEN" },
  { tokenId: 0x7f51, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F51_SCAT" },
  { tokenId: 0x7f52, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F52_SSLICE" },
  { tokenId: 0x7f53, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: true, hasDestReg: true, debugLabel: "OP_7F53_SFIND" },
  { tokenId: 0x7f54, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F54_SSPLIT" },
  { tokenId: 0x7f55, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F55_S2I" },
  { tokenId: 0x7f56, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F56_I2S" },
  { tokenId: 0x7f57, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: true, hasDestReg: true, debugLabel: "OP_7F57_SAN_SQL" },
  { tokenId: 0x7f58, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: true, hasDestReg: true, debugLabel: "OP_7F58_SAN_DOM" },
  { tokenId: 0x7f59, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: false, isSanitizer: true, hasDestReg: true, debugLabel: "OP_7F59_SAN_PRM" },
  { tokenId: 0x7f5a, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: true, hasDestReg: true, debugLabel: "OP_7F5A_SAN_PTH" },
  { tokenId: 0x7f5b, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: true, hasDestReg: true, debugLabel: "OP_7F5B_VAL_UUID" },
  { tokenId: 0x7f5c, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: true, hasDestReg: true, debugLabel: "OP_7F5C_VAL_JSON" },
  { tokenId: 0x7f5d, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F5D_FNV1A" },
  { tokenId: 0x7f5e, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F5E_HEX_ENC" },
  { tokenId: 0x7f5f, reqCapMask: 0, sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F5F_HEX_DEC" },
  { tokenId: 0x7f60, reqCapMask: capBit(CAP_ID_DB_W), sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F60_TX_BEG" },
  { tokenId: 0x7f61, reqCapMask: capBit(CAP_ID_DB_W), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F61_TX_COM" },
  { tokenId: 0x7f62, reqCapMask: capBit(CAP_ID_DB_W), sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F62_TX_RBK" },
  { tokenId: 0x7f63, reqCapMask: capBit(CAP_ID_DB_W), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F63_DB_INS" },
  { tokenId: 0x7f64, reqCapMask: capBit(CAP_ID_DB_R), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F64_DB_GET" },
  { tokenId: 0x7f65, reqCapMask: capBit(CAP_ID_DB_R), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F65_DB_QRY" },
  { tokenId: 0x7f66, reqCapMask: capBit(CAP_ID_DB_W), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F66_DB_UPD" },
  { tokenId: 0x7f67, reqCapMask: capBit(CAP_ID_DB_W), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F67_DB_DEL" },
  { tokenId: 0x7f68, reqCapMask: capBit(CAP_ID_DB_R), sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F68_DB_CNT" },
  { tokenId: 0x7f69, reqCapMask: capBit(CAP_ID_DB_W), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F69_TV_INS" },
  { tokenId: 0x7f6a, reqCapMask: capBit(CAP_ID_DB_R), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F6A_TV_SRCH" },
  { tokenId: 0x7f6b, reqCapMask: capBit(CAP_ID_DB_W), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F6B_KV_PUT" },
  { tokenId: 0x7f6c, reqCapMask: capBit(CAP_ID_NET), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F6C_ROUTE" },
  { tokenId: 0x7f6d, reqCapMask: capBit(CAP_ID_NET), sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F6D_REQ_PAR" },
  { tokenId: 0x7f6e, reqCapMask: capBit(CAP_ID_NET), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F6E_RSP_JSON" },
  { tokenId: 0x7f6f, reqCapMask: capBit(CAP_ID_NET), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F6F_RSP_HTML" },
  { tokenId: 0x7f70, reqCapMask: capBit(CAP_ID_AUTH), sinkRequiresSafe: false, isSanitizer: true, hasDestReg: true, debugLabel: "OP_7F70_AUTH_VER" },
  { tokenId: 0x7f71, reqCapMask: capBit(CAP_ID_NET), sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F71_RATELIM" },
  { tokenId: 0x7f72, reqCapMask: capBit(CAP_ID_SYS), sinkRequiresSafe: false, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F72_TIME_MS" },
  { tokenId: 0x7f73, reqCapMask: capBit(CAP_ID_SYS), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F73_LOG" },
  { tokenId: 0x7f74, reqCapMask: capBit(CAP_ID_SYS), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F74_ENV_GET" },
  { tokenId: 0x7f75, reqCapMask: capBit(CAP_ID_FS), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F75_FS_READ" },
  { tokenId: 0x7f76, reqCapMask: capBit(CAP_ID_DOM), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F76_DOM_NODE" },
  { tokenId: 0x7f77, reqCapMask: capBit(CAP_ID_DOM), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F77_DOM_ATTR" },
  { tokenId: 0x7f78, reqCapMask: capBit(CAP_ID_DOM), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F78_DOM_TEXT" },
  { tokenId: 0x7f79, reqCapMask: capBit(CAP_ID_DOM), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F79_DOM_EVT" },
  { tokenId: 0x7f7a, reqCapMask: capBit(CAP_ID_DOM), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F7A_DOM_MNT" },
  { tokenId: 0x7f7b, reqCapMask: capBit(CAP_ID_DOM), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F7B_WGPU" },
  { tokenId: 0x7f7c, reqCapMask: capBit(CAP_ID_AUDIO), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F7C_WAUDIO" },
  { tokenId: 0x7f7d, reqCapMask: capBit(CAP_ID_AI), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F7D_EMBED" },
  { tokenId: 0x7f7e, reqCapMask: capBit(CAP_ID_VISION), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F7E_SK_LOAD" },
  { tokenId: 0x7f7f, reqCapMask: capBit(CAP_ID_VISION), sinkRequiresSafe: true, isSanitizer: false, hasDestReg: true, debugLabel: "OP_7F7F_SK_EXEC" },
];

export function decodeOpcodeToken(tok: number): OpcodeSpec | null {
  if (tok >= OP_BASE && tok <= OP_MAX) {
    return OMNI_ISA_TABLE[tok - OP_BASE] ?? null;
  }
  return null;
}

export function isWasmAotSupportedOpcode(opTok: number): boolean {
  return (
    opTok === 0x7f00 ||
    (opTok >= 0x7f01 && opTok <= 0x7f06) ||
    opTok === 0x7f09 ||
    (opTok >= 0x7f0d && opTok <= 0x7f18) ||
    (opTok >= 0x7f1e && opTok <= 0x7f2d) ||
    (opTok >= 0x7f30 && opTok <= 0x7f31) ||
    (opTok >= 0x7f34 && opTok <= 0x7f35) ||
    (opTok >= 0x7f3f && opTok <= 0x7f45) ||
    opTok === 0x7f4a ||
    opTok === 0x7f50 ||
    (opTok >= 0x7f57 && opTok <= 0x7f59)
  );
}

export function isX86_64AotSupportedOpcode(opTok: number): boolean {
  return isWasmAotSupportedOpcode(opTok) || opTok === 0x7f51;
}

export function clampTrit(v: number): number {
  if (v < -1) return -1;
  if (v > 1) return 1;
  return v | 0;
}

export function tnot(a: number): number {
  const c = -clampTrit(a);
  return c === 0 ? 0 : c;
}

export function tand(a: number, b: number): number {
  return Math.min(clampTrit(a), clampTrit(b));
}

export function tor(a: number, b: number): number {
  return Math.max(clampTrit(a), clampTrit(b));
}

export function txor(a: number, b: number): number {
  const r = -(clampTrit(a) * clampTrit(b));
  return r === 0 ? 0 : r;
}

export function tmaj(a: number, b: number, c: number): number {
  const s = clampTrit(a) + clampTrit(b) + clampTrit(c);
  if (s > 0) return 1;
  if (s < 0) return -1;
  return 0;
}

export const PACK5_MAX_VALID = 242;
export const SENTINEL_SPECIAL_TOKEN = 243;
export const SENTINEL_SPARSE_RUN_ZERO = 244;
export const SENTINEL_PARITY_CHECK = 245;
export const SENTINEL_CRC_ANCHOR = 246;

export const LUT_UNPACK5: number[][] = (() => {
  const table: number[][] = [];
  for (let b = 0; b < 243; b++) {
    const row: number[] = [0, 0, 0, 0, 0];
    let rem = b;
    for (let k = 0; k < 5; k++) {
      const u = rem % 3;
      row[k] = u - 1;
      rem = Math.floor(rem / 3);
    }
    table.push(row);
  }
  return table;
})();

export function pack5ByteLen(nTrits: number): number {
  return Math.floor((nTrits + 4) / 5);
}

export function pack5Encode(trits: number[]): number[] {
  const out: number[] = [];
  for (let off = 0; off < trits.length; off += 5) {
    let acc = 0;
    let pow3 = 1;
    for (let i = 0; i < 5; i++) {
      const t = off + i < trits.length ? clampTrit(trits[off + i]) : 0;
      const u = t + 1;
      acc += u * pow3;
      if (i < 4) pow3 *= 3;
    }
    out.push(acc);
  }
  return out;
}

export function pack5Decode(bytes: number[], expectedTrits: number): number[] {
  const out: number[] = [];
  for (const b of bytes) {
    if (b > PACK5_MAX_VALID || b < 0) {
      throw new Error("error[CORRUPT_PACK5]: PACK5_SENTINEL_OR_CORRUPT_BYTE");
    }
    const unpacked = LUT_UNPACK5[b];
    for (const v of unpacked) {
      if (out.length < expectedTrits) {
        out.push(v);
      }
    }
  }
  return out;
}

export function pack5GetTrit(packed: number[], tritIdx: number): number {
  const byteIdx = Math.floor(tritIdx / 5);
  if (byteIdx >= packed.length || tritIdx < 0) {
    throw new Error("PACK5_INDEX_OUT_OF_BOUNDS");
  }
  const b = packed[byteIdx];
  if (b > PACK5_MAX_VALID || b < 0) {
    throw new Error("error[CORRUPT_PACK5]: PACK5_SENTINEL_OR_CORRUPT_BYTE");
  }
  return LUT_UNPACK5[b][tritIdx % 5];
}

export function pack5SetTrit(packed: number[], tritIdx: number, val: number): void {
  const byteIdx = Math.floor(tritIdx / 5);
  if (byteIdx >= packed.length || tritIdx < 0) {
    throw new Error("PACK5_INDEX_OUT_OF_BOUNDS");
  }
  const b = packed[byteIdx];
  if (b > PACK5_MAX_VALID || b < 0) {
    throw new Error("error[CORRUPT_PACK5]: PACK5_SENTINEL_OR_CORRUPT_BYTE");
  }
  const trits = [...LUT_UNPACK5[b]];
  trits[tritIdx % 5] = clampTrit(val);
  let acc = 0;
  let pow3 = 1;
  for (let i = 0; i < 5; i++) {
    acc += (trits[i] + 1) * pow3;
    if (i < 4) pow3 *= 3;
  }
  packed[byteIdx] = acc;
}

export function pack5TdotDirect(aPacked: number[], bPacked: number[], nTrits: number): number {
  let dot = 0;
  let rem = nTrits;
  const len = Math.min(aPacked.length, bPacked.length);
  for (let i = 0; i < len; i++) {
    if (rem === 0) break;
    const ba = aPacked[i];
    const bb = bPacked[i];
    if (ba > PACK5_MAX_VALID || bb > PACK5_MAX_VALID || ba < 0 || bb < 0) {
      throw new Error("error[CORRUPT_PACK5]: PACK5_SENTINEL_OR_CORRUPT_BYTE");
    }
    const ua = LUT_UNPACK5[ba];
    const ub = LUT_UNPACK5[bb];
    const take = Math.min(rem, 5);
    for (let k = 0; k < take; k++) {
      dot += ua[k] * ub[k];
    }
    rem -= take;
  }
  return dot;
}

export function pack5SumDirect(packed: number[], nTrits: number): number {
  let sum = 0;
  let rem = nTrits;
  for (const b of packed) {
    if (rem === 0) break;
    if (b > PACK5_MAX_VALID || b < 0) {
      throw new Error("error[CORRUPT_PACK5]: PACK5_SENTINEL_OR_CORRUPT_BYTE");
    }
    const u = LUT_UNPACK5[b];
    const take = Math.min(rem, 5);
    for (let k = 0; k < take; k++) {
      sum += u[k];
    }
    rem -= take;
  }
  return sum;
}

export function pack5GateDirect(packed: number[], nTrits: number): number {
  const s = pack5SumDirect(packed, nTrits);
  if (s > 0) return 1;
  if (s < 0) return -1;
  return 0;
}

export function fnv1a64Hash(bytes: Uint8Array | number[]): bigint {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < bytes.length; i++) {
    h ^= BigInt(bytes[i] & 0xff);
    h = (h * prime) & mask;
  }
  return h;
}

export function fnv1a32Positive(s: string): number {
  const bytes = new TextEncoder().encode(s);
  let h = 0x811c9dc5;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return Math.max(h & 0x3fffffff, 1);
}
