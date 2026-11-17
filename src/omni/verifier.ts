// Omni Static Verifier — O(n) ownership, taint, capability, SSA ordering,
// linear arena leak, and branch-label target verifier.

import {
  PAR_BASE,
  PAR_MAX,
  REG_BASE,
  REG_MAX,
  TY_ARENA,
  TY_MODEL_T,
  decodeOpcodeToken,
  unpackTc,
} from "./isa.ts";
import { BinFunction, BinOperand, decodeOmniTokenStream, extractLabelId } from "./ir.ts";

export type RegState =
  | { kind: "Uninit" }
  | { kind: "Owned" }
  | { kind: "BorrowedR"; readers: number }
  | { kind: "BorrowedW" }
  | { kind: "Moved" }
  | { kind: "Dropped" };

export type Taint = "Safe" | "Untrusted";

export interface RegSlot {
  state: RegState;
  taint: Taint;
  isHeapArena: boolean;
  parentSlot: number | null;
}

export interface BinVerificationReport {
  passed: boolean;
  fnIdx: number;
  instructionCount: number;
  errors: string[];
  elapsedNs: number;
  rlReward: number;
}

function defaultRegSlot(): RegSlot {
  return { state: { kind: "Uninit" }, taint: "Safe", isHeapArena: false, parentSlot: null };
}

export function tokenToSlot(tok: number): number | null {
  if (tok >= REG_BASE && tok <= REG_MAX) return tok - REG_BASE;
  if (tok >= PAR_BASE && tok <= PAR_MAX) return 192 + (tok - PAR_BASE);
  return null;
}

export function collectOperandSlots(op: BinOperand, out: number[]): void {
  if (op.kind === "Reg" || op.kind === "Param") {
    out.push(op.value);
  } else if (op.kind === "FieldKV") {
    collectOperandSlots(op.inner, out);
  }
}

function hex4(n: number): string {
  return `0x${n.toString(16).toUpperCase().padStart(4, "0")}`;
}

export function verifyBinaryFunction(func: BinFunction): BinVerificationReport {
  const t0 = performance.now();
  const errors: string[] = [];
  const regs: RegSlot[] = Array.from({ length: 224 }, defaultRegSlot);
  let nextSsaTok = REG_BASE;
  const definedLabels: number[] = [];

  for (const ins of func.instructions) {
    if (ins.opTok === 0x7f40 && ins.operands.length > 0) {
      const lid = extractLabelId(ins.operands[0]);
      if (lid !== null) definedLabels.push(lid);
    }
  }

  for (const [parTok, tyTok, tcTok] of func.params) {
    const slot = tokenToSlot(parTok);
    if (slot !== null) {
      const { untrusted } = unpackTc(tcTok);
      regs[slot] = {
        state: { kind: "Owned" },
        taint: untrusted ? "Untrusted" : "Safe",
        isHeapArena: tyTok === TY_ARENA || tyTok === TY_MODEL_T,
        parentSlot: null,
      };
    }
  }

  for (const ins of func.instructions) {
    const spec = decodeOpcodeToken(ins.opTok);
    if (!spec) {
      errors.push(`Offset ${ins.offset}: Unknown opcode token ${hex4(ins.opTok)}`);
      continue;
    }

    if (ins.opTok === 0x7f05) {
      const op0 = ins.operands[0];
      if (op0 && op0.kind === "Reg") {
        const slot = tokenToSlot(op0.value);
        if (slot !== null) {
          const st = regs[slot].state;
          if (st.kind === "Moved" || st.kind === "Dropped" || st.kind === "Uninit") {
            errors.push(`Offset ${ins.offset}: error[BORROW_CONFLICT]: Invalid DROP on ${hex4(op0.value)}`);
          } else if (st.kind === "BorrowedR" && st.readers > 0) {
            errors.push(
              `Offset ${ins.offset}: error[BORROW_CONFLICT]: Cannot DROP ${hex4(op0.value)} with ${st.readers} active readers`,
            );
          } else {
            const p = regs[slot].parentSlot;
            if (p !== null) {
              const pst = regs[p].state;
              regs[p].state =
                pst.kind === "BorrowedR" && pst.readers > 1
                  ? { kind: "BorrowedR", readers: pst.readers - 1 }
                  : { kind: "Owned" };
            }
            regs[slot].state = { kind: "Dropped" };
          }
        }
      }
      continue;
    }

    if (ins.opTok === 0x7f4a) {
      const op0 = ins.operands[0];
      if (op0) {
        const refToks: number[] = [];
        collectOperandSlots(op0, refToks);
        for (const rTok of refToks) {
          const slot = tokenToSlot(rTok);
          if (slot !== null) {
            if (regs[slot].taint === "Untrusted") {
              errors.push(`Offset ${ins.offset}: error[TAINT_ON_RET]: Taint violation on RET ${hex4(rTok)}`);
            }
            regs[slot].state = { kind: "Moved" };
          }
        }
      }
      for (let idx = 0; idx < 192; idx++) {
        const st = regs[idx].state.kind;
        if (regs[idx].isHeapArena && st !== "Moved" && st !== "Dropped" && st !== "Uninit") {
          errors.push(`Offset ${ins.offset}: error[LINEAR_LEAK]: Linear memory leak on ${hex4(REG_BASE + idx)}`);
        }
      }
      continue;
    }

    if (ins.opTok === 0x7f41 || ins.opTok === 0x7f42 || ins.opTok === 0x7f43 || ins.opTok === 0x7f44) {
      const lblOps = ins.opTok === 0x7f41 ? ins.operands : ins.operands.slice(1);
      if (lblOps.length === 0) {
        errors.push(`Offset ${ins.offset}: error[UNRESOLVED_LABEL]: Missing branch target label on ${hex4(ins.opTok)}`);
      }
      for (const lop of lblOps) {
        const lid = extractLabelId(lop);
        if (lid === null || !definedLabels.includes(lid)) {
          errors.push(`Offset ${ins.offset}: error[UNRESOLVED_LABEL]: Unresolved branch target label ${lid ?? "null"}`);
        }
      }
    }

    if (!spec.hasDestReg) {
      continue;
    }

    if (ins.destReg !== null) {
      const dTok = ins.destReg;
      if (dTok > nextSsaTok) {
        errors.push(
          `Offset ${ins.offset}: Non-canonical SSA token ${hex4(dTok)}, expected <= ${hex4(nextSsaTok)}`,
        );
        nextSsaTok = dTok + 1;
      } else if (dTok === nextSsaTok) {
        nextSsaTok += 1;
      } else {
        const dSlot = tokenToSlot(dTok);
        if (dSlot !== null) {
          const dstKind = regs[dSlot].state.kind;
          if (regs[dSlot].isHeapArena && dstKind !== "Dropped" && dstKind !== "Moved") {
            errors.push(`Offset ${ins.offset}: Cannot overwrite active heap arena ${hex4(dTok)}`);
          } else if (dstKind !== "Owned") {
            errors.push(`Offset ${ins.offset}: Cannot reassign register ${hex4(dTok)} in state ${dstKind}`);
          }
        }
      }

      const { untrusted: insUntrusted, capMask: insCapMask } = unpackTc(ins.tcTok);
      if ((insCapMask & ~func.capMask) !== 0 || (spec.reqCapMask & ~func.capMask) !== 0) {
        errors.push(`Offset ${ins.offset}: error[MISSING_CAP]: Capability violation on opcode ${hex4(ins.opTok)}`);
      }

      let anyUntrusted = false;
      let parentSlot: number | null = null;
      const opToks: number[] = [];
      for (const op of ins.operands) {
        if (op.kind === "EnglishSpan") {
          anyUntrusted = true;
        }
        collectOperandSlots(op, opToks);
      }

      for (const rTok of opToks) {
        const slot = tokenToSlot(rTok);
        if (slot !== null) {
          const info = regs[slot];
          if (
            info.state.kind === "Moved" ||
            info.state.kind === "Dropped" ||
            info.state.kind === "Uninit" ||
            info.state.kind === "BorrowedW"
          ) {
            errors.push(
              `Offset ${ins.offset}: error[BORROW_CONFLICT]: Invalid state ${info.state.kind} on token ${hex4(rTok)}`,
            );
          }
          if (info.taint === "Untrusted") {
            anyUntrusted = true;
          }
          if (ins.opTok === 0x7f03 && info.state.kind === "Owned") {
            regs[slot].state = { kind: "BorrowedW" };
            parentSlot = slot;
          } else if (ins.opTok === 0x7f02 && (info.state.kind === "Owned" || info.state.kind === "BorrowedR")) {
            const k = info.state.kind === "BorrowedR" ? info.state.readers : 0;
            regs[slot].state = { kind: "BorrowedR", readers: k + 1 };
            parentSlot = slot;
          } else if (ins.opTok === 0x7f04 && info.state.kind === "Owned") {
            regs[slot].state = { kind: "Moved" };
          }
        }
      }

      if (spec.sinkRequiresSafe && anyUntrusted) {
        errors.push(`Offset ${ins.offset}: error[TAINT_SINK]: Sink opcode ${hex4(ins.opTok)} received @UNTRUSTED operand`);
      }
      const outTaint: Taint = insUntrusted ? "Untrusted" : "Safe";
      if (!spec.isSanitizer && anyUntrusted && outTaint === "Safe") {
        errors.push(`Offset ${ins.offset}: error[TAINT_SINK]: Illegal taint upgrade to @SAFE in opcode ${hex4(ins.opTok)}`);
      }

      const dSlot = tokenToSlot(dTok);
      if (dSlot !== null) {
        regs[dSlot] = {
          state: { kind: "Owned" },
          taint: outTaint,
          isHeapArena: ins.opTok === 0x7f01 || ins.opTok === 0x7f7e,
          parentSlot,
        };
      }
    }
  }

  const elapsedNs = Math.round((performance.now() - t0) * 1_000_000);
  const passed = errors.length === 0;
  const rlReward = passed ? 1.0 : Math.max(0.4 - 0.3 * errors.length, -1.0);

  return {
    passed,
    fnIdx: func.fnIdx,
    instructionCount: func.instructions.length,
    errors,
    elapsedNs,
    rlReward,
  };
}

export function verifyU16Stream(stream: number[]): BinVerificationReport[] {
  const mod = decodeOmniTokenStream(stream);
  return mod.functions.map(verifyBinaryFunction);
}

export function formatDebugLens(stream: number[]): string {
  const mod = decodeOmniTokenStream(stream);
  const lines: string[] = [
    `=== OMNI DEBUG LENS (${stream.length} u16 tokens, ${mod.functions.length} functions, ${mod.schemas.length} schemas) ===`,
  ];
  for (const func of mod.functions) {
    const rep = verifyBinaryFunction(func);
    lines.push(
      `FN #${hex4(func.fnIdx)} (params=${func.params.length}, ret_ty=${hex4(func.retTyTok)}, cap_mask=${hex4(func.capMask)}, ins=${func.instructions.length}, verified=${rep.passed}):`,
    );
    func.instructions.forEach((ins, idx) => {
      const opName = decodeOpcodeToken(ins.opTok)?.debugLabel ?? "OP_UNKNOWN";
      const dstStr = ins.destReg !== null ? `%${ins.destReg - REG_BASE} (${hex4(ins.destReg)}) = ` : "";
      lines.push(
        `  [${String(idx).padStart(3, "0")} @${hex4(ins.offset)}] ${dstStr}${opName} (${hex4(ins.opTok)}) ${JSON.stringify(ins.operands)}`,
      );
    });
  }
  return lines.join("\n") + "\n";
}
