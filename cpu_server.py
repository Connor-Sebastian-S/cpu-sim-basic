#!/usr/bin/env python3

import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("Missing deps.  Run:  pip install flask flask-cors")
    sys.exit(1)

try:
    from cpu import CPU, CPUConfig, process_input, InterruptType, INSTRUCTIONS
except ImportError as exc:
    print(f"Cannot import cpu.py — is it in the same folder?\n  {exc}")
    sys.exit(1)

app = Flask(__name__, static_folder=".")
CORS(app)

CONFIGS = {
    "microcontroller": CPUConfig.microcontroller,
    "embedded_16bit":  CPUConfig.embedded_16bit,
    "modern_desktop":  CPUConfig.modern_desktop,
}

# Reverse instruction lookup for the assembler: mnemonic -> (opcode, operand_count, operand_kind)
MNEMONIC_TO_OP = {}
for _op, (_mn, _desc, _operands) in INSTRUCTIONS.items():
    MNEMONIC_TO_OP[_mn] = (_op, len(_operands), _operands[0] if _operands else None)


# ── serialisation helpers ────────────────────────────────────────────────────
# Everything below turns Python objects from cpu.py into plain JSON-able
# dicts. None of this touches cpu.py's internals beyond reading attributes
# that are already public on CPU/CPUConfig/MemoryMap/Cache — no changes to
# the simulation engine itself were needed for any of this.

def step_to_dict(step):
    """Serialise one SimulationStep (cpu.py) into a JSON-able dict.
    `mem_changed` is a list of (address, new_value) tuples logged whenever
    an instruction writes to memory — the frontend uses this to drive both
    the animated 'data packet' on the die and the Memory Map's running
    log of written cells."""
    d = {
        "phase":          step.phase,
        "cycle":          step.cycle,
        "description":    step.description,
        "detail":         step.detail,
        "cache_event":    step.cache_event or "",
        "reg_snapshot":   step.reg_snapshot or {},
        "mem_changed":    [[a, v] for a, v in (step.mem_changed or [])],
        "flags_snapshot": step.flags_snapshot,
    }
    if step.bus:
        d["bus"] = {
            "state":   step.bus.state.value,
            "address": step.bus.address,
            "data":    step.bus.data,
            "control": step.bus.control,
            "cycle":   step.bus.cycle,
            "notes":   step.bus.notes or "",
        }
    else:
        d["bus"] = None
    return d


def config_to_dict(cfg):
    """Serialise a CPUConfig into a flat dict. The frontend uses this both
    to size/label the die (cache_levels, gp_register_names, ...) and to
    drive purely-informational badges (clock speed, pipeline stages)."""
    return {
        "name":                 cfg.name,
        "description":          cfg.description,
        # The executable datapath is intentionally fixed at eight bits in
        # every preset; ``word_bits`` is retained as comparative profile
        # context for older saved traces and the die labels.
        "word_bits":            cfg.word_bits,
        "simulated_data_bits":  8,
        "address_bits":         cfg.address_bits,
        "cache_levels":         cfg.cache_levels,
        "gp_register_names":    cfg.gp_register_names,
        "pipeline_stages":      cfg.pipeline_stages,
        "has_branch_predictor": cfg.has_branch_predictor,
        "branch_penalty_cycles":cfg.branch_penalty_cycles,
        "has_out_of_order":     cfg.has_out_of_order,
        "has_virtual_memory":   cfg.has_virtual_memory,
        "page_size_bytes":      cfg.page_size_bytes,
        "l1_sets":  cfg.l1_sets,  "l1_ways":  cfg.l1_ways,
        "l1_line_bytes": cfg.l1_line_bytes, "l1_latency": cfg.l1_latency,
        "l2_sets":  cfg.l2_sets,  "l2_ways":  cfg.l2_ways,
        "l2_line_bytes": cfg.l2_line_bytes, "l2_latency": cfg.l2_latency,
        "l3_sets":  cfg.l3_sets,  "l3_ways":  cfg.l3_ways,
        "l3_line_bytes": cfg.l3_line_bytes, "l3_latency": cfg.l3_latency,
        "ram_latency_cycles":   cfg.ram_latency_cycles,
        "clock_speed_mhz":      cfg.clock_speed_mhz,
        "max_sim_cycles":       cfg.max_sim_cycles,
        "memory_size":          cfg.memory_size(),
        "model_scope": {
            "execution_core": "8-bit accumulator ISA and ALU",
            "addressing": f"{cfg.address_bits}-bit address bus / {cfg.memory_size():,}-byte map",
            "cache_hierarchy": "simulated",
            "branch_predictor": "simulated" if cfg.has_branch_predictor else "not present",
            "pipeline": "sequential phase trace; stages do not overlap",
            "clock": "informational label only",
            "virtual_memory": ("assumed TLB hit; page-table walking not modeled"
                               if cfg.has_virtual_memory else "not present"),
            "out_of_order": "not modeled",
        },
    }


def regions_to_dict(cpu):
    """Serialise cpu.memory.regions — the named address-range map (IVT,
    BIOS, Stack, Heap, Program Code, Data Segment, I/O ports, ...) that
    cpu.py's MemoryMap already builds internally. Powers the Memory Map
    drawer tab.

    NOTE: `type` here is the underlying *memory technology*
    (ROM / Flash / DRAM) from MemoryType, not a semantic category like
    "code" or "stack" — that distinction lives in `name` instead, which
    is stable across all three architecture profiles (e.g. "Program Code"
    is always spelled exactly that way). The frontend matches on `name`
    for anything that needs to find a specific region."""
    return [
        {
            "name":        r.name,
            "start":       r.start,
            "size":        r.size,
            "type":        r.mem_type.value,
            "writable":    r.writable,
            "description": r.description,
        }
        for r in cpu.memory.regions
    ]


def memory_meta(cpu):
    """A handful of useful absolute addresses + the total memory size,
    so the frontend doesn't need to recompute region boundaries itself."""
    return {
        "size":           cpu.memory.size,
        "code_base":      cpu.memory.code_base,
        "data_base":      cpu.memory.data_base,
        "stack_top":      cpu.memory.stack_top,
        "io_input_base":  cpu.memory.io_input_base,
        "io_output_base": cpu.memory.io_output_base,
    }


def disassemble(cpu, max_bytes=256):
    """Walk Program Code from code_base, decoding bytes via INSTRUCTIONS.
    Stops at an unrecognised opcode (natural zero-padding terminator), HLT,
    or the safety cap — works for both preset and custom-assembled programs
    without needing the original source.

    Mirrors cpu.py's CPU._fetch_operand() encoding exactly: "addr"-kind
    operands (LOAD/STORE/JMP/JZ/JNZ/CALL) are 2 bytes little-endian
    whenever cpu.config.address_bits > 8, everything else ("operand"/"port"
    kind, or any addr-kind operand on an 8-bit profile) is 1 byte. Getting
    this wrong would silently misalign every instruction after the first
    wide operand, so this MUST stay in lockstep with the engine."""
    start = cpu.memory.code_base
    wide_addr = cpu.config.address_bits > 8
    pos = start
    out = []
    while pos - start < max_bytes and pos < cpu.memory.size:
        opcode = cpu.memory.cells[pos].value
        if opcode not in INSTRUCTIONS:
            break
        mnemonic, desc, operands = INSTRUCTIONS[opcode]
        operand_val = None
        length = 1
        if operands:
            operand_kind = operands[0]
            is_wide = (operand_kind == "addr" and wide_addr)
            n_operand_bytes = 2 if is_wide else 1
            if pos + n_operand_bytes >= cpu.memory.size:
                break
            if is_wide:
                lo = cpu.memory.cells[pos + 1].value
                hi = cpu.memory.cells[pos + 2].value
                operand_val = lo | (hi << 8)
            else:
                operand_val = cpu.memory.cells[pos + 1].value
            length = 1 + n_operand_bytes
        out.append({
            "addr": pos, "opcode": opcode, "mnemonic": mnemonic,
            "operand": operand_val, "operand_kind": operands[0] if operands else None,
            "length": length, "desc": desc,
        })
        pos += length
        if mnemonic == "HLT":
            break
    return out


def build_run_summary(cpu, steps):
    """Assemble the single JSON payload returned by both /run and
    /run_asm, once a CPU object has finished executing a program. Keeping
    this in one place guarantees both endpoints' responses have an
    identical shape, so the frontend's setSimData() doesn't need to know
    or care which path produced the data."""
    return {
        "config":      config_to_dict(cpu.config),
        "steps":       [step_to_dict(s) for s in steps],
        "regions":     regions_to_dict(cpu),
        "memory_meta": memory_meta(cpu),
        "disassembly": disassemble(cpu),
        "io_history":  extract_io_history(steps),
        "summary": {
            "total_cycles":  cpu.cycle,
            "total_steps":   len(steps),
            "halted":        cpu.halted,
            "stop_reason":   cpu.stop_reason,
            "stop_detail":   cpu.stop_detail,
            "instructions_executed": cpu.instructions_executed,
            "max_cycles":    cpu.max_cycles_used,
            "timer_irq_enabled": cpu.timer_irq_enabled,
            "l1_hits":       cpu.l1_cache.hits,
            "l1_misses":     cpu.l1_cache.misses,
            "l1_evictions":  cpu.l1_cache.evictions,
            "l1_hit_rate":   cpu.l1_cache.hit_rate,
            "l2_hits":       cpu.l2_cache.hits,
            "l2_misses":     cpu.l2_cache.misses,
            "l2_evictions":  cpu.l2_cache.evictions,
            "l3_hits":       cpu.l3_cache.hits       if cpu.l3_cache else 0,
            "l3_misses":     cpu.l3_cache.misses     if cpu.l3_cache else 0,
            "l3_evictions":  cpu.l3_cache.evictions  if cpu.l3_cache else 0,
        },
    }


# ── tiny two-pass assembler ─────────────────────────────────────────────────
# Syntax:
#   ; comment            (or # comment)
#   label:                defines a label at the current address
#   MNEMONIC [operand]    operand is a label/symbol name, 0xNN hex, or decimal
#
# Operand width matches cpu.py's CPU._fetch_operand() exactly:
#   - "addr"-kind operands (LOAD/STORE/JMP/JZ/JNZ/CALL) are 1 byte on
#     profiles with address_bits <= 8, or 2 bytes little-endian on wider
#     profiles (16-bit/64-bit) — wide enough to address anywhere in a
#     64 KB program rather than only the first 256 bytes.
#   - "operand"-kind (ADD/SUB/AND/OR/XOR/CMP/LDI) and "port"-kind (IN/OUT)
#     operands are always 1 byte, on every profile — they're immediate
#     values, not addresses, so there's no truncation risk to fix.
#
# Symbolic constants (DMA_SRC_LO, DMA_SRC_HI, DMA_DST_LO, DMA_DST_HI,
# DMA_LEN_LO, DMA_LEN_HI, DMA_SCRATCH_LO, DMA_SCRATCH_HI) are pre-resolved
# per-profile addresses for the DMA control block — see build_dma_symbols().

class AsmError(Exception):
    """Raised for any assembly problem (bad mnemonic, missing/invalid
    operand, unknown label, oversized program). Always carries a 1-based
    line number so the frontend can show a compiler-style inline error."""
    def __init__(self, line_no, msg):
        super().__init__(f"Line {line_no}: {msg}")
        self.line_no = line_no
        self.msg = msg


def _parse_operand_token(tok, line_no):
    """Try to read an operand token as a number (0x-prefixed hex or plain
    decimal). Returns None if it isn't numeric at all — the caller then
    treats it as a label reference instead."""
    tok = tok.strip()
    try:
        if tok.lower().startswith("0x"):
            return int(tok, 16)
        return int(tok, 10)
    except ValueError:
        return None  # not numeric — caller treats as a label reference


def assemble(asm_text, load_addr, address_bits=8, max_bytes=256, symbols=None):
    """Assemble source into executable bytes and retain a source map.

    The two passes are the same as a conventional tiny assembler: pass 1
    calculates labels and instruction addresses; pass 2 emits bytes.  In
    addition to the program and label table, this returns one source-map
    record per emitted instruction so the teaching UI can keep source,
    disassembly and execution state synchronised.

    Returns: (program_bytes, labels_dict, source_map)
      source_map entries contain line_no, source, addr, length and mnemonic.
    """
    wide_addr = address_bits > 8
    operand_ceiling = 65535 if wide_addr else 255

    lines = asm_text.split("\n")
    # (line_no, original_source, mnemonic, operand_token, address, length)
    parsed = []
    label_line_entries = []
    labels = dict(symbols) if symbols else {}
    addr = load_addr

    def operand_width(kind):
        if kind == "addr" and wide_addr:
            return 2
        return 1 if kind else 0

    label_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")

    # Pass 1: validate syntax, calculate labels and instruction addresses.
    for i, raw in enumerate(lines, start=1):
        source_text = raw.rstrip("\r")
        line = source_text.split(";", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        m = label_re.match(line)
        if m:
            label_name = m.group(1)
            labels[label_name] = addr
            line = m.group(2).strip()
            if not line:
                label_line_entries.append({
                    "line_no": i, "source": source_text, "addr": addr,
                    "length": 0, "mnemonic": None, "label": label_name,
                    "label_only": True,
                })
                continue
        parts = line.split(None, 1)
        mnemonic = parts[0].upper()
        operand_tok = parts[1].strip() if len(parts) > 1 else None
        if mnemonic not in MNEMONIC_TO_OP:
            raise AsmError(i, f"unknown instruction '{parts[0]}'")
        opcode, n_operands, kind = MNEMONIC_TO_OP[mnemonic]
        if n_operands == 1 and operand_tok is None:
            raise AsmError(i, f"{mnemonic} requires an operand")
        if n_operands == 0 and operand_tok is not None:
            raise AsmError(i, f"{mnemonic} takes no operand (got '{operand_tok}')")
        length = 1 + operand_width(kind if n_operands else None)
        parsed.append((i, source_text, mnemonic, operand_tok, addr, length))
        addr += length
        if addr - load_addr > max_bytes:
            raise AsmError(i, f"program exceeds {max_bytes}-byte limit for this profile")

    if not parsed:
        raise AsmError(1, "program is empty")

    # Pass 2: emit bytes and build the address-to-source mapping.
    out = bytearray()
    source_map = list(label_line_entries)
    for i, source_text, mnemonic, operand_tok, ins_addr, length in parsed:
        opcode, n_operands, kind = MNEMONIC_TO_OP[mnemonic]
        out.append(opcode)
        if n_operands == 1:
            num = _parse_operand_token(operand_tok, i)
            is_wide = operand_width(kind) == 2
            if num is None:
                if operand_tok not in labels:
                    raise AsmError(i, f"unknown label '{operand_tok}'")
                num = labels[operand_tok] if is_wide else (labels[operand_tok] & 0xFF)
            ceiling = operand_ceiling if kind == "addr" else 255
            if not (0 <= num <= ceiling):
                raise AsmError(i, f"operand {num} out of range (0-{ceiling})")
            if is_wide:
                out.append(num & 0xFF)
                out.append((num >> 8) & 0xFF)
            else:
                out.append(num & 0xFF)
        source_map.append({
            "line_no": i,
            "source": source_text,
            "addr": ins_addr,
            "length": length,
            "mnemonic": mnemonic,
        })

    if out[-1] != 0xFF:
        implicit_addr = load_addr + len(out)
        out.append(0xFF)
        source_map.append({
            "line_no": None,
            "source": "        HLT            ; inserted automatically",
            "addr": implicit_addr,
            "length": 1,
            "mnemonic": "HLT",
            "implicit": True,
        })
    source_map.sort(key=lambda entry: (entry.get("line_no") is None, entry.get("line_no") or 10**9))
    return bytes(out), labels, source_map

def build_dma_symbols(cpu):
    """Pre-resolves the DMA control block's addresses to symbolic names a
    custom-assembly program can reference portably across architecture
    profiles — the DMA Buffer region lives at 0xE0 on the 8-bit MCU but
    0xA200 on the 16-bit/64-bit profiles, so hardcoding either would only
    work on one of the three.

    Layout (first 6 bytes of the "DMA Buffer" region, written via ordinary
    STORE instructions before the DMA opcode fires):
        +0 SRC_LO  +1 SRC_HI  +2 DST_LO  +3 DST_HI  +4 LEN_LO  +5 LEN_HI
    DMA_SCRATCH_LO/HI point just past the control block — a safe writable
    destination for demo transfers that doesn't collide with the control
    registers themselves."""
    region = cpu.memory._region("DMA Buffer")
    if not region:
        return {}
    base = region.start
    scratch = base + 6
    return {
        "DMA_SRC_LO": base + 0, "DMA_SRC_HI": base + 1,
        "DMA_DST_LO": base + 2, "DMA_DST_HI": base + 3,
        "DMA_LEN_LO": base + 4, "DMA_LEN_HI": base + 5,
        "DMA_SCRATCH_LO": scratch & 0xFF, "DMA_SCRATCH_HI": (scratch >> 8) & 0xFF,
    }


EXAMPLE_PROGRAMS = {
    # ── Foundations-view default (mirrors the course's own 4-line worked
    # example: load two numbers, add them, done) — see cpu_visualiser_simple.html
    "add_two_numbers": (
        "; Add two numbers together.\n"
        "; load a first\n"
        "; value, add a second value, then save the result.\n"
        "\n"
        "        LDI   9        ; A <- 9   (first number)\n"
        "        ADD   10       ; A <- A + 10   (second number)\n"
        "        STORE 0x20     ; save the result to memory\n"
        "        HLT\n"
    ),
    "string_echo": (
        "; STRING ECHO — type up to 5 characters into Stdin below,\n"
        "; then run: each one is read from the keyboard (a different\n"
        "; input port per character) and sent straight to the terminal.\n"
        ";\n"
        "; This is the simplest possible input -> output pipeline: no\n"
        "; transformation, just IN then OUT, repeated for each character.\n"
        "\n"
        "        IN    0         ; read 1st character from keyboard\n"
        "        OUT   0         ; send it to the terminal\n"
        "        IN    1         ; read 2nd character\n"
        "        OUT   0\n"
        "        IN    2         ; read 3rd character\n"
        "        OUT   0\n"
        "        IN    3         ; read 4th character\n"
        "        OUT   0\n"
        "        IN    4         ; read 5th character\n"
        "        OUT   0\n"
        "        HLT\n"
    ),
    "multiply_by_addition": (
        "; Multiply 6 x 7 using repeated addition.\n"
        "; There's no MUL instruction in this ISA, so multiplication is\n"
        "; built the way it's actually defined: adding one number to\n"
        "; itself the other number's worth of times.\n"
        "        LDI   0\n"
        "        STORE 0xA0      ; total <- 0\n"
        "        LDI   7\n"
        "        STORE 0xA1      ; counter <- 7\n"
        "loop:   LOAD  0xA0\n"
        "        ADD   6\n"
        "        STORE 0xA0      ; total += 6\n"
        "        LOAD  0xA1\n"
        "        SUB   1\n"
        "        STORE 0xA1      ; counter -= 1\n"
        "        CMP   0\n"
        "        JNZ   loop\n"
        "        LOAD  0xA0      ; A <- final total (42)\n"
        "        OUT   0\n"
        "        HLT\n"
    ),
    "sum_1_to_n": (
        "; Sum the numbers 1 through 5 (1+2+3+4+5 = 15).\n"
        "; Uses ADDM to add the CONTENTS of a memory cell rather than a\n"
        "; fixed constant -- that's what lets 'sum += i' work when i is\n"
        "; a variable, not a literal number baked into the instruction.\n"
        "        LDI   0\n"
        "        STORE 0xA0      ; sum <- 0\n"
        "        LDI   1\n"
        "        STORE 0xA1      ; i <- 1\n"
        "loop:   LOAD  0xA0\n"
        "        ADDM  0xA1      ; sum += i\n"
        "        STORE 0xA0\n"
        "        LOAD  0xA1\n"
        "        ADD   1\n"
        "        STORE 0xA1      ; i += 1\n"
        "        CMP   6\n"
        "        JNZ   loop\n"
        "        LOAD  0xA0\n"
        "        OUT   0\n"
        "        HLT\n"
    ),
    "max_of_two": (
        "; Output whichever of two numbers is larger.\n"
        "; This ISA only has 'jump if zero' / 'jump if not zero' -- no\n"
        "; direct 'jump if greater than'. So a > b is rebuilt from a\n"
        "; subtraction: if a - b comes out negative, its top bit (0x80)\n"
        "; is set, and masking that bit out turns 'is it negative' into\n"
        "; something JZ/JNZ can actually test.\n"
        "        LDI   42\n"
        "        STORE 0xA0      ; first number\n"
        "        LDI   17\n"
        "        STORE 0xA1      ; second number\n"
        "\n"
        "        LOAD  0xA0\n"
        "        SUBM  0xA1      ; A = first - second\n"
        "        AND   0x80      ; isolate just the sign bit\n"
        "        JNZ   second_bigger\n"
        "        LOAD  0xA0\n"
        "        OUT   0\n"
        "        JMP   done\n"
        "second_bigger:\n"
        "        LOAD  0xA1\n"
        "        OUT   0\n"
        "done:   HLT\n"
    ),
    "fibonacci_sequence": (
        "; Print the first six Fibonacci numbers: 0, 1, 1, 2, 3, 5.\n"
        "; Three C++ variables (prev, curr, next) become three memory\n"
        "; cells here, since there's only one register -- the\n"
        "; accumulator -- to actually compute with.\n"
        "        LDI   0\n"
        "        STORE 0xA0      ; prev <- 0\n"
        "        LDI   1\n"
        "        STORE 0xA1      ; curr <- 1\n"
        "        LDI   6\n"
        "        STORE 0xA2      ; counter <- 6 terms to print\n"
        "\n"
        "loop:   LOAD  0xA0\n"
        "        OUT   0         ; print prev\n"
        "\n"
        "        LOAD  0xA0\n"
        "        ADDM  0xA1      ; A = prev + curr\n"
        "        STORE 0xA3      ; next <- prev + curr\n"
        "\n"
        "        LOAD  0xA1\n"
        "        STORE 0xA0      ; prev <- curr\n"
        "        LOAD  0xA3\n"
        "        STORE 0xA1      ; curr <- next\n"
        "\n"
        "        LOAD  0xA2\n"
        "        SUB   1\n"
        "        STORE 0xA2      ; counter -= 1\n"
        "        CMP   0\n"
        "        JNZ   loop\n"
        "        HLT\n"
    ),
    "caesar_cipher": (
        "; Caesar cipher demo: read one keyboard character, shift it\n"
        "; forward by 3 (encode) and print that, then shift the SAME\n"
        "; value back by 3 (decode) and print the original again.\n"
        "        IN    0         ; read a character from the keyboard\n"
        "        ADD   3         ; shift forward by 3 -- encode\n"
        "        OUT   0         ; print the encoded character\n"
        "        SUB   3         ; shift back by 3 -- decode\n"
        "        OUT   0         ; print the original character back\n"
        "        HLT\n"
    ),
    "rocket_countdown": (
        "; Just for fun -- a rocket launch countdown, printed one\n"
        "; character at a time (this ISA has no way to print a whole\n"
        "; string in one instruction; every character gets its own OUT).\n"
        "        LDI   51        ; '3'\n"
        "        OUT   0\n"
        "        LDI   10        ; newline\n"
        "        OUT   0\n"
        "        LDI   50        ; '2'\n"
        "        OUT   0\n"
        "        LDI   10\n"
        "        OUT   0\n"
        "        LDI   49        ; '1'\n"
        "        OUT   0\n"
        "        LDI   10\n"
        "        OUT   0\n"
        "        LDI   71        ; 'G'\n"
        "        OUT   0\n"
        "        LDI   79        ; 'O'\n"
        "        OUT   0\n"
        "        LDI   33        ; '!'\n"
        "        OUT   0\n"
        "        HLT\n"
    ),
    "power_of_two_doubling": (
        "; Prints 1, 2, 4, 8, 16, 32 -- the first six powers of two.\n"
        "; There's no left-shift instruction in this ISA, so 'shift left\n"
        "; by one bit' is done the way it's actually defined underneath:\n"
        "; adding a number to itself doubles it, same as shifting every\n"
        "; bit up one place.\n"
        "        LDI   1\n"
        "        STORE 0xA0\n"
        "        LDI   6\n"
        "        STORE 0xA1      ; counter <- 6 terms to print\n"
        "\n"
        "loop:   LOAD  0xA0\n"
        "        OUT   0\n"
        "        LOAD  0xA0\n"
        "        ADDM  0xA0      ; A = A + A  (== A * 2)\n"
        "        STORE 0xA0\n"
        "\n"
        "        LOAD  0xA1\n"
        "        SUB   1\n"
        "        STORE 0xA1\n"
        "        CMP   0\n"
        "        JNZ   loop\n"
        "        HLT\n"
    ),
    "countdown": (
        "; Count down from 5 to 1, OUT each value, then halt.\n"
        "; Same shape as the counting-up loop, but decrementing —\n"
        "; a good side-by-side comparison of loop direction and CMP.\n"
        "        LDI   5\n"
        "loop:   OUT   0\n"
        "        SUB   1\n"
        "        CMP   0\n"
        "        JNZ   loop\n"
        "        HLT\n"
    ),
    "counting_loop": (
        "; Count from 0 up to 5, OUT each value, then halt\n"
        "        LOAD  0xA0      ; A <- counter cell (starts at 0)\n"
        "loop:   OUT   0\n"
        "        ADD   1\n"
        "        STORE 0xA0\n"
        "        CMP   5\n"
        "        JNZ   loop\n"
        "        HLT\n"
    ),
    "stack_push_pop": (
        "; Push three values, then pop them back off\n"
        "        LOAD  0xA0\n"
        "        PUSH\n"
        "        ADD   1\n"
        "        PUSH\n"
        "        ADD   1\n"
        "        PUSH\n"
        "        POP\n"
        "        OUT   0\n"
        "        POP\n"
        "        OUT   0\n"
        "        POP\n"
        "        OUT   0\n"
        "        HLT\n"
    ),
    "branch_demo": (
        "; Demonstrate CMP + conditional jumps\n"
        "        LOAD  0xA0\n"
        "        CMP   10\n"
        "        JZ    equal\n"
        "        OUT   0\n"
        "        JMP   done\n"
        "equal:  XOR   0xFF\n"
        "        OUT   0\n"
        "done:   HLT\n"
    ),
    "dma_demo": (
        "; Configure the DMA controller's control block (SRC/DST/LEN),\n"
        "; then trigger a transfer that copies 4 sentinel bytes into the\n"
        "; DMA buffer's scratch area -- entirely via the DMA controller,\n"
        "; the CPU never touches the data itself.\n"
        "        LDI   0x99\n"
        "        STORE 0xA0\n"
        "        LDI   0x98\n"
        "        STORE 0xA1\n"
        "        LDI   0x97\n"
        "        STORE 0xA2\n"
        "        LDI   0x96\n"
        "        STORE 0xA3\n"
        "        LDI   0xA0\n"
        "        STORE DMA_SRC_LO\n"
        "        LDI   0x00\n"
        "        STORE DMA_SRC_HI\n"
        "        LDI   DMA_SCRATCH_LO\n"
        "        STORE DMA_DST_LO\n"
        "        LDI   DMA_SCRATCH_HI\n"
        "        STORE DMA_DST_HI\n"
        "        LDI   4\n"
        "        STORE DMA_LEN_LO\n"
        "        LDI   0\n"
        "        STORE DMA_LEN_HI\n"
        "        DMA\n"
        "        HLT\n"
    ),

    # ── Module-aligned examples (SQA H175 34, Outcome 2) ─────────────────────
    # These three examples directly mirror the practical applications named
    # in the module's support notes: case conversion with masks, subnet mask
    # calculation, and XOR for data integrity — each one makes a bitwise
    # operation concrete rather than abstract, and the comments show the
    # 8-bit working the evidence requirements explicitly ask for.

    "letter_case_conversion": (
        "; ASCII CASE CONVERSION using bitwise OR and AND masks\n"
        ";\n"
        "; In ASCII, uppercase and lowercase letters differ by exactly ONE bit:\n"
        ";   bit 5 (0x20 = 0010 0000) is SET in lowercase, CLEAR in uppercase.\n"
        ";\n"
        "; Uppercase → Lowercase: OR  with 0x20 (SETS   bit 5)\n"
        "; Lowercase → Uppercase: AND with 0xDF (CLEARS bit 5)\n"
        ";   (0xDF = 1101 1111 — all bits set except bit 5)\n"
        ";\n"
        "; Step through with 'BIN' display mode to see the bit change!\n"
        "\n"
        "        LDI   0x48      ; A = 0x48 = 0100 1000 = ASCII 'H'\n"
        "        OR    0x20      ; A = 0100 1000 OR 0010 0000 = 0110 1000 = 0x68 = 'h'\n"
        "        OUT   0         ; Output: 0x68 ('h')\n"
        "\n"
        "        LDI   0x69      ; A = 0x69 = 0110 1001 = ASCII 'i'\n"
        "        AND   0xDF      ; A = 0110 1001 AND 1101 1111 = 0100 1001 = 0x49 = 'I'\n"
        "        OUT   0         ; Output: 0x49 ('I')\n"
        "        HLT\n"
    ),

    "ip_subnet_mask": (
        "; SUBNET MASK APPLICATION\n"
        ";\n"
        "; Network: 192.168.1.0/28  (last octet only is demonstrated)\n"
        ";   Host IP last octet: 100  (0x64 = 0110 0100)\n"
        ";   Subnet mask  /28:   240  (0xF0 = 1111 0000)  — 28 ones, 4 zeros\n"
        ";   Inverted mask:       15  (0x0F = 0000 1111)\n"
        ";\n"
        "; Network address = IP AND mask        → keep network bits, zero host bits\n"
        "; Host part       = IP AND NOT(mask)   → keep host bits, zero network bits\n"
        ";\n"
        ";   0110 0100  AND  1111 0000  =  0110 0000  = 96  (network)\n"
        ";   0110 0100  AND  0000 1111  =  0000 0100  =  4  (host)\n"
        "\n"
        "        LDI   0x64      ; A = 100 (0x64 = 0110 0100) -- host IP last octet\n"
        "        STORE 0xA0      ; Save copy for the second calculation\n"
        "        AND   0xF0      ; AND mask: 0110 0100 AND 1111 0000 = 0110 0000 = 96\n"
        "        OUT   0         ; Output network address last octet: 96 (0x60)\n"
        "        LOAD  0xA0      ; Restore original IP octet\n"
        "        AND   0x0F      ; Inv mask: 0110 0100 AND 0000 1111 = 0000 0100 = 4\n"
        "        OUT   0         ; Output host number: 4 (0x04)\n"
        "        HLT\n"
    ),

    "xor_checksum": (
        "; XOR CHECKSUM -- data integrity using XOR properties\n"
        ";\n"
        "; Key XOR properties:\n"
        ";   A XOR A = 0   (any value XORed with itself = 0)\n"
        ";   A XOR 0 = A   (XOR with 0 leaves value unchanged)\n"
        ";   XOR is its own inverse\n"
        ";\n"
        "; To send data + checksum:\n"
        ";   1. XOR all data bytes together  -> checksum\n"
        ";   2. Transmit data + checksum\n"
        ";   3. Receiver XORs all bytes incl. checksum -> 0 means no error\n"
        ";\n"
        "; Data: 'Math' = 0x4D 0x61 0x74 0x68\n"
        ";   0000 0000 XOR 0100 1101 = 0100 1101  (after M)\n"
        ";   0100 1101 XOR 0110 0001 = 0010 1100  (after a)\n"
        ";   0010 1100 XOR 0111 0100 = 0101 1000  (after t)\n"
        ";   0101 1000 XOR 0110 1000 = 0011 0000  (after h) <- checksum = 0x30\n"
        "\n"
        "        LDI   0x00      ; A = 0  (running XOR accumulator)\n"
        "        XOR   0x4D      ; XOR with 'M' = 0x4D\n"
        "        XOR   0x61      ; XOR with 'a' = 0x61\n"
        "        XOR   0x74      ; XOR with 't' = 0x74\n"
        "        XOR   0x68      ; XOR with 'h' = 0x68  -- A = 0x30 (checksum)\n"
        "        STORE 0xA0      ; Save checksum\n"
        "        OUT   0         ; Transmit checksum: 0x30\n"
        ";\n"
        "; Receiver re-XORs everything including the checksum:\n"
        "        LOAD  0xA0      ; Reload checksum\n"
        "        XOR   0x4D\n"
        "        XOR   0x61\n"
        "        XOR   0x74\n"
        "        XOR   0x68      ; A = 0x00 if no error (XOR cancels itself out)\n"
        "        OUT   0         ; Output 0x00 = no error\n"
        "        HLT\n"
    ),

    "odd_even_subroutine": (
        "; ODD/EVEN TEST using a reusable subroutine and bit masking\n"
        ";\n"
        "; The LSB (bit 0) tells us odd or even:\n"
        ";   even number: ...xxxx0  (bit 0 = 0)\n"
        ";   odd  number: ...xxxx1  (bit 0 = 1)\n"
        ";\n"
        "; AND with 0x01 = 0000 0001 isolates bit 0.\n"
        "; CALL pushes the return address onto the stack;\n"
        "; RET pops it back, making the subroutine reusable.\n"
        "\n"
        "        LDI   66        ; Test 66  (0100 0010 -> bit 0 = 0 -> even)\n"
        "        CALL  odd_test\n"
        "        OUT   0         ; Output: 0 (even)\n"
        "\n"
        "        LDI   67        ; Test 67  (0100 0011 -> bit 0 = 1 -> odd)\n"
        "        CALL  odd_test\n"
        "        OUT   0         ; Output: 1 (odd)\n"
        "\n"
        "        LDI   200       ; Test 200 (1100 1000 -> bit 0 = 0 -> even)\n"
        "        CALL  odd_test\n"
        "        OUT   0         ; Output: 0 (even)\n"
        "        HLT\n"
        "\n"
        "odd_test:               ; --- Subroutine: returns 0 (even) or 1 (odd) ---\n"
        "        AND   1         ; A AND 0000 0001 = isolate bit 0\n"
        "        RET             ; Return to caller with result in A\n"
    ),

    # ── Programs that show ADDM/SUBM and the IO terminal ─────────────────────

    "factorial_5": (
        "; FACTORIAL OF 5  (5! = 120 = 0x78)\n"
        ";\n"
        "; Algorithm: result = 1 x 2 x 3 x 4 x 5 via repeated addition.\n"
        "; Since the ISA has no multiply opcode, n x result is computed by\n"
        "; adding result to an accumulator n times — this is where ADDM shines:\n"
        ";   ADDM addr  ->  A = A + mem[addr]  (load from memory AND add, in one instruction)\n"
        ";\n"
        "; Scratch locations (writable Heap/Data region):\n"
        ";   0x40 = n (outer counter, counts 5 -> 0)\n"
        ";   0x41 = add_counter (inner counter, equals n each outer iteration)\n"
        ";   0x42 = result (the running product)\n"
        ";   0x43 = factor  (copy of result BEFORE zeroing, added add_counter times)\n"
        ";\n"
        "; NOTE: this program is 70 bytes on 16/64-bit profiles (2-byte addr operands)\n"
        ";       — use embedded_16bit or modern_desktop, not microcontroller.\n"
        ";       Needs ~3000 cycles — raise the Sim Length slider above default.\n"
        "\n"
        "        LDI   5\n"
        "        STORE 0x40      ; n = 5\n"
        "        LDI   1\n"
        "        STORE 0x42      ; result = 1\n"
        "\n"
        "start:  LOAD  0x40\n"
        "        JZ    done      ; if n == 0, finished\n"
        "\n"
        "        LOAD  0x42\n"
        "        STORE 0x43      ; factor = current result\n"
        "        LOAD  0x40\n"
        "        STORE 0x41      ; add_counter = n\n"
        "        LDI   0\n"
        "        STORE 0x42      ; result = 0 (rebuild via addition)\n"
        "\n"
        "mul:    LOAD  0x42\n"
        "        ADDM  0x43      ; result += factor  (ADDM: A = A + mem[0x43])\n"
        "        STORE 0x42\n"
        "        LOAD  0x41\n"
        "        SUB   1\n"
        "        STORE 0x41\n"
        "        JNZ   mul\n"
        "\n"
        "        LOAD  0x40\n"
        "        SUB   1\n"
        "        STORE 0x40\n"
        "        JMP   start\n"
        "\n"
        "done:   LOAD  0x42\n"
        "        OUT   0         ; Output result: 120 = 0x78\n"
        "        HLT\n"
    ),

    "cpu_benchmark": (
        "; ════════════════════════════════════════════════════════════════\n"
        "; MULTI-PHASE CPU BENCHMARK — 64-bit Desktop (modern_desktop)\n"
        "; Measured: 99,350 cycles — verified on modern_desktop profile\n"
        "; ════════════════════════════════════════════════════════════════\n"
        ";\n"
        "; Three computational phases that push the 64-bit pipeline hard.\n"
        "; Watch the Bottleneck Heatmap, Cache Inspector, and Branch\n"
        "; Predictor tabs fill up as phases progress.\n"
        ";\n"
        "; HOW TO RUN:\n"
        ";   1. Architecture: modern_desktop\n"
        ";   2. Drag the Sim Length slider to maximum (100,000 cycles)\n"
        ";   3. Click Assemble & Run\n"
        ";   4. Open IO Terminal drawer tab — three output values appear\n"
        ";\n"
        "; ── PHASE A: Factorial benchmark (12 repetitions) ───────────────\n"
        "; Computes 7! (mod 256 = 0xB0 = 176) using ADDM-based multiply.\n"
        "; Inner loop: result += factor, repeated n times per outer step.\n"
        "; Inner iterations per run: 7+6+5+4+3+2+1 = 28 ADDM calls.\n"
        "; Cost: ~7,864 cycles/run x 12 = ~94,368 cycles.\n"
        ";\n"
        "; ── PHASE B: Fibonacci generator (37 iterations) ────────────────\n"
        "; F(n) = F(n-1) + F(n-2) via ADDM on two memory cells.\n"
        "; Cost: ~345 cycles/iteration x 37 = ~12,753 cycles.\n"
        ";\n"
        "; ── PHASE C: Countdown accumulation (47 iterations) ─────────────\n"
        "; Counts 47→1, adding each value into a running sum via ADDM.\n"
        "; Cost: ~181 cycles/iteration x 47 = ~8,516 cycles.\n"
        ";\n"
        "; ── OUTPUTS (visible in IO Terminal tab) ────────────────────────\n"
        ";   0xB0 (176) — 7! mod 256:           proves Phase A ran fully\n"
        ";   0x01 (  1) — F(39) mod 256:        proves Phase B ran fully\n"
        ";   0x68 (104) — countdown sum mod 256: proves Phase C ran fully\n"
        ";\n"
        "; ── SCRATCH MEMORY ───────────────────────────────────────────────\n"
        ";   0x40 n      0x41 add_ctr  0x42 result  0x43 factor\n"
        ";   0x44 reps   0x45 fact_out\n"
        ";   0x50 fib_a  0x51 fib_b   0x52 fib_tmp 0x53 fib_count\n"
        ";   0x60 cd_ctr 0x61 cd_sum\n"
        "; ════════════════════════════════════════════════════════════════\n"
        "\n"
        "        LDI   0\n"
        "        STORE 0x45      ; fact_out = 0\n"
        "        STORE 0x61      ; cd_sum   = 0\n"
        "\n"
        "        ; ── PHASE A: 12 reps of factorial(7) via ADDM multiply ──\n"
        "        LDI   12\n"
        "        STORE 0x44      ; reps = 12\n"
        "\n"
        "phaseA: LOAD  0x44\n"
        "        JZ    phaseB\n"
        "\n"
        "        LDI   7\n"
        "        STORE 0x40      ; n = 7\n"
        "        LDI   1\n"
        "        STORE 0x42      ; result = 1\n"
        "\n"
        "outerA: LOAD  0x40\n"
        "        JZ    doneA     ; n==0: this rep is done\n"
        "\n"
        "        LOAD  0x42\n"
        "        STORE 0x43      ; factor = current result\n"
        "        LOAD  0x40\n"
        "        STORE 0x41      ; add_counter = n\n"
        "        LDI   0\n"
        "        STORE 0x42      ; result = 0 (rebuild as n copies of factor)\n"
        "\n"
        "innerA: LOAD  0x42\n"
        "        ADDM  0x43      ; result += factor\n"
        "        STORE 0x42\n"
        "        LOAD  0x41\n"
        "        SUB   1\n"
        "        STORE 0x41\n"
        "        JNZ   innerA    ; 27x taken per n, 1x not-taken (predictor trains here)\n"
        "\n"
        "        LOAD  0x40\n"
        "        SUB   1\n"
        "        STORE 0x40\n"
        "        JMP   outerA\n"
        "\n"
        "doneA:  LOAD  0x42\n"
        "        STORE 0x45      ; fact_out = 7! mod 256 = 0xB0 (176)\n"
        "        LOAD  0x44\n"
        "        SUB   1\n"
        "        STORE 0x44\n"
        "        JMP   phaseA\n"
        "\n"
        "        ; ── PHASE B: 37 Fibonacci iterations ────────────────────\n"
        "phaseB: LDI   1\n"
        "        STORE 0x50      ; fib_a = F(1) = 1\n"
        "        STORE 0x51      ; fib_b = F(2) = 1\n"
        "        LDI   37\n"
        "        STORE 0x53      ; fib_count = 37\n"
        "\n"
        "fibL:   LOAD  0x53\n"
        "        JZ    phaseC\n"
        "        LOAD  0x50\n"
        "        ADDM  0x51      ; A = fib_a + fib_b\n"
        "        STORE 0x52      ; fib_tmp = next Fibonacci number\n"
        "        LOAD  0x51\n"
        "        STORE 0x50      ; fib_a = old fib_b\n"
        "        LOAD  0x52\n"
        "        STORE 0x51      ; fib_b = next\n"
        "        LOAD  0x53\n"
        "        SUB   1\n"
        "        STORE 0x53\n"
        "        JMP   fibL\n"
        "\n"
        "        ; ── PHASE C: countdown 47→1, accumulate sum ─────────────\n"
        "        ; sum = 47+46+...+1 = 1128 = 0x468, mod 256 = 0x68 (104)\n"
        "phaseC: LDI   0\n"
        "        STORE 0x61\n"
        "        LDI   47\n"
        "        STORE 0x60\n"
        "\n"
        "cdL:    LOAD  0x61\n"
        "        ADDM  0x60      ; cd_sum += cd_counter\n"
        "        STORE 0x61\n"
        "        LOAD  0x60\n"
        "        SUB   1\n"
        "        STORE 0x60\n"
        "        JNZ   cdL\n"
        "\n"
        "        ; ── Output three proof-of-work values ────────────────────\n"
        "        LOAD  0x45\n"
        "        OUT   0         ; 0xB0 (176) = 7! mod 256\n"
        "        LOAD  0x51\n"
        "        OUT   0         ; 0x01 (  1) = F(39) mod 256\n"
        "        LOAD  0x61\n"
        "        OUT   0         ; 0x68 (104) = countdown sum mod 256\n"
        "        HLT\n"
    ),

    "keyboard_echo": (
        "; KEYBOARD ECHO with case conversion\n"
        ";\n"
        "; Reads one character from the keyboard (IN port 0) and echoes it\n"
        "; to the terminal three ways: original, lowercase, uppercase.\n"
        ";\n"
        "; Pre-fill the Stdin field below before running,\n"
        "; e.g. type  H  to see: 0x48 'H'  0x68 'h'  0x48 'H'\n"
        "\n"
        "        IN    0         ; A = character from keyboard (port 0)\n"
        "        STORE 0xA0      ; save original\n"
        "        OUT   0         ; echo original to terminal\n"
        "\n"
        "        LOAD  0xA0\n"
        "        OR    0x20      ; set bit 5  -> lowercase (0x41 'A' -> 0x61 'a')\n"
        "        OUT   0         ; echo lowercase\n"
        "\n"
        "        LOAD  0xA0\n"
        "        AND   0xDF      ; clear bit 5 -> uppercase (0x61 'a' -> 0x41 'A')\n"
        "        OUT   0         ; echo uppercase\n"
        "        HLT\n"
    ),

    "case_toggle_terminal": (
        "; CASE CONVERSION TERMINAL DEMO\n"
        ";\n"
        "; Reads 4 characters from ports 0-3 (fill Stdin below),\n"
        "; converts each to lowercase, and streams them to the IO Terminal.\n"
        "\n"
        "        IN    0         ; Read char 1\n"
        "        OR    0x20      ; Force lowercase\n"
        "        OUT   0         ; -> terminal\n"
        "\n"
        "        IN    1         ; Read char 2\n"
        "        OR    0x20\n"
        "        OUT   0\n"
        "\n"
        "        IN    2         ; Read char 3\n"
        "        OR    0x20\n"
        "        OUT   0\n"
        "\n"
        "        IN    3         ; Read char 4\n"
        "        OR    0x20\n"
        "        OUT   0\n"
        "        HLT\n"
    ),
}


# ── I/O history extraction ────────────────────────────────────────────────────

import re as _re
_IN_RE  = _re.compile(r'EXECUTE IN port (\d+) → A = 0x([0-9A-Fa-f]+)')
_OUT_RE = _re.compile(r'EXECUTE OUT A\(0x([0-9A-Fa-f]+)\) → port (\d+)')

def extract_io_history(steps):
    """Pre-extract every IN and OUT event from the step list.

    Returns a list of {step, cycle, dir, port, value} dicts, one per
    IN or OUT instruction executed. Pre-computing this server-side means
    the frontend's Terminal panel can filter to 'events up to step N'
    simply by slicing the list, rather than scanning all steps each time
    the user moves the scrubber."""
    events = []
    for i, s in enumerate(steps):
        m = _OUT_RE.search(s.description)
        if m:
            events.append({'step': i, 'cycle': s.cycle, 'dir': 'out',
                           'value': int(m.group(1), 16), 'port': int(m.group(2))})
            continue
        m = _IN_RE.search(s.description)
        if m:
            events.append({'step': i, 'cycle': s.cycle, 'dir': 'in',
                           'port': int(m.group(1)), 'value': int(m.group(2), 16)})
    return events


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the visualiser's single HTML file directly — there's no
    build step, so this is just a static file response."""
    return send_from_directory(".", "cpu_visualiser_simple.html")


@app.route("/simple")
def index_simple():
    """Serve the Foundations View — a separate, much smaller HTML
    rendering aimed at the course's own CPU-architecture scope (control
    unit, ALU, registers, buses, fetch-decode-execute). Same engine,
    same /run_asm and /examples endpoints as the full visualiser — this
    is purely a different front end over the same simulation."""
    return send_from_directory(".", "cpu_visualiser_simple.html")


@app.route("/run", methods=["POST"])
def run_simulation():
    """Preset-input run: delegates straight to cpu.py's process_input(),
    which builds a small machine-code program for the chosen mode
    (int/str/key) and value, then executes it.
    Body: {mode, value, config, timer_irq, max_cycles}."""
    body      = request.get_json(force=True) or {}
    mode      = body.get("mode", "int")
    value     = str(body.get("value", "42"))
    cfg_name  = body.get("config", "microcontroller")
    timer_irq = bool(body.get("timer_irq", False))
    max_cycles = body.get("max_cycles", None)  # None -> use profile default

    cfg = CONFIGS.get(cfg_name, CPUConfig.microcontroller)()
    if max_cycles:
        cfg.max_sim_cycles = int(max_cycles)

    try:
        cpu, steps = process_input(value, mode, cfg,
                                   timer_irq=timer_irq, max_cycles=max_cycles)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(build_run_summary(cpu, steps))


@app.route("/run_asm", methods=["POST"])
def run_custom_program():
    """Custom-assembly run: assemble() the user's source into bytes, then
    feed them straight into cpu.py's CPU.run_program() — the same
    execution path the preset programs use, just with hand-written
    machine code instead of process_input()'s auto-generated bytes.
    Body: {asm_text, config, timer_irq, max_cycles, stdin}.

    max_cycles: override the profile's default cycle limit — needed for
      complex programs like factorial whose iteration count exceeds the
      default budget. The UI exposes this as the Sim Length slider.

    stdin: string of ASCII characters (or space-separated hex like
      '0x48 0x65') pre-loaded into input registers before the run,
      so IN port-0 reads the first character, IN port-1 the second, etc."""
    body      = request.get_json(force=True) or {}
    asm_text  = body.get("asm_text", "")
    cfg_name  = body.get("config", "microcontroller")
    timer_irq = bool(body.get("timer_irq", False))
    max_cycles = body.get("max_cycles", None)
    stdin_data = body.get("stdin", "")

    cfg = CONFIGS.get(cfg_name, CPUConfig.microcontroller)()
    if max_cycles:
        cfg.max_sim_cycles = int(max_cycles)

    cpu = CPU(cfg)
    region = cpu.memory._region("Program Code")
    # 4096-byte cap for 16/64-bit profiles (was 256 — too tight for
    # legitimately larger programs like factorial). MCU stays at its
    # physical Flash limit (48 bytes).
    max_bytes = min(region.size, 4096) if region else 64

    try:
        program, labels, source_map = assemble(asm_text, cpu.memory.code_base,
                                                address_bits=cfg.address_bits, max_bytes=max_bytes,
                                                symbols=build_dma_symbols(cpu))
    except AsmError as exc:
        return jsonify({"error": str(exc), "line": exc.line_no}), 400
    except Exception as exc:
        return jsonify({"error": f"Assembly failed: {exc}"}), 400

    # Pre-load stdin characters into input registers so IN instructions
    # can read them. Accepts either a plain string ("Hello") or
    # space-separated hex bytes ("0x48 0x65 0x6C 0x6C 0x6F").
    if stdin_data:
        chars: list[int] = []
        parts = stdin_data.split()
        if len(parts) > 1 or (len(parts) == 1 and parts[0].startswith('0x')):
            for tok in parts:
                try:
                    chars.append(int(tok, 16) & 0xFF)
                except ValueError:
                    pass
        else:
            chars = [ord(c) & 0xFF for c in stdin_data]
        for i, val in enumerate(chars[:8]):   # max 8 input ports
            addr = cpu.memory.io_input_base + i
            if 0 <= addr < cpu.memory.size:
                try:
                    cpu.memory.write(addr, val)
                except Exception:
                    pass

    try:
        steps = cpu.run_program(list(program), timer_irq=timer_irq)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    result = build_run_summary(cpu, steps)
    result["assembled_bytes"] = list(program)
    result["labels"] = labels
    result["source_text"] = asm_text
    result["source_map"] = source_map
    source_by_addr = {entry["addr"]: entry for entry in source_map if not entry.get("label_only")}
    for ins in result.get("disassembly", []):
        mapped = source_by_addr.get(ins.get("addr"))
        if mapped:
            ins["source_line"] = mapped.get("line_no")
            ins["source"] = mapped.get("source")
    return jsonify(result)


@app.route("/configs", methods=["GET"])
def list_configs():
    """Expose the available CPUConfig presets (microcontroller / expanded
    address-space / modern memory-hierarchy demonstrator) so a front end
    can build an architecture-select dropdown from real config data —
    names, descriptions, register names, default cycle budget — instead
    of duplicating that information client-side."""
    return jsonify({key: config_to_dict(factory()) for key, factory in CONFIGS.items()})


@app.route("/examples", methods=["GET"])
def get_examples():
    """Returns the canned example programs shown in the Custom Assembly
    example dropdown, keyed by the same names used as <option> values
    in the frontend's #sel-example selector."""
    return jsonify(EXAMPLE_PROGRAMS)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  CPU Visualiser  →  http://localhost:{port}\n"
          f"     (cpu_visualiser_simple.html must be in the same folder)\n")
    app.run(debug=True, port=port, use_reloader=False)
