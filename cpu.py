"""
Computer System Simulation Engine
Simulates: CPU registers, ALU, buses, cache hierarchy, RAM, ROM, interrupts, DMA

─────────────────────────────────────────────────────────────────────────────
HARDWARE PROFILES  — pass a CPUConfig to CPU() to choose your target system:

    cpu = CPU()                              # default: 8-bit teaching computer
    cpu = CPU(CPUConfig.microcontroller())   # 8-bit teaching core (explicit)
    cpu = CPU(CPUConfig.embedded_16bit())    # expanded 16-bit address space
    cpu = CPU(CPUConfig.modern_desktop())    # modern memory-hierarchy demo

    # Build a fully custom profile:
    cfg = CPUConfig(
        name                = "My 32-bit CPU",
        word_bits           = 32,
        address_bits        = 16,     # 64 KB address space
        gp_register_names   = ["EAX","EBX","ECX","EDX","ESI","EDI","ESP","EBP"],
        pipeline_stages     = 5,
        has_branch_predictor= True,
        branch_penalty_cycles= 5,
        cache_levels        = 3,
        l1_sets=16, l1_ways=4, l1_line_bytes=16, l1_latency=2,
        l2_sets=64, l2_ways=8, l2_line_bytes=32, l2_latency=8,
        l3_sets=256,l3_ways=8, l3_line_bytes=64, l3_latency=30,
        ram_latency_cycles   = 80,
        has_virtual_memory  = False,
        clock_speed_mhz     = 500.0,
        max_sim_cycles      = 1500,
    )
    cpu = CPU(cfg)
    print(cpu.config.summary())              # inspect the active profile
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum



# ═══════════════════════════════════════════════════════════════════════════════
#  CPU CONFIGURATION  ── edit here, or pick a preset, to model different CPUs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CPUConfig:
    """
    All tunable hardware parameters for the simulation.

    Every field has a sensible default (8-bit MCU).  Override individual
    fields for fine-grained experiments, or start from a preset:

        CPUConfig.microcontroller()  8-bit teaching core, 256 B address space
        CPUConfig.embedded_16bit()  same 8-bit ISA, expanded 64 KB address space
        CPUConfig.modern_desktop()  same 8-bit ISA, modern-style memory hierarchy

    Important model boundary: all three presets execute the same educational
    8-bit accumulator ISA. Wider presets change address width, memory layout,
    cache hierarchy and selected timing characteristics; they do not turn the
    ALU or instruction set into a real 16-bit or 64-bit implementation.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    name:        str = "Custom CPU"
    description: str = ""

    # ── Datapath & address bus ────────────────────────────────────────────────
    word_bits:    int = 8   # Comparative profile label; executable ALU remains 8-bit
    address_bits: int = 8   # Address-bus width → total RAM = 2 ** address_bits bytes

    # ── General-purpose registers ─────────────────────────────────────────────
    # The FIRST FOUR names always map to A / B / C / D (the core accumulator and
    # scratch registers used throughout execute_instruction).  Names beyond index
    # 3 become "extended" GP registers shown in the register snapshot but not
    # directly addressable by the current ISA — exactly like how x86-64's R8-R15
    # are inaccessible to 8086 code.
    gp_register_names: list = field(
        default_factory=lambda: ["A", "B", "C", "D"]
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    pipeline_stages: int = 2
    # 2  = minimal two-stage (fetch → execute)            e.g. PIC
    # 3  = fetch → decode → execute                       e.g. AVR
    # 5  = classic 5-stage RISC (IF ID EX MEM WB)        e.g. MIPS R3000
    # 19 = approximate modern superscalar depth           e.g. Intel Core

    has_branch_predictor:  bool = False  # True adds branch-penalty stall steps
    branch_penalty_cycles: int  = 0     # cycles wasted on a mis-prediction
    has_out_of_order:      bool = False  # profile label only — NOT modeled; instructions
                                          # still issue/complete strictly in program order

    # ── Cache hierarchy ───────────────────────────────────────────────────────
    cache_levels: int = 2   # 1, 2, or 3

    l1_sets:       int = 4
    l1_ways:       int = 4
    l1_line_bytes: int = 4
    l1_latency:    int = 1

    l2_sets:       int = 8
    l2_ways:       int = 4
    l2_line_bytes: int = 4
    l2_latency:    int = 4

    # L3 — only used when cache_levels == 3
    l3_sets:       int = 0
    l3_ways:       int = 0
    l3_line_bytes: int = 0
    l3_latency:    int = 0

    # ── Main memory ───────────────────────────────────────────────────────────
    ram_latency_cycles: int = 10   # DRAM access time on a full cache miss

    # ── Virtual memory ────────────────────────────────────────────────────────
    has_virtual_memory: bool = False
    page_size_bytes:    int  = 256  # TLB page granularity (informational)

    # ── Simulation limits ─────────────────────────────────────────────────────
    clock_speed_mhz: float = 16.0  # shown in summary; does not scale timing math
    max_sim_cycles:  int   = 500   # hard stop — raise for longer programs

    # ─────────────────────────────────────────────────────────────────────────
    #  Preset factory methods
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def microcontroller(cls) -> "CPUConfig":
        """
        8-bit microcontroller (AVR ATmega / PIC class).

        • 8-bit ALU  •  256-byte flat address space
        • 4 GP registers (A B C D)
        • 2-level cache: tiny L1 + small L2
        • No branch predictor, no virtual memory
        • 16 MHz (typical for ATmega328P / Arduino Uno)
        """
        return cls(
            name="8-bit Teaching Computer",
            description=(
                "Educational 8-bit accumulator core with a 256 B address space. "
                "Cache and timing behaviour are simplified teaching models."
            ),
            word_bits=8,   address_bits=8,
            gp_register_names=["A","B","C","D"],
            pipeline_stages=2,
            has_branch_predictor=False, branch_penalty_cycles=0,
            has_out_of_order=False,
            cache_levels=2,
            l1_sets=4,  l1_ways=4,  l1_line_bytes=4,  l1_latency=1,
            l2_sets=8,  l2_ways=4,  l2_line_bytes=4,  l2_latency=4,
            l3_sets=0,  l3_ways=0,  l3_line_bytes=0,  l3_latency=0,
            ram_latency_cycles=10,
            has_virtual_memory=False, page_size_bytes=256,
            clock_speed_mhz=16.0,    max_sim_cycles=500,
        )

    @classmethod
    def embedded_16bit(cls) -> "CPUConfig":
        """
        Expanded-address-space teaching system.

        • Same executable 8-bit accumulator ISA as the base profile
        • 16-bit address bus  •  64 KB address space
        • Additional register names are illustrative/display-only
        • Larger two-level cache and latency settings
        • Pipeline depth and clock speed are contextual labels, not a
          genuinely overlapping 16-bit processor pipeline
        """
        return cls(
            name="Expanded Address-Space System",
            description=(
                "The 8-bit teaching ISA with a 16-bit address bus and 64 KB memory map. "
                "Register names, pipeline depth and clock are comparative context."
            ),
            word_bits=16,  address_bits=16,
            gp_register_names=["AX","BX","CX","DX","SI","DI","BP","IX"],
            pipeline_stages=5,
            has_branch_predictor=False, branch_penalty_cycles=3,
            has_out_of_order=False,
            cache_levels=2,
            l1_sets=8,  l1_ways=4,  l1_line_bytes=8,  l1_latency=2,
            l2_sets=32, l2_ways=8,  l2_line_bytes=8,  l2_latency=8,
            l3_sets=0,  l3_ways=0,  l3_line_bytes=0,  l3_latency=0,
            ram_latency_cycles=20,
            has_virtual_memory=False, page_size_bytes=256,
            clock_speed_mhz=25.0,     max_sim_cycles=1000,
        )

    @classmethod
    def modern_desktop(cls) -> "CPUConfig":
        """
        Modern memory-hierarchy demonstrator.

        NOTE: This remains the same executable 8-bit teaching ISA. It adds a
        modern-style three-level cache, an adaptive branch predictor, longer
        memory latencies and an assumed-hit TLB step. The 64-bit register names,
        19-stage pipeline depth, out-of-order capability and 3.6 GHz clock are
        comparative labels only; they are not implemented execution mechanisms.
        """
        return cls(
            name="Modern Memory-Hierarchy Demonstrator",
            description=(
                "The 8-bit teaching ISA attached to a modern-style three-level cache, "
                "branch predictor and assumed-hit TLB model. 64-bit naming, clock, deep "
                "pipeline and out-of-order execution are illustrative labels only."
            ),
            word_bits=64,  address_bits=16,
            gp_register_names=[
                "RAX","RBX","RCX","RDX","RSI","RDI",
                "R8","R9","R10","R11",
            ],
            pipeline_stages=19,
            has_branch_predictor=True,  branch_penalty_cycles=15,
            has_out_of_order=True,
            cache_levels=3,
            l1_sets=64,   l1_ways=8,  l1_line_bytes=64, l1_latency=4,
            l2_sets=512,  l2_ways=8,  l2_line_bytes=64, l2_latency=12,
            l3_sets=4096, l3_ways=16, l3_line_bytes=64, l3_latency=40,
            ram_latency_cycles=200,
            has_virtual_memory=True, page_size_bytes=4096,
            clock_speed_mhz=3600.0,  max_sim_cycles=2000,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Derived helpers
    # ─────────────────────────────────────────────────────────────────────────

    def word_mask(self) -> int:
        """Bitmask for the comparative profile width.

        The current executable ISA still masks ALU values to eight bits; this
        helper is retained for configuration/reporting compatibility.
        """
        return (1 << self.word_bits) - 1

    def memory_size(self) -> int:
        """Total addressable bytes."""
        return 1 << self.address_bits

    def summary(self) -> str:
        """Human-readable hardware profile — useful as a classroom reference."""
        cache_lines = [
            f"  L1:  {self.l1_sets} sets × {self.l1_ways}-way, "
            f"{self.l1_line_bytes} B lines, {self.l1_latency}-cycle latency",
            f"  L2:  {self.l2_sets} sets × {self.l2_ways}-way, "
            f"{self.l2_line_bytes} B lines, {self.l2_latency}-cycle latency",
        ]
        if self.cache_levels >= 3:
            cache_lines.append(
                f"  L3:  {self.l3_sets} sets × {self.l3_ways}-way, "
                f"{self.l3_line_bytes} B lines, {self.l3_latency}-cycle latency"
            )
        cache_lines.append(f"  RAM: {self.ram_latency_cycles}-cycle latency on full miss")

        bp  = (f"Yes (mis-predict penalty: {self.branch_penalty_cycles} cycles)"
               if self.has_branch_predictor else "No")
        vm  = (f"Yes ({self.page_size_bytes:,} B pages)"
               if self.has_virtual_memory else "No")
        sep = "═" * max(2, 48 - len(self.name))

        lines = [
            f"╔═══ {self.name} {sep}",
            f"║  {self.description}",
            f"╠═══ Architecture",
            f"║  Executable core:   8-bit accumulator ISA / ALU",
            f"║  Profile word label:{self.word_bits:>7}-bit (comparative only)",
            f"║  Address space:     {self.memory_size():,} bytes  ({self.address_bits}-bit bus)",
            f"║  GP registers:      {len(self.gp_register_names)}  "
            f"({', '.join(self.gp_register_names)})",
            f"║  Clock label:       {self.clock_speed_mhz:,.1f} MHz (informational)",
            f"╠═══ Pipeline",
            f"║  Stage-depth label: {self.pipeline_stages} (execution remains sequential)",
            f"║  Branch predictor:  {bp}",
            f"║  Out-of-order exec: {'label only, not modeled' if self.has_out_of_order else 'No'}",
            f"╠═══ Cache hierarchy  ({self.cache_levels} level"
            f"{'s' if self.cache_levels > 1 else ''})",
        ] + [f"║  {cl}" for cl in cache_lines] + [
            f"╚═══ Virtual memory:  {vm}",
        ]
        return "\n".join(lines)


# ── Memory types ──────────────────────────────────────────────────────────────
# Different types of memory technologies simulated in the system.

class MemoryType(Enum):
    SRAM   = "SRAM"          # Static RAM (used for fast cache memory)
    DRAM   = "DRAM"          # Dynamic RAM (used for main system memory)
    ROM    = "ROM"           # Mask ROM (fixed, unchangeable memory)
    PROM   = "PROM"          # Programmable ROM
    EPROM  = "EPROM"         # Erasable PROM (UV erasable)
    EEPROM = "EEPROM"        # Electrically erasable programmable ROM
    FLASH  = "Flash"         # Flash memory (typically used for BIOS/firmware)
    VMEM   = "Virtual"       # Virtual memory page (for OS-level paging)


"""Represents a single byte of memory at a specific address."""
@dataclass
class MemoryCell:
    address: int
    value: int = 0
    label: str = ""
    writable: bool = True
    mem_type: MemoryType = MemoryType.DRAM
    last_accessed: float = 0.0
    access_count: int = 0

    """Reads the cell value and updates access metrics."""
    def read(self) -> int:
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value

    """Writes an 8-bit value to the cell if it is writable."""
    def write(self, val: int):
        if not self.writable:
            raise PermissionError(f"Write to read-only address 0x{self.address:04X}")
        # Enforce 8-bit limit by applying a bitwise AND mask (0xFF)
        self.value = val & 0xFF
        self.last_accessed = time.time()
        self.access_count += 1


"""Defines a contiguous block of memory designated for a specific purpose."""
@dataclass
class MemoryRegion:
    name: str
    start: int
    size: int
    mem_type: MemoryType
    writable: bool = True
    description: str = ""
    cells: list[MemoryCell] = field(default_factory=list)

    def __post_init__(self):
        # Automatically populate the region with MemoryCell objects upon initialisation
        self.cells = [
            MemoryCell(self.start + i, 0, writable=self.writable, mem_type=self.mem_type)
            for i in range(self.size)
        ]


# ── Memory map ─────────────────────────────────────────────────────────────────
"""Simulates a flat address space with typed regions, sized by CPUConfig."""
class MemoryMap:
    def __init__(self, config: "CPUConfig"):
        self.config = config
        self.size   = config.memory_size()         # 256 B, 64 KB, etc.
        self.cells: list[MemoryCell] = [MemoryCell(i) for i in range(self.size)]
        self.regions: list[MemoryRegion] = []
        self._setup_regions()

    """Chooses an explicit layout or falls back to proportional scaling."""
    def _setup_regions(self):
        if self.size == 256:
            specs = self._layout_256()
        elif self.size == 65536:
            specs = self._layout_64k()
        else:
            specs = self._layout_proportional(self.size)

        self.regions = [
            MemoryRegion(name, start, size, mtype, writable, desc)
            for name, start, size, mtype, writable, desc in specs
        ]

        # Map regional cells back into the flat address array
        for region in self.regions:
            for i, cell in enumerate(region.cells):
                self.cells[region.start + i] = cell

        self._init_rom()

    # ── Layout definitions ─────────────────────────────────────────────────

    def _layout_256(self) -> list:
        """Original 256-byte layout (8-bit microcontroller)."""
        return [
            ("Interrupt Vector Table", 0x00, 16, MemoryType.ROM,   False, "Fixed jump addresses for interrupt handlers"),
            ("BIOS / Firmware",        0x10, 16, MemoryType.FLASH, False, "Startup code and hardware initialisation"),
            ("Stack",                  0x20, 32, MemoryType.DRAM,  True,  "LIFO store for return addresses & local vars"),
            ("Heap",                   0x40, 48, MemoryType.DRAM,  True,  "Dynamically allocated data"),
            ("Program Code",           0x70, 48, MemoryType.DRAM,  True,  "Loaded executable instructions"),
            ("Data Segment",           0xA0, 48, MemoryType.DRAM,  True,  "Global & static variables"),
            ("Input Registers",        0xD0,  8, MemoryType.DRAM,  True,  "Memory-mapped input ports (port 0=keyboard)"),
            ("Output Registers",       0xD8,  8, MemoryType.DRAM,  True,  "Memory-mapped output ports (port 0=display char)"),
            ("DMA Buffer",             0xE0, 16, MemoryType.DRAM,  True,  "Direct Memory Access transfer buffer"),
            ("Video / Display Buffer", 0xF0, 16, MemoryType.DRAM,  True,  "Frame buffer for display output"),
        ]
        # Sum: 16+16+32+48+48+48+8+8+16+16 = 256 ✓

    def _layout_64k(self) -> list:
        """64 KB layout for 16-bit and modern (64-bit scaled) profiles.

        Key teaching point — Zero-page RAM (0x0010–0x00FF):
        Many real 16-bit/64-bit CPUs keep a small block of fast, writable
        RAM at very low addresses so that 1-byte-operand instructions can
        still reach working data without needing a full 16-bit address.
        The 6502, Z80, and MSP430 all exploit this idea.  This simulation
        does the same: process_input uses zero-page addresses (0x10-0xFF)
        for scratch data, mirroring that architectural pattern.

        Layout (sum = 65 536 bytes):
          0x0000  16 B  IVT ROM   (8 vectors × 2 bytes)
          0x0010 240 B  Zero-page DRAM  (writable scratch; 1-byte addressable)
          0x0100 256 B  BIOS FLASH      (ISR stub at 0x01FE–0x01FF)
          0x0200  4 KB  Stack DRAM
          0x1200 11 KB  Heap DRAM
          0x4000 12 KB  Code DRAM      ← PC reset here
          0x7000 12 KB  Data DRAM
          0xA000 256 B  I/O Input DRAM
          0xA100 256 B  I/O Output DRAM
          0xA200  1 KB  DMA Buffer DRAM
          0xA600 23 KB  Video DRAM
        """
        return [
            ("Interrupt Vector Table", 0x0000,   16, MemoryType.ROM,   False, "8 interrupt vectors × 2 bytes"),
            ("Zero-page RAM",          0x0010,  240, MemoryType.DRAM,  True,  "Fast writable scratch (1-byte addressable, like 6502 zero-page)"),
            ("BIOS / Firmware",        0x0100,  256, MemoryType.FLASH, False, "Startup code; ISR stub at 0x01FE–0x01FF"),
            ("Stack",                  0x0200, 4096, MemoryType.DRAM,  True,  "Call stack (grows downward from 0x11FF)"),
            ("Heap",                   0x1200,11776, MemoryType.DRAM,  True,  "Dynamically allocated data"),
            ("Program Code",           0x4000,12288, MemoryType.DRAM,  True,  "Loaded executable instructions"),
            ("Data Segment",           0x7000,12288, MemoryType.DRAM,  True,  "Global & static variables"),
            ("Input Registers",        0xA000,  256, MemoryType.DRAM,  True,  "Memory-mapped input ports (port 0=keyboard)"),
            ("Output Registers",       0xA100,  256, MemoryType.DRAM,  True,  "Memory-mapped output ports (port 0=display)"),
            ("DMA Buffer",             0xA200, 1024, MemoryType.DRAM,  True,  "Direct Memory Access transfer buffer"),
            ("Video / Display Buffer", 0xA600,23040, MemoryType.DRAM,  True,  "Frame buffer for display output"),
        ]
        # Size check: 16+240+256+4096+11776+12288+12288+256+256+1024+23040 = 65536

    def _layout_proportional(self, total: int) -> list:
        """Fallback: scale the 256-byte fractions to any address space size."""
        # Fractions derived from the original 256-byte layout
        ratios = [16, 16, 32, 48, 48, 48, 8, 8, 16]   # first 9 regions
        sizes  = [max(8, int(total * r / 256) & ~7) for r in ratios]
        video  = max(8, total - sum(sizes))

        names_meta = [
            ("Interrupt Vector Table", MemoryType.ROM,   False, "Jump table for interrupt handlers"),
            ("BIOS / Firmware",        MemoryType.FLASH, False, "Startup code and hardware initialisation"),
            ("Stack",                  MemoryType.DRAM,  True,  "LIFO store for return addresses & local vars"),
            ("Heap",                   MemoryType.DRAM,  True,  "Dynamically allocated data"),
            ("Program Code",           MemoryType.DRAM,  True,  "Loaded executable instructions"),
            ("Data Segment",           MemoryType.DRAM,  True,  "Global & static variables"),
            ("Input Registers",        MemoryType.DRAM,  True,  "Memory-mapped input ports"),
            ("Output Registers",       MemoryType.DRAM,  True,  "Memory-mapped output ports"),
            ("DMA Buffer",             MemoryType.DRAM,  True,  "Direct Memory Access transfer buffer"),
        ]
        specs, pos = [], 0
        for (name, mtype, writable, desc), sz in zip(names_meta, sizes):
            specs.append((name, pos, sz, mtype, writable, desc))
            pos += sz
        specs.append(("Video / Display Buffer", pos, video, MemoryType.DRAM, True, "Frame buffer"))
        return specs

    # ── ROM initialisation ─────────────────────────────────────────────────

    def _init_rom(self):
        """Pre-load IVT and BIOS with a generic ISR stub at the end of BIOS."""
        ivt  = self._region("Interrupt Vector Table")
        bios = self._region("BIOS / Firmware")
        if not ivt or not bios:
            return

        # ISR stub: NOP then RET at the last two bytes of the BIOS region
        stub_addr = bios.start + bios.size - 2

        # IVT: fill with 16-bit little-endian entries pointing to the stub
        num_vectors = min(ivt.size // 2, 128)   # up to 128 vectors
        for i in range(num_vectors):
            ea = ivt.start + i * 2
            self.cells[ea].value     = stub_addr & 0xFF
            self.cells[ea + 1].value = (stub_addr >> 8) & 0xFF

        # BIOS body: NOPs
        for i in range(bios.size - 2):
            self.cells[bios.start + i].value = 0x12   # NOP opcode
        # Stub at last two bytes: NOP then RET
        self.cells[stub_addr].value     = 0x12   # NOP
        self.cells[stub_addr + 1].value = 0x0F   # RET (return from interrupt)

    # ── Address-property helpers (used by CPU — no more magic constants) ───

    def _region(self, name: str) -> Optional[MemoryRegion]:
        for r in self.regions:
            if r.name == name:
                return r
        return None

    @property
    def code_base(self) -> int:
        """Default program load address (start of Program Code region)."""
        r = self._region("Program Code")
        return r.start if r else 0

    @property
    def data_base(self) -> int:
        """Start of the Data Segment region."""
        r = self._region("Data Segment")
        return r.start if r else 0

    @property
    def stack_top(self) -> int:
        """Initial stack-pointer value (last byte of the Stack region)."""
        r = self._region("Stack")
        return (r.start + r.size - 1) if r else 0

    @property
    def io_input_base(self) -> int:
        """Base address for memory-mapped input ports."""
        r = self._region("Input Registers")
        return r.start if r else 0

    @property
    def io_output_base(self) -> int:
        """Base address for memory-mapped output ports."""
        r = self._region("Output Registers")
        return r.start if r else 0

    def region_for(self, addr: int) -> Optional[MemoryRegion]:
        """Returns the specific MemoryRegion an address belongs to."""
        for r in self.regions:
            if r.start <= addr < r.start + r.size:
                return r
        return None

    def read(self, addr: int) -> int:
        if 0 <= addr < self.size:
            return self.cells[addr].read()
        raise IndexError(f"Address 0x{addr:04X} out of range")

    def write(self, addr: int, val: int):
        if 0 <= addr < self.size:
            self.cells[addr].write(val)
        else:
            raise IndexError(f"Address 0x{addr:04X} out of range")


# ── Cache ──────────────────────────────────────────────────────────────────────
"""Represents a single line of memory inside the cache."""
@dataclass
class CacheLine:
    tag: int = -1
    data: list[int] = field(default_factory=lambda: [0]*4)
    valid: bool = False
    dirty: bool = False
    last_used: float = 0.0

"""N-way set-associative cache with LRU eviction. Fully configured by CPUConfig."""
class Cache:
    # NOTE: SETS / WAYS / LINE_SIZE are now instance variables set in __init__,
    # not class constants.  This allows each cache level (L1, L2, L3) to have
    # independent geometry — mirroring how real hardware cache levels differ.

    def __init__(self, name: str, sets: int, ways: int,
                 line_bytes: int, latency_cycles: int):
        self.name      = name
        self.SETS      = sets        # number of cache sets
        self.WAYS      = ways        # associativity (lines per set)
        self.LINE_SIZE = line_bytes  # bytes per cache line
        self.latency   = latency_cycles
        # Initialise a 2D list simulating cache sets and ways
        self.lines: list[list[CacheLine]] = [
            [CacheLine() for _ in range(self.WAYS)] for _ in range(self.SETS)
        ]
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    """Calculates the efficiency of the cache."""
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    """Determines which cache set an address maps to."""
    def _set_idx(self, addr: int) -> int:
        return (addr // self.LINE_SIZE) % self.SETS

    """Extracts the tag for cache lookup."""
    def _tag(self, addr: int) -> int:
        return addr // (self.LINE_SIZE * self.SETS)

    """Checks if a memory address is currently in the cache."""
    def lookup(self, addr: int) -> tuple[bool, Optional[CacheLine]]:
        s = self._set_idx(addr)
        t = self._tag(addr)
        for line in self.lines[s]:
            if line.valid and line.tag == t:
                self.hits += 1
                line.last_used = time.time() # Update LRU metric
                return True, line
        self.misses += 1
        return False, None

    """Load a cache line; evict LRU if set is full."""
    def load(self, addr: int, data: list[int]) -> Optional[CacheLine]:
        s = self._set_idx(addr)
        t = self._tag(addr)
        
        # Find empty slot first to avoid unnecessary evictions
        for line in self.lines[s]:
            if not line.valid:
                line.tag = t
                line.data = data[:self.LINE_SIZE]
                line.valid = True
                line.dirty = False
                line.last_used = time.time()
                return None  # no eviction
                
        # LRU eviction: Find the line accessed least recently
        lru = min(self.lines[s], key=lambda l: l.last_used)
        evicted = CacheLine(lru.tag, lru.data[:], lru.valid, lru.dirty)
        self.evictions += 1
        
        # Overwrite the LRU line
        lru.tag = t
        lru.data = data[:self.LINE_SIZE]
        lru.valid = True
        lru.dirty = False
        lru.last_used = time.time()
        return evicted

    """Clears the entire cache by marking all lines as invalid."""
    def invalidate(self):
        for s in self.lines:
            for line in s:
                line.valid = False

    def write(self, addr: int, val: int) -> bool:
        """Write-through update: if addr is currently cached, update the stored
        byte in place so subsequent reads don't return stale data.

        This is the critical companion to load(): without it, STORE instructions
        write to memory but leave the cache line holding the old value, so the
        very next LOAD from that address (a cache HIT) returns the pre-STORE byte.
        Practically: 'result += factor; result = 0' loops would never actually
        zero result because the cache kept serving the accumulated total.

        Returns True if the address was found and updated in cache, False if it
        was not currently cached (i.e., a cache miss — nothing to update)."""
        s = self._set_idx(addr)
        t = self._tag(addr)
        offset = addr % self.LINE_SIZE
        for line in self.lines[s]:
            if line.valid and line.tag == t:
                if 0 <= offset < len(line.data):
                    line.data[offset] = val & 0xFF
                return True
        return False


# ── CPU Registers ──────────────────────────────────────────────────────────────
"""Status flags generated by Arithmetic Logic Unit operations."""
@dataclass
class Flags:
    zero:     bool = False   # Z — result was zero
    carry:    bool = False   # C — arithmetic carry/borrow
    negative: bool = False   # N — result was negative
    overflow: bool = False   # V — signed overflow
    interrupt: bool = True   # I — interrupts enabled

    def as_dict(self) -> dict:
        return {"Z": self.zero, "C": self.carry, "N": self.negative,
                "V": self.overflow, "I": self.interrupt}


"""Holds the CPU's internal working memory (Registers)."""
@dataclass
class Registers:
    # ── Core general-purpose registers (always present) ───────────────────────
    # These map to the FIRST FOUR names in CPUConfig.gp_register_names.
    # E.g. in a modern profile: A→RAX, B→RBX, C→RCX, D→RDX.
    # All execute_instruction logic uses A/B/C/D directly, so no code change
    # is needed there — only the display names change via gp_names below.
    A: int = 0      # Accumulator
    B: int = 0      # General purpose
    C: int = 0      # General purpose
    D: int = 0      # General purpose

    # ── Extended GP registers (modern profiles only) ───────────────────────────
    # Keys: register names from gp_register_names[4:] (e.g. RSI, RDI, R8…).
    # These are shown in the register snapshot but are not addressable by the
    # current ISA — analogous to x86-64's R8–R15 being inaccessible to 8086 code.
    extra_gp: dict = field(default_factory=dict)

    # Display names for A/B/C/D — set by CPU to match the active profile.
    # E.g. microcontroller → ["A","B","C","D"], modern → ["RAX","RBX","RCX","RDX"]
    gp_names: list = field(default_factory=lambda: ["A", "B", "C", "D"])

    # ── Special registers (address-wide; hold any valid address) ──────────────
    # CPU resets PC → memory.code_base and SP → memory.stack_top on init.
    PC: int = 0x70  # Program Counter
    SP: int = 0x3F  # Stack Pointer

    # ── Internal CPU registers (invisible to programmer) ──────────────────────
    MAR: int = 0    # Memory Address Register
    MDR: int = 0    # Memory Data Register
    IR:  int = 0    # Instruction Register (opcode)
    CIR: str = ""   # Current Instruction (decoded mnemonic)

    # ── ALU internal state ────────────────────────────────────────────────────
    ALU_A:   int = 0
    ALU_B:   int = 0
    ALU_OUT: int = 0
    FLAGS: Flags = field(default_factory=Flags)

    def as_dict(self) -> dict:
        # Rename A/B/C/D to the profile's display names (e.g. RAX/RBX/RCX/RDX)
        names = self.gp_names or ["A", "B", "C", "D"]
        gp = dict(zip(names[:4], [self.A, self.B, self.C, self.D]))
        gp.update(self.extra_gp)   # append extended registers (RSI, R8, …)
        return {
            **gp,
            "PC": self.PC, "SP": self.SP,
            "MAR": self.MAR, "MDR": self.MDR,
            "IR": self.IR, "CIR": self.CIR,
            "ALU_A": self.ALU_A, "ALU_B": self.ALU_B, "ALU_OUT": self.ALU_OUT,
        }


# ── Bus ────────────────────────────────────────────────────────────────────────
"""Enumeration of possible actions the system bus can be performing."""
class BusState(Enum):
    IDLE     = "idle"
    FETCH    = "fetch"        # Instruction fetch
    READ     = "mem_read"     # Data read
    WRITE    = "mem_write"    # Data write
    IO_READ  = "io_read"
    IO_WRITE = "io_write"
    DMA      = "dma"
    IRQ      = "irq"

"""Captures the state of the bus at a specific clock cycle."""
@dataclass
class BusSnapshot:
    state:   BusState
    address: int
    data:    int
    control: str    # "RD", "WR", "INTA", "DMA", etc.
    cycle:   int
    notes:   str = ""


# ── Interrupt system ──────────────────────────────────────────────────────────
"""Categorises the different hardware and software interrupts."""
class InterruptType(Enum):
    NMI      = "NMI"        # Non-maskable interrupt (hardware fault)
    IRQ0     = "IRQ0"       # Timer tick
    IRQ1     = "IRQ1"       # Keyboard
    IRQ2     = "IRQ2"       # Serial port
    IRQ3     = "IRQ3"       # Disk controller
    SOFTWARE = "SW"         # Software interrupt (syscall)
    DMA_DONE = "DMA_DONE"   # DMA transfer complete

"""Contains information regarding a triggered interrupt."""
@dataclass
class InterruptEvent:
    itype: InterruptType
    vector: int
    priority: int
    description: str
    cycle_raised: int


# ── Instruction set ───────────────────────────────────────────────────────────
# Defines the machine code opcodes supported by the simulated CPU.

INSTRUCTIONS = {
    0x01: ("LOAD",  "Load value from address into A",       ["addr"]),
    0x02: ("STORE", "Store A to address",                   ["addr"]),
    0x03: ("ADD",   "A = A + operand",                      ["operand"]),
    0x04: ("SUB",   "A = A - operand",                      ["operand"]),
    0x05: ("AND",   "A = A AND operand (bitwise)",          ["operand"]),
    0x06: ("OR",    "A = A OR operand (bitwise)",           ["operand"]),
    0x07: ("XOR",   "A = A XOR operand (bitwise)",          ["operand"]),
    0x08: ("CMP",   "Compare A with operand, set flags",    ["operand"]),
    0x09: ("JMP",   "Unconditional jump to address",        ["addr"]),
    0x0A: ("JZ",    "Jump if Zero flag set",                ["addr"]),
    0x0B: ("JNZ",   "Jump if Zero flag clear",              ["addr"]),
    0x0C: ("PUSH",  "Push A onto stack, SP--",              []),
    0x0D: ("POP",   "Pop from stack into A, SP++",          []),
    0x0E: ("CALL",  "Push PC, jump to subroutine",          ["addr"]),
    0x0F: ("RET",   "Pop PC, return from subroutine",       []),
    0x10: ("IN",    "Read from I/O port into A",            ["port"]),
    0x11: ("OUT",   "Write A to I/O port",                  ["port"]),
    0x12: ("NOP",   "No operation",                         []),
    0x13: ("DMA",   "Trigger a DMA transfer configured via the DMA control block", []),
    0x14: ("LDI",   "Load an immediate 8-bit value directly into A", ["operand"]),
    # ── Memory-operand arithmetic ─────────────────────────────────────────────
    # Fills the most common gap in a single-accumulator ISA: adding or
    # subtracting the CONTENTS of a memory address, not just a constant.
    # Without ADDM, "result += factor" (factor in RAM) needs an awkward
    # LOAD/PUSH/LOAD/POP/ADD dance. The memory read goes through the cache
    # hierarchy exactly like LOAD, so miss stalls appear correctly.
    0x15: ("ADDM",  "A = A + mem[addr]  (add memory to accumulator)", ["addr"]),
    0x16: ("SUBM",  "A = A - mem[addr]  (subtract memory from accumulator, via 2's complement)", ["addr"]),
    0xFF: ("HLT",   "Halt the CPU",                         []),
}


# ── Simulation step log ────────────────────────────────────────────────────────
"""Logs the details of a single fetch-decode-execute step for debugging/tracing."""
@dataclass
class SimStep:
    phase: str          # "fetch" | "decode" | "execute" | "writeback" | "interrupt" | "dma"
    cycle: int
    description: str
    detail: str
    bus: Optional[BusSnapshot] = None
    reg_snapshot: Optional[dict] = None
    mem_changed: list[tuple[int,int]] = field(default_factory=list)  # [(addr, val)]
    cache_event: str = ""   # "hit" | "miss" | "evict" | ""
    flags_snapshot: Optional[dict] = None


# ── CPU ────────────────────────────────────────────────────────────────────────
"""
    Simulated 8-bit CPU with:
    - Full register set (A, B, C, D, PC, SP, MAR, MDR, IR, FLAGS)
    - Three-bus architecture (data, address, control)
    - Two-level cache (L1 SRAM, L2 SRAM)
    - Interrupt controller (IRQ + NMI)
    - DMA controller
    - Fetch-Decode-Execute-Writeback cycle logging
    """
class CPU:

    def __init__(self, config: Optional["CPUConfig"] = None):
        # Use the supplied profile, or fall back to the 8-bit microcontroller preset.
        self.config = config or CPUConfig.microcontroller()
        cfg = self.config

        # ── Core state ────────────────────────────────────────────────────────
        self.registers = Registers()
        self.memory    = MemoryMap(cfg)

        # Reset PC / SP to the correct addresses for this memory layout.
        # (Registers defaults to 0x70/0x3F which only suits the 256-byte map.)
        self.registers.PC = self.memory.code_base
        self.registers.SP = self.memory.stack_top

        # Apply GP register profile: rename A/B/C/D and add extended registers.
        names = cfg.gp_register_names
        self.registers.gp_names = names[:4]        # display names for A B C D
        if len(names) > 4:
            self.registers.extra_gp = {n: 0 for n in names[4:]}

        # ── Cache hierarchy ───────────────────────────────────────────────────
        self.l1_cache = Cache("L1 SRAM",
                              cfg.l1_sets, cfg.l1_ways, cfg.l1_line_bytes, cfg.l1_latency)
        self.l2_cache = Cache("L2 SRAM",
                              cfg.l2_sets, cfg.l2_ways, cfg.l2_line_bytes, cfg.l2_latency)
        # L3 is optional — only present when cache_levels >= 3
        self.l3_cache: Optional[Cache] = None
        if cfg.cache_levels >= 3 and cfg.l3_sets > 0:
            self.l3_cache = Cache("L3 SRAM",
                                  cfg.l3_sets, cfg.l3_ways, cfg.l3_line_bytes, cfg.l3_latency)

        # ── Other simulation state ────────────────────────────────────────────
        self.cycle      = 0
        self.halted     = False
        self.bus_log:   list[BusSnapshot]     = []
        self.irq_queue: list[InterruptEvent]  = []
        self.step_log:  list[SimStep]         = []
        self.dma_active = False
        self.dma_src = self.dma_dst = self.dma_len = 0

        # ── Run termination state ─────────────────────────────────────────────
        # Exposed through the Flask summary so the learner can distinguish a
        # normal HLT from a cycle-limit stop, invalid opcode or runtime fault.
        self.stop_reason = "not_started"
        self.stop_detail = "Simulation has not started."
        self.instructions_executed = 0
        self.max_cycles_used = cfg.max_sim_cycles
        self.timer_irq_enabled = False

        # ── Branch predictor state ────────────────────────────────────────────
        # Per-branch-SITE (keyed by the JZ/JNZ instruction's own address, so a
        # branch inside a loop accumulates history across iterations) 2-bit
        # saturating counters — the textbook predictor design. Counter value
        # 0-3: 0/1 predict NOT-TAKEN (weak/strong), 2/3 predict TAKEN
        # (weak/strong). Starts at 1 ("weakly not-taken") so the very first
        # encounter of any branch matches the old fixed "always predict
        # not-taken" behaviour — the adaptiveness only shows up on repeats.
        self._branch_predictor: dict[int, int] = {}
        # Address of the instruction currently being fetched/decoded/executed
        # — captured at the top of _fetch() (where MAR == PC, before either
        # gets reused for an operand fetch) so branch handlers in
        # execute_instruction() can still identify "which JZ/JNZ is this"
        # after MAR has long since been overwritten.
        self._current_instr_addr: int = self.registers.PC

    # ── Branch predictor helpers ────────────────────────────────────────────
    def _predict_branch(self, branch_addr: int) -> bool:
        """Returns the CURRENT prediction (True = predict taken) for the
        branch at this address, based on its saturating counter — without
        modifying any state. Unseen branches default to counter=1 (weakly
        not-taken), matching a predictor with no history yet."""
        counter = self._branch_predictor.get(branch_addr, 1)
        return counter >= 2

    def _update_branch_predictor(self, branch_addr: int, taken: bool):
        """Saturating increment (branch was taken) or decrement (was not
        taken) of this branch site's 2-bit counter, clamped to [0, 3]."""
        counter = self._branch_predictor.get(branch_addr, 1)
        counter = min(3, counter + 1) if taken else max(0, counter - 1)
        self._branch_predictor[branch_addr] = counter

    # ── Internal helpers ────────────────────────────────────────────────────
    """Advances the simulated hardware clock."""
    def _tick(self, n: int = 1):
        
        self.cycle += n

    """Advances the clock one cycle at a time, logging a visible stall step each cycle.
    This makes multi-cycle memory latencies visible in the step log rather than
    silently jumping the cycle counter — important for educational accuracy."""
    def _tick_stall(self, n: int, phase: str, reason: str, bus=None, detail: str = ""):
        for i in range(n):
            self.cycle += 1
            base = detail or f"CPU pipeline is frozen — waiting for: {reason}."
            full_detail = (
                f"Stall cycle {i + 1} of {n}. {base} "
                f"No new instruction can be issued until this operation completes."
            )
            self._step(
                phase=phase,
                desc=f"STALL ({i + 1}/{n}) — {reason}",
                detail=full_detail,
                bus=bus,
            )

    """Records the state of the bus lines to the bus log."""
    def _bus(self, state: BusState, addr: int, data: int, ctrl: str, notes: str = ""):
        snap = BusSnapshot(state, addr, data, ctrl, self.cycle, notes)
        self.bus_log.append(snap)
        return snap

    """Records a simulation step containing register and memory snapshots."""
    def _step(self, phase: str, desc: str, detail: str,
              bus=None, mem_changed=None, cache_event="", flags=None):
        snap = SimStep(
            phase=phase, cycle=self.cycle,
            description=desc, detail=detail,
            bus=bus,
            reg_snapshot=self.registers.as_dict(),
            mem_changed=mem_changed or [],
            cache_event=cache_event,
            flags_snapshot=self.registers.FLAGS.as_dict() if flags else None,
        )
        self.step_log.append(snap)
        return snap

    """Try L1 → L2 → (optional) L3 → RAM. Returns (value, event_str)."""
    def _cache_read(self, addr: int, phase: str = "fetch") -> tuple[int, str]:
        cfg = self.config

        # ── Optional TLB step (virtual-memory profiles) ──────────────────
        if cfg.has_virtual_memory:
            self._step(
                phase, f"TLB lookup — 0x{addr:04X}",
                f"Virtual address 0x{addr:04X} translated through the TLB. "
                f"Page size: {cfg.page_size_bytes:,} B. "
                f"TLB hit assumed (1 cycle). A TLB miss would require a "
                f"hardware page-table walk costing up to {cfg.ram_latency_cycles} additional cycles.",
            )
            self._tick(1)

        # ── L1 hit ───────────────────────────────────────────────────────
        hit, line = self.l1_cache.lookup(addr)
        if hit:
            offset = addr % self.l1_cache.LINE_SIZE
            self._tick_stall(
                self.l1_cache.latency, phase, "L1 cache lookup",
                detail=(
                    f"Address 0x{addr:04X} found in L1 SRAM. "
                    f"L1 has {self.l1_cache.latency}-cycle latency — the fastest memory tier. "
                    f"Data is ready at the end of this stall."
                )
            )
            return line.data[offset], "L1 hit"

        # ── L2 hit ───────────────────────────────────────────────────────
        hit2, line2 = self.l2_cache.lookup(addr)
        if hit2:
            offset = addr % self.l2_cache.LINE_SIZE
            val = line2.data[offset]
            data = [self.memory.read(addr - (addr % self.l1_cache.LINE_SIZE) + i)
                    for i in range(self.l1_cache.LINE_SIZE)]
            self.l1_cache.load(addr, data)
            latency = self.l1_cache.latency + self.l2_cache.latency
            self._tick_stall(
                latency, phase, "L2 → L1 cache fill",
                detail=(
                    f"Address 0x{addr:04X} not in L1 — L1 miss. Found in L2 SRAM. "
                    f"Line promoted from L2 ({self.l2_cache.latency}-cycle) to L1 "
                    f"({self.l1_cache.latency}-cycle). Total wait: {latency} cycles. "
                    f"This is why L2 hits are slower than L1 hits but far faster than RAM."
                )
            )
            return val, "L2 hit → L1 fill"

        # ── L3 hit (only when a third cache level is configured) ──────────
        if self.l3_cache is not None:
            hit3, line3 = self.l3_cache.lookup(addr)
            if hit3:
                offset = addr % self.l3_cache.LINE_SIZE
                val = line3.data[offset]
                data = [self.memory.read(addr - (addr % self.l1_cache.LINE_SIZE) + i)
                        for i in range(self.l1_cache.LINE_SIZE)]
                self.l1_cache.load(addr, data)
                self.l2_cache.load(addr, data)
                latency = (self.l1_cache.latency + self.l2_cache.latency
                           + self.l3_cache.latency)
                self._tick_stall(
                    latency, phase, "L3 → L2 → L1 cache fill",
                    detail=(
                        f"Address 0x{addr:04X} found in L3 (last-level cache). "
                        f"Line must traverse the full hierarchy: "
                        f"L3 ({self.l3_cache.latency}-cycle) → "
                        f"L2 ({self.l2_cache.latency}-cycle) → "
                        f"L1 ({self.l1_cache.latency}-cycle). "
                        f"Total stall: {latency} cycles. "
                        f"Still much faster than full DRAM "
                        f"({cfg.ram_latency_cycles} cycles)."
                    )
                )
                return val, "L3 hit → L2/L1 fill"

        # ── Full DRAM fetch (all caches missed) ───────────────────────────
        val = self.memory.read(addr)
        data = [self.memory.read(max(0, addr - addr % self.l1_cache.LINE_SIZE + i))
                for i in range(self.l1_cache.LINE_SIZE)]
        evicted = self.l1_cache.load(addr, data)
        self.l2_cache.load(addr, data)
        if self.l3_cache is not None:
            self.l3_cache.load(addr, data)

        ram_lat   = cfg.ram_latency_cycles
        total_lat = self.l1_cache.latency + ram_lat
        event     = "miss → RAM fetch"
        if evicted:
            event += " + eviction"
        dram_desc = "modern DDR5" if ram_lat >= 100 else "embedded DRAM"
        self._tick_stall(
            total_lat, phase, "RAM fetch (full cache miss)",
            detail=(
                f"Full cache miss — 0x{addr:04X} not in any cache level. "
                f"CPU stalls {ram_lat} cycles waiting for {dram_desc}. "
                f"A full cache line is loaded into every cache level for future use. "
                f"{'An L1 line was evicted (LRU) to make room. ' if evicted else ''}"
                f"Total stall: {total_lat} cycles — the most expensive memory operation."
            )
        )
        return val, event

    # ── ALU ────────────────────────────────────────────────────────────────
    """Returns how many hex digits are natural for this profile's address
    space (e.g. 2 for an 8-bit/256-byte layout, 4 for a 16-bit/64KB layout).
    Centralises a formula that used to be duplicated inline in several
    places (dma_transfer, PUSH/POP); also used by the address-width-aware
    operand fetch/display logic below."""
    def _aw(self) -> int:
        return max(2, (self.config.address_bits + 3) // 4)

    def _cache_write(self, addr: int, val: int):
        """Write-through store: update memory AND every cache level that currently
        holds this address, so subsequent cache hits return the new value.

        Without this, a STORE writes to memory but leaves stale bytes in every
        cache line that previously loaded that address, causing any cached LOAD
        to silently return the old data — a write-after-read hazard.

        Call this everywhere a value is written to data memory (STORE, PUSH,
        DMA inner loop, OUT) instead of calling self.memory.write() directly."""
        self.memory.write(addr, val)
        # Update all cache levels that hold this address (write-through policy).
        self.l1_cache.write(addr, val)
        self.l2_cache.write(addr, val)
        if self.l3_cache:
            self.l3_cache.write(addr, val)

    """Performs arithmetic or logical operations and updates status flags."""
    def _alu(self, op: str, a: int, b: int) -> int:
        r = self.registers
        r.ALU_A = a & 0xFF
        r.ALU_B = b & 0xFF
        result = {
            "ADD": a + b,
            "SUB": a - b,
            "AND": a & b,
            "OR":  a | b,
            "XOR": a ^ b,
            "CMP": a - b,
        }.get(op, a)
        
        # Calculate flags based on operation result
        r.FLAGS.zero     = (result & 0xFF) == 0
        r.FLAGS.negative = bool(result & 0x80)
        r.FLAGS.carry    = result > 0xFF or result < 0
        r.FLAGS.overflow = ((a ^ result) & (b ^ result) & 0x80) != 0
        
        r.ALU_OUT = result & 0xFF
        # Note: _tick(1) is deliberately omitted here. The caller logs the EXECUTE step
        # and then ticks — keeping the ALU result step at the same cycle as the execution.
        return r.ALU_OUT

    # ── Stack ──────────────────────────────────────────────────────────────
    """Pushes an 8-bit value onto the stack, decrementing the stack pointer.
    SP is masked to the full address width so PUSH works correctly for any
    configured address space (not just 8-bit / 256-byte layouts).
    """
    def _push(self, val: int):
        sp_mask = self.memory.size - 1   # e.g. 0xFF for 256 B, 0xFFFF for 64 KB
        aw      = max(2, (self.config.address_bits + 3) // 4)  # hex digit width
        self._cache_write(self.registers.SP, val & 0xFF)
        self.step_log.append(SimStep(
            phase="execute", cycle=self.cycle,
            description=f"PUSH 0x{val:02X} → [SP=0x{self.registers.SP:0{aw}X}]",
            detail=f"Address bus: 0x{self.registers.SP:0{aw}X}. Data bus: 0x{val:02X}. WR asserted. "
                   f"Value 0x{val:02X} written to stack address 0x{self.registers.SP:0{aw}X}. "
                   f"Each PUSH is one full memory write cycle.",
            reg_snapshot=self.registers.as_dict(),
            mem_changed=[(self.registers.SP, val & 0xFF)],
        ))
        self.registers.SP = (self.registers.SP - 1) & sp_mask
        self._tick(1)

    """Pops an 8-bit value from the stack, incrementing the stack pointer."""
    def _pop(self) -> int:
        sp_mask = self.memory.size - 1
        aw      = max(2, (self.config.address_bits + 3) // 4)
        self.registers.SP = (self.registers.SP + 1) & sp_mask
        val = self.memory.read(self.registers.SP)
        self.step_log.append(SimStep(
            phase="execute", cycle=self.cycle,
            description=f"POP [SP=0x{self.registers.SP:0{aw}X}] → 0x{val:02X}",
            detail=f"Address bus: 0x{self.registers.SP:0{aw}X}. RD asserted. "
                   f"Value 0x{val:02X} read from stack address 0x{self.registers.SP:0{aw}X}. "
                   f"Each POP is one full memory read cycle.",
            reg_snapshot=self.registers.as_dict(),
        ))
        self._tick(1)
        return val

    # ── Interrupt handling ─────────────────────────────────────────────────
    """Queues an interrupt based on its priority and hardcoded memory vector."""
    def raise_interrupt(self, itype: InterruptType, description: str = ""):
        vectors = {
            InterruptType.NMI:      0x00,
            InterruptType.IRQ0:     0x02,
            InterruptType.IRQ1:     0x04,
            InterruptType.IRQ2:     0x06,
            InterruptType.IRQ3:     0x08,
            InterruptType.SOFTWARE: 0x0A,
            InterruptType.DMA_DONE: 0x0C,
        }
        priorities = {
            InterruptType.NMI: 0,
            InterruptType.IRQ0: 1, InterruptType.IRQ1: 2,
            InterruptType.IRQ2: 3, InterruptType.IRQ3: 4,
            InterruptType.SOFTWARE: 5, InterruptType.DMA_DONE: 6,
        }
        evt = InterruptEvent(itype, vectors.get(itype, 0x00),
                             priorities.get(itype, 9),
                             description or itype.value,
                             self.cycle)
        self.irq_queue.append(evt)
        # Ensure highest priority interrupts (lowest number) are handled first
        self.irq_queue.sort(key=lambda e: e.priority)

    """Halts normal execution to service a queued interrupt."""
    def _handle_interrupt(self, evt: InterruptEvent):
        if not self.registers.FLAGS.interrupt and evt.itype != InterruptType.NMI:
            self._step("interrupt",
                       f"IRQ masked: {evt.itype.value}",
                       "Interrupt flag (I) is clear — IRQ ignored by CPU.")
            return
            
        bus = self._bus(BusState.IRQ, evt.vector, self.registers.PC, "INTA",
                        f"{evt.itype.value} acknowledged")
        self._step("interrupt",
                   f"Interrupt: {evt.itype.value}",
                   f"CPU acknowledges {evt.description}. "
                   f"INTA (Interrupt Acknowledge) asserted. Interrupts masked. "
                   f"PC=0x{self.registers.PC:04X} will be saved to the stack.",
                   bus=bus)
        self._tick(1)  # INTA cycle completes before stack saves begin
                   
        # Push current Program Counter to stack so we can return later
        self._push(self.registers.PC & 0xFF)
        self._push((self.registers.PC >> 8) & 0xFF)

        self.registers.FLAGS.interrupt = False # Mask further interrupts

        # Log the IVT lookup at the current cycle before the stall begins
        self._step("interrupt",
                   f"IVT lookup — reading handler address from 0x{evt.vector:02X}",
                   f"PC saved. CPU now reads the 16-bit handler address from the "
                   f"Interrupt Vector Table at 0x{evt.vector:02X}–0x{evt.vector+1:02X}. "
                   f"This ROM access incurs a 6-cycle latency.")

        # IVT stores 16-bit handler addresses as little-endian pairs
        lo = self.memory.read(evt.vector)
        hi = self.memory.read(evt.vector + 1)
        self.registers.PC = (hi << 8) | lo
        self._tick_stall(
            6, "interrupt", "interrupt vector fetch and context setup",
            detail=(
                f"CPU stalls while reading the IVT entry at 0x{evt.vector:02X} and loading "
                f"the handler PC. Interrupt flag cleared to prevent nested interrupts. "
                f"This overhead is why minimising interrupt latency matters in real-time systems."
            )
        )

    # ── DMA ────────────────────────────────────────────────────────────────
    """Simulate DMA block transfer, pausing CPU (bus hold)."""
    def dma_transfer(self, src: int, dst: int, length: int):
        self.dma_active = True
        # Use the full address width, not a hardcoded 8-bit mask.
        # & 0xFF was correct for the 256-byte MCU layout, but silently truncates
        # addresses like 0xA200 to 0x00 (IVT/ROM) on 64 KB layouts.
        addr_mask = self.memory.size - 1
        aw = self._aw()

        bus = self._bus(BusState.DMA, src, 0, "HOLD",
                        f"DMA: 0x{src:0{aw}X} -> 0x{dst:0{aw}X} x {length}")
        self._step("dma",
                   f"DMA Transfer: 0x{src:0{aw}X} -> 0x{dst:0{aw}X}, {length} bytes",
                   f"DMA controller asserts HOLD. CPU suspends bus access (HLDA). "
                   f"DMA autonomously transfers {length} bytes "
                   f"from 0x{src:0{aw}X} to 0x{dst:0{aw}X} "
                   f"without CPU involvement — saving {length * 4} CPU cycles.",
                   bus=bus)
        changed = []
        for i in range(length):
            s = (src + i) & addr_mask
            d = (dst + i) & addr_mask
            val = self.memory.read(s)
            self._cache_write(d, val)
            changed.append((d, val))
            self._tick_stall(
                2, "dma", f"DMA byte {i + 1}/{length}: 0x{s:0{aw}X} -> 0x{d:0{aw}X}",
                detail=(
                    f"DMA controller copies byte {i + 1} of {length}: "
                    f"value 0x{val:02X} read from 0x{s:0{aw}X}, written to 0x{d:0{aw}X}. "
                    f"CPU is in HLDA (Hold Acknowledge) — the bus is fully owned by the "
                    f"DMA controller. The CPU cannot fetch or execute instructions "
                    f"until the entire transfer completes and HOLD is released."
                )
            )
        self._step("dma",
                   "DMA Complete — bus released to CPU",
                   f"Transfer done. DMA raises DMA_DONE interrupt. CPU resumes (HLDA released).",
                   mem_changed=changed)
        # Inform the CPU the DMA transfer has finished
        self.raise_interrupt(InterruptType.DMA_DONE, "DMA block transfer complete")
        self.dma_active = False

    # ── Fetch-Decode-Execute-Writeback ─────────────────────────────────────
    """Fetch opcode at PC. Returns opcode byte."""
    def _fetch(self) -> int:
        r = self.registers
        r.MAR = r.PC
        self._current_instr_addr = r.MAR  # stable even after MAR is reused for operand fetches
        bus = self._bus(BusState.FETCH, r.MAR, 0, "RD",
                        f"Fetch opcode at 0x{r.MAR:04X}")
        self._step("fetch",
                   f"FETCH — address bus: 0x{r.MAR:04X}",
                   f"MAR ← PC (0x{r.MAR:04X}). "
                   f"Address placed on address bus. Control bus asserts RD. "
                   f"Waiting for memory response…",
                   bus=bus)

        val, cache_event = self._cache_read(r.MAR, phase="fetch")
        r.MDR = val
        r.IR  = val
        r.PC  = (r.PC + 1) & 0xFFFF
        bus2 = self._bus(BusState.FETCH, r.MAR, r.MDR, "RD",
                         f"Opcode 0x{r.MDR:02X} on data bus")
        self._step("fetch",
                   f"FETCH complete — opcode 0x{r.MDR:02X} received",
                   f"MDR ← 0x{r.MDR:02X} ({cache_event}). IR ← 0x{r.IR:02X}. PC advances to 0x{r.PC:04X}.",
                   bus=bus2, cache_event=cache_event)
        self._tick(1)
        return r.IR

    """Fetches the parameter required by the instruction currently executing.

    ADDRESS-WIDTH FIX: address-kind operands ("addr" — used by LOAD, STORE,
    JMP, JZ, JNZ, CALL) are fetched as TWO bytes, little-endian, whenever
    the profile's address space exceeds 8 bits. Previously every operand —
    including addresses — was a single byte (0-255), which meant a profile
    whose code loads above 0xFF (the 16-bit and 64-bit presets, both at
    0x4000) could never encode a jump/call back into its own program: the
    operand byte could only ever reach the first 256 bytes of memory.
    Immediate-value operands ("operand", used by ADD/SUB/AND/OR/XOR/CMP/DMA)
    and port numbers ("port", used by IN/OUT) are deliberately NOT widened —
    they represent small values, not addresses, and a 1-byte operand was
    never a bug for them. This also means 8-bit profiles (address_bits=8)
    are completely unaffected: `kind == "addr" and self.config.address_bits
    > 8` is always False there, so this path is byte-for-byte identical to
    the original single-byte implementation.
    """
    def _fetch_operand(self, kind: str = "operand") -> int:
        wide = (kind == "addr" and self.config.address_bits > 8)
        if not wide:
            return self._fetch_operand_byte()
        lo = self._fetch_operand_byte(byte_label="low byte")
        hi = self._fetch_operand_byte(byte_label="high byte")
        return lo | (hi << 8)

    """Fetches a single operand byte from [PC], advances PC, and ticks the
    clock — the original (pre-widening) _fetch_operand body, factored out
    so the dispatcher above can call it once for narrow operands or twice
    (low byte, then high byte) for a wide address operand."""
    def _fetch_operand_byte(self, byte_label: Optional[str] = None) -> int:
        r = self.registers
        suffix = f" ({byte_label})" if byte_label else ""
        r.MAR = r.PC
        bus_addr = self._bus(BusState.READ, r.MAR, 0, "RD",
                             f"Fetch operand{suffix} at 0x{r.MAR:04X}")
        self._step("fetch",
                   f"FETCH operand{suffix} — address bus: 0x{r.MAR:04X}",
                   f"MAR ← PC (0x{r.MAR:04X}). Address placed on address bus. "
                   f"Control bus asserts RD. Waiting for memory response…",
                   bus=bus_addr)
        val, cache_event = self._cache_read(r.MAR, phase="fetch")
        r.MDR = val
        r.PC  = (r.PC + 1) & 0xFFFF
        bus = self._bus(BusState.READ, r.MAR, r.MDR, "RD", f"Operand{suffix} on data bus")
        self._step("fetch",
                   f"FETCH operand{suffix} complete — 0x{r.MDR:02X} received",
                   f"MDR ← 0x{r.MDR:02X} ({cache_event}). PC advances to 0x{r.PC:04X}.",
                   bus=bus, cache_event=cache_event)
        self.cycle += 1
        self._step("fetch",
                   f"Operand{suffix} latched — ready for execute unit",
                   f"Operand byte 0x{r.MDR:02X} is now in MDR"
                   + (f" — this is the {byte_label} of a 2-byte address operand, "
                      f"combined with its other half before execute begins."
                      if byte_label else
                      " and will be forwarded to the ALU or address unit. "
                      "The fetch phase is complete."))
        self._tick(1)  # advance past latch before execute begins
        return r.MDR

    """Maps an opcode byte back to its mnemonic and parameter needs."""
    def _decode(self, opcode: int) -> tuple[str, str, list[str]]:
        mnemonic, desc, operands = INSTRUCTIONS.get(opcode, ("???", "Unknown opcode", []))
        self.registers.CIR = mnemonic
        self._step("decode",
                   f"DECODE — 0x{opcode:02X} → {mnemonic}",
                   f"Instruction decoder identifies opcode 0x{opcode:02X} as '{mnemonic}'. "
                   f"{desc}. Operands needed: {len(operands)}. "
                   f"Control unit generates microoperations.",
                   flags=True)
        self.cycle += 1
        self._step("decode",
                   f"DECODE complete — control signals asserted",
                   f"The control unit has finished decoding '{mnemonic}' and is now asserting "
                   f"the microoperation signals to the appropriate functional units. "
                   f"{'Operand fetch will follow.' if operands else 'No operand needed — proceeding straight to execute.'}")
        self._tick(1)  # advance past decode before next phase begins
        return mnemonic, desc, operands

    """Execute one full fetch-decode-execute-writeback cycle."""
    def execute_instruction(self, opcode: int):
        r = self.registers
        mnemonic, desc, operand_names = self._decode(opcode)
        operand = self._fetch_operand(operand_names[0]) if operand_names else None
        aw = self._aw()  # hex-digit width for this profile's address space — used
                          # below so LOAD/STORE/JMP/JZ/JNZ/CALL display the FULL
                          # address rather than truncating a >8-bit value to 2 digits
        result_addr = None
        result_val  = None

        if mnemonic == "LOAD":
            r.MAR = operand
            val, cache_event = self._cache_read(r.MAR, phase="execute")
            r.MDR = val
            r.ALU_A = val
            r.A = val
            bus = self._bus(BusState.READ, r.MAR, r.MDR, "RD")
            self._step("execute",
                       f"EXECUTE LOAD — A ← [0x{r.MAR:0{aw}X}] = 0x{val:02X}",
                       f"MAR = 0x{r.MAR:0{aw}X}. Memory read ({cache_event}). "
                       f"MDR = 0x{val:02X}. Accumulator A ← 0x{val:02X}.",
                       bus=bus, cache_event=cache_event, flags=True)

        elif mnemonic == "STORE":
            r.MAR = operand
            r.MDR = r.A
            self._cache_write(r.MAR, r.MDR)
            bus = self._bus(BusState.WRITE, r.MAR, r.MDR, "WR")
            self._step("execute",
                       f"EXECUTE STORE — [0x{r.MAR:0{aw}X}] ← A (0x{r.A:02X})",
                       f"MAR = 0x{r.MAR:0{aw}X}. MDR = A = 0x{r.A:02X}. "
                       f"Write cycle: data bus carries 0x{r.MDR:02X}, WR asserted.",
                       bus=bus, mem_changed=[(r.MAR, r.MDR)])
            result_addr, result_val = r.MAR, r.MDR

        elif mnemonic in ("ADD","SUB","AND","OR","XOR"):
            old_a = r.A
            r.A = self._alu(mnemonic, r.A, operand)
            self._step("execute",
                       f"EXECUTE {mnemonic} — A: 0x{old_a:02X} {mnemonic} 0x{operand:02X} = 0x{r.A:02X}",
                       f"ALU performs {mnemonic}. Inputs: A=0x{old_a:02X}, operand=0x{operand:02X}. "
                       f"Result: 0x{r.A:02X}. Flags updated: Z={r.FLAGS.zero} C={r.FLAGS.carry} N={r.FLAGS.negative}.",
                       flags=True)

        elif mnemonic == "CMP":
            # Compare doesn't store the result, it only updates flags
            self._alu("CMP", r.A, operand)
            self._step("execute",
                       f"EXECUTE CMP — A(0x{r.A:02X}) vs 0x{operand:02X}",
                       f"ALU subtracts without storing result. Only flags updated: "
                       f"Z={r.FLAGS.zero} C={r.FLAGS.carry} N={r.FLAGS.negative}.",
                       flags=True)

        elif mnemonic == "JMP":
            old_pc = r.PC
            r.PC = operand
            self._step("execute",
                       f"EXECUTE JMP → 0x{operand:0{aw}X}",
                       f"PC overwritten: 0x{old_pc:0{aw}X} → 0x{operand:0{aw}X}. "
                       f"Next fetch will come from 0x{operand:0{aw}X}.")

        elif mnemonic == "JZ":
            taken = bool(r.FLAGS.zero)
            zero_txt = "1" if taken else "0"
            if self.config.has_branch_predictor and self.config.branch_penalty_cycles > 0:
                branch_addr = self._current_instr_addr
                predicted = self._predict_branch(branch_addr)
                mispredicted = predicted != taken
                counter_before = self._branch_predictor.get(branch_addr, 1)
                self._update_branch_predictor(branch_addr, taken)
                pred_txt = "TAKEN" if predicted else "NOT-TAKEN"
                if mispredicted:
                    # Misprediction — same pipeline-flush penalty as before,
                    # but now only paid when the PREDICTOR was actually wrong,
                    # not on every taken branch. A branch seen for the first
                    # time always starts at counter=1 ("weakly not-taken"),
                    # so a loop's first iteration still mispredicts exactly
                    # like the old fixed behaviour — the difference shows up
                    # from the SECOND iteration onward, once the counter has
                    # adapted to the branch's real pattern.
                    counter_after = self._branch_predictor[branch_addr]
                    next_pred = "TAKEN" if counter_after >= 2 else "NOT-TAKEN"
                    self._tick_stall(
                        self.config.branch_penalty_cycles, "execute",
                        "branch mis-predict pipeline flush",
                        detail=(
                            f"Branch {'TAKEN' if taken else 'NOT TAKEN'} (Z={zero_txt}), but the "
                            f"predictor — a 2-bit saturating counter at counter={counter_before} for this branch "
                            f"site — predicted {pred_txt}. Misprediction: the "
                            f"{self.config.pipeline_stages}-stage pipeline must be squashed and refilled. "
                            f"Penalty: {self.config.branch_penalty_cycles} cycles. The counter now reads "
                            f"{counter_after}, so next time this exact branch executes it will predict {next_pred}."
                        )
                    )
                if taken:
                    r.PC = operand
                if mispredicted:
                    detail = (f"{'Zero flag is set' if taken else 'Zero flag is clear'}. "
                              f"{'PC ← 0x' + format(operand, f'0{aw}X') if taken else f'PC stays at 0x{r.PC:0{aw}X}'}. "
                              f"Predictor said {pred_txt} — ✗ mispredicted (pipeline flush above).")
                else:
                    detail = (f"{'Zero flag is set' if taken else 'Zero flag is clear'}. "
                              f"{'PC ← 0x' + format(operand, f'0{aw}X') if taken else f'PC stays at 0x{r.PC:0{aw}X}'}. "
                              f"Predictor said {pred_txt} — ✓ predicted correctly, no flush needed.")
                self._step("execute", f"EXECUTE JZ → {'TAKEN' if taken else 'NOT TAKEN'} (Z={zero_txt})", detail)
            else:
                # No predictor on this profile — original unconditional behaviour.
                if taken:
                    r.PC = operand
                    detail = f"Zero flag is set. PC ← 0x{operand:0{aw}X}."
                else:
                    detail = f"Zero flag is clear. PC stays at 0x{r.PC:0{aw}X}."
                self._step("execute", f"EXECUTE JZ → {'TAKEN' if taken else 'NOT TAKEN'} (Z={zero_txt})", detail)

        elif mnemonic == "JNZ":
            taken = not r.FLAGS.zero
            zero_txt = "0" if taken else "1"
            if self.config.has_branch_predictor and self.config.branch_penalty_cycles > 0:
                branch_addr = self._current_instr_addr
                predicted = self._predict_branch(branch_addr)
                mispredicted = predicted != taken
                counter_before = self._branch_predictor.get(branch_addr, 1)
                self._update_branch_predictor(branch_addr, taken)
                pred_txt = "TAKEN" if predicted else "NOT-TAKEN"
                if mispredicted:
                    counter_after = self._branch_predictor[branch_addr]
                    next_pred = "TAKEN" if counter_after >= 2 else "NOT-TAKEN"
                    self._tick_stall(
                        self.config.branch_penalty_cycles, "execute",
                        "branch mis-predict pipeline flush",
                        detail=(
                            f"Branch {'TAKEN' if taken else 'NOT TAKEN'} (Z={zero_txt}), but the "
                            f"predictor — a 2-bit saturating counter at counter={counter_before} for this branch "
                            f"site — predicted {pred_txt}. Misprediction: the "
                            f"{self.config.pipeline_stages}-stage pipeline must be squashed and refilled. "
                            f"Penalty: {self.config.branch_penalty_cycles} cycles. The counter now reads "
                            f"{counter_after}, so next time this exact branch executes it will predict {next_pred}."
                        )
                    )
                if taken:
                    r.PC = operand
                if mispredicted:
                    detail = (f"{'Zero flag is clear' if taken else 'Zero flag is set'}. "
                              f"{'PC ← 0x' + format(operand, f'0{aw}X') if taken else f'PC stays at 0x{r.PC:0{aw}X}'}. "
                              f"Predictor said {pred_txt} — ✗ mispredicted (pipeline flush above).")
                else:
                    detail = (f"{'Zero flag is clear' if taken else 'Zero flag is set'}. "
                              f"{'PC ← 0x' + format(operand, f'0{aw}X') if taken else f'PC stays at 0x{r.PC:0{aw}X}'}. "
                              f"Predictor said {pred_txt} — ✓ predicted correctly, no flush needed.")
                self._step("execute", f"EXECUTE JNZ → {'TAKEN' if taken else 'NOT TAKEN'} (Z={zero_txt})", detail)
            else:
                # No predictor on this profile — original unconditional behaviour.
                if taken:
                    r.PC = operand
                    detail = f"Zero flag is clear. PC ← 0x{operand:0{aw}X}."
                else:
                    detail = f"Zero flag is set. PC stays at 0x{r.PC:0{aw}X}."
                self._step("execute", f"EXECUTE JNZ → {'TAKEN' if taken else 'NOT TAKEN'} (Z={zero_txt})", detail)

        elif mnemonic == "PUSH":
            self._push(r.A)
            self._step("execute", f"EXECUTE PUSH A (0x{r.A:02X})",
                       f"A pushed to stack. SP was 0x{(r.SP+1)&(self.memory.size-1):0{aw}X}, now 0x{r.SP:0{aw}X}.")

        elif mnemonic == "POP":
            r.A = self._pop()
            self._step("execute", f"EXECUTE POP → A = 0x{r.A:02X}",
                       f"Popped 0x{r.A:02X} from stack. SP was 0x{(r.SP-1)&(self.memory.size-1):0{aw}X}, now 0x{r.SP:0{aw}X}.")

        elif mnemonic == "CALL":
            self._push(r.PC & 0xFF)
            self._push((r.PC >> 8) & 0xFF)
            r.PC = operand
            self._step("execute", f"EXECUTE CALL 0x{operand:0{aw}X}",
                       f"Return address pushed to stack. PC ← 0x{operand:0{aw}X}.")

        elif mnemonic == "RET":
            hi = self._pop()
            lo = self._pop()
            r.PC = ((hi << 8) | lo) & 0xFFFF
            self._step("execute", f"EXECUTE RET → PC = 0x{r.PC:04X}",
                       f"Return address popped. PC ← 0x{r.PC:04X}.")

        elif mnemonic == "IN":
            port = operand
            io_addr = self.memory.io_input_base + (port & 0x07)
            val = self.memory.read(io_addr)
            r.A = val
            bus = self._bus(BusState.IO_READ, port, val, "IOR",
                            f"Read I/O port {port}")
            self._step("execute", f"EXECUTE IN port {port} → A = 0x{val:02X}",
                       f"I/O read cycle. Port {port} mapped to Input Register 0x{io_addr:02X}. "
                       f"Data bus: 0x{val:02X}. A ← 0x{val:02X}.", bus=bus)

        elif mnemonic == "OUT":
            port = operand
            io_addr = self.memory.io_output_base + (port & 0x07)
            self._cache_write(io_addr, r.A)
            bus = self._bus(BusState.IO_WRITE, port, r.A, "IOW",
                            f"Write I/O port {port}")
            self._step("execute", f"EXECUTE OUT A(0x{r.A:02X}) → port {port}",
                       f"I/O write cycle. Port {port} mapped to Output Register 0x{io_addr:02X}. "
                       f"A = 0x{r.A:02X} placed on data bus. IOW asserted.",
                       bus=bus, mem_changed=[(io_addr, r.A)])

        elif mnemonic == "DMA":
            # Triggers a transfer using parameters the program has already
            # written into the DMA control block — a small 6-byte header
            # (SRC_LO, SRC_HI, DST_LO, DST_HI, LEN_LO, LEN_HI) carved out
            # of the start of the "DMA Buffer" region. This mirrors how
            # real DMA controllers work: the CPU configures memory-mapped
            # control registers via ordinary STORE instructions, then
            # writes to a "go" register/instruction to hand the bus over.
            # Length is clamped to keep the demo's cycle cost bounded —
            # dma_transfer() ticks 2 cycles per byte copied.
            dma_region = self.memory._region("DMA Buffer")
            ctrl_base = dma_region.start if dma_region else self.memory.io_output_base
            src    = self.memory.read(ctrl_base)     | (self.memory.read(ctrl_base + 1) << 8)
            dst    = self.memory.read(ctrl_base + 2) | (self.memory.read(ctrl_base + 3) << 8)
            length = self.memory.read(ctrl_base + 4) | (self.memory.read(ctrl_base + 5) << 8)
            length = max(1, min(length, 32))
            aw = self._aw()
            self._step("execute",
                       f"EXECUTE DMA — triggering transfer via control block at 0x{ctrl_base:0{aw}X}",
                       f"Reads SRC/DST/LEN from the DMA control block "
                       f"(0x{ctrl_base:0{aw}X}-0x{ctrl_base+5:0{aw}X}): "
                       f"src=0x{src:0{aw}X}, dst=0x{dst:0{aw}X}, length={length}. "
                       f"Hands off to the DMA controller, which now owns the bus.")
            self.dma_transfer(src, dst, length)

        elif mnemonic == "LDI":
            old_a = r.A
            r.A = operand & 0xFF
            self._step("execute", f"EXECUTE LDI — A ← 0x{r.A:02X}",
                       f"Immediate value 0x{operand:02X} loaded directly into A "
                       f"(was 0x{old_a:02X}). Unlike LOAD, no memory access occurs here — "
                       f"the value comes straight from the instruction stream itself.",
                       flags=True)

        elif mnemonic == "ADDM":
            # Memory-add: A = A + mem[addr]. Goes through the full cache
            # hierarchy (same path as LOAD), so miss stalls are simulated
            # correctly and show up in the Heat/Cache tabs.
            r.MAR = operand
            self._bus(BusState.READ, r.MAR, 0, "RD",
                      f"ADDM memory read: MAR=0x{r.MAR:0{aw}X} → cache")
            val, cache_event = self._cache_read(r.MAR)
            r.MDR = val
            old_a = r.A
            result = self._alu("ADD", r.A, val)
            r.A = result
            self._step("execute",
                       f"EXECUTE ADDM — A: 0x{old_a:02X} + [0x{operand:0{aw}X}]=0x{val:02X} = 0x{result:02X}",
                       f"Loads 0x{val:02X} from address 0x{operand:0{aw}X} (via cache), "
                       f"then adds it to A (0x{old_a:02X}). Result: 0x{result:02X}. "
                       f"Equivalent to LOAD followed by ADD, but in a single instruction — "
                       f"essential for loops that accumulate a value held in memory.",
                       flags=True, cache_event=cache_event)

        elif mnemonic == "SUBM":
            # Memory-subtract via 2's complement: A = A - mem[addr].
            r.MAR = operand
            self._bus(BusState.READ, r.MAR, 0, "RD",
                      f"SUBM memory read: MAR=0x{r.MAR:0{aw}X} → cache")
            val, cache_event = self._cache_read(r.MAR)
            r.MDR = val
            old_a = r.A
            result = self._alu("SUB", r.A, val)
            r.A = result
            self._step("execute",
                       f"EXECUTE SUBM — A: 0x{old_a:02X} - [0x{operand:0{aw}X}]=0x{val:02X} = 0x{result:02X}",
                       f"Loads 0x{val:02X} from address 0x{operand:0{aw}X} (via cache), "
                       f"then subtracts it from A (0x{old_a:02X}) using 2's complement. "
                       f"Result: 0x{result:02X}.",
                       flags=True, cache_event=cache_event)

        elif mnemonic == "NOP":
            self._step("execute", "EXECUTE NOP", "No operation performed. PC already advanced.")

        elif mnemonic == "HLT":
            self.halted = True
            self.stop_reason = "halted"
            self.stop_detail = "Program executed HLT normally."
            self._step("execute", "EXECUTE HLT — CPU halted",
                       "Program terminated normally by executing HLT. "
                       "A new run or RESET is required to begin again.")
            return

        # Writeback phase — always on a distinct cycle from execute
        self._tick(1)
        self._step("writeback",
                   f"WRITEBACK — {mnemonic} complete",
                   f"Results committed. Any pending register writes finalised. "
                   f"Interrupt check: {len(self.irq_queue)} pending. "
                   f"Next fetch at PC = 0x{r.PC:04X}.")
        self._tick(1)

        # Service pending interrupts
        if self.irq_queue:
            evt = self.irq_queue.pop(0)
            self._handle_interrupt(evt)

    """Load and run a program, returning a full, self-describing step log.

    ``stop_reason`` and ``stop_detail`` are always populated before return so
    callers can tell whether execution halted normally, hit its cycle budget,
    encountered an invalid opcode, or stopped on an expected runtime fault.
    """
    def run_program(self, program: list[int], load_addr: int = None,
                    timer_irq: bool = False, max_cycles: int = None) -> list[SimStep]:
        if load_addr is None:
            load_addr = self.memory.code_base

        self.step_log = []
        self.bus_log = []
        self.halted = False
        self.stop_reason = "running"
        self.stop_detail = "Simulation is running."
        self.instructions_executed = 0
        self.timer_irq_enabled = bool(timer_irq)
        _max_cycles = max_cycles if max_cycles is not None else self.config.max_sim_cycles
        self.max_cycles_used = int(_max_cycles)

        for i, byte in enumerate(program):
            if load_addr + i < self.memory.size:
                self.memory.write(load_addr + i, byte)
        self.registers.PC = load_addr

        self._step(
            "fetch",
            "CPU RESET — starting execution",
            f"Program loaded at 0x{load_addr:04X}. "
            f"PC = 0x{load_addr:04X}. SP = 0x{self.registers.SP:04X}. "
            f"Timer IRQ injection: {'enabled' if timer_irq else 'disabled'}. "
            f"Cycle budget: {self.max_cycles_used:,}. Profile: {self.config.name}."
        )

        while not self.halted and self.cycle < self.max_cycles_used:
            try:
                # Inject one teaching/demo timer interrupt after two instructions,
                # but only when the user has explicitly enabled the control.
                if timer_irq and self.instructions_executed == 2:
                    self.raise_interrupt(InterruptType.IRQ0, "Injected teaching timer tick")

                if (self.instructions_executed > 0 and self.irq_queue
                        and self.registers.FLAGS.interrupt):
                    evt = self.irq_queue.pop(0)
                    self._handle_interrupt(evt)
                    self.registers.FLAGS.interrupt = True

                opcode_addr = self.registers.PC
                opcode = self._fetch()
                if opcode not in INSTRUCTIONS:
                    self.stop_reason = "invalid_opcode"
                    self.stop_detail = (
                        f"Unknown opcode 0x{opcode:02X} fetched from "
                        f"0x{opcode_addr:0{self._aw()}X}."
                    )
                    self._step(
                        "fault",
                        f"INVALID OPCODE — 0x{opcode:02X}",
                        self.stop_detail + " Execution stopped before decode."
                    )
                    break

                self.execute_instruction(opcode)
                self.instructions_executed += 1

            except (IndexError, ValueError, PermissionError) as exc:
                self.stop_reason = "runtime_fault"
                self.stop_detail = f"{type(exc).__name__}: {exc}"
                self._step(
                    "fault",
                    f"RUNTIME FAULT — {type(exc).__name__}",
                    self.stop_detail
                )
                break

        if self.stop_reason == "running":
            if self.halted:
                self.stop_reason = "halted"
                self.stop_detail = "Program executed HLT normally."
            elif self.cycle >= self.max_cycles_used:
                self.stop_reason = "cycle_limit"
                self.stop_detail = (
                    f"Execution reached the {self.max_cycles_used:,}-cycle safety limit "
                    "before HLT. Increase Sim Length or inspect the program for a loop."
                )
                self._step("limit", "CYCLE LIMIT REACHED", self.stop_detail)
            else:
                self.stop_reason = "stopped"
                self.stop_detail = "Simulation stopped without executing HLT."

        return self.step_log


# ── Convenience: process user input ───────────────────────────────────────────
"""
    Build a small program based on user input, run it, return (cpu, steps).
    mode:   'key' | 'int' | 'str'
    config: optional CPUConfig (defaults to microcontroller preset)

    Example:
        cpu, steps = process_input("42", "int", CPUConfig.modern_desktop())

    NOTE — addressing in the 1-byte ISA:
    The ISA encodes addresses as single bytes (0x00–0xFF), so programs can
    only directly address the first 256 bytes of memory.  For the 8-bit MCU
    layout those bytes include real data regions (stack @ 0x20, data @ 0xA0).
    For 64 KB layouts we pre-copy I/O values to zero-page RAM (0x10–0xFF) so
    the same 1-byte programs work unchanged — exactly the "zero-page" trick
    used by the 6502, Z80, and MSP430.
    """
def process_input(raw: str, mode: str,
                  config: Optional[CPUConfig] = None,
                  timer_irq: bool = False,
                  max_cycles: Optional[int] = None) -> tuple[CPU, list[SimStep]]:
    cpu = CPU(config) 

    io_in  = cpu.memory.io_input_base
    io_out = cpu.memory.io_output_base
    data   = cpu.memory.data_base

    # ── Operand-width helper ───────────────────────────────────────────────
    # Must mirror CPU._fetch_operand()'s encoding exactly: "addr"-kind
    # operands (used here by LOAD=0x01 and STORE=0x02) are 2 bytes
    # little-endian whenever address_bits > 8, everything else stays 1
    # byte. Getting this out of sync with the engine desyncs the entire
    # rest of the byte stream — the next "instruction" the CPU reads would
    # actually be the tail of the previous operand.
    wide_addr = cpu.config.address_bits > 8
    def addr_bytes(addr_val):
        if wide_addr:
            return [addr_val & 0xFF, (addr_val >> 8) & 0xFF]
        return [addr_val & 0xFF]

    # ── Choose 1-byte-addressable scratch locations ───────────────────────────
    # For the MCU (256-byte) layout the I/O and data regions are already in the
    # 0x00–0xFF window, so we use them directly.
    # For larger layouts we mirror I/O values into zero-page RAM (0x10–0xFF)
    # and store results there — the same "zero-page" idiom used by many real CPUs.
    # (These addresses are deliberately small — 0x10-0xFF — so even when
    # addr_bytes() encodes them as a wide 2-byte operand, the high byte is
    # always 0x00; the zero-page idiom is about choosing small VALUES, which
    # is orthogonal to how many bytes the operand gets encoded in.)
    if cpu.memory.size <= 256:
        # Native layout: low-byte of each region address is the real address
        zp_io   = io_in  & 0xFF   # e.g. 0xD0 for MCU
        zp_data = data   & 0xFF   # e.g. 0xA0 for MCU
    else:
        # Large address space: use zero-page RAM (writable 0x10–0xFF region)
        zp_io   = 0x10   # zero-page mirror of I/O port 0 value
        zp_data = 0x20   # zero-page scratch data area

    if mode == "key":
        char = raw[0] if raw else 'A'
        val  = ord(char) & 0xFF
        prog = ([0x10, 0x00]                          # IN  port 0
                + [0x02] + addr_bytes(zp_data)         # STORE [zp_data]
                + [0x03, 0x01]                         # ADD 1
                + [0x02] + addr_bytes((zp_data + 1) & 0xFF)  # STORE [zp_data+1] (val+1)
                + [0x11, 0x00]                         # OUT  port 0
                + [0xFF])                              # HLT
        # Pre-load the I/O register — both native address and zero-page mirror
        if cpu.memory.size <= 256:
            cpu.memory.write(io_in, val)
        else:
            cpu.memory.write(io_in,  val)     # real I/O register (for completeness)
            cpu.memory.write(zp_io,  val)     # zero-page mirror (what IN reads via port 0)
        cpu.raise_interrupt(InterruptType.IRQ1, f"Keyboard: '{char}' (ASCII {val})")

    elif mode == "int":
        n   = max(0, min(255, int(raw) if raw.lstrip('-').isdigit() else 0))
        val = n & 0xFF

        # Demonstrates 8-bit overflow detection: ADD 1 wraps 255 -> 0, and
        # CMP 0 + JZ catches exactly that case. This is a genuinely useful
        # CPU concept (not just branching for branching's sake) — and the
        # ISA only has zero-flag-based jumps (JZ/JNZ; no "less/greater
        # than"), so testing "did the result wrap to exactly zero" is the
        # natural conditional to reach for here.
        #
        # Built incrementally with a running address counter (rather than
        # one large literal byte list) because JZ/JMP's "addr" operand can
        # be 1 or 2 bytes depending on profile (see addr_bytes() above) —
        # the jump targets below are patched in once their real addresses
        # are known, the same two-pass approach a real assembler uses.
        addr = cpu.memory.code_base
        prog = []
        def emit(byte_list):
            nonlocal addr
            prog.extend(byte_list)
            addr += len(byte_list)

        emit([0x01] + addr_bytes(zp_io))                   # LOAD  [zp_io]
        emit([0x02] + addr_bytes(zp_data))                 # STORE [zp_data]
        emit([0x03, 0x01])                                 # ADD 1   (wraps 255 -> 0)
        emit([0x02] + addr_bytes((zp_data + 1) & 0xFF))    # STORE [zp_data+1]
        emit([0x08, 0x00])                                 # CMP 0   — did we just wrap around?
        wrapped_jz_pos = len(prog)
        emit([0x0A] + addr_bytes(0))                       # JZ  wrapped   (target patched below)
        emit([0x11, 0x06])                                 # OUT port 6     — normal path
        done_jmp_pos = len(prog)
        emit([0x09] + addr_bytes(0))                       # JMP done      (target patched below)
        wrapped_addr = addr
        emit([0x11, 0x07])                                 # wrapped: OUT port 7 — overflow indicator
        done_addr = addr
        emit([0xFF])                                       # done: HLT

        def patch(jump_instr_pos, target_addr):
            target_bytes = addr_bytes(target_addr)
            for i, b in enumerate(target_bytes):
                prog[jump_instr_pos + 1 + i] = b
        patch(wrapped_jz_pos, wrapped_addr)
        patch(done_jmp_pos, done_addr)

        # Populate both the real I/O address and the zero-page mirror
        if cpu.memory.size <= 256:
            cpu.memory.write(io_in, val)
        else:
            cpu.memory.write(io_in, val)
            cpu.memory.write(zp_io, val)

    else:  # str
        chars = [ord(c) & 0xFF for c in raw[:16]]
        prog  = []
        for i, c in enumerate(chars):
            src = (zp_io   + i) & 0xFF
            dst = (zp_data + i) & 0xFF
            prog += ([0x01] + addr_bytes(src)    # LOAD  [zp_io+i]
                     + [0x02] + addr_bytes(dst))  # STORE [zp_data+i]
            if cpu.memory.size <= 256:
                cpu.memory.write(io_in + i, c)
            else:
                cpu.memory.write(io_in + i, c)
                cpu.memory.write(zp_io  + i, c)
        prog += [0x11, 0x00, 0xFF]             # OUT port 0, HLT

    steps = cpu.run_program(prog, timer_irq=timer_irq, max_cycles=max_cycles)
    return cpu, steps