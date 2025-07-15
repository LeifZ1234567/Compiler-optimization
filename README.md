# Compiler Auto-Tuning via Sequence Generation and Flag and Selection


## Overview
This framework implements a Compiler Auto-Tuning via Sequence Generation and Flag Selection. It supports:
- 12+ evolutionary algorithms (BA, GA, DE, PSO, etc.)
- Hybrid algorithm combinations (EnhancedHybridBA, EnhancedHybridDE, etc.)
- Some SOTA comparisons, such as BOCA, Opentuner, RIO
- Multi-objective optimization (performance + energy efficiency)
- Automated evaluation of GCC/LLVM compiler flag combinations

## Key Features
- ​**Evolutionary Algorithms**:
  - Classic algorithms: BA, CS, DE, EDA, FA, FPA, GA, GWO, HHO, JAYA, PSO, SCA, SSA, WOA
  - Enhanced hybrids: EnhancedHybrid[Algorithm] variants combined Sequences and Flags

- ​**Compiler Optimization**:
  - Automatic flag sequence generation
  - Performance benchmarking
  - Energy consumption measurement (via pyRAPL)
  - Cross-platform support (tested on x86 and ARM)

- ​**Analysis Tools**:
  - Flag effectiveness analysis (`calculate_fi()`)
  - Interaction matrix generation (`graph_matrix()`)
  - Automated result logging (JSON/Excel)

# Clone repository
git clone [repository_url]
cd compiler-autotuning

# Install dependencies
pip install -r requirements.txt
