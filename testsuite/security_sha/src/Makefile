# ************ CCC - passing optimization flags ************ 
ifeq ($(CCC_OPTS),)
 CCC_OPTS =  
endif
ifeq ($(CCC_OPTS_ADD),)
  CCC_OPTS_ADD =  -assumption-cache-tracker -instcombine -cg-profile -inline -mem2reg -scoped-noalias-aa -lower-expect -licm -lcssa -gvn -prune-eh -sroa -loop-unroll -early-cse-memssa -tbaa -targetlibinfo -verify -early-cse -profile-summary-info -annotation2metadata -ipsccp -globalopt -deadargelim -loops -lazy-branch-prob -lazy-block-freq -globals-aa -openmp-opt-cgscc -function-attrs -argpromotion -memoryssa -speculative-execution -correlated-propagation -postdomtree -branch-prob -block-freq -pgo-memop-opt -tailcallelim -reassociate -lcssa-verification -scalar-evolution -loop-rotate -indvars -loop-deletion -sccp -demanded-bits -barrier -rpo-function-attrs -globaldce -lower-constant-intrinsics -loop-accesses -vector-combine -transform-warning -alignment-from-assumptions -strip-dead-prototypes -constmerge -div-rem-pairs -annotation-remarks  
endif

# Declarations
ifeq ($(ZCC),) 
 ZCC = clang 
endif
ifeq ($(CC_OPTS),)
 CC_OPTS = -emit-llvm -Xclang -disable-O0-optnone  
endif
ifeq ($(LDCC),)
 LDCC = llvm-link 
endif
ifeq ($(LD_OPTS),)
 LD_OPTS = -f -o a.out
endif

# Actions
all:		
	@ $(ZCC) $(CCC_OPTS) $(CC_OPTS) -Wl -w -c *.c
	@ $(LDCC) *.bc -o tmp.bc
	@ opt $(CCC_OPTS_ADD) tmp.bc -o tmp.bc -enable-new-pm=0
	@ $(LDCC) tmp.bc $(LD_OPTS) 
clean:
	@ rm -f a.out *.o *.a *.s *.i *.I *.bc *.ll
