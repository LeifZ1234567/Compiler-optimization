import opentuner
from opentuner import ConfigurationManipulator
from opentuner import MeasurementInterface
from opentuner import Result
from opentuner.search.manipulator import BooleanParameter
import argparse
import logging
import os
import time
from util import Util # 假设 Util 类在 util.py 中

# 全局日志记录器
log = logging.getLogger(__name__)

# 全局变量：已知序列 (与您的 EnhancedHybridGA 使用的 known_sequences 一致)
# 请确保这些标志是您的 util.py 和 Makefile 能直接处理的格式
KNOWN_SEQUENCES = [
    ['-sroa', '-jump-threading'],
    ['-mem2reg', '-gvn', '-instcombine'],
    ['-mem2reg', '-gvn', '-prune-eh'],
    ['-mem2reg', '-gvn', '-dse'],
    ['-mem2reg', '-loop-sink', '-loop-distribute'],
    ['-early-cse-memssa', '-instcombine'],
    ['-early-cse-memssa', '-dse'],
    ['-lcssa', '-loop-unroll'],
    ['-licm', '-gvn', '-instcombine'],
    ['-licm', '-gvn', '-prune-eh'],
    ['-licm', '-gvn', '-dse'],
    ['-memcpyopt', '-loop-distribute']
]
# (如果您的 EnhancedHybridGA 使用了其他的 known_sequences 列表，请替换成那个)


# ... (导入和全局 KNOWN_SEQUENCES 定义保持不变) ...

class GAInspiredHybridInterface(MeasurementInterface): # <--- 类名已修改为英文
    def __init__(self, args):
        super(GAInspiredHybridInterface, self).__init__(args) # <--- 确保 super() 调用也使用新类名
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.suite_name = args.program
        self.util = Util()

        # ... (init 方法的其余部分，确保没有中文) ...
        # 1. 加载已知序列
        self.known_sequences_list = KNOWN_SEQUENCES
        self.num_known_sequences = len(self.known_sequences_list)
        self.log.info(f"Loaded {self.num_known_sequences} known sequences.")

        # 2. 获取程序类别并加载类别特定的基础标志
        self.program_cluster_index = self.util.get_cluster_index(self.suite_name)
        self.log.info(f"Program '{self.suite_name}' belongs to cluster {self.program_cluster_index}.")
        self.base_flags = self.util.gain_flags_cluster(self.program_cluster_index)
        if not self.base_flags:
            self.log.warning(f"No base flags loaded for cluster {self.program_cluster_index}. This might be an issue.")
        self.log.info(f"Loaded {len(self.base_flags)} base flags for cluster {self.program_cluster_index}.")

        # 3. 创建参数名映射
        self.param_configs = []
        for i in range(self.num_known_sequences):
            param_name = f"enable_seq_{i}"
            self.param_configs.append({'type': 'seq', 'id': i, 'name': param_name})
        for i, flag_str in enumerate(self.base_flags):
            param_name = f"enable_base_flag_{i}_{flag_str.replace('-', '_').replace('=', '_')[:20]}"
            self.param_configs.append({'type': 'base', 'id': i, 'flag': flag_str, 'name': param_name})

        # 4. 测量基准性能
        self.baseline_time = self.measure_baseline()
        if self.baseline_time == float('inf'):
            self.log.error(f"Failed to establish a valid baseline for {self.suite_name}. Tuning results might be misleading.")
    # ... (measure_baseline, manipulator, decode_opentuner_cfg_to_flags, run, save_final_config 方法保持不变，确保内部无中文) ...

    def measure_baseline(self):
        self.log.info(f"Measuring baseline for {self.suite_name} using -O3...")
        try:
            path = self.util.testsuite_path(self.suite_name)
            option_str = self.util.testsuite_option(self.suite_name)
            self.util.update_makefile(path, option_str, "-O3 ", "./data/Makefile.llvm")

            run_times = []
            for _ in range(3):
                runtime = self.util.get_runtime(path, self.suite_name)
                if runtime is not None and runtime > 0:
                    run_times.append(runtime)
                else:
                    self.log.warning(f"Invalid runtime ({runtime}) during baseline for {self.suite_name}")
            if not run_times:
                self.log.error(f"All baseline runs failed for {self.suite_name}.")
                return float('inf')
            baseline = sorted(run_times)[len(run_times) // 2]
            self.log.info(f"Baseline -O3 time for {self.suite_name}: {baseline:.4f}s")
            return baseline
        except Exception as e:
            self.log.error(f"Exception during baseline for {self.suite_name}: {e}", exc_info=True)
            return float('inf')

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for p_config in self.param_configs:
            manipulator.add_parameter(BooleanParameter(p_config['name'], default=False))
        return manipulator

    def decode_opentuner_cfg_to_flags(self, configuration_data):
        seq_options = []
        base_options_from_cfg = []

        for p_config in self.param_configs:
            param_name = p_config['name']
            is_active = configuration_data.get(param_name, False)
            if is_active:
                if p_config['type'] == 'seq':
                    sequence_index = p_config['id']
                    seq_options.extend(self.known_sequences_list[sequence_index])
                elif p_config['type'] == 'base':
                    base_options_from_cfg.append(p_config['flag'])
        
        combined = seq_options + base_options_from_cfg
        seen = {}
        for opt in reversed(combined):
            key = opt.split('=')[0]
            if key not in seen:
                seen[key] = opt
        final_flags_list = list(reversed(seen.values()))
        if not final_flags_list:
            self.log.debug(f"Config for {self.suite_name} (data: {configuration_data}) decoded to no flags.")
        return final_flags_list

    def run(self, desired_result, input, limit):
        cfg_data = desired_result.configuration.data
        decoded_flags_list = self.decode_opentuner_cfg_to_flags(cfg_data)
        final_flags_str = " ".join(decoded_flags_list)

        if not final_flags_str.strip():
            self.log.warning(f"No flags generated for {self.suite_name}. CFG: {cfg_data}. Skipping.")
            return Result(time=float('inf'))

        log_flags_display = final_flags_str if len(final_flags_str) < 200 else final_flags_str[:197] + "..."
        self.log.debug(f"Running {self.suite_name} with flags: {log_flags_display}")

        try:
            path = self.util.testsuite_path(self.suite_name)
            option_str = self.util.testsuite_option(self.suite_name)
            self.util.update_makefile(path, option_str, final_flags_str + " ", "./data/Makefile.llvm")

            run_times = []
            for i in range(3):
                runtime = self.util.get_runtime(path, self.suite_name)
                if runtime is not None and runtime > 0:
                    run_times.append(runtime)
                else:
                    self.log.warning(f"Run {i+1}/3 for {self.suite_name} with flags '{log_flags_display}' invalid time: {runtime}")
            if not run_times:
                self.log.warning(f"All runs failed for {self.suite_name} with flags: {log_flags_display}")
                return Result(time=float('inf'))
            median_time = sorted(run_times)[len(run_times) // 2]
            speedup = 0.0
            if self.baseline_time != float('inf') and self.baseline_time > 0 and median_time > 0:
                speedup = self.baseline_time / median_time
            self.log.info(f"Prog: {self.suite_name} | Time: {median_time:.4f}s | Speedup: {speedup:.3f}x | Flags: {log_flags_display}")
            return Result(time=median_time)
        except Exception as e:
            self.log.error(f"Exception for {self.suite_name} flags '{log_flags_display}': {e}", exc_info=True)
            return Result(time=float('inf'))

    def save_final_config(self, configuration):
        cfg_data = configuration.data
        best_flags_list = self.decode_opentuner_cfg_to_flags(cfg_data)
        best_flags_str = " ".join(best_flags_list)

        self.log.info("\n" + "="*30)
        self.log.info("Best Configuration Found by OpenTuner (GAInspiredHybridInterface)")
        self.log.info(f"Program: {self.suite_name} (Cluster: {self.program_cluster_index})")
        if best_flags_str.strip():
            self.log.info(f"Flags: {best_flags_str}")
            try:
                path = self.util.testsuite_path(self.suite_name)
                option_str = self.util.testsuite_option(self.suite_name)
                self.util.update_makefile(path, option_str, best_flags_str + " ", "./data/Makefile.llvm")
                final_runtimes = []
                for _ in range(5):
                    rt = self.util.get_runtime(path, self.suite_name)
                    if rt is not None and rt > 0: final_runtimes.append(rt)
                if final_runtimes:
                    final_time = sorted(final_runtimes)[len(final_runtimes)//2]
                    final_speedup = (self.baseline_time / final_time) if self.baseline_time > 0 and final_time > 0 else 0.0
                    self.log.info(f"Final Measured Time (median of 5): {final_time:.4f}s")
                    self.log.info(f"Final Speedup over -O3: {final_speedup:.3f}x")
                else: self.log.warning(f"Failed to re-measure final config for {self.suite_name}")
            except Exception as e: self.log.error(f"Error re-measuring final config: {e}", exc_info=True)
        else:
            self.log.info("Flags: (No flags active in the best configuration)")
        self.log.info("="*30 + "\n")


if __name__ == '__main__':
    # 1. 获取 OpenTuner 的默认参数解析器
    #    它包含了 --stop-after, --parallelism, --test-limit, -d (debug) 等标准选项
    ot_parser = opentuner.default_argparser()

    # 2. 创建你自己的解析器，以 OpenTuner 解析器为父解析器
    #    并设置 add_help=False 避免 -h/--help 冲突
    parser = argparse.ArgumentParser(
        parents=[ot_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Tune compilation flags with GA-like hybrid encoding using OpenTuner.",
        add_help=False
    )

    # 3. 添加你自己的特定参数
    parser.add_argument('--program',
                        required=True,
                        help='Program/benchmark to tune (e.g., automotive_susan_c from cBench)')

    args = parser.parse_args()

    # 4. 配置日志记录
    #    直接设置为 INFO 级别，与您原始代码行为一致。
    #    如果用户在命令行使用了 OpenTuner 的调试参数 (如 -d),
    #    OpenTuner 内部的日志系统仍然会响应这些参数并可能输出更详细的日志。
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # 获取当前模块的 logger，在 INFO 级别下，log.debug() 将不会显示
    log = logging.getLogger(__name__)
    log.info(f"Parsed arguments: {args}") # 这条会显示
    log.debug(f"This is a debug message, will not show if level is INFO.") # 这条不会显示

    if not KNOWN_SEQUENCES:
        log.warning("Global KNOWN_SEQUENCES list is empty. The 'sequence' part of hybrid encoding will be ineffective.")

    # 调用 OpenTuner 的主运行函数 (请确保类名正确)
    GAInspiredHybridInterface.main(args)