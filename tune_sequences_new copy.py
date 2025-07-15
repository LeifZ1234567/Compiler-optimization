import opentuner
from opentuner import ConfigurationManipulator
from opentuner import MeasurementInterface
from opentuner import Result
from opentuner.search.manipulator import BooleanParameter
import argparse
import logging
import os
import time
from util import Util

log = logging.getLogger(__name__)

class LLVMFlagsInterface(MeasurementInterface):
    def __init__(self, args):
        super(LLVMFlagsInterface, self).__init__(args)
        self.suite_name = args.program
        self.util = Util()
        
        # 获取所有标志并创建映射
        self.individual_flags = self.util.gain_flags()
        log.info(f"Loaded {len(self.individual_flags)} LLVM flags for tuning")
        
        # 创建标志到参数的映射
        self.flag_to_param = {}
        for i, flag in enumerate(self.individual_flags):
            # 创建规范的参数名
            param_name = f"flag_{i}"
            self.flag_to_param[flag] = param_name
        
        # 测量基准时间
        self.baseline_time = self.measure_baseline()

    def measure_baseline(self):
        """测量基准-O3性能"""
        try:
            path = self.util.testsuite_path(self.suite_name)
            option = self.util.testsuite_option(self.suite_name)
            
            # 设置-O3并编译
            self.util.update_makefile(path, option, "-O3 ", "./data/Makefile.llvm")
            
            # 运行3次取中位数
            run_times = []
            for _ in range(3):
                runtime = self.util.get_runtime(path, self.suite_name)
                if runtime > 0:
                    run_times.append(runtime)
            
            if not run_times:
                log.error("Failed to measure baseline performance")
                return float('inf')
            
            baseline = sorted(run_times)[len(run_times)//2]
            log.info(f"Baseline -O3 time: {baseline:.4f}s")
            return baseline
        except Exception as e:
            log.error(f"Error measuring baseline: {e}")
            return float('inf')

    def manipulator(self):
        """创建配置空间"""
        manipulator = ConfigurationManipulator()
        
        # 为每个标志添加布尔参数
        for flag, param_name in self.flag_to_param.items():
            manipulator.add_parameter(
                BooleanParameter(param_name)
            )
        
        return manipulator

    def run(self, desired_result, input, limit):
        """执行和测量配置"""
        cfg = desired_result.configuration.data
        active_flags = []
        
        # 收集激活的标志
        for flag, param_name in self.flag_to_param.items():
            if cfg.get(param_name, False):
                active_flags.append(flag)
        
        if not active_flags:
            return Result(time=float('inf'))
        
        log.debug(f"Testing flags: {' '.join(active_flags)}")
        
        try:
            path = self.util.testsuite_path(self.suite_name)
            option = self.util.testsuite_option(self.suite_name)
            
            # 更新Makefile并编译
            self.util.update_makefile(path, option, " ".join(active_flags) + " ", "./data/Makefile.llvm")
            
            # 运行3次取中位数
            run_times = []
            for _ in range(3):
                runtime = self.util.get_runtime(path, self.suite_name)
                if runtime > 0:  # 忽略无效运行时间
                    run_times.append(runtime)
            
            if not run_times:
                log.warning("All runs failed for this configuration")
                return Result(time=float('inf'))
            
            median_time = sorted(run_times)[len(run_times)//2]
            speedup = self.baseline_time / median_time if self.baseline_time > 0 else 0
            
            log.info(f"Time: {median_time:.4f}s | Speedup: {speedup:.3f}x | Flags: {' '.join(active_flags)}")
            return Result(time=median_time)
            
        except Exception as e:
            log.error(f"Run failed: {e}")
            return Result(time=float('inf'))

    def save_final_config(self, configuration):
        """保存最终配置"""
        best_flags = []
        for flag, param_name in self.flag_to_param.items():
            if configuration.data.get(param_name, False):
                best_flags.append(flag)
        
        print("\n=== Best Configuration ===")
        print(f"Program: {self.suite_name}")
        print(f"Flags: {' '.join(best_flags)}")
        
        # 重新测量最佳配置
        try:
            path = self.util.testsuite_path(self.suite_name)
            option = self.util.testsuite_option(self.suite_name)
            
            self.util.update_makefile(path, option, " ".join(best_flags) + " ", "./data/Makefile.llvm")
            
            run_times = []
            for _ in range(5):  # 更多次运行以获得稳定结果
                runtime = self.util.get_runtime(path, self.suite_name)
                if runtime > 0:
                    run_times.append(runtime)
            
            if run_times:
                final_time = sorted(run_times)[len(run_times)//2]
                speedup = self.baseline_time / final_time
                print(f"Final Time: {final_time:.4f}s")
                print(f"Speedup over -O3: {speedup:.3f}x")
            else:
                print("Failed to measure final performance")
        except Exception as e:
            print(f"Error measuring final config: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=opentuner.argparsers())
    parser.add_argument('--program', required=True, help='Program to tune')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    LLVMFlagsInterface.main(args)