#!/usr/bin/env python3
"""
GPU显存峰值监控脚本
使用pynvml高频采样（50ms间隔），捕获瞬时显存峰值
按Ctrl+C停止并显示统计结果
"""

import time
import signal
import sys
from datetime import datetime
from typing import Dict, List

try:
    import pynvml
except ImportError:
    print("错误: 请先安装pynvml")
    print("安装命令: pip install pynvml")
    sys.exit(1)


class GPUMemoryMonitor:
    def __init__(self, sample_interval_ms: int = 50):
        """
        初始化GPU显存监控器
        
        Args:
            sample_interval_ms: 采样间隔（毫秒），默认50ms
        """
        self.sample_interval = sample_interval_ms / 1000.0  # 转换为秒
        self.running = True
        self.peak_memory = {}  # {gpu_id: peak_mb}
        self.peak_time = {}    # {gpu_id: timestamp}
        self.current_memory = {}  # {gpu_id: current_mb}
        self.total_memory = {}    # {gpu_id: total_mb}
        self.sample_count = 0
        self.start_time = None
        
        # 初始化NVML
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            print(f"✅ 检测到 {self.device_count} 张GPU")
            
            # 初始化每张卡的数据
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mb = memory_info.total / (1024 ** 2)
                
                self.peak_memory[i] = 0.0
                self.peak_time[i] = None
                self.current_memory[i] = 0.0
                self.total_memory[i] = total_mb
                
                print(f"   GPU {i}: {name} ({total_mb:.0f} MB)")
                
        except Exception as e:
            print(f"❌ 初始化NVML失败: {e}")
            sys.exit(1)
    
    def sample_memory(self):
        """采样一次所有GPU的显存使用情况"""
        try:
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mb = memory_info.used / (1024 ** 2)
                
                self.current_memory[i] = used_mb
                
                # 更新峰值
                if used_mb > self.peak_memory[i]:
                    self.peak_memory[i] = used_mb
                    self.peak_time[i] = datetime.now()
            
            self.sample_count += 1
            
        except Exception as e:
            print(f"\n⚠️  采样出错: {e}")
    
    def display_current(self):
        """实时显示当前显存使用情况（同行刷新）"""
        # 构建显示字符串
        display_parts = []
        for i in range(self.device_count):
            current = self.current_memory[i]
            total = self.total_memory[i]
            percentage = (current / total) * 100 if total > 0 else 0
            
            # 颜色编码：<50%绿色，50-80%黄色，>80%红色
            if percentage < 50:
                color = '\033[92m'  # 绿色
            elif percentage < 80:
                color = '\033[93m'  # 黄色
            else:
                color = '\033[91m'  # 红色
            
            reset = '\033[0m'
            
            display_parts.append(
                f"GPU{i}: {color}{current:6.0f}/{total:.0f}MB ({percentage:5.1f}%){reset}"
            )
        
        # 采样计数
        elapsed = time.time() - self.start_time if self.start_time else 0
        samples_per_sec = self.sample_count / elapsed if elapsed > 0 else 0
        
        display_str = " | ".join(display_parts)
        display_str += f" | 采样: {self.sample_count} ({samples_per_sec:.1f}/s)"
        
        # 使用\r在同一行刷新
        print(f"\r{display_str}", end='', flush=True)
    
    def print_statistics(self):
        """打印最终统计结果"""
        print("\n\n" + "=" * 80)
        print("🎯 GPU显存峰值统计")
        print("=" * 80)
        
        # 计算运行时间
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\n📊 监控时长: {elapsed:.1f}秒")
        print(f"📊 采样次数: {self.sample_count}")
        print(f"📊 采样频率: {self.sample_count / elapsed:.1f} 次/秒")
        print(f"📊 采样间隔: {self.sample_interval * 1000:.0f} ms\n")
        
        # 表头
        print(f"{'GPU':<6} {'总显存':>10} {'峰值显存':>12} {'使用率':>10} {'峰值时间':<20} {'32G够用?':<10}")
        print("-" * 80)
        
        # 每张卡的统计
        max_peak = 0.0
        for i in range(self.device_count):
            total = self.total_memory[i]
            peak = self.peak_memory[i]
            percentage = (peak / total) * 100 if total > 0 else 0
            peak_time_str = self.peak_time[i].strftime("%H:%M:%S.%f")[:-3] if self.peak_time[i] else "N/A"
            
            # 判断32G是否够用（留10%余量）
            enough_32g = "✅ 够用" if peak < 32768 * 0.9 else "❌ 不够"
            
            # 高亮显示峰值最高的卡
            if peak > max_peak:
                max_peak = peak
                highlight = '\033[93m'  # 黄色高亮
                reset = '\033[0m'
            else:
                highlight = ''
                reset = ''
            
            print(f"{highlight}GPU {i:<3} {total:8.0f} MB  {peak:9.0f} MB  {percentage:8.1f}%  {peak_time_str:<20} {enough_32g:<10}{reset}")
        
        print("-" * 80)
        
        # 总体建议
        print(f"\n💡 峰值显存使用: {max_peak:.0f} MB")
        
        if max_peak < 32768 * 0.9:  # 32GB的90%
            margin = 32768 - max_peak
            print(f"✅ 结论: 32G显卡够用！（还有 {margin:.0f} MB 余量，约 {margin/1024:.1f} GB）")
        elif max_peak < 32768:
            margin = 32768 - max_peak
            print(f"⚠️  结论: 32G显卡勉强够用（仅剩 {margin:.0f} MB 余量，约 {margin/1024:.1f} GB）")
            print("   建议: 考虑使用48G显卡以获得更多安全余量")
        else:
            deficit = max_peak - 32768
            print(f"❌ 结论: 32G显卡不够！（超出 {deficit:.0f} MB，约 {deficit/1024:.1f} GB）")
            print("   建议: 必须使用48G或更大显存的显卡")
        
        print("\n" + "=" * 80)
    
    def signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        print("\n\n⏸️  收到停止信号，正在生成统计...")
        self.running = False
    
    def run(self):
        """主监控循环"""
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("\n🚀 开始监控GPU显存使用情况...")
        print(f"⚙️  采样间隔: {self.sample_interval * 1000:.0f} ms")
        print("📌 按 Ctrl+C 停止监控并查看统计\n")
        
        self.start_time = time.time()
        
        try:
            while self.running:
                self.sample_memory()
                self.display_current()
                time.sleep(self.sample_interval)
        
        except Exception as e:
            print(f"\n❌ 监控出错: {e}")
        
        finally:
            # 打印统计
            self.print_statistics()
            
            # 清理NVML
            try:
                pynvml.nvmlShutdown()
            except:
                pass


def main():
    """主函数"""
    # 可选：从命令行参数读取采样间隔
    sample_interval_ms = 50  # 默认50ms
    
    if len(sys.argv) > 1:
        try:
            sample_interval_ms = int(sys.argv[1])
            if sample_interval_ms < 10:
                print("⚠️  警告: 采样间隔过小可能影响性能，建议 >= 10ms")
        except ValueError:
            print(f"⚠️  无效的采样间隔参数，使用默认值 {sample_interval_ms}ms")
    
    monitor = GPUMemoryMonitor(sample_interval_ms=sample_interval_ms)
    monitor.run()


if __name__ == "__main__":
    main()

