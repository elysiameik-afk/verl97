#!/usr/bin/env python3
"""
vLLM采样测试脚本
用于测试不同采样配置下的生成质量，定位乱码问题
"""

import sys
from typing import List, Dict

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("错误: 请先安装vllm")
    print("安装命令: pip install vllm")
    sys.exit(1)


# ================ 配置部分 ================

# 模型路径（根据你的实际路径修改）
MODEL_PATH = "/root/autodl-tmp/LlaSMol-EGFR-Final-exp3"

# vLLM配置（对应你的训练配置）
# 如果报错，可以尝试调整这些参数
VLLM_CONFIG = {
    "tensor_parallel_size": 1,  # 先用单卡测试，避免张量并行问题
    "gpu_memory_utilization": 0.4,  # 降低到0.4
    "max_model_len": 512,  # 降低到512，减少显存需求
    "dtype": "bfloat16",
    "trust_remote_code": True,
    # 如果还报错，取消注释下面的选项
    # "enforce_eager": True,  # 禁用CUDA graph，可能更稳定但慢
    # "disable_custom_all_reduce": True,  # 禁用自定义all-reduce
}

# 测试prompt列表
TEST_PROMPTS = [
    # 化学分子生成任务
    "Generate a small molecule that could inhibit EGFR. Output format: <SMILES>molecule</SMILES>",
    
    # 更简单的任务
    "Design a drug-like molecule. Format: <SMILES>SMILES_string</SMILES>",
    
    # 包含示例的prompt
    """Task: Generate a molecule for EGFR inhibition.
Example output: <SMILES>CCN(CC)CCNC(=O)c1ccc2ncnc(Nc3ccc(OCc4ccccn4)c(Cl)c3)c2c1</SMILES>
Now generate a new molecule:""",
]

# 测试的采样配置
SAMPLING_CONFIGS = {
    "当前配置（完全随机）": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 512,
    },
    "保守配置（推荐）": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 512,
    },
    "中等配置": {
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 100,
        "max_tokens": 512,
    },
    "贪婪配置（确定性）": {
        "temperature": 0.0,  # 贪婪解码
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 512,
    },
}


# ================ 辅助函数 ================

def is_garbage_text(text: str) -> bool:
    """
    简单判断文本是否为乱码
    
    判断标准：
    - 包含过多大写字母连续
    - 包含过多特殊字符
    - 没有正常单词
    """
    # 统计大写字母连续出现
    upper_sequences = 0
    in_upper = False
    for char in text:
        if char.isupper():
            if not in_upper:
                upper_sequences += 1
                in_upper = True
        else:
            in_upper = False
    
    # 统计特殊字符
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / len(text) if len(text) > 0 else 0
    
    # 判断是否包含<SMILES>标签
    has_smiles_tag = "<SMILES>" in text.upper() or "</SMILES>" in text.upper()
    
    # 判断标准
    is_garbage = (
        upper_sequences > 20 or  # 过多大写字母段
        special_ratio > 0.5 or   # 过多特殊字符
        (len(text) > 100 and not has_smiles_tag)  # 长文本但没有SMILES标签
    )
    
    return is_garbage


def format_output(text: str, max_length: int = 200) -> str:
    """格式化输出文本，过长则截断"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


# ================ 主测试函数 ================

def test_vllm_sampling():
    """测试vLLM采样"""
    
    print("=" * 80)
    print("🧪 vLLM采样测试")
    print("=" * 80)
    
    # 1. 加载模型
    print(f"\n📦 正在加载模型: {MODEL_PATH}")
    print(f"   配置: {VLLM_CONFIG}")
    
    try:
        llm = LLM(
            model=MODEL_PATH,
            **VLLM_CONFIG
        )
        print("✅ 模型加载成功！\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 2. 对每个prompt进行测试
    for prompt_idx, prompt in enumerate(TEST_PROMPTS, 1):
        print("\n" + "=" * 80)
        print(f"📝 测试Prompt #{prompt_idx}")
        print("=" * 80)
        print(f"Prompt: {format_output(prompt, 150)}\n")
        
        # 3. 测试不同采样配置
        for config_name, config in SAMPLING_CONFIGS.items():
            print(f"\n🔧 [{config_name}]")
            print(f"   参数: temperature={config['temperature']}, "
                  f"top_p={config['top_p']}, top_k={config['top_k']}")
            
            # 创建采样参数
            sampling_params = SamplingParams(**config)
            
            # 生成
            try:
                outputs = llm.generate([prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text
                
                # 判断是否乱码
                is_garbage = is_garbage_text(generated_text)
                status = "❌ 乱码" if is_garbage else "✅ 正常"
                
                print(f"   结果: {status}")
                print(f"   生成长度: {len(generated_text)} 字符")
                print(f"   内容: {format_output(generated_text, 200)}")
                
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
    
    # 4. 总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    print("""
如果所有配置都生成乱码：
  → 可能是模型问题（未训练/损坏）或数据格式问题
  
如果只有"当前配置"生成乱码：
  → 采样参数太随机（top_p=1.0, top_k=-1），建议修改配置
  
如果"贪婪配置"生成正常：
  → 说明模型本身OK，需要调整temperature/top_p/top_k参数
    """)


# ================ 可选：从数据集读取真实prompt ================

def test_with_real_prompts(dataset_path: str = None):
    """
    使用真实数据集中的prompt测试
    
    Args:
        dataset_path: parquet数据集路径
    """
    if dataset_path is None:
        dataset_path = "/root/autodl-tmp/verl97/verl/data/molecule_generation_train.parquet"
    
    print(f"\n📂 尝试从数据集读取prompt: {dataset_path}")
    
    try:
        import pandas as pd
        df = pd.read_parquet(dataset_path)
        
        print(f"✅ 数据集加载成功，共 {len(df)} 条数据")
        
        # 显示前3条prompt
        print("\n数据集中的前3条prompt:")
        for i in range(min(3, len(df))):
            prompt = df.iloc[i]['prompt']
            print(f"\n样本 {i+1}:")
            print(format_output(prompt, 300))
        
        # 可以选择测试真实prompt
        choice = input("\n是否使用数据集中的prompt测试？(y/n): ").strip().lower()
        if choice == 'y':
            # 选择前3条测试
            real_prompts = df['prompt'].tolist()[:3]
            global TEST_PROMPTS
            TEST_PROMPTS = real_prompts
            print("✅ 已切换到真实prompt")
        
    except Exception as e:
        print(f"⚠️  无法加载数据集: {e}")
        print("将使用默认测试prompt")


# ================ 主入口 ================

def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         vLLM采样问题诊断工具                                  ║
║                                                                              ║
║  本脚本将测试不同采样配置下的生成质量，帮助定位乱码问题                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # 询问是否使用真实数据集
    use_real = input("是否尝试加载真实数据集中的prompt？(y/n，默认n): ").strip().lower()
    if use_real == 'y':
        test_with_real_prompts()
    
    # 确认继续
    print("\n准备开始测试...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"将测试 {len(SAMPLING_CONFIGS)} 种采样配置")
    print(f"将测试 {len(TEST_PROMPTS)} 个prompt\n")
    
    input("按Enter键开始测试...")
    
    # 运行测试
    test_vllm_sampling()


if __name__ == "__main__":
    main()

