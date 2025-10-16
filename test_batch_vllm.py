#!/usr/bin/env python3
"""
Batch vLLM测试 - 模拟verl的batch推理和n=8配置
测试是否batch+多样本生成会导致部分样本乱码
"""

import pandas as pd
from vllm import LLM, SamplingParams

# ============================================================================
# 配置（完全模拟verl训练）
# ============================================================================
MODEL_PATH = "/root/autodl-tmp/LlaSMol-EGFR-Final-exp3"
DATASET_PATH = "/root/autodl-tmp/verl97/verl/data/molecule_generation_train.parquet"

# vLLM配置（tensor_parallel=1，和现在训练一致）
VLLM_CONFIG = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.5,
    "max_model_len": 1024,
    "dtype": "bfloat16",
    "trust_remote_code": True,
}

# 采样参数（和verl训练时一致）
SAMPLING_PARAMS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 512,
    "n": 8,  # ← 关键：每个prompt生成8个response（verl配置）
    "repetition_penalty": 1.0,
}

# Batch配置
BATCH_SIZE = 16  # 和verl的train_batch_size一致


def is_garbage(text):
    """简单判断是否乱码"""
    if len(text) < 10:
        return True
    
    # 检查重复
    if "[/INST]" in text:
        inst_count = text.count("[/INST]")
        if inst_count > 2:  # 重复生成prompt
            return True
    
    # 检查数字重复
    if "230652306523065" in text or "5.5.5.5.5.5" in text:
        return True
    
    # 检查是否有SMILES标签
    if "<SMILES>" not in text.upper():
        return True
    
    return False


def main():
    print("=" * 80)
    print("🧪 Batch vLLM测试 - 模拟verl的batch+n=8配置")
    print("=" * 80)
    
    # 1. 加载数据集
    print(f"\n1️⃣ 加载数据集...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"✅ 数据集加载成功，共 {len(df)} 条")
    
    # 取前batch_size条
    test_prompts = df['prompt'].tolist()[:BATCH_SIZE]
    print(f"   测试样本数: {len(test_prompts)}")
    
    # 2. 加载vLLM
    print(f"\n2️⃣ 加载vLLM...")
    print(f"   配置: {VLLM_CONFIG}")
    llm = LLM(model=MODEL_PATH, **VLLM_CONFIG)
    print(f"✅ vLLM加载成功")
    
    # 3. Batch推理
    print(f"\n3️⃣ Batch推理...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   每个prompt生成数: {SAMPLING_PARAMS['n']}")
    print(f"   总生成数: {BATCH_SIZE * SAMPLING_PARAMS['n']}")
    print(f"   采样参数: temperature={SAMPLING_PARAMS['temperature']}, top_p={SAMPLING_PARAMS['top_p']}, top_k={SAMPLING_PARAMS['top_k']}")
    
    sampling_params = SamplingParams(**SAMPLING_PARAMS)
    
    # Batch生成
    outputs = llm.generate(test_prompts, sampling_params)
    
    # 4. 分析结果
    print(f"\n4️⃣ 分析结果...")
    
    total_responses = 0
    success_count = 0
    failure_count = 0
    
    for prompt_idx, output in enumerate(outputs):
        print(f"\n--- Prompt {prompt_idx+1}/{len(outputs)} ---")
        print(f"Prompt: {test_prompts[prompt_idx][:60]}...")
        
        for sample_idx, sample_output in enumerate(output.outputs):
            total_responses += 1
            response_text = sample_output.text
            
            is_bad = is_garbage(response_text)
            
            status = "❌ 乱码" if is_bad else "✅ 正常"
            print(f"  样本{sample_idx+1}/8: {status} - {response_text[:80]}...")
            
            if is_bad:
                failure_count += 1
            else:
                success_count += 1
    
    # 5. 统计
    print("\n" + "=" * 80)
    print("📊 统计结果")
    print("=" * 80)
    print(f"总生成数: {total_responses}")
    print(f"成功: {success_count} ({success_count/total_responses*100:.1f}%)")
    print(f"失败: {failure_count} ({failure_count/total_responses*100:.1f}%)")
    
    print(f"\n💡 结论：")
    if success_count == total_responses:
        print("✅ 全部成功 → batch+n=8配置没问题，问题在verl的其他逻辑")
    elif success_count > total_responses * 0.8:
        print("⚠️  大部分成功(>80%) → 采样随机性导致，可以通过降低temperature改善")
    elif success_count > total_responses * 0.3:
        print("⚠️  部分成功(30-80%) → 和verl训练类似，问题在batch+n配置或数据集")
    else:
        print("❌ 大部分失败(<30%) → 配置或数据集有严重问题")


if __name__ == "__main__":
    main()

