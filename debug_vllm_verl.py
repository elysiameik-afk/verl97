#!/usr/bin/env python3
"""
vLLM调用诊断脚本 - 完全模拟verl的加载和调用流程
用于定位乱码问题的根本原因
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 80)
print("🔍 vLLM vs Transformers 对比测试")
print("=" * 80)

# ============================================================================
# 配置（和verl训练一致）
# ============================================================================
MODEL_PATH = "/root/autodl-tmp/LlaSMol-EGFR-Final-exp3"
DATASET_PATH = "/root/autodl-tmp/verl97/verl/data/molecule_generation_train.parquet"

# vLLM配置（对应chem.sh）
VLLM_CONFIG = {
    "tensor_parallel_size": 4,  # 和你的训练配置一致
    "gpu_memory_utilization": 0.5,
    "max_model_len": 1024,
    "dtype": "bfloat16",
    "trust_remote_code": True,
}

# 采样参数（训练时）
TRAIN_SAMPLING = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "max_tokens": 512,
}

# 采样参数（验证时）
VAL_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
}

# ============================================================================
# 1. 加载数据集
# ============================================================================
print("\n1️⃣ 加载数据集...")
try:
    df = pd.read_parquet(DATASET_PATH)
    print(f"✅ 数据集加载成功，共 {len(df)} 条数据")
    print(f"   列名: {list(df.columns)}")
    
    # 显示前3条prompt
    print(f"\n前3条prompt:")
    for i in range(min(3, len(df))):
        prompt = df.iloc[i]['prompt']
        print(f"   {i+1}. {prompt[:100]}...")
    
    # 选择测试样本
    test_prompts = df['prompt'].tolist()[:3]
    
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    exit(1)

# ============================================================================
# 2. 加载Tokenizer
# ============================================================================
print("\n2️⃣ 加载Tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"✅ Tokenizer加载成功")
    print(f"   词表大小: {tokenizer.vocab_size}")
    print(f"   pad_token_id: {tokenizer.pad_token_id}")
    print(f"   eos_token_id: {tokenizer.eos_token_id}")
    print(f"   bos_token_id: {tokenizer.bos_token_id}")
    
except Exception as e:
    print(f"❌ Tokenizer加载失败: {e}")
    exit(1)

# ============================================================================
# 3. 测试Transformers生成（baseline）
# ============================================================================
print("\n3️⃣ 测试Transformers生成（baseline）...")

try:
    model_hf = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ HF模型加载成功")
    
    for i, prompt in enumerate(test_prompts[:2], 1):
        print(f"\n   测试样本 {i}:")
        print(f"   Prompt: {prompt[:80]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model_hf.device)
        
        # 使用和验证时相同的采样参数
        with torch.no_grad():
            outputs = model_hf.generate(
                **inputs,
                max_new_tokens=VAL_SAMPLING["max_tokens"],
                temperature=VAL_SAMPLING["temperature"],
                top_p=VAL_SAMPLING["top_p"],
                top_k=VAL_SAMPLING["top_k"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):].strip()
        
        print(f"   ✅ Transformers生成: {new_text[:200]}")
        
        # 判断是否包含SMILES标签
        has_smiles = "<SMILES>" in new_text.upper() or "smiles" in new_text.lower()
        print(f"   包含SMILES相关内容: {'✅ 是' if has_smiles else '❌ 否'}")
    
    # 释放显存
    del model_hf
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"❌ Transformers测试失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. 测试vLLM生成（verl使用的引擎）
# ============================================================================
print("\n4️⃣ 测试vLLM生成（verl使用的引擎）...")

try:
    from vllm import LLM, SamplingParams
    
    print(f"   vLLM配置: {VLLM_CONFIG}")
    
    llm = LLM(
        model=MODEL_PATH,
        **VLLM_CONFIG
    )
    print(f"✅ vLLM加载成功")
    
    # 测试训练时的采样参数
    print("\n   【测试A：训练时采样参数】")
    print(f"   参数: {TRAIN_SAMPLING}")
    
    sampling_params_train = SamplingParams(**TRAIN_SAMPLING)
    
    for i, prompt in enumerate(test_prompts[:2], 1):
        print(f"\n   测试样本 {i}:")
        print(f"   Prompt: {prompt[:80]}...")
        
        outputs = llm.generate([prompt], sampling_params_train)
        generated_text = outputs[0].outputs[0].text
        
        print(f"   🔍 vLLM生成（训练参数）: {generated_text[:200]}")
        
        # 判断是否乱码
        upper_count = sum(1 for c in generated_text if c.isupper())
        if len(generated_text) > 0:
            upper_ratio = upper_count / len(generated_text)
            if upper_ratio > 0.5:
                print(f"   ❌ 疑似乱码（大写字母比例: {upper_ratio:.1%}）")
            else:
                print(f"   ✅ 看起来正常")
    
    # 测试验证时的采样参数
    print("\n   【测试B：验证时采样参数】")
    print(f"   参数: {VAL_SAMPLING}")
    
    sampling_params_val = SamplingParams(**VAL_SAMPLING)
    
    for i, prompt in enumerate(test_prompts[:2], 1):
        print(f"\n   测试样本 {i}:")
        print(f"   Prompt: {prompt[:80]}...")
        
        outputs = llm.generate([prompt], sampling_params_val)
        generated_text = outputs[0].outputs[0].text
        
        print(f"   🔍 vLLM生成（验证参数）: {generated_text[:200]}")
        
        # 判断是否乱码
        upper_count = sum(1 for c in generated_text if c.isupper())
        if len(generated_text) > 0:
            upper_ratio = upper_count / len(generated_text)
            if upper_ratio > 0.5:
                print(f"   ❌ 疑似乱码（大写字母比例: {upper_ratio:.1%}）")
            else:
                print(f"   ✅ 看起来正常")
    
    # 测试贪婪解码（baseline）
    print("\n   【测试C：贪婪解码（baseline）】")
    sampling_params_greedy = SamplingParams(temperature=0.0, max_tokens=512)
    
    for i, prompt in enumerate(test_prompts[:2], 1):
        print(f"\n   测试样本 {i}:")
        print(f"   Prompt: {prompt[:80]}...")
        
        outputs = llm.generate([prompt], sampling_params_greedy)
        generated_text = outputs[0].outputs[0].text
        
        print(f"   🔍 vLLM生成（贪婪）: {generated_text[:200]}")
        
        has_smiles = "<SMILES>" in generated_text.upper()
        print(f"   包含SMILES标签: {'✅ 是' if has_smiles else '❌ 否'}")

except Exception as e:
    print(f"❌ vLLM测试失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 5. 总结
# ============================================================================
print("\n" + "=" * 80)
print("📊 诊断总结")
print("=" * 80)
print("""
如果Transformers正常，vLLM也正常：
  → 问题可能在verl的rollout wrapper或数据传递逻辑
  
如果Transformers正常，vLLM乱码：
  → 问题在vLLM的配置或模型加载方式
  → 可能是tensor_parallel、load_format等参数问题
  
如果vLLM贪婪解码正常，但采样乱码：
  → 问题在采样参数
  → 需要进一步调整temperature/top_p/top_k
  
如果所有都乱码：
  → 检查模型config.json（特别是rms_norm_eps）
""")

