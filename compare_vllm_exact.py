#!/usr/bin/env python3
"""
精确对比测试 - 使用和verl训练完全相同的token IDs
用于定位verl和独立vLLM的差异
"""

from vllm import LLM, SamplingParams

# ============================================================================
# 配置：从训练日志中复制
# ============================================================================
MODEL_PATH = "/root/autodl-tmp/LlaSMol-EGFR-Final-exp3"

# 从训练日志复制的第一个样本的token IDs
# 🔍 [DEBUG] 输入vLLM的第1个prompt: Token IDs (前30个)
VERL_INPUT_TOKEN_IDS = [
    28792, 16289, 28793, 1094, 5645, 413, 28777, 9790, 28733, 28738, 
    28796, 28737, 354, 1843, 28733, 9310, 3601, 14966, 8875, 325, 
    7016, 3100, 28743, 28731, 5827, 5938, 272, 2747, 1871, 298,
    # ... 把完整的token IDs粘贴在这里
    # 从日志的 "Token IDs (前30个)" 和 "Token IDs (后10个)" 拼接
]

# vLLM配置（和verl训练一致）
VLLM_CONFIG = {
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.5,
    "max_model_len": 1024,
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "enforce_eager": False,  # 和verl一样，先用False测试
}

# 验证时的采样参数（从训练日志复制）
VAL_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "repetition_penalty": 1.0,
}


def main():
    print("=" * 80)
    print("🔍 精确对比测试：verl vs 独立vLLM")
    print("=" * 80)
    
    # 步骤1：打印输入
    print(f"\n1️⃣ 输入token IDs（从verl训练日志复制）：")
    print(f"   长度: {len(VERL_INPUT_TOKEN_IDS)} tokens")
    print(f"   前10个: {VERL_INPUT_TOKEN_IDS[:10]}")
    print(f"   后10个: {VERL_INPUT_TOKEN_IDS[-10:]}")
    print(f"   第一个token: {VERL_INPUT_TOKEN_IDS[0]}")
    print(f"   是BOS(1): {'✅ 是' if VERL_INPUT_TOKEN_IDS[0] == 1 else '❌ 否'}")
    
    # 步骤2：加载vLLM
    print(f"\n2️⃣ 加载vLLM...")
    print(f"   配置: {VLLM_CONFIG}")
    
    try:
        llm = LLM(model=MODEL_PATH, **VLLM_CONFIG)
        print(f"   ✅ vLLM加载成功")
    except Exception as e:
        print(f"   ❌ vLLM加载失败: {e}")
        return
    
    # 步骤3：使用相同的token IDs生成
    print(f"\n3️⃣ 测试生成...")
    print(f"   采样参数: {VAL_SAMPLING}")
    
    sampling_params = SamplingParams(**VAL_SAMPLING)
    
    # 使用prompt_token_ids而不是字符串
    from vllm.inputs import TokensPrompt
    prompt = TokensPrompt(prompt_token_ids=VERL_INPUT_TOKEN_IDS)
    
    try:
        outputs = llm.generate(prompt, sampling_params)
        
        output_token_ids = outputs[0].outputs[0].token_ids
        
        print(f"\n4️⃣ 生成结果：")
        print(f"   输出长度: {len(output_token_ids)} tokens")
        print(f"   输出token IDs (前30个): {output_token_ids[:30]}")
        print(f"   输出token IDs (后10个): {output_token_ids[-10:]}")
        
        # Decode
        tokenizer = llm.get_tokenizer()
        decoded = tokenizer.decode(output_token_ids)
        print(f"\n   解码后的文本:")
        print(f"   {decoded[:300]}")
        
        # 判断
        has_smiles = "<SMILES>" in decoded.upper()
        is_repetitive = decoded.count("[/INST]") > 2
        
        print(f"\n5️⃣ 分析：")
        print(f"   包含SMILES标签: {'✅ 是' if has_smiles else '❌ 否'}")
        print(f"   有重复内容: {'❌ 是(重复{})次[/INST])'.format(decoded.count('[/INST]')) if is_repetitive else '✅ 否'}")
        
        if has_smiles and not is_repetitive:
            print(f"\n   ✅ 结论: 独立vLLM生成正常！")
            print(f"   → 问题在verl的其他配置（可能是CUDA graph、prefix caching等）")
        elif is_repetitive:
            print(f"\n   ❌ 结论: 独立vLLM也重复生成！")
            print(f"   → 问题在采样参数或vLLM配置本身")
        else:
            print(f"\n   ⚠️  结论: 独立vLLM生成异常但不重复")
            print(f"   → 需要进一步检查")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("📝 下一步：")
    print("=" * 80)
    print("""
1. 从verl训练日志中找到完整的token IDs
   - 找 "[DEBUG] 输入vLLM的第1个prompt"
   - 复制完整的前30个和后10个token IDs
   
2. 把完整的token IDs填入本脚本的 VERL_INPUT_TOKEN_IDS
   
3. 重新运行本脚本：python compare_vllm_exact.py
   
4. 对比结果
""")


if __name__ == "__main__":
    main()

