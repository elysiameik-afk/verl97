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

# 从训练日志复制的第一个样本的token IDs（完整的40个tokens）
# 🔍 [DEBUG-Final] 最终传给vLLM的prompt: 完整token IDs
VERL_INPUT_TOKEN_IDS = [
    1, 28792, 16289, 28793, 1094, 5645, 413, 28777, 9790, 28733, 
    28738, 28796, 28737, 354, 1843, 28733, 9310, 3601, 14966, 8875, 
    325, 7016, 3100, 28743, 28731, 5827, 5938, 272, 2747, 1871, 
    298, 2231, 264, 12160, 27969, 28723, 733, 28748, 16289, 28793
]

# vLLM配置（和verl训练完全一致）
VLLM_CONFIG = {
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.5,
    "max_model_len": 1024,
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "enforce_eager": False,  # verl默认值
    "enable_prefix_caching": True,  # verl默认开启
    "enable_chunked_prefill": True,  # verl配置
    "disable_custom_all_reduce": True,  # verl配置
}

# 采样参数（从训练日志复制）
# [vLLM采样参数] temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.0
SAMPLING_PARAMS = {
    "temperature": 0.9,
    "top_p": 0.95,
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
    print(f"   采样参数: {SAMPLING_PARAMS}")
    
    sampling_params = SamplingParams(**SAMPLING_PARAMS)
    
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
    print("📝 说明：")
    print("=" * 80)
    print("""
本脚本已填入从verl训练日志提取的实际token IDs和采样参数。

如果这个测试也出现重复生成：
  → 说明问题在vLLM配置（tensor_parallel=4 + 其他配置的组合）
  → 尝试修改配置：enforce_eager=True 或 tensor_parallel_size=1

如果这个测试生成正常：
  → 说明verl还有其他未知的配置影响了vLLM
  → 需要对比verl的LLM初始化参数
""")


if __name__ == "__main__":
    main()

