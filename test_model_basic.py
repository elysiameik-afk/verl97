#!/usr/bin/env python3
"""
基础模型测试脚本
使用HuggingFace transformers直接加载模型，测试模型是否正常
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径
MODEL_PATH = "/root/autodl-tmp/LlaSMol-EGFR-Final-exp3"

print("=" * 80)
print("🔍 基础模型测试（不使用vLLM）")
print("=" * 80)

# 1. 测试tokenizer加载
print("\n1️⃣ 测试Tokenizer加载...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"✅ Tokenizer加载成功")
    print(f"   词表大小: {tokenizer.vocab_size}")
    print(f"   特殊token: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}")
except Exception as e:
    print(f"❌ Tokenizer加载失败: {e}")
    exit(1)

# 2. 测试模型加载
print("\n2️⃣ 测试模型加载...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # 自动分配到GPU
        trust_remote_code=True
    )
    print(f"✅ 模型加载成功")
    print(f"   模型类型: {model.__class__.__name__}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("\n可能的原因:")
    print("  1. 模型文件损坏或不完整")
    print("  2. 模型格式不兼容")
    print("  3. 显存不足")
    exit(1)

# 3. 测试简单生成
print("\n3️⃣ 测试模型生成...")

test_prompts = [
    "Hello",
    "Generate a molecule",
    "Design a drug-like molecule. Format: <SMILES>SMILES_string</SMILES>"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n   测试 {i}: {prompt[:50]}...")
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成（贪婪解码，避免随机性）
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # 贪婪解码
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 只显示新生成的部分
        new_text = generated_text[len(prompt):].strip()
        
        print(f"   ✅ 生成成功")
        print(f"   生成内容: {new_text[:150]}")
        
        # 判断是否乱码
        if len(new_text) > 50:
            upper_count = sum(1 for c in new_text if c.isupper())
            upper_ratio = upper_count / len(new_text)
            if upper_ratio > 0.7:
                print(f"   ⚠️  警告: 生成内容可能是乱码（大写字母比例: {upper_ratio:.1%}）")
        
    except Exception as e:
        print(f"   ❌ 生成失败: {e}")

# 4. 总结
print("\n" + "=" * 80)
print("📊 测试总结")
print("=" * 80)
print("""
如果以上测试都通过：
  ✅ 模型本身是正常的
  → 问题在于vLLM的配置或版本
  → 建议：调整vLLM参数或降级vLLM版本

如果模型加载失败：
  ❌ 模型文件有问题
  → 需要重新下载或检查模型

如果生成内容是乱码：
  ⚠️  模型可能未训练完成或是随机初始化的
  → 检查模型训练状态
""")

