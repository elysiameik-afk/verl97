#!/usr/bin/env python3
"""检查模型是否有chat template"""

from transformers import AutoTokenizer

MODEL_PATH = "/root/autodl-tmp/LlaSMol-EGFR-Final-exp3"

print("检查模型的chat template...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"\n1. tokenizer.chat_template 是否存在: {hasattr(tokenizer, 'chat_template')}")

if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    print(f"2. 内置chat template:")
    print(tokenizer.chat_template)
    
    # 测试应用chat template
    print("\n3. 测试应用chat template:")
    messages = [{"role": "user", "content": "Generate a molecule"}]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"格式化结果:\n{formatted}")
    except Exception as e:
        print(f"应用失败: {e}")
else:
    print("2. 模型没有内置chat template")
    print("\n推荐使用自定义template:")
    print('[INST] {{ message.content }} [/INST]')

print("\n" + "="*60)

