# coding=utf-8
# run_hebbian_external_example.py

import torch
from transformers import AutoTokenizer, MambaForCausalLM, MambaConfig # 导入标准的 Mamba 类
from hebbian_utils import ( # 从我们创建的工具文件中导入函数
    enable_hebbian_hooks_on_model,
    disable_hebbian_hooks_on_model,
    hebbian_update_weights,
    store_attention_mask_for_hebbian,
    clear_hebbian_activations # 如果需要在更新权重之外清除
)

def run_external_example():
    """
    运行使用外部 Hebbian 工具函数微调 Mamba 模型的示例。
    """
    # 1. 配置并实例化一个标准的 Mamba 模型
    try:
        config = MambaConfig.from_pretrained("state-spaces/mamba-130m-hf")
        config.num_hidden_layers = 2
        config.hidden_size = 128
        config.intermediate_size = config.hidden_size * MambaConfig().expand
        config.state_size = 16
        config.vocab_size = 1000
        if not hasattr(config, 'time_step_rank') or config.time_step_rank is None:
             config.time_step_rank = config.intermediate_size // 16
        if not hasattr(config, 'time_step_init_scheme'):
             config.time_step_init_scheme = "random"
    except Exception as e:
        print(f"从 hub 加载配置失败，将使用一个简化的手动配置: {e}")
        config = MambaConfig(
            vocab_size=1000, hidden_size=64, state_size=8, num_hidden_layers=2,
            layer_norm_epsilon=1e-5, intermediate_size=128, conv_kernel=4,
            time_step_rank=32, time_step_scale=1.0, time_step_min=0.001,
            time_step_max=0.1, time_step_floor=1e-4, use_conv_bias=True,
            use_bias=False, hidden_act="silu", expand=2,
            rescale_prenorm_residual=False, residual_in_fp32=False,
            time_step_init_scheme="random",
        )
        if not hasattr(config, 'intermediate_size'):
            config.intermediate_size = config.hidden_size * config.expand


    model = MambaForCausalLM(config) # 使用标准的 MambaForCausalLM
    model.eval()

    # 2. 定义目标层和 Hebbian 更新参数
    target_layers_ext = [
        "backbone.layers.0.mixer.out_proj",
        "lm_head"
    ]
    hebbian_lr_ext = 0.001

    # 3. 在模型上启用 Hebbian hooks
    enable_hebbian_hooks_on_model(model, target_module_names=target_layers_ext)

    # 4. 准备输入数据
    batch_size_ext = 2
    seq_length_ext = 20
    dummy_input_ids_ext = torch.randint(0, config.vocab_size, (batch_size_ext, seq_length_ext))
    dummy_labels_ext = torch.randint(0, config.vocab_size, (batch_size_ext, seq_length_ext))
    dummy_attention_mask_ext = torch.ones_like(dummy_input_ids_ext)
    if batch_size_ext > 0 and seq_length_ext > 5:
        dummy_attention_mask_ext[0, -5:] = 0

    print(f"模块 {target_layers_ext[0]} 的初始权重均值: {model.get_submodule(target_layers_ext[0]).weight.data.mean().item()}")
    if len(target_layers_ext) > 1 and target_layers_ext[1] in dict(model.named_modules()):
         print(f"模块 {target_layers_ext[1]} 的初始权重均值: {model.get_submodule(target_layers_ext[1]).weight.data.mean().item()}")

    # --- 模拟训练迭代 ---
    print("\n--- 第一次 Hebbian 更新 (外部工具) ---")
    # 5. 执行模型的前向传播
    #    这将触发已注册的 hooks，并将激活存储在 hebbian_utils 的全局缓冲区中。
    #    同时，我们需要手动存储 attention_mask。
    store_attention_mask_for_hebbian(dummy_attention_mask_ext)
    outputs_ext = model(input_ids=dummy_input_ids_ext, labels=dummy_labels_ext, attention_mask=dummy_attention_mask_ext)
    loss_ext = outputs_ext.loss
    print(f"标准损失: {loss_ext.item()}")

    # 6. 计算奖励信号
    reward_signal_ext = -loss_ext.item()
    print(f"计算得到的奖励: {reward_signal_ext}")

    # 7. 调用外部函数应用 Hebbian 更新
    hebbian_update_weights(
        model=model,
        target_module_names=target_layers_ext,
        reward=reward_signal_ext,
        learning_rate=hebbian_lr_ext
    )
    # hebbian_update_weights 内部会调用 clear_hebbian_activations()

    print(f"模块 {target_layers_ext[0]} Hebbian 更新后的权重均值: {model.get_submodule(target_layers_ext[0]).weight.data.mean().item()}")
    if len(target_layers_ext) > 1 and target_layers_ext[1] in dict(model.named_modules()):
        print(f"模块 {target_layers_ext[1]} Hebbian 更新后的权重均值: {model.get_submodule(target_layers_ext[1]).weight.data.mean().item()}")


    # --- 模拟另一次迭代 ---
    print("\n--- 第二次 Hebbian 更新 (外部工具) ---")
    # 再次存储 attention_mask (如果它可能已更改) 并执行前向传播
    store_attention_mask_for_hebbian(dummy_attention_mask_ext)
    model(input_ids=dummy_input_ids_ext, attention_mask=dummy_attention_mask_ext) # 不需要 labels 来填充缓冲区

    reward_signal_ext_2 = 0.5
    print(f"第二次迭代计算得到的奖励: {reward_signal_ext_2}")
    hebbian_update_weights(
        model=model,
        target_module_names=target_layers_ext,
        reward=reward_signal_ext_2,
        learning_rate=hebbian_lr_ext
    )

    print(f"模块 {target_layers_ext[0]} 第二次 Hebbian 更新后的权重均值: {model.get_submodule(target_layers_ext[0]).weight.data.mean().item()}")
    if len(target_layers_ext) > 1 and target_layers_ext[1] in dict(model.named_modules()):
        print(f"模块 {target_layers_ext[1]} 第二次 Hebbian 更新后的权重均值: {model.get_submodule(target_layers_ext[1]).weight.data.mean().item()}")


    # 8. 禁用 Hebbian hooks
    disable_hebbian_hooks_on_model(model)
    print("\nHebbian hooks 已禁用。")

    # 9. (可选) 测试文本生成
    try:
        print("\n测试文本生成 (使用标准 Mamba 模型)...")
        tokenizer_ext = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        if tokenizer_ext.pad_token is None:
            tokenizer_ext.pad_token = tokenizer_ext.eos_token

        # 使用与上面相同的模型实例（其权重已被 Hebbian 更新修改）
        model.config.vocab_size = tokenizer_ext.vocab_size # 确保词汇表匹配
        model.eval()

        input_text_ext = "This is a test for Mamba"
        inputs_ext = tokenizer_ext(input_text_ext, return_tensors="pt", padding=True, truncation=True)

        generated_ids_ext = model.generate(
            inputs_ext.input_ids,
            attention_mask=inputs_ext.attention_mask,
            max_new_tokens=10,
            use_cache=True,
            pad_token_id=tokenizer_ext.pad_token_id
        )
        decoded_text_ext = tokenizer_ext.decode(generated_ids_ext[0], skip_special_tokens=True)
        print(f"生成的文本: {decoded_text_ext}")

    except Exception as e:
        print(f"生成测试过程中发生错误: {e}")

if __name__ == "__main__":
    run_external_example()