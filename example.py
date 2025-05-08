# coding=utf-8
# run_hebbian_external_example.py

import torch
from transformers import AutoTokenizer, MambaForCausalLM, MambaConfig
from hebbian_utils import (
    enable_hebbian_hooks_on_model,
    disable_hebbian_hooks_on_model,
    hebbian_update_weights,
    store_attention_mask_for_hebbian,
    clear_hebbian_activations
)
from transformers.utils import logging # 导入 logging

logger = logging.get_logger(__name__)


def prepare_data_for_mamba(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    准备一批文本数据用于 Mamba 模型。

    Args:
        texts (List[str]): 包含要处理的文本字符串的列表。
        tokenizer (AutoTokenizer): 用于分词的 Hugging Face 分词器。
        max_length (int): 分词后序列的最大长度。较短的序列将被填充，较长的将被截断。
        device (torch.device): 数据应该加载到的设备 (例如, torch.device("cuda"))。

    Returns:
        Dict[str, torch.Tensor]: 一个包含 "input_ids", "attention_mask", 和 "labels" 的字典。
    """
    # 确保分词器有 pad_token，如果没有，通常使用 eos_token
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad_token. Using eos_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # 分词文本
    # padding="max_length": 将所有序列填充到 max_length
    # truncation=True: 将所有序列截断到 max_length
    # return_tensors="pt": 返回 PyTorch 张量
    tokenized_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)

    # 创建 labels 用于因果语言模型
    # labels 通常是 input_ids 向左移动一位
    # 在被模型忽略的位置（例如，填充token或最后一个token的预测），labels 通常被设置为 -100
    labels = input_ids.clone()

    # 对于因果LM，我们不预测第一个token的下一个token，也不需要为最后一个token之后的token计算损失
    # 因此，通常的做法是将labels向左移，并将最后一个token的label设为-100
    # 或者，更常见的做法是，在计算损失时对logits和labels进行移位。
    # Hugging Face 的 MambaForCausalLM 内部会处理 logits 和 labels 的移位。
    # 所以，我们可以直接将 input_ids 作为 labels 传递，但要确保填充部分的 labels 被忽略。
    # CrossEntropyLoss 默认忽略 target 为 -100 的项。
    # 当 attention_mask 为 0 时，对应的 labels 应该被设为 -100。
    labels[attention_mask == 0] = -100 # 将填充位置的 labels 设为 -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def run_external_example():
    """
    运行使用外部 Hebbian 工具函数微调 Mamba 模型的示例。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. 配置并实例化一个标准的 Mamba 模型
    try:
        # 使用一个较小的、公开的 Mamba 配置进行快速测试
        # 如果 "state-spaces/mamba-130m-hf" 对于本地测试太大或太慢，可以考虑更小的配置
        # 或者如之前一样手动创建一个非常小的配置
        config_name = "state-spaces/mamba-130m-hf" # 或者其他你想要测试的 Mamba 模型
        config = MambaConfig.from_pretrained(config_name)

        # 为了快速演示和减少内存占用，我们通常会减小模型规模
        # 如果你想测试实际性能，应该使用原始配置或更大的配置
        config.num_hidden_layers = 2  # 示例：减少层数
        config.hidden_size = 256      # 示例：减小隐藏层大小
        # intermediate_size 通常是 hidden_size 的 expand 倍，MambaConfig 有默认的 expand
        config.intermediate_size = config.hidden_size * getattr(config, 'expand', MambaConfig().expand)
        config.state_size = 16        # 示例：减小状态大小

        # 词汇表大小将由分词器决定
        # 我们稍后会设置它

    except Exception as e:
        logger.error(f"从 hub 加载配置 {config_name} 失败，将使用一个简化的手动配置: {e}")
        config = MambaConfig(
            vocab_size=50257, # 临时值，会被分词器覆盖
            hidden_size=64, state_size=8, num_hidden_layers=2,
            layer_norm_epsilon=1e-5,
            intermediate_size=128, # hidden_size * expand (默认 expand=2)
            conv_kernel=4,
            time_step_rank=32,
            time_step_scale=1.0, time_step_min=0.001,
            time_step_max=0.1, time_step_floor=1e-4,
            use_conv_bias=True, use_bias=False, hidden_act="silu", expand=2,
            # 确保以下属性存在，Mamba 模型实现可能需要
            rescale_prenorm_residual=False, residual_in_fp32=False,
            time_step_init_scheme="random",
        )
    # 确保 MambaMixer 和 MambaPreTrainedModel._init_weights 所需的属性存在
    if not hasattr(config, 'time_step_rank') or config.time_step_rank is None:
        config.time_step_rank = config.intermediate_size // 16
    if not hasattr(config, 'time_step_init_scheme'):
        config.time_step_init_scheme = "random"


    # 加载分词器
    # 使用与模型配置对应的分词器，或者一个通用的分词器进行测试
    # "EleutherAI/gpt-neox-20b" 的分词器是一个不错的选择，因为它被广泛使用
    # "state-spaces/mamba-130m-hf" 模型本身可能没有关联特定的分词器，但它通常与 GPT-NeoX 风格的分词器一起使用
    try:
        tokenizer_name = "EleutherAI/gpt-neox-20b"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        logger.error(f"加载分词器 {tokenizer_name} 失败: {e}. 尝试 gpt2 分词器。")
        tokenizer_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


    # 更新模型配置的词汇表大小以匹配分词器
    config.vocab_size = tokenizer.vocab_size
    # 确保 pad_token_id 在配置中设置正确
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        logger.info("Tokenizer has no pad_token_id. Setting model's pad_token_id to eos_token_id.")
        config.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token # 确保分词器实例也同步
    else:
        # 如果两者都没有，需要添加一个新的 pad token
        logger.warning("Tokenizer has no pad_token_id or eos_token_id. Adding a new pad token.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        config.vocab_size = len(tokenizer) # 更新词汇表大小
        config.pad_token_id = tokenizer.pad_token_id


    model = MambaForCausalLM(config).to(device) # 使用标准的 MambaForCausalLM
    model.eval() # 或 model.train()

    # 2. 定义目标层和 Hebbian 更新参数
    target_layers_ext = [
        "backbone.layers.0.mixer.out_proj", # 示例：第一个 Mamba 块的混合器中的输出投影
        "lm_head"                           # 语言模型头
    ]
    hebbian_lr_ext = 0.001

    # 3. 在模型上启用 Hebbian hooks
    enable_hebbian_hooks_on_model(model, target_module_names=target_layers_ext)

    # 4. 准备输入数据 - 使用新的 prepare_data_for_mamba 函数
    sample_texts = [
        "Mamba is a fascinating new architecture for sequence modeling.",
        "The quick brown fox jumps over the lazy dog.",
        "This is a shorter sentence." # 测试不同长度和填充
    ]
    max_sequence_length = 64 # 定义最大序列长度

    data_batch = prepare_data_for_mamba(sample_texts, tokenizer, max_sequence_length, device)
    input_ids_ext = data_batch["input_ids"]
    attention_mask_ext = data_batch["attention_mask"]
    labels_ext = data_batch["labels"]

    logger.info(f"Input IDs shape: {input_ids_ext.shape}")
    logger.info(f"Attention Mask shape: {attention_mask_ext.shape}")
    logger.info(f"Labels shape: {labels_ext.shape}")

    # 打印一些初始权重信息
    if target_layers_ext[0] in dict(model.named_modules()):
        logger.info(f"模块 {target_layers_ext[0]} 的初始权重均值: {model.get_submodule(target_layers_ext[0]).weight.data.mean().item()}")
    if len(target_layers_ext) > 1 and target_layers_ext[1] in dict(model.named_modules()):
         logger.info(f"模块 {target_layers_ext[1]} 的初始权重均值: {model.get_submodule(target_layers_ext[1]).weight.data.mean().item()}")


    # --- 模拟训练迭代 ---
    logger.info("\n--- 第一次 Hebbian 更新 (外部工具) ---")
    # 5. 执行模型的前向传播
    store_attention_mask_for_hebbian(attention_mask_ext) # 存储当前批次的掩码
    outputs_ext = model(
        input_ids=input_ids_ext,
        attention_mask=attention_mask_ext,
        labels=labels_ext
    )
    loss_ext = outputs_ext.loss
    logger.info(f"标准损失: {loss_ext.item()}")

    # 6. 计算奖励信号
    reward_signal_ext = -loss_ext.item() # 示例奖励
    logger.info(f"计算得到的奖励: {reward_signal_ext}")

    # 7. 调用外部函数应用 Hebbian 更新
    hebbian_update_weights(
        model=model,
        target_module_names=target_layers_ext,
        reward=reward_signal_ext,
        learning_rate=hebbian_lr_ext
    )

    if target_layers_ext[0] in dict(model.named_modules()):
        logger.info(f"模块 {target_layers_ext[0]} Hebbian 更新后的权重均值: {model.get_submodule(target_layers_ext[0]).weight.data.mean().item()}")
    if len(target_layers_ext) > 1 and target_layers_ext[1] in dict(model.named_modules()):
        logger.info(f"模块 {target_layers_ext[1]} Hebbian 更新后的权重均值: {model.get_submodule(target_layers_ext[1]).weight.data.mean().item()}")


    # --- 模拟另一次迭代 ---
    logger.info("\n--- 第二次 Hebbian 更新 (外部工具) ---")
    # 假设我们有新的文本或只是想用不同的奖励再次运行
    # 为了简单起见，我们使用相同的数据，但模拟不同的奖励
    store_attention_mask_for_hebbian(attention_mask_ext) # 再次存储掩码
    # 如果不计算损失，可以不传 labels，但前向传播仍需运行以填充激活缓冲区
    model(input_ids=input_ids_ext, attention_mask=attention_mask_ext)

    reward_signal_ext_2 = 0.5 # 示例：正奖励
    logger.info(f"第二次迭代计算得到的奖励: {reward_signal_ext_2}")
    hebbian_update_weights(
        model=model,
        target_module_names=target_layers_ext,
        reward=reward_signal_ext_2,
        learning_rate=hebbian_lr_ext
    )

    if target_layers_ext[0] in dict(model.named_modules()):
        logger.info(f"模块 {target_layers_ext[0]} 第二次 Hebbian 更新后的权重均值: {model.get_submodule(target_layers_ext[0]).weight.data.mean().item()}")
    if len(target_layers_ext) > 1 and target_layers_ext[1] in dict(model.named_modules()):
        logger.info(f"模块 {target_layers_ext[1]} 第二次 Hebbian 更新后的权重均值: {model.get_submodule(target_layers_ext[1]).weight.data.mean().item()}")


    # 8. 禁用 Hebbian hooks
    disable_hebbian_hooks_on_model(model)
    logger.info("\nHebbian hooks 已禁用。")

    # 9. (可选) 测试文本生成
    try:
        logger.info("\n测试文本生成 (使用标准 Mamba 模型，其权重已被修改)...")
        model.eval() # 确保模型处于评估模式

        input_text_ext = "Mamba models are known for their"
        # 使用之前定义的同一个分词器
        inputs_ext_gen = tokenizer(
            input_text_ext,
            return_tensors="pt",
            padding=True, # 对于单个输入，padding 可能不是必需的，但保持一致性
            truncation=True,
            max_length=max_sequence_length # 与训练时使用的长度一致或更短
        ).to(device)

        logger.info(f"生成输入: '{input_text_ext}'")
        logger.info(f"模型配置 pad_token_id: {model.config.pad_token_id}, tokenizer pad_token_id: {tokenizer.pad_token_id}")
        logger.info(f"模型配置 eos_token_id: {model.config.eos_token_id}, tokenizer eos_token_id: {tokenizer.eos_token_id}")


        # 确保 generate 函数使用的 pad_token_id 与模型配置一致
        pad_token_id_for_gen = model.config.pad_token_id if model.config.pad_token_id is not None else tokenizer.eos_token_id

        generated_ids_ext = model.generate(
            inputs_ext_gen.input_ids,
            attention_mask=inputs_ext_gen.attention_mask,
            max_new_tokens=20, # 生成更长的序列
            use_cache=True,
            pad_token_id=pad_token_id_for_gen, # 明确传递 pad_token_id
            eos_token_id=model.config.eos_token_id # 也可以明确传递 eos_token_id
        )
        decoded_text_ext = tokenizer.decode(generated_ids_ext[0], skip_special_tokens=True)
        logger.info(f"生成的文本: {decoded_text_ext}")

    except Exception as e:
        logger.error(f"生成测试过程中发生错误: {e}", exc_info=True) # 打印完整的异常信息

if __name__ == "__main__":
    # 配置基本的日志记录，以便能看到 hebbian_utils 中的 logger 输出
    logging.set_verbosity_info()
    formatter = logging.StrftimeFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    logging.add_handler(logging.StreamHandler(logging.sys.stdout), formatter)

    run_external_example()