# coding=utf-8
# hebbian_utils.py

import torch
from torch import nn
from typing import List, Dict, Tuple, Optional, Any
from transformers.utils import logging # 从 transformers 导入 logging

logger = logging.get_logger(__name__)

# 全局或类属性来存储激活和 hook 句柄
# 使用字典，键是模块的唯一标识符（例如其在模型中的名称）
# 值是 (h1, h2) 或更复杂的结构
_hebbian_activations_buffer: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
_hebbian_hook_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
_current_attention_mask_for_hebbian_update: Optional[torch.Tensor] = None

def _hebbian_hook_fn_global(module: nn.Module, input_t: Tuple[torch.Tensor, ...], output_t: torch.Tensor):
    """
    全局的 forward hook 函数，用于捕获激活。
    它需要知道将激活存储到 _hebbian_activations_buffer 中的哪个键下。
    我们可以在注册 hook 时使用 functools.partial 来传递模块的名称。
    或者，我们可以通过 module 对象本身找到它的名称（但这可能比较复杂且不可靠）。
    一个更简单的方法是假设 module.__hebbian_id__ 已经被设置。
    """
    module_id = getattr(module, "__hebbian_id__", None)
    if module_id is None:
        logger.error("模块没有 __hebbian_id__ 属性，无法存储激活。")
        return

    h1 = input_t[0].detach().clone()
    h2 = output_t.detach().clone()

    if h1.ndim == 2:
        h1 = h1.unsqueeze(1)
    if h2.ndim == 2:
        h2 = h2.unsqueeze(1)

    if module_id not in _hebbian_activations_buffer:
        _hebbian_activations_buffer[module_id] = []
    _hebbian_activations_buffer[module_id].append((h1, h2)) # 支持一个模块被多次调用（例如在循环中）

def enable_hebbian_hooks_on_model(model: nn.Module, target_module_names: List[str]):
    """
    在给定模型的指定目标模块上启用 Hebbian forward hooks。

    Args:
        model (nn.Module): 要操作的 PyTorch 模型。
        target_module_names (List[str]): 需要应用 Hebbian 更新的模块名称列表。
                                         这些模块应该是 nn.Linear 类型。
    """
    disable_hebbian_hooks_on_model(model) # 清理旧的 hooks 和缓冲区

    for name, module in model.named_modules():
        if name in target_module_names:
            if not isinstance(module, nn.Linear):
                logger.warning(f"Hebbian 目标模块 {name} 不是 nn.Linear 类型。将跳过此模块。")
                continue

            # 为模块设置一个唯一ID，以便 hook 函数可以识别它
            # 这里我们简单地使用模块的名称作为ID
            setattr(module, "__hebbian_id__", name)
            
            handle = module.register_forward_hook(_hebbian_hook_fn_global)
            _hebbian_hook_handles[name] = handle
            _hebbian_activations_buffer[name] = [] # 为这个模块初始化激活列表
            logger.info(f"已为模块 {name} 启用 Hebbian hook。")

    if not _hebbian_hook_handles:
        logger.warning("未找到有效的目标模块，或者未指定任何目标模块。Hebbian hook 未被激活。")

def disable_hebbian_hooks_on_model(model: nn.Module):
    """
    移除模型上所有已注册的 Hebbian forward hooks 并清空缓冲区。

    Args:
        model (nn.Module): 之前被启用 Hebbian hooks 的模型。
    """
    global _current_attention_mask_for_hebbian_update
    for name, handle in _hebbian_hook_handles.items():
        handle.remove()
        # 清理我们添加的属性（可选，但保持整洁）
        module = model.get_submodule(name) # 获取模块
        if hasattr(module, "__hebbian_id__"):
            delattr(module, "__hebbian_id__")

    _hebbian_hook_handles.clear()
    _hebbian_activations_buffer.clear()
    _current_attention_mask_for_hebbian_update = None
    logger.info("已禁用所有 Hebbian hooks 并清空相关缓冲区。")

def clear_hebbian_activations():
    """
    仅清空激活缓冲区。这在每次 hebbian_update_weights 调用后执行。
    """
    for key in _hebbian_activations_buffer:
        _hebbian_activations_buffer[key] = []

def store_attention_mask_for_hebbian(attention_mask: Optional[torch.Tensor]):
    """
    存储当前前向传播的 attention_mask，供 Hebbian 更新使用。

    Args:
        attention_mask (Optional[torch.Tensor]): 当前的注意力掩码。
    """
    global _current_attention_mask_for_hebbian_update
    if attention_mask is not None:
        _current_attention_mask_for_hebbian_update = attention_mask.detach().clone()
    else:
        _current_attention_mask_for_hebbian_update = None

@torch.no_grad()
def hebbian_update_weights(
    model: nn.Module,
    target_module_names: List[str],
    reward: float,
    learning_rate: float
):
    """
    对模型的目标模块应用 Hebbian 权重更新。
    这个函数应该在模型的前向传播之后，并且在 store_attention_mask_for_hebbian (如果需要) 调用之后执行。

    Args:
        model (nn.Module): 要更新权重的模型。
        target_module_names (List[str]): 之前通过 enable_hebbian_hooks_on_model 指定的目标模块名称列表。
        reward (float): 一个标量奖励信号。
        learning_rate (float): Hebbian 更新的学习率。
    """
    global _current_attention_mask_for_hebbian_update

    if not _hebbian_activations_buffer:
        logger.debug("Hebbian 激活缓冲区为空。不执行权重更新。")
        return

    for module_name in target_module_names:
        if module_name not in _hebbian_activations_buffer or not _hebbian_activations_buffer[module_name]:
            # logger.debug(f"模块 {module_name} 没有捕获到激活。跳过更新。")
            continue

        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            logger.error(f"在模型中未找到名为 {module_name} 的子模块。跳过更新。")
            continue

        if not isinstance(module, nn.Linear) or not hasattr(module, "weight") or module.weight is None:
            logger.warning(f"模块 {module_name} 不是带有权重的 nn.Linear。跳过更新。")
            continue
        
        # 通常一个模块的hook只被调用一次（除非模型结构复杂，例如共享权重或循环）
        # 如果一个模块被多次调用（例如，在transformer的每一层都用同一个线性层），
        # 这里的逻辑需要决定是累积更新还是只用最后一次的激活。
        # 当前 _hebbian_hook_fn_global 会将所有调用都加入列表。
        # 为简单起见，我们假设对于每个目标模块，我们只关心最近一次（或唯一一次）捕获的激活。
        # 或者，我们可以迭代所有捕获到的激活对并分别应用更新或累积 delta_w。
        # 这里我们假设只取最后一次捕获的激活对。
        # 如果要处理多次调用，需要修改逻辑，例如累加 delta_w_avg。

        # 获取该模块最后一次（或唯一一次）捕获的激活
        # 如果一个模块在前向传播中被多次调用 (例如，在循环中或作为共享层)，
        # _hebbian_activations_buffer[module_name] 会是一个列表。
        # 这里我们简单地处理列表中的每一个激活对。
        
        accumulated_delta_w_avg = torch.zeros_like(module.weight.data)
        total_active_elements_for_module = 0

        for h1_stored, h2_stored in _hebbian_activations_buffer[module_name]:
            # h1_stored: (batch, seq_len, in_features)
            # h2_stored: (batch, seq_len, out_features)

            current_mask = _current_attention_mask_for_hebbian_update # 使用全局存储的mask

            num_active_elements_this_pair = 0

            if current_mask is not None:
                if current_mask.shape[0] != h1_stored.shape[0] or \
                   current_mask.shape[1] != h1_stored.shape[1]:
                    logger.warning(f"模块 {module_name} 的 Attention mask 形状与激活形状不兼容。忽略 mask。")
                    current_mask = None

            if current_mask is not None:
                mask_expanded = current_mask.unsqueeze(-1).to(h1_stored.dtype)
                h1_masked = h1_stored * mask_expanded
                h2_masked = h2_stored * mask_expanded
                delta_w_sum_this_pair = torch.einsum('bso,bsi->oi', h2_masked, h1_masked)
                
                num_active_elements_this_pair = current_mask.sum().item()
                if num_active_elements_this_pair == 0:
                    delta_w_avg_this_pair = torch.zeros_like(module.weight.data)
                else:
                    delta_w_avg_this_pair = delta_w_sum_this_pair / num_active_elements_this_pair
            else:
                delta_w_sum_this_pair = torch.einsum('bso,bsi->oi', h2_stored, h1_stored)
                num_elements_this_pair = h1_stored.size(0) * h1_stored.size(1)
                if num_elements_this_pair == 0:
                    delta_w_avg_this_pair = torch.zeros_like(module.weight.data)
                else:
                    delta_w_avg_this_pair = delta_w_sum_this_pair / num_elements_this_pair
                num_active_elements_this_pair = num_elements_this_pair # 在没有mask的情况下，所有元素都活跃

            accumulated_delta_w_avg.add_(delta_w_avg_this_pair)
            total_active_elements_for_module += num_active_elements_this_pair # 或者用 len(_hebbian_activations_buffer[module_name]) 来平均

        if total_active_elements_for_module > 0 : # 或者用 len(_hebbian_activations_buffer[module_name]) > 0
            # 如果有多个 (h1,h2) 对被捕获（例如模块在序列中被多次应用），
            # 我们需要决定如何平均。这里简单地对所有对的 delta_w_avg 求和，
            # 然后可以除以捕获对的数量 (len(_hebbian_activations_buffer[module_name]))
            # 或者，如果每个 delta_w_avg_this_pair 已经是平均值，那么这里的累加可能是正确的，
            # 然后直接应用这个累加值（如果 reward 和 lr 应该作用于每个子步骤的平均）。
            # 目前的实现是：每个 (h1,h2) 对计算自己的 delta_w_avg，然后将这些 delta_w_avg 加起来。
            # 这可能需要调整，取决于你希望如何处理模块的多次调用。
            # 一个简单的处理是，假设我们只关心最后一次调用，或者对所有调用的平均delta_W应用一次更新。
            # 为了简单，我们假设上面的 accumulated_delta_w_avg 是所有 (h1,h2) 对贡献的 *总和*，并且每个都已平均。
            # 因此，我们可以直接将这个总和（或其平均值，如果需要）应用到权重上。
            # 如果要对所有捕获的 delta_w_avg 再做一次平均：
            # final_delta_w_to_apply = accumulated_delta_w_avg / len(_hebbian_activations_buffer[module_name])

            # 当前 accumulated_delta_w_avg 是每个 (h1,h2) 对的 delta_w_avg 的和。
            # 如果希望 reward 和 lr 作用于这个总和，那么直接用。
            # 如果希望 reward 和 lr 作用于所有 (h1,h2) 对的平均贡献，则需要再除以对的数量。
            # 让我们假设 accumulated_delta_w_avg 就是我们想要施加的“平均”更新（可能已在内部平均过）。
            final_delta_w_to_apply = accumulated_delta_w_avg
            if len(_hebbian_activations_buffer[module_name]) > 1: # 如果模块被多次调用，对累积的平均值再取平均
                final_delta_w_to_apply = final_delta_w_to_apply / len(_hebbian_activations_buffer[module_name])


            module.weight.data.add_(final_delta_w_to_apply, alpha=(learning_rate * reward))
            logger.debug(f"模块 {module_name} 的权重已通过 Hebbian 规则更新。")

    # 在所有模块更新后，清空激活缓冲区，为下一次前向传播做准备
    clear_hebbian_activations()
    # 通常也在这里清除 attention_mask，因为它与特定的前向传播相关
    # store_attention_mask_for_hebbian(None) # 或者让调用者在需要时清除