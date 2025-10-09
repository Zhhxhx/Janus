# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image

# specify the path to the model
# model_path = "deepseek-ai/Janus-1.3B"
model_path = "./Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# conversation = [
#     {
#         "role": "User",
#         "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
#     },
#     {"role": "Assistant", "content": ""},
# ]

conversation = [
    {
        "role": "User",
        "content": "Give a picture of a face of a beautiful girl.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    # Classifier-Free Guidance: 在有条件 (cond: 有 prompt 的情况下) 和无条件 (uncond: 无 prompt 的情况下) 生成之间插值，增强条件信息对生成结果的引导作用
    for i in range(parallel_size*2):
        # CFG: 偶数行为正常 prompt，奇数行除了首尾都填充为 pad
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):
        # 使用 input_embeds 进行 inference
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        # 获得最后一个 hidden layer 的结果 [2*batch_size, seq_len, hidden_dim]
        hidden_states = outputs.last_hidden_state
        
        # 取最后一个 hidden_layer 的结果中每个 batch 中的最后一个 token，送入 gen_head 模块: [2*batch_size, 1, hidden_dim] -> [2*batch_size, 1, vocab_size]
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :] # 取偶数行: 有条件生成
        logit_uncond = logits[1::2, :] # 取奇数行: 无条件生成
        
        # CFG: 使用有条件生成和无条件生成进行插值计算最终的 logit
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        # 计算每个 vocab entry 对应的概率
        probs = torch.softmax(logits / temperature, dim=-1)

        # 根据上面计算的概率采样生成 token id，next_token [batch_size, 1]
        next_token = torch.multinomial(probs, num_samples=1)
        # print(next_token[0])
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        
        # 把每个 batch 的 token 复制一份（扩展为 cond/uncond 两份），保证下一轮自回归的输入和 CFG 保持一致
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        # 把 token id 映射到 embedding 上，并作为下一轮的 input
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    print(generated_tokens.shape)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    print(mmgpt.gen_vision_model.quantize.embedding.weight.shape)
    print(mmgpt.gen_vision_model.quantize.embedding.weight[:5])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)


generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
)

# 检查 checkpoint 是否有对应键并比较
import torch
ckpt = torch.load("./Janus-Pro-1B/pytorch_model.bin", map_location="cpu")
sd = ckpt.get("state_dict", ckpt)
key = "gen_vision_model.quantize.embedding.weight"
print("key in ckpt:", key in sd)
if key in sd:
    print("ckpt shape, dtype:", sd[key].shape, sd[key].dtype)
    # 假设 model 已经实例化或已从_pretrained 加载
    emb = vl_gpt.gen_vision_model.quantize.embedding.weight
    print("model shape, dtype:", tuple(emb.shape), emb.dtype)
    ck = sd[key].to(device=emb.device, dtype=emb.dtype)
    print("max abs diff (before load):", (ck - emb.detach()).abs().max().item())
# 尝试加载并查看返回信息
res = vl_gpt.load_state_dict(sd, strict=False)
print(res)  # 会显示 missing_keys / unexpected_keys