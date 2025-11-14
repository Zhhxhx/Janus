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

from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import numpy as np
import os
import PIL.Image

# specify the path to the model
# model_path = "deepseek-ai/Janus-1.3B"
# model_path = "deepseek-ai/Janus-Pro-1B"
model_path = "./Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
  model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation_und = [
  {
    "role": "User",
    "content": "<image_placeholder>\nConvert the formula into latex code.",
    "images": ["images/equation.png"],
  },
  {"role": "Assistant", "content": ""},
]

conversation_gen = [
    {
        "role": "User",
        # "content": "Give a picture of lovely Chinese Li Hua Cat.",
        "content": "Give a picture of a young woman with long, flowing hair, wearing a vintage sundress, standing in a field of wildflowers. She has a gentle smile, and the sunlight creates a soft, ethereal glow around her. The image is styled in a watercolor painting format, with delicate brushstrokes and pastel colors.",
    },
    {"role": "Assistant", "content": ""},
]

@torch.inference_mode()
def measure_und_inference_time():
  nr_run = 10

  events = [
    [torch.cuda.Event(enable_timing=True) for _ in range(8)] for _ in range(nr_run)
  ]

  # --- Warm-up Phase ---
  print("Warming up ...")
  pil_images = load_pil_images(conversation_und)
  prepare_inputs = vl_chat_processor(
    conversations=conversation_und, images=pil_images, force_batchify=True
  ).to(vl_gpt.device)

  for _ in range(3):
    # Warm-up the whole process
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    _ = vl_gpt.language_model.generate(
      inputs_embeds=inputs_embeds,
      attention_mask=prepare_inputs.attention_mask,
      pad_token_id=tokenizer.eos_token_id,
      bos_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      max_new_tokens=512,
      do_sample=False,
      use_cache=True,
    )
  torch.cuda.synchronize()
  print("Warming up done.")

  pil_images = load_pil_images(conversation_und)
  print("Starting inference time measurement ...")
  for i in range(nr_run):
    # load images and prepare for inputs
    events[i][0].record()
    prepare_inputs = vl_chat_processor(
        conversations=conversation_und, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)
    events[i][1].record()

    events[i][2].record()
    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    events[i][3].record()

    events[i][4].record()
    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
      inputs_embeds=inputs_embeds,
      attention_mask=prepare_inputs.attention_mask,
      pad_token_id=tokenizer.eos_token_id,
      bos_token_id=tokenizer.bos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      max_new_tokens=512,
      do_sample=False,
      use_cache=True,
    )
    events[i][5].record()

    events[i][6].record()
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    events[i][7].record()
  torch.cuda.synchronize()

  for i in range(nr_run):
    print(
      f"Run {i}: Preparation {events[i][0].elapsed_time(events[i][1])/1000:.6f} s, "
      f"Image Encoder {events[i][2].elapsed_time(events[i][3])/1000:.6f} s, "
      f"LM Inference {events[i][4].elapsed_time(events[i][5])/1000:.6f} s, "
      f"Decode {events[i][6].elapsed_time(events[i][7])/1000:.6f} s"
    )
  print(f"{prepare_inputs['sft_format'][0]}", answer)

@torch.inference_mode()
def prepare_inputs_embeds_for_gen(
  vl_chat_processor: VLChatProcessor,
  conversations: List[Dict[str, str]],
  sft_format: str = "deepseek",
  system_prompt: str = "",
  parallel_size: int = 16,
  events: Optional[List[torch.cuda.Event]] = None,
):
  sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversations,
    sft_format=vl_chat_processor.sft_format,
    system_prompt=system_prompt,
  )
  prompt = sft_format + vl_chat_processor.image_start_tag
  input_ids = vl_chat_processor.tokenizer.encode(prompt)
  input_ids = torch.LongTensor(input_ids)
  tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
  if events is not None:
    events[0].record()
    # Classifier-Free Guidance: 在有条件 (cond: 有 prompt 的情况下) 和无条件 (uncond: 无 prompt 的情况下) 生成之间插值，增强条件信息对生成结果的引导作用
    for i in range(parallel_size*2):
      # CFG: 偶数行为正常 prompt，奇数行除了首尾都填充为 pad
      tokens[i, :] = input_ids
      if i % 2 != 0:
        tokens[i, 1:-1] = vl_chat_processor.pad_id
    events[1].record()

    events[2].record()
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    events[3].record()
    return inputs_embeds
  else:
    # Classifier-Free Guidance: 在有条件 (cond: 有 prompt 的情况下) 和无条件 (uncond: 无 prompt 的情况下) 生成之间插值，增强条件信息对生成结果的引导作用
    for i in range(parallel_size*2):
      # CFG: 偶数行为正常 prompt，奇数行除了首尾都填充为 pad
      tokens[i, :] = input_ids
      if i % 2 != 0:
        tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    return inputs_embeds
@torch.inference_mode()
def generate(
  mmgpt: MultiModalityCausalLM,
  # vl_chat_processor: VLChatProcessor,
  # prompt: str,
  inputs_embeds: torch.Tensor,
  temperature: float = 1,
  parallel_size: int = 16,
  cfg_weight: float = 5,
  image_token_num_per_image: int = 576,
  # img_size: int = 384,
  # patch_size: int = 16,
):
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

    # CFG: 使用有条件生成和无条件生成进行插值计算最终的 logits
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

  return generated_tokens

@torch.inference_mode()
def measure_gen_inference_time():
  nr_run = 10
  parallel_size: int = 16
  img_size: int = 384 
  patch_size: int = 16

  events = [
    [torch.cuda.Event(enable_timing=True) for _ in range(10)] for _ in range(nr_run)
  ]

  # --- Warm-up Phase ---
  print("Warming up ...")
  inputs_embeds = prepare_inputs_embeds_for_gen(
    vl_chat_processor=vl_chat_processor,
    conversations=conversation_gen,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
    parallel_size=parallel_size,
  ).to(vl_gpt.device)
  
  for _ in range(3):
    # generated_tokens = generate(vl_gpt, vl_chat_processor, prompt)
    generated_tokens = generate(mmgpt=vl_gpt, inputs_embeds=inputs_embeds)
    _ = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
  print("Warming up done.")

  visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
  print("Starting inference time measurement ...")
  for i in range(nr_run):
    events[i][0].record()
    inputs_embeds = prepare_inputs_embeds_for_gen(
      vl_chat_processor=vl_chat_processor,
      conversations=conversation_gen,
      sft_format=vl_chat_processor.sft_format,
      system_prompt="",
      parallel_size=parallel_size,
      events=events[i][6:],
    ).to(vl_gpt.device)
    events[i][1].record()

    events[i][2].record()
    generated_tokens = generate(mmgpt=vl_gpt, inputs_embeds=inputs_embeds)
    events[i][3].record()

    events[i][4].record()
    dec = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    events[i][5].record()

    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    # visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
  torch.cuda.synchronize()

  for i in range(nr_run):
    print(
      f"Run {i}: Preparation {events[i][0].elapsed_time(events[i][1])/1000:.6f} s, "
      f"LM Inference {events[i][2].elapsed_time(events[i][3])/1000:.6f} s, "
      f"Decode {events[i][4].elapsed_time(events[i][5])/1000:.6f} s, "
      f"Prepare {events[i][6].elapsed_time(events[i][7])/1000:.6f} s, "
      f"Embedding {events[i][8].elapsed_time(events[i][9])/1000:.6f} s"
    )

  os.makedirs('generated_samples', exist_ok=True)
  for i in range(parallel_size):
    save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
    PIL.Image.fromarray(visual_img[i]).save(save_path)
  
  
if __name__ == "__main__":
  measure_und_inference_time()
  # measure_gen_inference_time()

  