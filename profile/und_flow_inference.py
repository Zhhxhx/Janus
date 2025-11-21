import argparse
import os
import PIL.Image
import torch
import numpy as np

import torch
from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from modeling.janusflow.janusflow import add_argus_arguments, init_attn_mlp_from_args

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
	parser = add_argus_arguments(parser)
	args = parser.parse_args()
	base_attention, mlp_args = init_attn_mlp_from_args(args, 30, 24)

	# specify the path to the model
	model_path = "./models/JanusFlow-1.3B"
	vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
	tokenizer = vl_chat_processor.tokenizer

	vl_gpt = MultiModalityCausalLM.from_pretrained(
			model_path, trust_remote_code=True, base_attn=base_attention, mlp_args=mlp_args
	)
	vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

	conversation = [
			{
					"role": "User",
					"content": "<image_placeholder>\nConvert the formula into latex code.",
					"images": ["images/equation.png"],
			},
			{"role": "Assistant", "content": ""},
	]

	# load images and prepare for inputs
	pil_images = load_pil_images(conversation)
	prepare_inputs = vl_chat_processor(
			conversations=conversation, images=pil_images, force_batchify=True
	).to(vl_gpt.device)

	# # run image encoder to get the image embeddings
	inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

	# # run the model to get the response
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

	answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
	print(f"{prepare_inputs['sft_format'][0]}", answer)