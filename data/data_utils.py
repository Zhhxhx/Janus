import math
import random
from PIL import Image

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks
def pil_img2rgb(image):
	if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
			image = image.convert("RGBA")
			white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
			white.paste(image, mask=image.split()[3])
			image = white
	else:
			image = image.convert("RGB")

	return image