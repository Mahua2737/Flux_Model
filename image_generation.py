import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt


pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)  # pre-trained Stable Diffusion model from Hugging Face
pipe.to(
    "cuda"
)  # if GPU is there we can use it otherwise we can run the same thing is colab to generate the images and use that for preprocessing


prompt = "A vibrant forest landscape with colorful trees and a bright sky"


num_images = 3
generated_images = []

for i in range(num_images):
    with torch.no_grad():
        image = pipe(prompt).images[0]
        generated_images.append(image)
        image.save(f"generated_forest_image_{i+1}.png")


fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

for i, img in enumerate(generated_images):
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Image {i+1}")

plt.show()
