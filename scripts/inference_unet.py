import argparse
import os
import torch
from pathlib import Path
import time
import pickle
import torchaudio
from torchvision.utils import save_image
from tqdm import tqdm
from diffusers import AudioDiffusionPipeline
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel, UNet2DModel, Mel)
from accelerate import Accelerator

def main(args):
    accelerator = Accelerator()

    model_path = args.pretrained_model_path
    output_path = os.path.join(model_path, 'samples')
    tag = f'sch_{args.scheduler}_nisteps_{args.num_inference_steps}'
    images_path = os.path.join(output_path, 'images', tag)
    audios_path = os.path.join(output_path, 'audio', tag)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(audios_path, exist_ok=True)

    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    
    pipeline = AudioDiffusionPipeline.from_pretrained(model_path)
    original_mel = pipeline.mel
    mel = Mel(
        x_res=original_mel.x_res,
        y_res=original_mel.y_res,
        hop_length=256,
        sample_rate=22050,
        n_fft=1024,
        n_iter=args.n_iter
    )
    model = pipeline.unet
    vqvae = accelerator.prepare(pipeline.vqvae) if hasattr(pipeline, "vqvae") else None
    
    # Load encodings if provided
    encodings = None
    if args.encodings is not None:
        with open(args.encodings, "rb") as f:
            encodings = pickle.load(f)
        if isinstance(encodings, dict):
            encodings = {k: torch.tensor(v).to(accelerator.device) for k, v in encodings.items()}
        else:
            # Handle the case where encodings is not a dictionary
            encodings = torch.tensor(encodings).to(accelerator.device)
     
    # Prepare the model for generation
    model = pipeline.unet
    model = accelerator.prepare(model)
    model.eval()
        
    # Calculate number of batches
    num_batches = (args.num_images + args.eval_batch_size - 1) // args.eval_batch_size

    # Generate and save images and audio
    generated_count = 0
    for batch_idx in tqdm(range(num_batches), desc=f'Saving images and audio into folder: {output_path}'):
        batch_size = min(args.eval_batch_size, args.num_images - generated_count)
        
        with torch.no_grad():
            # If encodings are used, sample them according to the batch size
            encoding_sample = None
            if encodings is not None:
                if isinstance(encodings, dict):
                    encoding_sample = torch.stack(list(encodings.values())[:batch_size])
                else:
                    encoding_sample = encodings[:batch_size]
            
            images, (sample_rate, audios) = pipeline(
                generator=generator, 
                batch_size=batch_size,
                return_dict=False, 
                steps=args.num_inference_steps,
                encoding=encoding_sample, 
                eta=0 if args.scheduler == 'ddpm' else 1
            )

        for i in range(batch_size):
            image_index = generated_count + i + 1
            image_path = os.path.join(images_path, f'image_{image_index}.png')
            audio_path = os.path.join(audios_path, f'audio_{image_index}.wav')

            # Save image
            images[i].save(image_path)

            # Save audio
            audio_tensor = torch.tensor(audios[i]).unsqueeze(0)  # Convert numpy array to tensor and add batch dimension
            torchaudio.save(audio_path, audio_tensor, sample_rate)

        generated_count += batch_size

    print(f"All images and audios have been generated and saved to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a trained model.")
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--n_iter", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--scheduler", type=str, default='ddim', choices=['ddim', 'ddpm'])
    parser.add_argument("--vae", type=str, default=None, help="Path to a pretrained VAE model.")
    parser.add_argument("--encodings", type=str, default=None, help="Path to pickle file containing encodings.")

    args = parser.parse_args()
    main(args)
