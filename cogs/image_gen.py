# cogs/image_gen.py
import io
import asyncio
import replicate
import discord
from discord.ext import commands
from discord import File
from state import user_generated_images  
from utils.prompt_manager import get_prompt_by_index
from utils.image_manager import add_images, get_image_by_index, list_images

class GenerationCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def flux(self, ctx, *args):
        """
        Generate images using the Flux Schnell model with either a stored prompt or a direct prompt.
        
        Usage examples:
          - Using a stored prompt: !flux prompt[1] 3
          - Using a direct prompt: !flux an astronaut riding a horse 3
        (The number indicates the number of outputs.)
        """
        stored_prompt = None
        direct_prompt_parts = []
        num_outputs = 1

        # Parse arguments for prompt[<index>], digits for number of outputs, or direct text.
        for arg in args:
            if arg.startswith("prompt[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("prompt["):-1])
                    stored_prompt = get_prompt_by_index(ctx.author.id, idx)
                    if not stored_prompt:
                        await ctx.send(f"No stored prompt found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid prompt index format.")
                    return
            elif arg.isdigit():
                num_outputs = int(arg)
            else:
                direct_prompt_parts.append(arg)
        
        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt either as a stored prompt (prompt[<index>]) or as direct text.")
            return
        
        num_outputs = max(1, min(num_outputs, 4))
        msg = await ctx.send(f"Generating {num_outputs} image(s) for prompt:\n> {prompt}")

        try:
            output = await asyncio.to_thread(
                replicate.run,
                "black-forest-labs/flux-schnell",
                input={"prompt": prompt, "num_outputs": num_outputs}
            )
        except Exception as e:
            await msg.edit(content=f"Flux generation failed: {e}")
            return

        generated_urls = []
        for idx, img_file in enumerate(output, start=1):
            image_bytes = img_file.read()
            file_data = io.BytesIO(image_bytes)
            sent = await ctx.send(
                content=f"> **Image {idx}** for prompt:\n> {prompt}",
                file=File(file_data, f"flux_{idx}.png")
            )
            if sent.attachments:
                generated_urls.append(sent.attachments[0].url)

        add_images(ctx.author.id, generated_urls)
        await msg.delete()

    @commands.command()
    async def listimages(self, ctx):
        """
        List all stored images (with their indexes) for the user.
        Usage: !listimages
        """
        images = list_images(ctx.author.id)
        if not images:
            await ctx.send("You have no stored images.")
            return

        message = "**Your Stored Images:**\n"
        for idx, url in images:
            message += f"**{idx}**: {url}\n"
        await ctx.send(message)

    @commands.command()
    async def redux(self, ctx, index_str: str = "1", aspect_ratio: str = "1:1"):
        """
        Take a previously generated image and run it through the Flux Redux model.
        Usage: !redux <image_index> [aspect_ratio]
        Example: !redux 2 3:2  (uses image at index 2 with aspect ratio 3:2)
        """
        try:
            index = int(index_str)
        except ValueError:
            await ctx.send("Invalid image index. Please provide a number.")
            return
    
        redux_image_url = get_image_by_index(ctx.author.id, index)
        if not redux_image_url:
            await ctx.send(f"Invalid image index {index}. Use `!listimages` to see your stored images.")
            return
    
        msg = await ctx.send(f"Running Redux on image #{index} with aspect_ratio={aspect_ratio}...")
    
        redux_input = {
            "redux_image": redux_image_url,
            "aspect_ratio": aspect_ratio,
        }
    
        try:
            redux_output = await asyncio.to_thread(
                replicate.run,
                "black-forest-labs/flux-redux-dev",
                input=redux_input
            )
        except Exception as e:
            await msg.edit(content=f"Flux Redux generation failed: {e}")
            return
    
        generated_urls = []
        for i, file_like in enumerate(redux_output, start=1):
            redux_bytes = file_like.read()
            redux_data = io.BytesIO(redux_bytes)
            sent = await ctx.send(
                content=f"Redux output {i} from image #{index}",
                file=File(redux_data, f"redux_output_{i}.webp")
            )
            if sent.attachments:
                generated_urls.append(sent.attachments[0].url)
    
        add_images(ctx.author.id, generated_urls)
        await msg.delete()

    @commands.command()
    async def stable35(self, ctx, *args):
        """
        Generate images using the 'stability-ai/stable-diffusion-3.5-large' model with either a stored prompt or a direct prompt.
        
        Usage examples:
          - Using a stored prompt and a stored image: !stable35 prompt[1] image[2]
          - Using a stored prompt only: !stable35 prompt[1]
          - Using a direct prompt with a stored image: !stable35 A landscape at sunset image[2]
          - Using a direct prompt only: !stable35 A landscape at sunset
        """
        stored_prompt = None
        direct_prompt_parts = []
        input_image_url = None

        for arg in args:
            if arg.startswith("prompt[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("prompt["):-1])
                    stored_prompt = get_prompt_by_index(ctx.author.id, idx)
                    if not stored_prompt:
                        await ctx.send(f"No stored prompt found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid prompt index format.")
                    return
            elif arg.startswith("image[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("image["):-1])
                    input_image_url = get_image_by_index(ctx.author.id, idx)
                    if not input_image_url:
                        await ctx.send(f"No stored image found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid image index format.")
                    return
            else:
                direct_prompt_parts.append(arg)

        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt either as a stored prompt (prompt[<index>]) or as direct text.")
            return

        cfg = 3.5
        steps = 28
        aspect_ratio = "1:1"
        output_format = "webp"
        output_quality = 90
        prompt_strength = 0.85

        msg_text = f"**Stable Diffusion 3.5** generation in progress...\nPrompt: `{prompt}`"
        if input_image_url:
            msg_text += f"\nUsing image as a starting point."
        msg = await ctx.send(msg_text)

        sd_input = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "cfg": cfg,
            "steps": steps,
            "output_format": output_format,
            "output_quality": output_quality
        }
        if input_image_url:
            sd_input["image"] = input_image_url
            sd_input["prompt_strength"] = prompt_strength

        try:
            output_files = await asyncio.to_thread(
                replicate.run,
                "stability-ai/stable-diffusion-3.5-large",
                input=sd_input
            )
        except Exception as e:
            await msg.edit(content=f"Stable Diffusion 3.5 generation failed: {e}")
            return

        generated_urls = []
        for i, file_like in enumerate(output_files, start=1):
            image_bytes = file_like.read()
            image_data = io.BytesIO(image_bytes)
            sent = await ctx.send(
                content=f"**Stable Diffusion 3.5 Output {i}**",
                file=File(image_data, f"sd35_output_{i}.{output_format}")
            )
            if sent.attachments:
                generated_urls.append(sent.attachments[0].url)

        add_images(ctx.author.id, generated_urls)
        await msg.delete()

    # ---- New Commands for Additional Image Models ----

    @commands.command()
    async def fluxpro(self, ctx, *args):
        """
        Generate an image using the Flux 1.1 Pro Ultra model.
        Usage: !fluxpro prompt[<index>] [image[<index>]] or direct prompt (optionally with image[<index>])
        """
        stored_prompt = None
        direct_prompt_parts = []
        input_image_url = None

        for arg in args:
            if arg.startswith("prompt[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("prompt["):-1])
                    stored_prompt = get_prompt_by_index(ctx.author.id, idx)
                    if not stored_prompt:
                        await ctx.send(f"No stored prompt found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid prompt index format.")
                    return
            elif arg.startswith("image[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("image["):-1])
                    input_image_url = get_image_by_index(ctx.author.id, idx)
                    if not input_image_url:
                        await ctx.send(f"No stored image found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid image index format.")
                    return
            else:
                direct_prompt_parts.append(arg)
        
        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt either as a stored prompt (prompt[<index>]) or as direct text.")
            return

        msg = await ctx.send(f"Generating image using Flux 1.1 Pro Ultra for prompt:\n> {prompt}")

        model_input = {
            "prompt": prompt,
            "aspect_ratio": "9:16"
        }
        if input_image_url:
            model_input["image_prompt"] = input_image_url

        try:
            output = await asyncio.to_thread(
                replicate.run,
                "black-forest-labs/flux-1.1-pro-ultra",
                input=model_input
            )
        except Exception as e:
            await msg.edit(content=f"Flux Pro generation failed: {e}")
            return

        try:
            image_bytes = output.read()
        except Exception as e:
            await msg.edit(content=f"Error reading output: {e}")
            return

        file_data = io.BytesIO(image_bytes)
        sent = await ctx.send(
            content=f"> **Flux Pro Output** for prompt:\n> {prompt}",
            file=File(file_data, "fluxpro_output.jpg")
        )
        if sent.attachments:
            add_images(ctx.author.id, [sent.attachments[0].url])
        await msg.delete()

    @commands.command()
    async def sdxl(self, ctx, *args):
        """
        Generate an image using the SDXL model.
        Usage: !sdxl prompt[<index>] (or direct prompt)
        """
        stored_prompt = None
        direct_prompt_parts = []
        # SDXL does not support image input in our example
        for arg in args:
            if arg.startswith("prompt[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("prompt["):-1])
                    stored_prompt = get_prompt_by_index(ctx.author.id, idx)
                    if not stored_prompt:
                        await ctx.send(f"No stored prompt found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid prompt index format.")
                    return
            else:
                direct_prompt_parts.append(arg)
        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt.")
            return

        msg = await ctx.send(f"Generating image using SDXL for prompt:\n> {prompt}")

        model_input = {
            "width": 768,
            "height": 768,
            "prompt": prompt,
            "refine": "expert_ensemble_refiner",
            "apply_watermark": False,
            "num_inference_steps": 25
        }
        try:
            output = await asyncio.to_thread(
                replicate.run,
                "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                input=model_input
            )
        except Exception as e:
            await msg.edit(content=f"SDXL generation failed: {e}")
            return

        generated_urls = []
        for i, file_like in enumerate(output, start=1):
            image_bytes = file_like.read()
            image_data = io.BytesIO(image_bytes)
            sent = await ctx.send(
                content=f"**SDXL Output {i}** for prompt:\n> {prompt}",
                file=File(image_data, f"sdxl_output_{i}.png")
            )
            if sent.attachments:
                generated_urls.append(sent.attachments[0].url)
        add_images(ctx.author.id, generated_urls)
        await msg.delete()

    @commands.command()
    async def imagen(self, ctx, *args):
        """
        Generate an image using Google Imagen 3.
        Usage: !imagen prompt[<index>] (or direct prompt)
        """
        stored_prompt = None
        direct_prompt_parts = []
        for arg in args:
            if arg.startswith("prompt[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("prompt["):-1])
                    stored_prompt = get_prompt_by_index(ctx.author.id, idx)
                    if not stored_prompt:
                        await ctx.send(f"No stored prompt found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid prompt index format.")
                    return
            else:
                direct_prompt_parts.append(arg)
        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt.")
            return

        msg = await ctx.send(f"Generating image using Imagen 3 for prompt:\n> {prompt}")

        model_input = {
            "prompt": prompt,
            "aspect_ratio": "1:1",
            "negative_prompt": "",
            "safety_filter_level": "block_medium_and_above"
        }
        try:
            output = await asyncio.to_thread(
                replicate.run,
                "google/imagen-3",
                input=model_input
            )
        except Exception as e:
            await msg.edit(content=f"Imagen generation failed: {e}")
            return

        try:
            image_bytes = output.read()
        except Exception as e:
            await msg.edit(content=f"Error reading Imagen output: {e}")
            return

        file_data = io.BytesIO(image_bytes)
        sent = await ctx.send(
            content=f"**Imagen 3 Output** for prompt:\n> {prompt}",
            file=File(file_data, "imagen_output.png")
        )
        if sent.attachments:
            add_images(ctx.author.id, [sent.attachments[0].url])
        await msg.delete()

    @commands.command()
    async def recraftv3(self, ctx, *args):
        """
        Generate an image using Recraft V3.
        Usage: !recraftv3 prompt[<index>] (or direct prompt)
        """
        stored_prompt = None
        direct_prompt_parts = []
        for arg in args:
            if arg.startswith("prompt[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("prompt["):-1])
                    stored_prompt = get_prompt_by_index(ctx.author.id, idx)
                    if not stored_prompt:
                        await ctx.send(f"No stored prompt found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid prompt index format.")
                    return
            else:
                direct_prompt_parts.append(arg)
        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt.")
            return

        msg = await ctx.send(f"Generating image using Recraft V3 for prompt:\n> {prompt}")

        model_input = {
            "prompt": prompt,
            "size": "1365x1024"
        }
        try:
            output = await asyncio.to_thread(
                replicate.run,
                "recraft-ai/recraft-v3",
                input=model_input
            )
        except Exception as e:
            await msg.edit(content=f"Recraft V3 generation failed: {e}")
            return

        try:
            image_bytes = output.read()
        except Exception as e:
            await msg.edit(content=f"Error reading Recraft V3 output: {e}")
            return

        file_data = io.BytesIO(image_bytes)
        sent = await ctx.send(
            content=f"**Recraft V3 Output** for prompt:\n> {prompt}",
            file=File(file_data, "recraftv3_output.webp")
        )
        if sent.attachments:
            add_images(ctx.author.id, [sent.attachments[0].url])
        await msg.delete()

    @commands.command()
    async def playground(self, ctx, *args):
        """
        Generate images using Playground V2.5 Aesthetic.
        Usage: !playground prompt[<index>] (or direct prompt) and optionally image[<index>]
        """
        stored_prompt = None
        direct_prompt_parts = []
        input_image_url = None

        for arg in args:
            if arg.startswith("prompt[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("prompt["):-1])
                    stored_prompt = get_prompt_by_index(ctx.author.id, idx)
                    if not stored_prompt:
                        await ctx.send(f"No stored prompt found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid prompt index format.")
                    return
            elif arg.startswith("image[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("image["):-1])
                    input_image_url = get_image_by_index(ctx.author.id, idx)
                    if not input_image_url:
                        await ctx.send(f"No stored image found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid image index format.")
                    return
            else:
                direct_prompt_parts.append(arg)

        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt.")
            return

        msg = await ctx.send(f"Generating image using Playground V2.5 Aesthetic for prompt:\n> {prompt}")

        model_input = {
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "scheduler": "DPMSolver++",
            "num_outputs": 1,
            "guidance_scale": 3,
            "apply_watermark": True,
            "negative_prompt": "ugly, deformed, noisy, blurry, distorted",
            "prompt_strength": 0.8,
            "num_inference_steps": 25,
            "disable_safety_checker": False
        }
        if input_image_url:
            model_input["image"] = input_image_url

        try:
            output = await asyncio.to_thread(
                replicate.run,
                "playgroundai/playground-v2.5-1024px-aesthetic:a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
                input=model_input
            )
        except Exception as e:
            await msg.edit(content=f"Playground generation failed: {e}")
            return

        generated_urls = []
        for i, file_like in enumerate(output, start=1):
            image_bytes = file_like.read()
            image_data = io.BytesIO(image_bytes)
            sent = await ctx.send(
                content=f"**Playground V2.5 Aesthetic Output {i}** for prompt:\n> {prompt}",
                file=File(image_data, f"playground_output_{i}.png")
            )
            if sent.attachments:
                generated_urls.append(sent.attachments[0].url)
        add_images(ctx.author.id, generated_urls)
        await msg.delete()
        
    @commands.command()
    async def multigen(self, ctx, *args):
        """
        Generate images using multiple models concurrently for the same prompt.
        
        Usage examples:
          - Using a stored prompt: !multigen prompt[1]
          - Using a direct prompt: !multigen A scenic landscape at sunset
          - Optionally, include an image: !multigen prompt[1] image[2]
        
        This command calls a set of predefined models and returns all outputs.
        """
        stored_prompt = None
        direct_prompt_parts = []
        input_image_url = None

        # Parse arguments for prompt[<index>], image[<index>], or direct text.
        for arg in args:
            if arg.startswith("prompt[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("prompt["):-1])
                    stored_prompt = get_prompt_by_index(ctx.author.id, idx)
                    if not stored_prompt:
                        await ctx.send(f"No stored prompt found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid prompt index format.")
                    return
            elif arg.startswith("image[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("image["):-1])
                    input_image_url = get_image_by_index(ctx.author.id, idx)
                    if not input_image_url:
                        await ctx.send(f"No stored image found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid image index format.")
                    return
            else:
                direct_prompt_parts.append(arg)
        
        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt either as a stored prompt (prompt[<index>]) or as direct text.")
            return

        msg = await ctx.send(f"Generating images concurrently for prompt:\n> {prompt}")

        # Define the models to use and a lambda to generate their input dictionaries.
        models = {
            
            "stable35": {
                "replicate_id": "stability-ai/stable-diffusion-3.5-large",
                "input": lambda prompt, image: {
                    "prompt": prompt,
                    "aspect_ratio": "1:1",
                    "cfg": 3.5,
                    "steps": 28,
                    "output_format": "webp",
                    "output_quality": 90,
                    **({"image": image, "prompt_strength": 0.85} if image else {})
                }
            },
            "sdxl": {
                "replicate_id": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                "input": lambda prompt, image: {
                    "width": 768,
                    "height": 768,
                    "prompt": prompt,
                    "refine": "expert_ensemble_refiner",
                    "apply_watermark": False,
                    "num_inference_steps": 25
                }
            },
            "imagen": {
                "replicate_id": "google/imagen-3",
                "input": lambda prompt, image: {
                    "prompt": prompt,
                    "aspect_ratio": "1:1",
                    "negative_prompt": "",
                    "safety_filter_level": "block_medium_and_above"
                }
            },
            "recraftv3": {
                "replicate_id": "recraft-ai/recraft-v3",
                "input": lambda prompt, image: {
                    "prompt": prompt,
                    "size": "1365x1024"
                }
            },
            "playground": {
                "replicate_id": "playgroundai/playground-v2.5-1024px-aesthetic:a45f82a1382bed5c7aeb861dac7c7d191b0fdf74d8d57c4a0e6ed7d4d0bf7d24",
                "input": lambda prompt, image: {
                    "prompt": prompt,
                    "width": 1024,
                    "height": 1024,
                    "scheduler": "DPMSolver++",
                    "num_outputs": 1,
                    "guidance_scale": 3,
                    "apply_watermark": True,
                    "negative_prompt": "ugly, deformed, noisy, blurry, distorted",
                    "prompt_strength": 0.8,
                    "num_inference_steps": 25,
                    "disable_safety_checker": False,
                    **({"image": image} if image else {})
                }
            },
            "fluxpro": {
                "replicate_id": "black-forest-labs/flux-1.1-pro-ultra",
                "input": lambda prompt, image: {
                    "prompt": prompt,
                    "aspect_ratio": "3:2",
                    **({"image_prompt": image} if image else {})
                }
            }
        }

        # Define an async function to run each model.
        async def run_model(model_key, model_info):
            input_dict = model_info["input"](prompt, input_image_url)
            try:
                result = await asyncio.to_thread(
                    replicate.run,
                    model_info["replicate_id"],
                    input=input_dict
                )
            except Exception as e:
                return model_key, f"Error: {e}", None

            # Process the output: if it's a list, read each item; otherwise, read single output.
            outputs = []
            if isinstance(result, list):
                for item in result:
                    try:
                        outputs.append(item.read())
                    except Exception as e:
                        continue
            else:
                try:
                    outputs.append(result.read())
                except Exception as e:
                    pass
            return model_key, None, outputs

        # Create and gather tasks for each model.
        tasks = [run_model(key, info) for key, info in models.items()]
        results = await asyncio.gather(*tasks)

        all_generated_urls = []
        for model_key, error, outputs in results:
            if error:
                await ctx.send(f"**{model_key}**: {error}")
                continue
            if not outputs:
                await ctx.send(f"**{model_key}**: No output generated.")
                continue
            for idx, output_bytes in enumerate(outputs, start=1):
                file_data = io.BytesIO(output_bytes)
                # Use appropriate file extension; here we default to PNG.
                filename = f"{model_key}_output_{idx}.png"
                sent = await ctx.send(
                    content=f"**{model_key.capitalize()} Output {idx}** for prompt:\n> {prompt}",
                    file=File(file_data, filename)
                )
                if sent.attachments:
                    all_generated_urls.append(sent.attachments[0].url)
        add_images(ctx.author.id, all_generated_urls)
        await msg.delete()

async def setup(bot):
    await bot.add_cog(GenerationCog(bot))
