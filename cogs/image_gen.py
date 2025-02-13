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

        # Parse arguments: check for stored prompt markers, digits (for num_outputs), or direct text.
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
        
        # Use the stored prompt if provided; otherwise, join the direct text.
        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt either as a stored prompt (prompt[<index>]) or as direct text.")
            return
        
        # Clamp number of outputs to between 1 and 4.
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
    
        # Retrieve image URL from the image manager.
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
                "black-forest-labs/flux-redux-schnell",
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

        # Parse the arguments:
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

        # Define parameters for Stable Diffusion 3.5.
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

    @commands.command()
    async def upscale(self, ctx, image_spec: str):
        """
        Upscale a stored image using the recraft-crisp-upscale model.
        Usage: !upscale image[<index>]
        Example: !upscale image[2]
        """
        # Validate image reference format.
        if not (image_spec.startswith("image[") and image_spec.endswith("]")):
            await ctx.send("Please specify the image to upscale using the format `image[<index>]`.")
            return

        try:
            image_index = int(image_spec[len("image["):-1])
        except ValueError:
            await ctx.send("Invalid image index format. Please use `image[<index>]` (e.g., image[2]).")
            return

        image_url = get_image_by_index(ctx.author.id, image_index)
        if image_url is None:
            await ctx.send(f"No stored image found at index {image_index}.")
            return

        msg = await ctx.send(f"Upscaling image from index {image_index}...")

        try:
            output = await asyncio.to_thread(
                replicate.run,
                "recraft-ai/recraft-crisp-upscale",
                input={"image": image_url}
            )
        except Exception as e:
            await ctx.send(f"Upscale failed: {e}")
            return

        # The upscale model returns a URI (string) as output.
        upscaled_image_url = output.strip()

        add_images(ctx.author.id, [upscaled_image_url])
        await msg.edit(content=f"Upscaled image stored and available: {upscaled_image_url}")

# This is required for the bot to load the cog
async def setup(bot):
    await bot.add_cog(GenerationCog(bot))
