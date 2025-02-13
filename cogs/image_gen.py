# cogs/image_gen.py
import io
import replicate
import discord
from discord.ext import commands
from discord import File
from state import user_generated_images  
from utils.prompt_manager import get_prompt_by_index
from utils.image_manager import add_images, get_image_by_index
class GenerationCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    
    async def flux(self, ctx, *args):
        """
        Generate images using the Flux Schnell model with a stored prompt.
        Usage: !flux prompt[<index>] [number_of_outputs]
        Example: !flux prompt[1] 3
        """
        if not args or not (args[0].startswith("prompt[") and args[0].endswith("]")):
            await ctx.send("Please specify a stored prompt, e.g. `!flux prompt[1] [number_of_outputs]`.")
            return

        try:
            prompt_index = int(args[0][len("prompt["):-1])
        except ValueError:
            await ctx.send("Invalid prompt index.")
            return

        # Retrieve the stored prompt.
        prompt = get_prompt_by_index(ctx.author.id, prompt_index)
        if not prompt:
            await ctx.send(f"No stored prompt found at index {prompt_index}. Use `!fluxgpt` to generate one.")
            return

        # Determine number of outputs.
        num_outputs = 1
        if len(args) > 1 and args[1].isdigit():
            num_outputs = int(args[1])
        num_outputs = max(1, min(num_outputs, 4))

        msg = await ctx.send(f"Using stored prompt (index {prompt_index}):\n> {prompt}\n> Generating {num_outputs} images...")

        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt, "num_outputs": num_outputs}
        )

        generated_urls = []
        for idx, img_file in enumerate(output, start=1):
            image_bytes = img_file.read()
            file_data = io.BytesIO(image_bytes)
            sent = await ctx.send(
                content=f"> **Image {idx}** for stored prompt (index {prompt_index})",
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
        from utils.image_manager import list_images
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
    
        # Retrieve image URL from centralized image storage
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
            redux_output = replicate.run(
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
    
        # Store the newly generated Redux images in image storage
        add_images(ctx.author.id, generated_urls)
    
        await msg.delete()
    

    @commands.command()
    async def stable35(self, ctx, *args):
        """
        Generate images using the 'stability-ai/stable-diffusion-3.5-large' model with a stored prompt.
        Usage: !stable35 prompt[<index>] [optional_input_image_index]
        Example: !stable35 prompt[1] 2
        If an optional input image index is provided, that previously generated image will be used as a starting point.
        """
        if not args or not (args[0].startswith("prompt[") and args[0].endswith("]")):
            await ctx.send("Please specify a stored prompt, e.g. `!stable35 prompt[1] [input_image_index]`.")
            return

        try:
            prompt_index = int(args[0][len("prompt["):-1])
        except ValueError:
            await ctx.send("Invalid prompt index.")
            return

        # Retrieve the stored prompt.
        prompt = get_prompt_by_index(ctx.author.id, prompt_index)
        if not prompt:
            await ctx.send(f"No stored prompt found at index {prompt_index}. Use `!fluxgpt` to generate one.")
            return

        # Optional second argument: input image index.
        input_image_url = None
        if len(args) > 1 and args[1].isdigit():
            image_index = int(args[1])
            # Use the image manager to get the input image.
            input_image_url = get_image_by_index(ctx.author.id, image_index)
            if not input_image_url:
                await ctx.send(f"Invalid image index. You may not have an image at index {image_index}.")
                return

        # Define parameters for Stable Diffusion 3.5.
        cfg = 3.5
        steps = 28
        aspect_ratio = "1:1"
        output_format = "webp"
        output_quality = 90
        prompt_strength = 0.85

        msg_text = f"**Stable Diffusion 3.5** generation in progress...\nStored Prompt (index {prompt_index}): `{prompt}`"
        if input_image_url:
            msg_text += f"\nUsing image #{args[1]} as a starting point..."
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
            output_files = replicate.run(
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
        # Validate that the user provided an image reference in the proper format.
        if not (image_spec.startswith("image[") and image_spec.endswith("]")):
            await ctx.send("Please specify the image to upscale using the format `image[<index>]`.")
            return
    
        try:
            image_index = int(image_spec[len("image["):-1])
        except ValueError:
            await ctx.send("Invalid image index format. Please use `image[<index>]` (e.g., image[2]).")
            return
    
        # Retrieve the stored image URL using the image manager.
        from utils.image_manager import get_image_by_index, add_images
        image_url = get_image_by_index(ctx.author.id, image_index)
        if image_url is None:
            await ctx.send(f"No stored image found at index {image_index}.")
            return
    
        msg = await ctx.send(f"Upscaling image from index {image_index}...")
    
        try:
            output = replicate.run(
                "recraft-ai/recraft-crisp-upscale",
                input={"image": image_url}
            )
        except Exception as e:
            await ctx.send(f"Upscale failed: {e}")
            return
    
        # The upscale model returns a URI (string) as output.
        upscaled_image_url = output
    
        # Store the upscaled image in the image manager.
        add_images(ctx.author.id, [upscaled_image_url])
    
        await msg.edit(content=f"Upscaled image stored and available: {upscaled_image_url}")

        
# This is required for the bot to load the cog
async def setup(bot):
    await bot.add_cog(GenerationCog(bot))
