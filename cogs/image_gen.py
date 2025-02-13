# cogs/image_gen.py
import io
import replicate
import discord
from discord.ext import commands
from discord import File
from state import user_generated_images  
from utils.prompt_manager import get_prompt_by_index
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

        user_generated_images[ctx.author.id] = generated_urls
        await msg.delete()


    @commands.command()
    async def redux(self, ctx, index_str: str = "1", aspect_ratio: str = "1:1"):
        """
        Take a previously generated image and run it through the Flux Redux model.
        """
        try:
            index = int(index_str)
        except ValueError:
            index = 1

        urls = user_generated_images.get(ctx.author.id, [])
        if not urls:
            await ctx.send("You have no generated images to redux. Use `!flux` first!")
            return

        if index < 1 or index > len(urls):
            await ctx.send(f"Invalid image index. Pick a number 1–{len(urls)}.")
            return

        redux_image_url = urls[index - 1]
        msg = await ctx.send(f"Running Redux on image #{index} with aspect_ratio={aspect_ratio}...")

        redux_input = {
            "redux_image": redux_image_url,
            "aspect_ratio": aspect_ratio,
        }

        redux_output = replicate.run(
            "black-forest-labs/flux-redux-schnell",
            input=redux_input
        )

        for i, file_like in enumerate(redux_output, start=1):
            redux_bytes = file_like.read()
            redux_data = io.BytesIO(redux_bytes)
            await ctx.send(
                content=f"Redux output {i} from image #{index}",
                file=File(redux_data, f"redux_output_{i}.webp")
            )

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
            urls = user_generated_images.get(ctx.author.id, [])
            if not urls:
                await ctx.send("You have no generated images to use as input. Please generate one first with `!flux`.")
                return
            if image_index < 1 or image_index > len(urls):
                await ctx.send(f"Invalid image index. Choose between 1 and {len(urls)}.")
                return
            input_image_url = urls[image_index - 1]

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

        user_generated_images[ctx.author.id] = generated_urls
        await msg.delete()

# This is required for the bot to load the cog
async def setup(bot):
    await bot.add_cog(GenerationCog(bot))
