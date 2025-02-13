# cogs/image_gen.py
import io
import replicate
import discord
from discord.ext import commands
from discord import File
from state import user_generated_images  

class GenerationCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def flux(self, ctx, *args):
        """
        Generate N images from a text prompt using the Flux Schnell model.
        """
        num_outputs = 1
        if args and args[0].isdigit():
            num_outputs = int(args[0])
            prompt = " ".join(args[1:])
        else:
            prompt = " ".join(args)

        num_outputs = max(1, min(num_outputs, 4))
        msg = await ctx.send(f"{prompt}\n> Generating {num_outputs} images...")

        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt, "num_outputs": num_outputs}
        )

        generated_urls = []
        for idx, img_file in enumerate(output, start=1):
            image_bytes = img_file.read()
            file_data = io.BytesIO(image_bytes)
            sent = await ctx.send(
                content=f"> **Image {idx}** for prompt: {prompt}",
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
        Generate images using the 'stability-ai/stable-diffusion-3.5-large' model on Replicate.
        """

        cfg = 3.5
        steps = 28
        aspect_ratio = "1:1"
        output_format = "webp"
        output_quality = 90
        prompt_strength = 0.85

        # 1) Check if first arg is a digit for image index:
        index = None
        if args and args[0].isdigit():
            index = int(args[0])
            prompt = " ".join(args[1:])
        else:
            prompt = " ".join(args)

        prompt = prompt.strip()
        if not prompt:
            await ctx.send("Please provide a prompt. Usage: `!stable35 <optional index> <prompt>`")
            return

        input_image_url = None
        if index is not None:
            urls = user_generated_images.get(ctx.author.id, [])
            if not urls:
                await ctx.send("You have no generated images to use. Please generate one first with `!flux`.")
                return
            if index < 1 or index > len(urls):
                await ctx.send(f"Invalid image index. Pick a number 1–{len(urls)}.")
                return
            input_image_url = urls[index - 1]

        msg_text = f"**Stable Diffusion 3.5** generation in progress...\nPrompt: `{prompt}`"
        if input_image_url:
            msg_text += f"\nUsing image #{index} as a starting point..."

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
