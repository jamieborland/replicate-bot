import io
import os
import replicate
import discord
from discord import Intents, File
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

intents = Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    description="Runs models on Replicate!",
    intents=intents,
)

@bot.command()
async def flux(ctx, *args):
    """Generate N images from a text prompt using the Flux Schnell model.

    Usage:
    - `!flux 3 an astronaut riding a horse` → Generates 3 images
    - `!flux an astronaut riding a horse`   → Generates 1 image (default)
    """
    # Default number of images
    num_outputs = 1  

    # Check if the first argument is an integer
    if args and args[0].isdigit():
        num_outputs = int(args[0])
        prompt = " ".join(args[1:])  # Use the rest as the prompt
    else:
        prompt = " ".join(args)  # Use everything as the prompt

    # Prevent users from generating too many images (adjust if needed)
    num_outputs = max(1, min(num_outputs, 4))

    msg = await ctx.send(f"{prompt}\n> Generating {num_outputs} images...")

    # Run the model; returns a list of file-like objects
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": prompt, "num_outputs": num_outputs}
    )

    # Iterate over all generated images and send them to Discord
    for idx, img_file in enumerate(output, start=1):
        image_bytes = img_file.read()
        file_data = io.BytesIO(image_bytes)

        await ctx.send(
            content=f"> **Image {idx}** for prompt: {prompt}",
            file=File(file_data, f"flux_{idx}.png")
        )

    # Clean up "Generating..." message
    await msg.delete()

bot.run(os.environ["DISCORD_TOKEN"])
