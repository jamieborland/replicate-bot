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

# user_id -> list of last generated image URLs (from flux)
user_generated_images = {}

# user_id -> one "pending" LLM prompt for text-to-image
pending_prompts = {}

#
# -- Existing flux command (unchanged) --
#
@bot.command()
async def flux(ctx, *args):
    """
    Generate N images from a text prompt using the Flux Schnell model.
    Usage:
    - !flux 3 an astronaut riding a horse  -> generates 3 images
    - !flux an astronaut riding a horse    -> generates 1 image (default)
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

    # Store these new image URLs
    user_generated_images[ctx.author.id] = generated_urls

    await msg.delete()

#
# -- Existing redux command (unchanged) --
#
@bot.command()
async def redux(ctx, index_str: str = "1", aspect_ratio: str = "1:1"):
    """
    Take a previously generated image and run it through the Flux Redux model.
    Usage:
    - !redux 2 3:2
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
        # Add other parameters (megapixels, output_format, etc.) as needed
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


@bot.command()
async def fluxgpt(ctx, *args):
    user_concept = " ".join(args).strip()
    if not user_concept:
        await ctx.send("Please provide a concept. E.g. `!fluxgpt cat astronaut`")
        return

    msg = await ctx.send(f"**Generating a text-to-image prompt** from your concept:\n> {user_concept}")

    llm_input = {
        "system_prompt": (
            "You are a helpful assistant that crafts short, descriptive prompts "
            "for text-to-image generation. The user has provided a brief concept. "
            "Please write a single, refined prompt focusing on relevant details. "
            "Do not include extra disclaimers, just the prompt text."
            "Generate structured text-to-image prompts following this strict order: (1) Main Subject, (2) Perspective & Camera Details, (3) Setting & Background, (4) Key Visual Features, (5) Lighting & Mood, (6) Art Style & Detail Level, (7) Color Palette & Effects (if applicable), (8) Negative Prompts (if supported), ensuring clarity, coherence, and prioritization of key details."
        ),
        "prompt": user_concept,
        "temperature": 0.7,
        "max_tokens": 250,
        "stop_sequences": "<|end_of_text|>,<|eot_id|>",
    }

    try:
        llm_output = replicate.run("meta/meta-llama-3-8b-instruct", input=llm_input)

        # If it's a list of strings, join them; if a single string, just strip it.
        if isinstance(llm_output, list):
            # Option A: take the first string
            # refined_prompt = llm_output[0].strip()

            # Option B: join all segments into one
            refined_prompt = "\n".join(llm_output).strip()
        else:
            # If it was a single string
            refined_prompt = llm_output.strip()

    except Exception as e:
        await msg.edit(content=f"LLM generation failed: {e}")
        return

    pending_prompts[ctx.author.id] = refined_prompt

    await msg.edit(content=(
        f"**LLM-Generated Prompt**:\n"
        f"```{refined_prompt}```\n"
        "If you're happy with this prompt, run `!approveflux` to generate images.\n"
        "If not, you can run `!fluxgpt` again or discard."
    ))

#
# --- NEW COMMAND: 2) Approve the stored prompt and run Flux ---
#
@bot.command()
async def approveflux(ctx):
    """
    Stage 2: Approve the LLM-generated prompt previously created by !fluxgpt.
    Then run it through Flux. 
    """
    refined_prompt = pending_prompts.get(ctx.author.id)
    if not refined_prompt:
        await ctx.send(
            "No pending prompt found. Please use `!fluxgpt` first to generate a prompt."
        )
        return

    msg = await ctx.send(f"**Generating images** from your LLM-based prompt:\n```{refined_prompt}```")

    # We'll just do 1 image for demonstration; you could let the user specify
    flux_output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": refined_prompt, "num_outputs": 1}
    )

    # Send resulting image(s)
    generated_urls = []
    for i, file_like in enumerate(flux_output, start=1):
        flux_bytes = file_like.read()
        flux_data = io.BytesIO(flux_bytes)
        sent = await ctx.send(
            content=f"**Flux Output {i}**",
            file=File(flux_data, f"flux_llm_{i}.png")
        )
        # Gather attachment URLs if needed
        if sent.attachments:
            generated_urls.append(sent.attachments[0].url)

    # Save these new image URLs for potential use in "redux"
    user_generated_images[ctx.author.id] = generated_urls

    # Once used, clear out the pending prompt (so user won't reuse it accidentally)
    del pending_prompts[ctx.author.id]

    await msg.delete()

bot.run(os.environ["DISCORD_TOKEN"])
