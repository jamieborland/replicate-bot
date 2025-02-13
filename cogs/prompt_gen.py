
# cogs/prompt_gen.py
import io
import replicate
import discord
from discord.ext import commands
from discord import File
from state import pending_prompts, user_generated_images

class PromptCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def fluxgpt(self, ctx, *args):
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
                "Generate structured text-to-image prompts following this strict order: "
                "(1) Main Subject, (2) Perspective & Camera Details, (3) Setting & Background, "
                "(4) Key Visual Features, (5) Lighting & Mood, (6) Art Style & Detail Level, "
                "(7) Color Palette & Effects (if applicable), (8) Negative Prompts (if supported), "
                "ensuring clarity, coherence, and prioritization of key details."
            ),
            "prompt": user_concept,
            "temperature": 0.7,
            "max_tokens": 350,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
        }

        try:
            llm_output = replicate.run("meta/meta-llama-3-8b-instruct", input=llm_input)

            if isinstance(llm_output, list):
                refined_prompt = "\n".join(llm_output).strip()
            else:
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

    @commands.command()
    async def approveflux(self, ctx):
        """
        Approves the LLM-generated prompt previously created by !fluxgpt,
        then runs it through Flux.
        """
        refined_prompt = pending_prompts.get(ctx.author.id)
        if not refined_prompt:
            await ctx.send("No pending prompt found. Please use `!fluxgpt` first.")
            return

        msg = await ctx.send(f"**Generating images** from your LLM-based prompt:\n```{refined_prompt}```")

        flux_output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": refined_prompt, "num_outputs": 1}
        )

        generated_urls = []
        for i, file_like in enumerate(flux_output, start=1):
            flux_bytes = file_like.read()
            flux_data = io.BytesIO(flux_bytes)
            sent = await ctx.send(
                content=f"**Flux Output {i}**",
                file=File(flux_data, f"flux_llm_{i}.png")
            )
            if sent.attachments:
                generated_urls.append(sent.attachments[0].url)

        user_generated_images[ctx.author.id] = generated_urls

        # Clear the pending prompt now that we've used it
        del pending_prompts[ctx.author.id]

        await msg.delete()

async def setup(bot):
    await bot.add_cog(PromptCog(bot))
