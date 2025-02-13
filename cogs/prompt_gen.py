
# cogs/prompt_gen.py
import replicate
from discord.ext import commands
from utils.prompt_manager import save_prompt, list_prompts

class PromptCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def fluxgpt(self, ctx, *args):
        """
        Generate a refined prompt using the LLM and save it.
        Usage: !fluxgpt <concept>
        """
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
                "Do not include extra disclaimers, just the prompt text. "
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

        # Save the refined prompt using our prompt manager.
        save_prompt(ctx.author.id, refined_prompt)
        # Determine the index of the newly added prompt.
        prompts = list_prompts(ctx.author.id)
        new_index = prompts[-1][0]  # last entry's index

        await msg.edit(content=(
            f"**LLM-Generated Prompt Saved as index {new_index}:**\n"
            f"```{refined_prompt}```\n"
            "You can view your stored prompts with `!listprompts`.\n"
            "Then use the prompt in image generation commands like `!flux prompt[1]`."
        ))

    @commands.command()
    async def listprompts(self, ctx):
        """
        Lists your stored prompts.
        Usage: !listprompts
        """
        prompts = list_prompts(ctx.author.id)
        if not prompts:
            await ctx.send("You have no stored prompts. Generate one using `!fluxgpt <concept>`.")
            return

        message = "**Your Stored Prompts:**\n"
        for index, prompt in prompts:
            # Show a preview (first 50 characters) for brevity.
            preview = prompt[:50] + ("..." if len(prompt) > 50 else "")
            message += f"**{index}**: {preview}\n"
        await ctx.send(message)

async def setup(bot):
    await bot.add_cog(PromptCog(bot))
