
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
                "You are a helpful assistant that generates structured text-to-image prompts. "
                "The user provides a brief concept, and you should return a single refined prompt "
                "that smoothly incorporates all the following structured elements into a single descriptive text:\n\n"
                "- A detailed description of the **main subject**.\n"
                "- Perspective & camera details blended into the description.\n"
                "- Background elements naturally included.\n"
                "- Distinctive features smoothly described.\n"
                "- Lighting & mood naturally incorporated.\n"
                "- Art style & detail level blended without section labels.\n"
                "- Color palette & effects integrated as part of the description.\n"
                "- Negative prompts smoothly mentioned if necessary.\n\n"
                "The final output should be a **single, natural-sounding paragraph** containing all these details. "
                "Do not include any section labels, numbering, or explicit category names like 'Main Subject:' or 'Lighting:'."
            ),
            "prompt": user_concept,
            "temperature": 0.7,
            "max_tokens": 300,  # Keeping it high for structured but concise
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
            # Show a preview (first 200 characters) for brevity.
            preview = prompt 
            message += f"**{index}**: {preview}\n"
        await ctx.send(message)

async def setup(bot):
    await bot.add_cog(PromptCog(bot))
