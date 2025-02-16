# cogs/prompt_gen.py
import discord
from discord.ext import commands
from utils.prompt_manager import save_prompt, list_prompts

class AddPromptCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def addprompt(self, ctx, *, prompt_text: str):
        """
        Add your own custom prompt to your stored prompts.
        
        Usage:
          !addprompt <your custom prompt>
        
        Example:
          !addprompt A dreamy, futuristic cityscape with neon lights and flying cars.
        """
        prompt_text = prompt_text.strip()
        if not prompt_text:
            await ctx.send("Please provide a prompt text to add.")
            return

        # Save the custom prompt
        save_prompt(ctx.author.id, prompt_text)
        # Retrieve the new index (1-based) from the stored prompts
        new_index = list_prompts(ctx.author.id)[-1][0]

        await ctx.send(f"Your custom prompt has been added as prompt[{new_index}].")
    

async def setup(bot):
    await bot.add_cog(AddPromptCog(bot))

