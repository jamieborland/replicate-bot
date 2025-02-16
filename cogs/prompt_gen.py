
# cogs/prompt_gen.py
import replicate
from discord.ext import commands
from utils.prompt_manager import save_prompt, list_prompts, get_prompt_by_index

class PromptCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def gpt(self, ctx, *, concept: str):
        """
        Generate a text-to-image prompt using Llama 3 70B Instruct.
        Usage: !gpt <concept>

        Example:
            !gpt A serene forest at sunrise
        """
        concept = concept.strip()
        if not concept:
            await ctx.send("Please provide a concept. E.g. `!gpt A city in the clouds`")
            return

        # Let the user know we're working
        msg = await ctx.send(f"**Generating a text-to-image prompt** from your concept:\n> {concept}")

                # Prepare your Llama 3 70B Instruct inputs
        # Example updated system_prompt for Llama 3 70B Instruct
        llm_input = {
            "system_prompt": (
                "You are an expert at creating concise, vivid text-to-image prompts. "
                "The user provides a short concept. You must integrate the following details in a single, natural paragraph:\n"
                "- Main subject\n"
                "- Scene composition\n"
                "- Background elements\n"
                "- Lighting and mood\n"
                "- Artistic style\n"
                "- Color palette\n"
                "- Lens or camera settings (if applicable)\n"
                "- Negative prompts (avoid certain elements)\n\n"
                "Output only one paragraph describing the entire scene, smoothly incorporating these elements. "
                "Do not add headings, bullet points, or extraneous explanations. "
                "Do not mention 'the concept' or 'the user'; simply describe the scene as if you are finalizing a text prompt. "
                "Do not start your response with phrases like 'Here is the...'—just provide the paragraph directly."
            ),
            "prompt": concept,
            "temperature": 0.2,
            "max_tokens": 200,
            "top_p": 0.9,
            "presence_penalty": 1.15,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
        }



        try:
            # Use the 70B Instruct model on Replicate
            # Name: "meta/meta-llama-3-70b-instruct"
            llm_output = replicate.run("meta/meta-llama-3-70b-instruct", input=llm_input)
            # llm_output can be a string or list of strings
            if isinstance(llm_output, list):
                final_prompt = "\n".join(llm_output).strip()
            else:
                final_prompt = llm_output.strip()
        except Exception as e:
            await msg.edit(content=f"LLM generation failed: {e}")
            return

        # Save the prompt
        save_prompt(ctx.author.id, final_prompt)
        # Get the index of the newly added prompt
        new_index = list_prompts(ctx.author.id)[-1][0]

        await msg.edit(content=(
            f"**LLM-Generated Prompt (Index {new_index}):**\n"
            f"```{final_prompt}```\n"
            "Use `!listprompts` to view all prompts.\n"
            "Then generate an image with e.g. `!flux prompt[<index>]` or refine it with `!refine <index> <instructions>`."
        ))

    @commands.command()
    async def refine(self, ctx, *args):
        """
        Refine an existing prompt using Claude 3.5 Sonnet.
        
        Usage:
          !refine prompt[<index>] <refinement_instructions>
        
        Example:
          !refine prompt[1] change the birch to be stone in the bridge
        """
        stored_prompt = None
        instructions_parts = []
    
        # Parse arguments: look for a stored prompt reference and remaining instructions.
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
            else:
                instructions_parts.append(arg)
        
        instructions = " ".join(instructions_parts).strip()
        if not stored_prompt:
            await ctx.send("Please provide a stored prompt reference like `prompt[<index>]`.")
            return
        if not instructions:
            await ctx.send("Please provide instructions for how to refine the prompt.")
            return
    
        # Let the user know we're refining
        msg = await ctx.send(
            f"Refining prompt (from stored prompt) with your instructions:\n> {instructions}"
        )
    
        # Build input for Claude 3.5 Sonnet
        claude_input = {
            "system_prompt": (
                "You are an assistant that refines text-to-image prompts. "
                "Keep the core idea but focus on making the changes requested by the user. "
                "Do NOT add extra commentary—only return the updated prompt as a single paragraph."
            ),
            "prompt": (
                f"Original prompt:\n{stored_prompt}\n\n"
                f"User's instructions:\n{instructions}\n\n"
                "Now return the fully refined prompt as a single paragraph."
            ),
            "max_tokens": 200  # Adjust as needed
        }
    
        try:
            # Use Claude 3.5 Sonnet model (Name: "anthropic/claude-3.5-sonnet")
            llm_output = replicate.run("anthropic/claude-3.5-sonnet", input=claude_input)
            if isinstance(llm_output, list):
                refined_prompt = "\n".join(llm_output).strip()
            else:
                refined_prompt = llm_output.strip()
        except Exception as e:
            await msg.edit(content=f"Refinement failed: {e}")
            return
    
        # Save the refined prompt as a new entry
        save_prompt(ctx.author.id, refined_prompt)
        new_index = list_prompts(ctx.author.id)[-1][0]
    
        await msg.edit(content=(
            f"**Refined Prompt Saved (Index {new_index}):**\n"
            f"```{refined_prompt}```\n"
            "You can view your prompts with `!listprompts` or refine further with `!refine prompt[<index>] <instructions>`."
        ))


    @commands.command()
    async def listprompts(self, ctx):
        """
        Lists your stored prompts.
        Usage: !listprompts
        """
        prompts = list_prompts(ctx.author.id)
        if not prompts:
            await ctx.send("You have no stored prompts. Generate one using `!gpt <concept>`.")
            return

        message = "**Your Stored Prompts:**\n"
        for idx, prompt in prompts:
            preview = prompt[:200] + ("..." if len(prompt) > 200 else "")
            message += f"**{idx}**: {preview}\n"
        await ctx.send(message)

async def setup(bot):
    await bot.add_cog(PromptCog(bot))
