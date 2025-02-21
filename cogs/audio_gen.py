# cogs/audio_gen.py
import io
import asyncio
import replicate
import discord
from discord.ext import commands
from utils.prompt_manager import get_prompt_by_index
from utils.video_manager import get_video_by_index

class AudioCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def audio(self, ctx, *args):
        """
        Generate audio using the MMAudio model.
        
        Usage examples:
          - Using a stored prompt and a stored video:
                !audio prompt[1] video[2]
          - Using a direct prompt and a stored video:
                !audio a cat meowing video[2]
        
        The command accepts:
          - Stored prompt markers (prompt[<index>])
          - Video markers (video[<index>]) for video-to-audio generation
          - Direct text prompt
        
        Other parameters use defaults:
          seed: -1, duration: 8, num_steps: 25, cfg_strength: 4.5, negative_prompt: "music"
        """
        stored_prompt = None
        video_url = None
        direct_prompt_parts = []

        # Set default parameters from the schema.
        seed = -1
        duration = 8
        num_steps = 25
        cfg_strength = 4.5
        negative_prompt = "music"

        # Parse arguments.
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
            elif arg.startswith("video[") and arg.endswith("]"):
                try:
                    idx = int(arg[len("video["):-1])
                    video_url = get_video_by_index(ctx.author.id, idx)
                    if not video_url:
                        await ctx.send(f"No stored video found at index {idx}.")
                        return
                except ValueError:
                    await ctx.send("Invalid video index format.")
                    return
            else:
                direct_prompt_parts.append(arg)

        # Determine which prompt to use: stored prompt takes precedence.
        prompt = stored_prompt if stored_prompt else " ".join(direct_prompt_parts).strip()
        if not prompt:
            await ctx.send("Please provide a prompt either as a stored prompt (prompt[<index>]) or as direct text.")
            return

        # Since a video is required, return an error if no video was provided.
        if not video_url:
            await ctx.send("This command requires a video. Please provide a stored video using the format video[<index>].")
            return

        # Build the input dictionary according to the schema.
        audio_input = {
            "prompt": prompt,
            "seed": seed,
            "duration": duration,
            "num_steps": num_steps,
            "cfg_strength": cfg_strength,
            "negative_prompt": negative_prompt,
            "video": video_url
        }

        msg = await ctx.send(
            f"Generating audio with prompt: `{prompt}` using your stored video."
        )

        try:
            # Run the blocking replicate.run in a separate thread.
            output = await asyncio.to_thread(
                replicate.run,
                "zsxkib/mmaudio:4b9f801a167b1f6cc2db6ba7ffdeb307630bf411841d4e8300e63ca992de0be9",
                input=audio_input
            )
        except Exception as e:
            await msg.edit(content=f"Audio generation failed: {e}")
            return

        # Read the file-like output.
        audio_bytes = output.read()
        audio_file = io.BytesIO(audio_bytes)

        # Send the audio file as an attachment.
        await ctx.send(
            content="Audio generated:",
            file=discord.File(audio_file, "output.mp4")
        )

        await msg.delete()

async def setup(bot):
    await bot.add_cog(AudioCog(bot))
