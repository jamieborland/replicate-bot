# cogs/video_upload.py
import discord
from discord.ext import commands
from utils.video_manager import add_videos

class VideoUploadCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def uploadvideo(self, ctx):
        """
        Upload one or more videos to be saved for later use.
        
        Usage: Attach one or more video files to your message and call:
          !uploadvideo
        The video URLs will be stored and can be viewed using !listvideos.
        """
        if not ctx.message.attachments:
            await ctx.send("Please attach one or more video files to upload.")
            return

        video_urls = []
        for attachment in ctx.message.attachments:
            # Check if the attachment is recognized as a video.
            if attachment.content_type and "video" in attachment.content_type:
                video_urls.append(attachment.url)
            else:
                await ctx.send(f"Attachment `{attachment.filename}` is not recognized as a video.")

        if video_urls:
            add_videos(ctx.author.id, video_urls)
            await ctx.send(f"Uploaded {len(video_urls)} video(s). Use `!listvideos` to view your saved videos.")
        else:
            await ctx.send("No valid video attachments were found.")

async def setup(bot):
    await bot.add_cog(VideoUploadCog(bot))

