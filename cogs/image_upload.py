# cogs/image_upload.py
import discord
from discord.ext import commands
from utils.image_manager import add_images

class ImageUploadCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def uploadimage(self, ctx):
        """
        Upload one or more images to be saved for later use.
        
        Usage: Attach one or more image files to your message and call:
          !uploadimage
        The image URLs will be stored and can be viewed using !listimages.
        """
        if not ctx.message.attachments:
            await ctx.send("Please attach one or more images to upload.")
            return

        image_urls = []
        for attachment in ctx.message.attachments:
            # Optionally, you can check if the attachment is an image.
            if attachment.content_type and "image" in attachment.content_type:
                image_urls.append(attachment.url)
            else:
                await ctx.send(f"Attachment `{attachment.filename}` is not recognized as an image.")

        if image_urls:
            add_images(ctx.author.id, image_urls)
            await ctx.send(f"Uploaded {len(image_urls)} image(s). Use `!listimages` to view your saved images.")
        else:
            await ctx.send("No valid image attachments were found.")

async def setup(bot):
    await bot.add_cog(ImageUploadCog(bot))

