import os
import asyncio
from dotenv import load_dotenv
import discord
from discord.ext import commands

import logging
logging.basicConfig(level=logging.INFO)
load_dotenv()  # Loads .env if present

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    description="Runs models on Replicate!",
    intents=intents
)

# List the cogs we want to load
initial_cogs = [
    "cogs.image_gen",
    "cogs.prompt_gen",
    "cogs.video_gen",
]

async def main():
    async with bot:
        # Load each cog
        for cog in initial_cogs:
            await bot.load_extension(cog)
        # Start the bot
        await bot.start(os.environ["DISCORD_TOKEN"])

if __name__ == "__main__":
    asyncio.run(main())
