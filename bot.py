import os
import asyncio
import threading
import logging
from dotenv import load_dotenv
import discord
from discord.ext import commands
from flask import Flask

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Setup Discord bot
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    description="Runs models on Replicate!",
    intents=intents
)

# List of cogs to load
initial_cogs = [
    "cogs.image_gen",
    "cogs.prompt_gen",
    "cogs.video_gen",
    "cogs.image_upload",
    "cogs.add_prompt",
    "cogs.video_upload",
]

async def main():
    async with bot:
        # Load each cog
        for cog in initial_cogs:
            await bot.load_extension(cog)
        # Start the bot
        await bot.start(os.environ["DISCORD_TOKEN"])

# Flask server to keep Render from shutting down
app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is running!"

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Run the bot
    asyncio.run(main())
