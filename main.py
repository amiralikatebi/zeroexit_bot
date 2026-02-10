'''
Some Notes from APM:

1- ALways remember, you must not insert your **Token** in online platform for miss used
2- Instead of raw path like MODEL_PATH , always with 'system' or 'os' library chck it exist and then start teelgram bot
3- also use longer prompt (better)


'''




import asyncio
import threading
import re
import random
import time
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_cpp import Llama

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


BOT_TOKEN = "8297582534:AAGWEEks9OzkMwEdIzQ9Mvc9UdisHLH61lw"
BOT_NAME = "zeroexit"
MODEL_PATH = "./llm/qwen2.5-7b-instruct-q4_k_m.gguf"

HEADERS = {"User-Agent": "Mozilla/5.0"}


app = FastAPI(title="ZeroExit AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    n_batch=256,
    n_gpu_layers=0,
    verbose=False
)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+$", "", text)
    text = re.sub(r"[\.…]+$", "", text)
    return text.strip()


def google_image_search(query: str):
    try:
        r = requests.get(
            "https://www.google.com/search",
            params={"q": query, "tbm": "isch"},
            headers=HEADERS,
            timeout=8
        )
        soup = BeautifulSoup(r.text, "html.parser")
        imgs = [img["src"] for img in soup.find_all("img") if img.get("src", "").startswith("http")]
        return random.choice(imgs[:10]) if imgs else None
    except:
        return None



def stream_answer(question: str):
    prompt = f"""تو یک دستیار فارسی دقیق و حرفه‌ای هستی.
قبل از پاسخ، سؤال را در ذهن خود تحلیل کن اما تحلیل را ننویس.
پاسخ نهایی باید:
- فقط فارسی باشد
- کوتاه، شفاف و مفید باشد
- حداکثر در ۲ تا ۳ جمله باشد
- از توضیح اضافی پرهیز کند

هرگز کلمات Human، User یا Assistant را ننویس.
بعد از اتمام پاسخ، فوراً متوقف شو.

سؤال:
{question}

پاسخ کوتاه:
"""

    for chunk in llm(
        prompt,
        max_tokens=120,
        temperature=0.45,
        top_p=0.9,
        repeat_penalty=1.15,
        stream=True,
        stop=[
            "\nHuman",
            "\nUser",
            "\nAssistant",
            "Human:",
            "User:",
            "Assistant:",
            "\n\n",
        ]
    ):
        token = chunk["choices"][0]["text"]
        if token:
            yield token

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"hi! i'm {BOT_NAME}, your AI assistant.\n\n"
    )

async def typing_loop(bot, chat_id, stop_event):
    while not stop_event.is_set():
        await bot.send_chat_action(chat_id, ChatAction.TYPING)
        await asyncio.sleep(3)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    stop_event = asyncio.Event()
    typing_task = asyncio.create_task(
        typing_loop(context.bot, chat_id, stop_event)
    )

    try:

        if text.startswith("عکس"):
            query = text.replace("عکس", "").strip() or text
            caption = clean_text("".join(stream_answer(query)))
            img = google_image_search(query)

            if img:
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=img,
                    caption=caption,
                    reply_to_message_id=update.message.message_id
                )
            else:
                await update.message.reply_text("❌ image not found")

        else:
            msg = await context.bot.send_message(
                chat_id=chat_id,
                text="⏳",
                reply_to_message_id=update.message.message_id
            )

            buffer = ""
            last_sent = ""
            last_update = time.time()

            for token in stream_answer(text):
                buffer += token
                if time.time() - last_update >= 0.4:
                    current = buffer.strip()
                    if current and current != last_sent:
                        await msg.edit_text(current)
                        last_sent = current
                    last_update = time.time()

            final = clean_text(buffer)
            if final and final != last_sent:
                await msg.edit_text(final)

    except Exception as e:
        print(e)
        await update.message.reply_text("❌ error")

    finally:
        stop_event.set()
        await typing_task

def run_fastapi():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

def run_bot():
    tg = ApplicationBuilder().token(BOT_TOKEN).build()
    tg.add_handler(CommandHandler("start", start))
    tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 zeroexit running...")
    tg.run_polling()

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    run_bot()