import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # DNSE API
    DNSE_API_KEY = os.getenv("DNSE_API_KEY", "")
    DNSE_API_SECRET = os.getenv("DNSE_API_SECRET", "")

    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

    # Trading
    ACCOUNT_NO = os.getenv("DNSE_ACCOUNT_NO", "")

    # Data
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod
    def validate(cls):
        if not cls.DNSE_API_KEY or not cls.DNSE_API_SECRET:
            raise ValueError("DNSE_API_KEY and DNSE_API_SECRET must be set in .env")
