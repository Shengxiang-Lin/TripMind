"""Shared bilingual lexical tokenizer; it does not call an embedding service."""
import re
import unicodedata
from functools import lru_cache


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC
    return OpenCC("t2s")


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    result = re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)*", text)
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        run = _converter().convert(run)
        result.extend(run[i:i+2] for i in range(len(run)-1))
        if len(run) == 1:
            result.append(run)
    return result


INTEREST_HINTS = {
    "photography": "摄影 拍照 风景 scenic photography",
    "food": "美食 餐饮 小吃 food cuisine restaurant",
    "culture": "文化 人文 culture heritage",
    "history": "历史 古迹 history temple heritage",
    "nature": "自然 山水 公园 nature park scenic",
    "museum": "博物馆 纪念馆 museum gallery",
    "architecture": "建筑 寺庙 architecture temple",
    "night_view": "夜景 夜游 night view",
    "shopping": "购物 商业街 shopping market",
}


def profile_query(profile) -> str:
    return " ".join([
        profile.city, profile.country, profile.travel_group, profile.pace,
        *[INTEREST_HINTS.get(x, x) for x in profile.interests],
        *profile.special_preferences,
    ])
