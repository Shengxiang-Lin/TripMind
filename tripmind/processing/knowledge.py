from __future__ import annotations
import bz2
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import quote
import mwparserfromhell
from tqdm import tqdm
from .common import canonical_city, CITY_NAMES, stable_id

KEEP_PARAMS = {"name","name-local","content","description","alt","nameextra"}


def clean_wikitext(text: str, depth: int = 0) -> str:
    if depth > 8:
        return ""
    code = mwparserfromhell.parse(text)
    for comment in list(code.filter_comments(recursive=False)):
        code.remove(comment)
    for template in list(code.filter_templates(recursive=False)):
        values = []
        for param in template.params:
            key = str(param.name).strip().casefold()
            if key in KEEP_PARAMS or re.fullmatch(r"region\d+(?:name|description)",key):
                values.append(clean_wikitext(str(param.value),depth+1))
        code.replace(template," ".join(values))
    for link in list(code.filter_wikilinks(recursive=False)):
        title = str(link.title).strip()
        if re.match(r"(?i)^(?:file|image|category|文件|檔案|分类|分類):",title):
            code.remove(link)
    for tag in list(code.filter_tags(recursive=False)):
        if str(tag.tag).casefold() in {"ref","references","gallery","math","score"}:
            code.remove(tag)
    text = code.strip_code(normalize=True,collapse=True)
    lines = [re.sub(r"^[*#:;]+\s*","",line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}","\n\n","\n".join(lines)).strip()


def text_chunks(text: str, size: int, overlap: int, minimum: int = 80):
    if size <= overlap or overlap < 0:
        raise ValueError("chunk_size must exceed nonnegative overlap")
    start = 0
    while start < len(text):
        stop = min(start+size,len(text))
        if stop < len(text):
            boundary = max(text.rfind("\n",start+size//2,stop),text.rfind("。",start+size//2,stop),text.rfind(". ",start+size//2,stop))
            if boundary > start: stop=boundary+1
        value = text[start:stop].strip()
        if len(value) >= minimum: yield value
        if stop >= len(text): break
        start = max(start+1,stop-overlap)


def page_documents(page, language: str, source: str, chunk_size: int, overlap: int):
    def child(element,name):
        return next((x for x in element if x.tag.rsplit("}",1)[-1]==name),None)
    title_node, ns_node = child(page,"title"),child(page,"ns")
    if title_node is None or ns_node is None or ns_node.text != "0" or child(page,"redirect") is not None:
        return
    title = title_node.text or ""
    revision = child(page,"revision")
    if revision is None: return
    text_node = child(revision,"text")
    if text_node is None or not text_node.text: return
    root_title = re.split(r"[/／]",title)[0]
    city = canonical_city(root_title)
    country = "China" if city in CITY_NAMES else "*"
    # General travel topic pages are eligible as global fallback documents.
    if re.search(r"\{\{\s*(?:topic|traveltopic|旅行话题|旅行話題)\b",text_node.text,re.I):
        city = "*"
    section = "Overview" if language=="en" else "概览"
    parts = re.split(r"(?m)^={2,6}[ \t]*(.*?)[ \t]*={2,6}[ \t]*$",text_node.text)
    source_url = f"https://{language}.wikivoyage.org/wiki/"+quote(title.replace(" ","_"))
    page_id = child(page,"id"); revision_id = child(revision,"id")
    for index,part in enumerate(parts):
        if index%2:
            section=clean_wikitext(part); continue
        cleaned = clean_wikitext(part)
        for chunk in text_chunks(cleaned,chunk_size,overlap):
            yield {
                "doc_id":stable_id(language,title,section,chunk)[:24],
                "title":title,"section":section,"city":city,"country":country,"language":language,
                "text":chunk,"source":source_url,"license":"CC BY-SA 4.0",
                "page_id":page_id.text if page_id is not None else None,
                "revision_id":revision_id.text if revision_id is not None else None,
                "source_file":source,
            }


def build_knowledge(raw: Path, out: Path, languages: list[str], chunk_size: int = 1200, overlap: int = 150) -> dict:
    folder = out/"knowledge"; folder.mkdir()
    db = sqlite3.connect(folder/"documents.sqlite")
    db.execute("CREATE TABLE documents (row_idx INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, city TEXT, country TEXT, language TEXT, metadata TEXT)")
    seen = set()
    summary = {}
    count = 0
    try:
        with (folder/"chunks.jsonl").open("w",encoding="utf-8") as dest:
            for language in languages:
                path = raw/f"wikivoyage/{language}/{language}wikivoyage-latest-pages-articles.xml.bz2"
                pages = skipped = duplicates = accepted_pages = 0
                city_counts = Counter(); before=count
                progress = tqdm(desc=f"Wikivoyage {language}",unit="pages",mininterval=2)
                with bz2.open(path,"rb") as stream:
                    iterator = ET.iterparse(stream,events=("start","end"))
                    _,root = next(iterator)
                    for event,page in iterator:
                        if event != "end" or not page.tag.endswith("}page"): continue
                        pages += 1; progress.update()
                        added = 0
                        for doc in page_documents(page,language,str(path.relative_to(raw)),chunk_size,overlap):
                            text_hash = stable_id(language,doc["city"]," ".join(doc["text"].casefold().split()))
                            if text_hash in seen:
                                duplicates += 1; continue
                            seen.add(text_hash)
                            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?)",(count,doc["doc_id"],doc["city"],doc["country"],language,json.dumps(doc,ensure_ascii=False)))
                            dest.write(json.dumps(doc,ensure_ascii=False,sort_keys=True)+"\n")
                            count+=1; added+=1; city_counts[doc["city"]]+=1
                        if added: accepted_pages+=1
                        else: skipped+=1
                        if pages%500==0: db.commit()
                        root.clear()
                progress.close()
                summary[language] = {"pages_seen":pages,"pages_with_chunks":accepted_pages,
                    "skipped_or_short_pages":skipped,"duplicate_chunks":duplicates,
                    "chunks":count-before,"destination_chunks":{c:city_counts[c] for c in CITY_NAMES}}
        db.execute("CREATE INDEX documents_destination ON documents(city,country,language)")
        db.commit()
    finally:
        db.close()
    if not count: raise ValueError("No Wikivoyage documents produced")
    return {"chunks":count,"languages":summary,"chunk_size":chunk_size,"overlap":overlap,
            "deduplication":"language + city + whitespace-normalized exact text"}
