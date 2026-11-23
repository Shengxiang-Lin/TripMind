from __future__ import annotations
from datetime import date
from pathlib import Path
import html,json
import os,re
import base64
import io

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# 没有已激活的处理数据包时，自动使用项目自带的演示数据。
# 必须在导入并初始化 TripMindService 之前设置，避免启动阶段报错。
_active_bundle = PROJECT_ROOT / "data" / "processed" / "production" / "current.json"
if not _active_bundle.is_file():
    os.environ["TRIPMIND_USE_PROCESSED"] = "false"

import gradio as gr
from inference import V2_ADAPTER, build_input, generate_recommendation
from PIL import Image
from tripmind.data import city_pois
from tripmind.models import UserProfile
from tripmind.service import TripMindService,itinerary_to_markdown
service=TripMindService();INTERESTS=["photography","food","culture","history","nature","museum","architecture","night_view","shopping"]

def load_poi_image_map():
    try:
        payload = json.loads((PROJECT_ROOT / "data/processed/images/poi_image_map.json").read_text(encoding="utf-8"))
        metadata = payload["images"]
        if not isinstance(metadata, dict):
            raise ValueError("Invalid image mapping")
        paths = {poi_id: item["local_path"] for poi_id, item in metadata.items()}
        return paths, metadata, ""
    except (OSError, ValueError, KeyError, TypeError):
        return {}, {}, "图片暂不可用：无法读取本地图片映射；文字行程仍可正常使用。"

POI_IMAGE_PATHS, POI_IMAGE_METADATA, IMAGE_MAP_WARNING = load_poi_image_map()

# 景点名称与图片映射名称的人工核对表
POI_IMAGE_ALIASES = {
    "西湖": "西湖风景名胜区",
    "灵隐寺": "灵隐寺",
    "河坊街": "清河坊街",
    "小河直街": "小河直街历史文化街区",
    "中国茶叶博物馆": "中国茶叶博物馆",
    "杭州动物园": "杭州动物园",
}


def find_poi_image(poi):
    """优先按 POI ID 查找，失败后按人工核对的景点名称查找。"""

    poi_id = getattr(poi, "poi_id", "")

    # 第一优先级：直接使用景点 ID
    if poi_id in POI_IMAGE_PATHS:
        return poi_id

    # 第二优先级：通过人工核对的名称匹配
    poi_name = getattr(poi, "name", "")
    target_name = POI_IMAGE_ALIASES.get(poi_name)

    if not target_name:
        return None

    for image_id, metadata in POI_IMAGE_METADATA.items():
        if (
            metadata.get("city") == "杭州"
            and metadata.get("name") == target_name
        ):
            return image_id

    return None

def itinerary_images(itinerary):
    gallery, credits, notices, seen = [], [], [], set()

    if IMAGE_MAP_WARNING:
        notices.append(IMAGE_MAP_WARNING)

    for day in itinerary.days:
        for item in day.items:
            poi = item.poi

            # 使用统一的图片匹配函数
            image_id = find_poi_image(poi)

            if not image_id or image_id in seen:
                continue

            seen.add(image_id)

            try:
                relative_path = Path(POI_IMAGE_PATHS[image_id])
                local_path = (PROJECT_ROOT / relative_path).resolve()

                if (
                    relative_path.is_absolute()
                    or not local_path.is_relative_to(PROJECT_ROOT)
                ):
                    raise ValueError("Image path must stay inside the project")

                with Image.open(local_path) as source:
                    picture = source.convert("RGB")

                metadata = POI_IMAGE_METADATA[image_id]
                author = metadata.get("author")
                license_name = metadata.get("license")

                caption = poi.name

                if author and license_name:
                    attribution = f"图片：{author} / {license_name}"
                    caption += "\n" + attribution

                    links = []

                    for label, key in [
                        ("来源", "source_url"),
                        ("许可", "license_url"),
                    ]:
                        url = metadata.get(key, "")

                        if (
                            isinstance(url, str)
                            and url.startswith("https://")
                        ):
                            links.append(
                                f'<a href="{html.escape(url, quote=True)}">'
                                f'{label}</a>'
                            )

                    credits.append(
                        f"<small>{html.escape(poi.name)} · "
                        f"{html.escape(attribution)} "
                        f"{' · '.join(links)}</small>"
                    )

                gallery.append((picture, caption))

            except Exception:
                notices.append(
                    f"{html.escape(poi.name)}：图片暂不可用；"
                    "文字行程不受影响。"
                )

    return gallery, "\n\n".join(notices + credits)

def render_itinerary_html(itinerary):
    """Render a personalized itinerary as image-rich HTML cards."""

    p = itinerary.profile
    total_places = sum(len(day.items) for day in itinerary.days)
    total_distance = sum(day.travel_distance_km for day in itinerary.days)

    def safe(value):
        return html.escape(str(value or ""))

    def time_text(value):
        return f"{value // 60:02d}:{value % 60:02d}"

    def image_data(poi):
        """Load a matched local POI image and return an embedded data URL."""
        try:
            image_id = find_poi_image(poi)
            relative = POI_IMAGE_PATHS.get(image_id) if image_id else None
            if not relative:
                return ""

            relative_path = Path(relative)
            if relative_path.is_absolute():
                return ""

            local_path = (PROJECT_ROOT / relative_path).resolve()
            if not local_path.is_relative_to(PROJECT_ROOT):
                return ""

            with Image.open(local_path) as source:
                picture = source.convert("RGB")
                picture.thumbnail((1200, 800))
                buffer = io.BytesIO()
                picture.save(buffer, format="JPEG", quality=84, optimize=True)

            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            # A missing or unreadable image must not break the itinerary.
            return ""

    location = f"{p.city}, {p.country}" if p.country else p.city

    parts = [
        '<div class="tm-itinerary">',
        '<div class="tm-itinerary-hero">',
        '<div class="tm-itinerary-eyebrow">YOUR PERSONALIZED JOURNEY</div>',
        f'<h2>{safe(location)}</h2>',
        f'<p>{p.days}-Day Personalized Travel Plan</p>',
        '</div>',
        '<div class="tm-itinerary-stats">',
        f'<div class="tm-stat"><strong>{total_places}</strong><span>Attractions</span></div>',
        f'<div class="tm-stat"><strong>{len(itinerary.days)}</strong><span>Travel days</span></div>',
        f'<div class="tm-stat"><strong>{total_distance:.1f} km</strong><span>Travel distance</span></div>',
        f'<div class="tm-stat"><strong>{safe(p.currency)} {itinerary.total_estimated_cost:.0f}</strong><span>Ticket cost</span></div>',
        '</div>',
    ]

    if p.start_date:
        parts.append(
            f'<div class="tm-itinerary-date">Travel date: {safe(p.start_date)}</div>'
        )

    if itinerary.warnings:
        parts.append(
            '<div class="tm-friendly-notice">'
            'Some planning conditions may affect the available attractions. '
            'Please confirm opening hours and ticket information before departure.'
            '</div>'
        )

    for day in itinerary.days:
        parts.extend([
            '<section class="tm-day-card">',
            '<div class="tm-day-heading">',
            f'<span class="tm-day-number">DAY {day.day:02d}</span>',
            '<span class="tm-day-label">YOUR DAILY ADVENTURE</span>',
            '</div>',
        ])

        if not day.items:
            parts.append(
                '<div class="tm-empty-state">'
                'No feasible attractions were found for this day. '
                'Please adjust your preferences or budget.'
                '</div>'
            )
            parts.append('</section>')
            continue

        parts.append('<div class="tm-timeline">')

        for item in day.items:
            poi = item.poi
            image_src = image_data(poi)
            tags = " · ".join(safe(tag) for tag in (poi.tags or [])[:4])
            duration = max(0, item.end_min - item.start_min)

            parts.append('<div class="tm-timeline-item">')
            parts.append(
                '<div class="tm-timeline-marker"><span></span></div>'
            )
            parts.append('<article class="tm-poi-card">')
            parts.append('<div class="tm-poi-content">')

            # Visit time and duration are separate elements for consistent spacing.
            parts.append(
                '<div class="tm-poi-time" style="display:flex!important;justify-content:flex-start!important;align-items:center!important;flex-wrap:wrap!important;column-gap:5ch!important;row-gap:8px!important;width:100%!important;">'
                f'<span class="tm-time-range" style="display:inline-block!important;white-space:nowrap!important;">'
                f'{time_text(item.start_min)} – {time_text(item.end_min)}'
                '</span>'
                f'<span class="tm-time-duration" style="display:inline-block!important;white-space:nowrap!important;">'
                f'{duration // 60}h {duration % 60:02d}m'
                '</span>'
                '</div>'
            )

            # Place name first; show the image immediately underneath it only if available.
            parts.append(f'<h3>{safe(poi.name)}</h3>')
            if image_src:
                parts.append(
                    f'<img class="tm-poi-image" src="{image_src}" '
                    f'alt="{safe(poi.name)}" loading="lazy">'
                )

            parts.append(
                f'<div class="tm-poi-category">{safe(poi.category)}</div>'
            )

            if poi.description:
                parts.append(
                    f'<p class="tm-poi-description">'
                    f'{safe(poi.description)}</p>'
                )

            parts.append(
                '<div class="tm-poi-meta" style="display:flex!important;align-items:center!important;flex-wrap:wrap!important;column-gap:28px!important;row-gap:10px!important;width:100%!important;margin-top:14px!important;">'
                f'<span class="tm-poi-price" style="display:inline-block!important;white-space:nowrap!important;margin-right:4px!important;">{safe(poi.currency)} {poi.ticket_price:.0f}</span>'
                f'<span class="tm-poi-score" style="display:inline-block!important;white-space:nowrap!important;">Preference {item.score:.2f}</span>'
                '</div>'
            )

            if tags:
                parts.append(f'<div class="tm-poi-tags">{tags}</div>')

            parts.append('</div>')  # tm-poi-content
            parts.append('</article>')
            parts.append('</div>')  # tm-timeline-item

            if item.travel_from_previous:
                leg = item.travel_from_previous
                parts.append(
                    '<div class="tm-route-info" style="display:flex!important;align-items:center!important;flex-wrap:wrap!important;column-gap:28px!important;row-gap:8px!important;width:calc(100% - 27px)!important;">'
                    f'<span class="tm-route-mode" style="display:inline-block!important;white-space:nowrap!important;">{safe(leg.mode)}</span>'
                    f'<span class="tm-route-distance" style="display:inline-block!important;white-space:nowrap!important;">{leg.distance_km:.1f} km</span>'
                    f'<span class="tm-route-duration" style="display:inline-block!important;white-space:nowrap!important;">About {leg.duration_min} min</span>'
                    '</div>'
                )

        parts.append('</div>')  # tm-timeline
        parts.append(
            '<div class="tm-day-summary">'
            '<span>Daily travel distance</span>'
            f'<strong>{day.travel_distance_km:.1f} km</strong>'
            '<span>Estimated travel time</span>'
            f'<strong>{day.travel_duration_min} min</strong>'
            '</div>'
        )
        parts.append('</section>')

    parts.append('</div>')
    return "".join(parts)

def history_items(v):return [x.strip() for x in re.split(r"[\n,;，；]+",v or "") if x.strip()]
def generate(country,city,start_date,days,budget,currency,age_group,travel_group,interests,pace,transport,travel_history,notes,exclude_poi_ids=None):
    try:
        p=UserProfile(country=(country or "China").strip(),city=city.strip() or "杭州",start_date=start_date.strip() or None,days=int(days),budget=float(budget),currency=currency,age_group=age_group,travel_group=travel_group,interests=interests or ["culture"],pace=pace,transport_preference=transport,travel_history=history_items(travel_history),special_preferences=[x.strip() for x in notes.split(",") if x.strip()])
        excluded = set(exclude_poi_ids or ())
        result = service.plan(p, exclude_poi_ids=excluded)
        candidates = city_pois(service.pois, result.profile.city, result.profile.country)
        selection = gr.CheckboxGroup(
            choices=[(poi.name, poi.poi_id) for poi in candidates],
            value=[poi.poi_id for poi in candidates if poi.poi_id in excluded],
        )
        gallery, image_details = itinerary_images(result)
        if not any(day.items for day in result.days):
            return (
    '<div class="tm-empty-state">'
    '<h3>暂时无法生成可行行程</h3>'
    '<p>当前景点可能无法满足开放时间、游玩时长或预算要求。'
    '请调整预算或减少排除的景点后重新尝试。</p>'
    '</div>',
    selection,
    gallery,
    image_details,
)
        text = render_itinerary_html(result)
        if any(not day.items for day in result.days):
            text = "部分日期无法安排可行 POI，请减少排除项或调整预算、天数。\n\n" + text
        return text, selection, gallery, image_details
    except ValueError as exc:
        selection = gr.skip() if exclude_poi_ids is not None else gr.CheckboxGroup(choices=[], value=[])
        return (
            '<div class="tm-friendly-notice">'
            '<strong>无法规划行程</strong><br>'
            + html.escape(str(exc))
            + '</div>',
            selection,
            [],
            "",
        )
    except Exception:
        selection = gr.skip() if exclude_poi_ids is not None else gr.CheckboxGroup(choices=[], value=[])
        return (
            '<div class="tm-friendly-notice">'
            '<strong>暂时无法生成行程</strong><br>'
            '请检查输入信息后重试。如问题持续存在，请查看 CMD 运行日志。'
            '</div>',
            selection,
            [],
            "",
        )

def ai_suggest(origin, people, city, days, budget, currency, interests, travel_group, pace):
    try:
        origin, city = origin.strip(), city.strip()
        if not origin or not city:
            return "请填写出发地和目的地。", ""
        people, days = int(people), int(days)
        interest_text = "、".join(interests or [])
        query = (
            f"当前位置{origin}。{travel_group}出行，共{people}人，计划{days}天，"
            f"预算{budget}{currency}，兴趣：{interest_text}，节奏：{pace}。"
        )
        text = generate_recommendation(build_input(query, origin, city, days, people))
        if not text:
            raise ValueError("Empty model response")
        return f"已就绪：{V2_ADAPTER.name}", text
    except Exception as exc:
        return f"暂时无法生成 AI 行程建议（{type(exc).__name__}）。请稍后重试。", ""


with gr.Blocks(
    title="TripMind | AI Travel Planner",
    analytics_enabled=False,
    elem_id="tripmind-app",
    theme=gr.themes.Base(),
    css_paths=[PROJECT_ROOT / "styles" / "tripmind.css"],
) as demo:

    # =========================
    # Main page container
    # =========================
    with gr.Column(elem_classes=["tm-shell"]):

        # =========================
        # 1. Top navigation / brand
        # =========================
        with gr.Row(elem_classes=["tm-topbar"]):
            gr.Markdown(
                "## ✦ TripMind",
                elem_classes=["tm-brand"],
            )

            gr.Markdown(
                "PERSONALIZED  ·  KNOWLEDGE-GROUNDED  ·  "
                "REALTIME-AWARE",
                elem_classes=["tm-topbar-note"],
            )

        # =========================
        # 2. Hero section
        # =========================
        gr.Markdown(
            """
# Plan less. Explore more.

Your personal AI travel companion. Discover destinations,
create smarter itineraries, and enjoy every journey.
            """,
            elem_classes=["tm-hero"],
        )

        gr.Markdown(
            f"**SUPPORTED DESTINATIONS**  ·  "
            f"{', '.join(service.available_cities)}"
            f"　 |　 **PREFERENCE RANKER**  ·  "
            f"{service.personalization_status}",
            elem_classes=["tm-meta-line"],
        )

        # =========================
        # 3. Main workspace
        # =========================
        with gr.Row(elem_classes=["tm-main-layout"]):

            # -------------------------
            # Left: Trip preferences
            # -------------------------
            with gr.Column(
                scale=1,
                min_width=340,
                elem_classes=["tm-panel", "tm-form-panel"],
            ):

                gr.Markdown(
                    "### Build your trip\n"
                    "Tell us a little about your next journey.",
                    elem_classes=["tm-panel-heading"],
                )

                # Section 01: Destination
                gr.Markdown(
                    "01 / DESTINATION",
                    elem_classes=["tm-section-title"],
                )

                country = gr.Textbox(
                    value="China",
                    label="Country",
                    elem_classes=["tm-field"],
                )

                city = gr.Textbox(
                    value="杭州",
                    label="City",
                    elem_classes=["tm-field"],
                )

                start_date = gr.Textbox(
                    value=date.today().isoformat(),
                    label="Start date (YYYY-MM-DD)",
                    elem_classes=["tm-field"],
                )

                days = gr.Slider(
                    1,
                    5,
                    value=2,
                    step=1,
                    label="Trip duration (days)",
                    elem_classes=["tm-field"],
                )

                # Section 02: Budget and group
                gr.Markdown(
                    "02 / BUDGET & TRAVEL GROUP",
                    elem_classes=["tm-section-title"],
                )

                budget = gr.Slider(
                    300,
                    8000,
                    value=1500,
                    step=100,
                    label="Travel budget",
                    elem_classes=["tm-field"],
                )

                currency = gr.Dropdown(
                    ["CNY", "JPY", "USD", "EUR", "GBP"],
                    value="CNY",
                    label="Currency",
                    elem_classes=["tm-field"],
                )

                age_group = gr.Dropdown(
                    ["<18", "18-25", "26-40", "41-60", "60+"],
                    value="18-25",
                    label="Age group",
                    elem_classes=["tm-field"],
                )

                travel_group = gr.Dropdown(
                    ["solo", "couple", "friends", "family", "parents"],
                    value="couple",
                    label="Travel group",
                    elem_classes=["tm-field"],
                )

                # Section 03: Preferences
                gr.Markdown(
                    "03 / TRAVEL PREFERENCES",
                    elem_classes=["tm-section-title"],
                )

                interests = gr.CheckboxGroup(
                    INTERESTS,
                    value=["photography", "food", "culture"],
                    label="Interests",
                    elem_classes=["tm-field"],
                )

                pace = gr.Radio(
                    ["relaxed", "normal", "intensive"],
                    value="relaxed",
                    label="Travel pace",
                    elem_classes=["tm-field"],
                )

                transport = gr.Radio(
                    ["walking", "driving", "cycling", "transit"],
                    value="walking",
                    label="Transport preference",
                    elem_classes=["tm-field"],
                )

                # Section 04: Personalization
                gr.Markdown(
                    "04 / PERSONALIZATION",
                    elem_classes=["tm-section-title"],
                )

                travel_history = gr.Textbox(
                    value="",
                    lines=3,
                    label="Previously visited / liked places",
                    placeholder="e.g. 故宫, 苏州博物馆, 西安城墙",
                    elem_classes=["tm-field"],
                )

                notes = gr.Textbox(
                    value="avoid_crowds,prefer_night_view",
                    lines=2,
                    label="Special preferences",
                    placeholder="Enter preferences separated by commas",
                    elem_classes=["tm-field"],
                )

                # Main generation button
                submit = gr.Button(
                    "✦ Generate my itinerary",
                    variant="primary",
                    elem_classes=["tm-primary-btn"],
                )

                gr.Markdown(
                    "Your preferences will be used to create a personalized "
                    "travel plan.",
                    elem_classes=["tm-helper-text"],
                )

            # -------------------------
            # Right: Itinerary results
            # -------------------------
            with gr.Column(
                scale=2,
                min_width=420,
                elem_classes=["tm-panel", "tm-results-panel"],
            ):

                with gr.Row(elem_classes=["tm-results-heading"]):

                    gr.Markdown(
                        "### Your itinerary\n"
                        "Your personalized travel plan will appear here.",
                        elem_classes=["tm-panel-heading"],
                    )

                    gr.Markdown(
                        "AI POWERED",
                        elem_classes=["tm-status-label"],
                    )

                # Itinerary text
                output = gr.HTML(
    value=(
        '<div class="tm-empty-state">'
        '<h3>Your journey starts here</h3>'
        '<p>Enter your travel preferences and generate your '
        'personalized itinerary.</p>'
        '</div>'
    ),
    elem_classes=["tm-itinerary-output"],
)

                # POI image gallery
                gallery = gr.Gallery(
    label="Places in your itinerary",
    columns=2,
    object_fit="contain",
    interactive=False,
    format="jpeg",
    visible=False,
    elem_classes=["tm-gallery"],
)

                # Image attribution and notices
                image_details = gr.Markdown(
                    value=IMAGE_MAP_WARNING,
                    elem_classes=["tm-image-details"],
                )

                # Advanced itinerary customization
                with gr.Accordion(
                    "Refine your itinerary",
                    open=False,
                    elem_classes=["tm-accordion"],
                ):

                    gr.Markdown(
                        "Select POIs that you would like to exclude, "
                        "then generate an updated itinerary."
                    )

                    excluded_pois = gr.CheckboxGroup(
                        choices=[],
                        value=[],
                        label="Exclude candidate POIs",
                        elem_classes=["tm-field"],
                    )

                    replan = gr.Button(
                        "Update itinerary",
                        variant="secondary",
                        elem_classes=["tm-secondary-btn"],
                    )

        with gr.Column(elem_classes=["tm-panel"]):
            gr.Markdown("### AI 行程建议", elem_classes=["tm-panel-heading"])
            with gr.Row():
                ai_origin = gr.Textbox(value="上海", label="出发地", elem_classes=["tm-field"])
                ai_people = gr.Slider(1, 10, value=2, step=1, label="人数", elem_classes=["tm-field"])
            ai_button = gr.Button("生成 AI 行程建议", elem_classes=["tm-secondary-btn"])
            ai_status = gr.Markdown("首次生成约 10–30 秒", elem_classes=["tm-helper-text"])
            ai_output = gr.Markdown()

    ai_button.click(
        ai_suggest,
        inputs=[ai_origin, ai_people, city, days, budget, currency, interests, travel_group, pace],
        outputs=[ai_status, ai_output],
        api_name="ai_suggest",
    )

    # =========================
    # 4. Preserve original events
    # =========================
    profile_inputs = [
        country,
        city,
        start_date,
        days,
        budget,
        currency,
        age_group,
        travel_group,
        interests,
        pace,
        transport,
        travel_history,
        notes,
    ]

    submit.click(
        generate,
        inputs=profile_inputs,
        outputs=[
            output,
            excluded_pois,
            gallery,
            image_details,
        ],
        api_name="plan",
    )

    replan.click(
        generate,
        inputs=[*profile_inputs, excluded_pois],
        outputs=[
            output,
            excluded_pois,
            gallery,
            image_details,
        ],
        api_name="replan",
    )


# =========================
# 5. Launch the application
# =========================
if __name__ == "__main__":
    demo.launch()
