from __future__ import annotations
from datetime import date
import os,re
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED","False")
import gradio as gr
from tripmind.models import UserProfile
from tripmind.service import TripMindService,itinerary_to_markdown
service=TripMindService();INTERESTS=["photography","food","culture","history","nature","museum","architecture","night_view","shopping"]
def history_items(v):return [x.strip() for x in re.split(r"[\n,;，；]+",v or "") if x.strip()]
def generate(country,city,start_date,days,budget,currency,age_group,travel_group,interests,pace,transport,travel_history,notes):
    try:
        p=UserProfile(country=(country or "China").strip(),city=city.strip() or "杭州",start_date=start_date.strip() or None,days=int(days),budget=float(budget),currency=currency,age_group=age_group,travel_group=travel_group,interests=interests or ["culture"],pace=pace,transport_preference=transport,travel_history=history_items(travel_history),special_preferences=[x.strip() for x in notes.split(",") if x.strip()]);return itinerary_to_markdown(service.plan(p))
    except Exception as exc:return f"### Planning failed\n\n`{type(exc).__name__}: {exc}`"
with gr.Blocks(title="TripMind",analytics_enabled=False) as demo:
    gr.Markdown(f"# TripMind\nPersonalized · Knowledge-grounded · Realtime-aware · Constraint-aware\n\nPreference ranker at startup: **{service.personalization_status}**.");gr.Markdown("Available cities: "+", ".join(service.available_cities))
    with gr.Row():
        with gr.Column(scale=1):
            country=gr.Textbox(value="China",label="Country");city=gr.Textbox(value="杭州",label="City");start_date=gr.Textbox(value=date.today().isoformat(),label="Start date (YYYY-MM-DD)");days=gr.Slider(1,5,value=2,step=1,label="Days");budget=gr.Slider(300,8000,value=1500,step=100,label="Budget");currency=gr.Dropdown(["CNY","JPY","USD","EUR","GBP"],value="CNY",label="Currency");age_group=gr.Dropdown(["<18","18-25","26-40","41-60","60+"],value="18-25",label="Age group");travel_group=gr.Dropdown(["solo","couple","friends","family","parents"],value="couple",label="Travel group");interests=gr.CheckboxGroup(INTERESTS,value=["photography","food","culture"],label="Interests");pace=gr.Radio(["relaxed","normal","intensive"],value="relaxed",label="Pace");transport=gr.Radio(["walking","driving","cycling","transit"],value="walking",label="Transport preference");travel_history=gr.Textbox(value="",lines=3,label="Previously visited/liked places (optional)",placeholder="e.g. 故宫, 苏州博物馆, 西安城墙");notes=gr.Textbox(value="avoid_crowds,prefer_night_view",label="Special preferences (comma separated)");submit=gr.Button("Generate itinerary",variant="primary")
        with gr.Column(scale=2):output=gr.Markdown()
    submit.click(generate,inputs=[country,city,start_date,days,budget,currency,age_group,travel_group,interests,pace,transport,travel_history,notes],outputs=output,api_name="plan")
if __name__=="__main__":demo.launch()
