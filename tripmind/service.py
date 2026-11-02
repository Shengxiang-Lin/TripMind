from __future__ import annotations
from dataclasses import replace
import json,math
from datetime import date
from typing import Any,List,Mapping
from .config import get_config,resolve_project_path
from .bundle import resolve_runtime_data
from .processing.common import normalized
from .data import city_pois,load_pois
from .map_provider import FallbackMapProvider,LocalMapProvider,MapProvider
from .models import Itinerary,RealtimeSignal,UserProfile
from .planner import GreedyConstraintPlanner
from .rag import LocalKnowledgeRetriever
from .realtime import CachingRealtimeProvider,DashScopeRealtimeProvider,NullRealtimeProvider,RealtimeProvider
from .recommender import HeuristicPersonalizedRanker,LearnedPersonalizedRanker,RealtimeAdjustmentEngine

def build_map_provider(config=None):
    config=dict(config or get_config());mode=str(config.get("providers",{}).get("route","auto")).strip().lower();ors=config.get("ors",{});key=str(ors.get("api_key","")).strip()
    if mode in {"auto","ors"} and key:
        from .integrations.openrouteservice import OpenRouteServiceProvider
        return FallbackMapProvider(OpenRouteServiceProvider(api_key=key,base_url=ors.get("base_url"),timeout=float(ors.get("timeout_seconds",15)),config=config),LocalMapProvider())
    if mode=="ors":raise RuntimeError("providers.route=ors but no ORS key is configured.")
    return LocalMapProvider()
def build_realtime_provider(config=None):
    config=dict(config or get_config());mode=str(config.get("providers",{}).get("realtime","auto")).strip().lower();key=str(config.get("dashscope",{}).get("api_key","")).strip()
    if mode in {"auto","dashscope"} and key:return CachingRealtimeProvider(DashScopeRealtimeProvider(config=config),ttl_seconds=int(config.get("realtime",{}).get("ttl_seconds",3600)))
    if mode=="dashscope":raise RuntimeError("providers.realtime=dashscope but no DashScope key is configured.")
    return NullRealtimeProvider()
def build_personalized_ranker(config=None):
    config=dict(config or get_config());mode=str(config.get("providers",{}).get("preference","auto")).strip().lower()
    if mode not in {"auto","heuristic","learned"}:raise ValueError("providers.preference must be auto, heuristic, or learned")
    if mode=="heuristic":return HeuristicPersonalizedRanker(),"heuristic",None
    mc=config.get("preference_model",{});cp=resolve_project_path(mc.get("checkpoint","checkpoints/tripmind-preference/best"));cf=cp/"preference_config.json"
    if not cf.is_file():
        if mode=="learned":raise FileNotFoundError(f"Learned preference checkpoint not found: {cf}")
        return HeuristicPersonalizedRanker(),"heuristic",f"Learned preference checkpoint not found; using heuristic ranker ({cp})."
    try:
        r=LearnedPersonalizedRanker.from_checkpoint(cp,device=str(mc.get("device","auto")),batch_size=int(mc.get("inference_batch_size",8)),max_length=int(mc["max_length"]) if mc.get("max_length") not in (None,"") else None,history_char_limit=int(mc["history_char_limit"]) if mc.get("history_char_limit") not in (None,"") else None,heuristic_blend=float(mc.get("heuristic_blend",.15)));return r,"learned-genup",None
    except Exception as exc:
        if mode=="learned":raise
        return HeuristicPersonalizedRanker(),"heuristic",f"Learned preference model failed to load ({type(exc).__name__}); using heuristic ranker."
class TripMindService:
    def __init__(self,map_provider=None,realtime_provider=None,config=None):
        self.config=dict(config or get_config());d=resolve_runtime_data(self.config.get("data",{}));self.data_bundle_id=d.get("bundle_id");af=d.get("city_aliases_file");self.city_registry=json.loads(resolve_project_path(af).read_text()) if af else {};self.city_lookup={normalized(a):c for c,m in self.city_registry.items() for a in m["aliases"]};poi_dir=resolve_project_path(d.get("poi_dir","data/poi"));kf=resolve_project_path(d.get("knowledge_file","data/knowledge/travel_knowledge.json"));self.pois=load_pois(poi_dir)
        if d.get("knowledge_index"):
            from .vector_rag import load_vector_retriever
            self.rag=load_vector_retriever(str(resolve_project_path(d["knowledge_index"])))
        else:self.rag=LocalKnowledgeRetriever(kf)
        self.available_cities=sorted({p.city for p in self.pois});self.ranker,self.personalization_status,self.personalization_notice=build_personalized_ranker(self.config);self.preference_mode=str(self.config.get("providers",{}).get("preference","auto")).strip().lower();self.realtime_adjuster=RealtimeAdjustmentEngine();self.map=map_provider or build_map_provider(self.config);self.realtime=realtime_provider or build_realtime_provider(self.config);self.planner=GreedyConstraintPlanner(self.map)
    def plan(self,profile):
        city=self.city_lookup.get(normalized(profile.city),profile.city);country="China" if normalized(profile.country) in {"china","中国","中华人民共和国","prc",""} else profile.country;profile=replace(profile,city=city,country=country)
        if isinstance(profile.days,bool) or not isinstance(profile.days,int) or profile.days<1:raise ValueError("Trip days must be a positive integer.")
        if not math.isfinite(profile.budget) or profile.budget<0:raise ValueError("Budget must be a finite nonnegative number.")
        if profile.start_date:date.fromisoformat(profile.start_date)
        candidates=city_pois(self.pois,profile.city,profile.country)
        if not candidates:raise ValueError(f"No POIs available for city={profile.city!r}, country={profile.country!r} in the current local dataset.")
        if any(p.currency!=profile.currency for p in candidates):raise ValueError("Budget currency must match the POI currency; exchange-rate conversion is not configured.")
        docs=self.rag.retrieve_documents(profile,top_k=5);knowledge=[x["text"] for x in docs];sources=[{k:x.get(k) for k in ["doc_id","title","source","license","language","score","city"]} for x in docs if x.get("source")];warnings=[]
        if self.personalization_notice:warnings.append(self.personalization_notice)
        status=self.personalization_status
        try:ranked=self.ranker.rank(profile,candidates,knowledge)
        except Exception as exc:
            if self.preference_mode!="auto" or status!="learned-genup":raise
            ranked=HeuristicPersonalizedRanker().rank(profile,candidates,knowledge);status="heuristic-fallback";warnings.append(f"Learned preference scoring failed ({type(exc).__name__}); this request used the heuristic ranker.")
        rt=[x.poi for x in ranked[:min(12,len(ranked))]];rt_errors=[]
        try:signals=self.realtime.fetch(profile,rt);rt_errors.extend(getattr(self.realtime,"last_errors",[]))
        except Exception as exc:signals=[];rt_errors.append(f"Realtime lookup failed ({type(exc).__name__}).")
        warnings.extend(rt_errors);rt_status="disabled" if isinstance(self.realtime,NullRealtimeProvider) else ("partial" if signals else "failed") if rt_errors else "ok";ranked=self.realtime_adjuster.apply(profile,ranked,signals);blocked={s.subject for s in signals if s.hard_constraint and s.effect=="avoid"};days=self.planner.plan(profile,ranked,blocked_pois=blocked);cost=round(sum(d.estimated_cost for d in days),2)
        return Itinerary(profile=profile,days=days,total_estimated_cost=cost,retrieved_knowledge=knowledge,knowledge_sources=sources,realtime_signals=signals,warnings=warnings,realtime_status=rt_status,personalization_status=status)
def minute_str(v):return f"{v//60:02d}:{v%60:02d}"
def itinerary_to_markdown(it):
    p=it.profile;loc=f"{p.city}, {p.country}" if p.country else p.city;lines=[f"# {loc} · {p.days}-Day Personalized Itinerary",f"Profile: **{p.travel_group}** · **{p.pace}** · interests: "+", ".join(p.interests),f"Personalization: **{it.personalization_status}**",f"Budget: {p.currency} {p.budget:.0f} · currently planned ticket cost: {p.currency} {it.total_estimated_cost:.0f}",""]
    if p.travel_history:lines.insert(3,"Preference history: "+", ".join(p.travel_history[:8]))
    if it.warnings:lines += ["## Planning notices",*[f"- {x}" for x in it.warnings],""]
    for day in it.days:
        lines.append(f"## Day {day.day}")
        if not day.items:lines.append("_No feasible POIs under current constraints._");continue
        for x in day.items:
            if x.travel_from_previous:lines.append(f"↳ {x.travel_from_previous.mode}: **{x.travel_from_previous.distance_km:.1f} km**, about **{x.travel_from_previous.duration_min} min**")
            lines.append(f"- **{minute_str(x.start_min)}–{minute_str(x.end_min)}** {x.poi.name} · {x.poi.currency} {x.poi.ticket_price:.0f} · preference {x.score:.2f}  \n  {x.poi.description}")
        lines.append(f"\nDay travel: **{day.travel_distance_km:.1f} km / {day.travel_duration_min} min**")
    return "\n".join(lines)
