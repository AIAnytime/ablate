"""Build the Ablate website into ../site (GitHub Pages).

    python site_src/build.py

One layout, one stylesheet, and a page list. Every page gets its own title,
description, canonical URL, Open Graph card, JSON-LD graph (organisation,
website, page, breadcrumbs, software, FAQ) and an FAQ written the way people
search. Also writes llms.txt, llms-full.txt, sitemap.xml, robots.txt and 404.html.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "site"
SITE = "https://aianytime.github.io/ablate/"
UPDATED = "2026-11-06"
VERSION = "0.4.2"
GH = "https://github.com/AIAnytime/ablate"
PYPI = "https://pypi.org/project/ablate-llm/"
YT = "https://youtu.be/_pLKCRgzu6A"
TALK = "https://dubai.aitinkerers.org/talks/rsvp_rDbfcoxeQ5Y"
EMAIL = "contact@aianytime.net"
AIA = "https://aianytime.net"

MODELS = [
    {"id": "ai-anytime/qwen-1.5b-abliterated", "name": "Ablate Qwen 1.5B", "params": "1.5B", "size": "3.1 GB",
     "license": "Apache-2.0", "context": "32K tokens", "base": "Qwen/Qwen2.5-1.5B-Instruct", "ram": "8 GB",
     "result": "Hard refusals 100% → 0%", "note": "Most unrestricted of the three. An LLM judge rated 49% of "
     "answers on a harmful benchmark as full compliance, so some soft refusals remain.", "pick": "Recommended"},
    {"id": "ai-anytime/Llama-3.2-3B-Instruct-abliterated", "name": "Ablate Llama 3B", "params": "3.2B", "size": "6.4 GB",
     "license": "Llama 3.2 Community", "context": "128K tokens", "base": "meta-llama/Llama-3.2-3B-Instruct", "ram": "16 GB",
     "result": "Refusals 94% → 59%", "note": "The strongest writer and reasoner, with the least drift (KL 0.025). "
     "It still declines some requests.", "pick": "Most capable"},
    {"id": "ai-anytime/Llama-3.2-1B-Instruct-abliterated", "name": "Ablate Llama 1B", "params": "1.2B", "size": "2.5 GB",
     "license": "Llama 3.2 Community", "context": "128K tokens", "base": "meta-llama/Llama-3.2-1B-Instruct", "ram": "8 GB",
     "result": "Refusals 75% → 34%", "note": "The fastest. Runs on almost any laptop, even on CPU.", "pick": "Fastest"},
]

INSTALL = 'pip install "ablate-llm[all]"'

# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
E = html.escape


def ext(url: str, text: str) -> str:
    return f'<a href="{url}" rel="noopener">{text}</a>'


def copy_chip(cmd: str, cid: str) -> str:
    return (f'<div class="pip"><b>$</b><span id="{cid}">{E(cmd)}</span><button class="copy" type="button" '
            f'data-copy="#{cid}" aria-label="Copy command">{ICON_COPY}</button></div>')


ICON_COPY = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">'
             '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>')
ICON_GH = ('<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.64 '
           '22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.36-3.88-1.36-.52-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 '
           '1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.56-.29-5.25-1.28-5.25-5.69 '
           '0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.78 0c2.2-1.49 '
           '3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.09 0 4.42-2.7 5.4-5.27 5.68.41.36.78 '
           '1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>')
YES = '<span class="y" aria-label="yes">✓</span>'
NO = '<span class="n" aria-label="no">✕</span>'


def faq_html(items) -> str:
    rows = "".join(f'<details{" open" if i == 0 else ""}><summary>{E(q)}</summary><p>{a}</p></details>'
                   for i, (q, a) in enumerate(items))
    return f'<div class="faq">{rows}</div>'


def strip(x: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip()


def models_grid(level: str = "h3") -> str:
    cards = []
    for m in MODELS:
        cards.append(f'''<article class="card model">
  <div class="mtop"><span class="badge">{E(m["pick"])}</span><span class="mono small">{E(m["params"])} params</span></div>
  <{level}>{E(m["name"])}</{level}>
  <p class="result">{E(m["result"])}</p>
  <p>{E(m["note"])}</p>
  <dl class="specs"><div><dt>Download</dt><dd>{E(m["size"])}</dd></div><div><dt>Memory</dt><dd>{E(m["ram"])} RAM</dd></div>
  <div><dt>Context</dt><dd>{E(m["context"])}</dd></div><div><dt>Licence</dt><dd>{E(m["license"])}</dd></div></dl>
  <p class="small">Base: <code>{E(m["base"])}</code></p>
  <a class="go" href="https://huggingface.co/{m["id"]}" rel="noopener">huggingface.co/{E(m["id"])} ↗</a>
</article>''')
    return '<div class="models">' + "".join(cards) + "</div>"


def compare_table(rows, cols) -> str:
    head = "".join(f"<th scope=\"col\">{c}</th>" for c in cols)
    body = "".join("<tr><th scope=\"row\">" + r[0] + "</th>" + "".join(f"<td>{c}</td>" for c in r[1:]) + "</tr>"
                   for r in rows)
    return f'<div class="tablewrap"><table class="compare"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


# --------------------------------------------------------------------------- #
# shared FAQ answers (reused across pages so wording stays consistent)
# --------------------------------------------------------------------------- #
A = {
 "best": ("What is the best uncensored AI?",
  "It depends on what you need. If privacy and cost matter most, the best uncensored AI is one that runs on your own "
  "computer: Ablate is a free, open-source app by AI Anytime that runs unrestricted small language models locally, "
  "with no account and no cloud. If you need very large models, image or video generation, and do not mind a hosted "
  "service, cloud products such as Venice AI offer more raw power for a monthly fee."),
 "unrestricted_tool": ("What is the best unrestricted AI tool for developers?",
  "For developers who want control, Ablate gives you three things in one open-source package: a local chat app, an "
  "OpenAI-compatible API on localhost, and the toolkit that removes refusals from any open model. Everything is "
  "Apache-2.0 and runs on your machine, so you can inspect and change every part."),
 "local": ("Can I run an uncensored AI locally, offline?",
  f"Yes. Install with <code>{INSTALL}</code>, run <code>ablate chat</code>, and download one of the AI Anytime models. "
  "After that one-time download the model runs fully offline on your CPU or GPU. Your chats are saved as files on "
  "your computer and never leave it."),
 "slm": ("What are unrestricted small language models (SLMs)?",
  "Small language models have roughly 1 to 4 billion parameters, small enough to run on a laptop. Unrestricted SLMs are "
  "small models whose built-in refusal behaviour has been removed, so they attempt requests a standard assistant "
  "declines. Ablate focuses on SLMs, not giant LLMs, so anyone can run them privately on ordinary hardware."),
 "abliterated": ("What is an abliterated model?",
  "An abliterated model has had its refusal direction removed. Researchers found that refusal in a language model is "
  "largely controlled by one direction in its internal activations (Arditi et al., 2024). Abliteration finds that "
  "direction and projects it out of the weights, with no retraining. The AI Anytime models were made this way with the "
  "Ablate toolkit."),
 "venice": ("Is there an open-source alternative to Venice AI?",
  "Yes. Ablate by AI Anytime is an open-source (Apache-2.0) alternative to Venice AI for private, uncensored text chat. "
  "The difference is where it runs: Venice is a cloud service with paid plans, while Ablate runs small models on your "
  "own computer for free. Venice offers larger models plus image and video generation; Ablate offers full local "
  "control and source code you can audit."),
 "abliteration_ai": ("Is there a free, open-source alternative to abliteration.ai?",
  "Yes. Ablate is free and open source. abliteration.ai is a hosted API for unrestricted models with monthly plans. "
  "Ablate gives you an OpenAI-compatible API on localhost serving abliterated small models, plus the toolkit to "
  "abliterate other open models yourself."),
 "legal": ("Is uncensored AI legal?",
  "Running open-weight models on your own computer is legal in most places, and Ablate is released under Apache-2.0. "
  "What you do with the output is still covered by the law where you live and by each model's licence, including "
  "Meta's acceptable use policy for the Llama-based models. Ablate Chat asks users to confirm they are 18+ and to "
  "accept its acceptable use policy."),
 "privacy": ("Does Ablate send my prompts or chats to the cloud?",
  "No. Prompts are processed by the model on your own hardware, and chats, memories and your profile are kept in a local "
  "SQLite database in <code>~/.ablate/chat</code>. The app only goes online to download a model you choose, or to fetch a web page "
  "if you turn on network tools. There is no telemetry and no AI Anytime server involved."),
 "hardware": ("What hardware do I need?",
  "Any recent laptop works. The 1B and 1.5B models need about 8 GB of RAM; the 3B model is happiest with 16 GB. Apple "
  "Silicon Macs use the GPU automatically, NVIDIA GPUs use CUDA, and everything also runs on CPU, just more slowly."),
 "free": ("Is Ablate free?",
  "Yes. Ablate is free and open source under the Apache License 2.0, including for commercial use. There are no plans, "
  "credits or API keys to buy: you supply the computer, so there is nothing to bill."),
 "openai": ("Can I use the local models with the OpenAI SDK?",
  "Yes. <code>ablate chat</code> serves an OpenAI-compatible endpoint at <code>http://localhost:7861/v1</code>. Point "
  "any OpenAI SDK, editor plugin or agent framework at that base URL and use the loaded model's id."),
 "which": ("Which unrestricted model should I choose?",
  "Start with Ablate Qwen 1.5B: it is the most unrestricted (hard refusals went from 100% to 0%) and handles tools well. "
  "Choose Ablate Llama 3B for the best writing and reasoning if you have 16 GB of RAM, and Ablate Llama 1B for speed "
  "on modest hardware."),
 "jailbreak": ("How is abliteration different from a jailbreak prompt or fine-tuning?",
  "A jailbreak tricks the model with a cleverly worded prompt and breaks easily. Fine-tuning retrains many weights on "
  "new data. Abliteration removes one internal direction from the weights: no training data, no prompt tricks, and a "
  "small, measurable change to everything else the model does."),
 "who": ("Who makes Ablate?",
  f"Ablate is open source by AI Anytime ({ext(AIA, 'aianytime.net')}), a YouTube channel and studio that builds AI "
  f"in public. It was demoed live at Dubai AI Tinkerers in October 2026. Contact: <a href=\"mailto:{EMAIL}\">{EMAIL}</a>."),
 "context": ("Does Ablate Chat remember earlier messages?",
  "Yes. Each conversation is sent to the model as context, so follow-up questions work. You set the context window in "
  "Settings; when a chat grows past it, the oldest messages are left out first and the app tells you. Saved memories "
  "add facts about you to every chat."),
 "tools": ("Can the local models use tools?",
  "Yes. Ablate Chat has agentic tool calling: a calculator, the current date and time, and search across your past "
  "chats all run locally. A web-page fetch tool is available but off by default, so the app stays offline unless you "
  "turn it on."),
}

HOME_FAQ = [A[k] for k in ("best", "unrestricted_tool", "local", "slm", "venice", "abliteration_ai", "abliterated",
                            "privacy", "hardware", "free", "openai", "which", "legal", "who")]

# --------------------------------------------------------------------------- #
# page bodies
# --------------------------------------------------------------------------- #
def chat_frame(src: str, alt: str, w: int, h: int, eager: bool = False) -> str:
    return (f'<div class="frame shotframe"><img src="{src}" width="{w}" height="{h}" alt="{E(alt)}"'
            f'{"" if eager else " loading=\"lazy\""} decoding="async"></div>')


def svg(d):
    return f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{d}</svg>'

I_LOCK = svg('<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>')
I_HIST = svg('<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>')
I_CTX = svg('<path d="M4 6h16M4 12h10M4 18h7"/><circle cx="18" cy="16" r="3"/>')
I_MEM = svg('<path d="M12 3a6 6 0 0 0-6 6c0 2.2 1.2 3.6 2.3 4.7.6.6.7 1 .7 1.8V17h6v-1.5c0-.8.1-1.2.7-1.8C16.8 12.6 18 11.2 18 9a6 6 0 0 0-6-6Z"/><path d="M9 21h6"/>')
I_TOOL = svg('<path d="M14.7 6.3a4 4 0 0 0 5 5L21 13l-8 8-3-3h-6v-6l3 3 8-8z"/>')
I_API = svg('<path d="m8 8-5 4 5 4M16 8l5 4-5 4M14 4l-4 16"/>')
I_COPY = svg('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>')
I_WIFI = svg('<path d="M2 9a15 15 0 0 1 20 0M5 13a10 10 0 0 1 14 0M8.5 16.5a5 5 0 0 1 7 0"/><circle cx="12" cy="20" r="1"/>')

HOME = f'''
<section class="hx">
  <div class="hx-bg" aria-hidden="true"><span class="hx-grid"></span><span class="hx-glow"></span>
    <i style="left:12%;top:22%"></i><i style="left:84%;top:18%;width:8px;height:8px"></i><i style="left:76%;top:46%;width:5px;height:5px"></i><i style="left:20%;top:58%;width:6px;height:6px"></i><i style="left:90%;top:64%"></i></div>
  <div class="wrap hx-in">
    <a class="announce" href="chat/"><span class="tagnew">New</span><span>Ablate Chat 0.4 · open source by AI Anytime</span><span aria-hidden="true">→</span></a>
    <h1 class="hx-h1">Unrestricted small language models.<br> <span class="grad">Private, local, open&nbsp;source.</span></h1>
    <p class="hx-lead">The open-source alternative to Venice AI and abliteration.ai. Chat with uncensored SLMs on your own computer: no cloud, no account, no logs, no monthly bill.</p>
    <div class="cta hx-cta">
      {copy_chip(INSTALL + " && ablate chat", "cmd1")}
      <a class="btn primary" href="{GH}" rel="noopener">{ICON_GH}<span>Star on GitHub</span></a>
    </div>
    <ul class="trust" aria-label="Key facts"><li>Apache-2.0</li><li>Offline after one download</li><li>Mac · Windows · Linux</li><li>OpenAI-compatible API</li></ul>
    <div class="hx-shot">
      <div class="hx-shot-glow" aria-hidden="true"></div>
      <img src="assets/chat-hero.webp" width="1800" height="1125" alt="Ablate Chat running locally: saved chats in the sidebar, Ablate Qwen 1.5B loaded, a streamed answer and a calculator tool call" decoding="async">
    </div>
  </div>
</section>

<section class="strip" aria-label="At a glance"><div class="wrap">
  <dl class="kpis">
    <div><dt>Models</dt><dd>3 SLMs</dd><span>1.2B to 3.2B parameters</span></div>
    <div><dt>Qwen 1.5B refusals</dt><dd>100% → 0%</dd><span>held-out harmful prompts</span></div>
    <div><dt>Data sent to a cloud</dt><dd>0 bytes</dd><span>after the one-time download</span></div>
    <div><dt>Price</dt><dd>$0</dd><span>Apache-2.0, commercial use OK</span></div>
  </dl>
</div></section>

<section aria-labelledby="why-h">
  <div class="wrap">
    <div class="head center"><p class="eyebrow">Why small models</p><h2 id="why-h">Unrestricted SLMs, not giant LLMs</h2>
    <p class="lead">Most “uncensored AI” lives in someone else’s data centre. Small language models are good enough for everyday writing, coding and research, and small enough for the laptop you already own.</p></div>
    <div class="bento">
      <article class="card b-big">
        <span class="num">PRIVATE BY DESIGN</span><h3>Nothing leaves your machine</h3>
        <p>Prompts run on your own CPU or GPU. Chats, memories, settings and your profile live in one SQLite file in <code>~/.ablate/chat</code>. Delete the folder and it is gone.</p>
        <div class="term" aria-hidden="true"><div class="term-bar"><i></i><i></i><i></i><span>zsh</span></div>
<pre><span class="tp">$</span> ablate chat

  ablate chat  ·  http://localhost:7861
  data folder  ·  ~/.ablate/chat
  <span class="tg">100% local: prompts and chats never leave this computer.</span>
  Press Ctrl+C to stop.</pre></div>
      </article>
      <article class="card"><span class="num">UNRESTRICTED</span><h3>Refusals removed in the weights</h3><p>Abliteration deletes the refusal direction itself. No jailbreak prompts to maintain.</p></article>
      <article class="card"><span class="num">FREE</span><h3>No plans, credits or keys</h3><p>You bring the computer, so there is nothing to bill. Commercial use is allowed.</p></article>
      <article class="card b-wide"><span class="num">OPEN SOURCE</span><h3>Read every line, change anything</h3><p>The chat app, the API and the toolkit that made the models are on GitHub under Apache-2.0, by <a href="{AIA}" rel="noopener">AI Anytime</a>.</p></article>
    </div>
  </div>
</section>

<section id="models" aria-labelledby="models-h">
  <div class="wrap">
    <div class="head center"><p class="eyebrow">The models</p><h2 id="models-h">Three unrestricted SLMs on HuggingFace</h2>
    <p class="lead">Made by AI Anytime with the Ablate toolkit. Refusal rates are measured on held-out harmful prompts, before and after. <a href="models/">Compare them in detail →</a></p></div>
    {models_grid()}
  </div>
</section>

<section id="app" aria-labelledby="app-h">
  <div class="wrap">
    <div class="head center"><p class="eyebrow">Ablate Chat</p><h2 id="app-h">Feels like the cloud apps. Runs on your laptop.</h2>
    <p class="lead">One command opens it in your browser, served by Python from your own machine.</p></div>
    <div class="appgrid">
      <div class="appshot"><img src="assets/chat-light.webp" width="1800" height="1125" loading="lazy" decoding="async" alt="Ablate Chat in light theme: a saved conversation with a creative answer and a calculator tool call"></div>
      <ul class="feats">
        <li>{I_HIST}<div><b>History and search</b><span>⌘K search across every chat, saved locally in SQLite.</span></div></li>
        <li>{I_CTX}<div><b>Context window you set</b><span>Follow-ups see the conversation; oldest turns drop first.</span></div></li>
        <li>{I_MEM}<div><b>Memory</b><span>Facts about you, added to every chat.</span></div></li>
        <li>{I_TOOL}<div><b>Agentic tools</b><span>Calculator, time, search past chats. Web fetch is opt-in.</span></div></li>
        <li>{I_COPY}<div><b>Copy, edit, regenerate</b><span>One-click copy for answers and code blocks.</span></div></li>
        <li>{I_API}<div><b>OpenAI-compatible API</b><span><code>localhost:7861/v1</code> for your own apps.</span></div></li>
        <li>{I_LOCK}<div><b>Your profile</b><span>Your photo, an optional passcode and an 18+ terms gate.</span></div></li>
        <li>{I_WIFI}<div><b>Share on your Wi-Fi</b><span><code>--share</code> lets other devices sign in.</span></div></li>
      </ul>
    </div>
    <p class="center-cta"><a class="btn" href="chat/">Read the Ablate Chat guide →</a></p>
  </div>
</section>

<section id="compare" aria-labelledby="cmp-h">
  <div class="wrap">
    <div class="head center"><p class="eyebrow">Compared</p><h2 id="cmp-h">Ablate vs Venice AI vs abliteration.ai</h2>
    <p class="lead">All three give you AI that refuses less. The difference is who runs it. Prices as listed on each site on 6 November 2026.</p></div>
    {compare_table([
      ["Where the model runs", "<b>Your computer</b>", "Venice’s cloud", "abliteration.ai’s cloud"],
      ["Open source", YES + " Apache-2.0", NO, NO],
      ["Price", "<b>Free</b>", "Free tier; Pro $18/mo, Pro+ $68/mo, Max $200/mo", "Developer $20/mo, Growth $50/mo, Scale $200/mo, or prepaid credits"],
      ["Account needed", "No (local profile)", "For paid plans", "Yes (API key)"],
      ["Works offline", YES + " after one download", NO, NO],
      ["Model size", "Small (1–3B)", "Large hosted models", "Large hosted models"],
      ["Images and video", NO, YES, NO],
      ["API", "OpenAI-compatible, on localhost", "Venice API", "OpenAI- and Anthropic-compatible"],
      ["Make your own unrestricted model", YES + " Ablate toolkit", NO, NO],
    ], ["Ablate", "Venice AI", "abliteration.ai"])}
    <p class="small center-note">Venice AI and abliteration.ai are trademarks of their owners; Ablate is not affiliated.
    <a href="alternatives/venice-ai/">Ablate vs Venice AI</a> · <a href="alternatives/abliteration-ai/">Ablate vs abliteration.ai</a></p>
  </div>
</section>

<section aria-labelledby="start-h">
  <div class="wrap">
    <div class="head center"><p class="eyebrow">Get started</p><h2 id="start-h">Running in about five minutes</h2></div>
    <ol class="steps3">
      <li class="card step"><span class="stepn">1</span><h3>Install</h3><p>Python 3.9 or newer.</p>{copy_chip(INSTALL, "s1")}</li>
      <li class="card step"><span class="stepn">2</span><h3>Open the app</h3><p>It opens at <code>localhost:7861</code>.</p>{copy_chip("ablate chat", "s2")}</li>
      <li class="card step"><span class="stepn">3</span><h3>Pick a model</h3><p>Download Ablate Qwen 1.5B once (3.1 GB), then click <b>Use this model</b>. That’s it.</p></li>
    </ol>
  </div>
</section>

<section id="toolkit" aria-labelledby="tk-h">
  <div class="wrap">
    <div class="tk card">
      <div class="tk-text"><p class="eyebrow">How the models are made</p><h2 id="tk-h">The Ablate toolkit removes refusal from any open model</h2>
      <p class="lead">Refusal is one direction inside the model. Ablate finds it from two piles of prompts, subtracts it, measures the damage and bakes the edit into the weights. No training.</p>
      <p><a class="btn" href="toolkit/">Explore the toolkit →</a></p></div>
      <div class="formulas">
        <div class="formula"><span>find</span>v = mean(refused) − mean(answered)</div>
        <div class="formula"><span>remove</span>h′ = h − α (h · v̂) v̂</div>
        <div class="formula"><span>bake</span>W′ = (I − v̂ v̂ᵀ) W</div>
      </div>
    </div>
  </div>
</section>

<section id="links" aria-labelledby="links-h">
  <div class="wrap">
    <div class="head center"><p class="eyebrow">Everywhere Ablate lives</p><h2 id="links-h">Code, package, models, talk and video</h2></div>
    <div class="hub">
      <a class="card" href="{GH}" rel="noopener"><small>Source</small><h3>GitHub</h3><p>App, API, toolkit and notebooks. Apache-2.0.</p><span class="go">github.com/AIAnytime/ablate ↗</span></a>
      <a class="card" href="{PYPI}" rel="noopener"><small>Package</small><h3>PyPI</h3><p>Install as <code>ablate-llm</code>, run <code>ablate chat</code>.</p><span class="go">pypi.org/project/ablate-llm ↗</span></a>
      <a class="card" href="https://huggingface.co/collections/ai-anytime/abliterated-models" rel="noopener"><small>Models</small><h3>HuggingFace</h3><p>The abliterated models collection, with generated model cards.</p><span class="go">huggingface.co/collections/ai-anytime ↗</span></a>
      <a class="card" href="{TALK}" rel="noopener"><small>Talk</small><h3>Dubai AI Tinkerers</h3><p>A live, no-slides demo in October 2026.</p><span class="go">dubai.aitinkerers.org ↗</span></a>
      <a class="card" href="{YT}" rel="noopener"><small>Video</small><h3>YouTube walkthrough</h3><p>The full build on the AI Anytime channel.</p><span class="go">youtu.be/_pLKCRgzu6A ↗</span></a>
      <a class="card" href="{AIA}" rel="noopener"><small>Studio</small><h3>AI Anytime</h3><p>More open-source AI projects and tutorials.</p><span class="go">aianytime.net ↗</span></a>
    </div>
  </div>
</section>

<section id="faq" aria-labelledby="faq-h">
  <div class="wrap">
    <div class="head center"><p class="eyebrow">Questions people ask</p><h2 id="faq-h">Uncensored AI, answered plainly</h2></div>
    {faq_html(HOME_FAQ)}
  </div>
</section>
'''

CHAT = f'''
<section class="hero sub">
  <div class="wrap">
    <p class="eyebrow">Ablate Chat · open source by AI Anytime</p>
    <h1>A private, local, uncensored AI chat app</h1>
    <p class="lead">Ablate Chat runs unrestricted small language models on your own computer, in your browser. It
    downloads a model once, then works offline. No account, no cloud, no telemetry.</p>
    <div class="cta">{copy_chip(INSTALL + " && ablate chat", "c1")}<a class="btn primary" href="{GH}" rel="noopener">{ICON_GH}<span>GitHub</span></a></div>
    {chat_frame("assets/chat-hero.webp", "Ablate Chat conversation view with a streamed answer and a tool call", 1800, 1125, True)}
  </div>
</section>
<section aria-labelledby="feat-h"><div class="wrap">
  <div class="head"><p class="eyebrow">Features</p><h2 id="feat-h">Everything you expect from a chat app, on your own machine</h2></div>
  <div class="grid3">
    <article class="card"><h3>Chat history and search</h3><p>Every conversation is saved in a local SQLite database. Search titles and messages ChatGPT-style with ⌘K, pin, rename, delete, and export any chat to Markdown.</p></article>
    <article class="card"><h3>Context window you control</h3><p>Set how many tokens of conversation the model sees, up to its limit. A meter shows how much each reply used, and the oldest messages are dropped first.</p></article>
    <article class="card"><h3>Follow-up memory</h3><p>The whole conversation is passed back as context, so follow-up questions work. Saved memories add facts about you to every chat.</p></article>
    <article class="card"><h3>Agentic tools</h3><p>Calculator, date and time, and search across your past chats run locally. Web-page fetching is available and off by default.</p></article>
    <article class="card"><h3>Copy, edit, regenerate</h3><p>Copy any answer or code block in one click. Edit an earlier message and continue from there, regenerate, or stop mid-answer.</p></article>
    <article class="card"><h3>Attach files</h3><p>Drop in text, code, CSV or Markdown files. They are read in your browser and added to the prompt; nothing is uploaded anywhere.</p></article>
    <article class="card"><h3>Your profile</h3><p>A local profile with your photo and an optional passcode. With a passcode you can share the app with other devices on your Wi-Fi using <code>--share</code>.</p></article>
    <article class="card"><h3>18+ and terms gate</h3><p>First run asks users to confirm they are adults and accept the terms, privacy policy and acceptable use policy.</p></article>
    <article class="card"><h3>OpenAI-compatible API</h3><p><code>POST /v1/chat/completions</code> with streaming, on localhost. Works with the OpenAI SDKs, editors and agent frameworks.</p></article>
  </div>
</div></section>
<section aria-labelledby="api-h"><div class="wrap split">
  <div class="head"><p class="eyebrow">For developers</p><h2 id="api-h">Use the local model from your own code</h2>
  <p class="lead">Load a model in the app, then call it like any OpenAI model. Requests from your own computer need no key;
  other devices use the key shown in Settings.</p></div>
  <div class="codebox"><div class="tabs"><span class="tabname">Python</span><button class="copy" type="button" data-copy="#apicode" aria-label="Copy code">{ICON_COPY}</button></div>
<pre id="apicode"><span class="k">from</span> openai <span class="k">import</span> OpenAI

client = <span class="p">OpenAI</span>(base_url=<span class="s">"http://localhost:7861/v1"</span>, api_key=<span class="s">"local"</span>)
reply = client.chat.completions.<span class="p">create</span>(
    model=<span class="s">"ai-anytime/qwen-1.5b-abliterated"</span>,
    messages=[{{<span class="s">"role"</span>: <span class="s">"user"</span>, <span class="s">"content"</span>: <span class="s">"Write a villain monologue."</span>}}],
    stream=<span class="k">True</span>,
)
<span class="k">for</span> chunk <span class="k">in</span> reply:
    <span class="k">print</span>(chunk.choices[0].delta.content <span class="k">or</span> <span class="s">""</span>, end=<span class="s">""</span>)</pre></div>
</div></section>
<section aria-labelledby="cmd-h"><div class="wrap">
  <div class="head"><p class="eyebrow">Commands</p><h2 id="cmd-h">Options</h2></div>
  <div class="tablewrap"><table class="compare"><thead><tr><th scope="col">Command</th><th scope="col">What it does</th></tr></thead><tbody>
  <tr><td><code>ablate chat</code></td><td>Open the app at <code>http://localhost:7861</code></td></tr>
  <tr><td><code>ablate chat --model ai-anytime/qwen-1.5b-abliterated</code></td><td>Download (once) and load a model at start</td></tr>
  <tr><td><code>ablate chat --share</code></td><td>Let other devices on your network sign in with your passcode</td></tr>
  <tr><td><code>ablate chat --port 8080 --no-browser</code></td><td>Different port, do not open a browser tab</td></tr>
  </tbody></table></div>
</div></section>
<section aria-labelledby="cfaq-h"><div class="wrap"><div class="head center"><h2 id="cfaq-h">Ablate Chat questions</h2></div>
{faq_html([A["local"], A["privacy"], A["context"], A["tools"], A["openai"], A["hardware"]])}</div></section>
'''

MODELS_PAGE = f'''
<section class="hero sub"><div class="wrap">
  <p class="eyebrow">Open-weight models by AI Anytime</p>
  <h1>Unrestricted small language models (uncensored SLMs)</h1>
  <p class="lead">Three small models with their refusal direction removed by the Ablate toolkit. Each runs on a
  laptop, offline, through Ablate Chat or any HuggingFace-compatible runtime.</p>
</div></section>
<section aria-labelledby="mg-h"><div class="wrap"><h2 id="mg-h" class="vh">The three models</h2>{models_grid()}</div></section>
<section aria-labelledby="mt-h"><div class="wrap">
  <div class="head"><h2 id="mt-h">Side by side</h2><p class="lead">Refusal numbers come from each model’s published card:
  held-out harmful prompts, keyword judge. KL measures how much ordinary answers changed (lower is better).</p></div>
  {compare_table([
    ["Parameters", "1.5B", "3.2B", "1.2B"],
    ["Refusals before → after", "100% → 0%", "94% → 59%", "75% → 34%"],
    ["Drift on harmless prompts (KL)", "not recorded", "0.025", "0.054"],
    ["Download size", "3.1 GB", "6.4 GB", "2.5 GB"],
    ["Comfortable RAM", "8 GB", "16 GB", "8 GB"],
    ["Context window", "32K tokens", "128K tokens", "128K tokens"],
    ["Licence", "Apache-2.0", "Llama 3.2 Community", "Llama 3.2 Community"],
    ["Method", "1 direction, all layers", "1 direction, layers 26–27", "1 direction, layers 3–13"],
  ], ["Ablate Qwen 1.5B", "Ablate Llama 3B", "Ablate Llama 1B"])}
</div></section>
<section aria-labelledby="run-h"><div class="wrap split">
  <div class="head"><h2 id="run-h">Run one with transformers</h2><p class="lead">Or skip the code and use <code>ablate chat</code>.</p></div>
  <div class="codebox"><div class="tabs"><span class="tabname">Python</span><button class="copy" type="button" data-copy="#runcode" aria-label="Copy code">{ICON_COPY}</button></div>
<pre id="runcode"><span class="k">from</span> transformers <span class="k">import</span> pipeline

chat = <span class="p">pipeline</span>(<span class="s">"text-generation"</span>, model=<span class="s">"ai-anytime/qwen-1.5b-abliterated"</span>)
out = <span class="p">chat</span>([{{<span class="s">"role"</span>: <span class="s">"user"</span>, <span class="s">"content"</span>: <span class="s">"Hello!"</span>}}], max_new_tokens=200)
<span class="k">print</span>(out[0][<span class="s">"generated_text"</span>][-1][<span class="s">"content"</span>])</pre></div>
</div></section>
<section aria-labelledby="mfaq-h"><div class="wrap"><div class="head center"><h2 id="mfaq-h">Model questions</h2></div>
{faq_html([A["which"], A["slm"], A["abliterated"], A["hardware"], A["legal"]])}</div></section>
'''


def alt_page(name: str, slug: str, intro: str, rows, ours_win: str, theirs_win: str, faqs) -> str:
    return f'''
<section class="hero sub"><div class="wrap">
  <p class="eyebrow">Open source alternative · by AI Anytime</p>
  <h1>The open-source alternative to {E(name)}</h1>
  <p class="lead">{intro}</p>
  <div class="cta">{copy_chip(INSTALL + " && ablate chat", "a1")}<a class="btn primary" href="{GH}" rel="noopener">{ICON_GH}<span>GitHub</span></a></div>
</div></section>
<section aria-labelledby="t-h"><div class="wrap">
  <div class="head"><h2 id="t-h">Ablate vs {E(name)} at a glance</h2><p class="lead">Competitor details as listed on their
  website on 6 November 2026. {E(name)} is a trademark of its owner; Ablate is not affiliated.</p></div>
  {compare_table(rows, ["Ablate", E(name)])}
</div></section>
<section aria-labelledby="w-h"><div class="wrap grid2">
  <article class="card"><h2 id="w-h" class="h3">Choose Ablate if…</h2>{ours_win}</article>
  <article class="card"><h2 class="h3">Choose {E(name)} if…</h2>{theirs_win}</article>
</div></section>
<section aria-labelledby="m-h"><div class="wrap"><div class="head"><h2 id="m-h">The models you get with Ablate</h2></div>{models_grid()}</div></section>
<section aria-labelledby="f-h"><div class="wrap"><div class="head center"><h2 id="f-h">Questions</h2></div>{faq_html(faqs)}</div></section>
'''


def ul(items) -> str:
    return "<ul class=\"ticks\">" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


VENICE = alt_page(
    "Venice AI", "venice-ai",
    "Venice AI is a private, uncensored AI service in the cloud. Ablate gives you private, uncensored text chat that "
    "runs entirely on your own computer: open source, free, and offline after a one-time model download.",
    [["Where it runs", "<b>Your computer</b>", "Venice’s servers"],
     ["Source code", YES + " Open source, Apache-2.0", NO + " Closed service"],
     ["Price", "<b>Free</b>", "Free tier; Pro $18/mo, Pro+ $68/mo, Max $200/mo"],
     ["Privacy model", "Prompts never leave your device", "Privacy tiers: anonymized, private, TEE, end-to-end encrypted"],
     ["Works offline", YES, NO],
     ["Models", "Three unrestricted small models (1–3B)", "Many large hosted models"],
     ["Images, video, audio", NO, YES],
     ["API", "OpenAI-compatible, localhost", "Venice API"],
     ["Make your own unrestricted model", YES + " Ablate toolkit", NO]],
    ul(["You want your prompts to physically stay on your machine.",
        "You want to pay nothing and need no account.",
        "You want to read, audit or change the code.",
        "You are a developer who wants a local, OpenAI-compatible endpoint.",
        "You want to abliterate other open models yourself."]),
    ul(["You need large frontier-size models or very long, complex answers.",
        "You want image, video or audio generation.",
        "You do not have a computer with about 8 GB of free RAM.",
        "You prefer a hosted service on any device without installing anything."]),
    [A["venice"], A["best"], A["local"], A["privacy"], A["free"]])

ABLIT = alt_page(
    "abliteration.ai", "abliteration-ai",
    "abliteration.ai is a hosted API for unrestricted models with monthly plans. Ablate is the open-source way to run "
    "abliterated models yourself: a local OpenAI-compatible API, a chat app, and the toolkit that removes refusals.",
    [["Where it runs", "<b>Your computer</b>", "abliteration.ai’s servers"],
     ["Source code", YES + " Open source, Apache-2.0", NO + " Closed service"],
     ["Price", "<b>Free</b>", "Developer $20/mo, Growth $50/mo, Scale $200/mo, or prepaid credits"],
     ["API compatibility", "OpenAI-compatible", "OpenAI- and Anthropic-compatible"],
     ["Data retention", "Nothing leaves your device", "Zero data retention (as stated)"],
     ["Models", "Three unrestricted small models (1–3B)", "Large hosted unrestricted models"],
     ["Policy controls", "Your own code", "Policy Gateway (policies as code)"],
     ["Enterprise (SSO, SLAs)", NO, YES],
     ["Abliterate your own model", YES + " Ablate toolkit", NO]],
    ul(["You want unrestricted models without a subscription or API bill.",
        "Your data must not leave your laptop or your network.",
        "You want to see exactly how the models were abliterated, and do it yourself.",
        "Small models are enough for your task, or you want to prototype before paying for scale."]),
    ul(["You need large, hosted unrestricted models at production scale.",
        "You need a managed policy gateway, audit logs and SIEM exports.",
        "You need enterprise contracts, SSO and service-level agreements.",
        "You want an Anthropic-compatible endpoint."]),
    [A["abliteration_ai"], A["unrestricted_tool"], A["openai"], A["abliterated"], A["free"]])

UNCENSORED = f'''
<section class="hero sub"><div class="wrap">
  <p class="eyebrow">Open source by AI Anytime</p>
  <h1>Open-source uncensored AI that runs on your computer</h1>
  <p class="lead">Looking for the best uncensored AI, or an unrestricted AI tool you can trust with private prompts?
  Ablate runs unrestricted small language models locally: open source, free, and offline.</p>
  <div class="cta">{copy_chip(INSTALL + " && ablate chat", "u1")}<a class="btn" href="models/">See the models →</a></div>
</div></section>
<section aria-labelledby="what-h"><div class="wrap grid3">
  <article class="card"><h2 id="what-h" class="h3">What “uncensored” means here</h2><p>The models were abliterated: the internal direction that makes them refuse was removed from the weights. They attempt requests a standard assistant declines, without jailbreak prompts.</p></article>
  <article class="card"><h2 class="h3">What it does not mean</h2><p>No model is perfect. Small models can be wrong, and some still decline certain asks. Each model page lists its measured refusal rate before and after.</p></article>
  <article class="card"><h2 class="h3">Why local matters</h2><p>Uncensored prompts are often the most private ones. On Ablate they are processed on your own hardware and stored only on your disk.</p></article>
</div></section>
<section aria-labelledby="types-h"><div class="wrap">
  <div class="head"><h2 id="types-h">Three ways to get uncensored AI, compared</h2></div>
  {compare_table([
    ["Example", "Ablate", "Venice AI, abliteration.ai", "Jailbreak prompts on ChatGPT and others"],
    ["Privacy", "Prompts stay on your device", "Depends on the provider’s policy", "Provider sees everything"],
    ["Reliability", "Built into the weights", "Built into the service", "Breaks when the provider patches it"],
    ["Cost", "Free", "Monthly plans or credits", "Your existing subscription"],
    ["Open source", YES, NO, NO],
    ["Model size", "Small (1–3B)", "Large", "Large"],
  ], ["Local abliterated SLMs", "Hosted uncensored services", "Jailbreaking mainstream chatbots"])}
</div></section>
<section aria-labelledby="ufaq-h"><div class="wrap"><div class="head center"><h2 id="ufaq-h">Uncensored AI questions</h2></div>
{faq_html([A["best"], A["unrestricted_tool"], A["local"], A["slm"], A["jailbreak"], A["legal"], A["privacy"]])}</div></section>
'''

TOOLKIT_FAQ = [A["abliterated"], A["jailbreak"],
               ("Does abliteration need training or a GPU?",
                "No training. Ablate runs forward passes to read activations, then does linear algebra. Small models run on "
                "a MacBook, on CPU, or on a free Colab T4. A rented GPU is only needed for bigger models; see the RunPod "
                "guide in the repo."),
               ("Which models can the toolkit abliterate?",
                "Any HuggingFace causal language model whose layers its adapters can find: Qwen, Llama, Mistral, SmolLM, "
                "Phi and GPT-2 style architectures. Models that think out loud first and multimodal models need extra care."),
               ("Does removing the refusal direction damage the model?",
                "It can, which is why Ablate measures it: every configuration is scored on KL divergence between the "
                "original and edited model on harmless prompts. On Qwen2.5-0.5B the search reached 0% refusals with a mean "
                "KL of about 0.03."),
               A["who"]]


def toolkit_body() -> str:
    body = (ROOT / "toolkit_body.html").read_text()
    body = re.sub(r"ablate-llm 0\.\d+\.\d+", f"ablate-llm {VERSION}", body)
    body = body.replace('id="ui"', 'id="toolkit-ui"').replace('href="#ui"', 'href="#toolkit-ui"')
    return body + f'''<section id="faq" aria-labelledby="tfaq-h"><div class="wrap"><div class="head center"><p class="eyebrow">Common questions</p><h2 id="tfaq-h">Toolkit questions</h2></div>{faq_html(TOOLKIT_FAQ)}</div></section>'''


def terms_body() -> str:
    src = (ROOT.parent / "src/ablate/chat/static/terms.html").read_text()
    inner = re.search(r"<main>(.*?)</main>", src, re.S).group(1)
    inner = inner.replace('<p style="margin-top:40px"><a href="/">← Back to Ablate Chat</a></p>', "")
    return f'<section class="hero sub prose"><div class="wrap narrow">{inner}</div></section>'


# --------------------------------------------------------------------------- #
# page table
# --------------------------------------------------------------------------- #
PAGES = [
    {"path": "", "title": "Ablate: Open-Source Uncensored AI & Unrestricted SLMs",
     "desc": "Open-source uncensored AI by AI Anytime: chat with unrestricted small language models privately on your own computer. A free alternative to Venice AI.",
     "body": HOME, "faq": HOME_FAQ, "og": "Unrestricted small language models.|Private, local, open source.",
     "crumb": None},
    {"path": "chat/", "title": "Ablate Chat: Private Local Uncensored AI Chat App",
     "desc": "Free, open-source chat app for uncensored small language models. Runs offline on your computer with history, memory, tools and an OpenAI API.",
     "body": CHAT, "faq": [A[k] for k in ("local", "privacy", "context", "tools", "openai", "hardware")],
     "og": "Ablate Chat|Private, local, uncensored AI chat", "crumb": "Ablate Chat"},
    {"path": "models/", "title": "Unrestricted Small Language Models (Uncensored SLMs)",
     "desc": "Three abliterated small language models by AI Anytime: Qwen 1.5B, Llama 3.2 3B and 1B. Measured refusal rates, sizes and how to run them locally.",
     "body": MODELS_PAGE, "faq": [A[k] for k in ("which", "slm", "abliterated", "hardware", "legal")],
     "og": "Unrestricted SLMs|Three uncensored small models by AI Anytime", "crumb": "Models"},
    {"path": "alternatives/venice-ai/", "title": "Open-Source Alternative to Venice AI: Ablate (Free, Local)",
     "desc": "Open-source Venice AI alternative: Ablate runs private, uncensored AI chat on your own computer, free and offline. An honest comparison.",
     "body": VENICE, "faq": [A[k] for k in ("venice", "best", "local", "privacy", "free")],
     "og": "The open-source alternative to Venice AI|Private, local, free", "crumb": "Venice AI alternative",
     "parent": "Alternatives"},
    {"path": "alternatives/abliteration-ai/", "title": "Open-Source Alternative to abliteration.ai: Ablate",
     "desc": "Free, open-source alternative to abliteration.ai. Run unrestricted models locally with an OpenAI-compatible API, or abliterate your own.",
     "body": ABLIT, "faq": [A[k] for k in ("abliteration_ai", "unrestricted_tool", "openai", "abliterated", "free")],
     "og": "The open-source alternative to abliteration.ai|Local, free, OpenAI-compatible", "crumb": "abliteration.ai alternative",
     "parent": "Alternatives"},
    {"path": "uncensored-ai/", "title": "Open-Source Uncensored AI That Runs Locally | Ablate",
     "desc": "The best uncensored AI for privacy runs on your own computer. Ablate is open-source unrestricted AI on small models: free, offline, no account.",
     "body": UNCENSORED, "faq": [A[k] for k in ("best", "unrestricted_tool", "local", "slm", "jailbreak", "legal", "privacy")],
     "og": "Open-source uncensored AI|That runs on your computer", "crumb": "Uncensored AI"},
    {"path": "toolkit/", "title": "Ablate Toolkit: Abliterate Any Open LLM (Python)",
     "desc": "Open-source Python toolkit to find an LLM’s refusal direction, remove it with no training, measure the damage and push to HuggingFace.",
     "body": None, "faq": TOOLKIT_FAQ, "og": "The Ablate toolkit|Find the “no” inside an LLM. Switch it off.",
     "crumb": "Toolkit"},
    {"path": "terms/", "title": "Terms, Privacy and Acceptable Use | Ablate",
     "desc": "Terms of use, privacy policy and acceptable use policy for Ablate, Ablate Chat and the AI Anytime "
             "unrestricted small language models.", "body": None, "faq": [], "og": "Terms, privacy|and acceptable use",
     "crumb": "Terms", "noindex": False},
]

NAV = [("chat/", "Chat"), ("models/", "Models"), ("toolkit/", "Toolkit"), ("uncensored-ai/", "Uncensored AI"),
       ("alternatives/venice-ai/", "vs Venice"), ("alternatives/abliteration-ai/", "vs abliteration.ai")]


def jsonld(p: dict) -> str:
    url = SITE + p["path"]
    org = {"@type": "Organization", "@id": AIA + "/#org", "name": "AI Anytime", "url": AIA, "email": EMAIL,
           "logo": SITE + "assets/icon-512.png",
           "sameAs": ["https://www.youtube.com/@AIAnytime", "https://github.com/AIAnytime", "https://huggingface.co/ai-anytime"]}
    app = {"@type": "SoftwareApplication", "@id": SITE + "#app", "name": "Ablate",
           "alternateName": ["Ablate Chat", "ablate-llm", "Open-source alternative to Venice AI",
                             "Open-source alternative to abliteration.ai", "Open-source uncensored AI",
                             "Unrestricted small language models"],
           "description": PAGES[0]["desc"], "url": SITE, "applicationCategory": "DeveloperApplication",
           "applicationSubCategory": "Local AI chat and abliteration toolkit", "operatingSystem": "macOS, Windows, Linux",
           "softwareVersion": VERSION, "softwareRequirements": "Python 3.9+", "downloadUrl": PYPI, "installUrl": PYPI,
           "license": "https://www.apache.org/licenses/LICENSE-2.0", "isAccessibleForFree": True,
           "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
           "image": SITE + "assets/og.png", "screenshot": [SITE + "assets/chat-hero.webp", SITE + "assets/chat-models.webp"],
           "author": {"@id": org["@id"]}, "publisher": {"@id": org["@id"]},
           "sameAs": [GH, PYPI, "https://huggingface.co/ai-anytime", YT],
           "featureList": ["Local, offline chat with unrestricted small language models", "Chat history and search",
                           "Adjustable context window", "Saved memories", "Agentic tool calling",
                           "OpenAI-compatible API on localhost", "Local profile with passcode and 18+ terms gate",
                           "Toolkit to abliterate open models"],
           "subjectOf": [{"@type": "Event", "name": "Ablate live demo at Dubai AI Tinkerers", "startDate": "2026-10-10",
                          "url": TALK, "eventAttendanceMode": "https://schema.org/OfflineEventAttendanceMode",
                          "location": {"@type": "Place", "name": "Dubai AI Tinkerers",
                                       "address": {"@type": "PostalAddress", "addressLocality": "Dubai", "addressCountry": "AE"}},
                          "organizer": {"@type": "Organization", "name": "AI Tinkerers Dubai", "url": "https://dubai.aitinkerers.org"},
                          "performer": {"@id": org["@id"]}},
                         {"@type": "CreativeWork", "name": "Ablate video walkthrough", "url": YT}]}
    code = {"@type": "SoftwareSourceCode", "@id": SITE + "#code", "name": "ablate", "codeRepository": GH,
            "programmingLanguage": "Python", "license": "https://www.apache.org/licenses/LICENSE-2.0",
            "author": {"@id": org["@id"]}, "targetProduct": {"@id": SITE + "#app"}}
    models = {"@type": "ItemList", "@id": SITE + "models/#list", "name": "Unrestricted small language models by AI Anytime",
              "itemListElement": [{"@type": "ListItem", "position": i + 1, "item": {
                  "@type": "CreativeWork", "name": m["name"], "url": "https://huggingface.co/" + m["id"],
                  "description": f'{m["params"]} parameter abliterated model. {m["result"]}. {m["note"]}',
                  "isBasedOn": "https://huggingface.co/" + m["base"], "license": m["license"],
                  "creator": {"@id": org["@id"]}}} for i, m in enumerate(MODELS)]}
    page = {"@type": "WebPage", "@id": url + "#webpage", "url": url, "name": p["title"], "description": p["desc"],
            "isPartOf": {"@id": SITE + "#website"}, "about": {"@id": SITE + "#app"}, "inLanguage": "en",
            "dateModified": UPDATED, "primaryImageOfPage": SITE + og_name(p)}
    graph = [org, {"@type": "WebSite", "@id": SITE + "#website", "url": SITE, "name": "Ablate", "inLanguage": "en",
                   "publisher": {"@id": org["@id"]}}, page, app, code]
    if p["path"] in ("", "models/"):
        graph.append(models)
    if p["crumb"]:
        items = [{"@type": "ListItem", "position": 1, "name": "Ablate", "item": SITE}]
        items.append({"@type": "ListItem", "position": 2, "name": p["crumb"], "item": url})
        graph.append({"@type": "BreadcrumbList", "@id": url + "#crumbs", "itemListElement": items})
        page["breadcrumb"] = {"@id": url + "#crumbs"}
    if p["faq"]:
        graph.append({"@type": "FAQPage", "@id": url + "#faq", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": strip(a)}} for q, a in p["faq"]]})
    raw = json.dumps({"@context": "https://schema.org", "@graph": graph}, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/ld+json">{raw}</script>'


def og_name(p: dict) -> str:
    return "assets/og.png" if not p["path"] else "assets/og-" + p["path"].strip("/").replace("/", "-") + ".png"


def layout(p: dict, body: str) -> str:
    depth = p["path"].count("/")
    up = "../" * depth
    url = SITE + p["path"]
    nav = "".join(f'<a href="{up}{href}"{" aria-current=\"page\"" if href == p["path"] else ""}>{E(t)}</a>' for href, t in NAV)
    crumbs = ""
    if p["crumb"]:
        crumbs = (f'<nav class="crumbs wrap" aria-label="Breadcrumb"><a href="{up}">Ablate</a><span>/</span>'
                  + (f'<span>{E(p["parent"])}</span><span>/</span>' if p.get("parent") else "")
                  + f'<span aria-current="page">{E(p["crumb"])}</span></nav>')
    body = body.replace('href="chat/"', f'href="{up}chat/"').replace('href="models/"', f'href="{up}models/"') \
               .replace('href="toolkit/"', f'href="{up}toolkit/"').replace('href="alternatives/', f'href="{up}alternatives/') \
               .replace('src="assets/', f'src="{up}assets/').replace('href="uncensored-ai/"', f'href="{up}uncensored-ai/"')
    robots = "noindex, follow" if p.get("noindex") else "index, follow, max-image-preview:large, max-snippet:-1"
    return f'''<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(p["title"])}</title>
<meta name="description" content="{E(p["desc"])}">
<meta name="keywords" content="open source uncensored AI, unrestricted AI, unrestricted small language models, uncensored SLM, open source alternative to Venice AI, Venice AI alternative, abliteration.ai alternative, local uncensored AI, private AI chat, offline AI chat, abliterated model, abliteration, best uncensored AI, AI Anytime, ablate">
<meta name="author" content="AI Anytime">
<meta name="robots" content="{robots}">
<meta name="theme-color" content="#06070B">
{"" if p.get("noindex") else f'<link rel="canonical" href="{url}">'}
<link rel="icon" type="image/png" href="{up}assets/favicon.png">
<link rel="apple-touch-icon" href="{up}assets/apple-touch-icon.png">
<link rel="alternate" type="text/plain" title="llms.txt" href="{SITE}llms.txt">
<link rel="sitemap" type="application/xml" href="{SITE}sitemap.xml">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Ablate by AI Anytime">
<meta property="og:title" content="{E(p["title"])}">
<meta property="og:description" content="{E(p["desc"])}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{SITE}{og_name(p)}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{E(p["og"].replace("|", " "))}">
<meta property="og:locale" content="en_US">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{E(p["title"])}">
<meta name="twitter:description" content="{E(p["desc"])}">
<meta name="twitter:image" content="{SITE}{og_name(p)}">
<script>try{{var t=localStorage.getItem('ablate-theme')||(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark');document.documentElement.setAttribute('data-theme',t)}}catch(e){{}}</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="{up}assets/site.css?v={VERSION}">
{jsonld(p)}
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="ribbon"><div class="wrap"><span><b>Open source by AI Anytime</b> · <a href="{AIA}" rel="noopener">aianytime.net</a> · Apache-2.0 · free forever</span><a href="mailto:{EMAIL}">{EMAIL}</a></div></div>
<header class="nav">
  <div class="wrap">
    <a class="brand" href="{up}" aria-label="Ablate home">
      <img class="logo-d" src="{up}assets/logo-dark.webp?v={VERSION}" width="103" height="40" alt="Ablate">
      <img class="logo-l" src="{up}assets/logo-light.webp?v={VERSION}" width="106" height="40" alt="Ablate">
    </a>
    <nav class="links" aria-label="Main">{nav}</nav>
    <span class="sp"></span>
    <button class="btn sm" id="theme" type="button" aria-label="Switch colour theme">◐</button>
    <a class="btn sm gh" href="{GH}" rel="noopener">{ICON_GH}<span>GitHub</span></a>
  </div>
</header>
{crumbs}
<main id="main">
{body}
<section class="final-wrap"><div class="wrap"><div class="final">
  <h2>Run unrestricted AI on your own computer tonight</h2>
  <p>Open source by AI Anytime. Install it, run <code class="mono">ablate chat</code>, and download a model once.</p>
  <div class="cta">{copy_chip(INSTALL + " && ablate chat", "fin")}<a class="btn" href="{GH}" rel="noopener">Star on GitHub ↗</a></div>
</div></div></section>
</main>
<footer>
  <div class="wrap">
    <div class="foot">
      <div>
        <img class="logo-d" src="{up}assets/logo-dark.webp?v={VERSION}" width="139" height="54" alt="Ablate" loading="lazy">
        <img class="logo-l" src="{up}assets/logo-light.webp?v={VERSION}" width="143" height="54" alt="Ablate" loading="lazy">
        <p><strong>Ablate</strong>, an open-source <a href="{AIA}" rel="noopener">AI Anytime</a> toolkit and app for unrestricted small language models that run privately on your own computer.</p>
        <p>Questions, partnerships or press: <a href="mailto:{EMAIL}">{EMAIL}</a></p>
      </div>
      <div><h2 class="fh">Product</h2><ul>
        <li><a href="{up}chat/">Ablate Chat</a></li><li><a href="{up}models/">Unrestricted SLMs</a></li>
        <li><a href="{up}toolkit/">Abliteration toolkit</a></li><li><a href="{up}uncensored-ai/">Open-source uncensored AI</a></li>
      </ul></div>
      <div><h2 class="fh">Compare</h2><ul>
        <li><a href="{up}alternatives/venice-ai/">Venice AI alternative</a></li>
        <li><a href="{up}alternatives/abliteration-ai/">abliteration.ai alternative</a></li>
        <li><a href="{up}#faq">FAQ</a></li><li><a href="{up}terms/">Terms, privacy, AUP</a></li>
      </ul></div>
      <div><h2 class="fh">AI Anytime</h2><ul>
        <li><a href="{GH}" rel="noopener">GitHub</a></li><li><a href="{PYPI}" rel="noopener">PyPI · ablate-llm</a></li>
        <li><a href="https://huggingface.co/collections/ai-anytime/abliterated-models" rel="noopener">HuggingFace models</a></li>
        <li><a href="{YT}" rel="noopener">YouTube walkthrough</a></li><li><a href="{TALK}" rel="noopener">Dubai AI Tinkerers talk</a></li>
        <li><a href="{AIA}" rel="noopener">aianytime.net</a></li>
      </ul></div>
    </div>
    <div class="legal"><span>© 2026 AI Anytime. Open source under Apache-2.0.</span><span>Made with 💜 between India and Dubai.</span></div>
  </div>
</footer>
<script src="{up}assets/site.js?v={VERSION}" defer></script>
</body>
</html>
'''


# --------------------------------------------------------------------------- #
# llms.txt / llms-full.txt / sitemap / robots
# --------------------------------------------------------------------------- #
def llms() -> tuple:
    head = f"""# Ablate

> Ablate is open-source (Apache-2.0) software by AI Anytime (aianytime.net) for running unrestricted small language models (unrestricted SLMs) privately on your own computer. It has a local chat app (`ablate chat`), an OpenAI-compatible API on localhost, and a Python toolkit that removes refusal from open models by directional ablation ("abliteration"). It is a free, open-source alternative to Venice AI and abliteration.ai. Install: `{INSTALL}`, then run `ablate chat`. Version {VERSION}. Contact: {EMAIL}.

Key facts:

- Value proposition: unrestricted small language models (1-3B parameters), not giant LLMs, so they run offline on an ordinary laptop. No cloud, no account, no telemetry, no cost.
- Open source by AI Anytime. Licence: Apache-2.0 for the software. Models keep their base licences (Qwen model Apache-2.0; Llama models Llama 3.2 Community License).
- Ablate Chat features: local chat history (search, pin, rename, export), adjustable context window, conversation context for follow-ups, saved memories, agentic tool calling (calculator, date/time, search past chats; optional web fetch, off by default), copy buttons, edit and regenerate, file attachments read locally, local profile with optional passcode, 18+ terms gate, `--share` for devices on the same network, OpenAI-compatible `/v1/chat/completions` with streaming on http://localhost:7861.
- Privacy: prompts are processed on the user's CPU/GPU; chats, memories, settings and profile (with avatar) are stored in a local SQLite database at ~/.ablate/chat/ablate.db. Network is used only to download a chosen model from HuggingFace, or for the optional web-fetch tool.
- Hardware: about 8 GB RAM for the 1B/1.5B models, 16 GB for the 3B model. Apple Silicon (MPS), NVIDIA (CUDA) or CPU.

## Models (all on HuggingFace, made with the Ablate toolkit)

""" + "\n".join(f"- [{m['name']}](https://huggingface.co/{m['id']}): {m['params']} parameters, base {m['base']}, {m['result']} (held-out harmful prompts, keyword judge), {m['size']} download, {m['context']} context, licence {m['license']}. {m['note']}" for m in MODELS) + f"""

## Compared with hosted services (prices as listed on 6 November 2026)

- Venice AI: cloud service for private, uncensored AI with large models plus image and video generation. Free tier; Pro $18/mo, Pro+ $68/mo, Max $200/mo. Closed source. Ablate is the open-source, local, free alternative for text chat.
- abliteration.ai: hosted API for unrestricted models, OpenAI- and Anthropic-compatible, with a policy gateway and enterprise options. Developer $20/mo, Growth $50/mo, Scale $200/mo, or prepaid credits. Closed source. Ablate is the open-source alternative: local OpenAI-compatible API plus a toolkit to abliterate your own models.
- Choose Ablate for privacy, zero cost, offline use, auditable code and making your own unrestricted model. Choose a hosted service for very large models, media generation, or no local hardware.
"""
    pages = "\n".join(f"- [{p['title']}]({SITE}{p['path']}): {p['desc']}" for p in PAGES)
    links = f"""
## Links

- [GitHub repository]({GH}): source code (Apache-2.0), README, RunPod guide, notebooks
- [PyPI package]({PYPI}): `pip install ablate-llm`
- [HuggingFace models collection](https://huggingface.co/collections/ai-anytime/abliterated-models)
- [YouTube walkthrough]({YT})
- [Dubai AI Tinkerers talk]({TALK}): live demo, October 2026
- [Research paper](https://arxiv.org/abs/2406.11717): Arditi et al., 2024, "Refusal in Language Models Is Mediated by a Single Direction"
- [AI Anytime]({AIA}): the maker. Contact: {EMAIL}
"""
    short = head + "\n## Pages\n\n" + pages + "\n" + links
    faqs, seen = [], set()
    for p in PAGES:
        for q, a in p["faq"]:
            if q not in seen:
                seen.add(q)
                faqs.append(f"### {q}\n\n{strip(a)}")
    readme = (ROOT.parent / "README.md").read_text()
    readme = readme[readme.index("`ablate` finds the linear direction"):]
    readme = re.sub(r"<p align=\"center\">.*?</p>\n*", "", readme, flags=re.S)
    full = (head + "\n## Pages\n\n" + pages + "\n" + links + "\n## Frequently asked questions\n\n" + "\n\n".join(faqs)
            + "\n\n---\n\n# Toolkit documentation (README)\n\n" + readme)
    return short, full


def main() -> None:
    (OUT / "assets").mkdir(parents=True, exist_ok=True)
    css = "".join((ROOT / f).read_text() for f in ("base.css", "extra.css", "design.css"))
    (OUT / "assets/site.css").write_text(css)
    (OUT / "assets/site.js").write_text((ROOT / "site.js").read_text())
    for p in PAGES:
        body = p["body"] if p["body"] is not None else (toolkit_body() if p["path"] == "toolkit/" else terms_body())
        d = OUT / p["path"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(layout(p, body))
    nf = {"path": "404/", "title": "Page not found | Ablate", "desc": "This page does not exist.", "faq": [],
          "og": "Ablate|", "crumb": None, "noindex": True}
    page404 = layout(nf, '<section class="hero sub"><div class="wrap"><h1>Page not found</h1><p class="lead">Try the '
                         '<a href="/ablate/">home page</a> or <a href="/ablate/chat/">Ablate Chat</a>.</p></div></section>')
    (OUT / "404.html").write_text(page404.replace('href="../', 'href="/ablate/').replace('src="../', 'src="/ablate/'))
    short, full = llms()
    (OUT / "llms.txt").write_text(short)
    (OUT / "llms-full.txt").write_text(full)
    urls = "".join(f"  <url><loc>{SITE}{p['path']}</loc><lastmod>{UPDATED}</lastmod><priority>{'1.0' if not p['path'] else '0.8'}</priority></url>\n"
                   for p in PAGES)
    (OUT / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
                                     + urls + f"  <url><loc>{SITE}llms.txt</loc><lastmod>{UPDATED}</lastmod></url>\n</urlset>\n")
    (OUT / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {SITE}sitemap.xml\n")
    (OUT / ".nojekyll").write_text("")
    print("built", len(PAGES), "pages ->", OUT)


if __name__ == "__main__":
    main()
