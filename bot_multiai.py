#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TALOS - Multi-AI Hub pour Discord (Mistral, Claude, Gemini)
Hackathon Robot Humanoïde Défense 2026 (AID / Agorize / Unitree G1 & Go2)
=============================================================================
Ce bot connecte vos abonnements et clés IA directement à Discord :
- #🤖・mistral-ai -> Route automatiquement vers Mistral AI (Large / Codestral)
- #🤖・claude-ai  -> Route automatiquement vers Anthropic Claude (3.5 Sonnet / 3.7)
- #🤖・gemini-ai  -> Route automatiquement vers Google Gemini (2.5 Pro / Flash)
=============================================================================
"""

import os
import sys
import asyncio
import discord

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# --- CONFIGURATION TOKENS & CLÉS API ---
DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Clients paresseux (lazy-loaded)
mistral_client = None
anthropic_client = None
gemini_client = None

SYSTEM_PROMPT = (
    "Tu es l'assistant IA technique officiel de l'équipe d'ingénieurs TALOS "
    "(Hackathon Robot Humanoïde Défense 2026 organisé par l'AID et Agorize). "
    "Tu es un expert en :\n"
    "- Robotique humanoïde (Unitree G1, 29 DDL, mains Dex3-1) et quadrupède (Unitree Go2)\n"
    "- Simulation physique avancée (MuJoCo, Isaac Gym, PyBullet)\n"
    "- Middleware robotique (ROS 2 Humble, DDS, RTOS, micro-ROS)\n"
    "- Contrôle moteur, apprentissage par renforcement (RL, PPO, SAC) et cinématique\n"
    "- SLAM 3D, fusion Lidar/Caméras de profondeur et franchissement d'obstacles tactiques\n"
    "- Développement bas niveau C++20, firmware STM32/Raspberry Pi et Python (SDK2 Unitree)\n\n"
    "Réponds toujours avec rigueur, concision et pertinence technique en français. "
    "Fournis des exemples de code commentés, robustes et optimisés lorsque nécessaire."
)

# Mémoire de conversation par channel_id : [ {"role": ..., "content": ...} ]
conversations = {}
MAX_HISTORY = 10


def get_mistral():
    global mistral_client
    if not mistral_client and MISTRAL_API_KEY:
        from mistralai import Mistral
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    return mistral_client


def get_anthropic():
    global anthropic_client
    if not anthropic_client and ANTHROPIC_API_KEY:
        import anthropic
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return anthropic_client


def get_gemini():
    global gemini_client
    if not gemini_client and GEMINI_API_KEY:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return gemini_client


def split_message(text, max_len=1950):
    chunks = []
    while len(text) > max_len:
        split_idx = text.rfind("\n", 0, max_len)
        if split_idx == -1:
            split_idx = max_len
        chunks.append(text[:split_idx])
        text = text[split_idx:].lstrip()
    if text:
        chunks.append(text)
    return chunks


async def call_mistral(history):
    client = get_mistral()
    if not client:
        return "⚠️ **Clé API Mistral non configurée !** Renseignez `MISTRAL_API_KEY`."
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    response = await asyncio.to_thread(
        client.chat.complete,
        model="mistral-large-latest",
        messages=msgs,
    )
    return response.choices[0].message.content


async def call_claude(history):
    client = get_anthropic()
    if not client:
        return "⚠️ **Clé API Anthropic (Claude) non configurée !** Renseignez `ANTHROPIC_API_KEY`."
    msgs = []
    for m in history:
        if m["role"] in ["user", "assistant"]:
            msgs.append({"role": m["role"], "content": m["content"]})
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=msgs,
    )
    return response.content[0].text


async def call_gemini(history):
    client = get_gemini()
    if not client:
        return "⚠️ **Clé API Google Gemini non configurée !** Renseignez `GEMINI_API_KEY`."
    contents = []
    for m in history:
        if m["role"] == "user":
            contents.append(f"Utilisateur : {m['content']}")
        elif m["role"] == "assistant":
            contents.append(f"Assistant : {m['content']}")

    full_prompt = f"System Instruction: {SYSTEM_PROMPT}\n\n" + "\n\n".join(contents)
    response = await asyncio.to_thread(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=full_prompt,
    )
    return response.text


intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"\n[+] Bot TALOS Multi-AI connecté : {client.user} (ID: {client.user.id})")
    print("=" * 65)
    print("  STATUS DES MODÈLES D'INTELLIGENCE ARTIFICIELLE :")
    print(f"  • Mistral AI  : {'[OK] Clé configurée' if MISTRAL_API_KEY else '[!] Clé en attente'}")
    print(f"  • Claude AI   : {'[OK] Clé configurée' if ANTHROPIC_API_KEY else '[!] Clé en attente'}")
    print(f"  • Gemini AI   : {'[OK] Clé configurée' if GEMINI_API_KEY else '[!] Clé en attente'}")
    print("=" * 65)
    print("  EN ATTENTE DES MESSAGES SUR DISCORD...\n")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    ch_name = getattr(message.channel, "name", "").lower()
    is_mistral = "mistral" in ch_name
    is_claude = "claude" in ch_name
    is_gemini = "gemini" in ch_name
    is_mentioned = client.user in message.mentions

    if not (is_mistral or is_claude or is_gemini or is_mentioned):
        return

    clean_content = message.clean_content.replace(f"@{client.user.name}", "").strip()
    if not clean_content:
        return

    # Routage du modèle selon le salon
    ai_provider = "mistral"
    if is_claude:
        ai_provider = "claude"
    elif is_gemini:
        ai_provider = "gemini"
    elif is_mistral:
        ai_provider = "mistral"
    elif is_mentioned:
        # Si mentionné en dehors, par défaut Mistral ou Claude
        if ANTHROPIC_API_KEY:
            ai_provider = "claude"
        elif MISTRAL_API_KEY:
            ai_provider = "mistral"
        elif GEMINI_API_KEY:
            ai_provider = "gemini"

    # Gestion de l'historique
    cid = message.channel.id
    if cid not in conversations:
        conversations[cid] = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversations[cid].append({"role": "user", "content": clean_content})
    if len(conversations[cid]) > (MAX_HISTORY * 2 + 1):
        conversations[cid] = [conversations[cid][0]] + conversations[cid][-(MAX_HISTORY * 2):]

    async with message.channel.typing():
        try:
            if ai_provider == "mistral":
                reply = await call_mistral(conversations[cid])
            elif ai_provider == "claude":
                reply = await call_claude(conversations[cid])
            elif ai_provider == "gemini":
                reply = await call_gemini(conversations[cid])
            else:
                reply = "Aucune clé d'IA n'est actuellement configurée."

            conversations[cid].append({"role": "assistant", "content": reply})

            chunks = split_message(reply)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)

        except Exception as e:
            print(f"[-] Erreur {ai_provider} : {e}")
            await message.reply(f"❌ **Erreur ({ai_provider.capitalize()})** : `{e}`")


def main():
    global MISTRAL_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
    print("=" * 65)
    print("  TALOS MULTI-AI DISCORD BOT (Mistral, Claude, Gemini)")
    print("=" * 65)

    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
