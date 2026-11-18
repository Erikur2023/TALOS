#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TALOS - Assistant Mistral AI pour Discord
Hackathon Robot Humanoïde Défense (AID / Agorize / Unitree G1 & Go2)
=============================================================================
Ce bot connecte votre abonnement / clé Mistral AI directement à Discord :
- Répond automatiquement dans le salon #🤖・mistral-ai (ou en mention)
- Expert en Robotique (Unitree G1, Go2, MuJoCo, ROS 2, RL, C++, Python)
- Gestion de la mémoire de conversation par fil / salon
- Découpage automatique des réponses longues (> 2000 caractères)
=============================================================================
"""

import os
import sys
import asyncio
import discord
from mistralai import Mistral

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
TARGET_CHANNEL_NAME = "mistral-ai"
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")

SYSTEM_PROMPT = (
    "Tu es l'assistant IA technique officiel de l'équipe d'ingénieurs TALOS "
    "(Hackathon Robot Humanoïde Défense 2026 organisé par l'AID et Agorize). "
    "Tu es un expert mondial en :\n"
    "- Robotique humanoïde (Unitree G1, 29 DDL, mains Dex3-1) et quadrupède (Unitree Go2)\n"
    "- Simulation physique avancée (MuJoCo, Isaac Gym, PyBullet)\n"
    "- Middleware robotique (ROS 2 Humble/Iron, DDS, micro-ROS, RTOS)\n"
    "- Contrôle moteur, apprentissage par renforcement (RL, PPO, SAC) et cinématique inverse\n"
    "- SLAM 3D, fusion capteurs Lidar/Caméras de profondeur et franchissement d'obstacles tactiques\n"
    "- Développement bas niveau C++20, firmware STM32/Raspberry Pi et Python 3.12 (SDK2 Unitree)\n\n"
    "Réponds toujours avec rigueur, concision et pertinence technique en français. "
    "Fournis des exemples de code commentés, robustes et optimisés lorsque nécessaire."
)

# Historique local de conversation (par channel_id)
conversation_history = {}
MAX_HISTORY_MESSAGES = 10


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


intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"\n[+] Bot TALOS Mistral connecté : {client.user} (ID: {client.user.id})")
    print(f"[*] Modèle Mistral configuré : {MISTRAL_MODEL}")
    print(f"[*] Salon cible : #{TARGET_CHANNEL_NAME}")
    print("=" * 60)
    print("  EN ATTENTE DE MESSAGES DANS DISCORD...")
    print("=" * 60 + "\n")


@client.event
async def on_message(message: discord.Message):
    # Ignorer ses propres messages et ceux des autres bots
    if message.author.bot:
        return

    is_mistral_channel = TARGET_CHANNEL_NAME in message.channel.name
    is_mentioned = client.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)

    if not (is_mistral_channel or is_mentioned or is_dm):
        return

    clean_content = message.clean_content.replace(f"@{client.user.name}", "").strip()
    if not clean_content:
        return

    if not MISTRAL_API_KEY:
        await message.reply(
            "⚠️ **Clé API Mistral manquante !**\n"
            "Pour que je puisse répondre avec ton abonnement Pro, définis la variable d'environnement `MISTRAL_API_KEY` "
            "ou lance le script avec ta clé API récupérée sur **https://console.mistral.ai/api-keys/**."
        )
        return

    # Gestion de l'historique
    channel_id = message.channel.id
    if channel_id not in conversation_history:
        conversation_history[channel_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    conversation_history[channel_id].append({"role": "user", "content": clean_content})

    # Limiter la taille de l'historique
    if len(conversation_history[channel_id]) > (MAX_HISTORY_MESSAGES * 2 + 1):
        conversation_history[channel_id] = [
            conversation_history[channel_id][0]
        ] + conversation_history[channel_id][-(MAX_HISTORY_MESSAGES * 2):]

    async with message.channel.typing():
        try:
            mistral_client = Mistral(api_key=MISTRAL_API_KEY)
            response = await asyncio.to_thread(
                mistral_client.chat.complete,
                model=MISTRAL_MODEL,
                messages=conversation_history[channel_id],
            )
            reply_text = response.choices[0].message.content

            # Enregistrer la réponse dans l'historique
            conversation_history[channel_id].append({"role": "assistant", "content": reply_text})

            # Envoyer la réponse (découpée si > 2000 caractères)
            chunks = split_message(reply_text)
            for idx, chunk in enumerate(chunks):
                if idx == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)

        except Exception as e:
            print(f"[-] Erreur appel Mistral : {e}")
            await message.reply(f"❌ **Erreur Mistral AI** : `{e}`")


def main():
    global MISTRAL_API_KEY
    if not MISTRAL_API_KEY:
        if len(sys.argv) > 1:
            MISTRAL_API_KEY = sys.argv[1].strip()
        else:
            print("\n" + "=" * 60)
            print("  CONNEXION DU BOT TALOS À MISTRAL AI (PRO / LE CHAT)")
            print("=" * 60)
            print("Récupérez votre clé API sur : https://console.mistral.ai/api-keys/")
            try:
                MISTRAL_API_KEY = input("Entrez votre clé API Mistral : ").strip()
            except EOFError:
                pass

    try:
        client.run(DISCORD_TOKEN)
    except Exception as e:
        print(f"[-] Erreur connexion Discord : {e}")


if __name__ == "__main__":
    main()
