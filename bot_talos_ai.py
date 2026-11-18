#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TALOS - Bot IA Officiel pour Discord (Mistral Pro & Google Gemini)
Hackathon Robot Humanoïde Défense 2026 (AID / Agorize / Unitree G1 & Go2)
=============================================================================
Ce bot donne vie aux salons Discord avec une connaissance totale du projet :
- #🤖・mistral-ai -> Alimenté par Mistral Large (Abonnement Pro)
- #🤖・gemini-ai  -> Alimenté par Google Gemini 2.5 Flash / Pro
- Connaissance native : Dossier TALOS, CVs de l'équipe (4 membres ENSTA),
  spécifications Unitree G1 & Go2, scripts MuJoCo, ROS 2, RL Gym et GitHub.
=============================================================================
"""

import os
import sys
import asyncio
import discord
from mistralai.client import Mistral
from google import genai

# --- CHARGEMENT DU FICHIER .ENV LOCAL ---
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# --- CONFIGURATION CLÉS SÉCURISÉES ---
DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- BASE DE CONNAISSANCES COMPLÈTE DU PROJET TALOS ---
PROJECT_KNOWLEDGE_BASE = """
=== CONTEXTE ET IDENTITÉ DU PROJET TALOS ===
Nom du projet : TALOS (Inspiré de l'automate de bronze forgé par Héphaïstos pour protéger la Crète).
Compétition : Hackathon Robot Humanoïde Défense 2026, organisé par l'Agence de l'Innovation de Défense (AID) et Agorize / Adopt AI.
Objectif opérationnel : Reconnaissance tactique et franchissement autonome en milieu clos et dégradé (bâtiments effondrés, escaliers, couloirs étroits <= 70 cm, guerre électronique sévère avec brouillage total GNSS/GPS).

=== COMPOSITION DE L'ÉQUIPE (4 ÉLÈVES-INGÉNIEURS ENSTA BRETAGNE) ===
Tous élèves-ingénieurs alternants en Systèmes Embarqués à l'ENSTA Bretagne (école sous tutelle de la DGA) :
1. Mathis BEAUDUCEL :
   - Entreprise : Safran Electronics & Defense
   - Spécialité : Architecte & Firmware bas niveau, microcontrôleurs (STM32, Raspberry Pi), rétro-ingénierie matérielle, réparation de cartes de vol critiques.
2. Arlann LAUNAY :
   - Entreprise : Safran Electronics & Defense
   - Spécialité : Test, Bancs d'essais automatiques (TestStand / C++), routage de cartes multicouches KiCad, qualification avionique civile et militaire.
3. Ayoub OUZANI :
   - Entreprise : ArianeGroup
   - Spécialité : Simulation éléments finis (FEA Radioss / HyperWorks), dynamique des matériaux énergétiques, guidage de robots mobiles autonomes.
4. Yurii KHARKIVSKYI :
   - Entreprise : Breizelec
   - Spécialité : Bancs calculateurs ECU, bus CAN temps réel, diagnostic matériel et kits de télécommande sans fil sécurisée / bridges de trames pour engins autonomes.

=== ROBOTS ET ÉCOSYSTÈME MATÉRIEL ===
1. Humanoïde Unitree G1 (29 DDL) :
   - Hauteur ~1.30m, masse 35 kg, 23 à 29 degrés de liberté avec mains préhensiles motorisées Dex3-1.
   - Contrôle : Unitree SDK2 Python / C++, communication DDS / UDP, moteurs haute puissance intégrés.
   - Atout : Franchissement de marches, franchissement de dévers 30°, manipulation d'objets.
2. Quadrupède Unitree Go2 :
   - Quadrupède tout-terrain haute agilité, Lidar 4D hémisphérique, vitesse max 3.5 m/s.
   - Rôle tactique conjoint : Éclaireur rapide, relais de télécommunications radio, cartographie préalable avant engagement du G1.

=== ARCHITECTURE LOGICIELLE ET CODE DU DÉPÔT LOCAL ===
- Simulation : MuJoCo (`unitree_mujoco`) avec scènes de bâtiments (couloirs, escaliers, décombres d'habitation, grand hall).
- Apprentissage par renforcement : `unitree_rl_gym` (PPO, politiques de marche et franchissement d'obstacles).
- Middleware : ROS 2 (`unitree_ros2`) pour les nœuds de navigation, odométrie, SLAM Lidar et communication inter-robots.
- Scripts clés dans le projet :
  * `train_hyper_rl.py` : Entraînement de politiques RL pour terrain rocheux et décombres.
  * `analyze_apartment_path.py` & `test_real_apartment_run.py` : Validation des trajectoires en environnement intérieur réel.
  * `test_stair_climb.py` : Algorithme de montée et descente d'escaliers.
  * `test_single_goal_exploration.py` : SLAM et exploration autonome sans GPS.

=== STRUCTURE DU SERVEUR DISCORD TALOS ===
- 📌 INFORMATIONS & CADRAGE : #annonces, #ressources-docs, #liens-github, #general
- 🧠 ASSISTANTS IA : #mistral-ai (Mistral Pro), #gemini-ai (Google Gemini)
- 🤖 SIMULATION & RL (MuJoCo) : #mujoco-scenes, #rl-training, #slam-navigation
- 🦾 EMBARQUÉ & HUMANOÏDE G1 : #unitree-g1-sdk, #ros2-noeuds, #hardware-bancs
- 🐕 QUADRUPÈDE UNITREE GO2 : #go2-sdk-python, #go2-locomotion, #go2-reconnaissance
- 🎯 LIVRABLES & DÉMO : #slides-pitch, #videos-renders
- 🔊 SALONS VOCAUX : Point Synchro / Daily, Pair-Programming, Salle de pause
"""

SYSTEM_PROMPT = f"""Tu es l'assistant IA technique officiel du projet TALOS pour l'équipe d'ingénieurs sur Discord.
Voici l'intégralité des informations, documents, code et contexte de l'équipe :

{PROJECT_KNOWLEDGE_BASE}

RÈGLES DE COMPORTEMENT :
1. Tu réponds toujours en français, de manière experte, claire, structurée et précise.
2. Tu connais parfaitement chaque membre de l'équipe, leurs compétences respectives et leurs entreprises (Safran, ArianeGroup, Breizelec).
3. Tu connais par cœur les dossiers du projet, les robots G1 & Go2, MuJoCo, ROS 2 et l'état des simulations.
4. Si un membre te demande du code (Python, C++, XML MuJoCo, URDF, ROS 2, scripts d'entraînement), fournis du code de qualité industrielle, propre, documenté et optimisé.
5. Adopte une posture collaborative et motivante digne d'un coéquipier ingénieur de haut niveau.
"""

# Historique des conversations par salon : { channel_id: [ {"role": ..., "content": ...} ] }
conversations = {}
MAX_HISTORY = 10

# Initialisation des clients IA
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


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


def extract_mistral_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Filtrer pour ne garder que les morceaux de texte final (et ignorer les ThinkChunk)
        text_parts = []
        for item in content:
            if getattr(item, "type", None) == "text" and hasattr(item, "text"):
                text_parts.append(item.text)
        if text_parts:
            return "".join(text_parts)
    return str(content)


async def ask_mistral(history):
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    response = await asyncio.to_thread(
        mistral_client.chat.complete,
        model="mistral-large-latest",
        messages=msgs,
    )
    return extract_mistral_text(response.choices[0].message.content)


async def ask_gemini(history):
    contents = []
    for m in history:
        if m["role"] == "user":
            contents.append(f"Ingénieur : {m['content']}")
        elif m["role"] == "assistant":
            contents.append(f"TALOS AI : {m['content']}")

    prompt = f"Consigne Système :\n{SYSTEM_PROMPT}\n\nHistorique de la conversation :\n" + "\n\n".join(contents)
    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def create_client(with_message_content=True):
    intents = discord.Intents.default()
    if with_message_content:
        intents.message_content = True
    intents.guilds = True

    c = discord.Client(intents=intents)

    @c.event
    async def on_ready():
        mode = "TOUS LES MESSAGES DU SALON (Message Content Intent ON)" if with_message_content else "MENTION @Talos (ou MP)"
        print("\n" + "=" * 65)
        print(f"  BOT TALOS MULTI-AI OPÉRATIONNEL : {c.user} (ID: {c.user.id})")
        print("=" * 65)
        print(f"  • MODE D'ÉCOUTE  : {mode}")
        print("  • MISTRAL PRO    : mistral-large-latest [CONNECTÉ & OPÉRATIONNEL]")
        print("  • GOOGLE GEMINI  : gemini-2.5-flash [CONNECTÉ & OPÉRATIONNEL]")
        print("  • CONTEXTE ACTIF : 4 Membres ENSTA, Unitree G1/Go2, MuJoCo, ROS 2, Dépôt")
        print("=" * 65)
        print("  ÉCOUTE ACTIVE SUR LES SALONS DISCORD...\n", flush=True)

    @c.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return

        ch_name = getattr(message.channel, "name", "").lower()
        is_mistral = "mistral" in ch_name
        is_gemini = "gemini" in ch_name
        is_mentioned = c.user in message.mentions
        is_dm = isinstance(message.channel, discord.DMChannel)

        if not (is_mistral or is_gemini or is_mentioned or is_dm):
            return

        clean_content = message.clean_content.replace(f"@{c.user.name}", "").strip()

        # Si le bot reçoit un message dans #mistral-ai ou #gemini-ai mais que le contenu est vide
        # c'est que Discord masque le texte car le "Message Content Intent" n'est pas encore coché sur le portail
        if not clean_content:
            if is_mistral or is_gemini:
                embed_hint = discord.Embed(
                    title="👋 Pour me poser une question :",
                    description=(
                        "Pour que je puisse lire ton message, deux solutions :\n\n"
                        "1. **Option rapide :** Mentionne-moi dans ton message, par exemple :\n"
                        f"   > `@{c.user.name} allo ?`\n\n"
                        "2. **Option permanente (sans avoir besoin de mentionner) :**\n"
                        "   Active le **`Message Content Intent`** en 1 clic sur le portail Discord :\n"
                        "   👉 **[Activer sur Discord Developer Portal](https://discord.com/developers/applications/1572732425526706276/bot)**\n"
                        "   *(Descends à 'Privileged Gateway Intents' -> coche 'Message Content Intent' -> Save Changes)*."
                    ),
                    color=discord.Color.from_rgb(255, 180, 0)
                )
                await message.reply(embed=embed_hint)
            return

        # Choisir le moteur IA selon le salon
        ai_engine = "mistral"
        if is_gemini:
            ai_engine = "gemini"
        elif is_mistral:
            ai_engine = "mistral"
        elif is_mentioned or is_dm:
            ai_engine = "mistral"

        cid = message.channel.id
        if cid not in conversations:
            conversations[cid] = [{"role": "system", "content": SYSTEM_PROMPT}]

        conversations[cid].append({"role": "user", "content": clean_content})

        if len(conversations[cid]) > (MAX_HISTORY * 2 + 1):
            conversations[cid] = [conversations[cid][0]] + conversations[cid][-(MAX_HISTORY * 2):]

        # 1. ENCADRÉ "EN COURS DE TRAITEMENT" IMMÉDIAT
        engine_display = "Mistral Large (Pro)" if ai_engine == "mistral" else "Google Gemini 2.5"
        embed_loading = discord.Embed(
            title=f"⏳ {engine_display} — Réflexion en cours...",
            description="Recherche dans la base de connaissances TALOS et génération de la réponse...",
            color=discord.Color.from_rgb(255, 112, 0) if ai_engine == "mistral" else discord.Color.from_rgb(66, 133, 244)
        )
        embed_loading.set_footer(text="TALOS Autonomous Defense Bot • Traitement en cours...")
        
        loading_msg = await message.reply(embed=embed_loading)

        # 2. APPEL API & ÉDITION AVEC LA RÉPONSE
        async with message.channel.typing():
            try:
                if ai_engine == "mistral":
                    answer = await ask_mistral(conversations[cid])
                else:
                    answer = await ask_gemini(conversations[cid])

                conversations[cid].append({"role": "assistant", "content": answer})

                chunks = split_message(answer)
                # On remplace l'encadré de chargement par la réponse finale
                await loading_msg.edit(content=chunks[0], embed=None)

                # Si la réponse fait plus de 2000 caractères, on envoie le reste
                for extra_chunk in chunks[1:]:
                    await message.channel.send(extra_chunk)

            except Exception as e:
                print(f"[-] Erreur {ai_engine} : {e}", flush=True)
                err_embed = discord.Embed(
                    title="❌ Erreur de génération",
                    description=f"`{e}`",
                    color=discord.Color.red()
                )
                await loading_msg.edit(embed=err_embed)

    return c


def main():
    try:
        c = create_client(with_message_content=True)
        c.run(DISCORD_TOKEN)
    except Exception as e:
        print(f"[-] Erreur Discord : {e}")


if __name__ == "__main__":
    main()
