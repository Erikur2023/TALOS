#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
TALOS - Script de Configuration Automatique du Serveur Discord
Hackathon Robot Humanoide Defense (AID / Agorize / Unitree G1)
=============================================================================
Ce script configure l'integralite du serveur Discord de l'equipe TALOS :
- Creation des roles avec couleurs et hierarchies
- Creation des categories et salons textuels / vocaux avec descriptions
- Nettoyage propre des salons par defaut
- Envoi du message d'accueil et cadrage de mission dans #annonces
=============================================================================
"""

import sys
import asyncio
import os
import discord

# --- CONFIGURATION DES ROLES ---
# Pas de rôles rigides : l'équipe participe pour le fun, tout le monde touche à tout !
ROLES_CONFIG = [
    {
        "name": "TALOS",
        "color": discord.Color.from_rgb(0, 229, 255),
        "hoist": True,
        "mentionable": True,
    },
    {
        "name": "🤖 Bot & Automations",
        "color": discord.Color.from_rgb(149, 165, 166),
        "hoist": False,
        "mentionable": False,
    },
]

# --- STRUCTURE DES SALONS & CATEGORIES ---
STRUCTURE_CONFIG = [
    {
        "category": "📌 INFORMATIONS & CADRAGE",
        "channels": [
            {
                "type": "text",
                "name": "📢・annonces",
                "topic": "Annonces officielles, calendrier des livrables Agorize / AID et jalons de mission.",
            },
            {
                "type": "text",
                "name": "📁・ressources-docs",
                "topic": "Dossier TALOS, reglement du hackathon, documentations Unitree G1 et liens utiles.",
            },
            {
                "type": "text",
                "name": "🔗・liens-github",
                "topic": "Depots git, suivi des commits, pull requests et scripts partages.",
            },
            {
                "type": "text",
                "name": "💬・general",
                "topic": "Echanges quotidiens, questions rapides et discussions de l'equipe.",
            },
        ],
    },
    {
        "category": "🤖 SIMULATION & RL (MuJoCo)",
        "channels": [
            {
                "type": "text",
                "name": "💻・mujoco-scenes",
                "topic": "Scenes XML MuJoCo, maillages STL/OBJ, decors tactiques et environnements de test.",
            },
            {
                "type": "text",
                "name": "🧠・rl-training",
                "topic": "Apprentissage par renforcement, hyperparametres, courbes de recompenses et convergence.",
            },
            {
                "type": "text",
                "name": "🗺️・slam-navigation",
                "topic": "Nuages de points Lidar, cartographie 3D, planification de trajectoire et franchissement.",
            },
        ],
    },
    {
        "category": "🦾 EMBARQUE & HUMANOIDE G1",
        "channels": [
            {
                "type": "text",
                "name": "⚡・unitree-g1-sdk",
                "topic": "SDK2 Python, protocoles moteurs, telemetrie temps reel et mains Dex3-1.",
            },
            {
                "type": "text",
                "name": "🔌・ros2-noeuds",
                "topic": "Architecture ROS 2, packages, topics capteurs/actionneurs et bridge DDS.",
            },
            {
                "type": "text",
                "name": "🔧・hardware-bancs",
                "topic": "Bancs de test, alimentations, mesures electriques et cartes additionnelles.",
            },
        ],
    },
    {
        "category": "🐕 QUADRUPEDE UNITREE GO2",
        "channels": [
            {
                "type": "text",
                "name": "⚡・go2-sdk-python",
                "topic": "SDK2 Python Unitree Go2, communication bas niveau, telemetrie et protocoles DDS.",
            },
            {
                "type": "text",
                "name": "🐾・go2-locomotion",
                "topic": "Controle dynamique, allures quadrupedes (marche, trot, sprint) et stabilite terrain accidente.",
            },
            {
                "type": "text",
                "name": "📡・go2-reconnaissance",
                "topic": "Lidar 4D / Cameras, cartographie SLAM eclaireur et coordination tactique G1/Go2.",
            },
        ],
    },
    {
        "category": "🎯 LIVRABLES & DEMO",
        "channels": [
            {
                "type": "text",
                "name": "📝・slides-pitch",
                "topic": "Support de presentation PowerPoint / Marp, argumentaire oral et structure du pitch.",
            },
            {
                "type": "text",
                "name": "🎬・videos-renders",
                "topic": "Captures video de simulation, montages de demonstration et rendus HUD.",
            },
        ],
    },
    {
        "category": "🔊 SALONS VOCAUX",
        "channels": [
            {
                "type": "voice",
                "name": "🎙️ Point Synchro / Daily",
                "user_limit": 0,
            },
            {
                "type": "voice",
                "name": "💻 Pair-Programming",
                "user_limit": 4,
            },
            {
                "type": "voice",
                "name": "☕ Salle de pause",
                "user_limit": 0,
            },
        ],
    },
]

WELCOME_EMBED = discord.Embed(
    title="🛡️ Projet TALOS — Hackathon Robot Humanoïde Défense",
    description=(
        "Bienvenue sur l'espace opérationnel de l'équipe **TALOS** !\n\n"
        "**Mission :** Système d'exploration et de franchissement autonome en milieu dégradé "
        "pour le robot humanoïde **Unitree G1** (29 DDL).\n\n"
        "**Organisation :**\n"
        "• 📌 **Informations & Cadrage** : Annonces, calendrier et documentation du projet.\n"
        "• 🤖 **Simulation & RL** : Scènes MuJoCo, modèles Gym et SLAM 3D.\n"
        "• 🦾 **Embarqué & G1** : Intégration ROS 2, SDK2 Python et hardware.\n"
        "• 🎯 **Livrables & Démo** : Préparation du pitch et des vidéos pour le jury.\n"
        "• 🔊 **Salons Vocaux** : Synchronisation quotidienne et dev en direct."
    ),
    color=discord.Color.from_rgb(41, 128, 185),
)
WELCOME_EMBED.add_field(
    name="👥 Les 4 Pôles Opérationnels",
    value=(
        "• 🔵 **Architecte & Systèmes Embarqués**\n"
        "• 🟣 **Simulation & Contrôle RL**\n"
        "• 🟢 **Guidage, SLAM & Navigation**\n"
        "• 🟠 **Test, Instrumentation & ROS2**"
    ),
    inline=False,
)
WELCOME_EMBED.set_footer(text="TALOS Team • Agorize / AID Hackathon 2026")


class ServerSetupClient(discord.Client):
    def __init__(self, target_guild_id=None, clean_old=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_guild_id = target_guild_id
        self.clean_old = clean_old
        self._setup_done = False

    async def setup_guild(self, guild):
        if self._setup_done:
            return
        self._setup_done = True

        print(f"\n[*] Configuration du serveur : {guild.name} (ID: {guild.id})", flush=True)

        # 1. Nettoyage optionnel des anciens salons
        if self.clean_old:
            print("[*] Nettoyage des salons par defaut...", flush=True)
            for ch in list(guild.channels):
                try:
                    await ch.delete(reason="Reinitialisation TALOS")
                    print(f"  - Supprime : {ch.name}", flush=True)
                except Exception as e:
                    print(f"  ! Impossible de supprimer {ch.name}: {e}", flush=True)

        # 2. Creation des Roles
        print("\n[*] Creation des roles d'equipe...", flush=True)
        created_roles = {}
        existing_role_names = [r.name for r in guild.roles]

        for r_conf in reversed(ROLES_CONFIG):
            if r_conf["name"] in existing_role_names:
                print(f"  = Role deja existant : {r_conf['name']}", flush=True)
                continue
            try:
                role = await guild.create_role(
                    name=r_conf["name"],
                    color=r_conf["color"],
                    hoist=r_conf["hoist"],
                    mentionable=r_conf["mentionable"],
                    reason="Setup roles TALOS",
                )
                created_roles[r_conf["name"]] = role
                print(f"  + Role cree : {role.name}", flush=True)
            except Exception as e:
                print(f"  ! Erreur creation role {r_conf['name']}: {e}", flush=True)

        # 3. Creation des Categories et Salons
        print("\n[*] Creation de l'arborescence des salons...", flush=True)
        annonces_channel = None

        for cat_conf in STRUCTURE_CONFIG:
            try:
                category = await guild.create_category(
                    name=cat_conf["category"],
                    reason="Setup categorie TALOS",
                )
                print(f"\n  📁 Categorie creee : {category.name}", flush=True)

                for ch_conf in cat_conf["channels"]:
                    if ch_conf["type"] == "text":
                        ch = await guild.create_text_channel(
                            name=ch_conf["name"],
                            category=category,
                            topic=ch_conf.get("topic", ""),
                            reason="Setup salon textuel TALOS",
                        )
                        print(f"     💬 #{ch.name}", flush=True)
                        if "annonces" in ch.name:
                            annonces_channel = ch
                    elif ch_conf["type"] == "voice":
                        v_ch = await guild.create_voice_channel(
                            name=ch_conf["name"],
                            category=category,
                            user_limit=ch_conf.get("user_limit", 0),
                            reason="Setup salon vocal TALOS",
                        )
                        print(f"     🔊 {v_ch.name}", flush=True)
            except Exception as e:
                print(f"  ! Erreur creation categorie {cat_conf['category']}: {e}", flush=True)

        # 4. Publication du message d'accueil
        if annonces_channel:
            print("\n[*] Publication du message de cadrage dans #annonces...", flush=True)
            try:
                msg = await annonces_channel.send(embed=WELCOME_EMBED)
                await msg.pin(reason="Message d'accueil epingle")
                print("  + Message publie et epingle avec succes !", flush=True)
            except Exception as e:
                print(f"  ! Erreur envoi message : {e}", flush=True)

        print("\n" + "=" * 60, flush=True)
        print("  CONFIGURATION DU SERVEUR TALOS TERMINEE AVEC SUCCES !", flush=True)
        print("=" * 60 + "\n", flush=True)
        await self.close()

    async def on_ready(self):
        print(f"\n[+] Connecte avec succes sous le compte : {self.user} (ID: {self.user.id})", flush=True)

        if len(self.guilds) > 0:
            target = None
            if self.target_guild_id:
                target = self.get_guild(int(self.target_guild_id))
            if not target:
                target = self.guilds[0]
            await self.setup_guild(target)
        else:
            invite_url = f"https://discord.com/oauth2/authorize?client_id={self.user.id}&permissions=8&scope=bot"
            print("\n[!] Le bot n'est encore present sur aucun serveur !", flush=True)
            print(f"[*] Cliquez sur ce lien pour inviter le bot en 1 clic sur votre serveur :\n    {invite_url}\n", flush=True)
            print("[*] En attente de l'invitation sur le serveur...", flush=True)

    async def on_guild_join(self, guild):
        print(f"\n[+] Le bot vient de rejoindre le serveur : {guild.name} (ID: {guild.id})", flush=True)
        await self.setup_guild(guild)


def main():
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1].strip()

    if not token:
        print("=" * 65)
        print("  CONFIGURATION AUTOMATIQUE DU SERVEUR DISCORD - TALOS")
        print("=" * 65)
        print("Pour lancer la creation automatique, entrez votre Discord Bot Token.")
        print("(Vous pouvez creer un bot en 1 min sur https://discord.com/developers/applications)\n")
        try:
            token = input("Discord Bot Token : ").strip()
        except EOFError:
            pass

    if not token:
        print("[-] Aucun token fourni. Sortie.")
        sys.exit(1)

    guild_id = os.environ.get("DISCORD_GUILD_ID")
    if not guild_id and sys.stdin.isatty():
        try:
            g_input = input("ID du serveur (Laissez vide pour detecter automatiquement) : ").strip()
            if g_input:
                guild_id = g_input
        except Exception:
            pass

    clean_old = True
    if sys.stdin.isatty():
        try:
            clean_answer = input("Supprimer les anciens salons par defaut ? (O/n) [Defaut: O] : ").strip().lower()
            clean_old = clean_answer not in ["n", "non", "no"]
        except Exception:
            clean_old = True

    intents = discord.Intents.default()
    intents.guilds = True

    client = ServerSetupClient(target_guild_id=guild_id, clean_old=clean_old, intents=intents)

    try:
        client.run(token)
    except discord.errors.LoginFailure:
        print("\n[-] Erreur : Le Bot Token fourni est invalide.")
    except Exception as e:
        print(f"\n[-] Erreur inattendue : {e}")


if __name__ == "__main__":
    main()
