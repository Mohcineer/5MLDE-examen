"""Créer ou mettre à jour l'automation d'alerte : flow run Failed/Crashed -> notification.

    uv run python flows/setup_alerts.py --channel webhook   # destination HTTP de test (défaut)
    uv run python flows/setup_alerts.py --channel teams     # webhook Teams (URL en saisie masquée)

Les tentatives de tâche et `AwaitingRetry` ne déclenchent rien : un incident transitoire résolu par
les retries reste silencieux. Idempotent : l'automation existante (même nom) est modifiée, jamais
dupliquée. L'URL du webhook Teams est un secret : jamais écrite dans le dépôt ni affichée.
"""
import argparse
import getpass
import os
import socket
from datetime import timedelta

from prefect.automations import Automation, EventTrigger, Posture
from prefect.blocks.notifications import CustomWebhookNotificationBlock, MicrosoftTeamsWebhook
from prefect.events.actions import SendNotification

NAME = "Alerte : échec du pipeline Bikeshare"
FLOW = "Pipeline Bikeshare"
BLOCKS = {"webhook": "bikeshare-alerte-http", "teams": "bikeshare-alerte-teams"}

SUBJECT = "ALERTE : {{ flow.name }} en échec ({{ flow_run.state.name }})"
BODY = (
    "Flow : {{ flow.name }}\n"
    "Flow run : {{ flow_run.name }} (ID {{ flow_run.id }})\n"
    "État : {{ flow_run.state.name }} à {{ flow_run.state.timestamp }}\n"
    "Cause : {{ flow_run.state.message or 'indisponible' }}\n"
    "Références (batch, run MLflow, étape en erreur) : {{ flow_run.tags | join(', ') or 'indisponible' }}\n"
    "Poste : " + socket.gethostname() + "\n"
    "Aucun modèle n'est publié après un échec du pipeline ; le champion en place reste servi."
)


def save_block(channel):
    if channel == "teams":
        url = os.getenv("TEAMS_WEBHOOK_URL") or getpass.getpass("URL du webhook Teams (saisie masquée) : ")
        if not url.startswith("https://"):
            raise SystemExit("L'URL doit commencer par https://")
        MicrosoftTeamsWebhook(url=url).save(BLOCKS[channel], overwrite=True)
        return MicrosoftTeamsWebhook.load(BLOCKS[channel])
    url = os.getenv("ALERT_WEBHOOK_URL", "http://127.0.0.1:8765/alert")
    CustomWebhookNotificationBlock(
        name="alerte-http", url=url, method="POST",
        json_data={"subject": "{{subject}}", "body": "{{body}}"}).save(BLOCKS[channel], overwrite=True)
    return CustomWebhookNotificationBlock.load(BLOCKS[channel])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", choices=BLOCKS, default="webhook")
    args = parser.parse_args()
    if not os.getenv("PREFECT_API_URL"):
        raise SystemExit("Définir PREFECT_API_URL (ex. http://127.0.0.1:4200/api).")
    block = save_block(args.channel)
    trigger = EventTrigger(
        expect={"prefect.flow-run.Failed", "prefect.flow-run.Crashed"},
        match_related={"prefect.resource.role": "flow", "prefect.resource.name": [FLOW]},
        posture=Posture.Reactive, threshold=1, within=timedelta(0))
    actions = [SendNotification(block_document_id=block._block_document_id, subject=SUBJECT, body=BODY)]
    try:
        automation = Automation.read(name=NAME)
    except Exception:
        automation = None
    if automation is None:
        automation = Automation(name=NAME, description="Alerte sur échec définitif du pipeline",
                                trigger=trigger, actions=actions).create()
        print(f"Automation créée ({args.channel}) : {automation.id}")
    else:
        automation.trigger, automation.actions = trigger, actions
        automation.update()
        print(f"Automation mise à jour ({args.channel}, même identifiant) : {automation.id}")


if __name__ == "__main__":
    main()
