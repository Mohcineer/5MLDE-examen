"""Flow Prefect du pipeline : ingest -> validate -> preprocess -> train -> evaluate -> publish.

    uv run python flows/pipeline_flow.py                    # un passage (prochain lot)
    uv run python flows/pipeline_flow.py serve              # déploiement local planifié (flow.serve)

Retries bornés sur les seules erreurs transitoires ; aucun retry sur une donnée invalide. Les
références (batch_id, run MLflow, étape en erreur) sont écrites dans les tags du flow run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from prefect import flow, task  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402
from prefect.runtime import flow_run  # noqa: E402

from bikeshare import pipeline  # noqa: E402
from bikeshare.config import load_config  # noqa: E402
from flows.orchestration import annotate_flow_run, retry_on_transient  # noqa: E402

FLOW_NAME = "Pipeline Bikeshare"


def step_task(name, fn):
    """Tâche nommée autour d'une étape ; retries bornés, jamais sur une donnée invalide."""
    settings = load_config()["pipeline"]
    return task(fn, name=name, retries=settings["transient_retries"],
                retry_delay_seconds=settings["transient_retry_delay_seconds"],
                retry_condition_fn=retry_on_transient, cache_policy=NO_CACHE, persist_result=False)


t_open = step_task("Ouverture du run MLflow", pipeline.step_open)
t_ingest = step_task("Ingestion du lot", pipeline.step_ingest)
t_validate = step_task("Contrôles qualité", pipeline.step_validate)
t_preprocess = step_task("Prétraitement", pipeline.step_preprocess)
t_train = step_task("Entraînement", pipeline.step_train)
t_evaluate = step_task("Évaluation et traçabilité", pipeline.step_evaluate)
t_publish = step_task("Publication (qualification, mise en service)", pipeline.step_publish)


@flow(name=FLOW_NAME, log_prints=True)
def bikeshare_pipeline(batch_id: str | None = None, data_dir: str | None = None):
    """Traiter un lot : par défaut, le prochain lot non encore traité."""
    config = load_config()
    if data_dir:  # lot alternatif (démo d'incident) : rapports écrits à côté, jamais sur les vrais
        config["data"]["batches_dir"] = str(Path(data_dir).resolve())
        config["validation"]["reports_dir"] = str(Path(data_dir).resolve() / "reports")
    batch_id = batch_id or pipeline.next_batch_id(config)
    ctx = pipeline.PipelineContext(config=config, batch_id=batch_id, flow_run_id=flow_run.id)
    annotate_flow_run(batch_id=batch_id)
    try:
        t_open(ctx)
        annotate_flow_run(mlflow_run_id=ctx.run_id)
        data = t_ingest(ctx)
        t_validate(ctx, data)
        prep = t_preprocess(ctx, data)
        trained = t_train(ctx, prep)
        evaluation = t_evaluate(ctx, data, prep, trained)
        published = t_publish(ctx, evaluation)
        pipeline.step_close_ok(ctx, {**published, "metrics": evaluation["metrics"]})
    except Exception as exc:
        pipeline.step_close_failed(ctx, exc)
        annotate_flow_run(failed_step=ctx.step)     # visible dans le message d'alerte
        raise
    print(f"{batch_id} : version {ctx.result['model_version']} {ctx.result['qualification']}, "
          f"promue={bool(ctx.result['promoted'])}")
    summary = {k: v for k, v in ctx.result.items() if k != "metrics"}
    return {"batch_id": batch_id, "mlflow_run_id": ctx.run_id, **summary}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        bikeshare_pipeline.serve(name="bikeshare-prochain-lot",
                                 interval=load_config()["pipeline"]["schedule_seconds"])
    else:
        bikeshare_pipeline()
