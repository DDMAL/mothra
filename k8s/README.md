# Mothra — Kubernetes manifests (app)

Plain-YAML manifests for the Mothra app, deployed to the **`mothra`** namespace.
These mirror the `docker-compose.yml` stack: redis + ic + text-service + backend +
Celery worker. **Postgres lives in its own repo** (deployed to the `postgres`
namespace); the app reaches it cross-namespace at
`mothra-postgres.postgres.svc.cluster.local:5432` via `DATABASE_URL`.

- **Ingress:** Traefik — `mothra.simssa.ca` → backend, `mothra-ic.simssa.ca` → ic.
  TLS is terminated by the campus proxy; the app is served over https.
- **Storage:** static NFS PersistentVolume (RWX) for `stored_models`, shared by
  backend + worker.
- **Images:** `ghcr.io/ddmal/mothra-{backend,ic,text-service}`, CPU-only, pulled with
  the `ghcr-pull-secret` imagePullSecret. `worker` reuses the `mothra-backend` image.

```
k8s/
  secret.yaml.example    # template → copy to secret.yaml (gitignored)
  secret.yaml            # MOTHRA_SECRET, DATABASE_URL   (gitignored, not committed)
  configmap.yaml         # broker + service URLs, IC_PUBLIC_URL, ALLOWED_ORIGINS
  stored-models-pv.yaml  # static NFS PV (RWX)
  stored-models-pvc.yaml # PVC
  redis.yaml             # Deployment + Service :6379
  ic.yaml                # Deployment + Service :8000
  text-service.yaml      # Deployment + Service :8002
  backend.yaml           # Deployment + Service :8001
  worker.yaml            # Deployment (celery worker; no Service)
  ingress.yaml           # Traefik ingress (both hosts)
```

## First-time setup
1. **Create the app Secret** (never committed):
   ```
   cp k8s/secret.yaml.example k8s/secret.yaml
   # set MOTHRA_SECRET (`openssl rand -hex 32`) and the DATABASE_URL password
   # (must match the Postgres deployment in its own repo).
   ```
2. Ensure the `mothra` namespace and the `ghcr-pull-secret` imagePullSecret exist,
   and that Postgres (separate repo) is running in the `postgres` namespace.

## Deploy
**CI/CD (preferred)** — `.github/workflows/build-images.yml` (`ci-cd`): on push to
`main` / `develop` / `k8s-deployment` it builds the images, then applies the
manifests and rolls out the deployments (all except redis). Requires the
`KUBECONFIG` repo secret. The deploy job pins images to the commit's `sha-<short>` tag.

**Manual** (redis/postgres excluded from CD; apply them by hand when needed):
```
kubectl apply -f k8s/secret.yaml -f k8s/configmap.yaml
kubectl apply -f k8s/stored-models-pv.yaml -f k8s/stored-models-pvc.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/ic.yaml -f k8s/text-service.yaml -f k8s/backend.yaml -f k8s/worker.yaml
kubectl apply -f k8s/ingress.yaml
```

## Verify
```
kubectl -n mothra get pods,pvc,svc,ingress          # all Ready, stored-models PVC Bound (RWX)
kubectl -n mothra logs deploy/backend | head        # tables created; Celery connected
kubectl -n mothra logs deploy/worker  | head        # "celery ... ready" (threads pool)
curl -k https://mothra.simssa.ca/                   # SPA HTML
# smoke test: POST /api/register → authed GET /api/projects (200)
```

## Known follow-ups
- No real `/healthz` yet → probes are TCP/exec. Adding `/healthz` is recommended.
- `init_db()`/`_migrate_db()` run at import → keep backend/worker at **1 replica**
  until a one-shot migration Job is added, then scale out.
- text-service `/batch-download/{id}` uses local disk keyed by batch_id → needs
  shared storage or a single replica if batch downloads are used.
