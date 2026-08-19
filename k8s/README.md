# Mothra — Kubernetes manifests (app)

Plain-YAML manifests for the Mothra app. **Two environments — production and
staging — both live in the `mothra` namespace.** They are kept apart entirely by
naming: the manifest *directory* selects the environment (`k8s/` = production,
`k8s/staging/` = staging), filenames are identical in both, and every staging
object carries a `-staging` suffix on its `metadata.name`, its `app` label **and its
selectors**. Drop that suffix from an `app` label and production's Service starts
load-balancing into a staging pod — that is the one invariant to respect when
editing anything here.

|  | production | staging |
|---|---|---|
| manifests | `k8s/` | `k8s/staging/` |
| object names | `backend`, `mothra-config`, … | `backend-staging`, `mothra-config-staging`, … |
| deployed by | push to `main` (automatic) | `workflow_dispatch` from the branch (any non-`main` ref) |
| app host | `mothra.simssa.ca` | `mothra.staging.simssa.ca` |
| IC host | `mothra-ic.simssa.ca` | `mothra-ic.staging.simssa.ca` |
| Postgres | `mothra-postgres.postgres…` | `mothra-staging-postgres.postgres…` |
| NFS path | `/srv/nfs/mothra/stored_models` | `/srv/nfs/mothra-staging/stored_models` |

Both mirror the `docker-compose.yml` stack: redis + ic + text-service +
paco-classifier-service + backend + Celery worker. **Postgres lives in its own
repo** (deployed to the `postgres`
namespace); each environment reaches its own instance cross-namespace via
`DATABASE_URL`. Staging's instance is a *separate deployment* serving a database
also called `mothra` — only the host differs.

- **Ingress:** Traefik, one Ingress object per host (see below). TLS is terminated
  by the campus proxy; the app is served over https. Staging's two hosts need DNS
  records and a campus-proxy vhost before they resolve.
- **IC iframe:** the campus proxy injects a blanket `X-Frame-Options: SAMEORIGIN`,
  which blocks the cross-host IC iframe. A Traefik `Middleware` in each
  `ingress.yaml` adds `Content-Security-Policy: frame-ancestors https://<app host>`
  to the IC host, which browsers enforce *instead of* XFO. Needs Traefik's CRDs —
  check the API group with `kubectl get crd | grep middlewares`.
- **Storage:** static NFS PersistentVolume (RWX) for `stored_models`, shared by
  backend + worker. Separate PV/PVC per environment, since `models_api.py` deletes
  files and a shared volume would let staging destroy production checkpoints.
- **Images:** `ghcr.io/ddmal/mothra-{backend,ic,text-service,paco-classifier-service}`,
  pulled with the `ghcr-pull-secret` imagePullSecret (namespace-scoped, so staging
  reuses it as-is). `worker` reuses the `mothra-backend` image. **Both environments
  share these four image repos** — only the tag differs, so `latest` is published
  from `main` only.
- **GPU:** both `worker` Deployments are pinned to `k3s-gpu-node-1` and share one
  MIG instance. Neither requests `nvidia.com/gpu` (see the comment in
  `k8s/worker.yaml`), so the scheduler cannot arbitrate: concurrent inference in
  both environments can surface as `torch.OutOfMemoryError` in *either* one, and it
  will not show as `OOMKilled` (the memory limit covers host RAM, not VRAM). If it
  bites, scale `worker-staging` to 0 and scale it up only for GPU tests.

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
  paco-classifier-service.yaml  # Deployment + Service :8003
  backend.yaml           # Deployment + Service :8001
  worker.yaml            # Deployment (celery worker; no Service)
  migrate-job.yaml       # one-shot Job: DB schema migration, run+waited-on before backend/worker
  ingress.yaml           # Traefik: ic-frame-ancestors Middleware + one Ingress per host
  staging/               # same 13 filenames, same `mothra` namespace,
                         #   every object name/label/selector suffixed -staging
```

## First-time setup

### Production
1. **Create the app Secret** (never committed):
   ```
   cp k8s/secret.yaml.example k8s/secret.yaml
   # set MOTHRA_SECRET (`openssl rand -hex 32`) and the DATABASE_URL password
   # (must match the Postgres deployment in its own repo).
   ```
2. Ensure the `mothra` namespace and the `ghcr-pull-secret` imagePullSecret exist,
   and that Postgres (separate repo) is running in the `postgres` namespace.

### Staging (one-time, all outside this repo except step 1)
1. **Create the staging Secret** — required *before* the first staging CD run: a
   missing Secret referenced by `envFrom` leaves pods in
   `CreateContainerConfigError` until `rollout status` times out.
   ```
   cp k8s/staging/secret.yaml.example k8s/staging/secret.yaml
   # set a MOTHRA_SECRET distinct from production's, and the staging DB password
   ```
   Double-check the host in `DATABASE_URL`: staging pods can resolve production's
   `mothra-secrets`/`mothra-config` too, so a wrong name here boots green and lets
   `_migrate_db()` ALTER **production**'s tables with no error anywhere.
2. **Deploy `mothra-staging-postgres` in the `postgres` namespace** (Postgres repo —
   nothing here creates it), serving a `mothra` database, with credentials that
   differ from production's.
3. **Create and export the staging NFS path** on `192.168.236.201`:
   `/srv/nfs/mothra-staging/stored_models`. This is outside the existing
   `/srv/nfs/mothra` export, so it needs an `/etc/exports` entry + `exportfs -ra`,
   not just a `mkdir`, and must sit inside the NFSv4 pseudo-root. A static PV binds
   `Bound` regardless — a missing path only shows at pod start as
   `mount.nfs4: ... No such file or directory` with the pod in `ContainerCreating`.
4. **DNS + campus-proxy vhost** for `mothra.staging.simssa.ca` and
   `mothra-ic.staging.simssa.ca`.
5. ~~**First boot only:** bring `backend-staging` up alone...~~ **No longer
   needed (mothra#220 row 31)** — `migrate-job.yaml` creates the schema once,
   before `backend`/`worker` are ever applied, so there's no first-boot race
   between them to worry about on a brand-new empty database.

## Deploy
**CI/CD (preferred)** — `.github/workflows/build-images.yml` (`ci-cd`) builds the
images, then applies one environment's manifests and rolls out its deployments (all
except redis). Requires the `KUBECONFIG` repo secret. The deploy job pins images to
the commit's `sha-<short>` tag.

| trigger | environment |
|---|---|
| `workflow_dispatch` from any non-`main` branch, `environment: auto` (default) | staging (`k8s/staging/`) |
| `workflow_dispatch`, `environment: staging` | staging, from any branch (incl. `main`) |
| push to `main` | production (`k8s/`) |
| `workflow_dispatch` from `main`, `environment: auto` | production |
| `workflow_dispatch`, `environment: production` | production — **refused unless run from `main`** |

**Deploying a branch to staging:** Actions → **ci-cd** → **Run workflow** → pick
the branch in the dropdown → leave `environment` at `auto` → **Run workflow**. The
run shows up as `manual · auto · <branch>`. It builds that branch's four images,
tags them `sha-<short>`, and rolls staging onto them.

Staging deploys are **not** automatic on push. There is one shared staging
environment (one deployment set, one `mothra-staging-postgres`) and a ~25-minute
four-image build, so ~20 active branches auto-deploying would just thrash both;
whoever dispatches last owns staging either way.

**Caveat — a dispatched run uses the *selected branch's* files.** That is both the
point (you can test edits to `k8s/staging/*.yaml` on the branch that makes them) and
the footgun: a branch cut before the staging commit (`654f18e`) carries the *old*
workflow, which has no `resolve` job and deploys the **production** manifests. Merge
or rebase `main` into a branch before dispatching it. Likewise, a stale branch
redeploys whatever `k8s/staging/` looked like on that branch.

Committed staging manifests carry the placeholder tag `sha-0000000`, which is not a
real tag: if the deploy job's `sed` ever fails to rewrite it, the rollout fails
loudly instead of silently redeploying a stale image.

**Manual** (redis/postgres/secrets/PV excluded from CD; apply them by hand when needed):
```
# production
kubectl apply -f k8s/secret.yaml -f k8s/configmap.yaml
kubectl apply -f k8s/stored-models-pv.yaml -f k8s/stored-models-pvc.yaml
kubectl apply -f k8s/redis.yaml
kubectl -n mothra delete job/migrate --ignore-not-found && kubectl apply -f k8s/migrate-job.yaml \
  && kubectl -n mothra wait --for=condition=complete job/migrate --timeout=120s
kubectl apply -f k8s/ic.yaml -f k8s/text-service.yaml \
              -f k8s/paco-classifier-service.yaml \
              -f k8s/backend.yaml -f k8s/worker.yaml
kubectl apply -f k8s/ingress.yaml

# staging (same filenames — note the directory)
kubectl apply -f k8s/staging/secret.yaml -f k8s/staging/configmap.yaml
kubectl apply -f k8s/staging/stored-models-pv.yaml -f k8s/staging/stored-models-pvc.yaml
kubectl apply -f k8s/staging/redis.yaml
kubectl -n mothra delete job/migrate-staging --ignore-not-found && kubectl apply -f k8s/staging/migrate-job.yaml \
  && kubectl -n mothra wait --for=condition=complete job/migrate-staging --timeout=120s
kubectl apply -f k8s/staging/ic.yaml -f k8s/staging/text-service.yaml \
              -f k8s/staging/paco-classifier-service.yaml \
              -f k8s/staging/backend.yaml -f k8s/staging/worker.yaml
kubectl apply -f k8s/staging/ingress.yaml
```
`kubectl apply -f k8s/` does **not** recurse into `k8s/staging/` (that needs `-R`),
so a directory-wide production apply can't pick up staging by accident. The
migration Job (`migrate-job.yaml`, mothra#220 row 31) must complete before
`backend.yaml`/`worker.yaml` are applied -- they no longer create the schema
themselves at import.

## Verify
```
kubectl -n mothra get pods,pvc,svc,ingress          # all Ready, stored-models PVC Bound (RWX)
kubectl -n mothra logs deploy/backend | head        # tables created; Celery connected
kubectl -n mothra logs deploy/worker  | head        # "celery ... ready" (threads pool)
curl -k https://mothra.simssa.ca/                   # SPA HTML
# smoke test: POST /api/register → authed GET /api/projects (200)

# IC iframe not frame-blocked — expect the frame-ancestors CSP. The proxy's
# X-Frame-Options: SAMEORIGIN stays in the response; CSP takes precedence.
curl -sSI https://mothra-ic.simssa.ca/ | grep -iE 'content-security|frame-options'
kubectl -n mothra describe ingress mothra-ic | grep -i middlewares   # annotation present
```

Staging equivalents — plus the two checks that only matter in a shared namespace:
```
kubectl -n mothra get all -l mothra.env=staging      # staging objects only
kubectl -n mothra get endpoints backend ic text-service paco-classifier-service
# → production pods only (no staging IPs)

# staging really is wired to staging config/secrets — a copy-paste slip here is
# the one mistake that fails *open* (staging writing to the production database)
kubectl -n mothra get deploy backend-staging worker-staging -o jsonpath=\
'{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[*].envFrom[*].secretRef.name}{"\t"}{.spec.template.spec.containers[*].envFrom[*].configMapRef.name}{"\n"}{end}'
# → must print only mothra-secrets-staging / mothra-config-staging

kubectl -n mothra logs deploy/backend-staging | head
kubectl -n mothra logs deploy/worker-staging  | head
curl -k https://mothra.staging.simssa.ca/
curl -sSI https://mothra-ic.staging.simssa.ca/ | grep -iE 'content-security|frame-options'
kubectl -n mothra describe ingress mothra-ic-staging | grep -i middlewares
```

## Known follow-ups
- ~~No real `/healthz` yet~~ **done (mothra#220 row 29)** — `backend`/`text-service`
  now have real `httpGet` probes too, matching `paco-classifier-service`'s
  existing `/health` + `/ready`. `backend`'s readinessProbe (`/healthz`) checks
  Postgres + Celery broker reachability; its livenessProbe (`/healthz/live`)
  deliberately does not, so a transient DB/broker outage pulls the pod out of
  rotation instead of killing and restarting it. `text-service` has no
  DB/broker of its own, so both its probes point at the same `/healthz`.
- ~~`init_db()`/`_migrate_db()` run at import → keep backend/worker at 1 replica~~
  **done (mothra#220 row 31)** — a one-shot `migrate-job.yaml` now runs the
  schema migration once per deploy, applied and waited-on by
  `build-images.yml`'s `deploy` job before `backend`/`worker` are applied.
  `backend`/`worker` no longer create/alter the schema themselves. Both are
  still left at **1 replica** as an operational choice, not a code
  constraint — except `worker`, which has a *different* reason to stay at 1:
  its embedded Celery beat scheduler (row 28) would double-fire periodic
  tasks if run by more than one replica.
- text-service `/batch-download/{id}` uses local disk keyed by batch_id → needs
  shared storage or a single replica if batch downloads are used.
- Sharing the `mothra` namespace means no `ResourceQuota` headroom check happens
  automatically — staging adds ≈2 CPU / 8.6Gi of *requests* (paco-classifier-service
  alone is 500m / 3Gi of that). Confirm with
  `kubectl -n mothra describe quota,limitrange` if pods start failing admission.
