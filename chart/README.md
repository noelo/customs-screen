# customs-screen Helm chart

Deploys the Border Classifier (three-tier HTS classification demo) to
Kubernetes/OpenShift. Target: aarch64 OpenShift SNO, namespace `deepsec`,
StorageClass `lvms-vg1` (topolvm, WaitForFirstConsumer).

## Architecture

```
init pod (app image)                app pod
┌─────────────────────┐            ┌──────────────────────────┐
│ /app/docker/        │   writes   │ server.py :8000          │
│   init-data.sh      │──────────▶ │ loads /data/hts_index.json│
│ curl USITC export → │    PV      │ BM25 + logprob scoring    │
│ build_index.py      │            │ TMM_MAAS_API_KEY (secret) │
└─────────────────────┘            └──────────────────────────┘
         └──────────▶ PVC 5Gi lvms-vg1 (RWO) ◀──────┘
```

- The 36 MB HTS raw export + built index are **not baked into the image**.
  The init container downloads the USITC export and builds the index into
  the PV once; re-deploys skip when `hts_index.json` already exists.
- Probes hit the app's `/healthz`.

## Deploy

```bash
# 1. Push the image (built on the Pi: podman build -t quay.io/noeloc/customs-screen .)
podman push quay.io/noeloc/customs-screen:latest

# 2. Wire the API key — either an existing secret...
helm upgrade --install customs ./chart -n deepsec \
  --set llm.existingSecret=llm-access --set llm.secretKey=LLMAPIKEY

# ...or let the chart create one (NOT recommended for real keys in shell history):
helm upgrade --install customs ./chart -n deepsec \
  --set llm.apiKey="$TMM_MAAS_API_KEY" \
  --set llm.baseUrl="https://maas.apps.ocp.cloud.rhai-tmm.dev/prelude-maas/glm-53-flash/v1" \
  --set llm.model=glm-53-flash

# 3. Verify
kubectl rollout status deployment/customs-customs-screen -n deepsec
kubectl port-forward svc/customs-customs-screen 8000:8000 -n deepsec
curl http://127.0.0.1:8000/healthz
```

## Values that matter

| Value | Default | Note |
|---|---|---|
| `image.repository` | `quay.io/noeloc/customs-screen` | aarch64 image |
| `persistence.existingClaim` | `""` | set to reuse a PVC instead of creating one |
| `persistence.storageClassName` | `lvms-vg1` | cluster default |
| `initContainer.enabled` | `true` | set `false` only if the PV is pre-seeded |
| `initContainer.imageOverride` | `""` | empty = reuse the app image (has curl + python) |
| `llm.existingSecret` | `""` | secret holding `TMM_MAAS_API_KEY` |
| `serviceAccount.create` | `true` | set `false` + `serviceAccount.name` to reuse an SA |

## SCC

The image runs as `USER 1001` with `allowPrivilegeEscalation: false` and all
capabilities dropped — no SCC binding needed on restricted/baseline namespaces.
