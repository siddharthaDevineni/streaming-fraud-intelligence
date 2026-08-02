# Cloud Deployment (GKE)

Deploys the full Streaming Fraud Intelligence stack to **Google Kubernetes Engine (GKE Autopilot)** – Kafka via the
Strimzi Operator in KRaft mode, ChromaDB as a standalone client-server deployment, secrets from GCP Secret Manager, and
every service running as its own container image pulled from Artifact Registry. One script (`deploy.sh`) handles the
entire rollout.

---

## Prerequisites (one-time, per GCP project)

1. **A GCP project with billing enabled.** Even the free tier requires billing to be linked before APIs can be enabled.

2. **`gcloud` CLI installed and authenticated:**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud config set compute/region europe-west3   # or your preferred region
   ```

3. **Required APIs enabled:**
   ```bash
   gcloud services enable \
     container.googleapis.com \
     artifactregistry.googleapis.com \
     secretmanager.googleapis.com \
     compute.googleapis.com
   ```

4. **`kubectl` installed:**
   ```bash
   gcloud components install kubectl gke-gcloud-auth-plugin
   ```

5. **An Artifact Registry repository:**
   ```bash
   gcloud artifacts repositories create streaming-fraud-intelligence \
     --repository-format=docker \
     --location=europe-west3 \
     --description="Docker images for Streaming Fraud Intelligence"

   gcloud auth configure-docker europe-west3-docker.pkg.dev
   ```

6. **API keys registered in Secret Manager** — the scripts *read* from Secret Manager, they don't create the underlying
   values. Store your own keys once:
   ```bash
   echo -n "your-actual-groq-api-key" | \
     gcloud secrets create groq-api-key --data-file=-

   echo -n "your-actual-langsmith-api-key" | \
     gcloud secrets create langchain-api-key --data-file=-
   ```

7. **IAM binding** so the in-cluster service account can read those secrets (Workload Identity):
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:fraud-detection-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

8. **Cross-platform builds, if building on Apple Silicon / ARM64:** GKE Autopilot nodes are `amd64`. Building on an
   `arm64` dev machine without emulation produces images that fail at pod startup with `exec format error`. One-time
   host setup:
   ```bash
   docker run --privileged --rm tonistiigi/binfmt --install all
   ```
   (Registers QEMU handlers with the kernel — not part of the build pipeline itself, just a machine prerequisite, like
   installing Docker.)

---

## Deploying

```bash
# 1. Build and push all three images (spring-boot, python-ml, testgen)
./k8s/build-and-push.sh

# 2. Deploy everything — creates the cluster if it doesn't exist, installs Strimzi, waits on readiness at each dependency step,
#    trains the model before starting services that depend on it
./k8s/deploy.sh
```

`deploy.sh` is idempotent and self-healing — safe to re-run at any point. It will skip steps that are already
satisfied (existing cluster, existing Strimzi CRDs, an already-trained model) rather than repeating them.

### Access the dashboard and Kafka UI

```bash
kubectl get service streamlit -n fraud-detection   # → open EXTERNAL-IP:8501
kubectl get service kafka-ui  -n fraud-detection   # → open EXTERNAL-IP:8090
```

`EXTERNAL-IP` may show `<pending>` for 1–2 minutes after first deploy while the GCP LoadBalancer provisions.

### Run a test scenario

`TestDataGenerator` is a `src/test/java` class — Maven doesn't bundle it into the production `app.jar`, so it runs as
its own Job, inside the cluster network, talking to Kafka via its internal DNS name (no port-forwarding, nothing exposed
outside the cluster):

```bash
kubectl apply -f k8s/test-data-generator-job.yaml
kubectl logs -f job/test-data-generator -n fraud-detection
```

Jobs are immutable once created — to run it again:

```bash
kubectl delete job test-data-generator -n fraud-detection --ignore-not-found
kubectl apply -f k8s/test-data-generator-job.yaml
```

### Watch logs

```bash
kubectl logs -f deployment/inference-consumer -n fraud-detection
kubectl logs -f deployment/spring-boot -n fraud-detection
kubectl logs -f deployment/feedback-embedder -n fraud-detection
```

### Tear down (stop billing between sessions)

```bash
./k8s/teardown.sh
```

Deletes the `fraud-detection` namespace and everything in it. The GKE cluster itself keeps running at minimal idle
cost — to stop that too:

```bash
gcloud container clusters delete streaming-fraud-intel --location=europe-west3
```

---

## Architecture decisions specific to this deployment

### ChromaDB runs as its own client-server Deployment

Locally (Docker Compose), each Python service opens ChromaDB directly as an embedded `PersistentClient` against a shared
volume. On Kubernetes this breaks two ways at once:

- **Storage class mismatch** - GKE's default `standard-rwo` storage class only supports `ReadWriteOnce`, meaning only
  one pod can mount the volume at all. `ReadWriteMany` requires Filestore (NFS-backed), which is expensive and overkill
  for this workload.
- **Multi-process corruption** – even where multiple mounts *are* possible, concurrent embedded ChromaDB clients writing
  to the same SQLite-backed store from separate processes causes HNSW index corruption (`Nothing found on disk`
  errors) — the same class of bug hit during local development.

The fix: `chromadb-server` runs as its own single-replica Deployment, the **only** process touching the storage volume (
`ReadWriteOnce` is fine — only one pod ever needs it). Every other service connects over HTTP (`chromadb.HttpClient`)
instead of opening the file directly. This is also how ChromaDB is meant to be run in production generally, not a
workaround specific to Kubernetes.

### Kafka runs via Strimzi in KRaft mode, not a raw StatefulSet

Kubernetes has no native concept of "a Kafka cluster" — Strimzi is a **Kubernetes Operator**: it watches for `Kafka` and
`KafkaTopic` custom resources and reconciles the actual StatefulSets, Services, and configs needed to run a real broker
cluster.

### Kafka topic naming - `spec.topicName` vs `metadata.name`

Kubernetes resource names must be lowercase DNS-1123 subdomains — no uppercase characters allowed. Kafka topic names
have no such restriction, and this project's code uses `customerProfiles` (camelCase) throughout. Strimzi's `KafkaTopic`
CRD separates the two: the Kubernetes object can be named `customer-profiles` (satisfies K8s naming rules) while
`spec.topicName: customerProfiles` sets the *actual* Kafka topic name to exactly what the application code expects.

### `TestDataGenerator` needs its own image, separate from the app

`Dockerfile.streams` (the production Spring Boot image) is a lean, multi-stage build — source code is discarded after
compilation, only the final `app.jar` ships. `TestDataGenerator` lives under `src/test/java`, which Maven never bundles
into that jar. `Dockerfile.testgen` is a dedicated image carrying the full Maven build environment and source, so
`mvn exec:java -Dexec.classpathScope=test`works identically to how it runs locally from an IDE — without bloating the
production runtime image with a build toolchain it doesn't need.

### Security contexts - non-root containers need explicit volume ownership

Both `Dockerfile.streams` and `Dockerfile.ml` run as non-root users (`appuser`, `mluser`) for security.
Kubernetes-managed volumes (`emptyDir`, PVC mounts) default to root ownership unless told otherwise. Two fixes were
needed:

- `spring-boot` Deployment: `securityContext.fsGroup: 999` (matching `appuser`'s GID) so it can write Kafka Streams'
  RocksDB state directory
- `model-trainer` Job: `securityContext.runAsUser: 0` — MLflow's file store needs to create its own directory tree at
  `/models/mlruns`; acceptable for a short-lived, one-shot batch job with no persistent exposure

### Regional disk quota (`SSD_TOTAL_GB`)

Free-tier GCP projects have a hard, **non-adjustable** regional quota on total persistent disk (500GB by default) — this
isn't something you can request an increase for via the standard Quotas UI on a trial account. GKE Autopilot's node
auto-provisioning, combined with several rounds of`kubectl apply` churn during initial setup, can approach this ceiling
faster than expected, since every auto-provisioned node carries its own boot disk on top of whatever PVCs you've
explicitly requested. If you hit `FailedScaleUp: GCE quota exceeded`, check actual disk usage directly:

```bash
gcloud compute disks list --filter="NOT name~pvc-"
```

and trim `resources.requests` across Deployments rather than requesting a quota increase that likely isn't available at
this account tier.

---

## Troubleshooting quick reference

| Symptom                                                               | Likely cause                                                                                                                                                                                            |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `exec format error` at container startup                              | Image built for wrong CPU architecture — see prerequisite #8                                                                                                                                            |
| `UNKNOWN_TOPIC_OR_PARTITION` in Spring Boot logs                      | Topic name mismatch — check `spec.topicName` in `kafka-topics.yaml` matches the Java/Python code exactly                                                                                                |
| `Operation not permitted` writing to a mounted volume                 | Missing `securityContext.fsGroup` for the container's non-root user                                                                                                                                     |
| `FailedScaleUp: GCE quota exceeded`                                   | Regional `SSD_TOTAL_GB` quota — see architecture note above                                                                                                                                             |
| Job `spec.template...: Not found` on apply                            | `volumeMounts` references a volume name with no matching entry in `volumes:`                                                                                                                            |
| RAG retrieval always returns 0 results despite a populated collection | Check every ChromaDB client instantiation uses the shared `get_chromadb_client()` helper — a hardcoded `PersistentClient` anywhere silently reads from an empty local path instead of `chromadb-server` |
