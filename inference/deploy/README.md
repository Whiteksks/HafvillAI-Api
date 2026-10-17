# Deploy inference on Brev

These files prepare a standalone GPU service. They do not provision or start
Brev resources. Select hardware and review its price separately before deployment.

## Prerequisites

- A running Linux x86-64 GPU instance with BF16 support and sufficient VRAM.
- Docker Engine, Docker Compose v2, and NVIDIA Container Toolkit configured for GPU access.
- The private bundle produced by `python3 -m inference.bundle`, transferred to the instance.
- Network access for the first image build. Model loading itself is offline.

The two roles previously used about 5.69 GiB of allocated tensor memory on an
RTX 3060 12 GiB. Allow additional memory for the process, input lengths, and
kernel compilation; this measurement is not a hardware guarantee.

## Configure on the instance

Run from the exported bundle root, after transferring it privately:

```bash
python3 -m inference.prepare_artifacts verify
cp inference/.env.example inference/.env
```

Edit `inference/.env` and set a unique `HAFVILLAI_API_KEY`. A key can be generated
locally with `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`.
Do not paste the key into source files, public logs, or a frontend bundle.

## Build and start

```bash
bash inference/deploy/compose.sh build
bash inference/deploy/compose.sh up -d api
bash inference/deploy/compose.sh logs -f api
```

The API listens inside its container on port 8000. Host access is bound to
`127.0.0.1:8000`, so it is not exposed publicly by this command. The API requires
its bearer key because the container listener binds to a non-loopback address.
The private artifact directory is mounted read-only; weights are never copied
into the Docker image. Writable kernel caches use a separate persistent volume.

The image uses CUDA Toolkit 12.8.1, Python 3.12, the pinned model stack, and a
non-root process. Compose requests one GPU and restarts failed processes or
services after Docker restarts. Docker itself must be configured to start at
boot. Stopped/deleted cloud instances cannot serve requests.

Wait for readiness after checksum verification and kernel warmup:

```bash
curl http://127.0.0.1:8000/health
```

For the real model smoke check, use `bash inference/run.sh smoke` from an installed
native environment, or run inside the service:

```bash
bash inference/deploy/compose.sh exec api python -m inference.smoke --role all --max-new-tokens 128
```

The smoke check takes its API key from the container environment and checks two
independent requests per role. It does not train or modify model files.

## Optional public HTTPS endpoint

Set `HAFVILLAI_DOMAIN` in `inference/.env` to a real DNS name pointing to the
instance. Allow inbound TCP ports 80 and 443 in the cloud/Brev network settings.
The instance must have a reachable route for these ports. Then:

```bash
bash inference/deploy/compose.sh --profile https up -d
```

The Caddy service obtains TLS certificates and forwards requests to `api:8000`.
The model API retains its bearer-key check. TLS state is stored in separate
persistent volumes. `localhost` is a development default, not a public endpoint.
Do not expose Caddy's admin endpoint.

A Brev browser-authenticated secure link is not automatically suitable for
server-to-server requests. Verify the chosen external URL from the actual
Next.js host, including authentication and request timeouts. Infrastructure
access and DNS/TLS setup are completed when the instance is available.

## Next.js connection

The intended path is browser → Next.js server → Brev HTTPS endpoint. Configure
the endpoint URL and model API key only in the Next.js server environment.
The server calls `/chat` or `/summarize`, forwards conversation messages, and
reads `content` from the full JSON response. The local WSL machine is not part
of this request path. Add application-session checks and suitable timeouts in
the site's server handler; the model service does not manage user accounts or
conversation storage.

## References

- [Docker Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).
- [NVIDIA Brev connectivity](https://docs.nvidia.com/brev/cli/connectivity).
- [Caddy reverse proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy).
- [Next.js server/client boundaries](https://nextjs.org/docs/app/getting-started/server-and-client-components).
