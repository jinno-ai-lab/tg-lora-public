# TG-LoRA Docker Development & Deployment Guide

This document describes how to use Docker to validate, test, and deploy the TG-LoRA training and evaluation environment, particularly when scaling up on cloud GPU platforms like Vast.ai.

---

## 1. Local Dev & Validation Quickstart

We use `docker compose` to build and manage containers with GPU access. Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host.

### Build the Image
```bash
make docker-build
```

### Run Tests Inside the Container
This runs the test suite inside the isolated container to ensure editable packages and configurations match the local environment:
```bash
make docker-test
```

### Start an Interactive Session with GPU
To launch an interactive bash shell with GPU acceleration:
```bash
make docker-run
```

### Run Downstream Evaluations in Container
```bash
make docker-eval
```

---

## 2. Docker Image Hosting & Registry Selection

To pull your pre-built environment onto cloud instances (like Vast.ai) without building it from scratch every time, you should publish the Docker image to a registry.

We compare the three primary registry candidates:

| Registry | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **GitHub Container Registry (GHCR)**<br>`ghcr.io/your-username/tg-lora` | • Fully integrated with GitHub Actions/CI.<br>• Free for public repositories.<br>• Secure and private access via GitHub PAT. | • Requires PAT authentication on the remote instance for private images. | **Recommended** for CI/CD automation and automated builds. |
| **Docker Hub**<br>`your-username/tg-lora:latest` | • The most standard, globally accessible registry.<br>• No authentication needed for public pulls. | • Rate limits on free accounts.<br>• Limited private repositories for free tier. | **Recommended** for quick, public sharing with Vast.ai. |
| **Vast.ai Direct Build / Base PyTorch** | • Bypasses registry upload latency.<br>• Simple docker-compose build on host. | • Renting billing starts during build time.<br>• Installing PyTorch + dependencies on every start adds 5–10 min latency. | **Alternative fallback** if registries are blocked or offline. |

### How to Push to GHCR
1. Authenticate with GHCR:
   ```bash
   echo $CR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
   ```
2. Tag the image:
   ```bash
   docker tag tg-lora:latest ghcr.io/YOUR_GITHUB_USERNAME/tg-lora:latest
   ```
3. Push the image:
   ```bash
   docker push ghcr.io/YOUR_GITHUB_USERNAME/tg-lora:latest
   ```

### Pulling on Vast.ai
On the Vast.ai instance, pull the precompiled image:
```bash
docker pull ghcr.io/YOUR_GITHUB_USERNAME/tg-lora:latest
```
This guarantees that training starts in under 30 seconds from instance creation, saving compute time and setup costs.
