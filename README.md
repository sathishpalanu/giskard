# 🧪 Giskard Proof of Concept on OpenShift

This project demonstrates how to deploy the [Giskard AI Testing Platform](https://github.com/Giskard-AI/giskard) on an **OpenShift cluster**, including:

- Self-hosted Giskard server using Docker image
- PostgreSQL as backend database
- External access via OpenShift Route
- Sample Python SDK usage to upload models and scan results

---

## 📌 What is Giskard?

**Giskard** is an open-source framework to:
- Scan ML models for bugs, biases, and vulnerabilities
- Generate test suites and reports
- Collaborate with teams using a web UI

Giskard has two main components:
- 🧪 **Python SDK** — to test and scan models locally
- 🌐 **Web Server (UI)** — to view and manage test results

---

## 📁 Project Structure

```bash
.
├── postgresql.yaml            # PostgreSQL Deployment + Service
├── giskard-deployment.yaml    # Giskard Server Deployment
├── giskard-service.yaml       # Giskard Service
├── giskard-route.yaml         # OpenShift Route to expose Giskard
└── README.md


🚀 Step-by-Step Deployment Guide
1️⃣ Pre-requisites

✅ OpenShift CLI (oc) installed

✅ Access to an OpenShift cluster with permission to create resources

✅ Docker-enabled OpenShift environment

✅ Python 3.8+ and pip if running SDK locally

2️⃣ Deploy to OpenShift

Switch to or create a new project:

oc new-project giskard-poc
oc project giskard-poc

Apply all deployment files:

oc apply -f postgresql.yaml
oc apply -f giskard-deployment.yaml
oc apply -f giskard-service.yaml
oc apply -f giskard-route.yaml

Check that everything is running:

oc get pods
oc get svc
oc get route

Open the Giskard UI using the URL from:

oc get route giskard

It should be similar to:

https://giskard-giskard-poc.apps.<your-cluster-domain>

3️⃣ Use Python SDK to Upload Model

Install the SDK (in a virtual environment is best):

pip install giskard scikit-learn pandas

Create a script (giskard_poc.py) to train and upload a model:

🔐 Security Notes

Default setup uses no authentication for the Giskard server (intended for PoC).

For production:

Use OpenShift OAuth or service mesh to secure the route.

Replace hardcoded DB credentials with Kubernetes Secrets.

Use persistent volumes for PostgreSQL storage.

🧰 Resources

🔗 Giskard GitHub Repo

📘 Giskard Docs

🐙 OpenShift Docs
