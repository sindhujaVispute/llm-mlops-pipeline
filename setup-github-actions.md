# GitHub Actions CI/CD Setup Guide

## Prerequisites

1. GitHub repository
2. Render account (for deployment)
3. Docker Hub account (optional)

## Step 1: Repository Setup

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit with MLOps pipeline"

# Add remote origin
git remote add origin https://github.com/yourusername/llm-mlops-pipeline.git
git branch -M main
git push -u origin main