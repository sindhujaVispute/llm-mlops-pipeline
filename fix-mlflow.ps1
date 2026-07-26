# fix-mlflow.ps1 - Fix MLflow backend issue

Write-Host "🔧 Fixing MLflow backend configuration..." -ForegroundColor Cyan

# Update the workflow file
$workflowPath = ".github/workflows/mlops-pipeline.yml"

if (Test-Path $workflowPath) {
    Write-Host "📝 Updating workflow file..." -ForegroundColor Yellow
    
    # Backup existing workflow
    Copy-Item $workflowPath "$workflowPath.bak"
    
    # Read and update the file
    $content = Get-Content $workflowPath -Raw
    
    # Replace the mlflow server command with SQLite version
    $content = $content -replace 'mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \.\/mlflow', 
                              'mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --serve-artifacts'
    
    # Save the updated content
    $content | Out-File -FilePath $workflowPath -Encoding utf8
    
    Write-Host "✅ Workflow updated!" -ForegroundColor Green
}

# Create a simple MLflow config for local development
@'
# .mlflow_config
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
'@ | Out-File -FilePath ".env.mlflow" -Encoding utf8

Write-Host "✅ Created .env.mlflow" -ForegroundColor Green

Write-Host ""
Write-Host "📌 Changes made:" -ForegroundColor Cyan
Write-Host "  - Using SQLite backend instead of filesystem"
Write-Host "  - Adding --default-artifact-root and --serve-artifacts flags"
Write-Host ""
Write-Host "📌 Next steps:" -ForegroundColor Cyan
Write-Host "1. git add ."
Write-Host "2. git commit -m 'Fix MLflow backend: Use SQLite instead of filesystem'"
Write-Host "3. git push origin main"