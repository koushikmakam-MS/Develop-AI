// ============================================================
// BCRD DeveloperAI — Azure Bicep Deployment
// ============================================================
// Deploys:
//   1. Azure Container Registry (ACR) — stores the Docker image
//   2. Azure Container Apps Environment — hosting platform
//   3. Azure Container App — runs the BCRD DeveloperAI bot + all services
//   4. Azure Bot Service — connects to Microsoft Teams
//
// Usage:
//   az deployment group create \
//     --resource-group bcrd-devai-rg \
//     --template-file deploy/main.bicep \
//     --parameters botAppId=<YOUR_BOT_APP_ID> \
//                  botAppPassword=<YOUR_BOT_APP_PASSWORD> \
//                  llmApiKey=<YOUR_LLM_KEY> \
//                  azureDevOpsPat=<YOUR_PAT>
// ============================================================

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Base name for all resources')
param baseName string = 'bcrd-devai'

@description('Bot Framework App ID (from Azure Bot registration)')
param botAppId string

@secure()
@description('Bot Framework App Password')
param botAppPassword string

@secure()
@description('Azure OpenAI API key')
param llmApiKey string

@secure()
@description('Azure DevOps PAT for repo sync')
param azureDevOpsPat string

@description('Kusto cluster URL')
param kustoClusterUrl string = 'https://mabprod1.kusto.windows.net'

@description('Kusto database name')
param kustoDatabase string = 'MABKustoProd1'

// ── Computed names ──────────────────────────────────────────

var acrName = replace('${baseName}acr', '-', '')
var envName = '${baseName}-env'
var appName = '${baseName}-bot'
var botServiceName = '${baseName}-bot-service'
var logWorkspaceName = '${baseName}-logs'

// ── Log Analytics Workspace ─────────────────────────────────

resource logWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logWorkspaceName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Azure Container Registry ────────────────────────────────

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

// ── Container Apps Environment ──────────────────────────────

resource containerEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logWorkspace.properties.customerId
        sharedKey: logWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// ── Container App ───────────────────────────────────────────

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 3978
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'acr-password', value: acr.listCredentials().passwords[0].value }
        { name: 'bot-app-password', value: botAppPassword }
        { name: 'llm-api-key', value: llmApiKey }
        { name: 'devops-pat', value: azureDevOpsPat }
      ]
    }
    template: {
      containers: [
        {
          name: 'bcrd-devai-bot'
          image: '${acr.properties.loginServer}/${baseName}:latest'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'BCRD_DEVAI_TEAMS_BOT_APP_ID', value: botAppId }
            { name: 'BCRD_DEVAI_TEAMS_BOT_APP_PASSWORD', secretRef: 'bot-app-password' }
            { name: 'BCRD_DEVAI_LLM_API_KEY', secretRef: 'llm-api-key' }
            { name: 'BCRD_DEVAI_AZURE_DEVOPS_PAT', secretRef: 'devops-pat' }
            { name: 'BCRD_DEVAI_KUSTO_CLUSTER_URL', value: kustoClusterUrl }
            { name: 'BCRD_DEVAI_KUSTO_DATABASE', value: kustoDatabase }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/api/health'
                port: 3978
              }
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/api/health'
                port: 3978
              }
              initialDelaySeconds: 15
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-requests'
            http: { metadata: { concurrentRequests: '50' } }
          }
        ]
      }
    }
  }
}

// ── Azure Bot Service ───────────────────────────────────────

resource botService 'Microsoft.BotService/botServices@2022-09-15' = {
  name: botServiceName
  location: 'global'
  kind: 'azurebot'
  sku: { name: 'F0' }    // Free tier — change to S1 for production
  properties: {
    displayName: 'BCRD DeveloperAI'
    description: 'AI assistant for Azure Backup Management'
    msaAppId: botAppId
    endpoint: 'https://${containerApp.properties.configuration.ingress.fqdn}/api/messages'
    msaAppType: 'SingleTenant'
  }

  // Enable the Microsoft Teams channel
  resource teamsChannel 'channels' = {
    name: 'MsTeamsChannel'
    location: 'global'
    properties: {
      channelName: 'MsTeamsChannel'
    }
  }
}

// ── Outputs ─────────────────────────────────────────────────

@description('Container App FQDN — use this as the Bot messaging endpoint')
output appUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'

@description('Bot messaging endpoint')
output botEndpoint string = 'https://${containerApp.properties.configuration.ingress.fqdn}/api/messages'

@description('ACR login server — for docker push')
output acrLoginServer string = acr.properties.loginServer

@description('Azure Bot Service name')
output botServiceNameOutput string = botServiceName
