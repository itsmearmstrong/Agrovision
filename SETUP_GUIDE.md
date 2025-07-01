# Google Cloud Vision API Setup Guide

## Step 1: Google Cloud Project Setup

1. **Visit Google Cloud Console**: Go to [console.cloud.google.com](https://console.cloud.google.com)

2. **Create/Select Project**:
   - Click the project dropdown at the top
   - Click "New Project" or select an existing one
   - Name it something like "Strawberry Image Analysis"
   - Click "Create"

## Step 2: Enable Cloud Vision API

1. **Navigate to API Library**:
   - In the left sidebar, go to "APIs & Services" > "Library"
   - Search for "Cloud Vision API"
   - Click on "Cloud Vision API"
   - Click "Enable"

## Step 3: Create Service Account

1. **Go to Credentials**:
   - In the left sidebar, go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"

2. **Fill in Service Account Details**:
   - Service account name: `strawberry-vision-api`
   - Description: `Service account for strawberry image analysis`
   - Click "Create and Continue"

3. **Assign Roles**:
   - Click "Select a role"
   - Search for "Cloud Vision"
   - Select "Cloud Vision API User"
   - Click "Continue"
   - Click "Done"

## Step 4: Create and Download API Key

1. **Create Key**:
   - Click on your newly created service account
   - Go to the "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose "JSON" format
   - Click "Create"
   - The JSON file will download automatically

2. **Place the Key File**:
   - Move the downloaded JSON file to your project directory
   - Rename it to something like `google-cloud-credentials.json`

## Step 5: Set Environment Variable

### Windows PowerShell:
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\JOSLAN\Desktop\Strawberry Image\google-cloud-credentials.json"
```

### Windows Command Prompt:
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\JOSLAN\Desktop\Strawberry Image\google-cloud-credentials.json
```

### For permanent setup (Windows):
1. Open System Properties > Environment Variables
2. Add new User Variable:
   - Variable name: `GOOGLE_APPLICATION_CREDENTIALS`
   - Variable value: `C:\Users\JOSLAN\Desktop\Strawberry Image\google-cloud-credentials.json`

## Step 6: Install Dependencies

1. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment**:
   ```bash
   # Windows PowerShell
   .\venv\Scripts\activate
   
   # Windows Command Prompt
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Step 7: Test the Setup

