#!/bin/bash

# === STEP 0: Ask for GitHub PAT ===
read -sp "Enter your GitHub Personal Access Token: " GITHUB_TOKEN
echo ""

# === STEP 1: Clean old credentials ===
git credential-cache exit
git config --global --unset credential.helper
git credential reject

# === STEP 2: Set Keychain helper ===
git config --global credential.helper osxkeychain

# === STEP 3: Remove old remote (if any) and add correct one ===
git remote remove origin 2>/dev/null
git remote add origin https://adi-2030:$GITHUB_TOKEN@github.com/adi-2030/clv_prediction.git

# === STEP 4: Commit changes ===
git add .
git commit -m "Push from automated script"

# === STEP 5: Push to GitHub ===
git branch -M main
git push -u origin main

echo "✅ Project successfully pushed to GitHub!"

