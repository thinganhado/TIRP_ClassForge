# app/database/powerbi_embed.py
import os
import requests
from flask import current_app

# configure these in your Config
TENANT_ID       = os.getenv("PBI_TENANT_ID")
CLIENT_ID       = os.getenv("PBI_CLIENT_ID")
CLIENT_SECRET   = os.getenv("PBI_CLIENT_SECRET")
WORKSPACE_ID    = os.getenv("PBI_WORKSPACE_ID")    # also called groupId
REPORT_ID       = os.getenv("PBI_REPORT_ID")
   
def get_embed_token():
    # 1) get AAD access token for Power BI
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type":    "client_credentials",
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope":         "https://analysis.windows.net/powerbi/api/.default"
    }
    resp = requests.post(url, data=data)
    resp.raise_for_status()
    aad_token = resp.json()["access_token"]

    # 2) call Power BI REST to generate an embed token
    embed_url = (
      f"https://api.powerbi.com/v1.0/myorg/groups/"
      f"{WORKSPACE_ID}/reports/{REPORT_ID}"
    )
    headers = {
      "Authorization": f"Bearer {aad_token}",
      "Content-Type":  "application/json"
    }
    body = {
      "accessLevel": "View"
    }
    r2 = requests.post(embed_url + "/GenerateToken", headers=headers, json=body)
    r2.raise_for_status()
    token = r2.json()["token"]
    return {
      "embedToken": token,
      "embedUrl":   r2.json().get("embedUrl", embed_url),
      "reportId":   REPORT_ID
    }