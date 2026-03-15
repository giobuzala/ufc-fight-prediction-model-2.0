from azure.identity import InteractiveBrowserCredential
from azure.keyvault.secrets import SecretClient

# Set up ----
key_vault_name = "kv-azure-lab-ufc"
kv_uri = f"https://{key_vault_name}.vault.azure.net"

credential = InteractiveBrowserCredential()
client = SecretClient(vault_url=kv_uri, credential=credential)

# Read secret ----
secret = client.get_secret("AZURE-STORAGE-CONNECTION-STRING").value

print(secret)