import re
import os
import datetime
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request


def file_validity_checker(filepath):
    #remove_double_slashes_if_any
    filepath = re.sub(r'/+', '/', filepath)
    try:
        with open(filepath, "r") as file:
            #extract data
            #print(f"{filepath} is good.")
            return filepath

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except PermissionError:
        print(f"Error: Permission for reading '{filepath}' not availble.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def create_service(client_secret_file, api_name, api_version, *scopes, prefix=''):
    CLIENT_SECRET_FILE = client_secret_file
    API_SERVICE_NAME = api_name
    API_VERSION = api_version
    SCOPES = [scope for scope in scopes[0]]

    print(f" SCOPES in APITrials = {SCOPES} \n\n")

    
    creds = None
    working_dir = os.getcwd()
    token_dir = 'token files'
    token_file = f'token_{API_SERVICE_NAME}_{API_VERSION}{prefix}.json'

    ### Check if token dir exists first, if not, create the folder
    if not os.path.exists(os.path.join(working_dir, token_dir)):
        os.mkdir(os.path.join(working_dir, token_dir))

    if os.path.exists(os.path.join(working_dir, token_dir, token_file)):
        fpath = os.path.join(working_dir, token_dir, token_file)
        if file_validity_checker(fpath):
            creds = Credentials.from_authorized_user_file(fpath, SCOPES)

    if creds:
        print(f"Cred created with Access Token: {creds.token}")

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(os.path.join(working_dir, token_dir, token_file), 'w') as token:
            token.write(creds.to_json())

    print(f"Cred just before build : {creds.token}")
    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=creds, static_discovery=False)
        #service = build(API_SERVICE_NAME, API_VERSION, credentials=creds)
        print(API_SERVICE_NAME, API_VERSION, 'service created successfully')
        return service
    except Exception as e:
        print(f" The exceptions is {e}")
        print(f'Failed to create service instance for {API_SERVICE_NAME}')
        os.remove(os.path.join(working_dir, token_dir, token_file))
        return None

def convert_to_RFC_datetime(year=1900, month=1, day=1, hour=0, minute=0):
    dt = datetime.datetime(year, month, day, hour, minute, 0).isoformat() + 'Z'
    return dt
