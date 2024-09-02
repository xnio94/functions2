# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
#
#
# def save_to_drive(file, folder):
#     gauth = GoogleAuth()
#     gauth.LocalWebserverAuth()  # client_secrets.json need to be in the same directory as the script
#     drive = GoogleDrive(gauth)
#
#     # View all folders and file in your Google Drive
#     fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
#     for file in fileList:
#         print('Title: %s, ID: %s' % (file['title'], file['id']))
#         # Get the folder ID that you want
#         if (file['title'] == "anas"):
#             fileID = file['id']
#     return fileList


from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# If modifying these scopes, delete the file token.json.
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive']


def save_to_drive(file, title):
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)
        # # Call the Drive v3 API
        # results = service.files().list(
        #     pageSize=10, fields="nextPageToken, files(id, name)").execute()
        # items = results.get('files', [])
        # if not items:
        #     print('No files found.')
        #     return
        # print('Files:')
        # for item in items:
        #     print(u'{0} ({1})'.format(item['name'], item['id']))
        # print('3')
        #
        file_metadata = {
            'name': file + '_' + title,
            'mimeType': '*/*',
            'parents': ['1QYSXP7WB2t9xo2G0tK9B6XIzzMaiDRNF'],
        }
        media = MediaFileUpload(file, mimetype='*/*', resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print('File ID: ' + file.get('id'))
        return "Done"

    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')
