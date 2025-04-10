from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2 import service_account

from dotenv import load_dotenv

import os
import io
from tqdm import tqdm
from pathlib import Path
from typing import Any, Tuple, Union

class GoogleDriveHelper():
    def __init__(self, project_root_dir:str, 
                 google_client_key_path:str, 
                 google_drive_owner_email:str,
                 drive_root_folder_name:str,
                 local_target_root_dir_name:str,
                 do_not_download_list:list=[],
                 do_not_upload_list:list=[],
                 test_mode:bool=False):
        self.GOOGLE_DRIVE_FOLDER_TYPE = 'application/vnd.google-apps.folder'
        self.MIME_TYPE_OCTET = 'application/octet-stream'

        self.mime_type = {
            'folder': 'application/vnd.google-apps.folder',
            'octet': 'application/octet-stream'
        }

        self.default_chunk_size = 20 * 1024 * 1024 # 20mb

        self.google_client_key_path = google_client_key_path
        self.project_root_dir = project_root_dir
        self.test_mode = test_mode
        self.google_drive_owner_email = google_drive_owner_email
        self.do_not_download_list = do_not_download_list
        self.do_not_upload_list = do_not_upload_list
        self.drive_root_folder_name = drive_root_folder_name
        self.local_target_root_dir = os.path.join(project_root_dir, local_target_root_dir_name)

        self._load_google_service()
        self.create_folder_and_file_metadatas()
    # ============================================ google service ============================================
    def _load_google_service(self):
        # with open(self.google_client_key_path, 'rb') as token:
        #     credential_info = json.load(token)
        # credentials = service_account.Credentials.from_service_account_info(credential_info)
        
        credentials = service_account.Credentials.from_service_account_file(self.google_client_key_path)
        service = build('drive', 'v3', credentials=credentials)
        self._file_service = service.files()
        self._permission_service = service.permissions()

    # ============================================ create metadata ============================================
    def create_folder_and_file_metadatas(self):
        # 모든 폴더를 가져오는 코드
        self.directory_struct = dict()

        self._get_folders_metadata()
        
        # 폴더 내의 모든 파일을 가져오는 코드
        for folder_name, v in self.directory_struct.items():
            folder_id = v['id']
            self._get_files_metadata(folder_id=folder_id,
                                     folder_name=folder_name)
            
    def _get_folders_metadata(self):
        results = []
        page_token = None
        while True:
            response = self._file_service.list(
                q=f"mimeType='{self.mime_type.get('folder')}' and name != 'dataset-backup'",
                spaces="drive",
                fields="nextPageToken, files(id, name, parents)",
                pageToken=page_token
            ).execute()
            results.extend(response.get("files", []))
            if (page_token := response.get("nextPageToken", None)) is None:
                break
            
        for folder in results:
            # if self.test_mode and folder.get("name").startswith("test"):
            #     self.delete_file(folder.get("id"))
            #     continue

            self.directory_struct[folder.get("name")] = {"id": folder.get("id"),
                                                    "parent_id": folder.get("parents", ["[ROOT]"])[0]}

    def _get_files_metadata(self, folder_id:str, folder_name:str):
        page_token = None
        results = []
        while True:
            response = self._file_service.list(
                q=f"'{folder_id}' in parents and mimeType != '{self.mime_type.get('folder')}'",
                spaces="drive",
                fields="nextPageToken, files(id, name, parents)",
                pageToken=page_token
            ).execute()
            results.extend(response.get("files", []))
            if (page_token := response.get("nextPageToken", None)) is None:
                break
        
        for file in results:
            self.directory_struct[folder_name][file.get("name")] = {"id": file.get("id"),
                                                                    "parent_id": file.get("parents", [""])[0]}
        
    def print_directory_metadata(self):
        for directory_name, folder in self.directory_struct.items():
            print(f"∴ {directory_name} ∴")
            for filename, file in folder.items():
                if filename in ["id", "parent_id"]:
                    print(f"\t{filename} - {file}")
                else:
                    print(f"\t∴  {filename} ∴")
                    for key, val in file.items():
                        print(f"\t\t{key} - {val}")

    def get_directory_id(self, directory_name:str) -> str:
        directory = self.directory_struct.get(directory_name, None)
        return (directory or {}).get("id", "")
        
    # ============================================ file or folder exists ============================================
    def _check_folder_exists(self, folder_name:str="") -> bool:
        return folder_name in self.directory_struct
    
    def _check_file_exists(self, folder_name:str = "", filename:str = "") -> bool:
        directory = self.directory_struct.get(folder_name, None)
        if directory is None:
            return False
        if filename in ["id", "parent_id"]:
            return False
        return filename in directory

    def is_exists(self, folder_name:str="", filename:str="") -> bool:
        if not folder_name:
            raise ValueError("폴더 이름은 필수입니다.")
        
        if filename == "":
            return self._check_folder_exists(folder_name=folder_name)
        else:
            return self._check_file_exists(folder_name=folder_name,
                                          filename=filename)

    # ============================================ create folder ============================================
    def create_folder(self, folder_name:str="", parent_folder_name:str=""):
        if folder_name in self.directory_struct:
            return
        
        try:
            parent_folder_id = self.directory_struct.get(parent_folder_name).get("id")

            metadata = {"name": folder_name, "parents": [parent_folder_id], "mimeType": self.mime_type.get('folder')}
            result = self._file_service.create(body=metadata, fields="id").execute()

            folder_id = result.get("id")
            self.directory_struct[folder_name] = {"id": folder_id,
                                                  "parent_id": parent_folder_id}
            self._change_permission(file_id=folder_id)
            
            print(f'create folder <{folder_name}> succeed')
        except Exception as e:
            print(f'{e}')

    # ============================================ upload file ============================================
    def upload_all_files(self):
        model_path = Path(self.local_target_root_dir)
        for file in model_path.rglob('*'):
            if file.is_file():
                self.upload_file(file)

    def upload_file(self, file_path:str, parent_folder_name:str='comment-filtering', additional_path:str='model'):
        remove_path = Path(os.path.join(self.project_root_dir, additional_path))
        file_path: Path = Path(file_path)

        relative_path = file_path.relative_to(remove_path)
        if relative_path.parts[-1] in self.do_not_upload_list:
            return
    
        for folder_name in relative_path.parts[:-1]:
            if not self.is_exists(folder_name=folder_name):
                self.create_folder(folder_name=folder_name, parent_folder_name=parent_folder_name)
            parent_folder_name = folder_name
        try:
            filename = relative_path.parts[-1]
            with open(file_path, 'rb') as file:
                file_buffer = io.BytesIO(file.read())

            media = MediaIoBaseUpload(file_buffer, mimetype=self.mime_type.get('octet'), 
                                        chunksize=self.default_chunk_size, 
                                        resumable=True)

            if filename in self.directory_struct[parent_folder_name]:
                request, upload_type = self._update_file(media=media, parent_folder_name=parent_folder_name, filename=filename)
            else:
                request, upload_type = self._create_file(media=media, parent_folder_name=parent_folder_name, filename=filename)
        
            file_id = self._upload_with_progress(request=request,
                                                filename=filename,
                                                file_size=file_path.stat().st_size)
            
            if upload_type == 'c':
                self.directory_struct[parent_folder_name][filename] = {'id': file_id,
                                                                        'parent_id': self.directory_struct[parent_folder_name]['id']}
                self._change_permission(file_id=file_id)
        except Exception as e:
            print(e)

    def _create_file(self, media:MediaIoBaseUpload, parent_folder_name:str, filename:str) -> Tuple[Any, str]:
        parent_folder_id = self.directory_struct[parent_folder_name].get("id")
        metadata = {'name': filename, 'parents': [parent_folder_id]}
        return self._file_service.create(media_body=media, body=metadata, fields="id"), 'c'

    def _update_file(self, media:MediaIoBaseUpload, parent_folder_name:str, filename:str) -> Tuple[Any, str]:
        file_id = self.directory_struct[parent_folder_name][filename].get("id")
        return self._file_service.update(fileId=file_id, media_body=media), 'u'
    
    def _upload_with_progress(self, request:Any, filename:str, file_size:int) -> Union[str, None]:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Upload - {filename}') as pbar:
            response = None
            while response is None:
                status, response = request.next_chunk()
                pbar.update(max(0, (status.progress() if status is not None else 1) * file_size - pbar.n))
        
        return response.get('id', None)

    # ============================================ download file ============================================
    def download_all_files(self, specific_file:Union[str, None]=None):
        print('download starts...')
        self._download_recursive(folder_id=self.directory_struct[self.drive_root_folder_name].get('id'),
                                 curr_local_path=os.path.join(self.project_root_dir, 'model'),
                                 specific_file=specific_file)
        print('download finished!')

    def _download_recursive(self, folder_id:str, curr_local_path:str, specific_file:Union[str, None]=None):
        if not os.path.exists(curr_local_path):
            os.mkdir(curr_local_path)

        items = self.get_all_items(folder_id)
        for item in items:
            if item.get("mimeType") == self.mime_type.get('folder'): # folder
                self._download_recursive(folder_id=item.get('id'), 
                                         curr_local_path=os.path.join(curr_local_path, item.get('name')),
                                         specific_file=specific_file)
            else: # file
                self._download_file(file=item,
                                    curr_local_path=curr_local_path,
                                    specific_file=specific_file)

    def _download_file(self, file:dict, curr_local_path:str, specific_file:Union[str, None]=None):
        file_id = file.get('id')
        filename = file.get('name')

        if specific_file is not None and filename != specific_file:
            return

        request = self._file_service.get_media(fileId=file_id)

        download_file = io.BytesIO()
        downloader = MediaIoBaseDownload(fd=download_file, request=request, chunksize=self.default_chunk_size)

        self._download_file_with_progress(downloader, filename)

        with open(os.path.join(curr_local_path, filename), 'wb') as save_file:
            save_file.write(download_file.getvalue())
    
    def _download_file_with_progress(self, downloader:Any, filename:str):
        with tqdm(unit='B', unit_scale=True, desc=f'Download - {filename}') as pbar:
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if pbar.total is None:
                    pbar.total = status.total_size
                
                if status is not None:
                    pbar.update(max(0, status.progress() * status.total_size - pbar.n))

    # ============================================ search ============================================
    def get_all_items(self, folder_id:str):
        results = []
        page_token = None
        while True:
            response = self._file_service.list(
                q=f"'{folder_id}' in parents",
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()
            results.extend(response.get("files", []))
            if (page_token := response.get('nextPageToken', None)) is None:
                break

        results = list(filter(lambda x: x.get('name') not in self.do_not_download_list, results))
        return results

    # ============================================ delete file ============================================
    def delete_file(self, file_id:str):
        self._file_service.delete(fileId=file_id).execute()

    def delete_folder(self, folder_id: str, force_delete: bool = False):
        page_token = None
        file_and_folders = []
        while True:
            response = self._file_service.list(
                q=f"'{folder_id}' in parents",
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()
            file_and_folders.extend(response.get('files', []))
            if (page_token := response.get('nextPageToken', None)) is None:
                break

        for item in file_and_folders:
            if item.get('mimeType') == self.mime_type.get('folder'):
                self.delete_folder(item.get('id'))
            else:
                if force_delete:
                    self._file_service.delete(fileId=item.get('id')).execute()
                else:
                    self.move_to_trash(item.get('id'))

        folder_name = self._file_service.get(fileId=folder_id, fields='name').execute().get('name')
        if force_delete:
            self._file_service.delete(fileId=folder_id).execute()
        else:
            self.move_to_trash(folder_id)
        self.directory_struct.pop(folder_name)

    def move_to_trash(self, file_id:str):
        body_value = {'trashed': True}
        self._file_service.update(fileId=file_id, body=body_value).execute()

    # ============================================ permission ============================================
    def _change_permission(self, file_id:str):
        permission_writer = {'type': 'user', 'role': 'writer', 'emailAddress': self.google_drive_owner_email}
        permission_reader = {'type': 'anyone', 'role': 'reader'}

        self._permission_service.create(fileId=file_id, body=permission_writer, sendNotificationEmail=False).execute()
        self._permission_service.create(fileId=file_id, body=permission_reader, sendNotificationEmail=False).execute()
        
        print(f"{file_id}'s permission has changed")

    def check_permissions(self, file_id:str):
        results = self._permission_service.list(fileId=file_id,
                                                fields="permissions(id, type, role, emailAddress)").execute()
        print(results)


if __name__ == "__main__":
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(curr_dir)
        load_dotenv(os.path.join(root_dir, 'env', '.env'))

        google_client_key_path = os.path.join(root_dir, 'env', 'ml-server-key.json')

        google_drive_owner_email = os.getenv("GOOGLE_DRIVE_OWNER_EMAIL")

        do_not_download_list = ['dataset-backup']

        downloader = GoogleDriveHelper(project_root_dir=root_dir,
                                       google_client_key_path=google_client_key_path,
                                       google_drive_owner_email=google_drive_owner_email,
                                       do_not_download_list=do_not_download_list,
                                       local_target_root_dir_name="model",
                                       drive_root_folder_name='comment-filtering',
                                       test_mode=True)
        
        # downloader.print_directory_metadata()

        # downloader.upload_all_files()
        
        downloader.print_directory_metadata()

        force_delete_flag=True
        downloader.delete_folder('1ije26xN9QM_geAmymNFKIO1gE6nnidVX', force_delete_flag)

        
        downloader.print_directory_metadata()
        # force_delete_flag=True
        # delete_target_list = ['1EPNDi3SFqXx8mzJWQlG_35mHGcUNpXu-', '1JxJls3lfHCda3X7AcozXqE_EBA8LPVb_', '18hpiV4xS-DbX4HSzyuuPqD9BuqVhlw7d', '1A0zN8HyqWXX8_NrmIRgFV9cq9Q3NZvRP', '1xmVE2OLNmB8aDePB2FvrUw7xQcmK8NEv', '1VtdlbkU-00cicCNUgBcZgpo5z8bAr0lQ', '1RHuW4cW2IdThfT6PjnqiDm4ojPHGpi1r']
        # for target in delete_target_list:
        #     downloader.delete_folder(target, force_delete_flag)
        # file_ids = ['100MP4h6eLWq8iOEizmz-qgjOP8bK02Gn', '1F2UhFQVB1MLJmpvQld3yZF8TrQ4OxaKg', '1f02rDOaUUL9EIO4d7Ka6sD6qx3x1JNnf', ]
        # for id in 

        # downloader._change_permission('1VtdlbkU-00cicCNUgBcZgpo5z8bAr0lQ')
        # downloader._change_permission('1RHuW4cW2IdThfT6PjnqiDm4ojPHGpi1r')
        

        # print(downloader.is_exists(folder_name="nickname_model",
        #                                    filename="model.safetensors"))
        # print(downloader.is_exists(folder_name="nickname_model",
        #                                    filename="model"))
        # print(downloader.is_exists(folder_name="nickname_model"))

        # try:
        #     print(downloader.is_exists(filename="model"))
        # except Exception as e:
        #     print(e)

        # try:
        #     print(downloader.is_exists())
        # except Exception as e:
        #     print(e)

        # downloader.create_folder(folder_name="testing!",
        #                          parent_folder_name="comment-filtering")

        # while True:
        #     if (text := input("폴더 이름 입력(나가기 - exit()): ").strip()).lower() == "exit()":
        #         break
        #     print(downloader.get_directory_id(text), downloader.is_exists(text))

        # print('upload file')
        # downloader._change_permission("1zri2WCCBscBiDDqiXlOdE_SwIMu1zGIz")
        # downloader._change_permission("1LC7fGjU_C0VtvNgKrQnzWkO1-FXWOgqQ")
        # downloader._change_permission("1cMetdGNA-xAkOXxFe-ad9x7lKPHQgZbP")
        # downloader._change_permission("1U4KAdzEPfpOhIJWDWnnoYUxvCyxzaGNb")
        # downloader.check_permissions("12r4FhHkRQ_JoRMPWcl7D2YVav8iutRTt")

        # downloader.upload_file('/home/sh/youtube-comment-ml-server/model/nickname_model/testing/tester/test/test.txt')

        # downloader.download_all_files()

        # downloader.upload_all_files()
        

    except Exception as e:
        print(e)